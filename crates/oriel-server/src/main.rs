//! Oriel server entrypoint.
//!
//! Launches an isolated headless Wayland session + private D-Bus + virtual-only
//! PipeWire/WirePlumber audio bridge inside a private session `XDG_RUNTIME_DIR`
//! and streams rootless windows, audio, and remote PTYs over QUIC.

use oriel_server::{audio, compositor, flatpak_bwrap, gpu, quic, session, shell_env};

use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal, SignalKind};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

/// TTL for single-use tokens minted via `attach` / control socket.
const ATTACH_TOKEN_TTL: Duration = Duration::from_secs(60);
/// TTL for the initial single-use token in foreground mode.
const FOREGROUND_TOKEN_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Parser)]
#[command(
    name = "oriel-server",
    about = "Oriel headless app & PTY server",
    // Server options belong after a subcommand (`attach --no-gpu`); before it they'd be ignored.
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<ServerSubcommand>,

    /// IP to bind the QUIC server on, optionally with a UDP port or inclusive port range:
    /// `0.0.0.0`, `0.0.0.0:7443`, `0.0.0.0:7443-7462`, `[::]:7443`. Without a port, any
    /// free port is used; with one, the server only starts if it can bind a port in it.
    #[arg(long, default_value = "0.0.0.0", value_name = "IP[:PORT[-END]]")]
    listen: quic::ListenSpec,

    /// Run quietly as a background daemon (used internally by `attach`).
    #[arg(long, hide = true)]
    daemon: bool,

    /// Disable embedded XWayland support (Wayland-only session).
    #[arg(long)]
    no_xwayland: bool,

    #[command(flatten)]
    gpu: GpuArgs,

    /// Target GUI application and arguments to launch inside the isolated session.
    #[arg(last = true)]
    command: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum ServerSubcommand {
    /// Attach to an existing persistent user session (or start one) and print a fresh connect URI.
    Attach {
        /// Where a newly started daemon listens, as for the server's own `--listen`. With a
        /// port or range, attaching to an already running daemon also requires its port to
        /// be in it.
        #[arg(long, default_value = "0.0.0.0", value_name = "IP[:PORT[-END]]")]
        listen: quic::ListenSpec,

        /// Hostname or IP to embed in the printed URI (defaults to listen IP).
        #[arg(long)]
        host: Option<String>,

        /// Disable embedded XWayland support when starting a new daemon.
        #[arg(long)]
        no_xwayland: bool,

        /// Bind the printed URI's token to the client certificate with this SHA-256 hex
        /// fingerprint: only a client holding that certificate's private key can use it.
        #[arg(long, value_name = "SHA256", value_parser = parse_cert_fingerprint)]
        client_fp: Option<String>,

        /// If the running server speaks a different protocol version than this binary (it
        /// is from an older or newer install), stop it and start this version. Its apps
        /// are closed.
        #[arg(long)]
        replace: bool,

        #[command(flatten)]
        gpu: GpuArgs,
    },
    /// Stop this user's running server, closing its apps.
    Stop,
    /// Print the environment for `attach`'s login-shell environment resolution.
    #[command(name = shell_env::PRINT_SUBCOMMAND, hide = true)]
    PrintShellEnv,
}

/// GPU acceleration options for apps in the session.
#[derive(Debug, Clone, clap::Args)]
struct GpuArgs {
    /// DRM render node apps render on (default: the first hardware GPU EGL reports).
    #[arg(long, value_name = "PATH", conflicts_with = "no_gpu")]
    render_node: Option<PathBuf>,

    /// Disable GPU acceleration; apps render in software and share frames via `wl_shm`.
    #[arg(long)]
    no_gpu: bool,
}

impl GpuArgs {
    fn mode(&self) -> gpu::GpuMode {
        match (&self.render_node, self.no_gpu) {
            (_, true) => gpu::GpuMode::Disabled,
            (Some(path), false) => gpu::GpuMode::Node(path.clone()),
            (None, false) => gpu::GpuMode::Auto,
        }
    }

    fn push_args(&self, cmd: &mut Command) {
        if self.no_gpu {
            cmd.arg("--no-gpu");
        }
        if let Some(path) = &self.render_node {
            cmd.arg("--render-node").arg(path);
        }
    }
}

/// What `attach` asks the daemon for over the control socket, as one JSON line.
#[derive(Debug, Serialize, Deserialize)]
struct ControlSockRequest {
    /// Client certificate fingerprint to bind the minted token to.
    #[serde(default)]
    client_fp: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ControlSockInfo {
    port: u16,
    fp: String,
    token: String,
    /// The daemon's [`oriel_proto::PROTOCOL_VERSION`]; older daemons do not send it.
    #[serde(default)]
    protocol: Option<u32>,
}

/// How long `stop` waits for a stopped daemon to clean up its session and exit.
const STOP_TIMEOUT: Duration = Duration::from_secs(20);

/// How long the daemon waits for `attach`'s request line.
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// A SHA-256 certificate fingerprint: 64 hex digits, normalized to lowercase.
fn parse_cert_fingerprint(s: &str) -> Result<String, String> {
    if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(s.to_ascii_lowercase())
    } else {
        Err("expected a SHA-256 fingerprint of 64 hex digits".into())
    }
}

fn main() -> Result<()> {
    // Flatpak runs this binary as its bwrap in our sessions (see `flatpak_bwrap`).
    flatpak_bwrap::run_if_wrapper();
    let cli = Cli::parse();
    if matches!(cli.subcommand, Some(ServerSubcommand::PrintShellEnv)) {
        return shell_env::print_env();
    }
    // A server started by hand, from a service or from cron gets the login-shell environment
    // here, before the runtime starts any threads (`attach` resolves it for its own daemon).
    let startup_env = cli.subcommand.is_none().then(shell_env::apply_at_startup);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the tokio runtime")?
        .block_on(run(cli, startup_env))
}

async fn run(cli: Cli, startup_env: Option<shell_env::StartupOutcome>) -> Result<()> {
    // Validate host XDG_RUNTIME_DIR up front (strict 0700 owned by current UID, no /tmp fallback).
    let host_runtime_dir = session::validate_host_runtime_dir()?;

    if let Some(ServerSubcommand::Attach {
        listen,
        host,
        no_xwayland,
        client_fp,
        replace,
        gpu,
    }) = cli.subcommand
    {
        let request = ControlSockRequest { client_fp };
        return run_attach(
            &host_runtime_dir,
            listen,
            host,
            no_xwayland,
            replace,
            &gpu,
            &request,
        )
        .await;
    }
    if matches!(cli.subcommand, Some(ServerSubcommand::Stop)) {
        let sock_path = session::control_socket_path(&host_runtime_dir);
        return match stop_daemon(&sock_path).await? {
            Some(pid) => {
                eprintln!("stopped the Oriel server (pid {pid})");
                Ok(())
            }
            None => {
                eprintln!("no Oriel server is running for this user");
                Ok(())
            }
        };
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?)
                // Render nodes never grant DRM master; smithay warns about it regardless.
                .add_directive("smithay::backend::drm::device::fd=error".parse()?),
        )
        .init();

    match startup_env {
        Some(shell_env::StartupOutcome::Applied(count)) => {
            info!(count, "applied variables from the login shell environment")
        }
        Some(shell_env::StartupOutcome::Fallback(count, err)) => warn!(
            count,
            error = format!("{err:#}"),
            "login shell environment unavailable; using a minimal login environment"
        ),
        Some(shell_env::StartupOutcome::NoLoginShell(reason)) => warn!(
            reason,
            "no login shell to read an environment from; using the inherited environment"
        ),
        Some(shell_env::StartupOutcome::NotApplied(err)) => warn!(
            error = format!("{err:#}"),
            "login shell environment not applied; using the inherited environment"
        ),
        Some(shell_env::StartupOutcome::Skipped) => {
            info!("login shell environment skipped (ORIEL_SKIP_SHELL_ENV)")
        }
        Some(shell_env::StartupOutcome::ResolvedByAttach) | None => {}
    }

    // Capture control socket path in the host XDG_RUNTIME_DIR before switching
    // the server process to its isolated session XDG_RUNTIME_DIR.
    let control_sock = session::control_socket_path(&host_runtime_dir);

    // There is deliberately no flag to supply a token: another local user could read it
    // from this process's arguments. Clients get one from `attach` instead.
    let token = generate_ephemeral_token()?;
    let initial_ttl = if cli.daemon {
        ATTACH_TOKEN_TTL
    } else {
        FOREGROUND_TOKEN_TTL
    };
    let creds = Arc::new(quic::generate_ephemeral_credentials(token, initial_ttl)?);

    // Bind before setting up the session, so a server that cannot get a port in the
    // requested range fails without starting anything.
    let endpoint = quic::bind_endpoint(creds.server_config.clone(), cli.listen)?;
    let port = endpoint
        .local_addr()
        .context("failed to read the bound QUIC address")?
        .port();

    let mut session_guard = session::SessionGuard::init(&host_runtime_dir)?;

    // Start the isolated virtual-only PipeWire + WirePlumber + pipewire-pulse stack eagerly
    // so GUI apps, Flatpaks, and remote PTYs immediately see live audio sockets.
    if let Err(err) = audio::ensure_virtual_audio_devices() {
        warn!(error = %err, "isolated PipeWire audio session unavailable at startup");
    }

    // Files the client puts on the clipboard appear here, streamed on demand.
    let xfer_mount = match oriel_proto::xfer_fuse::XferMount::mount(
        &session_guard.runtime_dir().join(session::XFER_MOUNT_DIR),
        tokio::runtime::Handle::current(),
    ) {
        Ok(mount) => {
            let mount = Arc::new(mount);
            session_guard.set_xfer_mount(Arc::clone(&mount));
            Some(mount)
        }
        Err(err) => {
            warn!(error = %format!("{err:#}"), "file transfer unavailable: cannot mount FUSE");
            None
        }
    };

    let (comp_handle, event_rx, comp_exited) =
        compositor::start_compositor(cli.command, !cli.no_xwayland, cli.gpu.mode())?;
    info!(
        socket = ?comp_handle.socket_name,
        session_dir = %session_guard.runtime_dir().display(),
        "headless Wayland compositor ready"
    );
    session_guard.set_wayland_socket_and_start_portal(&comp_handle.socket_name);
    // The initial command waits for this, so it reads settings from the session's portals
    // instead of D-Bus-activating a portal without the session's configuration.
    let _ = comp_handle
        .cmd_tx
        .send(compositor::CompositorCommand::SessionReady);

    let owns_control_sock = Arc::new(AtomicBool::new(false));
    let sock_creds = Arc::clone(&creds);
    let sock_path_task = control_sock.clone();
    let owns_flag = Arc::clone(&owns_control_sock);
    tokio::spawn(async move {
        if let Err(err) = run_control_socket(&sock_path_task, port, sock_creds, owns_flag).await {
            warn!(error = %err, "control Unix socket failed");
        }
    });

    let server_creds = quic::EphemeralCredentials {
        server_config: creds.server_config.clone(),
        cert_sha256_hex: creds.cert_sha256_hex.clone(),
        valid_tokens: Arc::clone(&creds.valid_tokens),
    };

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    // A closed terminal or SSH session sends SIGHUP, whose default action would skip the
    // session cleanup below.
    let mut sighup = signal(SignalKind::hangup())?;

    let server_res = tokio::select! {
        res = quic::run_server(endpoint, server_creds, comp_handle, event_rx, xfer_mount.clone()) => res,
        _ = sigterm.recv() => {
            info!("received SIGTERM; shutting down isolated session");
            Ok(())
        }
        _ = sigint.recv() => {
            info!("received SIGINT; shutting down isolated session");
            Ok(())
        }
        _ = sighup.recv() => {
            info!("received SIGHUP; shutting down isolated session");
            Ok(())
        }
        _ = comp_exited => {
            // Keeping the server up would leave `attach` handing out URIs for a session
            // that can show nothing; shut down so the next `attach` starts a fresh one.
            error!("headless compositor stopped; shutting down isolated session");
            Err(anyhow::anyhow!("headless compositor stopped"))
        }
    };

    if owns_control_sock.load(Ordering::SeqCst) {
        let _ = std::fs::remove_file(&control_sock);
    }
    // Unmounts the transfer filesystem after the session's apps are gone.
    drop(session_guard);

    server_res
}

async fn run_control_socket(
    sock_path: &Path,
    port: u16,
    creds: Arc<quic::EphemeralCredentials>,
    owns_control_sock: Arc<AtomicBool>,
) -> Result<()> {
    if sock_path.exists() {
        if UnixStream::connect(sock_path).await.is_ok() {
            warn!(
                path = %sock_path.display(),
                "another server owns the control socket; `oriel-server attach` will \
                 connect to that one, not this one"
            );
            return Ok(());
        }
        let _ = std::fs::remove_file(sock_path);
    }

    let listener = UnixListener::bind(sock_path)
        .with_context(|| format!("failed to bind control socket {}", sock_path.display()))?;
    session::set_mode(sock_path, 0o600)?;
    owns_control_sock.store(true, Ordering::SeqCst);
    info!(path = %sock_path.display(), "control Unix socket listening (mode 0600)");
    info!("run `oriel-server attach --host <addr>` to print a connect URI");

    loop {
        let (stream, _) = listener.accept().await?;
        let creds = Arc::clone(&creds);
        tokio::spawn(async move {
            if let Err(err) = answer_control_request(stream, port, &creds).await {
                warn!(error = %err, "control socket request failed");
            }
        });
    }
}

/// Mint a token for one `attach` on the control socket and send it the connect details.
async fn answer_control_request(
    stream: UnixStream,
    port: u16,
    creds: &quic::EphemeralCredentials,
) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let n = tokio::time::timeout(CONTROL_REQUEST_TIMEOUT, reader.read_line(&mut line))
        .await
        .context("timed out waiting for the control socket request")?
        .context("failed to read the control socket request")?;
    if n == 0 {
        // A server checking whether this socket is live connects and leaves.
        return Ok(());
    }
    let request: ControlSockRequest =
        serde_json::from_str(line.trim()).context("invalid control socket request")?;
    let client_fp = match request.client_fp {
        Some(fp) => Some(parse_cert_fingerprint(&fp).map_err(anyhow::Error::msg)?),
        None => None,
    };
    let token = generate_ephemeral_token().context("failed to generate CSPRNG session token")?;
    creds.add_token(token.clone(), ATTACH_TOKEN_TTL, client_fp);
    let info = ControlSockInfo {
        port,
        fp: creds.cert_sha256_hex.clone(),
        token,
        protocol: Some(oriel_proto::PROTOCOL_VERSION),
    };
    let mut json = serde_json::to_string(&info)?;
    json.push('\n');
    reader.get_mut().write_all(json.as_bytes()).await?;
    Ok(())
}

async fn run_attach(
    host_runtime_dir: &Path,
    listen: quic::ListenSpec,
    host: Option<String>,
    no_xwayland: bool,
    replace: bool,
    gpu: &GpuArgs,
    request: &ControlSockRequest,
) -> Result<()> {
    let sock_path = session::control_socket_path(host_runtime_dir);
    let mut running = query_control_socket(&sock_path, request).await.ok();
    if let Some(mismatch) = running.as_ref().and_then(protocol_mismatch) {
        let known = running.as_ref().is_some_and(|info| info.protocol.is_some());
        if !replace && !known {
            // It may well speak this version; only the client can tell.
            eprintln!(
                "warning: {mismatch}; if the client reports a protocol version mismatch, \
                 run `oriel-server attach --replace` to stop it (closing its apps) and \
                 start this version"
            );
        } else if !replace {
            anyhow::bail!(
                "{mismatch}, so clients of this version cannot connect to it. Run \
                 `oriel-server attach --replace` to stop it (closing its apps) and start \
                 this version, or `oriel-server stop`"
            );
        } else {
            eprintln!("{mismatch}; stopping it to start this version");
            stop_daemon(&sock_path).await?;
            running = None;
        }
    }
    if let Some(info) = running {
        if let Some(range) = listen.ports.filter(|r| !r.contains(info.port)) {
            anyhow::bail!(
                "a Oriel server is already running for this user on UDP port {}, \
                 outside the requested ports {range}; stop that server or allow port {}",
                info.port,
                info.port
            );
        }
        if gpu.no_gpu || gpu.render_node.is_some() {
            eprintln!(
                "warning: a Oriel server is already running for this user; \
                 --no-gpu/--render-node only apply when attach starts a new server"
            );
        }
        print_attach_uri(&host, listen, &info);
        return Ok(());
    }

    let _ = std::fs::remove_file(&sock_path);
    let exe = std::env::current_exe().context("failed to locate oriel-server binary")?;
    // Give the daemon the user's login environment: a one-off SSH command gets no login
    // shell, so it lacks /etc/profile.d additions such as Flatpak's XDG_DATA_DIRS. Resolve
    // it here rather than in the daemon so failures reach this command's stderr, and so the
    // daemon's startup isn't delayed past the wait below; the daemon is told to skip it.
    let (shell_env, shell_env_note) = match shell_env::resolve(&exe).await {
        shell_env::Resolution::Resolved(env) => {
            let note = format!("attach: using {} variables from the login shell", env.len());
            (Some(env), note)
        }
        shell_env::Resolution::Fallback { env, error } => {
            eprintln!("warning: {error:#}; starting the server with a minimal login environment");
            let note = format!("attach: login shell environment unavailable: {error:#}");
            (Some(env), note)
        }
        shell_env::Resolution::NoLoginShell(reason) => (
            None,
            format!("attach: no login shell ({reason}); keeping the SSH environment"),
        ),
        shell_env::Resolution::Skipped => (
            None,
            "attach: login shell environment skipped (ORIEL_SKIP_SHELL_ENV)".into(),
        ),
    };

    let log_path = session::daemon_log_path(host_runtime_dir);
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(&log_path)
        .ok();
    let _ = session::set_mode(&log_path, 0o600);
    // Where this daemon's output starts, to report its error if it fails to start.
    let log_start = log_file
        .as_ref()
        .and_then(|f| f.metadata().ok())
        .map_or(0, |m| m.len());
    if let Some(file) = log_file.as_mut() {
        use std::io::Write;
        let _ = writeln!(file, "{shell_env_note}");
    }

    let mut cmd = Command::new(exe);
    // The daemon's environment is the login one only: nothing from this SSH connection.
    if let Some(env) = &shell_env {
        cmd.env_clear().envs(env);
    }
    cmd.env(shell_env::ATTACH_RESOLVED_VAR, "1");
    cmd.arg("--daemon")
        .arg("--listen")
        .arg(listen.to_string())
        .stdin(Stdio::null());
    if no_xwayland {
        cmd.arg("--no-xwayland");
    }
    gpu.push_args(&mut cmd);
    if let Some(file) = log_file {
        match file.try_clone() {
            Ok(err_file) => {
                cmd.stdout(Stdio::from(file));
                cmd.stderr(Stdio::from(err_file));
            }
            Err(_) => {
                cmd.stdout(Stdio::from(file));
                cmd.stderr(Stdio::null());
            }
        }
    } else {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = cmd.spawn().context("failed to spawn oriel-server daemon")?;

    for _ in 0..80 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if let Ok(info) = query_control_socket(&sock_path, request).await {
            print_attach_uri(&host, listen, &info);
            return Ok(());
        }
        if let Ok(Some(status)) = child.try_wait() {
            match daemon_startup_error(&log_path, log_start) {
                Some(err) => anyhow::bail!("oriel-server failed to start: {err}"),
                None => anyhow::bail!(
                    "oriel-server exited during startup ({status}); see {}",
                    log_path.display()
                ),
            }
        }
    }

    anyhow::bail!(
        "timed out waiting for oriel-server daemon to start; see {}",
        log_path.display()
    )
}

/// The error a daemon printed on exit (`Error: ...` from `main`), from its part of the log.
fn daemon_startup_error(log_path: &Path, log_start: u64) -> Option<String> {
    let log = std::fs::read(log_path).ok()?;
    let ours = log.get(usize::try_from(log_start).ok()?..)?;
    let text = String::from_utf8_lossy(ours);
    let start = text
        .match_indices("Error: ")
        .map(|(i, _)| i)
        .filter(|&i| i == 0 || text.as_bytes()[i - 1] == b'\n')
        .last()?;
    let err = text[start + "Error: ".len()..].trim();
    (!err.is_empty()).then(|| err.to_string())
}

/// Why the daemon that sent `info` is not this binary's protocol version, if it is not.
fn protocol_mismatch(info: &ControlSockInfo) -> Option<String> {
    let ours = oriel_proto::PROTOCOL_VERSION;
    match info.protocol {
        Some(v) if v == ours => None,
        Some(v) => Some(format!(
            "the running Oriel server speaks protocol version {v}, this one {ours}"
        )),
        None => Some(format!(
            "the running Oriel server is from an older install that does not report its \
             protocol version (this one speaks {ours})"
        )),
    }
}

/// Stop the daemon that owns the control socket at `sock_path`: SIGTERM, then wait for it
/// to clean up its session and exit. Returns its pid, or `None` if no daemon is running.
///
/// The daemon is found through the socket's peer credentials rather than a request, so
/// daemons from any older install can be stopped too.
async fn stop_daemon(sock_path: &Path) -> Result<Option<i32>> {
    let Ok(stream) = UnixStream::connect(sock_path).await else {
        return Ok(None);
    };
    let cred = stream
        .peer_cred()
        .context("failed to read the Oriel server's credentials from its control socket")?;
    drop(stream);
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    let pid = match cred.pid() {
        Some(pid) if pid > 0 && cred.uid() == uid => pid,
        _ => anyhow::bail!(
            "the control socket {} is not owned by a server of this user",
            sock_path.display()
        ),
    };
    // SAFETY: plain syscall on a pid; a gone process gives ESRCH.
    if unsafe { libc::kill(pid, libc::SIGTERM) } != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return Ok(Some(pid));
        }
        return Err(err).with_context(|| format!("failed to stop the Oriel server (pid {pid})"));
    }
    let deadline = tokio::time::Instant::now() + STOP_TIMEOUT;
    while process_running(pid) {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "the Oriel server (pid {pid}) did not exit within {}s of SIGTERM",
                STOP_TIMEOUT.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(Some(pid))
}

/// Whether `pid` is still running. A process that has exited but not been reaped (its
/// parent does not wait for it) still exists as a zombie, and counts as stopped.
fn process_running(pid: i32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    if unsafe { libc::kill(pid, 0) } != 0 {
        return false;
    }
    // The state follows the parenthesized command name, which may itself contain ") ".
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let (_, rest) = stat.rsplit_once(')')?;
            rest.trim_start().chars().next()
        })
        .is_none_or(|state| state != 'Z' && state != 'X')
}

async fn query_control_socket(
    path: &Path,
    request: &ControlSockRequest,
) -> Result<ControlSockInfo> {
    let mut stream = UnixStream::connect(path).await?;
    let mut json = serde_json::to_string(request)?;
    json.push('\n');
    stream.write_all(json.as_bytes()).await?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let info: ControlSockInfo =
        serde_json::from_str(line.trim()).context("invalid control socket response")?;
    Ok(info)
}

fn print_attach_uri(host: &Option<String>, listen: quic::ListenSpec, info: &ControlSockInfo) {
    let host_str = host.clone().unwrap_or_else(|| match listen.ip {
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
        ip => ip.to_string(),
    });
    println!(
        "oriel://{}:{}?fp={}&token={}",
        host_str, info.port, info.fp, info.token
    );
}

/// Generate a 256-bit cryptographically random session token via `getrandom(2)`.
fn generate_ephemeral_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    session::fill_random_bytes(&mut bytes).context("failed to generate session token")?;

    let mut hex = String::with_capacity("oriel-".len() + bytes.len() * 2);
    hex.push_str("oriel-");
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csprng_tokens_have_256_bits_and_are_unique() {
        let t1 = generate_ephemeral_token().unwrap();
        let t2 = generate_ephemeral_token().unwrap();
        assert!(t1.starts_with("oriel-"));
        assert_eq!(t1.len(), "oriel-".len() + 64);
        assert_ne!(t1, t2);
    }

    #[test]
    fn exited_processes_count_as_stopped_before_they_are_reaped() {
        assert!(process_running(std::process::id() as i32));
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id() as i32;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        // Until it is waited for, the exited child is a zombie that signal 0 still finds.
        while process_running(pid) {
            assert!(std::time::Instant::now() < deadline, "child still running");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(unsafe { libc::kill(pid, 0) }, 0, "child was reaped early");
        child.wait().unwrap();
        assert!(!process_running(pid));
    }

    #[test]
    fn attach_spots_daemons_of_another_protocol_version() {
        let info = |json: &str| serde_json::from_str::<ControlSockInfo>(json).unwrap();
        let ours = oriel_proto::PROTOCOL_VERSION;
        let same = info(&format!(
            r#"{{"port":1,"fp":"f","token":"t","protocol":{ours}}}"#
        ));
        assert_eq!(protocol_mismatch(&same), None);
        let other = info(&format!(
            r#"{{"port":1,"fp":"f","token":"t","protocol":{}}}"#,
            ours + 1
        ));
        assert!(protocol_mismatch(&other)
            .unwrap()
            .contains("speaks protocol version"));
        // Daemons from before the field existed still parse, as of unknown version.
        let old = info(r#"{"port":1,"fp":"f","token":"t"}"#);
        assert_eq!(old.protocol, None);
        assert!(protocol_mismatch(&old).unwrap().contains("does not report"));
    }
}
