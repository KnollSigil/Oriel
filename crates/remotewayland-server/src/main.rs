//! RemoteWayland server entrypoint.
//!
//! Launches an isolated headless Wayland session + private D-Bus + virtual-only
//! PipeWire/WirePlumber audio bridge inside a private session `XDG_RUNTIME_DIR`
//! and streams rootless windows, audio, and remote PTYs over QUIC.

mod apps;
mod audio;
mod compositor;
mod gpu;
mod media;
mod pty;
mod quic;
mod session;
mod settings_portal;

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
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// TTL for single-use tokens minted via `attach` / control socket.
const ATTACH_TOKEN_TTL: Duration = Duration::from_secs(60);
/// TTL for the initial single-use token (e.g. one supplied via `--token`) in foreground mode.
const FOREGROUND_TOKEN_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Parser)]
#[command(
    name = "remotewayland-server",
    about = "RemoteWayland headless app & PTY server",
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

    /// Ephemeral session token (auto-generated via CSPRNG if omitted).
    #[arg(long)]
    token: Option<String>,

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

        #[command(flatten)]
        gpu: GpuArgs,
    },
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

#[derive(Debug, Serialize, Deserialize)]
struct ControlSockInfo {
    port: u16,
    fp: String,
    token: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate host XDG_RUNTIME_DIR up front (strict 0700 owned by current UID, no /tmp fallback).
    let host_runtime_dir = session::validate_host_runtime_dir()?;

    if let Some(ServerSubcommand::Attach {
        listen,
        host,
        no_xwayland,
        gpu,
    }) = cli.subcommand
    {
        return run_attach(&host_runtime_dir, listen, host, no_xwayland, &gpu).await;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?)
                // Render nodes never grant DRM master; smithay warns about it regardless.
                .add_directive("smithay::backend::drm::device::fd=error".parse()?),
        )
        .init();

    // Capture control socket path in the host XDG_RUNTIME_DIR before switching
    // the server process to its isolated session XDG_RUNTIME_DIR.
    let control_sock = session::control_socket_path(&host_runtime_dir);

    let token = match cli.token {
        Some(t) => t,
        None => generate_ephemeral_token()?,
    };
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

    let (comp_handle, event_rx) =
        compositor::start_compositor(cli.command, !cli.no_xwayland, cli.gpu.mode())?;
    info!(
        socket = ?comp_handle.socket_name,
        session_dir = %session_guard.runtime_dir().display(),
        "headless Wayland compositor ready"
    );
    session_guard.set_wayland_socket_and_start_portal(&comp_handle.socket_name);

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

    let server_res = tokio::select! {
        res = quic::run_server(endpoint, server_creds, comp_handle, event_rx) => res,
        _ = sigterm.recv() => {
            info!("received SIGTERM; shutting down isolated session");
            Ok(())
        }
        _ = sigint.recv() => {
            info!("received SIGINT; shutting down isolated session");
            Ok(())
        }
    };

    if owns_control_sock.load(Ordering::SeqCst) {
        let _ = std::fs::remove_file(&control_sock);
    }
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
                "another server owns the control socket; `remotewayland-server attach` will \
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
    info!("run `remotewayland-server attach --host <addr>` to print a connect URI");

    loop {
        let (mut stream, _) = listener.accept().await?;
        let token = match generate_ephemeral_token() {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, "failed to generate CSPRNG session token");
                continue;
            }
        };
        creds.add_token(token.clone(), ATTACH_TOKEN_TTL);
        let info = ControlSockInfo {
            port,
            fp: creds.cert_sha256_hex.clone(),
            token,
        };
        if let Ok(mut json) = serde_json::to_string(&info) {
            json.push('\n');
            let _ = stream.write_all(json.as_bytes()).await;
        }
    }
}

async fn run_attach(
    host_runtime_dir: &Path,
    listen: quic::ListenSpec,
    host: Option<String>,
    no_xwayland: bool,
    gpu: &GpuArgs,
) -> Result<()> {
    let sock_path = session::control_socket_path(host_runtime_dir);
    if let Ok(info) = query_control_socket(&sock_path).await {
        if let Some(range) = listen.ports.filter(|r| !r.contains(info.port)) {
            anyhow::bail!(
                "a RemoteWayland server is already running for this user on UDP port {}, \
                 outside the requested ports {range}; stop that server or allow port {}",
                info.port,
                info.port
            );
        }
        if gpu.no_gpu || gpu.render_node.is_some() {
            eprintln!(
                "warning: a RemoteWayland server is already running for this user; \
                 --no-gpu/--render-node only apply when attach starts a new server"
            );
        }
        print_attach_uri(&host, listen, &info);
        return Ok(());
    }

    let _ = std::fs::remove_file(&sock_path);
    let exe = std::env::current_exe().context("failed to locate remotewayland-server binary")?;
    let log_path = session::daemon_log_path(host_runtime_dir);
    let log_file = std::fs::OpenOptions::new()
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

    let mut cmd = Command::new(exe);
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
    let mut child = cmd
        .spawn()
        .context("failed to spawn remotewayland-server daemon")?;

    for _ in 0..80 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if let Ok(info) = query_control_socket(&sock_path).await {
            print_attach_uri(&host, listen, &info);
            return Ok(());
        }
        if let Ok(Some(status)) = child.try_wait() {
            match daemon_startup_error(&log_path, log_start) {
                Some(err) => anyhow::bail!("remotewayland-server failed to start: {err}"),
                None => anyhow::bail!(
                    "remotewayland-server exited during startup ({status}); see {}",
                    log_path.display()
                ),
            }
        }
    }

    anyhow::bail!(
        "timed out waiting for remotewayland-server daemon to start; see {}",
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

async fn query_control_socket(path: &Path) -> Result<ControlSockInfo> {
    let stream = UnixStream::connect(path).await?;
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
        "remotewayland://{}:{}?fp={}&token={}",
        host_str, info.port, info.fp, info.token
    );
}

/// Generate a 256-bit cryptographically random session token via `getrandom(2)`.
fn generate_ephemeral_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    session::fill_random_bytes(&mut bytes).context("failed to generate session token")?;

    let mut hex = String::with_capacity(3 + bytes.len() * 2);
    hex.push_str("rw-");
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
        assert!(t1.starts_with("rw-"));
        assert_eq!(t1.len(), 3 + 64);
        assert_ne!(t1, t2);
    }
}
