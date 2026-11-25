//! Local terminal <-> remote PTY QUIC / Unix-tunnel stream bridge with silent auto-reattach.

use std::ffi::{OsStr, OsString};
use std::fs::{self, DirBuilder, OpenOptions, Permissions};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use remotewayland_proto::{
    read_pty_message, write_bidi_stream_header, write_pty_message, BidiStreamHeader, PtyMessage,
};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::mpsc;

struct RawModeGuard {
    orig: Option<libc::termios>,
}

impl RawModeGuard {
    fn enter() -> Self {
        unsafe {
            if libc::isatty(libc::STDIN_FILENO) != 1 {
                return Self { orig: None };
            }
            let mut orig: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut orig) != 0 {
                return Self { orig: None };
            }
            let mut raw = orig;
            libc::cfmakeraw(&mut raw);
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
                return Self { orig: None };
            }
            Self { orig: Some(orig) }
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if let Some(orig) = &self.orig {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, orig);
            }
        }
    }
}

fn current_winsize() -> (u16, u16) {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ as _, &mut ws) == 0
            && ws.ws_col > 0
            && ws.ws_row > 0
        {
            return (ws.ws_col, ws.ws_row);
        }
    }
    (80, 24)
}

fn spawn_stdin_reader() -> Result<mpsc::Receiver<Vec<u8>>> {
    let (in_tx, in_rx) = mpsc::channel::<Vec<u8>>(64);
    std::thread::Builder::new()
        .name("rw-pty-stdin".into())
        .spawn(move || {
            let mut stdin = std::io::stdin().lock();
            let mut buf = [0u8; 4096];
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if in_tx.blocking_send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        })
        .context("failed to spawn PTY stdin reader")?;
    Ok(in_rx)
}

fn verify_private_dir(path: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(path)
        .with_context(|| format!("directory {} cannot be inspected", path.display()))?;
    let euid = unsafe { libc::geteuid() };
    if !meta.file_type().is_dir() || meta.uid() != euid || (meta.permissions().mode() & 0o077) != 0
    {
        bail!(
            "{} must be a 0700 directory owned by current uid (symlinks are not allowed)",
            path.display()
        );
    }
    Ok(())
}

fn ensure_private_dir(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path).is_err() {
        DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
    }
    let meta = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    let euid = unsafe { libc::geteuid() };
    if !meta.file_type().is_dir() || meta.uid() != euid {
        bail!(
            "{} must be a directory owned by current uid (symlinks are not allowed)",
            path.display()
        );
    }
    if (meta.permissions().mode() & 0o077) != 0 {
        fs::set_permissions(path, Permissions::from_mode(0o700))
            .with_context(|| format!("failed to chmod 0700 {}", path.display()))?;
    }
    Ok(())
}

fn parse_flatpak_info_app_id(contents: &str) -> Option<String> {
    let mut in_application = false;
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_application = line == "[Application]";
            continue;
        }
        if !in_application {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "name" {
                let id = v.trim();
                if is_valid_flatpak_app_id(id) {
                    return Some(id.to_string());
                }
            }
        }
    }
    None
}

fn is_valid_flatpak_app_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('-')
        && !id.starts_with('.')
        && !id.ends_with('.')
        && !id.contains("..")
        && id.contains('.')
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// Return the Flatpak application ID when running inside a Flatpak sandbox, or `None` otherwise.
fn flatpak_app_id() -> Option<String> {
    if !Path::new("/.flatpak-info").exists() {
        return None;
    }
    // Prefer the read-only kernel-mounted `/.flatpak-info` over the mutable `FLATPAK_ID` env var.
    if let Some(id) = fs::read_to_string("/.flatpak-info")
        .ok()
        .and_then(|contents| parse_flatpak_info_app_id(&contents))
    {
        return Some(id);
    }
    std::env::var("FLATPAK_ID")
        .ok()
        .map(|id| id.trim().to_string())
        .filter(|id| is_valid_flatpak_app_id(id))
}

fn pty_tunnel_socket_path_with(
    runtime_dir: &Path,
    flatpak_id: Option<&str>,
    session_key: &str,
) -> Result<PathBuf> {
    verify_private_dir(runtime_dir)?;

    let dir = match flatpak_id {
        Some(app_id) => {
            if !is_valid_flatpak_app_id(app_id) {
                bail!("invalid Flatpak application ID '{app_id}'");
            }
            // Flatpak bind-mounts `$XDG_RUNTIME_DIR/app/$FLATPAK_ID` from the host at the
            // exact same path across all instances of the app AND on the host itself, whereas
            // `$XDG_RUNTIME_DIR` itself is a private per-sandbox tmpfs.
            let app_dir = runtime_dir.join("app").join(app_id);
            ensure_private_dir(&app_dir)?;
            app_dir.join("remotewayland")
        }
        None => runtime_dir.join("remotewayland"),
    };
    ensure_private_dir(&dir)?;

    let mut sanitized: String = session_key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        sanitized.push_str("default");
    }
    Ok(dir.join(format!("pty-{sanitized}.sock")))
}

/// Compute the local `0600` Unix tunnel socket path for a connected host session
/// under `$XDG_RUNTIME_DIR/remotewayland/pty-<sanitized>.sock` (or
/// `$XDG_RUNTIME_DIR/app/$FLATPAK_ID/remotewayland/pty-<sanitized>.sock` inside Flatpak,
/// with no `/tmp` fallback).
pub fn pty_tunnel_socket_path(session_key: &str) -> Result<PathBuf> {
    let raw = std::env::var_os("XDG_RUNTIME_DIR")
        .context("XDG_RUNTIME_DIR is not set; cannot create private PTY tunnel socket")?;
    let base = PathBuf::from(&raw);
    let fp_id = flatpak_app_id();
    pty_tunnel_socket_path_with(&base, fp_id.as_deref(), session_key)
}

enum BridgeOutcome {
    /// Remote shell exited cleanly with this exit code (or user closed tab).
    Exited(i32),
    /// Server rejected attach because another terminal is already attached to this PTY.
    Busy,
    /// Transport dropped unexpectedly; caller may reconnect with `pty_id`.
    Disconnected,
}

async fn run_single_pty_stream<W, R>(
    mut send: W,
    mut recv: R,
    pty_id: &mut Option<u64>,
    received_bytes: &mut u64,
    ever_attached: &mut bool,
    command: Option<String>,
    in_rx: &mut mpsc::Receiver<Vec<u8>>,
    sigwinch: &mut tokio::signal::unix::Signal,
    sighup: &mut tokio::signal::unix::Signal,
) -> BridgeOutcome
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let (cols, rows) = current_winsize();
    if write_bidi_stream_header(
        &mut send,
        &BidiStreamHeader::Pty {
            pty_id: *pty_id,
            resume_offset: *received_bytes,
            cols,
            rows,
            command,
        },
    )
    .await
    .is_err()
    {
        return BridgeOutcome::Disconnected;
    }

    let mut stdout = std::io::stdout();
    loop {
        tokio::select! {
            msg = read_pty_message(&mut recv) => {
                match msg {
                    Ok(Some(PtyMessage::Attached {
                        pty_id: id,
                        stream_offset,
                    })) => {
                        *pty_id = Some(id);
                        *received_bytes = stream_offset;
                        *ever_attached = true;
                    }
                    Ok(Some(PtyMessage::Replay(bytes))) => {
                        let _ = stdout.write_all(&bytes);
                        let _ = stdout.flush();
                    }
                    Ok(Some(PtyMessage::Busy)) => {
                        return BridgeOutcome::Busy;
                    }
                    Ok(Some(PtyMessage::Data(bytes))) => {
                        *received_bytes = received_bytes.saturating_add(bytes.len() as u64);
                        let _ = stdout.write_all(&bytes);
                        let _ = stdout.flush();
                    }
                    Ok(Some(PtyMessage::Exit(code))) => {
                        return BridgeOutcome::Exited(code);
                    }
                    Ok(Some(PtyMessage::Resize { .. } | PtyMessage::Close)) => {}
                    Ok(None) | Err(_) => {
                        return BridgeOutcome::Disconnected;
                    }
                }
            }
            chunk = in_rx.recv() => {
                match chunk {
                    Some(bytes) => {
                        if write_pty_message(&mut send, &PtyMessage::Data(bytes)).await.is_err() {
                            return BridgeOutcome::Disconnected;
                        }
                    }
                    None => {
                        let _ = write_pty_message(&mut send, &PtyMessage::Close).await;
                        let _ = send.shutdown().await;
                        return BridgeOutcome::Exited(0);
                    }
                }
            }
            _ = sigwinch.recv() => {
                let (c, r) = current_winsize();
                if write_pty_message(&mut send, &PtyMessage::Resize { cols: c, rows: r }).await.is_err() {
                    return BridgeOutcome::Disconnected;
                }
            }
            _ = sighup.recv() => {
                let _ = write_pty_message(&mut send, &PtyMessage::Close).await;
                let _ = send.shutdown().await;
                return BridgeOutcome::Exited(0);
            }
        }
    }
}

/// Print `err`, and when stdin is a terminal, wait for Enter so a terminal window opened just
/// for this process stays open long enough to read it.
pub fn report_error_and_wait(err: &anyhow::Error) {
    use std::io::IsTerminal;
    eprintln!("remotewayland: {err:#}");
    if std::io::stdin().is_terminal() {
        eprint!("Press Enter to close.");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
}

/// Attach local stdin/stdout to a remote PTY over a fresh direct QUIC connection.
pub async fn run_pty_attach(
    server: SocketAddr,
    fingerprint: String,
    token: String,
    mut pty_id: Option<u64>,
    command: Option<String>,
) -> Result<i32> {
    let (conn, _ctrl_send, _ctrl_recv, _selected_codec) = crate::quic::connect_and_auth(
        server,
        fingerprint,
        token,
        remotewayland_proto::ClientRole::PtyOnly,
        false,
        1.0,
    )
    .await?;

    let (send, recv) = conn
        .open_bi()
        .await
        .context("failed to open QUIC PTY stream")?;

    let _raw_guard = RawModeGuard::enter();
    let mut in_rx = spawn_stdin_reader()?;
    let mut sigwinch = signal(SignalKind::window_change())?;
    let mut sighup = signal(SignalKind::hangup())?;
    let mut received_bytes = 0u64;
    let mut ever_attached = false;

    match run_single_pty_stream(
        send,
        recv,
        &mut pty_id,
        &mut received_bytes,
        &mut ever_attached,
        command,
        &mut in_rx,
        &mut sigwinch,
        &mut sighup,
    )
    .await
    {
        BridgeOutcome::Exited(code) => Ok(code),
        BridgeOutcome::Busy => {
            drop(_raw_guard);
            eprintln!("remotewayland: PTY is already attached to another terminal");
            Ok(1)
        }
        BridgeOutcome::Disconnected => Ok(1),
    }
}

/// Attach local stdin/stdout to a persistent remote PTY tunneled through an active
/// client's local Unix socket (`--socket`). Silently waits and auto-reattaches by
/// `pty_id` across network interruptions without injecting text into scrollback.
pub async fn run_pty_attach_socket(
    socket_path: &Path,
    initial_pty_id: Option<u64>,
    command: Option<String>,
) -> Result<i32> {
    let _raw_guard = RawModeGuard::enter();
    let mut in_rx = spawn_stdin_reader()?;
    let mut sigwinch = signal(SignalKind::window_change())?;
    let mut sighup = signal(SignalKind::hangup())?;

    let mut current_pty_id = initial_pty_id;
    let mut received_bytes = 0u64;
    let mut backoff = Duration::from_millis(200);
    let mut ever_attached = false;

    loop {
        let stream = match UnixStream::connect(socket_path).await {
            Ok(s) => {
                backoff = Duration::from_millis(200);
                s
            }
            Err(err) => {
                if !ever_attached && current_pty_id.is_none() {
                    bail!(
                        "failed to connect to local PTY tunnel socket {} (is the RemoteWayland session connected?): {err}",
                        socket_path.display()
                    );
                }
                // Silently wait for the main client to reconnect and re-bind the tunnel socket.
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = sighup.recv() => return Ok(0),
                }
                backoff = (backoff * 2).min(Duration::from_secs(2));
                continue;
            }
        };

        let (recv, send) = stream.into_split();
        let cmd = if current_pty_id.is_none() {
            command.clone()
        } else {
            None
        };

        let outcome = run_single_pty_stream(
            send,
            recv,
            &mut current_pty_id,
            &mut received_bytes,
            &mut ever_attached,
            cmd,
            &mut in_rx,
            &mut sigwinch,
            &mut sighup,
        )
        .await;

        match outcome {
            BridgeOutcome::Exited(code) => return Ok(code),
            BridgeOutcome::Busy => {
                if !ever_attached {
                    // Fresh terminal trying to attach to an already-attached PTY: reject immediately.
                    drop(_raw_guard);
                    eprintln!("remotewayland: PTY is already attached to another terminal");
                    return Ok(1);
                }
                // Reconnecting after a sudden network blackhole while the server's old QUIC
                // stream is still timing out: silently wait until it detaches.
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = sighup.recv() => return Ok(0),
                }
                backoff = (backoff * 2).min(Duration::from_secs(2));
            }
            BridgeOutcome::Disconnected => {
                // Tunnel dropped: stay completely silent and retry until the client reconnects.
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = sighup.recv() => return Ok(0),
                }
                backoff = (backoff * 2).min(Duration::from_secs(2));
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InitialTabMode {
    /// Initial tab uses the Konsole profile's `Command=` directly.
    ProfileDefault,
    /// Initial tab overrides the command via `-e` (e.g. reattaching `--pty-id` or one-shot `--command`).
    CustomCommand,
}

/// Spawn the user's preferred local terminal emulator attached to `remotewayland-client pty-attach --socket <path>`.
///
/// When launching `konsole`, also prepares an isolated per-session profile overlay
/// under `$XDG_RUNTIME_DIR/remotewayland/` so clicking Konsole's **New Tab** button
/// (`Ctrl+Shift+T` / `+`) opens another remote PTY on the same host by default without
/// modifying the user's `~/.config/konsolerc` or `~/.local/share/konsole/`.
#[cfg(not(feature = "gtk4"))]
pub fn spawn_local_terminal_pty_socket(
    socket_path: &Path,
    pty_id: Option<u64>,
    command: Option<&str>,
) -> Result<std::process::Child> {
    spawn_local_terminal_pty_socket_with_terminal(socket_path, pty_id, command, None)
}

/// Spawn a specific (or default when `terminal` is `None`) local terminal emulator
/// attached to `remotewayland-client pty-attach --socket <path>`.
pub fn spawn_local_terminal_pty_socket_with_terminal(
    socket_path: &Path,
    pty_id: Option<u64>,
    command: Option<&str>,
    terminal: Option<&str>,
) -> Result<std::process::Child> {
    let mut attach_args = vec![
        "pty-attach".to_string(),
        "--socket".to_string(),
        socket_path.to_string_lossy().into_owned(),
    ];
    if let Some(id) = pty_id {
        attach_args.push("--pty-id".to_string());
        attach_args.push(id.to_string());
    }
    if let Some(cmd) = command {
        attach_args.push("--command".to_string());
        attach_args.push(cmd.to_string());
    }
    let initial_mode = if pty_id.is_some() || command.is_some() {
        InitialTabMode::CustomCommand
    } else {
        InitialTabMode::ProfileDefault
    };
    spawn_local_terminal_with_args(&attach_args, Some(socket_path), initial_mode, terminal)
}

/// Spawn the user's preferred local terminal emulator running
/// `remotewayland-client pty-attach --ssh <target>`, which gets its own connect URI over SSH.
pub fn spawn_local_terminal_pty_ssh(
    ssh_target: &str,
    server_bin: Option<&str>,
    port: Option<remotewayland_proto::PortRange>,
    command: Option<&str>,
) -> Result<std::process::Child> {
    let mut attach_args = vec![
        "pty-attach".to_string(),
        "--ssh".to_string(),
        ssh_target.to_string(),
    ];
    if let Some(bin) = server_bin {
        attach_args.push("--server-bin".to_string());
        attach_args.push(bin.to_string());
    }
    if let Some(port) = port {
        attach_args.push("--port".to_string());
        attach_args.push(port.to_string());
    }
    if let Some(cmd) = command {
        attach_args.push("--command".to_string());
        attach_args.push(cmd.to_string());
    }
    spawn_local_terminal_with_args(&attach_args, None, InitialTabMode::CustomCommand, None)
}

#[derive(Debug, Default)]
struct XdgEnv {
    home: Option<PathBuf>,
    config_home: Option<PathBuf>,
    config_dirs: Option<OsString>,
    data_home: Option<PathBuf>,
    data_dirs: Option<OsString>,
}

impl XdgEnv {
    fn from_process() -> Self {
        let abs_dir = |key: &str| {
            std::env::var_os(key)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
        };
        Self {
            home: abs_dir("HOME"),
            config_home: abs_dir("XDG_CONFIG_HOME"),
            config_dirs: std::env::var_os("XDG_CONFIG_DIRS").filter(|v| !v.is_empty()),
            data_home: abs_dir("XDG_DATA_HOME"),
            data_dirs: std::env::var_os("XDG_DATA_DIRS").filter(|v| !v.is_empty()),
        }
    }

    fn effective_config_home(&self) -> Option<PathBuf> {
        self.config_home
            .clone()
            .or_else(|| self.home.as_ref().map(|h| h.join(".config")))
    }

    fn effective_data_home(&self) -> Option<PathBuf> {
        self.data_home
            .clone()
            .or_else(|| self.home.as_ref().map(|h| h.join(".local").join("share")))
    }
}

struct KonsoleProfileOverlay {
    config_home: PathBuf,
    config_dirs: String,
    data_home: PathBuf,
    data_dirs: String,
}

fn quote_konsole_arg(arg: &str) -> String {
    let needs_quote = arg.is_empty()
        || arg.chars().any(|c| {
            c.is_whitespace()
                || matches!(
                    c,
                    '\'' | '"'
                        | '\\'
                        | '$'
                        | '`'
                        | ';'
                        | '&'
                        | '|'
                        | '<'
                        | '>'
                        | '('
                        | ')'
                        | '{'
                        | '}'
                        | '*'
                        | '?'
                        | '['
                        | ']'
                        | '~'
                        | '#'
                )
        });
    if !needs_quote {
        return arg.to_string();
    }
    format!("'{}'", arg.replace('\'', "'\\''"))
}

/// Escape a string for a KConfig INI value (KConfig unescapes `\\`, `\n`, `\r`, `\t`
/// before Konsole passes `Command=` to `KShell::splitArgs`).
fn escape_kconfig_value(val: &str) -> String {
    let mut out = String::with_capacity(val.len());
    for c in val.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

fn format_konsole_command_with_prefix(cmd_prefix: &[String], socket_path: &Path) -> Result<String> {
    if cmd_prefix.is_empty() {
        bail!("Konsole command prefix cannot be empty");
    }
    let sock_str = socket_path
        .to_str()
        .context("PTY socket path is not valid UTF-8")?;
    for token in cmd_prefix.iter().map(String::as_str).chain([sock_str]) {
        if token.chars().any(|c| matches!(c, '\0' | '\n' | '\r')) {
            bail!("path contains control characters unsupported in Konsole profile");
        }
    }
    let shell_cmd = cmd_prefix
        .iter()
        .map(String::as_str)
        .chain(["pty-attach", "--socket", sock_str])
        .map(quote_konsole_arg)
        .collect::<Vec<_>>()
        .join(" ");
    Ok(escape_kconfig_value(&shell_cmd))
}

#[cfg(test)]
fn format_konsole_command(exe: &Path, socket_path: &Path) -> Result<String> {
    let exe_str = exe
        .to_str()
        .context("remotewayland-client executable path is not valid UTF-8")?;
    format_konsole_command_with_prefix(&[exe_str.to_string()], socket_path)
}

fn parse_konsole_default_profile(konsolerc: &str) -> Option<String> {
    let mut in_desktop_entry = false;
    let mut found: Option<String> = None;
    for raw_line in konsolerc.split(['\n', '\r']) {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let section = line.strip_suffix("[$i]").unwrap_or(line);
            in_desktop_entry = section == "[Desktop Entry]";
            continue;
        }
        if !in_desktop_entry {
            continue;
        }
        if let Some((raw_key, raw_val)) = line.split_once('=') {
            let key = raw_key.trim();
            let key = key.strip_suffix("[$i]").unwrap_or(key);
            if key == "DefaultProfile" {
                let val = raw_val.trim();
                let invalid = val.is_empty()
                    || val == "RemoteWayland.profile"
                    || val == "RemoteWayland"
                    || val.contains('/')
                    || val.contains('\\')
                    || val.contains("..")
                    || val.chars().any(|c| c.is_control());
                if !invalid {
                    found = Some(val.to_string());
                } else {
                    found = None;
                }
            }
        }
    }
    found
}

fn join_xdg_dirs(primary: Option<&Path>, existing: Option<&OsString>, fallback: &str) -> String {
    let mut dirs: Vec<String> = Vec::new();
    if let Some(p) = primary {
        dirs.push(p.to_string_lossy().into_owned());
    }
    match existing.filter(|v| !v.is_empty()) {
        Some(v) => dirs.push(v.to_string_lossy().into_owned()),
        None => dirs.push(fallback.to_string()),
    }
    dirs.join(":")
}

fn write_private_file_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{name}.{pid}.{seq}.tmp"));

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&tmp)
        .with_context(|| format!("failed to create temp file {}", tmp.display()))?;
    if let Err(err) = file.write_all(contents) {
        let _ = fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("failed to write {}", tmp.display()));
    }
    drop(file);
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("failed to rename {} -> {}", tmp.display(), path.display()));
    }
    Ok(())
}

fn prepare_konsole_remote_profile(
    cmd_prefix: &[String],
    socket_path: &Path,
    flatpak: bool,
) -> Result<KonsoleProfileOverlay> {
    prepare_konsole_remote_profile_with_env(
        cmd_prefix,
        socket_path,
        &XdgEnv::from_process(),
        flatpak,
    )
}

fn prepare_konsole_remote_profile_with_env(
    cmd_prefix: &[String],
    socket_path: &Path,
    env: &XdgEnv,
    flatpak: bool,
) -> Result<KonsoleProfileOverlay> {
    let parent_dir = socket_path
        .parent()
        .context("PTY socket path has no parent directory")?;
    verify_private_dir(parent_dir)?;

    let stem = socket_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default");

    let overlay_root = parent_dir.join(format!("konsole-{stem}"));
    let config_home = overlay_root.join("config");
    let data_home = overlay_root.join("data");
    let profiles_dir = data_home.join("konsole");

    for dir in [&overlay_root, &config_home, &data_home, &profiles_dir] {
        ensure_private_dir(dir)?;
    }

    // Inside Flatpak, XDG_CONFIG_HOME/XDG_DATA_HOME/XDG_*_DIRS point to sandbox-private
    // paths (`~/.var/app/<id>/...`, `/app/share`). Use the host user's standard XDG
    // directories so host Konsole inherits their themes and profiles instead.
    let (orig_config_home, orig_data_home, existing_config_dirs, existing_data_dirs) = if flatpak {
        (
            env.home.as_ref().map(|h| h.join(".config")),
            env.home.as_ref().map(|h| h.join(".local").join("share")),
            None,
            None,
        )
    } else {
        (
            env.effective_config_home(),
            env.effective_data_home(),
            env.config_dirs.as_ref(),
            env.data_dirs.as_ref(),
        )
    };

    let parent_profile = orig_config_home
        .as_deref()
        .and_then(|dir| fs::read_to_string(dir.join("konsolerc")).ok())
        .and_then(|content| parse_konsole_default_profile(&content))
        .unwrap_or_else(|| "FALLBACK/".to_string());

    let cmd = format_konsole_command_with_prefix(cmd_prefix, socket_path)?;
    let profile_contents =
        format!("[General]\nCommand={cmd}\nName=RemoteWayland\nParent={parent_profile}\n");
    write_private_file_atomic(
        &profiles_dir.join("RemoteWayland.profile"),
        profile_contents.as_bytes(),
    )?;

    let konsolerc_contents = "[Desktop Entry]\nDefaultProfile=RemoteWayland.profile\n";
    write_private_file_atomic(
        &config_home.join("konsolerc"),
        konsolerc_contents.as_bytes(),
    )?;

    let config_dirs = join_xdg_dirs(
        orig_config_home.as_deref(),
        existing_config_dirs,
        "/etc/xdg",
    );
    // The sandbox's XDG_DATA_DIRS is useless on the host, and the profile's `flatpak run`
    // inherits this value: without Flatpak's export dirs the host `flatpak` CLI prints a
    // "directories are not in the search path set by XDG_DATA_DIRS" warning into the PTY.
    let flatpak_data_dirs_fallback = env.home.as_ref().map(|h| {
        format!(
            "{}:/var/lib/flatpak/exports/share:/usr/local/share:/usr/share",
            h.join(".local/share/flatpak/exports/share").display()
        )
    });
    let data_dirs_fallback = match (flatpak, flatpak_data_dirs_fallback.as_deref()) {
        (true, Some(dirs)) => dirs,
        (true, None) => "/var/lib/flatpak/exports/share:/usr/local/share:/usr/share",
        (false, _) => "/usr/local/share:/usr/share",
    };
    let data_dirs = join_xdg_dirs(
        orig_data_home.as_deref(),
        existing_data_dirs,
        data_dirs_fallback,
    );

    Ok(KonsoleProfileOverlay {
        config_home,
        config_dirs,
        data_home,
        data_dirs,
    })
}

fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
}

fn executable_exists_in_path(program: &str, path_var: Option<&OsStr>) -> bool {
    let path = Path::new(program);
    if path.components().count() > 1 {
        return is_executable_file(path);
    }
    let Some(path_var) = path_var else {
        return false;
    };
    std::env::split_paths(path_var).any(|dir| is_executable_file(&dir.join(program)))
}

fn executable_exists(program: &str) -> bool {
    let path_var = std::env::var_os("PATH");
    executable_exists_in_path(program, path_var.as_deref())
}

const KNOWN_TERMINALS: &[&str] = &[
    "konsole",
    "alacritty",
    "kitty",
    "foot",
    "ghostty",
    "wezterm",
    "gnome-terminal",
    "xfce4-terminal",
    "lxterminal",
    "xterm",
];

/// Extract the display/matching executable basename for a terminal command or path.
pub(crate) fn terminal_basename(term: &str) -> &str {
    Path::new(term)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(term)
}

fn terminal_candidates_with(terminal_env: Option<&str>) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(trimmed) = terminal_env.map(str::trim).filter(|s| !s.is_empty()) {
        candidates.push(trimmed.to_string());
    }
    for term in KNOWN_TERMINALS {
        if !candidates.iter().any(|c| terminal_basename(c) == *term) {
            candidates.push((*term).to_string());
        }
    }
    candidates
}

#[cfg(any(feature = "gtk4", test))]
fn terminal_candidates() -> Vec<String> {
    let env_term = std::env::var("TERMINAL").ok();
    terminal_candidates_with(env_term.as_deref())
}

/// Resolve the ordered candidate list to try, validating `preferred_terminal` against
/// the configured allowlist (`$TERMINAL` + `KNOWN_TERMINALS`) so external callers
/// (e.g. GTK D-Bus `win.launch-terminal` actions) cannot execute arbitrary binaries.
fn resolve_terminal_candidates(
    preferred_terminal: Option<&str>,
    terminal_env: Option<&str>,
) -> Result<Vec<String>> {
    let allowed = terminal_candidates_with(terminal_env);
    match preferred_terminal.map(str::trim).filter(|s| !s.is_empty()) {
        Some(term) => {
            if allowed.iter().any(|a| a == term) {
                Ok(vec![term.to_string()])
            } else {
                bail!("terminal emulator '{term}' is not in the allowed terminal list");
            }
        }
        None => Ok(allowed),
    }
}

/// Probe which candidate terminal executables exist on the host via `flatpak-spawn --host`
/// in a single batched invocation with a bounded timeout, preserving candidate priority order.
fn host_existing_executables(candidates: &[String]) -> Vec<String> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut child = match Command::new("flatpak-spawn")
        .arg("--host")
        .arg("--")
        .args([
            "sh",
            "-c",
            "for t in \"$@\"; do command -v -- \"$t\" >/dev/null 2>&1 && printf '%s\\n' \"$t\"; done",
            "_",
        ])
        .args(candidates)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(err) => {
            tracing::warn!(error = %err, "failed to invoke flatpak-spawn --host for terminal probe");
            return Vec::new();
        }
    };

    let Some(mut stdout_pipe) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Vec::new();
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(1);
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout_pipe.read_to_string(&mut buf);
        let _ = tx.send(buf);
    });

    let stdout = match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(out) => out,
        Err(_) => {
            tracing::warn!("flatpak-spawn --host terminal probe timed out");
            let _ = child.kill();
            let _ = child.wait();
            return Vec::new();
        }
    };

    match child.wait() {
        Ok(status) if status.success() => {
            let found: std::collections::HashSet<&str> = stdout.lines().collect();
            candidates
                .iter()
                .filter(|c| found.contains(c.as_str()))
                .cloned()
                .collect()
        }
        Ok(status) => {
            tracing::warn!(%status, "flatpak-spawn --host terminal probe failed");
            Vec::new()
        }
        Err(err) => {
            tracing::warn!(error = %err, "failed to wait for flatpak-spawn --host terminal probe");
            Vec::new()
        }
    }
}

/// Build the argv prefix used by a local terminal emulator to invoke `remotewayland-client`.
fn client_attach_prefix(flatpak_id: Option<&str>) -> Result<Vec<String>> {
    if let Some(app_id) = flatpak_id {
        return Ok(vec![
            "flatpak".to_string(),
            "run".to_string(),
            "--command=remotewayland-client".to_string(),
            app_id.to_string(),
        ]);
    }
    let exe = std::env::current_exe().context("failed to locate remotewayland-client binary")?;
    let exe_str = exe
        .to_str()
        .context("remotewayland-client executable path is not valid UTF-8")?;
    Ok(vec![exe_str.to_string()])
}

/// Return installed local terminal emulators in priority order (index 0 is the default).
#[cfg(any(feature = "gtk4", test))]
pub fn available_local_terminals() -> Vec<String> {
    let candidates = terminal_candidates();
    if flatpak_app_id().is_some() {
        host_existing_executables(&candidates)
    } else {
        candidates
            .into_iter()
            .filter(|term| executable_exists(term))
            .collect()
    }
}

/// Wrap the `flatpak run` attach prefix for Konsole's profile overlay so the overridden
/// `XDG_DATA_HOME` (pointing at our temporary overlay dir) does not prevent `flatpak run`
/// from finding `--user` installations under `~/.local/share/flatpak`.
///
/// `XDG_DATA_DIRS` is deliberately kept: the overlay's value includes Flatpak's export
/// dirs, whereas unsetting it makes `flatpak` warn that they are missing from the path.
fn konsole_overlay_attach_prefix(attach_prefix: &[String], is_flatpak: bool) -> Vec<String> {
    if !is_flatpak {
        return attach_prefix.to_vec();
    }
    let mut wrapped = vec![
        "env".to_string(),
        "-u".to_string(),
        "XDG_CONFIG_HOME".to_string(),
        "-u".to_string(),
        "XDG_CONFIG_DIRS".to_string(),
        "-u".to_string(),
        "XDG_DATA_HOME".to_string(),
    ];
    wrapped.extend(attach_prefix.iter().cloned());
    wrapped
}

fn spawn_local_terminal_with_args(
    attach_args: &[String],
    new_tab_socket: Option<&Path>,
    initial_mode: InitialTabMode,
    preferred_terminal: Option<&str>,
) -> Result<std::process::Child> {
    let fp_id = flatpak_app_id();
    let is_flatpak = fp_id.is_some();
    let attach_prefix = client_attach_prefix(fp_id.as_deref())?;
    let env_term = std::env::var("TERMINAL").ok();
    let all_candidates = resolve_terminal_candidates(preferred_terminal, env_term.as_deref())?;
    let candidates = if is_flatpak {
        host_existing_executables(&all_candidates)
    } else {
        all_candidates
            .into_iter()
            .filter(|term| executable_exists(term))
            .collect()
    };
    let mut last_err: Option<(String, std::io::Error)> = None;

    for term in candidates {
        let mut env_overrides: Vec<(String, String)> = Vec::new();
        let mut term_args: Vec<String> = Vec::new();
        let base = terminal_basename(&term);
        match base {
            "konsole" => {
                if let Some(sock) = new_tab_socket {
                    // Scope XDG_CONFIG_HOME / XDG_DATA_HOME to this `--separate` Konsole
                    // instance so New Tab defaults to a remote PTY while `pty-attach` talks
                    // only over `--socket` and never leaks these vars to the remote shell.
                    let konsole_prefix = konsole_overlay_attach_prefix(&attach_prefix, is_flatpak);
                    match prepare_konsole_remote_profile(&konsole_prefix, sock, is_flatpak) {
                        Ok(overlay) => {
                            env_overrides.push((
                                "XDG_CONFIG_HOME".to_string(),
                                overlay.config_home.to_string_lossy().into_owned(),
                            ));
                            env_overrides
                                .push(("XDG_CONFIG_DIRS".to_string(), overlay.config_dirs));
                            env_overrides.push((
                                "XDG_DATA_HOME".to_string(),
                                overlay.data_home.to_string_lossy().into_owned(),
                            ));
                            env_overrides.push(("XDG_DATA_DIRS".to_string(), overlay.data_dirs));
                            term_args.push("--separate".to_string());
                            term_args.push("--profile".to_string());
                            term_args.push("RemoteWayland".to_string());
                            if initial_mode == InitialTabMode::CustomCommand {
                                term_args.push("-e".to_string());
                                term_args.extend(konsole_prefix.iter().cloned());
                                term_args.extend(attach_args.iter().cloned());
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                "failed to prepare Konsole remote profile overlay; falling back to -e"
                            );
                            term_args.push("-e".to_string());
                            term_args.extend(attach_prefix.iter().cloned());
                            term_args.extend(attach_args.iter().cloned());
                        }
                    }
                } else {
                    term_args.push("-e".to_string());
                    term_args.extend(attach_prefix.iter().cloned());
                    term_args.extend(attach_args.iter().cloned());
                }
            }
            "gnome-terminal" => {
                term_args.push("--".to_string());
                term_args.extend(attach_prefix.iter().cloned());
                term_args.extend(attach_args.iter().cloned());
            }
            "wezterm" => {
                term_args.push("start".to_string());
                term_args.push("--".to_string());
                term_args.extend(attach_prefix.iter().cloned());
                term_args.extend(attach_args.iter().cloned());
            }
            _ => {
                term_args.push("-e".to_string());
                term_args.extend(attach_prefix.iter().cloned());
                term_args.extend(attach_args.iter().cloned());
            }
        }

        let mut proc = if is_flatpak {
            let mut cmd = Command::new("flatpak-spawn");
            cmd.arg("--host");
            for (k, v) in &env_overrides {
                cmd.arg(format!("--env={k}={v}"));
            }
            cmd.arg("--").arg(&term).args(&term_args);
            cmd
        } else {
            let mut cmd = Command::new(&term);
            for (k, v) in &env_overrides {
                cmd.env(k, v);
            }
            cmd.args(&term_args);
            cmd
        };

        match proc.spawn() {
            Ok(child) => return Ok(child),
            Err(err) => last_err = Some((term, err)),
        }
    }

    if let Some((term, err)) = last_err {
        bail!("failed to launch terminal emulator '{term}': {err}");
    }
    if let Some(term) = preferred_terminal.map(str::trim).filter(|s| !s.is_empty()) {
        bail!("terminal emulator '{term}' not found");
    }
    anyhow::bail!(
        "no local terminal emulator found ($TERMINAL, {})",
        KNOWN_TERMINALS.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_konsole_command_quotes_and_escapes_kconfig() {
        assert_eq!(
            format_konsole_command(
                Path::new("/usr/bin/remotewayland-client"),
                Path::new("/run/user/1000/remotewayland/pty-a.sock"),
            )
            .unwrap(),
            "/usr/bin/remotewayland-client pty-attach --socket /run/user/1000/remotewayland/pty-a.sock"
        );
        // Single-quote shell escaping (`'\''`) must also have its backslash escaped for KConfig (`'\\''`).
        assert_eq!(
            format_konsole_command(
                Path::new("/tmp/my app/remotewayland-client"),
                Path::new("/run/user/1000/a'b.sock"),
            )
            .unwrap(),
            "'/tmp/my app/remotewayland-client' pty-attach --socket '/run/user/1000/a'\\\\''b.sock'"
        );
        // Leading ~ and # must be quoted for KShell::splitArgs.
        assert_eq!(
            format_konsole_command(Path::new("~user/bin/rw"), Path::new("/tmp/#test.sock"))
                .unwrap(),
            "'~user/bin/rw' pty-attach --socket '/tmp/#test.sock'"
        );
        // Control characters (newlines) must be rejected.
        assert!(
            format_konsole_command(Path::new("/bin/rw\nevil"), Path::new("/tmp/a.sock")).is_err()
        );
    }

    #[test]
    fn parse_konsole_default_profile_handles_last_wins_immutability_and_sanitization() {
        let sample = "\
            # comment
            [General]
            DefaultProfile=Wrong.profile

            [Desktop Entry][$i]
            DefaultProfile=First.profile
            DefaultProfile[$i]=MyCustom.profile
        ";
        assert_eq!(
            parse_konsole_default_profile(sample).as_deref(),
            Some("MyCustom.profile")
        );
        assert_eq!(
            parse_konsole_default_profile(
                "[Desktop Entry]\nDefaultProfile=RemoteWayland.profile\n"
            ),
            None
        );
        assert_eq!(
            parse_konsole_default_profile("[Desktop Entry]\nDefaultProfile=RemoteWayland\n"),
            None
        );
        assert_eq!(
            parse_konsole_default_profile("[Desktop Entry]\nDefaultProfile=../evil.profile\n"),
            None
        );
        assert_eq!(
            parse_konsole_default_profile("[General]\nConfigVersion=1\n"),
            None
        );
    }

    #[test]
    fn join_xdg_dirs_handles_all_combinations() {
        let p = Path::new("/home/u/.config");
        let existing = OsString::from("/etc/xdg/custom");
        assert_eq!(
            join_xdg_dirs(Some(p), Some(&existing), "/etc/xdg"),
            "/home/u/.config:/etc/xdg/custom"
        );
        assert_eq!(
            join_xdg_dirs(Some(p), None, "/etc/xdg"),
            "/home/u/.config:/etc/xdg"
        );
        assert_eq!(join_xdg_dirs(None, None, "/etc/xdg"), "/etc/xdg");
    }

    #[test]
    fn prepare_konsole_remote_profile_writes_private_overlay_and_inherits_parent() {
        let dir = std::env::temp_dir().join(format!("rw-konsole-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        ensure_private_dir(&dir).unwrap();

        let fake_cfg = dir.join("user_cfg");
        let fake_data = dir.join("user_data");
        fs::create_dir_all(&fake_cfg).unwrap();
        fs::create_dir_all(&fake_data).unwrap();
        fs::write(
            fake_cfg.join("konsolerc"),
            "[Desktop Entry]\nDefaultProfile=UserTheme.profile\n",
        )
        .unwrap();

        let env = XdgEnv {
            home: None,
            config_home: Some(fake_cfg.clone()),
            config_dirs: None,
            data_home: Some(fake_data.clone()),
            data_dirs: Some(OsString::from("/usr/share")),
        };

        let sock = dir.join("pty-test.sock");
        let exe = vec!["/usr/local/bin/remotewayland-client".to_string()];

        let overlay = prepare_konsole_remote_profile_with_env(&exe, &sock, &env, false).unwrap();
        assert_eq!(
            overlay.config_dirs,
            format!("{}:/etc/xdg", fake_cfg.display())
        );
        assert_eq!(
            overlay.data_dirs,
            format!("{}:/usr/share", fake_data.display())
        );

        let konsolerc_path = overlay.config_home.join("konsolerc");
        let konsolerc = fs::read_to_string(&konsolerc_path).unwrap();
        assert!(konsolerc.contains("DefaultProfile=RemoteWayland.profile"));
        assert_eq!(
            fs::symlink_metadata(&konsolerc_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );

        let profile_path = overlay
            .data_home
            .join("konsole")
            .join("RemoteWayland.profile");
        let profile = fs::read_to_string(&profile_path).unwrap();
        assert!(profile.contains("Name=RemoteWayland"));
        assert!(profile.contains("Parent=UserTheme.profile"));
        assert!(profile.contains(&format!(
            "Command=/usr/local/bin/remotewayland-client pty-attach --socket {}",
            sock.display()
        )));
        assert_eq!(
            fs::symlink_metadata(&profile_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert_eq!(
            fs::symlink_metadata(&overlay.config_home)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn terminal_candidates_deduplicates_by_basename_and_validates_allowlist() {
        let list = terminal_candidates_with(Some("  /usr/bin/konsole  "));
        assert_eq!(list[0], "/usr/bin/konsole");
        assert_eq!(
            list.iter()
                .filter(|c| terminal_basename(c) == "konsole")
                .count(),
            1
        );

        // Allowlist accepts $TERMINAL and KNOWN_TERMINALS, rejects arbitrary binaries.
        assert_eq!(
            resolve_terminal_candidates(Some("/usr/bin/konsole"), Some("/usr/bin/konsole"))
                .unwrap(),
            vec!["/usr/bin/konsole".to_string()]
        );
        assert_eq!(
            resolve_terminal_candidates(Some("xterm"), None).unwrap(),
            vec!["xterm".to_string()]
        );
        assert!(resolve_terminal_candidates(Some("/bin/sh"), None).is_err());
    }

    #[test]
    fn executable_exists_in_path_checks_execute_bits() {
        let dir = std::env::temp_dir().join(format!("rw-exec-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let non_exec = dir.join("not-exec");
        fs::write(&non_exec, b"#!/bin/sh\n").unwrap();
        fs::set_permissions(&non_exec, Permissions::from_mode(0o644)).unwrap();

        let exec = dir.join("is-exec");
        fs::write(&exec, b"#!/bin/sh\n").unwrap();
        fs::set_permissions(&exec, Permissions::from_mode(0o755)).unwrap();

        let path_env = OsString::from(&dir);
        assert!(!executable_exists_in_path(
            "not-exec",
            Some(path_env.as_os_str())
        ));
        assert!(executable_exists_in_path(
            "is-exec",
            Some(path_env.as_os_str())
        ));
        assert!(!executable_exists_in_path(non_exec.to_str().unwrap(), None));
        assert!(executable_exists_in_path(exec.to_str().unwrap(), None));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_flatpak_info_app_id_reads_application_name_and_validates() {
        let sample = "\
[Application]
name=dev.knoll.Oriel
runtime=runtime/org.gnome.Platform/x86_64/51

[Instance]
name=other
";
        assert_eq!(
            parse_flatpak_info_app_id(sample).as_deref(),
            Some("dev.knoll.Oriel")
        );
        assert_eq!(
            parse_flatpak_info_app_id("[Application]\nname=../evil\n"),
            None
        );
        assert_eq!(
            parse_flatpak_info_app_id("[Instance]\nname=dev.knoll.Oriel\n"),
            None
        );
        assert!(is_valid_flatpak_app_id("dev.knoll.Oriel"));
        assert!(!is_valid_flatpak_app_id("."));
        assert!(!is_valid_flatpak_app_id(".."));
        assert!(!is_valid_flatpak_app_id("--flag"));
        assert!(!is_valid_flatpak_app_id("no_dot_id"));
        assert!(!is_valid_flatpak_app_id(".leading.dot"));
        assert!(!is_valid_flatpak_app_id("trailing.dot."));
    }

    #[test]
    fn pty_tunnel_socket_path_uses_shared_flatpak_app_dir() {
        let dir = std::env::temp_dir().join(format!("rw-fp-sock-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        ensure_private_dir(&dir).unwrap();

        let native = pty_tunnel_socket_path_with(&dir, None, "user@host").unwrap();
        assert_eq!(native, dir.join("remotewayland/pty-user_host.sock"));

        let fp = pty_tunnel_socket_path_with(&dir, Some("dev.knoll.Oriel"), "user@host").unwrap();
        assert_eq!(
            fp,
            dir.join("app/dev.knoll.Oriel/remotewayland/pty-user_host.sock")
        );

        assert!(pty_tunnel_socket_path_with(&dir, Some("../bad"), "user@host").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flatpak_konsole_profile_uses_flatpak_run_and_host_xdg_dirs() {
        let dir = std::env::temp_dir().join(format!("rw-fp-konsole-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        ensure_private_dir(&dir).unwrap();

        let env = XdgEnv {
            home: Some(PathBuf::from("/home/alice")),
            config_home: Some(PathBuf::from("/home/alice/.var/app/dev.knoll.Oriel/config")),
            config_dirs: Some(OsString::from("/app/etc/xdg:/etc/xdg")),
            data_home: Some(PathBuf::from("/home/alice/.var/app/dev.knoll.Oriel/data")),
            data_dirs: Some(OsString::from("/app/share:/usr/share")),
        };

        let sock = dir.join("pty-host.sock");
        let prefix = client_attach_prefix(Some("dev.knoll.Oriel")).unwrap();
        assert_eq!(
            prefix,
            vec![
                "flatpak",
                "run",
                "--command=remotewayland-client",
                "dev.knoll.Oriel"
            ]
        );

        let konsole_prefix = konsole_overlay_attach_prefix(&prefix, true);
        let overlay =
            prepare_konsole_remote_profile_with_env(&konsole_prefix, &sock, &env, true).unwrap();
        assert_eq!(overlay.config_dirs, "/home/alice/.config:/etc/xdg");
        assert_eq!(
            overlay.data_dirs,
            "/home/alice/.local/share:/home/alice/.local/share/flatpak/exports/share:/var/lib/flatpak/exports/share:/usr/local/share:/usr/share"
        );

        let profile = fs::read_to_string(
            overlay
                .data_home
                .join("konsole")
                .join("RemoteWayland.profile"),
        )
        .unwrap();
        assert!(profile.contains(&format!(
            "Command=env -u XDG_CONFIG_HOME -u XDG_CONFIG_DIRS -u XDG_DATA_HOME flatpak run --command=remotewayland-client dev.knoll.Oriel pty-attach --socket {}",
            sock.display()
        )));

        let _ = fs::remove_dir_all(&dir);
    }
}
