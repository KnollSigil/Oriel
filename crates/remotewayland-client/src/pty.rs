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

/// Compute the local `0600` Unix tunnel socket path for a connected host session
/// under `$XDG_RUNTIME_DIR/remotewayland/pty-<sanitized>.sock` (no `/tmp` fallback).
pub fn pty_tunnel_socket_path(session_key: &str) -> Result<PathBuf> {
    let raw = std::env::var_os("XDG_RUNTIME_DIR")
        .context("XDG_RUNTIME_DIR is not set; cannot create private PTY tunnel socket")?;
    let base = PathBuf::from(&raw);
    verify_private_dir(&base)?;

    let dir = base.join("remotewayland");
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

/// Attach local stdin/stdout to a remote PTY over a fresh direct QUIC connection (`--uri`).
pub async fn run_pty_attach(
    server: SocketAddr,
    fingerprint: String,
    token: String,
    mut pty_id: Option<u64>,
    command: Option<String>,
) -> Result<i32> {
    let (conn, _ctrl_send, _ctrl_recv) =
        crate::quic::connect_and_auth(server, fingerprint, token, false).await?;

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

/// Spawn the user's preferred local terminal emulator attached to `remotewayland-client pty-attach --uri <uri>`.
pub fn spawn_local_terminal_pty(uri: &str, command: Option<&str>) -> Result<std::process::Child> {
    let mut attach_args = vec![
        "pty-attach".to_string(),
        "--uri".to_string(),
        uri.to_string(),
    ];
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

fn format_konsole_command(exe: &Path, socket_path: &Path) -> Result<String> {
    let exe_str = exe
        .to_str()
        .context("remotewayland-client executable path is not valid UTF-8")?;
    let sock_str = socket_path
        .to_str()
        .context("PTY socket path is not valid UTF-8")?;
    for token in [exe_str, sock_str] {
        if token.chars().any(|c| matches!(c, '\0' | '\n' | '\r')) {
            bail!("path contains control characters unsupported in Konsole profile");
        }
    }
    let shell_cmd = [exe_str, "pty-attach", "--socket", sock_str]
        .iter()
        .map(|p| quote_konsole_arg(p))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(escape_kconfig_value(&shell_cmd))
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
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
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
        return Err(err).with_context(|| {
            format!("failed to rename {} -> {}", tmp.display(), path.display())
        });
    }
    Ok(())
}

fn prepare_konsole_remote_profile(
    exe: &Path,
    socket_path: &Path,
) -> Result<KonsoleProfileOverlay> {
    prepare_konsole_remote_profile_with_env(exe, socket_path, &XdgEnv::from_process())
}

fn prepare_konsole_remote_profile_with_env(
    exe: &Path,
    socket_path: &Path,
    env: &XdgEnv,
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

    let orig_config_home = env.effective_config_home();
    let orig_data_home = env.effective_data_home();

    let parent_profile = orig_config_home
        .as_deref()
        .and_then(|dir| fs::read_to_string(dir.join("konsolerc")).ok())
        .and_then(|content| parse_konsole_default_profile(&content))
        .unwrap_or_else(|| "FALLBACK/".to_string());

    let cmd = format_konsole_command(exe, socket_path)?;
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
        env.config_dirs.as_ref(),
        "/etc/xdg",
    );
    let data_dirs = join_xdg_dirs(
        orig_data_home.as_deref(),
        env.data_dirs.as_ref(),
        "/usr/local/share:/usr/share",
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

#[cfg(any(feature = "gtk4", feature = "qt", test))]
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

/// Return installed local terminal emulators in priority order (index 0 is the default).
#[cfg(any(feature = "gtk4", feature = "qt", test))]
pub fn available_local_terminals() -> Vec<String> {
    terminal_candidates()
        .into_iter()
        .filter(|term| executable_exists(term))
        .collect()
}

fn spawn_local_terminal_with_args(
    attach_args: &[String],
    new_tab_socket: Option<&Path>,
    initial_mode: InitialTabMode,
    preferred_terminal: Option<&str>,
) -> Result<std::process::Child> {
    let exe = std::env::current_exe().context("failed to locate remotewayland-client binary")?;
    let env_term = std::env::var("TERMINAL").ok();
    let candidates = resolve_terminal_candidates(preferred_terminal, env_term.as_deref())?;
    let mut last_err: Option<(String, std::io::Error)> = None;

    for term in candidates {
        if !executable_exists(&term) {
            continue;
        }
        let mut proc = Command::new(&term);
        let base = terminal_basename(&term);
        match base {
            "konsole" => {
                if let Some(sock) = new_tab_socket {
                    // Scope XDG_CONFIG_HOME / XDG_DATA_HOME to this `--separate` Konsole
                    // instance so New Tab defaults to a remote PTY while `pty-attach` talks
                    // only over `--socket` and never leaks these vars to the remote shell.
                    match prepare_konsole_remote_profile(&exe, sock) {
                        Ok(overlay) => {
                            proc.env("XDG_CONFIG_HOME", &overlay.config_home)
                                .env("XDG_CONFIG_DIRS", &overlay.config_dirs)
                                .env("XDG_DATA_HOME", &overlay.data_home)
                                .env("XDG_DATA_DIRS", &overlay.data_dirs)
                                .arg("--separate")
                                .args(["--profile", "RemoteWayland"]);
                            if initial_mode == InitialTabMode::CustomCommand {
                                proc.arg("-e").arg(&exe).args(attach_args);
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                "failed to prepare Konsole remote profile overlay; falling back to -e"
                            );
                            proc.arg("-e").arg(&exe).args(attach_args);
                        }
                    }
                } else {
                    proc.arg("-e").arg(&exe).args(attach_args);
                }
            }
            "gnome-terminal" => {
                proc.arg("--").arg(&exe).args(attach_args);
            }
            "wezterm" => {
                proc.arg("start").arg("--").arg(&exe).args(attach_args);
            }
            _ => {
                proc.arg("-e").arg(&exe).args(attach_args);
            }
        }
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
        assert!(format_konsole_command(Path::new("/bin/rw\nevil"), Path::new("/tmp/a.sock")).is_err());
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
            parse_konsole_default_profile("[Desktop Entry]\nDefaultProfile=RemoteWayland.profile\n"),
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
        let exe = Path::new("/usr/local/bin/remotewayland-client");

        let overlay = prepare_konsole_remote_profile_with_env(exe, &sock, &env).unwrap();
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
            fs::symlink_metadata(&konsolerc_path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        let profile_path = overlay.data_home.join("konsole").join("RemoteWayland.profile");
        let profile = fs::read_to_string(&profile_path).unwrap();
        assert!(profile.contains("Name=RemoteWayland"));
        assert!(profile.contains("Parent=UserTheme.profile"));
        assert!(profile.contains(&format!(
            "Command=/usr/local/bin/remotewayland-client pty-attach --socket {}",
            sock.display()
        )));
        assert_eq!(
            fs::symlink_metadata(&profile_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::symlink_metadata(&overlay.config_home).unwrap().permissions().mode() & 0o777,
            0o700
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn terminal_candidates_deduplicates_by_basename_and_validates_allowlist() {
        let list = terminal_candidates_with(Some("  /usr/bin/konsole  "));
        assert_eq!(list[0], "/usr/bin/konsole");
        assert_eq!(
            list.iter().filter(|c| terminal_basename(c) == "konsole").count(),
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
        assert!(!executable_exists_in_path("not-exec", Some(path_env.as_os_str())));
        assert!(executable_exists_in_path("is-exec", Some(path_env.as_os_str())));
        assert!(!executable_exists_in_path(
            non_exec.to_str().unwrap(),
            None
        ));
        assert!(executable_exists_in_path(exec.to_str().unwrap(), None));

        let _ = fs::remove_dir_all(&dir);
    }
}
