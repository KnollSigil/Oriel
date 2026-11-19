//! Local terminal <-> remote PTY QUIC / Unix-tunnel stream bridge with silent auto-reattach.

use std::fs::{self, DirBuilder, Permissions};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
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

/// Compute the local `0600` Unix tunnel socket path for a connected host session
/// under `$XDG_RUNTIME_DIR/remotewayland/pty-<sanitized>.sock` (no `/tmp` fallback).
pub fn pty_tunnel_socket_path(session_key: &str) -> Result<PathBuf> {
    let raw = std::env::var_os("XDG_RUNTIME_DIR")
        .context("XDG_RUNTIME_DIR is not set; cannot create private PTY tunnel socket")?;
    let base = PathBuf::from(&raw);
    let meta = fs::symlink_metadata(&base)
        .with_context(|| format!("XDG_RUNTIME_DIR ({}) cannot be inspected", base.display()))?;
    let euid = unsafe { libc::geteuid() };
    if !meta.file_type().is_dir() || meta.uid() != euid || (meta.permissions().mode() & 0o077) != 0
    {
        bail!(
            "XDG_RUNTIME_DIR ({}) must be a 0700 directory owned by current uid",
            base.display()
        );
    }

    let dir = base.join("remotewayland");
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    fs::set_permissions(&dir, Permissions::from_mode(0o700))?;

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

/// Spawn the user's preferred local terminal emulator attached to `remotewayland-client pty-attach --socket <path>`.
pub fn spawn_local_terminal_pty_socket(
    socket_path: &Path,
    pty_id: Option<u64>,
    command: Option<&str>,
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
    spawn_local_terminal_with_args(&attach_args)
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
    spawn_local_terminal_with_args(&attach_args)
}

fn spawn_local_terminal_with_args(attach_args: &[String]) -> Result<std::process::Child> {
    let exe = std::env::current_exe().context("failed to locate remotewayland-client binary")?;

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(term) = std::env::var("TERMINAL") {
        if !term.trim().is_empty() {
            candidates.push(term);
        }
    }
    for term in [
        "konsole",
        "alacritty",
        "kitty",
        "foot",
        "wezterm",
        "gnome-terminal",
        "xfce4-terminal",
        "xterm",
    ] {
        candidates.push(term.to_string());
    }

    for term in candidates {
        let mut proc = Command::new(&term);
        let base = Path::new(&term)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&term);
        match base {
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
        if let Ok(child) = proc.spawn() {
            return Ok(child);
        }
    }

    anyhow::bail!(
        "no local terminal emulator found ($TERMINAL, konsole, alacritty, kitty, foot, xterm)"
    )
}
