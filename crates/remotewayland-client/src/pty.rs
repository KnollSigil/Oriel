//! Local terminal <-> remote PTY QUIC stream bridge.

use std::io::{Read, Write};
use std::net::SocketAddr;
use std::process::Command;

use anyhow::{Context, Result};
use remotewayland_proto::{
    read_pty_message, write_bidi_stream_header, write_pty_message, BidiStreamHeader, PtyMessage,
};
use tokio::signal::unix::{signal, SignalKind};

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

/// Attach local stdin/stdout to a remote PTY over QUIC (`StreamHeader::Pty`).
pub async fn run_pty_attach(
    server: SocketAddr,
    fingerprint: String,
    token: String,
    command: Option<String>,
) -> Result<i32> {
    let (conn, _ctrl_send, _ctrl_recv) =
        crate::quic::connect_and_auth(server, fingerprint, token, false).await?;

    let (cols, rows) = current_winsize();
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .context("failed to open QUIC PTY stream")?;
    write_bidi_stream_header(
        &mut send,
        &BidiStreamHeader::Pty {
            cols,
            rows,
            command,
        },
    )
    .await?;

    let _raw_guard = RawModeGuard::enter();

    let (in_tx, mut in_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
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

    let mut sigwinch = signal(SignalKind::window_change())?;

    let writer_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                chunk = in_rx.recv() => {
                    match chunk {
                        Some(bytes) => {
                            if write_pty_message(&mut send, &PtyMessage::Data(bytes)).await.is_err() {
                                break;
                            }
                        }
                        None => {
                            let _ = send.finish();
                            break;
                        }
                    }
                }
                _ = sigwinch.recv() => {
                    let (c, r) = current_winsize();
                    if write_pty_message(&mut send, &PtyMessage::Resize { cols: c, rows: r }).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut exit_code = 0;
    let mut stdout = std::io::stdout();
    while let Ok(Some(msg)) = read_pty_message(&mut recv).await {
        match msg {
            PtyMessage::Data(bytes) => {
                let _ = stdout.write_all(&bytes);
                let _ = stdout.flush();
            }
            PtyMessage::Exit(code) => {
                exit_code = code;
                break;
            }
            PtyMessage::Resize { .. } => {}
        }
    }

    writer_task.abort();
    Ok(exit_code)
}

/// Spawn the user's preferred local terminal emulator attached to `remotewayland-client pty-attach`.
pub fn spawn_local_terminal_pty(uri: &str, command: Option<&str>) -> Result<()> {
    let exe = std::env::current_exe().context("failed to locate remotewayland-client binary")?;
    let mut attach_args = vec![
        "pty-attach".to_string(),
        "--uri".to_string(),
        uri.to_string(),
    ];
    if let Some(cmd) = command {
        attach_args.push("--command".to_string());
        attach_args.push(cmd.to_string());
    }

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
        let base = std::path::Path::new(&term)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&term);
        match base {
            "gnome-terminal" => {
                proc.arg("--").arg(&exe).args(&attach_args);
            }
            "wezterm" => {
                proc.arg("start").arg("--").arg(&exe).args(&attach_args);
            }
            _ => {
                proc.arg("-e").arg(&exe).args(&attach_args);
            }
        }
        if proc.spawn().is_ok() {
            return Ok(());
        }
    }

    anyhow::bail!("no local terminal emulator found ($TERMINAL, konsole, alacritty, kitty, foot, xterm)")
}
