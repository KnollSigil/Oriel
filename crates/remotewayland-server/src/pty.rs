//! Remote PTY handler over bidirectional QUIC streams (no Wayland/PipeWire required).

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use quinn::{RecvStream, SendStream};
use remotewayland_proto::{read_pty_message, write_pty_message, PtyMessage};
use tracing::{debug, warn};

/// Spawn a shell or command inside a fresh PTY and bridge I/O + resize over QUIC.
pub async fn handle_pty_stream(
    mut send: SendStream,
    mut recv: RecvStream,
    cols: u16,
    rows: u16,
    command: Option<String>,
) -> Result<()> {
    let (master_fd, slave_fd) = open_pty(cols.max(20), rows.max(5))?;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut cmd = if let Some(ref c) = command {
        let mut p = Command::new(&shell);
        p.arg("-c").arg(c);
        p
    } else {
        let mut p = Command::new(&shell);
        p.arg("-i");
        p
    };

    let slave_raw = slave_fd.as_raw_fd();
    let master_raw = master_fd.as_raw_fd();

    cmd.env("TERM", "xterm-256color")
        .env("COLORTERM", "truecolor")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // SAFETY: pre_exec runs in the forked child before exec; only async-signal-safe syscalls are used.
    unsafe {
        cmd.pre_exec(move || {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(slave_raw, libc::TIOCSCTTY as _, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::dup2(slave_raw, libc::STDIN_FILENO) == -1
                || libc::dup2(slave_raw, libc::STDOUT_FILENO) == -1
                || libc::dup2(slave_raw, libc::STDERR_FILENO) == -1
            {
                return Err(std::io::Error::last_os_error());
            }
            if slave_raw > 2 {
                libc::close(slave_raw);
            }
            if master_raw > 2 {
                libc::close(master_raw);
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn().context("failed to spawn PTY child process")?;
    drop(slave_fd);

    let reader_fd = master_fd
        .try_clone()
        .context("failed to dup PTY master fd for reader")?;
    let writer_fd = master_fd
        .try_clone()
        .context("failed to dup PTY master fd for writer")?;

    let (tx_out, mut rx_out) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let read_task = tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::from(reader_fd);
        let mut buf = [0u8; 8192];
        loop {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx_out.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    });

    let wait_task = tokio::task::spawn_blocking(move || -> i32 {
        match child.wait() {
            Ok(status) => status.code().unwrap_or(1),
            Err(_) => 1,
        }
    });

    let send_task = tokio::spawn(async move {
        while let Some(chunk) = rx_out.recv().await {
            if write_pty_message(&mut send, &PtyMessage::Data(chunk))
                .await
                .is_err()
            {
                break;
            }
        }
        let code = wait_task.await.unwrap_or(1);
        let _ = write_pty_message(&mut send, &PtyMessage::Exit(code)).await;
        let _ = send.finish();
    });

    let mut writer_file = std::fs::File::from(writer_fd);
    while let Ok(Some(msg)) = read_pty_message(&mut recv).await {
        match msg {
            PtyMessage::Data(bytes) => {
                if writer_file.write_all(&bytes).is_err() {
                    break;
                }
            }
            PtyMessage::Resize { cols, rows } => {
                set_pty_winsize(master_fd.as_raw_fd(), cols.max(1), rows.max(1));
            }
            PtyMessage::Exit(_) => break,
        }
    }

    drop(writer_file);
    drop(master_fd);
    let _ = read_task.await;
    let _ = send_task.await;
    debug!("PTY session ended");
    Ok(())
}

fn open_pty(cols: u16, rows: u16) -> Result<(OwnedFd, OwnedFd)> {
    let mut master: libc::c_int = -1;
    let mut slave: libc::c_int = -1;
    let mut ws = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let rc = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut ws,
        )
    };
    if rc != 0 {
        warn!("openpty failed: {}", std::io::Error::last_os_error());
        anyhow::bail!("openpty failed: {}", std::io::Error::last_os_error());
    }
    let master_fd = unsafe { OwnedFd::from_raw_fd(master) };
    let slave_fd = unsafe { OwnedFd::from_raw_fd(slave) };
    Ok((master_fd, slave_fd))
}

fn set_pty_winsize(fd: libc::c_int, cols: u16, rows: u16) {
    let ws = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    unsafe {
        libc::ioctl(fd, libc::TIOCSWINSZ as _, &ws);
    }
}
