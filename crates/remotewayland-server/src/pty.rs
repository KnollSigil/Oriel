//! Persistent remote PTY manager over bidirectional QUIC streams.
//!
//! PTYs are owned by a server-wide [`PtyManager`] rather than individual QUIC
//! streams so terminal sessions survive network interruptions:
//! - On unexpected QUIC disconnect, the stream detaches and output produced
//!   while detached is buffered in a bounded 64 KB ring buffer.
//! - On reattach (`BidiStreamHeader::Pty { pty_id: Some(id), .. }`), missed
//!   output is replayed and `SIGWINCH` is delivered so TUI apps repaint cleanly.
//! - On intentional terminal close (`PtyMessage::Close`), the PTY process group
//!   is terminated and cleaned up immediately.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use quinn::{RecvStream, SendStream};
use remotewayland_proto::{
    read_pty_message, write_pty_message, LaunchState, PtyMessage, PTY_LAUNCH_ID_BASE,
};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use crate::compositor::CompositorEvent;

/// Maximum bytes kept in the rolling output ring buffer per PTY (64 KB).
const OUTPUT_BUFFER_CAP: usize = 64 * 1024;

enum PtyOutEvent {
    Data(Vec<u8>),
    Exit(i32),
}

struct PtySharedState {
    /// Active attached stream `(generation, sender)`, or `None` while detached.
    /// At most one terminal may be attached to a PTY at any time.
    attached: Option<(u64, mpsc::Sender<PtyOutEvent>)>,
    /// Rolling 64 KB ring buffer of recent PTY output (covers bytes
    /// `[total_bytes - output_buf.len() .. total_bytes]`) so reconnects can
    /// replay exact missed output from `resume_offset` even if bytes were lost
    /// in an unacknowledged QUIC send buffer before disconnect detection.
    output_buf: VecDeque<u8>,
    /// Total output bytes ever produced by this PTY.
    total_bytes: u64,
    /// Set once the child process has exited.
    exited: Option<i32>,
}

struct ManagedPty {
    pty_id: u64,
    base_name: String,
    pgid: i32,
    master_raw: libc::c_int,
    writer: Arc<Mutex<std::fs::File>>,
    shared: Arc<Mutex<PtySharedState>>,
}

/// Server-wide registry of persistent PTY sessions.
pub struct PtyManager {
    next_id: AtomicU64,
    next_gen: AtomicU64,
    sessions: Mutex<HashMap<u64, Arc<ManagedPty>>>,
    event_tx: broadcast::Sender<CompositorEvent>,
}

impl PtyManager {
    pub fn new(event_tx: broadcast::Sender<CompositorEvent>) -> Arc<Self> {
        Arc::new(Self {
            next_id: AtomicU64::new(1),
            next_gen: AtomicU64::new(1),
            sessions: Mutex::new(HashMap::new()),
            event_tx,
        })
    }

    /// `AppState` events for all currently alive (attached or detached) PTYs.
    pub fn state_events(&self) -> Vec<CompositorEvent> {
        let snapshot: Vec<(u64, String)> = {
            let Ok(map) = self.sessions.lock() else {
                return Vec::new();
            };
            map.values()
                .map(|pty| {
                    let detached = pty
                        .shared
                        .lock()
                        .map(|s| s.attached.is_none())
                        .unwrap_or(false);
                    let display = if detached {
                        format!("{} (detached)", pty.base_name)
                    } else {
                        pty.base_name.clone()
                    };
                    (pty.pty_id, display)
                })
                .collect()
        };

        snapshot
            .into_iter()
            .map(|(pty_id, name)| CompositorEvent::AppState {
                launch_id: PTY_LAUNCH_ID_BASE | pty_id,
                name,
                state: LaunchState::Running,
            })
            .collect()
    }

    /// Terminate a server-managed PTY by `pty_id` (signaling the entire PTY session,
    /// including any foreground job-control process groups like `sleep` in a bash loop).
    pub fn terminate(&self, pty_id: u64, force: bool) {
        let pty = {
            let Ok(map) = self.sessions.lock() else {
                return;
            };
            map.get(&pty_id).cloned()
        };
        if let Some(pty) = pty {
            if pty.pgid > 0 {
                signal_pty_session(pty.master_raw, pty.pgid, force);
            }
        }
    }

    fn emit_state(&self, pty_id: u64, name: String, state: LaunchState) {
        let _ = self.event_tx.send(CompositorEvent::AppState {
            launch_id: PTY_LAUNCH_ID_BASE | pty_id,
            name,
            state,
        });
    }

    fn spawn_pty(
        self: &Arc<Self>,
        cols: u16,
        rows: u16,
        command: Option<String>,
    ) -> Result<Arc<ManagedPty>> {
        let pty_id = self.next_id.fetch_add(1, Ordering::Relaxed);
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

        crate::session::apply_session_env(&mut cmd, None);
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
        let pgid = child.id() as i32;
        drop(slave_fd);

        let reader_fd = master_fd
            .try_clone()
            .context("failed to dup PTY master fd for reader")?;
        let writer_file = std::fs::File::from(master_fd);
        let master_raw = writer_file.as_raw_fd();

        let base_name = match command.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(c) => {
                let short = c.split_whitespace().next().unwrap_or(c);
                format!("Terminal #{pty_id} ({short})")
            }
            None => format!("Terminal #{pty_id}"),
        };

        let shared = Arc::new(Mutex::new(PtySharedState {
            attached: None,
            output_buf: VecDeque::new(),
            total_bytes: 0,
            exited: None,
        }));

        let managed = Arc::new(ManagedPty {
            pty_id,
            base_name: base_name.clone(),
            pgid,
            master_raw,
            writer: Arc::new(Mutex::new(writer_file)),
            shared: Arc::clone(&shared),
        });

        if let Ok(mut map) = self.sessions.lock() {
            map.insert(pty_id, Arc::clone(&managed));
        }

        let mgr = Arc::clone(self);
        let pty_for_reader = Arc::clone(&managed);
        std::thread::Builder::new()
            .name(format!("rw-pty-read-{pty_id}"))
            .spawn(move || {
                let mut file = std::fs::File::from(reader_fd);
                let mut buf = [0u8; 8192];
                loop {
                    match file.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let chunk = &buf[..n];
                            let sender = {
                                let Ok(mut st) = pty_for_reader.shared.lock() else {
                                    break;
                                };
                                let st = &mut *st;
                                push_output_bytes(&mut st.output_buf, &mut st.total_bytes, chunk);
                                st.attached.as_ref().map(|(_, tx)| tx.clone())
                            };
                            if let Some(tx) = sender {
                                let _ = tx.blocking_send(PtyOutEvent::Data(chunk.to_vec()));
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }

                let code = child
                    .wait()
                    .ok()
                    .and_then(|s| s.code())
                    .unwrap_or(1);

                let sender = {
                    let Ok(mut st) = pty_for_reader.shared.lock() else {
                        return;
                    };
                    st.exited = Some(code);
                    st.attached.take().map(|(_, tx)| tx)
                };
                if let Some(tx) = sender {
                    let _ = tx.blocking_send(PtyOutEvent::Exit(code));
                }

                if let Ok(mut map) = mgr.sessions.lock() {
                    map.remove(&pty_id);
                }
                mgr.emit_state(pty_id, base_name, LaunchState::Exited(code));
                debug!(pty_id, code, "persistent PTY exited");
            })
            .context("failed to spawn PTY reader thread")?;

        info!(pty_id, pgid, "spawned persistent PTY session");
        Ok(managed)
    }

    /// Handle a bidirectional QUIC PTY stream (either spawning a new PTY or reattaching to `req_pty_id`).
    pub async fn handle_pty_stream(
        self: Arc<Self>,
        mut send: SendStream,
        mut recv: RecvStream,
        req_pty_id: Option<u64>,
        resume_offset: u64,
        cols: u16,
        rows: u16,
        command: Option<String>,
    ) -> Result<()> {
        let (pty, is_reattach) = if let Some(id) = req_pty_id {
            let found = self
                .sessions
                .lock()
                .ok()
                .and_then(|m| m.get(&id).cloned());
            match found {
                Some(p) => (p, true),
                None => {
                    warn!(pty_id = id, "reattach requested for unknown/exited PTY");
                    let _ = write_pty_message(&mut send, &PtyMessage::Exit(1)).await;
                    let _ = send.finish();
                    return Ok(());
                }
            }
        } else {
            (self.spawn_pty(cols, rows, command)?, false)
        };

        let pty_id = pty.pty_id;
        let gen = self.next_gen.fetch_add(1, Ordering::Relaxed);
        let (out_tx, mut out_rx) = mpsc::channel::<PtyOutEvent>(128);

        enum AttachOutcome {
            Attached {
                stream_offset: u64,
                replay: Vec<u8>,
            },
            Busy,
            Exited(i32),
        }

        let outcome = {
            let mut st = pty
                .shared
                .lock()
                .map_err(|_| anyhow::anyhow!("PTY state lock poisoned"))?;
            if let Some(code) = st.exited {
                AttachOutcome::Exited(code)
            } else if st.attached.is_some() {
                AttachOutcome::Busy
            } else {
                st.attached = Some((gen, out_tx));
                let stream_offset = st.total_bytes;
                let replay = if is_reattach {
                    slice_missed_output(&st.output_buf, stream_offset, resume_offset)
                } else {
                    Vec::new()
                };
                AttachOutcome::Attached {
                    stream_offset,
                    replay,
                }
            }
        };

        let (stream_offset, replay_bytes) = match outcome {
            AttachOutcome::Exited(code) => {
                let _ = write_pty_message(&mut send, &PtyMessage::Exit(code)).await;
                let _ = send.finish();
                return Ok(());
            }
            AttachOutcome::Busy => {
                info!(pty_id, "rejecting duplicate attach; PTY already has an active terminal");
                let _ = write_pty_message(&mut send, &PtyMessage::Busy).await;
                let _ = send.finish();
                return Ok(());
            }
            AttachOutcome::Attached {
                stream_offset,
                replay,
            } => (stream_offset, replay),
        };

        self.emit_state(pty_id, pty.base_name.clone(), LaunchState::Running);

        write_pty_message(
            &mut send,
            &PtyMessage::Attached {
                pty_id,
                stream_offset,
            },
        )
        .await?;
        if !replay_bytes.is_empty() {
            write_pty_message(&mut send, &PtyMessage::Replay(replay_bytes)).await?;
        }

        if is_reattach {
            set_pty_winsize(pty.master_raw, cols.max(1), rows.max(1));
            if pty.pgid > 0 {
                unsafe {
                    libc::kill(-pty.pgid, libc::SIGWINCH);
                }
            }
            info!(pty_id, resume_offset, "client reattached to persistent PTY");
        }

        let mut intentional_close = false;
        loop {
            tokio::select! {
                out = out_rx.recv() => {
                    match out {
                        Some(PtyOutEvent::Data(chunk)) => {
                            if write_pty_message(&mut send, &PtyMessage::Data(chunk)).await.is_err() {
                                break;
                            }
                        }
                        Some(PtyOutEvent::Exit(code)) => {
                            let _ = write_pty_message(&mut send, &PtyMessage::Exit(code)).await;
                            let _ = send.finish();
                            break;
                        }
                        None => break,
                    }
                }
                incoming = read_pty_message(&mut recv) => {
                    match incoming {
                        Ok(Some(PtyMessage::Data(bytes))) => {
                            let write_res = pty
                                .writer
                                .lock()
                                .map_err(|_| std::io::Error::other("writer lock poisoned"))
                                .and_then(|mut f| f.write_all(&bytes));
                            if write_res.is_err() {
                                break;
                            }
                        }
                        Ok(Some(PtyMessage::Resize { cols, rows })) => {
                            set_pty_winsize(pty.master_raw, cols.max(1), rows.max(1));
                        }
                        Ok(Some(PtyMessage::Close)) => {
                            intentional_close = true;
                            info!(pty_id, "client requested intentional PTY close");
                            self.terminate(pty_id, false);
                            break;
                        }
                        Ok(Some(PtyMessage::Exit(_))) | Ok(None) | Err(_) => break,
                        Ok(Some(
                            PtyMessage::Attached { .. }
                            | PtyMessage::Replay(_)
                            | PtyMessage::Busy,
                        )) => {}
                    }
                }
            }
        }

        if !intentional_close {
            let became_detached = {
                if let Ok(mut st) = pty.shared.lock() {
                    if st.exited.is_none()
                        && st.attached.as_ref().is_some_and(|(g, _)| *g == gen)
                    {
                        st.attached = None;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if became_detached {
                info!(pty_id, "client detached from persistent PTY (keeping alive)");
                self.emit_state(
                    pty_id,
                    format!("{} (detached)", pty.base_name),
                    LaunchState::Running,
                );
            }
        }

        Ok(())
    }
}

fn signal_pty_session(master_raw: libc::c_int, sid: i32, force: bool) {
    let fg_pgid = unsafe { libc::tcgetpgrp(master_raw) };
    let session_pids = pids_in_session(sid);

    unsafe {
        if force {
            if fg_pgid > 0 {
                libc::kill(-fg_pgid, libc::SIGKILL);
            }
            libc::kill(-sid, libc::SIGKILL);
            for pid in session_pids {
                libc::kill(pid, libc::SIGKILL);
            }
        } else {
            if fg_pgid > 0 {
                libc::kill(-fg_pgid, libc::SIGHUP);
                libc::kill(-fg_pgid, libc::SIGCONT);
                libc::kill(-fg_pgid, libc::SIGTERM);
            }
            libc::kill(-sid, libc::SIGHUP);
            libc::kill(-sid, libc::SIGCONT);
            for pid in session_pids {
                libc::kill(pid, libc::SIGHUP);
                libc::kill(pid, libc::SIGCONT);
                if pid != sid {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
        }
    }
}

/// Enumerate all Linux PIDs belonging to session ID `target_sid` via `/proc/<pid>/stat`.
fn pids_in_session(target_sid: i32) -> Vec<i32> {
    let mut pids = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return pids;
    };
    for entry in entries.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let Ok(pid) = name.parse::<i32>() else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // Format: `pid (comm) state ppid pgrp session ...`
        // Find the last ')' because `comm` can contain spaces or parentheses.
        let Some(rparen) = stat.rfind(')') else {
            continue;
        };
        let rest = &stat[rparen + 1..];
        let mut fields = rest.split_whitespace();
        // fields[0]=state, [1]=ppid, [2]=pgrp, [3]=session
        if let Some(sid_str) = fields.nth(3) {
            if sid_str.parse::<i32>().ok() == Some(target_sid) {
                pids.push(pid);
            }
        }
    }
    pids
}

fn push_output_bytes(buf: &mut VecDeque<u8>, total_bytes: &mut u64, chunk: &[u8]) {
    *total_bytes = total_bytes.saturating_add(chunk.len() as u64);
    if chunk.len() >= OUTPUT_BUFFER_CAP {
        buf.clear();
        buf.extend(&chunk[chunk.len() - OUTPUT_BUFFER_CAP..]);
        return;
    }
    let overflow = (buf.len() + chunk.len()).saturating_sub(OUTPUT_BUFFER_CAP);
    if overflow > 0 {
        buf.drain(..overflow);
    }
    buf.extend(chunk);
}

/// Return the slice of `buf` produced after `resume_offset` without duplicating
/// bytes the client already received, stripping historical terminal capability
/// queries (DA1/DA2/XTVERSION/OSC color/DSR/Kitty `?u`/DECRQM) so a reattaching
/// terminal emulator doesn't inject unsolicited replies onto stdin.
fn slice_missed_output(buf: &VecDeque<u8>, total_bytes: u64, resume_offset: u64) -> Vec<u8> {
    if resume_offset >= total_bytes || buf.is_empty() {
        return Vec::new();
    }
    let buf_start = total_bytes.saturating_sub(buf.len() as u64);
    let skip = resume_offset
        .saturating_sub(buf_start)
        .min(buf.len() as u64) as usize;
    let raw: Vec<u8> = buf.iter().skip(skip).copied().collect();
    strip_terminal_queries(&raw)
}

/// Strip ANSI/VT terminal query sequences from historical replay output so
/// reattaching terminals (Konsole, Kitty, Alacritty, etc.) don't reply to old
/// shell-startup probes and echo garbage onto stdin.
fn strip_terminal_queries(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] != 0x1b || i + 1 >= data.len() {
            out.push(data[i]);
            i += 1;
            continue;
        }

        match data[i + 1] {
            // CSI: ESC [ <params/intermediates> <final 0x40..=0x7e>
            b'[' => {
                let mut j = i + 2;
                while j < data.len() && !(0x40..=0x7e).contains(&data[j]) {
                    j += 1;
                }
                if j < data.len() {
                    let params = &data[i + 2..j];
                    let final_byte = data[j];
                    if is_csi_query(params, final_byte) {
                        i = j + 1;
                        continue;
                    }
                    out.extend_from_slice(&data[i..=j]);
                    i = j + 1;
                    continue;
                }
            }
            // OSC: ESC ] ... (BEL | ESC \)
            b']' => {
                let mut j = i + 2;
                let mut end = None;
                let mut body_end = i + 2;
                while j < data.len() {
                    if data[j] == 0x07 {
                        body_end = j;
                        end = Some(j + 1);
                        break;
                    }
                    if data[j] == 0x1b && j + 1 < data.len() && data[j + 1] == b'\\' {
                        body_end = j;
                        end = Some(j + 2);
                        break;
                    }
                    j += 1;
                }
                if let Some(next_i) = end {
                    let body = &data[i + 2..body_end];
                    // OSC color/palette queries end with `;?` (e.g. `10;?`, `11;?`, `4;0;?`).
                    if body.ends_with(b";?") {
                        i = next_i;
                        continue;
                    }
                    out.extend_from_slice(&data[i..next_i]);
                    i = next_i;
                    continue;
                }
            }
            // DCS: ESC P ... ESC \
            b'P' => {
                let mut j = i + 2;
                let mut end = None;
                while j + 1 < data.len() {
                    if data[j] == 0x1b && data[j + 1] == b'\\' {
                        end = Some(j + 2);
                        break;
                    }
                    j += 1;
                }
                if let Some(next_i) = end {
                    let body = &data[i + 2..next_i - 2];
                    // DECRQSS (`$q...`) and XTGETTCAP (`+q...`) queries.
                    if body.starts_with(b"$q") || body.starts_with(b"+q") {
                        i = next_i;
                        continue;
                    }
                    out.extend_from_slice(&data[i..next_i]);
                    i = next_i;
                    continue;
                }
            }
            _ => {}
        }

        out.push(data[i]);
        i += 1;
    }
    out
}

fn is_csi_query(params: &[u8], final_byte: u8) -> bool {
    match final_byte {
        // DA1 (`ESC [ c`, `ESC [ 0 c`), DA2 (`ESC [ > c`, `ESC [ > 0 c`), DA3 (`ESC [ = c`, `ESC [ = 0 c`)
        b'c' => matches!(params, b"" | b"0" | b">" | b">0" | b"=" | b"=0"),
        // XTVERSION (`ESC [ > q`, `ESC [ > 0 q`)
        b'q' => matches!(params, b">" | b">0"),
        // Kitty keyboard protocol query (`ESC [ ? u`)
        b'u' => params == b"?",
        // DSR / CPR (`ESC [ 5 n`, `ESC [ 6 n`, `ESC [ ? 6 n`, etc.)
        b'n' => matches!(params, b"5" | b"6" | b"?6" | b"?15" | b"?25" | b"?26" | b"?53" | b"?55" | b"?56" | b"?62" | b"?63" | b"?75" | b"?85"),
        // DECRQM (`ESC [ ? <n> $ p` or `ESC [ <n> $ p`)
        b'p' => params.ends_with(b"$"),
        // XTWINOPS report queries (`14t`, `16t`, `18t`, `19t`)
        b't' => matches!(params, b"14" | b"16" | b"18" | b"19"),
        _ => false,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_ring_buffer_caps_at_64kb_and_replays_from_resume_offset() {
        let mut buf = VecDeque::new();
        let mut total = 0u64;
        let block1 = vec![b'a'; 40 * 1024];
        push_output_bytes(&mut buf, &mut total, &block1);
        assert_eq!(buf.len(), 40 * 1024);
        assert_eq!(total, 40 * 1024);

        let block2 = vec![b'b'; 40 * 1024];
        push_output_bytes(&mut buf, &mut total, &block2);
        assert_eq!(buf.len(), OUTPUT_BUFFER_CAP);
        assert_eq!(total, 80 * 1024);
        assert_eq!(buf.front(), Some(&b'a'));
        assert_eq!(buf.back(), Some(&b'b'));

        // Client already received 79 KB -> only the last 1 KB of 'b' is replayed.
        let missed = slice_missed_output(&buf, total, 79 * 1024);
        assert_eq!(missed.len(), 1024);
        assert!(missed.iter().all(|&b| b == b'b'));

        // Client already up-to-date -> zero replay.
        assert!(slice_missed_output(&buf, total, total).is_empty());

        // Fresh attach (offset 0) -> replays full 64 KB buffer.
        assert_eq!(slice_missed_output(&buf, total, 0).len(), OUTPUT_BUFFER_CAP);
    }

    #[test]
    fn strip_terminal_queries_removes_probes_and_preserves_colors() {
        let input = concat!(
            "\x1b[?u",
            "\x1b[>0q",
            "\x1b]11;?\x1b\\",
            "\x1b]10;?\x07",
            "\x1b[c",
            "\x1b[6n",
            "\x1b[?2026$p",
            "\x1bP+q544e\x1b\\",
            "\x1b[32mWelcome to fish\x1b[0m\r\n"
        );
        let cleaned = strip_terminal_queries(input.as_bytes());
        assert_eq!(
            String::from_utf8(cleaned).unwrap(),
            "\x1b[32mWelcome to fish\x1b[0m\r\n"
        );
    }

    #[test]
    fn terminate_kills_bash_while_loop_in_pty() {
        let (tx, _rx) = broadcast::channel(16);
        let mgr = PtyManager::new(tx);
        let pty = mgr.spawn_pty(80, 24, None).expect("spawn_pty failed");
        let pty_id = pty.pty_id;

        // Start an interactive shell loop (with job control putting `sleep 1` in its own pgid).
        {
            let mut w = pty.writer.lock().unwrap();
            w.write_all(b"while true; do date; sleep 1; done\n")
                .expect("write loop command");
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
        mgr.terminate(pty_id, false);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if !mgr.sessions.lock().unwrap().contains_key(&pty_id) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("interactive PTY running bash while loop did not terminate within 2s");
    }
}
