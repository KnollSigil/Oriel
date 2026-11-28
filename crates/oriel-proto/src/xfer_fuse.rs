//! FUSE presentation of file offers received from the peer (see [`crate::xfer`]).
//!
//! One read-only mount lives for the whole process. Each offer appears as a numbered
//! directory holding the offer's roots; directory listings and file contents are fetched on
//! demand over QUIC streams to the peer that owns the files. Replaced offers stay readable
//! for [`OFFER_GRACE`] (longer while files are open), then disappear.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use fuser::{
    AccessFlags, Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation,
    INodeNo, KernelConfig, LockOwner, MountOption, OpenFlags, ReplyAttr, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyXattr, Request,
};
use tracing::{debug, warn};

use crate::transfers::{self, TransferDirection, TransferGuard};
use crate::xfer::{read_frame, FileEntry, FileKind, FileListReply, FileReadReply, OfferSlot};
use crate::{write_bidi_stream_header, BidiStreamHeader};

/// How long a replaced offer stays readable when none of its files are open.
pub const OFFER_GRACE: Duration = Duration::from_secs(60);
/// Deadline for one FUSE read. File managers cannot cancel a read blocked inside FUSE, so
/// a stalled peer must turn into an error eventually.
const READ_DEADLINE: Duration = Duration::from_secs(30);
/// Deadline for fetching a directory listing.
const LIST_DEADLINE: Duration = Duration::from_secs(30);
/// Attribute/entry cache lifetime handed to the kernel.
const TTL: Duration = Duration::from_secs(5);
/// Bytes kept behind the stream position so slightly out-of-order reads (kernel
/// readahead) do not restart the stream.
const READ_WINDOW: u64 = 8 * 1024 * 1024;
/// Forward gap that is skipped by reading through the stream instead of reopening it.
const MAX_SKIP: u64 = 16 * 1024 * 1024;
/// QUIC application error code used when a reader abandons a file stream.
const STOP_CODE: u32 = 0;

const ROOT: u64 = 1;

/// A mounted transfer filesystem. Dropping it unmounts.
pub struct XferMount {
    shared: Arc<Shared>,
    mountpoint: PathBuf,
    session: Mutex<Option<fuser::BackgroundSession>>,
}

struct Shared {
    rt: tokio::runtime::Handle,
    uid: u32,
    gid: u32,
    state: Mutex<State>,
    /// Set when unmounting: reads held by a paused transfer are let go.
    closing: tokio::sync::watch::Sender<bool>,
}

struct State {
    next_ino: u64,
    next_fh: u64,
    next_offer_seq: u64,
    nodes: HashMap<u64, Node>,
    /// Offer directory inode -> offer.
    offers: HashMap<u64, OfferInfo>,
    handles: HashMap<u64, Arc<tokio::sync::Mutex<ReadState>>>,
}

struct Node {
    parent: u64,
    /// Offer directory inode this node belongs to (0 for the mount root).
    offer_dir: u64,
    /// Root name and components below it (empty for the root and offer directories).
    path: Vec<String>,
    kind: FileKind,
    size: u64,
    mtime_s: i64,
    mode: u32,
    /// Children by name, once known.
    children: Option<BTreeMap<String, u64>>,
}

struct OfferInfo {
    conn: quinn::Connection,
    offer_id: u64,
    slot: OfferSlot,
    retired_at: Option<Instant>,
    open_files: usize,
}

struct ReadState {
    conn: quinn::Connection,
    offer_id: u64,
    path: Vec<String>,
    size: u64,
    stream: Option<quinn::RecvStream>,
    /// Bytes `[buf_start, buf_start + buf.len())` of the file; the stream continues after.
    buf: Vec<u8>,
    buf_start: u64,
    /// This handle's part in the transfer shown to the user.
    transfer: TransferGuard,
}

impl Drop for XferMount {
    fn drop(&mut self) {
        // Before the session unmounts: a held read would keep the unmount waiting.
        self.shared.closing.send_replace(true);
    }
}

impl XferMount {
    /// Mount an empty transfer filesystem at `mountpoint` (created if missing). A stale
    /// mount left behind by a crashed process is unmounted first.
    pub fn mount(mountpoint: &Path, rt: tokio::runtime::Handle) -> Result<Self> {
        unmount_stale(mountpoint);
        std::fs::create_dir_all(mountpoint)
            .with_context(|| format!("failed to create {}", mountpoint.display()))?;
        let mut nodes = HashMap::new();
        nodes.insert(
            ROOT,
            Node {
                parent: ROOT,
                offer_dir: 0,
                path: Vec::new(),
                kind: FileKind::Dir,
                size: 0,
                mtime_s: now_s(),
                mode: 0o500,
                children: Some(BTreeMap::new()),
            },
        );
        let shared = Arc::new(Shared {
            rt,
            // SAFETY: getuid/getgid have no preconditions.
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            state: Mutex::new(State {
                next_ino: ROOT + 1,
                next_fh: 1,
                next_offer_seq: 1,
                nodes,
                offers: HashMap::new(),
                handles: HashMap::new(),
            }),
            closing: tokio::sync::watch::channel(false).0,
        });
        let mut config = Config::default();
        config.mount_options = vec![
            MountOption::FSName("oriel".into()),
            MountOption::Subtype("oriel-xfer".into()),
            MountOption::RO,
            MountOption::NoSuid,
            MountOption::NoDev,
            MountOption::NoExec,
            MountOption::DefaultPermissions,
        ];
        config.n_threads = Some(1);
        let session = fuser::spawn_mount(
            XferFs {
                shared: Arc::clone(&shared),
            },
            mountpoint,
            &config,
        )
        .with_context(|| format!("failed to mount FUSE at {}", mountpoint.display()))?;

        let weak = Arc::downgrade(&shared);
        shared.rt.spawn(sweep_loop(weak));
        Ok(Self {
            shared,
            mountpoint: mountpoint.to_path_buf(),
            session: Mutex::new(Some(session)),
        })
    }

    /// Unmount now (also done on drop). Later offers resolve to paths that don't exist.
    pub fn unmount(&self) {
        self.shared.closing.send_replace(true);
        let session = self.session.lock().unwrap().take();
        if let Some(session) = session {
            if let Err(err) = session.umount_and_join() {
                // Busy (a process still has a file or its working directory in it): detach
                // it lazily so it does not outlive us as a dead mount.
                warn!(error = %err, mountpoint = %self.mountpoint.display(), "failed to unmount transfer filesystem; detaching it lazily");
                unmount_stale(&self.mountpoint);
            }
        }
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Present a new offer from `conn` and return the local paths of its roots. The other
    /// offers in `slot` are retired: they stay readable for [`OFFER_GRACE`] and while open.
    pub fn add_offer(
        &self,
        conn: quinn::Connection,
        slot: OfferSlot,
        offer_id: u64,
        roots: Vec<FileEntry>,
    ) -> Vec<PathBuf> {
        let mut st = self.shared.state.lock().unwrap();
        let now = Instant::now();
        for offer in st.offers.values_mut().filter(|o| o.slot == slot) {
            offer.retired_at.get_or_insert(now);
        }
        let seq = st.next_offer_seq;
        st.next_offer_seq += 1;
        let dir_name = seq.to_string();
        let dir_ino = st.alloc_ino();
        let mut children = BTreeMap::new();
        let mut paths = Vec::new();
        let dir_path = self.mountpoint.join(&dir_name);
        for root in roots {
            if !crate::xfer::is_valid_name(&root.name) || children.contains_key(&root.name) {
                continue;
            }
            let ino = st.alloc_ino();
            paths.push(dir_path.join(&root.name));
            children.insert(root.name.clone(), ino);
            st.nodes.insert(
                ino,
                Node::from_entry(dir_ino, dir_ino, vec![root.name.clone()], &root),
            );
        }
        st.nodes.insert(
            dir_ino,
            Node {
                parent: ROOT,
                offer_dir: dir_ino,
                path: Vec::new(),
                kind: FileKind::Dir,
                size: 0,
                mtime_s: now_s(),
                mode: 0o500,
                children: Some(children),
            },
        );
        st.nodes
            .get_mut(&ROOT)
            .and_then(|n| n.children.as_mut())
            .expect("root node")
            .insert(dir_name, dir_ino);
        st.offers.insert(
            dir_ino,
            OfferInfo {
                conn,
                offer_id,
                slot,
                retired_at: None,
                open_files: 0,
            },
        );
        sweep(&mut st);
        paths
    }
}

impl Node {
    fn from_entry(parent: u64, offer_dir: u64, path: Vec<String>, e: &FileEntry) -> Self {
        Node {
            parent,
            offer_dir,
            path,
            kind: e.kind,
            size: e.size,
            mtime_s: e.mtime_s,
            mode: e.mode & 0o777,
            children: if e.kind == FileKind::Dir {
                None
            } else {
                Some(BTreeMap::new())
            },
        }
    }
}

impl State {
    fn alloc_ino(&mut self) -> u64 {
        let ino = self.next_ino;
        self.next_ino += 1;
        ino
    }

    fn attr(&self, ino: u64, uid: u32, gid: u32) -> Option<FileAttr> {
        let n = self.nodes.get(&ino)?;
        let mtime = UNIX_EPOCH + Duration::from_secs(n.mtime_s.max(0) as u64);
        Some(FileAttr {
            ino: INodeNo(ino),
            size: n.size,
            blocks: n.size.div_ceil(512),
            atime: mtime,
            mtime,
            ctime: mtime,
            crtime: mtime,
            kind: match n.kind {
                FileKind::File => FileType::RegularFile,
                FileKind::Dir => FileType::Directory,
            },
            perm: n.mode as u16,
            nlink: if n.kind == FileKind::Dir { 2 } else { 1 },
            uid,
            gid,
            rdev: 0,
            blksize: 128 * 1024,
            flags: 0,
        })
    }

    /// Remove `ino` and everything below it.
    fn remove_subtree(&mut self, ino: u64) {
        let mut stack = vec![ino];
        while let Some(i) = stack.pop() {
            if let Some(node) = self.nodes.remove(&i) {
                if let Some(children) = node.children {
                    stack.extend(children.values().copied());
                }
            }
        }
    }
}

/// Drop retired offers whose grace period is over and that have no open files.
fn sweep(st: &mut State) {
    let expired: Vec<u64> = st
        .offers
        .iter()
        .filter(|(_, o)| {
            o.open_files == 0 && o.retired_at.is_some_and(|t| t.elapsed() >= OFFER_GRACE)
        })
        .map(|(&ino, _)| ino)
        .collect();
    for dir_ino in expired {
        st.offers.remove(&dir_ino);
        if let Some(root) = st.nodes.get_mut(&ROOT).and_then(|n| n.children.as_mut()) {
            root.retain(|_, &mut ino| ino != dir_ino);
        }
        st.remove_subtree(dir_ino);
    }
}

async fn sweep_loop(shared: Weak<Shared>) {
    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;
        let Some(shared) = shared.upgrade() else {
            return;
        };
        sweep(&mut shared.state.lock().unwrap());
    }
}

fn now_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Remove mount directories in `dir` whose names start with `prefix` and that belong to
/// processes that are gone: stale mounts (`ENOTCONN`) are unmounted, and plain leftover
/// directories removed. Live mounts of other processes are left alone.
/// How old an unmounted directory must be before [`cleanup_stale_mounts`] removes it.
const UNMOUNTED_DIR_MIN_AGE: Duration = Duration::from_secs(60);

pub fn cleanup_stale_mounts(dir: &Path, prefix: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with(prefix) {
            continue;
        }
        let path = entry.path();
        match std::fs::metadata(&path) {
            Err(err) if err.raw_os_error() == Some(libc::ENOTCONN) => unmount_stale(&path),
            // An empty directory may belong to a process that is mounting on it right now
            // (it creates the directory first); leave young ones alone.
            Ok(meta)
                if meta.is_dir()
                    && !is_mountpoint(&path, &meta)
                    && meta
                        .modified()
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age >= UNMOUNTED_DIR_MIN_AGE) => {}
            _ => continue,
        }
        // Only succeeds for an empty directory that is no longer a mount.
        let _ = std::fs::remove_dir(&path);
    }
}

fn is_mountpoint(path: &Path, meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    path.parent()
        .and_then(|p| std::fs::metadata(p).ok())
        .is_some_and(|parent| parent.dev() != meta.dev())
}

/// Lazily unmount whatever is mounted at `mountpoint`, such as a mount left by a previous
/// process (a dead FUSE server makes every access fail with `ENOTCONN`). Does nothing when
/// nothing is mounted there.
pub fn unmount_stale(mountpoint: &Path) {
    let stale = match std::fs::metadata(mountpoint) {
        Err(err) => err.raw_os_error() == Some(libc::ENOTCONN),
        Ok(meta) => is_mountpoint(mountpoint, &meta),
    };
    if stale {
        detach(mountpoint);
    }
}

/// Lazily detach whatever is mounted at `mountpoint` (`fusermount3 -u -z`), without
/// touching it first: a stat could hang on a mount whose server is stuck.
pub fn detach(mountpoint: &Path) {
    let bin = std::env::var_os("FUSERMOUNT_PATH").unwrap_or_else(|| "fusermount3".into());
    match std::process::Command::new(&bin)
        .arg("-u")
        .arg("-z")
        .arg(mountpoint)
        .status()
    {
        Ok(status) if status.success() => {
            debug!(mountpoint = %mountpoint.display(), "unmounted stale transfer mount")
        }
        Ok(status) => warn!(%status, mountpoint = %mountpoint.display(), "fusermount3 -u failed"),
        Err(err) => warn!(error = %err, "failed to run fusermount3 -u"),
    }
}

// ---------------------------------------------------------------------------------------
// Peer requests
// ---------------------------------------------------------------------------------------

fn errno_from_quic<E: std::fmt::Display>(err: E) -> i32 {
    debug!(error = %err, "transfer stream failed");
    libc::EIO
}

async fn fetch_listing(
    conn: &quinn::Connection,
    offer_id: u64,
    path: Vec<String>,
) -> Result<Vec<FileEntry>, i32> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(errno_from_quic)?;
    write_bidi_stream_header(&mut send, &BidiStreamHeader::FileList { offer_id, path })
        .await
        .map_err(errno_from_quic)?;
    let _ = send.finish();
    match read_frame::<FileListReply, _>(&mut recv)
        .await
        .map_err(errno_from_quic)?
    {
        FileListReply::Ok(entries) => Ok(entries),
        FileListReply::Err(errno) => Err(errno),
    }
}

async fn open_read_stream(
    conn: &quinn::Connection,
    offer_id: u64,
    path: Vec<String>,
    offset: u64,
) -> Result<quinn::RecvStream, i32> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(errno_from_quic)?;
    write_bidi_stream_header(
        &mut send,
        &BidiStreamHeader::FileRead {
            offer_id,
            path,
            offset,
        },
    )
    .await
    .map_err(errno_from_quic)?;
    let _ = send.finish();
    match read_frame::<FileReadReply, _>(&mut recv)
        .await
        .map_err(errno_from_quic)?
    {
        FileReadReply::Ok => Ok(recv),
        FileReadReply::Err(errno) => Err(errno),
    }
}

impl ReadState {
    fn stream_pos(&self) -> u64 {
        self.buf_start + self.buf.len() as u64
    }

    fn stop_stream(&mut self) {
        if let Some(mut s) = self.stream.take() {
            let _ = s.stop(quinn::VarInt::from_u32(STOP_CODE));
        }
    }

    async fn read(&mut self, offset: u64, size: u32) -> Result<Vec<u8>, i32> {
        if self.transfer.is_cancelled() {
            return Err(libc::ECANCELED);
        }
        if offset >= self.size {
            return Ok(Vec::new());
        }
        let end = self.size.min(offset + u64::from(size));
        let pos = self.stream_pos();
        if self.stream.is_none() && offset >= self.buf_start && end <= pos {
            // Fully inside the window of an already finished stream.
        } else if self.stream.is_none() || offset < self.buf_start || offset > pos + MAX_SKIP {
            self.stop_stream();
            self.buf.clear();
            self.buf_start = offset;
            self.stream =
                Some(open_read_stream(&self.conn, self.offer_id, self.path.clone(), offset).await?);
        }
        let mut chunk = vec![0u8; 256 * 1024];
        while self.stream_pos() < end {
            let Some(stream) = self.stream.as_mut() else {
                break;
            };
            match stream.read(&mut chunk).await.map_err(errno_from_quic)? {
                Some(n) => {
                    self.buf.extend_from_slice(&chunk[..n]);
                    self.transfer.advance(0, n as u64);
                }
                None => {
                    // EOF before the announced size: the file shrank on the owner's side.
                    self.stream = None;
                    break;
                }
            }
        }
        let have_end = end.min(self.stream_pos());
        let data = if offset < have_end {
            self.buf[(offset - self.buf_start) as usize..(have_end - self.buf_start) as usize]
                .to_vec()
        } else {
            Vec::new()
        };
        self.transfer.advance(have_end, 0);
        // Keep only the trailing window.
        let pos = self.stream_pos();
        if pos - self.buf_start > READ_WINDOW {
            let keep_from = pos - READ_WINDOW;
            self.buf.drain(..(keep_from - self.buf_start) as usize);
            self.buf_start = keep_from;
        }
        Ok(data)
    }
}

/// The process whose thread `tid` made a request, looked up while the request is new: a
/// thread can end before its request does (readahead, short-lived worker threads), and
/// only its process ending says the reader is gone.
fn requester(tid: u32) -> Option<(u32, u32)> {
    if tid == 0 || !pids_resolvable() {
        // Not a process's request, or one we cannot look up.
        return None;
    }
    let status = std::fs::read_to_string(format!("/proc/{tid}/status")).ok()?;
    let tgid = status
        .lines()
        .find_map(|l| l.strip_prefix("Tgid:"))?
        .trim()
        .parse()
        .ok()?;
    Some((tid, tgid))
}

/// Resolves once the process that made a request (see [`requester`]) is gone or being
/// killed (a fatal signal is turned into SIGKILL for every thread of the process); never
/// for an unknown requester. Checked every half second.
async fn being_killed(requester: Option<(u32, u32)>) {
    let Some((tid, tgid)) = requester else {
        return std::future::pending().await;
    };
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let state = match kill_state(tid) {
            // The thread ended; its process may live on.
            KillState::Gone => kill_state(tgid),
            state => state,
        };
        if state != KillState::Alive {
            return;
        }
    }
}

/// Whether the pids FUSE reports are ours to look up in `/proc`. In a Flatpak sandbox the
/// mount is made by the host's fusermount3 (see `packaging/flatpak/fusermount3-wrapper.sh`),
/// so FUSE reports host pids, which the sandbox's `/proc` does not show (or shows as other
/// processes).
fn pids_resolvable() -> bool {
    static RESOLVABLE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| !Path::new("/.flatpak-info").exists());
    *RESOLVABLE
}

#[derive(Debug, PartialEq, Eq)]
enum KillState {
    Alive,
    /// SIGKILL is pending.
    Killed,
    Gone,
}

/// Thread or process `id`'s state, from `/proc/<id>/status`. A status that cannot be read
/// for another reason than its absence (a restricted `/proc`) counts as alive.
fn kill_state(id: u32) -> KillState {
    let status = match std::fs::read_to_string(format!("/proc/{id}/status")) {
        Ok(status) => status,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return KillState::Gone,
        Err(_) => return KillState::Alive,
    };
    let kill_bit = 1u64 << (libc::SIGKILL - 1);
    let killed = status.lines().any(|line| {
        let Some((name, value)) = line.split_once(':') else {
            return false;
        };
        (name == "SigPnd" || name == "ShdPnd")
            && u64::from_str_radix(value.trim(), 16).is_ok_and(|v| v & kill_bit != 0)
    });
    if killed {
        KillState::Killed
    } else {
        KillState::Alive
    }
}

impl Drop for ReadState {
    fn drop(&mut self) {
        self.stop_stream();
    }
}

// ---------------------------------------------------------------------------------------
// FUSE callbacks
// ---------------------------------------------------------------------------------------

struct XferFs {
    shared: Arc<Shared>,
}

/// Make sure the children of directory `ino` are known, fetching them from the peer.
async fn ensure_listed(shared: &Shared, ino: u64) -> Result<(), i32> {
    let (conn, offer_id, path) = {
        let st = shared.state.lock().unwrap();
        let node = st.nodes.get(&ino).ok_or(libc::ENOENT)?;
        if node.children.is_some() {
            return Ok(());
        }
        let offer = st.offers.get(&node.offer_dir).ok_or(libc::ENOENT)?;
        (offer.conn.clone(), offer.offer_id, node.path.clone())
    };
    let entries = tokio::time::timeout(LIST_DEADLINE, fetch_listing(&conn, offer_id, path))
        .await
        .map_err(|_| libc::ETIMEDOUT)??;
    let mut st = shared.state.lock().unwrap();
    let Some(node) = st.nodes.get(&ino) else {
        return Err(libc::ENOENT);
    };
    if node.children.is_some() {
        return Ok(());
    }
    let (offer_dir, base_path) = (node.offer_dir, node.path.clone());
    let mut children = BTreeMap::new();
    for entry in entries {
        if !crate::xfer::is_valid_name(&entry.name) || children.contains_key(&entry.name) {
            continue;
        }
        let child = st.alloc_ino();
        let mut path = base_path.clone();
        path.push(entry.name.clone());
        st.nodes
            .insert(child, Node::from_entry(ino, offer_dir, path, &entry));
        children.insert(entry.name, child);
    }
    if let Some(node) = st.nodes.get_mut(&ino) {
        node.children = Some(children);
    }
    Ok(())
}

/// The mount's limit on outstanding background requests (readahead), 12 by default. Each
/// file of a paused transfer keeps its readers' requests outstanding (held), and at the
/// limit the kernel queues every other file's readahead behind them. The kernel caps it
/// for unprivileged mounts (`max_user_bgreq`, thousands by default).
const MAX_BACKGROUND: u16 = 1024;

impl Filesystem for XferFs {
    fn init(&mut self, _req: &Request, config: &mut KernelConfig) -> std::io::Result<()> {
        // Congestion only slows the kernel's own writeback: keep the default 3/4.
        if let Err(max) = config.set_max_background(MAX_BACKGROUND) {
            warn!(
                max,
                "cannot raise the transfer mount's background request limit"
            );
        }
        Ok(())
    }

    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let shared = Arc::clone(&self.shared);
        let parent = parent.0;
        let name = name.to_owned();
        self.shared.rt.spawn(async move {
            if let Err(errno) = ensure_listed(&shared, parent).await {
                reply.error(Errno::from_i32(errno));
                return;
            }
            let st = shared.state.lock().unwrap();
            let child = name.to_str().and_then(|n| {
                st.nodes
                    .get(&parent)
                    .and_then(|p| p.children.as_ref())
                    .and_then(|c| c.get(n).copied())
            });
            match child.and_then(|ino| st.attr(ino, shared.uid, shared.gid)) {
                Some(attr) => reply.entry(&TTL, &attr, Generation(0)),
                None => reply.error(Errno::ENOENT),
            }
        });
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        let st = self.shared.state.lock().unwrap();
        match st.attr(ino.0, self.shared.uid, self.shared.gid) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn access(&self, _req: &Request, ino: INodeNo, mask: AccessFlags, reply: ReplyEmpty) {
        let st = self.shared.state.lock().unwrap();
        if !st.nodes.contains_key(&ino.0) {
            reply.error(Errno::ENOENT);
        } else if mask.contains(AccessFlags::W_OK) {
            reply.error(Errno::EROFS);
        } else {
            reply.ok();
        }
    }

    fn open(
        &self,
        _req: &Request,
        ino: INodeNo,
        flags: OpenFlags,
        _kill_suid_gid: bool,
        reply: ReplyOpen,
    ) {
        if flags.0 & libc::O_ACCMODE != libc::O_RDONLY {
            reply.error(Errno::EROFS);
            return;
        }
        let mut st = self.shared.state.lock().unwrap();
        let Some(node) = st.nodes.get(&ino.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        if node.kind != FileKind::File {
            reply.error(Errno::EISDIR);
            return;
        }
        let (offer_dir, path, size) = (node.offer_dir, node.path.clone(), node.size);
        let Some(offer) = st.offers.get_mut(&offer_dir) else {
            reply.error(Errno::ENOENT);
            return;
        };
        offer.open_files += 1;
        let transfer = transfers::track(
            offer.conn.stable_id(),
            TransferDirection::Incoming,
            offer.offer_id,
            &path,
            size,
        );
        let cancelled = transfer.cancelled();
        let read_state = ReadState {
            transfer,
            conn: offer.conn.clone(),
            offer_id: offer.offer_id,
            path,
            size,
            stream: None,
            buf: Vec::new(),
            buf_start: 0,
        };
        let fh = st.next_fh;
        st.next_fh += 1;
        let handle = Arc::new(tokio::sync::Mutex::new(read_state));
        // A cancel stops the stream at once, even while the app is not reading (paused).
        let weak = Arc::downgrade(&handle);
        self.shared.rt.spawn(async move {
            if !cancelled.await {
                return;
            }
            if let Some(handle) = weak.upgrade() {
                handle.lock().await.stop_stream();
            }
        });
        st.handles.insert(fh, handle);
        reply.opened(FileHandle(fh), FopenFlags::FOPEN_KEEP_CACHE);
    }

    fn read(
        &self,
        req: &Request,
        ino: INodeNo,
        fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let (handle, conn) = {
            let st = self.shared.state.lock().unwrap();
            let conn = st
                .nodes
                .get(&ino.0)
                .and_then(|n| st.offers.get(&n.offer_dir))
                .map(|o| o.conn.clone());
            (st.handles.get(&fh.0).cloned(), conn)
        };
        let Some(handle) = handle else {
            reply.error(Errno::EBADF);
            return;
        };
        let requester = requester(req.pid());
        let mut closing = self.shared.closing.subscribe();
        self.shared.rt.spawn(async move {
            // A paused transfer holds the read here, outside the deadline: pausing must not
            // turn into an error. Reads of the same handle queue behind this one's lock,
            // and are held the same way. fuser ignores FUSE_INTERRUPT, so a reader killed
            // meanwhile would leave the request (and the transfer, listed as paused) held
            // forever: let it go once its reader is being killed or gone, when the
            // connection closes (nobody is left to resume it), or when unmounting.
            let acquire = async {
                let rs = Arc::clone(&handle).lock_owned().await;
                rs.transfer.unpaused().await;
                rs
            };
            let conn_closed = async {
                match &conn {
                    Some(conn) => drop(conn.closed().await),
                    None => std::future::pending().await,
                }
            };
            let mut rs = tokio::select! {
                biased;
                rs = acquire => rs,
                () = being_killed(requester) => {
                    reply.error(Errno::EINTR);
                    return;
                }
                () = conn_closed => {
                    reply.error(Errno::EIO);
                    return;
                }
                _ = closing.wait_for(|c| *c) => {
                    reply.error(Errno::EIO);
                    return;
                }
            };
            let cancelled = rs.transfer.cancelled();
            let read = async {
                tokio::select! {
                    res = rs.read(offset, size) => res,
                    true = cancelled => Err(libc::ECANCELED),
                }
            };
            let res = tokio::time::timeout(READ_DEADLINE, read).await;
            match res {
                Ok(Ok(data)) => reply.data(&data),
                Ok(Err(errno)) => {
                    rs.stop_stream();
                    reply.error(Errno::from_i32(errno));
                }
                Err(_) => {
                    warn!(offset, "transfer read timed out");
                    rs.stop_stream();
                    reply.error(Errno::EIO);
                }
            }
        });
    }

    fn flush(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _lock_owner: LockOwner,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn getxattr(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _name: &OsStr,
        _size: u32,
        reply: ReplyXattr,
    ) {
        reply.error(Errno::ENODATA);
    }

    fn listxattr(&self, _req: &Request, _ino: INodeNo, size: u32, reply: ReplyXattr) {
        if size == 0 {
            reply.size(0);
        } else {
            reply.data(&[]);
        }
    }

    fn release(
        &self,
        _req: &Request,
        ino: INodeNo,
        fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        let mut st = self.shared.state.lock().unwrap();
        st.handles.remove(&fh.0);
        if let Some(offer_dir) = st.nodes.get(&ino.0).map(|n| n.offer_dir) {
            if let Some(offer) = st.offers.get_mut(&offer_dir) {
                offer.open_files = offer.open_files.saturating_sub(1);
            }
        }
        reply.ok();
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let shared = Arc::clone(&self.shared);
        let ino = ino.0;
        self.shared.rt.spawn(async move {
            if let Err(errno) = ensure_listed(&shared, ino).await {
                reply.error(Errno::from_i32(errno));
                return;
            }
            let st = shared.state.lock().unwrap();
            let Some(node) = st.nodes.get(&ino) else {
                reply.error(Errno::ENOENT);
                return;
            };
            if node.kind != FileKind::Dir {
                reply.error(Errno::ENOTDIR);
                return;
            }
            let mut entries: Vec<(u64, FileType, &str)> = vec![
                (ino, FileType::Directory, "."),
                (node.parent, FileType::Directory, ".."),
            ];
            for (name, &child) in node.children.iter().flatten() {
                let kind = match st.nodes.get(&child).map(|n| n.kind) {
                    Some(FileKind::Dir) => FileType::Directory,
                    Some(FileKind::File) => FileType::RegularFile,
                    None => continue,
                };
                entries.push((child, kind, name.as_str()));
            }
            for (i, (child, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
                if reply.add(INodeNo(child), (i + 1) as u64, kind, name) {
                    break;
                }
            }
            reply.ok();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_state_sees_live_and_gone_processes() {
        // SAFETY: gettid has no preconditions.
        let me = unsafe { libc::gettid() } as u32;
        assert_eq!(kill_state(me), KillState::Alive);
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let gone = child.id();
        child.wait().unwrap();
        assert_eq!(
            kill_state(gone),
            KillState::Gone,
            "a reaped process is gone"
        );
    }

    #[tokio::test]
    async fn a_requester_outlives_its_thread_while_its_process_lives() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // SAFETY: gettid has no preconditions.
            tx.send(requester(unsafe { libc::gettid() } as u32))
                .unwrap();
        })
        .join()
        .unwrap();
        let who = rx.recv().unwrap().expect("resolvable outside a sandbox");
        assert_eq!(who.1, std::process::id());
        // Its thread has ended, but this process lives on.
        assert!(
            tokio::time::timeout(Duration::from_millis(1200), being_killed(Some(who)))
                .await
                .is_err()
        );
    }

    #[test]
    fn cleanup_spares_young_unmounted_dirs() {
        let base = std::env::temp_dir().join(format!("oriel-xfer-cleanup-{}", std::process::id()));
        let young = base.join("xfer-young");
        let old = base.join("xfer-old");
        let other = base.join("keep-old");
        for dir in [&young, &old, &other] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let past = SystemTime::now() - UNMOUNTED_DIR_MIN_AGE - Duration::from_secs(5);
        for dir in [&old, &other] {
            std::fs::File::open(dir)
                .unwrap()
                .set_modified(past)
                .unwrap();
        }

        cleanup_stale_mounts(&base, "xfer-");

        assert!(
            young.exists(),
            "a directory being mounted right now must survive"
        );
        assert!(!old.exists());
        assert!(other.exists(), "only the given prefix is cleaned");
        let _ = std::fs::remove_dir_all(&base);
    }
}
