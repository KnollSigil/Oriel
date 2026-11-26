//! Client side of streamed clipboard file transfer (see `oriel_proto::xfer`).

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use oriel_proto::xfer::{FileEntry, OfferSlot, XferOwner};
use oriel_proto::xfer_fuse::{cleanup_stale_mounts, XferMount};
use oriel_proto::{read_bidi_stream_header, BidiStreamHeader};
use tracing::{debug, warn};
use wayland_client::protocol::wl_data_offer::WlDataOffer;

/// Local files offered to servers. Shared by every session in this process; offer ids are
/// random, so a server can only read what it was offered.
pub static OWNER: LazyLock<Arc<XferOwner>> = LazyLock::new(|| Arc::new(XferOwner::new()));

/// Prefix of this client's transfer mount directories in its runtime dir. Each process
/// mounts its own (random suffix: PIDs repeat across Flatpak sandboxes).
const MOUNT_PREFIX: &str = "xfer-";

/// Runtime for FUSE request handling. It is the process's own, so the mount keeps working
/// when a session's runtime goes away.
static FUSE_RUNTIME: LazyLock<Option<tokio::runtime::Runtime>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("oriel-xfer")
        .enable_all()
        .build()
        .inspect_err(|err| warn!(error = %err, "failed to start file transfer runtime"))
        .ok()
});

enum MountSlot {
    Unmounted,
    Mounted(Arc<XferMount>),
    Failed(Instant),
    /// After [`unmount`]: the process is exiting, so never mount again.
    Closed,
}

/// How long after a failed mount the next offer tries again.
const MOUNT_RETRY: Duration = Duration::from_secs(30);

static MOUNT: Mutex<MountSlot> = Mutex::new(MountSlot::Unmounted);

/// Where the transfer mount is, readable while `MOUNT` is locked by a stuck unmount.
static MOUNTPOINT: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The transfer mount where files offered by servers appear, mounted on first use.
fn mount() -> Option<Arc<XferMount>> {
    let mut slot = MOUNT.lock().unwrap();
    match &*slot {
        MountSlot::Mounted(m) => return Some(Arc::clone(m)),
        MountSlot::Failed(at) if at.elapsed() < MOUNT_RETRY => return None,
        MountSlot::Closed => return None,
        MountSlot::Unmounted | MountSlot::Failed(_) => {}
    }
    let mounted = try_mount();
    *slot = match &mounted {
        Some(m) => MountSlot::Mounted(Arc::clone(m)),
        None => MountSlot::Failed(Instant::now()),
    };
    mounted
}

fn try_mount() -> Option<Arc<XferMount>> {
    let rt = FUSE_RUNTIME.as_ref()?;
    let dir = crate::pty::oriel_runtime_dir()
        .inspect_err(|err| warn!(error = %format!("{err:#}"), "file transfer unavailable"))
        .ok()?;
    cleanup_stale_mounts(&dir, MOUNT_PREFIX);
    let mountpoint = dir.join(format!("{MOUNT_PREFIX}{:016x}", rand_suffix()));
    match XferMount::mount(&mountpoint, rt.handle().clone()) {
        Ok(m) => {
            *MOUNTPOINT.lock().unwrap() = Some(mountpoint);
            Some(Arc::new(m))
        }
        Err(err) => {
            warn!(error = %format!("{err:#}"), "file transfer unavailable: cannot mount FUSE");
            let _ = std::fs::remove_dir(&mountpoint);
            None
        }
    }
}

fn rand_suffix() -> u64 {
    let mut buf = [0u8; 8];
    let _ = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf));
    u64::from_ne_bytes(buf) ^ u64::from(std::process::id())
}

/// Unmount transfer mounts left behind by client processes that crashed. Runs in the
/// background: unmounting goes through `fusermount3`.
pub fn cleanup_stale_in_background() {
    std::thread::spawn(|| {
        if let Ok(dir) = crate::pty::oriel_runtime_dir() {
            cleanup_stale_mounts(&dir, MOUNT_PREFIX);
        }
    });
}

/// Present files the server offered and return their local paths (empty if the mount is
/// unavailable). Blocks on the first call while mounting.
pub fn receive_offer(
    conn: quinn::Connection,
    slot: OfferSlot,
    offer_id: u64,
    roots: Vec<FileEntry>,
) -> Vec<PathBuf> {
    match mount() {
        Some(m) => m.add_offer(conn, slot, offer_id, roots),
        None => Vec::new(),
    }
}

/// Unmount the transfer mount, if any, and remove its directory. Call before exiting.
pub fn unmount() {
    let slot = std::mem::replace(&mut *MOUNT.lock().unwrap(), MountSlot::Closed);
    if let MountSlot::Mounted(m) = slot {
        m.unmount();
        let _ = std::fs::remove_dir(m.mountpoint());
    }
}

/// Fetches types of a server's selection on demand (see `oriel_proto::selection`).
#[derive(Clone)]
pub struct SelectionFetcher {
    pub conn: quinn::Connection,
    pub rt: tokio::runtime::Handle,
}

impl std::fmt::Debug for SelectionFetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SelectionFetcher")
    }
}

impl SelectionFetcher {
    /// Fetch type `mime` of the server's selection `offer_id` into the pipe `fd`.
    pub fn fetch(&self, offer_id: u64, mime: String, fd: std::os::fd::OwnedFd) {
        let conn = self.conn.clone();
        self.rt.spawn(async move {
            if let Err(err) =
                oriel_proto::selection::fetch_selection_data(&conn, offer_id, mime, fd).await
            {
                debug!(error = %format!("{err:#}"), "clipboard fetch from server failed");
            }
        });
    }
}

/// Session bus connection for FileTransfer portal transfers, kept for the process
/// lifetime: the portal ends a transfer when the connection that started it closes.
fn portal_bus() -> anyhow::Result<zbus::blocking::Connection> {
    static BUS: Mutex<Option<zbus::blocking::Connection>> = Mutex::new(None);
    let mut slot = BUS.lock().unwrap();
    if let Some(bus) = slot.as_ref() {
        return Ok(bus.clone());
    }
    let bus = zbus::blocking::Connection::session()?;
    *slot = Some(bus.clone());
    Ok(bus)
}

/// Answer a sandboxed app's request for files from a server: register them with the
/// host's FileTransfer portal and write the key to `fd`, on a worker thread.
pub fn write_portal_key(paths: Vec<PathBuf>, fd: std::os::fd::OwnedFd) {
    std::thread::spawn(move || {
        let key =
            portal_bus().and_then(|bus| oriel_proto::selection::portal_transfer_key(&bus, &paths));
        match key {
            Ok(key) => {
                let _ = std::io::Write::write_all(&mut std::fs::File::from(fd), &key);
            }
            Err(err) => {
                warn!(error = %format!("{err:#}"), "FileTransfer portal registration failed")
            }
        }
    });
}

/// Local selections (clipboard contents, drags) offered to servers, by offer id, so a
/// server's paste can read any type from the app that owns it. wayland-client objects can
/// send requests from any thread, so reads need no trip through a window loop.
static LIVE_SOURCES: Mutex<Vec<LiveSource>> = Mutex::new(Vec::new());

struct LiveSource {
    slot: OfferSlot,
    offer_id: u64,
    offer: WlDataOffer,
    conn: wayland_client::Connection,
}

/// Live sources kept per session (Wayland connection) and slot: the current one, and the
/// one before it (a dropped drag may still be read while the next one starts).
const KEPT_LIVE_SOURCES: usize = 2;

/// Make local selection `offer` (the clipboard or a drag, per `slot`) readable by servers
/// as `offer_id`.
pub fn register_live_source(
    slot: OfferSlot,
    offer_id: u64,
    offer: WlDataOffer,
    conn: wayland_client::Connection,
) {
    let mut sources = LIVE_SOURCES.lock().unwrap();
    let same = |s: &LiveSource| s.slot == slot && s.conn == conn;
    while sources.iter().filter(|s| same(s)).count() >= KEPT_LIVE_SOURCES {
        let oldest = sources.iter().position(same).expect("counted above");
        sources.remove(oldest);
    }
    sources.push(LiveSource {
        slot,
        offer_id,
        offer,
        conn,
    });
}

/// Ask the owner of live source `offer_id` for type `mime`: the read end of the pipe it
/// writes to. `None` for an unknown offer. A replaced offer has been destroyed, so the
/// request is ignored and the pipe ends empty.
fn open_live_source(offer_id: u64, mime: String) -> Option<std::os::fd::OwnedFd> {
    let (offer, conn) = {
        let sources = LIVE_SOURCES.lock().unwrap();
        let source = sources.iter().find(|s| s.offer_id == offer_id)?;
        (source.offer.clone(), source.conn.clone())
    };
    let pipe =
        smithay_client_toolkit::data_device_manager::data_offer::receive(&offer, mime).ok()?;
    let _ = conn.flush();
    Some(pipe.into())
}

/// Write end of the pipe the termination signal handler reports signals on.
static SIGNAL_PIPE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

extern "C" fn on_termination_signal(sig: libc::c_int) {
    let fd = SIGNAL_PIPE.load(std::sync::atomic::Ordering::Relaxed);
    let byte = sig as u8;
    // SAFETY: write(2) is async-signal-safe; `byte` outlives the call.
    unsafe { libc::write(fd, (&byte as *const u8).cast(), 1) };
}

/// How long a clean unmount may take after a termination signal.
const UNMOUNT_GRACE: Duration = Duration::from_secs(3);

/// The next signal number reported on the signal pipe `fd`; `None` on error or after
/// `timeout`.
fn read_signal(fd: libc::c_int, timeout: Option<Duration>) -> Option<i32> {
    let deadline = timeout.map(|t| Instant::now() + t);
    loop {
        if let Some(deadline) = deadline {
            let rem = deadline.saturating_duration_since(Instant::now());
            let mut pfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: `pfd` is valid for the call.
            let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis() as i32) };
            if rc == 0 {
                return None;
            }
            if rc < 0 {
                if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return None;
            }
        }
        let mut byte = 0u8;
        // SAFETY: reads one byte into `byte`.
        let n = unsafe { libc::read(fd, (&mut byte as *mut u8).cast(), 1) };
        if n == 1 {
            return Some(i32::from(byte));
        }
        if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return None;
    }
}

/// On SIGTERM, SIGINT or SIGHUP, unmount the transfer mount and exit (128 + signal), so a
/// killed client does not leave a dead mount behind. A handler, not a blocked signal mask:
/// handlers reset on exec, so the programs this client starts keep default behaviour.
pub fn unmount_on_termination_signals() {
    let mut fds = [0; 2];
    // SAFETY: `fds` has room for the two descriptors pipe2 returns.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        warn!("cannot watch termination signals: pipe2 failed");
        return;
    }
    let [read_fd, write_fd] = fds;
    SIGNAL_PIPE.store(write_fd, std::sync::atomic::Ordering::Relaxed);
    let spawned = std::thread::Builder::new()
        .name("oriel-signals".into())
        .spawn(move || {
            let Some(sig) = read_signal(read_fd, None) else {
                return;
            };
            // Unmount cleanly, unless that hangs (a FUSE request stuck on a dead
            // connection) or another signal insists: then detach lazily and exit anyway.
            std::thread::spawn(move || {
                unmount();
                std::process::exit(128 + sig);
            });
            let _ = read_signal(read_fd, Some(UNMOUNT_GRACE));
            if let Some(mountpoint) = MOUNTPOINT.lock().unwrap().clone() {
                oriel_proto::xfer_fuse::detach(&mountpoint);
            }
            std::process::exit(128 + sig);
        });
    if spawned.is_err() {
        warn!("cannot watch termination signals: thread spawn failed");
        return;
    }
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: installs a handler that only calls async-signal-safe write(2).
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_termination_signal as *const () as usize;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(sig, &action, std::ptr::null_mut());
        }
    }
}

/// Serve the server's reads of files this client offered, until the connection closes.
pub fn spawn_stream_acceptor(conn: quinn::Connection) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut streams = tokio::task::JoinSet::new();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            streams.spawn(async move {
                match read_bidi_stream_header(&mut recv).await {
                    Ok(BidiStreamHeader::SelectionData { offer_id, mime }) => {
                        let source = open_live_source(offer_id, mime);
                        oriel_proto::selection::serve_selection_data(source, &mut send).await;
                    }
                    Ok(
                        header @ (BidiStreamHeader::FileList { .. }
                        | BidiStreamHeader::FileRead { .. }),
                    ) => {
                        oriel_proto::xfer::serve_request(&OWNER, &header, &mut send).await;
                    }
                    Ok(other) => debug!(?other, "ignoring unexpected stream from server"),
                    Err(err) => warn!(error = %err, "failed to read stream header from server"),
                }
            });
            while streams.try_join_next().is_some() {}
        }
    })
}

/// Host directories the Flatpak manifest's `--filesystem=host:ro` shows at the same path
/// inside the sandbox. Other host paths (`/usr`, `/etc`, `/tmp`, ...) are hidden or replaced
/// by the sandbox's own, so a `file://` URI naming them must not be read here.
const SANDBOX_SHARED_HOST_DIRS: &[&str] = &[
    "/home",
    "/var/home",
    "/media",
    "/run/media",
    "/mnt",
    "/opt",
    "/srv",
    "/afs",
];

/// Whether host path `path` names the same file inside this process. Always true outside
/// Flatpak.
fn host_path_visible(path: &Path, in_flatpak: bool) -> bool {
    if !in_flatpak {
        return true;
    }
    // Lexical check: `..` could climb out of a shared directory.
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return false;
    }
    SANDBOX_SHARED_HOST_DIRS
        .iter()
        .any(|dir| path.starts_with(dir))
}

/// The paths of a `text/uri-list` selection from a host app that this process can read
/// (in Flatpak, those under [`SANDBOX_SHARED_HOST_DIRS`] or Oriel's own runtime dir, where
/// every Oriel client mounts the files its servers offer: a copy in one remote session
/// pasted into another). `None` when there are none.
pub fn visible_host_paths(paths: Vec<PathBuf>) -> Option<Vec<PathBuf>> {
    let in_flatpak = crate::pty::flatpak_app_id().is_some();
    let own_dir = crate::pty::oriel_runtime_dir().ok();
    let visible: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| {
            host_path_visible(p, in_flatpak)
                || own_dir.as_deref().is_some_and(|dir| {
                    !p.components().any(|c| matches!(c, Component::ParentDir)) && p.starts_with(dir)
                })
        })
        .collect();
    (!visible.is_empty()).then_some(visible)
}

/// Resolve an `org.freedesktop.portal.FileTransfer` key from a drag or clipboard offer
/// into paths this sandboxed process can read.
pub fn retrieve_portal_files(key: &str) -> Option<Vec<PathBuf>> {
    let key = key.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if key.is_empty() {
        return None;
    }
    let result = (|| -> zbus::Result<Vec<String>> {
        let conn = zbus::blocking::Connection::session()?;
        let proxy = zbus::blocking::Proxy::new(
            &conn,
            "org.freedesktop.portal.Documents",
            "/org/freedesktop/portal/documents",
            "org.freedesktop.portal.FileTransfer",
        )?;
        let options: HashMap<&str, zbus::zvariant::Value<'_>> = HashMap::new();
        proxy.call("RetrieveFiles", &(key, options))
    })();
    match result {
        Ok(files) if !files.is_empty() => Some(files.into_iter().map(PathBuf::from).collect()),
        Ok(_) => None,
        Err(err) => {
            warn!(error = %err, "FileTransfer portal RetrieveFiles failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_paths_visible_in_flatpak_only_under_shared_dirs() {
        let visible = |p: &str| host_path_visible(Path::new(p), true);
        assert!(visible("/home/u/file.txt"));
        assert!(visible("/run/media/u/usb/a"));
        assert!(visible("/mnt/data"));
        assert!(!visible("/homework/a"));
        assert!(!visible("/tmp/a"));
        assert!(!visible("/usr/share/a"));
        assert!(!visible("/home/../etc/passwd"));
        assert!(host_path_visible(Path::new("/tmp/a"), false));
    }
}
