//! Client side of streamed clipboard file transfer (see `oriel_proto::xfer`).

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use oriel_proto::xfer::{FileEntry, XferOwner};
use oriel_proto::xfer_fuse::{cleanup_stale_mounts, XferMount};
use oriel_proto::{read_bidi_stream_header, BidiStreamHeader};
use tracing::{debug, warn};

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
        Ok(m) => Some(Arc::new(m)),
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
    offer_id: u64,
    roots: Vec<FileEntry>,
) -> Vec<PathBuf> {
    match mount() {
        Some(m) => m.add_offer(conn, offer_id, roots),
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

/// Serve the server's reads of files this client offered, until the connection closes.
pub fn spawn_stream_acceptor(conn: quinn::Connection) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut streams = tokio::task::JoinSet::new();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            streams.spawn(async move {
                match read_bidi_stream_header(&mut recv).await {
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
/// (in Flatpak, those under [`SANDBOX_SHARED_HOST_DIRS`]). `None` when there are none.
pub fn visible_host_paths(paths: Vec<PathBuf>) -> Option<Vec<PathBuf>> {
    let in_flatpak = crate::pty::flatpak_app_id().is_some();
    let visible: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| host_path_visible(p, in_flatpak))
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
