//! Streamed file transfer for clipboard file offers.
//!
//! The side that owns the files registers them in an [`XferOwner`] and announces them with
//! [`ControlMessage::ClipboardOffer`](crate::ControlMessage::ClipboardOffer). The receiving
//! side presents them through FUSE ([`crate::xfer_fuse`]) and fetches directory listings and
//! file contents on demand over bidirectional QUIC streams opened with
//! [`BidiStreamHeader::FileList`](crate::BidiStreamHeader::FileList) and
//! [`BidiStreamHeader::FileRead`](crate::BidiStreamHeader::FileRead). Nothing is staged.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::Metadata;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeekExt, AsyncWrite, AsyncWriteExt};

use crate::transfers::{self, TransferDirection};

/// Most roots one offer may carry.
pub const MAX_OFFER_ROOTS: usize = 4096;
/// Most entries one directory listing may carry; larger directories are truncated.
pub const MAX_DIR_ENTRIES: usize = 65_536;
/// Largest reply frame accepted on a file stream (a full listing of long names fits).
const MAX_REPLY_FRAME: usize = 32 * 1024 * 1024;
/// Chunk size used when streaming file contents.
const READ_CHUNK: usize = 256 * 1024;
/// How many past offers an owner keeps servable after newer ones replace them.
const KEPT_OFFERS: usize = 16;

/// What an offer is for. Each slot keeps its own offers, so dragging files does not expire
/// the files on the clipboard (and the other way round).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OfferSlot {
    Clipboard,
    Drag,
}

/// Files offered to the peer: an [`XferOwner`] offer and its roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileOffer {
    pub offer_id: u64,
    pub roots: Vec<FileEntry>,
}

/// Kind of an offered filesystem entry. Symlinks and special files are never offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileKind {
    File,
    Dir,
}

/// Metadata of one offered file or directory, sent up front so that `stat()` on the
/// receiving side never waits for the network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// Single path component (see [`is_valid_name`]).
    pub name: String,
    pub kind: FileKind,
    /// Size in bytes (0 for directories).
    pub size: u64,
    /// Modification time, seconds since the Unix epoch.
    pub mtime_s: i64,
    /// Permission bits (`st_mode & 0o777`).
    pub mode: u32,
}

/// Reply to [`BidiStreamHeader::FileList`](crate::BidiStreamHeader::FileList).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileListReply {
    Ok(Vec<FileEntry>),
    /// The listing failed with this `errno`.
    Err(i32),
}

/// Reply to [`BidiStreamHeader::FileRead`](crate::BidiStreamHeader::FileRead); `Ok` is
/// followed by the file's raw bytes from the requested offset to EOF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileReadReply {
    Ok,
    /// Opening the file failed with this `errno`.
    Err(i32),
}

/// Whether `name` is usable as a single path component: non-empty, not `.` or `..`,
/// without `/` or NUL, and at most 255 bytes.
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name.len() <= 255
        && !name.bytes().any(|b| b == b'/' || b == 0)
}

/// Build a [`FileEntry`] from `lstat`-style metadata; `None` for symlinks, special files
/// and names that are not valid UTF-8 components.
pub fn entry_from_metadata(name: &str, meta: &Metadata) -> Option<FileEntry> {
    if !is_valid_name(name) {
        return None;
    }
    let ft = meta.file_type();
    let kind = if ft.is_file() {
        FileKind::File
    } else if ft.is_dir() {
        FileKind::Dir
    } else {
        return None;
    };
    Some(FileEntry {
        name: name.to_string(),
        kind,
        size: if kind == FileKind::File {
            meta.len()
        } else {
            0
        },
        mtime_s: meta.mtime(),
        mode: meta.mode() & 0o777,
    })
}

/// Files registered for transfer, keyed by offer id. Old offers stay servable until
/// [`KEPT_OFFERS`] newer ones exist, so a paste that is still copying keeps working
/// after the clipboard changes.
#[derive(Default)]
pub struct XferOwner {
    offers: Mutex<VecDeque<Offer>>,
}

struct Offer {
    id: u64,
    slot: OfferSlot,
    /// Root name (as announced) and the canonical path it resolves to.
    roots: Vec<(String, PathBuf)>,
}

impl XferOwner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `paths` as a new offer and return its id and announced roots. Paths that
    /// are not regular files or directories are skipped; returns `None` when nothing is
    /// left. Duplicate base names get a ` (n)` suffix.
    pub fn offer(&self, slot: OfferSlot, paths: &[PathBuf]) -> Option<(u64, Vec<FileEntry>)> {
        let mut roots = Vec::new();
        let mut entries: Vec<FileEntry> = Vec::new();
        for path in paths.iter().take(MAX_OFFER_ROOTS) {
            let Ok(canonical) = std::fs::canonicalize(path) else {
                continue;
            };
            let Ok(meta) = std::fs::metadata(&canonical) else {
                continue;
            };
            let Some(base) = path
                .file_name()
                .or_else(|| canonical.file_name())
                .and_then(|n| n.to_str())
            else {
                continue;
            };
            let name = unique_name(base, &entries);
            let Some(entry) = entry_from_metadata(&name, &meta) else {
                continue;
            };
            roots.push((name, canonical));
            entries.push(entry);
        }
        if entries.is_empty() {
            return None;
        }
        let id = random_u64();
        let mut offers = self.offers.lock().unwrap();
        offers.push_back(Offer { id, slot, roots });
        // Drop the oldest offer of this slot beyond the limit.
        if offers.iter().filter(|o| o.slot == slot).count() > KEPT_OFFERS {
            if let Some(i) = offers.iter().position(|o| o.slot == slot) {
                offers.remove(i);
            }
        }
        Some((id, entries))
    }

    /// Resolve `path` (root name, then components) inside offer `offer_id` to a local
    /// path, refusing invalid components and symlinks below the root.
    pub fn resolve(&self, offer_id: u64, path: &[String]) -> Result<PathBuf, i32> {
        let (root_name, rest) = path.split_first().ok_or(libc::EINVAL)?;
        let root = {
            let offers = self.offers.lock().unwrap();
            let offer = offers
                .iter()
                .find(|o| o.id == offer_id)
                .ok_or(libc::ENOENT)?;
            offer
                .roots
                .iter()
                .find(|(name, _)| name == root_name)
                .map(|(_, p)| p.clone())
                .ok_or(libc::ENOENT)?
        };
        let mut resolved = root;
        for component in rest {
            if !is_valid_name(component) {
                return Err(libc::EINVAL);
            }
            resolved.push(component);
            let meta = std::fs::symlink_metadata(&resolved).map_err(|e| errno_of(&e))?;
            if meta.file_type().is_symlink() {
                return Err(libc::ENOENT);
            }
        }
        Ok(resolved)
    }
}

fn unique_name(base: &str, taken: &[FileEntry]) -> String {
    if !taken.iter().any(|e| e.name == base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base} ({n})"))
        .find(|candidate| !taken.iter().any(|e| &e.name == candidate))
        .unwrap()
}

/// A random id (for offers).
pub fn random_u64() -> u64 {
    let mut buf = [0u8; 8];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf))
        .is_err()
    {
        // Offer ids only need to be unique; unpredictability is a bonus.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        return nanos ^ u64::from(std::process::id()).rotate_left(32);
    }
    u64::from_ne_bytes(buf)
}

fn errno_of(err: &std::io::Error) -> i32 {
    err.raw_os_error().unwrap_or(libc::EIO)
}

/// List the directory at `dir`, skipping symlinks, special files and non-UTF-8 names.
pub fn list_dir(dir: &Path) -> Result<Vec<FileEntry>, i32> {
    let meta = std::fs::symlink_metadata(dir).map_err(|e| errno_of(&e))?;
    if !meta.is_dir() {
        return Err(libc::ENOTDIR);
    }
    let mut entries = Vec::new();
    for item in std::fs::read_dir(dir).map_err(|e| errno_of(&e))? {
        let Ok(item) = item else { continue };
        let name = item.file_name();
        let Some(name) = name.to_str() else { continue };
        let Ok(meta) = item.metadata() else { continue };
        if let Some(entry) = entry_from_metadata(name, &meta) {
            entries.push(entry);
            if entries.len() >= MAX_DIR_ENTRIES {
                break;
            }
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Serve a [`BidiStreamHeader::FileList`](crate::BidiStreamHeader::FileList) request.
pub async fn serve_list<W: AsyncWrite + Unpin>(
    owner: &Arc<XferOwner>,
    offer_id: u64,
    path: &[String],
    writer: &mut W,
) -> Result<()> {
    let reply = match resolve_blocking(owner, offer_id, path).await {
        Ok(dir) => match tokio::task::spawn_blocking(move || list_dir(&dir)).await {
            Ok(Ok(entries)) => FileListReply::Ok(entries),
            Ok(Err(errno)) => FileListReply::Err(errno),
            Err(_) => FileListReply::Err(libc::EIO),
        },
        Err(errno) => FileListReply::Err(errno),
    };
    write_frame(writer, &reply).await
}

/// [`XferOwner::resolve`] off the async runtime: each path component is a `stat`, which
/// for files offered from another transfer mount (a copy between two remote sessions) is
/// a FUSE lookup that may wait on the network.
async fn resolve_blocking(
    owner: &Arc<XferOwner>,
    offer_id: u64,
    path: &[String],
) -> Result<PathBuf, i32> {
    let owner = Arc::clone(owner);
    let path = path.to_vec();
    tokio::task::spawn_blocking(move || owner.resolve(offer_id, &path))
        .await
        .unwrap_or(Err(libc::EIO))
}

/// Serve a [`BidiStreamHeader::FileRead`](crate::BidiStreamHeader::FileRead) request from
/// the peer on connection `conn` (`quinn::Connection::stable_id`): stream the file from
/// `offset` until EOF, until the reader stops the stream, or until the user cancels the
/// transfer (an error, so the stream is reset rather than finished short).
pub async fn serve_read<W: AsyncWrite + Unpin>(
    owner: &Arc<XferOwner>,
    conn: usize,
    offer_id: u64,
    path: &[String],
    offset: u64,
    writer: &mut W,
) -> Result<()> {
    let file = match resolve_blocking(owner, offer_id, path).await {
        Ok(p) => open_regular(&p, offset).await,
        Err(errno) => Err(errno),
    };
    let (mut file, size) = match file {
        Ok(f) => f,
        Err(errno) => return write_frame(writer, &FileReadReply::Err(errno)).await,
    };
    let transfer = transfers::track(conn, TransferDirection::Outgoing, offer_id, path, size);
    if transfer.is_cancelled() {
        return write_frame(writer, &FileReadReply::Err(libc::ECANCELED)).await;
    }
    write_frame(writer, &FileReadReply::Ok).await?;
    let cancelled = transfer.cancelled();
    tokio::pin!(cancelled);
    let mut buf = vec![0u8; READ_CHUNK];
    let mut position = offset;
    loop {
        let n = file.read(&mut buf).await.context("file read failed")?;
        if n == 0 {
            return Ok(());
        }
        if transfer.is_cancelled() {
            bail!("transfer cancelled");
        }
        // A paused reader leaves the write waiting on flow control. A cancel comes first.
        tokio::select! {
            biased;
            true = &mut cancelled => bail!("transfer cancelled"),
            res = writer.write_all(&buf[..n]) => res?,
        }
        position += n as u64;
        transfer.advance(position, n as u64);
    }
}

/// Serve a [`BidiStreamHeader::FileList`](crate::BidiStreamHeader::FileList) or
/// [`BidiStreamHeader::FileRead`](crate::BidiStreamHeader::FileRead) stream and finish
/// it. Returns `false` (leaving the stream alone) for any other header.
pub async fn serve_request(
    owner: &Arc<XferOwner>,
    conn: usize,
    header: &crate::BidiStreamHeader,
    send: &mut quinn::SendStream,
) -> bool {
    let res = match header {
        crate::BidiStreamHeader::FileList { offer_id, path } => {
            serve_list(owner, *offer_id, path, send).await
        }
        crate::BidiStreamHeader::FileRead {
            offer_id,
            path,
            offset,
        } => serve_read(owner, conn, *offer_id, path, *offset, send).await,
        _ => return false,
    };
    match res {
        Ok(()) => {
            let _ = send.finish();
        }
        // The reader stopping the stream (cancel or seek) is the normal way reads end.
        // Reset rather than finish: a finished stream reads as a file that got shorter.
        Err(err) => {
            tracing::debug!(error = %err, "file transfer stream ended early");
            let _ = send.reset(quinn::VarInt::from_u32(0));
        }
    }
    true
}

/// The regular file at `path`, positioned at `offset`, and its size.
async fn open_regular(path: &Path, offset: u64) -> Result<(tokio::fs::File, u64), i32> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| errno_of(&e))?;
    let meta = file.metadata().await.map_err(|e| errno_of(&e))?;
    if !meta.is_file() {
        return Err(libc::EISDIR);
    }
    file.seek(std::io::SeekFrom::Start(offset))
        .await
        .map_err(|e| errno_of(&e))?;
    Ok((file, meta.len()))
}

/// Write one length-prefixed postcard frame.
pub async fn write_frame<T: Serialize, W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &T,
) -> Result<()> {
    let encoded = postcard::to_allocvec(msg).context("failed to serialize frame")?;
    if encoded.len() > MAX_REPLY_FRAME {
        bail!("frame too large: {}", encoded.len());
    }
    writer.write_u32(encoded.len() as u32).await?;
    writer.write_all(&encoded).await?;
    Ok(())
}

/// Read one length-prefixed postcard frame written by [`write_frame`].
pub async fn read_frame<T: DeserializeOwned, R: AsyncRead + Unpin>(reader: &mut R) -> Result<T> {
    let len = reader.read_u32().await? as usize;
    if len > MAX_REPLY_FRAME {
        bail!("frame too large: {len}");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    postcard::from_bytes(&buf).context("failed to deserialize frame")
}

// ---------------------------------------------------------------------------------------
// Selection formats
// ---------------------------------------------------------------------------------------

/// MIME type of a list of URIs (RFC 2483), understood by every toolkit.
pub const URI_LIST_MIME: &str = "text/uri-list";
/// Nautilus' copy/cut clipboard format: `copy` or `cut`, then one URI per line.
pub const GNOME_COPIED_FILES_MIME: &str = "x-special/gnome-copied-files";
/// KDE's cut marker: `1` for cut, `0` for copy.
pub const KDE_CUT_SELECTION_MIME: &str = "application/x-kde-cutselection";
/// Key of an `org.freedesktop.portal.FileTransfer` session.
pub const PORTAL_FILE_TRANSFER_MIME: &str = "application/vnd.portal.filetransfer";

/// MIME types a receiver offers for a file selection, in preference order. Plain-text
/// types (the paths, one per line) are appended by the caller.
pub const FILE_SELECTION_MIMES: &[&str] = &[
    URI_LIST_MIME,
    GNOME_COPIED_FILES_MIME,
    KDE_CUT_SELECTION_MIME,
];

/// Percent-encode `path` as a `file://` URI.
pub fn file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for &b in path.as_os_str().as_bytes() {
        if b.is_ascii_alphanumeric() || b"/-._~".contains(&b) {
            uri.push(b as char);
        } else {
            uri.push_str(&format!("%{b:02X}"));
        }
    }
    uri
}

/// Parse a `text/uri-list` body into local paths. Returns `None` unless every URI is a
/// local `file://` URI (so non-file selections such as web links stay text).
pub fn parse_file_uri_list(body: &str) -> Option<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for line in body.lines() {
        let line = line.trim_end_matches(['\r', '\0']).trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let rest = line.strip_prefix("file://")?;
        // Accept an empty authority or `localhost`; anything else is a remote host.
        let path = if rest.starts_with('/') {
            rest
        } else {
            rest.strip_prefix("localhost")
                .filter(|p| p.starts_with('/'))?
        };
        let bytes = percent_decode(path)?;
        if bytes.contains(&0) {
            return None;
        }
        paths.push(PathBuf::from(OsString::from_vec(bytes)));
    }
    if paths.is_empty() {
        None
    } else {
        Some(paths)
    }
}

fn percent_decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Some(out)
}

/// Body of a file selection for `mime`, or `None` if `mime` is not a file format.
/// Plain-text types get the paths, one per line.
pub fn file_selection_body(mime: &str, paths: &[PathBuf]) -> Option<Vec<u8>> {
    let uris = || paths.iter().map(|p| file_uri(p)).collect::<Vec<_>>();
    match mime {
        URI_LIST_MIME => {
            let mut body = uris().join("\r\n");
            body.push_str("\r\n");
            Some(body.into_bytes())
        }
        GNOME_COPIED_FILES_MIME => Some(format!("copy\n{}", uris().join("\n")).into_bytes()),
        KDE_CUT_SELECTION_MIME => Some(b"0".to_vec()),
        m if crate::SUPPORTED_TEXT_MIMES.contains(&m) => {
            let mut body = Vec::new();
            for (i, p) in paths.iter().enumerate() {
                if i > 0 {
                    body.push(b'\n');
                }
                body.extend_from_slice(p.as_os_str().as_bytes());
            }
            Some(body)
        }
        _ => None,
    }
}

/// All MIME types offered for a file selection (file formats, then plain text).
pub fn file_selection_mimes() -> Vec<String> {
    FILE_SELECTION_MIMES
        .iter()
        .chain(crate::SUPPORTED_TEXT_MIMES)
        .map(|s| (*s).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_validated() {
        assert!(is_valid_name("a b.txt"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("."));
        assert!(!is_valid_name(".."));
        assert!(!is_valid_name("a/b"));
        assert!(!is_valid_name("a\0b"));
        assert!(!is_valid_name(&"x".repeat(256)));
    }

    #[test]
    fn uri_round_trip() {
        let p = PathBuf::from("/tmp/a dir/ü%#?.txt");
        let uri = file_uri(&p);
        assert_eq!(uri, "file:///tmp/a%20dir/%C3%BC%25%23%3F.txt");
        assert_eq!(parse_file_uri_list(&format!("{uri}\r\n")), Some(vec![p]));
    }

    #[test]
    fn uri_list_parsing() {
        let body = "# comment\r\nfile:///a/b\r\nfile://localhost/c%20d\r\n\0";
        assert_eq!(
            parse_file_uri_list(body),
            Some(vec![PathBuf::from("/a/b"), PathBuf::from("/c d")])
        );
        assert_eq!(parse_file_uri_list("https://example.com/\n"), None);
        assert_eq!(parse_file_uri_list("file:///a\nhttps://x/\n"), None);
        assert_eq!(parse_file_uri_list("file://host/a\n"), None);
        assert_eq!(parse_file_uri_list("file:///a%2\n"), None);
        assert_eq!(parse_file_uri_list("file:///a%00b\n"), None);
        assert_eq!(parse_file_uri_list(""), None);
    }

    #[test]
    fn selection_bodies() {
        let paths = vec![PathBuf::from("/x/a b"), PathBuf::from("/x/c")];
        assert_eq!(
            file_selection_body(URI_LIST_MIME, &paths).unwrap(),
            b"file:///x/a%20b\r\nfile:///x/c\r\n"
        );
        assert_eq!(
            file_selection_body(GNOME_COPIED_FILES_MIME, &paths).unwrap(),
            b"copy\nfile:///x/a%20b\nfile:///x/c"
        );
        assert_eq!(
            file_selection_body(KDE_CUT_SELECTION_MIME, &paths).unwrap(),
            b"0"
        );
        assert_eq!(
            file_selection_body("text/plain;charset=utf-8", &paths).unwrap(),
            b"/x/a b\n/x/c"
        );
        assert_eq!(file_selection_body("image/png", &paths), None);
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oriel-xfer-{tag}-{}", random_u64()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn owner_resolves_only_offered_paths() {
        let dir = scratch_dir("resolve");
        let root = dir.join("root");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/f"), b"hi").unwrap();
        std::fs::write(dir.join("secret"), b"no").unwrap();
        std::os::unix::fs::symlink(dir.join("secret"), root.join("link")).unwrap();

        let owner = XferOwner::new();
        let (id, roots) = owner
            .offer(OfferSlot::Clipboard, &[root.clone(), root.clone()])
            .unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].name, "root");
        assert_eq!(roots[1].name, "root (2)");
        assert_eq!(roots[0].kind, FileKind::Dir);

        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(owner.resolve(id, &s(&["root", "sub", "f"])).is_ok());
        assert_eq!(owner.resolve(id, &s(&["root", ".."])), Err(libc::EINVAL));
        assert_eq!(owner.resolve(id, &s(&["root", "link"])), Err(libc::ENOENT));
        assert_eq!(owner.resolve(id, &s(&["nope"])), Err(libc::ENOENT));
        assert_eq!(owner.resolve(id ^ 1, &s(&["root"])), Err(libc::ENOENT));

        let listing = list_dir(&root).unwrap();
        assert_eq!(listing.len(), 1, "symlink must be skipped: {listing:?}");
        assert_eq!(listing[0].name, "sub");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn serve_read_streams_from_offset() {
        let dir = scratch_dir("read");
        std::fs::write(dir.join("f"), b"0123456789").unwrap();
        let owner = Arc::new(XferOwner::new());
        let (id, _) = owner.offer(OfferSlot::Clipboard, &[dir.join("f")]).unwrap();
        let mut out = Vec::new();
        serve_read(&owner, 0, id, &["f".to_string()], 4, &mut out)
            .await
            .unwrap();
        let mut cursor = std::io::Cursor::new(out);
        let reply: FileReadReply = read_frame(&mut cursor).await.unwrap();
        assert_eq!(reply, FileReadReply::Ok);
        let mut rest = Vec::new();
        cursor.read_to_end(&mut rest).await.unwrap();
        assert_eq!(rest, b"456789");

        let mut out = Vec::new();
        serve_read(&owner, 0, id, &["g".to_string()], 0, &mut out)
            .await
            .unwrap();
        let reply: FileReadReply = read_frame(&mut std::io::Cursor::new(out)).await.unwrap();
        assert_eq!(reply, FileReadReply::Err(libc::ENOENT));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
