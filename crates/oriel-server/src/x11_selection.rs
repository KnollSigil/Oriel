//! Read one format of the X11 `CLIPBOARD` selection over a connection of our own to the
//! embedded XWayland.
//!
//! Smithay 0.7's `X11Wm::send_selection` can only convert `UTF8_STRING` and `TEXT` (for
//! any other MIME type it looks the atom up in a `TARGETS` property it never stores), and
//! for large data sent incrementally (INCR) it passes the INCR size header on as data. All
//! clipboard reads from X11 apps go through here instead.
//!
//! A second connection reads window properties smithay's window manager drops (see
//! [`X11Display::is_modal`]).

use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ConnectionExt, CreateWindowAux, EventMask, GetPropertyReply, Property, Window,
    WindowClass,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::{DefaultStream, PollMode, RustConnection, Stream};
use x11rb::utils::RawFdContainer;
use x11rb::CURRENT_TIME;

use oriel_proto::MAX_SELECTION_BYTES;

use crate::compositor::SELECTION_IO_TIMEOUT;

const COOKIE_AUTH: &[u8] = b"MIT-MAGIC-COOKIE-1";

/// How long a streamed write may wait for the reader (the client's paste over the network).
const STREAM_WRITE_IDLE: std::time::Duration = std::time::Duration::from_secs(15);

/// Largest selection read with incremental (INCR) transfers, which X11 apps use for
/// anything bigger than a request (images, mostly).
const MAX_INCR_BYTES: usize = 256 * 1024 * 1024;

/// How to reach the embedded XWayland. Clones share one connection.
#[derive(Clone, Debug)]
pub struct X11Display {
    number: u32,
    xauthority: Option<PathBuf>,
    reader: Arc<Mutex<Option<Reader>>>,
    /// Separate from `reader` so a long clipboard transfer never holds up window mapping.
    props: Arc<Mutex<PropsSlot>>,
}

/// How long a window property read may wait for XWayland, in all.
const PROP_READ_TIMEOUT: Duration = Duration::from_millis(250);
/// How long to go without property reads after one failed, before connecting again.
const PROP_RETRY_DELAY: Duration = Duration::from_secs(5);

/// The property connection: made on a thread of its own, so the compositor thread never
/// waits for a connection to XWayland.
#[derive(Debug)]
enum PropsSlot {
    /// Not connected; connect again once `retry_at` has passed (at once when `None`).
    Idle {
        retry_at: Option<Instant>,
    },
    Connecting,
    Ready(Box<PropReader>),
}

impl Default for PropsSlot {
    fn default() -> Self {
        PropsSlot::Idle { retry_at: None }
    }
}

/// A connection for reading window properties, with the atoms it needs.
struct PropReader {
    conn: RustConnection<DeadlineStream>,
    net_wm_state: Atom,
    modal: Atom,
}

impl std::fmt::Debug for PropReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropReader").finish_non_exhaustive()
    }
}

/// A connection and the requestor window conversions are delivered to. Both live as long
/// as the XWayland: smithay's window manager inspects every new top-level window, and a
/// window created per request would be gone before it got there.
struct Reader {
    conn: RustConnection,
    window: Window,
    clipboard: Atom,
    property: Atom,
    incr: Atom,
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("window", &self.window)
            .finish()
    }
}

impl X11Display {
    pub fn new(number: u32, xauthority: Option<PathBuf>) -> Self {
        Self {
            number,
            xauthority,
            reader: Arc::default(),
            props: Arc::default(),
        }
    }

    /// Start connecting for [`Self::is_modal`] in the background, so the connection is
    /// ready by the time windows map.
    pub fn prepare_props(&self) {
        let mut slot = self.props.lock().unwrap();
        self.maybe_connect_props(&mut slot);
    }

    /// Connect on a new thread if not connected and not waiting out a failure.
    fn maybe_connect_props(&self, slot: &mut PropsSlot) {
        let PropsSlot::Idle { retry_at } = *slot else {
            return;
        };
        if retry_at.is_some_and(|at| Instant::now() < at) {
            return;
        }
        *slot = PropsSlot::Connecting;
        let display = self.clone();
        std::thread::spawn(move || {
            let result = display.connect_props();
            let mut slot = display.props.lock().unwrap();
            *slot = match result {
                Ok(props) => PropsSlot::Ready(Box::new(props)),
                Err(err) => {
                    tracing::debug!(error = %err, "cannot connect to XWayland for window state");
                    PropsSlot::Idle {
                        retry_at: Some(Instant::now() + PROP_RETRY_DELAY),
                    }
                }
            };
        });
    }

    /// Whether X11 window `window` asks to be modal: `_NET_WM_STATE_MODAL` in its
    /// `_NET_WM_STATE`. Smithay never reads the state an app sets before mapping (GTK and
    /// Qt set the modal state that way) and later replaces the property with its own, so
    /// call this from `map_window_request`, before the window is activated.
    ///
    /// It waits for XWayland's reply on the calling thread (the compositor's), at most
    /// [`PROP_READ_TIMEOUT`] in all: XWayland is a Wayland client of that thread and could
    /// be waiting on it. It never connects on that thread (see [`PropsSlot`]): without a
    /// connection, or when the read fails, the window counts as not modal. After a failure
    /// reads stop for [`PROP_RETRY_DELAY`], so a stuck XWayland stalls one map, not each.
    pub fn is_modal(&self, window: u32) -> bool {
        let mut slot = self.props.lock().unwrap();
        let PropsSlot::Ready(props) = &*slot else {
            self.maybe_connect_props(&mut slot);
            return false;
        };
        match props.is_modal(window) {
            Ok(modal) => modal,
            Err(err) => {
                tracing::debug!(error = %err, window, "cannot read X11 window state");
                *slot = PropsSlot::Idle {
                    retry_at: Some(Instant::now() + PROP_RETRY_DELAY),
                };
                false
            }
        }
    }

    fn connect_props(&self) -> anyhow::Result<PropReader> {
        let conn = self.open(|inner| DeadlineStream {
            inner,
            deadline: Mutex::new(None),
        })?;
        let net_wm_state = conn.intern_atom(false, b"_NET_WM_STATE")?;
        let modal = conn.intern_atom(false, b"_NET_WM_STATE_MODAL")?;
        let (net_wm_state, modal) = (net_wm_state.reply()?.atom, modal.reply()?.atom);
        Ok(PropReader {
            conn,
            net_wm_state,
            modal,
        })
    }

    /// Stream the `CLIPBOARD` selection converted to `mime` into `fd` as it arrives, at most
    /// `max` bytes (the reader sees more data than it wants and stops). Writes wait for a
    /// slow reader; a reader that stops reading ends the transfer.
    pub fn read_clipboard_to(&self, mime: &str, fd: &std::os::fd::OwnedFd, max: usize) {
        let raw = std::os::fd::AsRawFd::as_raw_fd(fd);
        self.read_clipboard_with(mime, max.min(MAX_INCR_BYTES), &mut |chunk| {
            // As patient as the stream draining the pipe to the client.
            crate::compositor::write_fd_idle(raw, chunk, STREAM_WRITE_IDLE)
        });
    }

    /// Convert the selection to `mime`, passing the data to `sink` chunk by chunk (it
    /// returns `false` to stop) up to `max` bytes. `true` when all of it was passed.
    fn read_clipboard_with(
        &self,
        mime: &str,
        max: usize,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> bool {
        let mut slot = self.reader.lock().unwrap();
        if slot.is_none() {
            match self.connect() {
                Ok(reader) => *slot = Some(reader),
                Err(err) => {
                    tracing::debug!(error = %err, "cannot connect to XWayland for the clipboard");
                    return false;
                }
            }
        }
        let Some(reader) = slot.as_ref() else {
            return false;
        };
        match reader.read(mime, max, sink) {
            Ok(complete) => complete,
            Err(err) => {
                // An abandoned incremental transfer leaves the owner writing chunks to our
                // window: start over with a new connection and window next time.
                tracing::debug!(error = %err, mime, "X11 clipboard conversion ended early");
                *slot = None;
                false
            }
        }
    }

    /// A new authenticated connection to the XWayland.
    fn open<S: Stream>(
        &self,
        wrap: impl FnOnce(DefaultStream) -> S,
    ) -> anyhow::Result<RustConnection<S>> {
        let socket = UnixStream::connect(format!("/tmp/.X11-unix/X{}", self.number))?;
        let (stream, _) = DefaultStream::from_unix_stream(socket)?;
        let stream = wrap(stream);
        let cookie = self
            .xauthority
            .as_deref()
            .and_then(|path| read_cookie(path, self.number));
        let (auth_name, auth_data) = match cookie {
            Some(cookie) => (COOKIE_AUTH.to_vec(), cookie),
            None => (Vec::new(), Vec::new()),
        };
        Ok(RustConnection::connect_to_stream_with_auth_info(
            stream, 0, auth_name, auth_data,
        )?)
    }

    fn connect(&self) -> anyhow::Result<Reader> {
        let conn = self.open(|s| s)?;
        let root = conn
            .setup()
            .roots
            .first()
            .ok_or_else(|| anyhow::anyhow!("no screen"))?
            .root;
        let clipboard = conn.intern_atom(false, b"CLIPBOARD")?;
        let property = conn.intern_atom(false, b"_ORIEL_SELECTION")?;
        let incr = conn.intern_atom(false, b"INCR")?;
        let (clipboard, property, incr) = (
            clipboard.reply()?.atom,
            property.reply()?.atom,
            incr.reply()?.atom,
        );
        let window = conn.generate_id()?;
        conn.create_window(
            0,
            window,
            root,
            0,
            0,
            1,
            1,
            0,
            WindowClass::INPUT_ONLY,
            0,
            // Never mapped; override-redirect keeps it out of window management anyway.
            // Property changes drive incremental (INCR) transfers.
            &CreateWindowAux::new()
                .override_redirect(1)
                .event_mask(EventMask::PROPERTY_CHANGE),
        )?;
        conn.flush()?;
        Ok(Reader {
            conn,
            window,
            clipboard,
            property,
            incr,
        })
    }
}

/// The X11 target for MIME type `mime`: the plain-text types map to the X11 text atoms
/// (as smithay's window manager announces them), everything else is its own atom name.
fn target_name(mime: &str) -> &str {
    match mime {
        "text/plain;charset=utf-8" => "UTF8_STRING",
        "text/plain" => "TEXT",
        other => other,
    }
}

/// A stream whose waits give up at `deadline` (or, without one, after
/// [`SELECTION_IO_TIMEOUT`] each), failing the request with `TimedOut` instead of blocking
/// for good.
#[derive(Debug)]
struct DeadlineStream {
    inner: DefaultStream,
    deadline: Mutex<Option<Instant>>,
}

impl Stream for DeadlineStream {
    fn poll(&self, mode: PollMode) -> std::io::Result<()> {
        let mut events = 0;
        if mode.readable() {
            events |= libc::POLLIN;
        }
        if mode.writable() {
            events |= libc::POLLOUT;
        }
        let deadline = self
            .deadline
            .lock()
            .unwrap()
            .unwrap_or_else(|| Instant::now() + SELECTION_IO_TIMEOUT);
        loop {
            let rem = deadline.saturating_duration_since(Instant::now());
            if rem.is_zero() {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
            let mut pfd = libc::pollfd {
                fd: self.inner.as_raw_fd(),
                events,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis().max(1) as i32) };
            if rc > 0 {
                return Ok(());
            }
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::Interrupted {
                    return Err(err);
                }
            }
        }
    }

    fn read(&self, buf: &mut [u8], fd_storage: &mut Vec<RawFdContainer>) -> std::io::Result<usize> {
        self.inner.read(buf, fd_storage)
    }

    fn write(&self, buf: &[u8], fds: &mut Vec<RawFdContainer>) -> std::io::Result<usize> {
        self.inner.write(buf, fds)
    }

    fn write_vectored(
        &self,
        bufs: &[std::io::IoSlice<'_>],
        fds: &mut Vec<RawFdContainer>,
    ) -> std::io::Result<usize> {
        self.inner.write_vectored(bufs, fds)
    }
}

impl PropReader {
    fn is_modal(&self, window: u32) -> anyhow::Result<bool> {
        *self.conn.stream().deadline.lock().unwrap() = Some(Instant::now() + PROP_READ_TIMEOUT);
        // A window has a few states; 64 atoms is plenty.
        let reply = self
            .conn
            .get_property(false, window, self.net_wm_state, AtomEnum::ATOM, 0, 64)?
            .reply()?;
        Ok(reply
            .value32()
            .is_some_and(|mut atoms| atoms.any(|a| a == self.modal)))
    }
}

impl Reader {
    /// Convert the selection to `mime` into `sink` (see `read_clipboard_with`). `Ok(false)`
    /// when the owner refused or did not answer; an error once an incremental transfer
    /// started and was abandoned (the window must not be reused).
    fn read(
        &self,
        mime: &str,
        max: usize,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> anyhow::Result<bool> {
        let deadline = Instant::now() + SELECTION_IO_TIMEOUT;
        let conn = &self.conn;
        let target = conn
            .intern_atom(false, target_name(mime).as_bytes())?
            .reply()?
            .atom;
        // Answers to earlier requests that timed out are stale.
        while conn.poll_for_event()?.is_some() {}
        // Clear a stale value so only the owner's answer is read.
        conn.delete_property(self.window, self.property)?;
        conn.convert_selection(
            self.window,
            self.clipboard,
            target,
            self.property,
            CURRENT_TIME,
        )?;
        conn.flush()?;

        let answered = loop {
            if let Some(event) = conn.poll_for_event()? {
                if let Event::SelectionNotify(n) = event {
                    if n.requestor == self.window
                        && n.selection == self.clipboard
                        && n.target == target
                    {
                        break n.property == self.property;
                    }
                }
                continue;
            }
            if !wait_readable(conn.stream(), deadline) {
                return Ok(false);
            }
        };
        if !answered {
            return Ok(false);
        }

        let reply = self.take_property()?;
        if reply.type_ != self.incr {
            if reply.bytes_after != 0 || reply.format != 8 {
                return Ok(false);
            }
            let take = reply.value.len().min(max);
            return Ok(sink(&reply.value[..take]) && take == reply.value.len());
        }
        // INCR (ICCCM 2.7.2): deleting the property (done by `take_property`) asks for the
        // first chunk; each chunk arrives as a new property value, which we take (deleting
        // it asks for the next) until an empty one ends the transfer.
        let mut total = 0usize;
        let mut deadline = Instant::now() + SELECTION_IO_TIMEOUT;
        loop {
            let Some(event) = conn.poll_for_event()? else {
                if !wait_readable(conn.stream(), deadline) {
                    anyhow::bail!("incremental transfer stalled");
                }
                continue;
            };
            let Event::PropertyNotify(n) = event else {
                continue;
            };
            if n.window != self.window || n.atom != self.property || n.state != Property::NEW_VALUE
            {
                continue;
            }
            let chunk = self.take_property()?;
            if chunk.format != 8 || chunk.bytes_after != 0 {
                anyhow::bail!("unexpected incremental chunk");
            }
            if chunk.value.is_empty() {
                return Ok(true);
            }
            let take = chunk.value.len().min(max - total);
            if !sink(&chunk.value[..take]) {
                anyhow::bail!("reader stopped");
            }
            total += take;
            if take < chunk.value.len() {
                anyhow::bail!("larger than {max} bytes");
            }
            deadline = Instant::now() + SELECTION_IO_TIMEOUT;
        }
    }

    /// Read and delete our property.
    fn take_property(&self) -> anyhow::Result<GetPropertyReply> {
        let max_words = (MAX_SELECTION_BYTES / 4) as u32;
        Ok(self
            .conn
            .get_property(
                true,
                self.window,
                self.property,
                AtomEnum::ANY,
                0,
                max_words,
            )?
            .reply()?)
    }
}

/// Wait until the connection has data to read; `false` at `deadline`.
fn wait_readable(stream: &DefaultStream, deadline: Instant) -> bool {
    loop {
        let rem = deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            return false;
        }
        let mut pfd = libc::pollfd {
            fd: stream.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis().min(i32::MAX as u128) as i32) };
        if rc > 0 {
            return true;
        }
        if rc < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return false;
        }
    }
}

/// The MIT-MAGIC-COOKIE-1 for display `number` in Xauthority file `path`
/// (as written by `session::write_xauthority_file`).
fn read_cookie(path: &Path, number: u32) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;
    parse_cookie(&data, number.to_string().as_bytes())
}

fn parse_cookie(mut data: &[u8], display: &[u8]) -> Option<Vec<u8>> {
    fn field<'a>(data: &mut &'a [u8]) -> Option<&'a [u8]> {
        let len = u16::from_be_bytes([*data.first()?, *data.get(1)?]) as usize;
        let value = data.get(2..2 + len)?;
        *data = &data[2 + len..];
        Some(value)
    }
    while !data.is_empty() {
        data = data.get(2..)?; // family
        let _address = field(&mut data)?;
        let number = field(&mut data)?;
        let name = field(&mut data)?;
        let cookie = field(&mut data)?;
        if number == display && name == COOKIE_AUTH {
            return Some(cookie.to_vec());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cookie_written_by_session() {
        let dir = tempfile_dir();
        let path = dir.join("Xauthority");
        crate::session::write_xauthority_file(&path, 7).unwrap();
        let cookie = read_cookie(&path, 7).expect("cookie");
        assert_eq!(cookie.len(), 16);
        assert!(read_cookie(&path, 8).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oriel-x11sel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
