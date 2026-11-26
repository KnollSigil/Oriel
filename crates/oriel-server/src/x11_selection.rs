//! Read one format of the X11 `CLIPBOARD` selection over a connection of our own to the
//! embedded XWayland.
//!
//! Smithay 0.7's `X11Wm::send_selection` can only convert `UTF8_STRING` and `TEXT` (for
//! any other MIME type it looks the atom up in a `TARGETS` property it never stores), and
//! for large data sent incrementally (INCR) it passes the INCR size header on as data. All
//! clipboard reads from X11 apps go through here instead.

use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ConnectionExt, CreateWindowAux, EventMask, GetPropertyReply, Property, Window,
    WindowClass,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::{DefaultStream, RustConnection};
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
        }
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

    fn connect(&self) -> anyhow::Result<Reader> {
        let socket = UnixStream::connect(format!("/tmp/.X11-unix/X{}", self.number))?;
        let (stream, _) = DefaultStream::from_unix_stream(socket)?;
        let cookie = self
            .xauthority
            .as_deref()
            .and_then(|path| read_cookie(path, self.number));
        let (auth_name, auth_data) = match cookie {
            Some(cookie) => (COOKIE_AUTH.to_vec(), cookie),
            None => (Vec::new(), Vec::new()),
        };
        let conn =
            RustConnection::connect_to_stream_with_auth_info(stream, 0, auth_name, auth_data)?;
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
