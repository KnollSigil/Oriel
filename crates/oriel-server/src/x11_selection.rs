//! Read one format of the X11 `CLIPBOARD` selection over a connection of our own to the
//! embedded XWayland.
//!
//! Smithay 0.7's `X11Wm::send_selection` can only convert `UTF8_STRING` and `TEXT`: for any
//! other MIME type it looks the atom up in a `TARGETS` property it never stores, and fails.
//! File copies (`text/uri-list`) and other formats from X11 apps go through here instead.

use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ConnectionExt, CreateWindowAux, Window, WindowClass,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::{DefaultStream, RustConnection};
use x11rb::CURRENT_TIME;

use oriel_proto::MAX_SELECTION_BYTES;

use crate::compositor::SELECTION_IO_TIMEOUT;

const COOKIE_AUTH: &[u8] = b"MIT-MAGIC-COOKIE-1";

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

    /// The `CLIPBOARD` selection converted to `mime`, or `None` when the owner refuses,
    /// answers incrementally (INCR, only used for large data), sends more than
    /// `MAX_SELECTION_BYTES`, or does not answer within `SELECTION_IO_TIMEOUT`.
    pub fn read_clipboard(&self, mime: &str) -> Option<Vec<u8>> {
        let mut slot = self.reader.lock().unwrap();
        if slot.is_none() {
            match self.connect() {
                Ok(reader) => *slot = Some(reader),
                Err(err) => {
                    tracing::debug!(error = %err, "cannot connect to XWayland for the clipboard");
                    return None;
                }
            }
        }
        let reader = slot.as_ref()?;
        match reader.read(mime) {
            Ok(data) => data,
            Err(err) => {
                tracing::debug!(error = %err, mime, "X11 clipboard conversion failed");
                // Reconnect next time.
                *slot = None;
                None
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
            &CreateWindowAux::new().override_redirect(1),
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

impl Reader {
    fn read(&self, mime: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let deadline = Instant::now() + SELECTION_IO_TIMEOUT;
        let conn = &self.conn;
        let target = conn.intern_atom(false, mime.as_bytes())?.reply()?.atom;
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
                return Ok(None);
            }
        };
        if !answered {
            return Ok(None);
        }

        let max_words = (MAX_SELECTION_BYTES / 4) as u32;
        let reply = conn
            .get_property(
                true,
                self.window,
                self.property,
                AtomEnum::ANY,
                0,
                max_words,
            )?
            .reply()?;
        if reply.type_ == self.incr || reply.bytes_after != 0 || reply.format != 8 {
            return Ok(None);
        }
        Ok(Some(reply.value))
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
