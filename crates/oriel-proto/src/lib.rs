//! Shared wire protocol and stream framing for Oriel.

pub mod selection;
pub mod smart_tile;
pub mod xfer;
pub mod xfer_fuse;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Current protocol version, sent in the control-stream preamble.
pub const PROTOCOL_VERSION: u32 = 21;

/// Whether `layout` is a plausible GNOME `button-layout` (e.g. `appmenu:minimize,maximize,close`,
/// or `close,minimize` with no `:` for left-side buttons only) for
/// [`ControlMessage::ButtonLayout`]. The allowed characters keep the value safe to embed in
/// GSettings keyfile and GTK `settings.ini` values (no quotes, escapes or newlines).
pub fn is_valid_button_layout(layout: &str) -> bool {
    !layout.is_empty()
        && layout.len() <= 256
        && layout
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | ',' | '-' | '_'))
}

/// Magic bytes that open the control-stream preamble (see [`write_protocol_preamble`]).
pub const PROTOCOL_MAGIC: [u8; 4] = *b"RWLD";

/// QUIC application close code the server sends when the client speaks a different
/// protocol version. The close reason carries a human-readable explanation.
pub const CLOSE_CODE_PROTOCOL_MISMATCH: u32 = 0x5257_0002;

/// QUIC application close code the server sends to a display client whose session was
/// taken over by a newer display client. Clients must not auto-reconnect on it, or two
/// clients would keep taking the session from each other.
pub const CLOSE_CODE_SUPERSEDED: u32 = 0x5257_0001;

/// An inclusive range of UDP ports the server may listen on; a single port is a range
/// of one. Written as `7443` or `7443-7462`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortRange {
    start: u16,
    end: u16,
}

impl PortRange {
    /// `start..=end`; port 0 is not allowed, since it means "any port" to the OS.
    pub fn new(start: u16, end: u16) -> Result<Self> {
        if start == 0 {
            bail!("port 0 is not a valid port (leave the port out to use any free port)");
        }
        if start > end {
            bail!("port range {start}-{end} is empty (the first port must not exceed the last)");
        }
        Ok(Self { start, end })
    }

    pub fn start(&self) -> u16 {
        self.start
    }

    pub fn end(&self) -> u16 {
        self.end
    }

    pub fn contains(&self, port: u16) -> bool {
        (self.start..=self.end).contains(&port)
    }

    pub fn ports(&self) -> std::ops::RangeInclusive<u16> {
        self.start..=self.end
    }
}

impl std::fmt::Display for PortRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

impl std::str::FromStr for PortRange {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let parse_port = |p: &str| -> Result<u16> {
            let p = p.trim();
            p.parse::<u16>()
                .with_context(|| format!("'{p}' is not a port number (1-65535)"))
        };
        match s.split_once('-') {
            Some((start, end)) => Self::new(parse_port(start)?, parse_port(end)?),
            None => {
                let port = parse_port(s)?;
                Self::new(port, port)
            }
        }
    }
}

impl Serialize for PortRange {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// Accepts the string form, or a bare port number (as saved by older clients).
impl<'de> Deserialize<'de> for PortRange {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = PortRange;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a UDP port or a port range like \"7443-7462\"")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<PortRange, E> {
                let port =
                    u16::try_from(v).map_err(|_| E::custom(format!("port {v} is out of range")))?;
                PortRange::new(port, port).map_err(E::custom)
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<PortRange, E> {
                let v =
                    u64::try_from(v).map_err(|_| E::custom(format!("port {v} is out of range")))?;
                self.visit_u64(v)
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<PortRange, E> {
                v.parse().map_err(E::custom)
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// Maximum allowed control frame size (16 MiB) to prevent unbounded allocations.
pub const MAX_CONTROL_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Maximum allowed encoded video frame payload size (64 MiB) to accommodate 4K/5K HiDPI frames.
pub const MAX_VIDEO_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Maximum UTF-8 selection payload size (4 MiB) read from clipboard or primary selection pipes.
pub const MAX_SELECTION_BYTES: usize = 4 * 1024 * 1024;

/// Read a selection pipe until end of file, at most [`MAX_SELECTION_BYTES`], giving up at
/// `timeout` so a misbehaving owner cannot pin a thread. Returns the bytes read and whether
/// they are complete (end of file reached within the limits).
pub fn read_selection_fd(
    fd: std::os::fd::OwnedFd,
    timeout: std::time::Duration,
) -> (Vec<u8>, bool) {
    read_selection_fd_max(fd, timeout, MAX_SELECTION_BYTES)
}

/// [`read_selection_fd`] with a limit of `max` bytes.
pub fn read_selection_fd_max(
    fd: std::os::fd::OwnedFd,
    timeout: std::time::Duration,
    max: usize,
) -> (Vec<u8>, bool) {
    use std::os::fd::AsRawFd;
    let raw = fd.as_raw_fd();
    let deadline = std::time::Instant::now() + timeout;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let rem = deadline.saturating_duration_since(std::time::Instant::now());
        if rem.is_zero() {
            return (buf, false);
        }
        let mut pfd = libc::pollfd {
            fd: raw,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is a valid pollfd for the duration of the call.
        let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis().min(i32::MAX as u128) as i32) };
        if rc < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return (buf, false);
        }
        if rc == 0 {
            return (buf, false);
        }
        // One byte past the limit tells "exactly at the limit" from "over it".
        let max_read = chunk.len().min(max + 1 - buf.len());
        // SAFETY: `chunk` has room for `max_read` bytes.
        let n = unsafe { libc::read(raw, chunk.as_mut_ptr().cast::<libc::c_void>(), max_read) };
        if n == 0 {
            return (buf, true);
        }
        if n < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return (buf, false);
        }
        buf.extend_from_slice(&chunk[..n as usize]);
        if buf.len() > max {
            buf.truncate(max);
            return (buf, false);
        }
    }
}

/// Maximum allowed cursor bitmap width/height in pixels.
pub const MAX_CURSOR_DIM: u32 = 512;

/// Ordered preference list of text MIME types for clipboard and primary selection sync.
pub const SUPPORTED_TEXT_MIMES: &[&str] = &[
    "text/plain;charset=utf-8",
    "text/plain;charset=UTF-8",
    "UTF8_STRING",
    "text/plain",
    "STRING",
    "TEXT",
];

/// Unique identifier for a remote Wayland window (`xdg_toplevel` or `xdg_popup`).
pub type WindowId = u64;

/// Supported video codecs negotiated between server and client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoCodec {
    Lz4Delta,
    H264,
    Hevc,
    Av1,
    SmartTile,
}

/// Identifies the role of a newly opened unidirectional QUIC stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamHeader {
    Video {
        window_id: WindowId,
        codec: VideoCodec,
    },
    AudioPlayback,
    AudioPlaybackApp {
        stream_id: u32,
        app_name: String,
        media_name: String,
        icon_name: String,
        media_role: String,
    },
    AudioCapture,
}

/// High-bit tag applied to `launch_id` values representing server-managed persistent PTYs.
/// Disjoint from `WINDOW_LAUNCH_ID_BASE` (`1 << 62`) used for GUI windows.
pub const PTY_LAUNCH_ID_BASE: u64 = 1u64 << 61;

/// Identifies the role of a newly opened secondary bidirectional QUIC stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BidiStreamHeader {
    /// Remote PTY terminal session (zero Wayland/PipeWire required on server).
    Pty {
        /// If `Some(id)`, reattach to an existing persistent server PTY;
        /// if `None`, spawn a new persistent PTY.
        pty_id: Option<u64>,
        /// Byte offset already received by the client; on reattach the server
        /// replays any buffered output produced after this offset.
        resume_offset: u64,
        cols: u16,
        rows: u16,
        command: Option<String>,
    },
    /// List a directory inside a file offer (see [`ControlMessage::ClipboardOffer`]). The
    /// owner replies with one [`xfer::FileListReply`] frame.
    FileList {
        offer_id: u64,
        /// Root name followed by the components below it.
        path: Vec<String>,
    },
    /// Read a file inside a file offer from `offset`. The owner replies with one
    /// [`xfer::FileReadReply`] frame followed by the raw bytes up to EOF; the reader stops
    /// the stream to cancel.
    FileRead {
        offer_id: u64,
        /// Root name followed by the components below it.
        path: Vec<String>,
        offset: u64,
    },
    /// Read type `mime` of selection `offer_id` (see [`ControlMessage::ClipboardOffer`]).
    /// The owner replies with one [`selection::SelectionDataReply`] frame followed by the
    /// raw data up to EOF; the reader stops the stream to cancel.
    SelectionData { offer_id: u64, mime: String },
}

/// Framed messages exchanged over a bidirectional PTY QUIC stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtyMessage {
    /// Sent by the server immediately upon creating or reattaching to a PTY,
    /// carrying the current byte offset so future reconnects stay exact.
    Attached {
        pty_id: u64,
        stream_offset: u64,
    },
    /// Historical buffered output replayed on reattach (already accounted for in `stream_offset`).
    Replay(Vec<u8>),
    /// Sent by the server when a reattach is rejected because another terminal is already attached.
    Busy,
    Data(Vec<u8>),
    Resize {
        cols: u16,
        rows: u16,
    },
    /// Sent by the client on intentional close (e.g., terminal tab closed via SIGHUP).
    Close,
    Exit(i32),
}

/// Metadata + optional icon bytes for a discovered remote `.desktop` application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteAppEntry {
    pub id: String,
    pub name: String,
    pub generic_name: Option<String>,
    pub keywords: Vec<String>,
    pub exec: String,
    pub categories: Vec<String>,
    /// `Terminal=true`: the program needs a terminal, so the client runs `exec` in a
    /// remote PTY shown in a local terminal emulator instead of as a GUI launch.
    pub terminal: bool,
    pub icon_mime: Option<String>,
    pub icon_data: Vec<u8>,
    /// `icon_data` is a symbolic (monochrome) icon: clients recolour it to their
    /// foreground colour instead of drawing its stock grey.
    pub icon_symbolic: bool,
}

/// Lifecycle state of a launched remote application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchState {
    Launching,
    Running,
    Exited(i32),
}

/// What an authenticated connection is used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientRole {
    /// Full desktop session: windows, input, clipboard and audio. The server serves one
    /// display client at a time; a newly authenticated one takes over the session and the
    /// previous one is closed with [`CLOSE_CODE_SUPERSEDED`].
    Display,
    /// Only tunnels remote PTY streams. Never takes over, or is taken over by, a display
    /// client.
    PtyOnly,
}

/// Bidirectional control-stream messages (QUIC stream 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client -> Server authentication and capability advertisement.
    AuthRequest {
        token: String,
        role: ClientRole,
        supported_codecs: Vec<VideoCodec>,
        enable_audio_bridge: bool,
        /// Initial effective display scale factor (>= 1.0) so the server can configure
        /// its virtual output and XWayland DPI before the first app launches.
        initial_scale: f64,
    },
    /// Server -> Client authentication result.
    AuthResponse {
        accepted: bool,
        reason: Option<String>,
        selected_codec: VideoCodec,
    },

    /// Server -> Client: a new `xdg_toplevel` window appeared.
    WindowCreated {
        window_id: WindowId,
        title: String,
        app_id: String,
        width: u32,
        height: u32,
    },
    /// Server -> Client: a new `xdg_popup` appeared relative to a parent window.
    PopupCreated {
        window_id: WindowId,
        parent_id: WindowId,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    /// Server -> Client: an `xdg_popup` was repositioned relative to its parent.
    PopupRepositioned {
        window_id: WindowId,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    /// Server -> Client: encoded BGRA (`Argb8888`) pixel buffer for an `xdg_popup`,
    /// using the session's negotiated `VideoCodec` (`SmartTile` by default).
    PopupFrame {
        window_id: WindowId,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        /// The popup's window rectangle inside the frame; see [`VideoFrameHeader::geometry`].
        geometry: FrameRect,
        /// See [`VideoFrameHeader::input_bounds`].
        input_bounds: FrameRect,
        is_keyframe: bool,
        bgra: Vec<u8>,
    },
    /// Server -> Client: remote app requested interactive window move (`xdg_toplevel.move`).
    StartMove { window_id: WindowId },
    /// Server -> Client: remote app requested interactive window resize (`xdg_toplevel.resize`).
    StartResize { window_id: WindowId, edge: u32 },
    /// Server -> Client: remote app requested maximize or unmaximize (`xdg_toplevel.set_maximized` / `unset_maximized`).
    SetMaximized {
        window_id: WindowId,
        maximized: bool,
    },
    /// Server -> Client: remote app requested minimize (`xdg_toplevel.set_minimized`).
    SetMinimized { window_id: WindowId },
    /// Server -> Client: remote app requested fullscreen or unfullscreen (`xdg_toplevel.set_fullscreen` / `unset_fullscreen`).
    SetFullscreen {
        window_id: WindowId,
        fullscreen: bool,
    },
    /// Server -> Client: remote app requested the compositor window menu (`xdg_toplevel.show_window_menu`).
    ShowWindowMenu { window_id: WindowId, x: i32, y: i32 },
    /// Server -> Client: remote app requested server-side or client-side decorations (`zxdg_toplevel_decoration_v1`).
    SetDecorationMode {
        window_id: WindowId,
        server_side: bool,
    },
    /// Server -> Client or Client -> Server: window size/state update.
    WindowConfigured {
        window_id: WindowId,
        width: u32,
        height: u32,
        scale_factor: f64,
        maximized: bool,
        fullscreen: bool,
        activated: bool,
        tiled: bool,
    },
    /// Server -> Client or Client -> Server: window closed.
    WindowClosed { window_id: WindowId },

    /// Client -> Server: pointer moved in surface-local logical coordinates.
    PointerMotion { window_id: WindowId, x: f64, y: f64 },
    /// Client -> Server: pointer button pressed or released (Linux evdev button code).
    PointerButton {
        window_id: WindowId,
        button: u32,
        pressed: bool,
    },
    /// Client -> Server: scroll wheel / touchpad axis event.
    PointerAxis {
        window_id: WindowId,
        dx: f64,
        dy: f64,
    },
    /// Client -> Server: keyboard key event (evdev keycode + UTF-8 optional text).
    KeyEvent {
        window_id: WindowId,
        keycode: u32,
        pressed: bool,
    },

    /// Bidirectional: the sender's clipboard changed (see [`selection`]). `mimes` are the
    /// types the receiver fetches on paste with [`BidiStreamHeader::SelectionData`];
    /// `text` is the text when small; `files` are files presented as paths, read on demand
    /// with [`BidiStreamHeader::FileList`] / [`BidiStreamHeader::FileRead`].
    ClipboardOffer {
        offer_id: u64,
        mimes: Vec<String>,
        text: Option<String>,
        files: Option<xfer::FileOffer>,
    },
    /// Client -> Server: a local drag entered remote window `window_id` at surface-local
    /// `x`/`y`. The server drags it over the window's apps until
    /// [`ControlMessage::DragLeave`] or [`ControlMessage::DragDrop`]. Its data is offered
    /// like [`ControlMessage::ClipboardOffer`]: `mimes` fetched from offer `offer_id` with
    /// [`BidiStreamHeader::SelectionData`] (until [`ControlMessage::DragFinished`]),
    /// `files` presented as paths.
    DragEnter {
        window_id: WindowId,
        x: f64,
        y: f64,
        offer_id: u64,
        mimes: Vec<String>,
        files: Option<xfer::FileOffer>,
    },
    /// Client -> Server: the drag from [`ControlMessage::DragEnter`] moved.
    DragMotion { window_id: WindowId, x: f64, y: f64 },
    /// Client -> Server: the drag left the window it entered (cancelled there).
    DragLeave,
    /// Client -> Server: the drag was dropped where it last moved to.
    DragDrop,
    /// Server -> Client: the app the drag from [`ControlMessage::DragEnter`] was dropped on
    /// is done with its data (or the drop failed): the client can end the local drag.
    DragFinished,
    /// Server -> Client: an app in remote window `window_id` started a drag that can leave
    /// the session (files, text, images). The client continues it on the local desktop;
    /// its data is offered like [`ControlMessage::ClipboardOffer`] (`mimes` fetched from
    /// offer `offer_id`). Over remote windows the client keeps sending ordinary pointer
    /// motion and the button release, which the server's drag follows.
    RemoteDragStart {
        window_id: WindowId,
        offer_id: u64,
        mimes: Vec<String>,
        files: Option<xfer::FileOffer>,
    },
    /// Client -> Server: the drag from [`ControlMessage::RemoteDragStart`] left the remote
    /// windows, so it is over no session app.
    RemoteDragLeave,
    /// Client -> Server: the drag from [`ControlMessage::RemoteDragStart`] ended outside the
    /// remote windows (dropped on a local app or cancelled); end it in the session too.
    RemoteDragEnd,
    /// Bidirectional: UTF-8 primary selection (middle-click paste) text synchronization.
    PrimarySelectionText { text: String },
    /// Server -> Client: enable/disable host microphone capture based on remote app demand.
    SetMicActive { active: bool },
    /// Server -> Client: cursor visibility or named cursor shape update.
    CursorUpdate {
        window_id: WindowId,
        visible: bool,
        shape_name: String,
    },
    /// Server -> Client: surface-based cursor bitmap (`Argb8888` BGRA bytes + logical hotspot).
    CursorBitmap {
        window_id: WindowId,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        bgra: Vec<u8>,
    },

    /// Client -> Server: request list of discovered `.desktop` applications.
    ListApps,
    /// Server -> Client: list of discovered `.desktop` applications with optional icons.
    AppList { apps: Vec<RemoteAppEntry> },
    /// Client -> Server: launch a remote GUI application inside the headless Wayland session.
    LaunchApp {
        launch_id: u64,
        name: Option<String>,
        exec: String,
    },
    /// Server -> Client: lifecycle state update for a launched remote application.
    AppState {
        launch_id: u64,
        name: String,
        state: LaunchState,
    },
    /// Client -> Server: terminate a launched remote application (`SIGTERM`, or `SIGKILL` if `force`).
    TerminateApp { launch_id: u64, force: bool },
    /// Client -> Server: XDG `org.freedesktop.appearance` `color-scheme` preference
    /// (`0` = no preference, `1` = prefer dark, `2` = prefer light).
    ColorScheme { scheme: u32 },
    /// Client -> Server: local keyboard focus entered a remote window with the
    /// given evdev keycodes already depressed (mirrors `wl_keyboard::enter`).
    KeyboardEnter {
        window_id: WindowId,
        pressed_keys: Vec<u32>,
    },
    /// Client -> Server: local keyboard focus left a remote window
    /// (mirrors `wl_keyboard::leave`).
    KeyboardLeave { window_id: WindowId },
    /// Client -> Server: acknowledges decoding of a video frame (`pts_ns`) for
    /// end-to-end in-flight frame pacing and low-latency frame dropping.
    FrameAck { window_id: WindowId, pts_ns: u64 },
    /// Client -> Server: updates the session's active display scale factor (>= 1.0)
    /// used for virtual `wl_output` fallback and global XWayland HiDPI scaling.
    DisplayScale { scale_factor: f64 },
    /// Client -> Server: GNOME `org.gnome.desktop.wm.preferences` `button-layout`
    /// (e.g. `appmenu:minimize,maximize,close`) so remote apps draw the same
    /// titlebar buttons as local ones. `None` means the client's desktop has no
    /// preference, and the server uses its default layout.
    ButtonLayout { layout: Option<String> },
    /// Client -> Server: XDG `org.freedesktop.appearance` `accent-color` as sRGB components
    /// in `[0, 1]`. `None` means the client's desktop has no accent preference.
    AccentColor { rgb: Option<[f64; 3]> },
    /// Client -> Server: XDG `org.freedesktop.appearance` `contrast`
    /// (`0` = no preference, `1` = higher contrast).
    Contrast { contrast: u32 },
    /// Client -> Server: the local pointer left a remote window (mirrors `wl_pointer::leave`).
    PointerLeave { window_id: WindowId },
    /// Client -> Server: whether the server downscales frames an app renders above its
    /// window's display scale (an integer-scale app on a fractional-scale display) to that
    /// scale before streaming them. Applies to the session from then on; enabled by default.
    ServerDownscale { enabled: bool },
}

/// Whether `rgb` is a valid XDG `accent-color`: three finite components in `[0, 1]`.
pub fn is_valid_accent_color(rgb: &[f64; 3]) -> bool {
    rgb.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c))
}

/// A rectangle inside a frame, in logical pixels relative to the frame's top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl FrameRect {
    /// A zero-size rectangle: for [`VideoFrameHeader::input_bounds`], a surface that takes
    /// no pointer input (a click-through surface).
    pub const NONE: Self = Self {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    /// Whether the rectangle has no area.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// A rectangle covering a whole `width` x `height` frame.
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// Header preceding each encoded video access unit on a per-window video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoFrameHeader {
    pub pts_ns: u64,
    pub width: u32,
    pub height: u32,
    pub logical_width: u32,
    pub logical_height: u32,
    /// The window's visible rectangle (`xdg_surface.set_window_geometry`). Apps that
    /// draw their own titlebar paint their drop shadow around it, outside this rectangle.
    pub geometry: FrameRect,
    /// Bounding box of the app's input region: the window plus whatever resize margin
    /// the app keeps in its shadow. Pointer input outside it belongs to windows below;
    /// empty ([`FrameRect::NONE`]) when the surface takes no input at all.
    pub input_bounds: FrameRect,
    pub is_keyframe: bool,
    pub payload_len: u32,
}

/// Header preceding each Opus audio packet on an audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPacketHeader {
    pub pts_ns: u64,
    pub payload_len: u32,
}

/// The protocol a peer announced at the start of the control stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerProtocol {
    /// The peer sent the preamble with this protocol version.
    Version(u32),
    /// The peer did not send the preamble: it predates protocol version 9, which
    /// introduced it, and starts directly with a length-prefixed message.
    Legacy,
}

/// Write the fixed-size preamble that opens the control stream: [`PROTOCOL_MAGIC`]
/// followed by [`PROTOCOL_VERSION`] as a big-endian `u32`. Its layout never depends on
/// the message encoding, so peers can always detect a version mismatch.
pub async fn write_protocol_preamble<W: AsyncWrite + Unpin>(writer: &mut W) -> Result<()> {
    writer.write_all(&PROTOCOL_MAGIC).await?;
    writer.write_u32(PROTOCOL_VERSION).await?;
    Ok(())
}

/// Read the control-stream preamble written by [`write_protocol_preamble`].
pub async fn read_protocol_preamble<R: AsyncRead + Unpin>(reader: &mut R) -> Result<PeerProtocol> {
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .await
        .context("failed to read protocol preamble")?;
    if magic != PROTOCOL_MAGIC {
        return Ok(PeerProtocol::Legacy);
    }
    let version = reader
        .read_u32()
        .await
        .context("failed to read protocol version")?;
    Ok(PeerProtocol::Version(version))
}

/// Serialize and write a length-prefixed `ControlMessage` to an async writer.
pub async fn write_control_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &ControlMessage,
) -> Result<()> {
    let encoded = postcard::to_allocvec(msg).context("failed to serialize ControlMessage")?;
    if encoded.len() > MAX_CONTROL_FRAME_BYTES {
        bail!("control message exceeds max frame size: {}", encoded.len());
    }
    let len = u32::try_from(encoded.len()).context("frame length overflow")?;
    writer.write_u32(len).await?;
    writer.write_all(&encoded).await?;
    Ok(())
}

/// Read and deserialize a length-prefixed `ControlMessage` from an async reader.
pub async fn read_control_message<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<ControlMessage>> {
    let len = match reader.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    if len > MAX_CONTROL_FRAME_BYTES {
        bail!("incoming control message exceeds max frame size: {len}");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    let msg = postcard::from_bytes(&buf).context("failed to deserialize ControlMessage")?;
    Ok(Some(msg))
}

/// Write a `StreamHeader` at the start of a unidirectional QUIC stream.
pub async fn write_stream_header<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &StreamHeader,
) -> Result<()> {
    let encoded = postcard::to_allocvec(header).context("failed to serialize StreamHeader")?;
    let len = u32::try_from(encoded.len()).context("stream header length overflow")?;
    writer.write_u32(len).await?;
    writer.write_all(&encoded).await?;
    Ok(())
}

/// Read a `StreamHeader` from the start of a unidirectional QUIC stream.
pub async fn read_stream_header<R: AsyncRead + Unpin>(reader: &mut R) -> Result<StreamHeader> {
    let len = reader.read_u32().await? as usize;
    if len > 4096 {
        bail!("stream header too large: {len}");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    postcard::from_bytes(&buf).context("failed to deserialize StreamHeader")
}

/// Write a `BidiStreamHeader` at the start of a secondary bidirectional QUIC stream.
pub async fn write_bidi_stream_header<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &BidiStreamHeader,
) -> Result<()> {
    let encoded = postcard::to_allocvec(header).context("failed to serialize BidiStreamHeader")?;
    let len = u32::try_from(encoded.len()).context("bidi stream header length overflow")?;
    writer.write_u32(len).await?;
    writer.write_all(&encoded).await?;
    Ok(())
}

/// Read a `BidiStreamHeader` from the start of a secondary bidirectional QUIC stream.
pub async fn read_bidi_stream_header<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<BidiStreamHeader> {
    let len = reader.read_u32().await? as usize;
    if len > 16 * 1024 {
        bail!("bidi stream header too large: {len}");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    postcard::from_bytes(&buf).context("failed to deserialize BidiStreamHeader")
}

/// Write a length-prefixed `PtyMessage` to a PTY QUIC stream.
pub async fn write_pty_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &PtyMessage,
) -> Result<()> {
    let encoded = postcard::to_allocvec(msg).context("failed to serialize PtyMessage")?;
    if encoded.len() > MAX_CONTROL_FRAME_BYTES {
        bail!("pty message exceeds max frame size: {}", encoded.len());
    }
    let len = u32::try_from(encoded.len()).context("pty frame length overflow")?;
    writer.write_u32(len).await?;
    writer.write_all(&encoded).await?;
    Ok(())
}

/// Read a length-prefixed `PtyMessage` from a PTY QUIC stream.
pub async fn read_pty_message<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Option<PtyMessage>> {
    let len = match reader.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    if len > MAX_CONTROL_FRAME_BYTES {
        bail!("incoming pty message exceeds max frame size: {len}");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    let msg = postcard::from_bytes(&buf).context("failed to deserialize PtyMessage")?;
    Ok(Some(msg))
}

/// Write a `VideoFrameHeader` followed by its encoded payload bytes.
pub async fn write_video_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &VideoFrameHeader,
    payload: &[u8],
) -> Result<()> {
    let encoded_hdr =
        postcard::to_allocvec(header).context("failed to serialize VideoFrameHeader")?;
    let hdr_len = u32::try_from(encoded_hdr.len()).context("video header length overflow")?;
    writer.write_u32(hdr_len).await?;
    writer.write_all(&encoded_hdr).await?;
    writer.write_all(payload).await?;
    Ok(())
}

/// Read a `VideoFrameHeader` and its encoded payload bytes (`None` on clean EOF).
pub async fn read_video_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<(VideoFrameHeader, Vec<u8>)>> {
    let hdr_len = match reader.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    if hdr_len > 4096 {
        bail!("video frame header too large: {hdr_len}");
    }
    let mut hdr_buf = vec![0u8; hdr_len];
    reader.read_exact(&mut hdr_buf).await?;
    let header: VideoFrameHeader =
        postcard::from_bytes(&hdr_buf).context("failed to deserialize VideoFrameHeader")?;
    if header.payload_len as usize > MAX_VIDEO_FRAME_BYTES {
        bail!("video payload too large: {}", header.payload_len);
    }
    let mut payload = vec![0u8; header.payload_len as usize];
    reader.read_exact(&mut payload).await?;
    Ok(Some((header, payload)))
}

/// Write an `AudioPacketHeader` followed by its Opus payload bytes.
pub async fn write_audio_packet<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &AudioPacketHeader,
    payload: &[u8],
) -> Result<()> {
    let encoded_hdr =
        postcard::to_allocvec(header).context("failed to serialize AudioPacketHeader")?;
    let hdr_len = u32::try_from(encoded_hdr.len()).context("audio header length overflow")?;
    writer.write_u32(hdr_len).await?;
    writer.write_all(&encoded_hdr).await?;
    writer.write_all(payload).await?;
    Ok(())
}

/// Read an `AudioPacketHeader` and its Opus payload bytes (`None` on clean EOF).
pub async fn read_audio_packet<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<(AudioPacketHeader, Vec<u8>)>> {
    let hdr_len = match reader.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    if hdr_len > 4096 {
        bail!("audio packet header too large: {hdr_len}");
    }
    let mut hdr_buf = vec![0u8; hdr_len];
    reader.read_exact(&mut hdr_buf).await?;
    let header: AudioPacketHeader =
        postcard::from_bytes(&hdr_buf).context("failed to deserialize AudioPacketHeader")?;
    if header.payload_len as usize > MAX_CONTROL_FRAME_BYTES {
        bail!("audio payload too large: {}", header.payload_len);
    }
    let mut payload = vec![0u8; header.payload_len as usize];
    reader.read_exact(&mut payload).await?;
    Ok(Some((header, payload)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_selection_fd_reports_completeness() {
        use std::io::Write;
        use std::time::Duration;

        let (r, mut w) = std::io::pipe().unwrap();
        w.write_all(b"file:///a\n").unwrap();
        drop(w);
        assert_eq!(
            read_selection_fd(r.into(), Duration::from_secs(1)),
            (b"file:///a\n".to_vec(), true)
        );

        // The writer stalls: what arrived is returned, marked incomplete.
        let (r, mut w) = std::io::pipe().unwrap();
        w.write_all(b"file:///Docu").unwrap();
        let (buf, complete) = read_selection_fd(r.into(), Duration::from_millis(50));
        assert_eq!(buf, b"file:///Docu");
        assert!(!complete);
        drop(w);
    }

    #[test]
    fn port_ranges_parse_and_display() {
        let single: PortRange = "7443".parse().unwrap();
        assert_eq!((single.start(), single.end()), (7443, 7443));
        assert_eq!(single.to_string(), "7443");

        let range: PortRange = " 7443 - 7462 ".parse().unwrap();
        assert_eq!((range.start(), range.end()), (7443, 7462));
        assert_eq!(range.to_string(), "7443-7462");
        assert!(range.contains(7450) && !range.contains(7463) && !range.contains(7442));
        assert_eq!(range.ports().count(), 20);

        for bad in [
            "",
            "0",
            "0-10",
            "7462-7443",
            "65536",
            "a",
            "1-",
            "-5",
            "1-2-3",
        ] {
            assert!(
                bad.parse::<PortRange>().is_err(),
                "{bad:?} should not parse"
            );
        }
    }

    #[tokio::test]
    async fn protocol_preamble_roundtrip() {
        let (mut tx, mut rx) = tokio::io::duplex(64);
        write_protocol_preamble(&mut tx).await.unwrap();
        assert_eq!(
            read_protocol_preamble(&mut rx).await.unwrap(),
            PeerProtocol::Version(PROTOCOL_VERSION)
        );
    }

    #[tokio::test]
    async fn legacy_peer_is_detected_without_preamble() {
        // Pre-v9 peers open the control stream with a big-endian u32 frame length.
        let (mut tx, mut rx) = tokio::io::duplex(64);
        tx.write_u32(57).await.unwrap();
        tx.write_all(&[0u8; 8]).await.unwrap();
        assert_eq!(
            read_protocol_preamble(&mut rx).await.unwrap(),
            PeerProtocol::Legacy
        );
    }

    #[tokio::test]
    async fn control_message_roundtrip() {
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        let original = ControlMessage::WindowCreated {
            window_id: 42,
            title: "Alacritty".into(),
            app_id: "Alacritty".into(),
            width: 1280,
            height: 720,
        };

        write_control_message(&mut tx, &original).await.unwrap();
        drop(tx);

        let decoded = read_control_message(&mut rx).await.unwrap().unwrap();
        assert_eq!(original, decoded);
        assert!(read_control_message(&mut rx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn video_stream_roundtrip() {
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        let stream_hdr = StreamHeader::Video {
            window_id: 7,
            codec: VideoCodec::H264,
        };
        write_stream_header(&mut tx, &stream_hdr).await.unwrap();

        let frame_hdr = VideoFrameHeader {
            pts_ns: 16_666_666,
            width: 1200,
            height: 900,
            logical_width: 800,
            logical_height: 600,
            geometry: FrameRect {
                x: 24,
                y: 18,
                width: 752,
                height: 558,
            },
            input_bounds: FrameRect {
                x: 12,
                y: 6,
                width: 776,
                height: 582,
            },
            is_keyframe: true,
            payload_len: 4,
        };
        write_video_frame(&mut tx, &frame_hdr, &[1, 2, 3, 4])
            .await
            .unwrap();
        drop(tx);

        assert_eq!(read_stream_header(&mut rx).await.unwrap(), stream_hdr);
        let (dec_hdr, dec_payload) = read_video_frame(&mut rx).await.unwrap().unwrap();
        assert_eq!(dec_hdr, frame_hdr);
        assert_eq!(dec_payload, vec![1, 2, 3, 4]);
        assert!(read_video_frame(&mut rx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn audio_stream_roundtrip() {
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        let stream_hdr = StreamHeader::AudioPlayback;
        write_stream_header(&mut tx, &stream_hdr).await.unwrap();

        let pkt_hdr = AudioPacketHeader {
            pts_ns: 20_000_000,
            payload_len: 3,
        };
        write_audio_packet(&mut tx, &pkt_hdr, &[9, 8, 7])
            .await
            .unwrap();
        drop(tx);

        assert_eq!(read_stream_header(&mut rx).await.unwrap(), stream_hdr);
        let (dec_hdr, dec_payload) = read_audio_packet(&mut rx).await.unwrap().unwrap();
        assert_eq!(dec_hdr, pkt_hdr);
        assert_eq!(dec_payload, vec![9, 8, 7]);
        assert!(read_audio_packet(&mut rx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn primary_selection_and_cursor_bitmap_roundtrip() {
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        let primary = ControlMessage::PrimarySelectionText {
            text: "middle-click paste".into(),
        };
        let cursor = ControlMessage::CursorBitmap {
            window_id: 9,
            width: 4,
            height: 4,
            logical_width: 2,
            logical_height: 2,
            hotspot_x: 1,
            hotspot_y: 1,
            bgra: [0u8, 0, 0, 255].repeat(16),
        };
        let scheme = ControlMessage::ColorScheme { scheme: 1 };
        let kbd_enter = ControlMessage::KeyboardEnter {
            window_id: 9,
            pressed_keys: vec![29, 42],
        };
        let kbd_leave = ControlMessage::KeyboardLeave { window_id: 9 };
        let frame_ack = ControlMessage::FrameAck {
            window_id: 9,
            pts_ns: 123_456_789,
        };
        let display_scale = ControlMessage::DisplayScale { scale_factor: 1.5 };
        let button_layout = ControlMessage::ButtonLayout {
            layout: Some("appmenu:minimize,maximize,close".to_string()),
        };
        let no_button_layout = ControlMessage::ButtonLayout { layout: None };
        let accent = ControlMessage::AccentColor {
            rgb: Some([0.21, 0.52, 0.89]),
        };
        let no_accent = ControlMessage::AccentColor { rgb: None };
        let contrast = ControlMessage::Contrast { contrast: 1 };
        let downscale = ControlMessage::ServerDownscale { enabled: false };

        write_control_message(&mut tx, &primary).await.unwrap();
        write_control_message(&mut tx, &cursor).await.unwrap();
        write_control_message(&mut tx, &scheme).await.unwrap();
        write_control_message(&mut tx, &kbd_enter).await.unwrap();
        write_control_message(&mut tx, &kbd_leave).await.unwrap();
        write_control_message(&mut tx, &frame_ack).await.unwrap();
        write_control_message(&mut tx, &display_scale)
            .await
            .unwrap();
        write_control_message(&mut tx, &button_layout)
            .await
            .unwrap();
        write_control_message(&mut tx, &no_button_layout)
            .await
            .unwrap();
        for msg in [&accent, &no_accent, &contrast, &downscale] {
            write_control_message(&mut tx, msg).await.unwrap();
        }
        drop(tx);

        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            primary
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            cursor
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            scheme
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            kbd_enter
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            kbd_leave
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            frame_ack
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            display_scale
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            button_layout
        );
        assert_eq!(
            read_control_message(&mut rx).await.unwrap().unwrap(),
            no_button_layout
        );
        for msg in [accent, no_accent, contrast, downscale] {
            assert_eq!(read_control_message(&mut rx).await.unwrap().unwrap(), msg);
        }
        assert!(read_control_message(&mut rx).await.unwrap().is_none());
    }

    #[test]
    fn is_valid_accent_color_requires_unit_range() {
        assert!(is_valid_accent_color(&[0.0, 0.5, 1.0]));
        assert!(!is_valid_accent_color(&[-1.0, -1.0, -1.0]));
        assert!(!is_valid_accent_color(&[0.2, 1.5, 0.2]));
        assert!(!is_valid_accent_color(&[f64::NAN, 0.2, 0.2]));
    }

    #[test]
    fn is_valid_button_layout_rejects_unsafe_values() {
        assert!(is_valid_button_layout("appmenu:minimize,maximize,close"));
        assert!(is_valid_button_layout("close,minimize:"));
        assert!(is_valid_button_layout(":"));
        assert!(is_valid_button_layout("close,minimize,maximize"));
        assert!(!is_valid_button_layout(""));
        assert!(!is_valid_button_layout("x:close'\n[evil]"));
        assert!(!is_valid_button_layout(&format!(
            "{}:close",
            "a".repeat(300)
        )));
    }

    #[tokio::test]
    async fn app_list_roundtrip_preserves_search_metadata() {
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        let msg = ControlMessage::AppList {
            apps: vec![
                RemoteAppEntry {
                    id: "firefox.desktop".into(),
                    name: "Firefox".into(),
                    generic_name: Some("Web Browser".into()),
                    keywords: vec!["internet".into(), "www".into()],
                    exec: "firefox".into(),
                    categories: vec!["Network".into()],
                    terminal: false,
                    icon_mime: None,
                    icon_data: Vec::new(),
                    icon_symbolic: false,
                },
                RemoteAppEntry {
                    id: "xterm.desktop".into(),
                    name: "XTerm".into(),
                    generic_name: None,
                    keywords: Vec::new(),
                    exec: "xterm".into(),
                    categories: Vec::new(),
                    terminal: true,
                    icon_mime: None,
                    icon_data: Vec::new(),
                    icon_symbolic: false,
                },
            ],
        };
        write_control_message(&mut tx, &msg).await.unwrap();
        drop(tx);
        assert_eq!(read_control_message(&mut rx).await.unwrap().unwrap(), msg);
    }
}
