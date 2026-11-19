//! Shared wire protocol and stream framing for RemoteWayland.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Current protocol version negotiated during the initial handshake.
pub const PROTOCOL_VERSION: u32 = 3;

/// Maximum allowed control frame size (16 MiB) to prevent unbounded allocations.
pub const MAX_CONTROL_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Maximum UTF-8 selection payload size (4 MiB) read from clipboard or primary selection pipes.
pub const MAX_SELECTION_BYTES: usize = 4 * 1024 * 1024;

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
}

/// Framed messages exchanged over a bidirectional PTY QUIC stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtyMessage {
    /// Sent by the server immediately upon creating or reattaching to a PTY,
    /// carrying the current byte offset so future reconnects stay exact.
    Attached { pty_id: u64, stream_offset: u64 },
    /// Historical buffered output replayed on reattach (already accounted for in `stream_offset`).
    Replay(Vec<u8>),
    /// Sent by the server when a reattach is rejected because another terminal is already attached.
    Busy,
    Data(Vec<u8>),
    Resize { cols: u16, rows: u16 },
    /// Sent by the client on intentional close (e.g., terminal tab closed via SIGHUP).
    Close,
    Exit(i32),
}

/// Metadata + optional icon bytes for a discovered remote `.desktop` application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteAppEntry {
    pub id: String,
    pub name: String,
    pub exec: String,
    pub categories: Vec<String>,
    pub icon_mime: Option<String>,
    pub icon_data: Vec<u8>,
}

/// Lifecycle state of a launched remote application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchState {
    Launching,
    Running,
    Exited(i32),
}

/// Bidirectional control-stream messages (QUIC stream 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client -> Server authentication and capability advertisement.
    AuthRequest {
        protocol_version: u32,
        token: String,
        supported_codecs: Vec<VideoCodec>,
        enable_audio_bridge: bool,
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
    /// Server -> Client: lossless XOR-delta + LZ4 BGRA (`Argb8888`) pixel buffer for an `xdg_popup`.
    PopupFrame {
        window_id: WindowId,
        width: u32,
        height: u32,
        is_keyframe: bool,
        bgra: Vec<u8>,
    },
    /// Server -> Client: remote app requested interactive window move (`xdg_toplevel.move`).
    StartMove {
        window_id: WindowId,
    },
    /// Server -> Client: remote app requested interactive window resize (`xdg_toplevel.resize`).
    StartResize {
        window_id: WindowId,
        edge: u32,
    },
    /// Server -> Client: remote app requested maximize or unmaximize (`xdg_toplevel.set_maximized` / `unset_maximized`).
    SetMaximized {
        window_id: WindowId,
        maximized: bool,
    },
    /// Server -> Client: remote app requested minimize (`xdg_toplevel.set_minimized`).
    SetMinimized {
        window_id: WindowId,
    },
    /// Server -> Client: remote app requested fullscreen or unfullscreen (`xdg_toplevel.set_fullscreen` / `unset_fullscreen`).
    SetFullscreen {
        window_id: WindowId,
        fullscreen: bool,
    },
    /// Server -> Client: remote app requested the compositor window menu (`xdg_toplevel.show_window_menu`).
    ShowWindowMenu {
        window_id: WindowId,
        x: i32,
        y: i32,
    },
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
    WindowClosed {
        window_id: WindowId,
    },

    /// Client -> Server: pointer moved in surface-local logical coordinates.
    PointerMotion {
        window_id: WindowId,
        x: f64,
        y: f64,
    },
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

    /// Bidirectional: UTF-8 clipboard text synchronization.
    ClipboardText {
        text: String,
    },
    /// Bidirectional: UTF-8 primary selection (middle-click paste) text synchronization.
    PrimarySelectionText {
        text: String,
    },
    /// Server -> Client: enable/disable host microphone capture based on remote app demand.
    SetMicActive {
        active: bool,
    },
    /// Server -> Client: cursor visibility or named cursor shape update.
    CursorUpdate {
        window_id: WindowId,
        visible: bool,
        shape_name: String,
    },
    /// Server -> Client: surface-based cursor bitmap (`Argb8888` BGRA bytes + hotspot).
    CursorBitmap {
        window_id: WindowId,
        width: u32,
        height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        bgra: Vec<u8>,
    },

    /// Client -> Server: request list of discovered `.desktop` applications.
    ListApps,
    /// Server -> Client: list of discovered `.desktop` applications with optional icons.
    AppList {
        apps: Vec<RemoteAppEntry>,
    },
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
    TerminateApp {
        launch_id: u64,
        force: bool,
    },
}

/// Header preceding each encoded video access unit on a per-window video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoFrameHeader {
    pub pts_ns: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub payload_len: u32,
}

/// Header preceding each Opus audio packet on an audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPacketHeader {
    pub pts_ns: u64,
    pub payload_len: u32,
}

/// Serialize and write a length-prefixed `ControlMessage` to an async writer.
pub async fn write_control_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &ControlMessage,
) -> Result<()> {
    let encoded = bincode::serialize(msg).context("failed to serialize ControlMessage")?;
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
    let msg = bincode::deserialize(&buf).context("failed to deserialize ControlMessage")?;
    Ok(Some(msg))
}

/// Write a `StreamHeader` at the start of a unidirectional QUIC stream.
pub async fn write_stream_header<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &StreamHeader,
) -> Result<()> {
    let encoded = bincode::serialize(header).context("failed to serialize StreamHeader")?;
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
    bincode::deserialize(&buf).context("failed to deserialize StreamHeader")
}

/// Write a `BidiStreamHeader` at the start of a secondary bidirectional QUIC stream.
pub async fn write_bidi_stream_header<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &BidiStreamHeader,
) -> Result<()> {
    let encoded = bincode::serialize(header).context("failed to serialize BidiStreamHeader")?;
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
    bincode::deserialize(&buf).context("failed to deserialize BidiStreamHeader")
}

/// Write a length-prefixed `PtyMessage` to a PTY QUIC stream.
pub async fn write_pty_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &PtyMessage,
) -> Result<()> {
    let encoded = bincode::serialize(msg).context("failed to serialize PtyMessage")?;
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
    let msg = bincode::deserialize(&buf).context("failed to deserialize PtyMessage")?;
    Ok(Some(msg))
}

/// Write a `VideoFrameHeader` followed by its encoded payload bytes.
pub async fn write_video_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &VideoFrameHeader,
    payload: &[u8],
) -> Result<()> {
    let encoded_hdr =
        bincode::serialize(header).context("failed to serialize VideoFrameHeader")?;
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
        bincode::deserialize(&hdr_buf).context("failed to deserialize VideoFrameHeader")?;
    if header.payload_len as usize > MAX_CONTROL_FRAME_BYTES {
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
        bincode::serialize(header).context("failed to serialize AudioPacketHeader")?;
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
        bincode::deserialize(&hdr_buf).context("failed to deserialize AudioPacketHeader")?;
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
            width: 800,
            height: 600,
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
            width: 2,
            height: 2,
            hotspot_x: 1,
            hotspot_y: 1,
            bgra: [0u8, 0, 0, 255].repeat(4),
        };

        write_control_message(&mut tx, &primary).await.unwrap();
        write_control_message(&mut tx, &cursor).await.unwrap();
        drop(tx);

        assert_eq!(read_control_message(&mut rx).await.unwrap().unwrap(), primary);
        assert_eq!(read_control_message(&mut rx).await.unwrap().unwrap(), cursor);
        assert!(read_control_message(&mut rx).await.unwrap().is_none());
    }
}
