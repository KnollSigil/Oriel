//! QUIC client transport with SHA-256 certificate fingerprint pinning.

use std::fs::{self, Permissions};
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use oriel_proto::{
    is_valid_accent_color, is_valid_button_layout, read_audio_packet, read_control_message,
    read_stream_header, read_video_frame, write_audio_packet, write_control_message,
    write_protocol_preamble, write_stream_header, AudioPacketHeader, ClientRole, ControlMessage,
    LaunchState, RemoteAppEntry, StreamHeader, VideoCodec, WindowId, CLOSE_CODE_PROTOCOL_MISMATCH,
    CLOSE_CODE_SUPERSEDED,
};
use quinn::{ClientConfig, ConnectionError, Endpoint, TransportConfig, VarInt};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as RustlsError, SignatureScheme};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::audio::{start_mic_capture, AudioPlaybackSink};
use crate::media::WindowVideoDecoder;
use crate::window::{ClientUiEvent, UiSender};

/// This client process's TLS identity: a self-signed certificate made at startup whose
/// private key never leaves memory. Its fingerprint goes to `oriel-server attach
/// --client-fp`, which binds the minted token to it, so a token that leaks (through a
/// process list, a log, a shoulder) is useless without this process's key.
pub struct ClientIdentity {
    cert: CertificateDer<'static>,
    key: PrivatePkcs8KeyDer<'static>,
    /// SHA-256 hex fingerprint of `cert`.
    pub fingerprint: String,
}

/// The process-wide client identity, generated on first use.
pub fn client_identity() -> Result<&'static ClientIdentity> {
    static IDENTITY: OnceLock<ClientIdentity> = OnceLock::new();
    if let Some(identity) = IDENTITY.get() {
        return Ok(identity);
    }
    let cert = rcgen::generate_simple_self_signed(vec!["oriel-client.local".to_string()])
        .context("failed to generate the client TLS certificate")?;
    let cert_der = CertificateDer::from(cert.cert);
    let fingerprint = Sha256::digest(cert_der.as_ref())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let identity = ClientIdentity {
        key: PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()),
        cert: cert_der,
        fingerprint,
    };
    // Another thread may have won the race; either identity is fine, but use the stored one.
    Ok(IDENTITY.get_or_init(|| identity))
}

#[derive(Debug)]
struct PinnedCertVerifier {
    expected_sha256_hex: String,
    supported: rustls::crypto::WebPkiSupportedAlgorithms,
}

impl PinnedCertVerifier {
    fn new(expected_sha256_hex: String) -> Self {
        Self {
            expected_sha256_hex: expected_sha256_hex.to_ascii_lowercase(),
            supported: rustls::crypto::ring::default_provider().signature_verification_algorithms,
        }
    }
}

impl ServerCertVerifier for PinnedCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        let mut hasher = Sha256::new();
        hasher.update(end_entity.as_ref());
        let actual: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        if actual != self.expected_sha256_hex {
            return Err(RustlsError::General(format!(
                "server certificate fingerprint mismatch: expected {}, got {actual}",
                self.expected_sha256_hex
            )));
        }

        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.supported)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.supported)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported.supported_schemes()
    }
}

/// The server ended the session because another display client took it over. Reconnecting
/// automatically would take it back, so callers should stop instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionTakenOver;

impl std::fmt::Display for SessionTakenOver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("session opened by another client")
    }
}

impl std::error::Error for SessionTakenOver {}

/// Whether the server closed `conn` because another display client took over.
fn closed_as_superseded(conn: &quinn::Connection) -> bool {
    matches!(
        conn.close_reason(),
        Some(ConnectionError::ApplicationClosed(close))
            if close.error_code == VarInt::from_u32(CLOSE_CODE_SUPERSEDED)
    )
}

/// Establish a QUIC connection to `server_addr`, presenting this process's client
/// certificate and accepting only a server certificate with SHA-256 `fingerprint_sha256`.
/// Nothing is sent yet; [`connect_and_auth`] also authenticates.
pub async fn connect_raw(
    server_addr: SocketAddr,
    fingerprint_sha256: String,
) -> Result<quinn::Connection> {
    let verifier = Arc::new(PinnedCertVerifier::new(fingerprint_sha256));
    let identity = client_identity()?;
    let mut rustls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(
            vec![identity.cert.clone()],
            PrivateKeyDer::Pkcs8(identity.key.clone_key()),
        )
        .context("failed to set the client TLS certificate")?;
    rustls_config.alpn_protocols = vec![b"oriel/1".to_vec()];

    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(rustls_config)
        .context("failed to create QUIC client crypto config")?;
    let mut client_config = ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = TransportConfig::default();
    transport.congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));
    transport.keep_alive_interval(Some(Duration::from_secs(3)));
    if let Ok(idle) = Duration::from_secs(10).try_into() {
        transport.max_idle_timeout(Some(idle));
    }
    client_config.transport_config(Arc::new(transport));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
    endpoint.set_default_client_config(client_config);

    let conn = endpoint
        .connect(server_addr, "oriel.local")?
        .await
        .with_context(|| format!("failed to connect to {server_addr}"))?;
    info!(%server_addr, "QUIC connection established");
    Ok(conn)
}

/// Establish a pinned QUIC connection and perform the `AuthRequest`/`AuthResponse` handshake.
/// A [`ClientRole::Display`] connection takes over the server's session from any other
/// display client; a [`ClientRole::PtyOnly`] one coexists with it.
pub async fn connect_and_auth(
    server_addr: SocketAddr,
    fingerprint_sha256: String,
    token: String,
    role: ClientRole,
    enable_audio_bridge: bool,
    initial_scale: f64,
) -> Result<(
    quinn::Connection,
    quinn::SendStream,
    quinn::RecvStream,
    VideoCodec,
)> {
    let conn = connect_raw(server_addr, fingerprint_sha256).await?;

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .context("failed to open control stream")?;

    write_protocol_preamble(&mut send).await?;
    write_control_message(
        &mut send,
        &ControlMessage::AuthRequest {
            token,
            role,
            supported_codecs: vec![VideoCodec::SmartTile, VideoCodec::Lz4Delta],
            enable_audio_bridge,
            initial_scale,
        },
    )
    .await?;

    let response = match read_control_message(&mut recv).await {
        Ok(response) => response,
        Err(err) => return Err(auth_read_error(&conn, err)),
    };
    let selected_codec = match response {
        Some(ControlMessage::AuthResponse {
            accepted: true,
            selected_codec,
            ..
        }) => {
            info!(?selected_codec, "authenticated with server");
            selected_codec
        }
        Some(ControlMessage::AuthResponse {
            accepted: false,
            reason,
            ..
        }) => {
            bail!(
                "server rejected authentication: {}",
                reason.unwrap_or_else(|| "unknown reason".into())
            );
        }
        other => bail!("unexpected auth response: {other:?}"),
    };

    Ok((conn, send, recv, selected_codec))
}

/// Explain why reading the server's `AuthResponse` failed, using the QUIC close reason
/// when the server closed the connection.
fn auth_read_error(conn: &quinn::Connection, err: anyhow::Error) -> anyhow::Error {
    match conn.close_reason() {
        Some(ConnectionError::ApplicationClosed(close))
            if close.error_code == VarInt::from_u32(CLOSE_CODE_PROTOCOL_MISMATCH) =>
        {
            anyhow::anyhow!(
                "server rejected the connection: {}. If the server is still running an \
                 older or newer install, run `oriel-server attach --replace` there to \
                 restart it with the installed version (this closes its apps)",
                String::from_utf8_lossy(&close.reason)
            )
        }
        // Servers older than protocol version 9 do not understand the preamble and drop
        // the connection without a reason.
        Some(ConnectionError::ApplicationClosed(close)) if close.reason.is_empty() => err.context(
            "server closed the connection during authentication; \
             it may be running an older, incompatible version",
        ),
        _ => err.context("failed to read authentication response"),
    }
}

/// Events sent from the QUIC client to the launcher UI.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "gtk4"), allow(dead_code))]
pub enum LauncherNetEvent {
    /// The QUIC handshake and authentication succeeded. Always the first event.
    Connected,
    AppList(Vec<RemoteAppEntry>),
    AppState {
        launch_id: u64,
        name: String,
        state: LaunchState,
    },
    /// See [`ControlMessage::AppWindow`].
    AppWindow {
        launch_id: u64,
        window_id: oriel_proto::WindowId,
    },
    /// Sent every second.
    NetStats {
        rtt_ms: u32,
        congested: bool,
        /// Bytes per second received and sent over the connection, everything included.
        rx_rate: u64,
        tx_rate: u64,
        /// File transfers over the connection, oldest first.
        transfers: Vec<TransferStatus>,
        /// Windows and app audio streams receiving at least [`MIN_LISTED_RATE`], busiest
        /// first.
        sources: Vec<SourceRate>,
    },
    Disconnected(Option<String>),
    /// The server closed the session because another display client took it over.
    TakenOver,
}

/// Received traffic slower than this (bytes per second) is not listed per window: an idle
/// window still gets the odd cursor blink or clock tick.
pub const MIN_LISTED_RATE: u64 = 1000;
/// Most windows and audio streams listed in [`LauncherNetEvent::NetStats`].
const MAX_LISTED_SOURCES: usize = 8;

/// A window's or app audio stream's received traffic in [`LauncherNetEvent::NetStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRate {
    pub source: crate::traffic::Source,
    /// The window title or the audio stream's app.
    pub name: String,
    /// The window's app id, or what the audio stream plays.
    pub detail: String,
    /// Bytes per second received over the last second.
    pub rate: u64,
}

/// One file transfer in [`LauncherNetEvent::NetStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferStatus {
    pub info: oriel_proto::transfers::TransferInfo,
    /// Bytes per second it moved over the last second.
    pub rate: u64,
}

/// Listen on the local 0600 Unix socket `sock_path` and tunnel each `oriel-client pty-attach
/// --socket` connection over its own QUIC bidirectional stream on `conn`.
fn spawn_pty_tunnel(
    conn: &quinn::Connection,
    sock_path: PathBuf,
) -> Option<tokio::task::JoinHandle<()>> {
    let _ = fs::remove_file(&sock_path);
    match tokio::net::UnixListener::bind(&sock_path) {
        Ok(listener) => {
            if let Err(err) = fs::set_permissions(&sock_path, Permissions::from_mode(0o600)) {
                warn!(path = %sock_path.display(), error = %err, "failed to chmod 0600 PTY tunnel socket");
            }
            info!(path = %sock_path.display(), "local PTY Unix tunnel listening (mode 0600)");
            let tunnel_conn = conn.clone();
            Some(tokio::spawn(async move {
                let mut streams = tokio::task::JoinSet::new();
                while let Ok((unix_stream, _)) = listener.accept().await {
                    while streams.try_join_next().is_some() {}
                    let c = tunnel_conn.clone();
                    streams.spawn(async move {
                        let Ok((mut quic_send, mut quic_recv)) = c.open_bi().await else {
                            warn!("failed to open QUIC bidi stream for tunneled PTY");
                            return;
                        };
                        let (mut unix_read, mut unix_write) = unix_stream.into_split();
                        let up = async {
                            let _ = tokio::io::copy(&mut unix_read, &mut quic_send).await;
                            let _ = quic_send.finish();
                        };
                        let down = async {
                            let _ = tokio::io::copy(&mut quic_recv, &mut unix_write).await;
                        };
                        tokio::join!(up, down);
                    });
                }
            }))
        }
        Err(err) => {
            warn!(path = %sock_path.display(), error = %err, "failed to bind local PTY tunnel socket");
            None
        }
    }
}

/// Apply clipboard and drag offers from the server in order: a file offer takes a moment to
/// set up (the first one mounts the transfer filesystem), and text copied after it must not
/// be overtaken. Returns where to send the offers.
fn spawn_clipboard_applier(
    conn: quinn::Connection,
    ui_tx: UiSender,
) -> (
    mpsc::UnboundedSender<ControlMessage>,
    tokio::task::JoinHandle<()>,
) {
    let (clipboard_tx, mut clipboard_rx) = mpsc::unbounded_channel::<ControlMessage>();
    let task = tokio::spawn(async move {
        while let Some(msg) = clipboard_rx.recv().await {
            match msg {
                ControlMessage::ClipboardOffer {
                    offer_id,
                    mimes,
                    text,
                    files,
                } => {
                    let files =
                        receive_files(&conn, oriel_proto::xfer::OfferSlot::Clipboard, files).await;
                    if files.is_none() && text.is_none() && mimes.is_empty() {
                        continue;
                    }
                    let clip = oriel_proto::selection::ReceivedClipboard {
                        offer_id,
                        mimes,
                        text,
                        files,
                    };
                    let fetch = crate::xfer::SelectionFetcher {
                        conn: conn.clone(),
                        rt: tokio::runtime::Handle::current(),
                    };
                    let _ = ui_tx.send(ClientUiEvent::ClipboardOffer { clip, fetch });
                }
                ControlMessage::RemoteDragStart {
                    window_id,
                    offer_id,
                    mimes,
                    files,
                } => {
                    let files =
                        receive_files(&conn, oriel_proto::xfer::OfferSlot::Drag, files).await;
                    if files.is_none() && mimes.is_empty() {
                        continue;
                    }
                    let clip = oriel_proto::selection::ReceivedClipboard {
                        offer_id,
                        mimes,
                        text: None,
                        files,
                    };
                    let fetch = crate::xfer::SelectionFetcher {
                        conn: conn.clone(),
                        rt: tokio::runtime::Handle::current(),
                    };
                    let _ = ui_tx.send(ClientUiEvent::RemoteDragStart {
                        window_id,
                        clip,
                        fetch,
                    });
                }
                _ => {}
            }
        }
    });
    (clipboard_tx, task)
}

/// Present the files of a server offer through the transfer mount; `None` when there are
/// none or none could be presented.
async fn receive_files(
    conn: &quinn::Connection,
    slot: oriel_proto::xfer::OfferSlot,
    files: Option<oriel_proto::xfer::FileOffer>,
) -> Option<Vec<PathBuf>> {
    let f = files?;
    let conn = conn.clone();
    let paths = tokio::task::spawn_blocking(move || {
        crate::xfer::receive_offer(conn, slot, f.offer_id, f.roots)
    })
    .await
    .unwrap_or_default();
    (!paths.is_empty()).then_some(paths)
}

/// Report the connection's round trip, loss, rates, file transfers and busiest traffic
/// sources to the launcher once a second, until it goes away.
fn spawn_stats_reporter(
    stats_conn: quinn::Connection,
    tx: mpsc::UnboundedSender<LauncherNetEvent>,
    traffic: Arc<crate::traffic::TrafficMeter>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Count from now: what connecting sent (the handshake, the first window images)
        // is not one second's traffic.
        let start = stats_conn.stats();
        let mut prev_lost = start.path.lost_packets;
        let mut prev_source_bytes = std::collections::HashMap::<crate::traffic::Source, u64>::new();
        let mut prev_bytes = (start.udp_rx.bytes, start.udp_tx.bytes);
        let mut prev_moved = std::collections::HashMap::<u64, u64>::new();
        let mut prev_at = std::time::Instant::now();
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            let secs = prev_at.elapsed().as_secs_f64().max(0.001);
            prev_at = std::time::Instant::now();
            let per_sec = |delta: u64| (delta as f64 / secs) as u64;
            let rtt_ms = stats_conn.rtt().as_millis() as u32;
            let stats = stats_conn.stats();
            let lost = stats.path.lost_packets;
            let loss_delta = lost.saturating_sub(prev_lost);
            prev_lost = lost;
            let congested = rtt_ms > 150 || loss_delta > 5;
            let bytes = (stats.udp_rx.bytes, stats.udp_tx.bytes);
            let rx_rate = per_sec(bytes.0.saturating_sub(prev_bytes.0));
            let tx_rate = per_sec(bytes.1.saturating_sub(prev_bytes.1));
            prev_bytes = bytes;
            let transfers: Vec<TransferStatus> =
                oriel_proto::transfers::list(stats_conn.stable_id())
                    .into_iter()
                    .map(|info| {
                        // A transfer listed for the first time has moved what it has
                        // since it started, more than one tick ago.
                        let rate = match prev_moved.get(&info.id) {
                            Some(&before) => per_sec(info.moved.saturating_sub(before)),
                            None => (info.moved as f64 / info.age.as_secs_f64().max(secs)) as u64,
                        };
                        TransferStatus { info, rate }
                    })
                    .collect();
            prev_moved = transfers
                .iter()
                .map(|t| (t.info.id, t.info.moved))
                .collect();
            let totals = traffic.totals();
            let mut sources: Vec<SourceRate> = totals
                .iter()
                .filter_map(|t| {
                    // A source seen for the first time is listed from its second
                    // tick: what it received before is not from one tick.
                    let before = prev_source_bytes.get(&t.source).copied()?;
                    let rate = per_sec(t.bytes.saturating_sub(before));
                    (rate >= MIN_LISTED_RATE).then(|| SourceRate {
                        source: t.source,
                        name: t.name.clone(),
                        detail: t.detail.clone(),
                        rate,
                    })
                })
                .collect();
            prev_source_bytes = totals.iter().map(|t| (t.source, t.bytes)).collect();
            sources.sort_by(|a, b| b.rate.cmp(&a.rate).then_with(|| a.name.cmp(&b.name)));
            sources.truncate(MAX_LISTED_SOURCES);
            let event = LauncherNetEvent::NetStats {
                rtt_ms,
                congested,
                rx_rate,
                tx_rate,
                transfers,
                sources,
            };
            if tx.send(event).is_err() {
                break;
            }
        }
    })
}

/// Accept the server's unidirectional streams (window video, audio playback) and serve
/// each in a task of its own until the connection closes.
fn spawn_uni_stream_acceptor(
    conn: quinn::Connection,
    ui_tx: UiSender,
    frame_ack_tx: mpsc::UnboundedSender<(WindowId, u64)>,
    traffic: Arc<crate::traffic::TrafficMeter>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut streams = tokio::task::JoinSet::new();
        while let Ok(recv_uni) = conn.accept_uni().await {
            while streams.try_join_next().is_some() {}
            streams.spawn(serve_uni_stream(
                recv_uni,
                ui_tx.clone(),
                frame_ack_tx.clone(),
                Arc::clone(&traffic),
            ));
        }
    })
}

/// Serve one unidirectional stream from the server by its [`StreamHeader`]: decode and show a
/// window's video, or play an audio stream.
async fn serve_uni_stream(
    mut recv_uni: quinn::RecvStream,
    tx: UiSender,
    ack_tx: mpsc::UnboundedSender<(WindowId, u64)>,
    traffic: Arc<crate::traffic::TrafficMeter>,
) {
    match read_stream_header(&mut recv_uni).await {
        Ok(StreamHeader::Video { window_id, codec }) => {
            let mut decoder = Some(WindowVideoDecoder::new(codec));
            while let Ok(Some((hdr, payload))) = read_video_frame(&mut recv_uni).await {
                traffic.add_window(window_id, payload.len());
                let mut dec = decoder.take().unwrap();
                let decode_res = tokio::task::spawn_blocking(move || {
                    let res = dec.decode(hdr.width, hdr.height, hdr.is_keyframe, &payload);
                    (dec, res)
                })
                .await;
                let (dec, frame_res) = match decode_res {
                    Ok(pair) => pair,
                    Err(err) => {
                        warn!(window_id, error = %err, "video decoder task panicked");
                        break;
                    }
                };
                decoder = Some(dec);
                match frame_res {
                    Ok(frame) => {
                        tx.push_frame(
                            window_id,
                            frame.width,
                            frame.height,
                            hdr.logical_width,
                            hdr.logical_height,
                            hdr.geometry,
                            hdr.input_bounds,
                            frame.bgra,
                            frame.damage,
                        );
                        let _ = ack_tx.send((window_id, hdr.pts_ns));
                    }
                    Err(err) => warn!(window_id, error = %err, "video decode error"),
                }
            }
        }
        Ok(StreamHeader::AudioPlayback) => match AudioPlaybackSink::new() {
            Ok(sink) => {
                while let Ok(Some((hdr, payload))) = read_audio_packet(&mut recv_uni).await {
                    if let Err(err) = sink.push_audio(hdr.pts_ns, &payload) {
                        warn!(error = %err, "audio playback push error");
                        break;
                    }
                }
            }
            Err(err) => warn!(error = %err, "client audio playback unavailable"),
        },
        Ok(StreamHeader::AudioPlaybackApp {
            stream_id,
            app_name,
            media_name,
            icon_name,
            media_role,
        }) => match AudioPlaybackSink::new_for_app(&app_name, &media_name, &icon_name, &media_role)
        {
            Ok(sink) => {
                info!(stream_id, %app_name, %media_name, "opened per-app host audio stream");
                traffic.audio_opened(stream_id, &app_name, &media_name);
                while let Ok(Some((hdr, payload))) = read_audio_packet(&mut recv_uni).await {
                    traffic.add_audio(stream_id, payload.len());
                    if let Err(err) = sink.push_audio(hdr.pts_ns, &payload) {
                        warn!(stream_id, error = %err, "audio playback push error");
                        break;
                    }
                }
                traffic.audio_closed(stream_id);
            }
            Err(err) => {
                warn!(stream_id, %app_name, error = %err, "client per-app audio playback unavailable")
            }
        },
        Ok(other) => {
            info!(?other, "received unexpected uni stream");
        }
        Err(err) => warn!(error = %err, "failed to read StreamHeader"),
    }
}

/// Connect to the Oriel server, verify its TLS fingerprint, and run the session loops.
pub async fn run_client(
    server_addr: SocketAddr,
    fingerprint_sha256: String,
    token: String,
    enable_audio_bridge: bool,
    initial_scale: f64,
    ui_tx: UiSender,
    mut ctrl_rx: mpsc::UnboundedReceiver<ControlMessage>,
    launcher_tx: Option<mpsc::UnboundedSender<LauncherNetEvent>>,
    pty_socket_path: Option<PathBuf>,
) -> Result<()> {
    let (conn, mut send, mut recv, selected_codec) = connect_and_auth(
        server_addr,
        fingerprint_sha256,
        token,
        ClientRole::Display,
        enable_audio_bridge,
        initial_scale,
    )
    .await?;
    // Forget the connection's cancelled and paused files however this returns.
    let _forget_transfers = oriel_proto::transfers::ConnectionGuard::new(conn.stable_id());
    if let Some(tx) = &launcher_tx {
        let _ = tx.send(LauncherNetEvent::Connected);
    }

    // Per-stream tasks live in `JoinSet`s owned by their accept loops, so aborting an accept
    // loop on shutdown also aborts every stream it spawned (the runtime may be shared with
    // other connections and outlive this one).

    // Optional local 0600 Unix socket that tunnels `oriel-client pty-attach --socket <path>`
    // directly over multiplexed QUIC bidirectional streams on this authenticated connection.
    let pty_tunnel_task = pty_socket_path
        .clone()
        .and_then(|sock_path| spawn_pty_tunnel(&conn, sock_path));

    // Serves the server's reads of files this client put on the clipboard.
    let xfer_task = crate::xfer::spawn_stream_acceptor(conn.clone());

    // Clipboard updates from the server, applied in order: a file offer takes a moment to
    // set up (the first one mounts the transfer filesystem), and text copied after it must
    // not be overtaken.
    let (clipboard_tx, clipboard_task) = spawn_clipboard_applier(conn.clone(), ui_tx.clone());

    let traffic = Arc::new(crate::traffic::TrafficMeter::default());
    let stats_task = launcher_tx
        .as_ref()
        .map(|tx| spawn_stats_reporter(conn.clone(), tx.clone(), Arc::clone(&traffic)));

    // Task: query and monitor local desktop settings (`color-scheme`, titlebar `button-layout`)
    // and forward changes to the server.
    let (theme_tx, mut theme_rx) = mpsc::unbounded_channel::<ControlMessage>();
    let color_scheme_task = tokio::spawn(watch_local_desktop_settings(theme_tx));

    // Per-window decoded video frame acknowledgments for end-to-end in-flight frame pacing.
    let (frame_ack_tx, mut frame_ack_rx) = mpsc::unbounded_channel::<(WindowId, u64)>();

    // Task: forward local input/resize/close/theme/frame-ack messages to server control stream.
    let (ctrl_closed_tx, mut ctrl_closed_rx) = tokio::sync::oneshot::channel::<()>();
    let conn_id = conn.stable_id();
    let writer_task = tokio::spawn(async move {
        // Pause and resume requests for outgoing transfers (see `oriel_proto::transfers`),
        // sent in the order they were queued.
        let mut requests_queued = oriel_proto::transfers::peer_requests_queued();
        let mut requests: std::collections::VecDeque<_> =
            oriel_proto::transfers::take_peer_requests(conn_id).into();
        loop {
            if let Some(req) = requests.pop_front() {
                let oriel_proto::transfers::PeerRequest {
                    offer_id,
                    path,
                    paused,
                } = req;
                let msg = ControlMessage::PauseTransfer {
                    offer_id,
                    path,
                    paused,
                };
                if let Err(err) = write_control_message(&mut send, &msg).await {
                    warn!(error = %err, "failed to send control message");
                    break;
                }
                continue;
            }
            let msg = tokio::select! {
                m = ctrl_rx.recv() => match m {
                    Some(m) => m,
                    None => break,
                },
                Ok(()) = requests_queued.changed() => {
                    requests.extend(oriel_proto::transfers::take_peer_requests(conn_id));
                    continue;
                }
                Some(msg) = theme_rx.recv() => msg,
                Some((window_id, pts_ns)) = frame_ack_rx.recv() => ControlMessage::FrameAck {
                    window_id,
                    pts_ns,
                },
            };
            if let Err(err) = write_control_message(&mut send, &msg).await {
                warn!(error = %err, "failed to send control message");
                break;
            }
        }
        let _ = ctrl_closed_tx.send(());
    });

    // On-demand host microphone capture: only opened when a remote app requests input audio.
    let mut active_mic: Option<(crate::audio::MicCaptureHandle, tokio::task::JoinHandle<()>)> =
        None;

    // Task: accept per-window video and audio playback unidirectional streams.
    let uni_task = spawn_uni_stream_acceptor(
        conn.clone(),
        ui_tx.clone(),
        frame_ack_tx.clone(),
        Arc::clone(&traffic),
    );

    // Read server control events and dispatch to Wayland UI thread.
    let mut popup_decoders: std::collections::HashMap<WindowId, WindowVideoDecoder> =
        std::collections::HashMap::new();
    let mut server_mic_wanted = false;
    let mut mic_cfg_rx = crate::audio::subscribe_mic_config();

    let sync_mic = |wanted: bool,
                    cfg: &crate::audio::MicConfig,
                    active_mic: &mut Option<(
        crate::audio::MicCaptureHandle,
        tokio::task::JoinHandle<()>,
    )>| {
        let should_run = enable_audio_bridge && wanted && !cfg.muted;
        if !should_run {
            if let Some((handle, task)) = active_mic.take() {
                task.abort();
                drop(handle);
                info!("host microphone capture deactivated");
            }
            return;
        }
        if active_mic.is_some() {
            return;
        }
        match start_mic_capture(cfg.target.as_deref()) {
            Ok((handle, mut rx)) => {
                let mic_conn = conn.clone();
                let task = tokio::spawn(async move {
                    match mic_conn.open_uni().await {
                        Ok(mut stream) => {
                            if let Err(err) =
                                write_stream_header(&mut stream, &StreamHeader::AudioCapture).await
                            {
                                warn!(error = %err, "failed to write AudioCapture stream header");
                                return;
                            }
                            while let Some(pkt) = rx.recv().await {
                                let hdr = AudioPacketHeader {
                                    pts_ns: pkt.pts_ns,
                                    payload_len: pkt.data.len() as u32,
                                };
                                if let Err(err) =
                                    write_audio_packet(&mut stream, &hdr, &pkt.data).await
                                {
                                    warn!(error = %err, "AudioCapture stream write error");
                                    break;
                                }
                            }
                            let _ = stream.finish();
                        }
                        Err(err) => warn!(error = %err, "failed to open AudioCapture uni stream"),
                    }
                });
                *active_mic = Some((handle, task));
                info!(target = ?cfg.target, "host microphone capture activated");
            }
            Err(err) => warn!(error = %err, "client mic capture unavailable"),
        }
    };

    // Dedicated control-stream reader task: `read_control_message` performs multiple
    // `read_exact` calls and is NOT cancel-safe inside `tokio::select!`.
    let (ctrl_in_tx, mut ctrl_in_rx) = mpsc::channel::<Result<ControlMessage>>(256);
    let ctrl_reader_task = tokio::spawn(async move {
        loop {
            match read_control_message(&mut recv).await {
                Ok(Some(msg)) => {
                    if ctrl_in_tx.send(Ok(msg)).await.is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    let _ = ctrl_in_tx.send(Err(err)).await;
                    break;
                }
            }
        }
    });

    let loop_res: Result<()> = async {
        loop {
            let msg = tokio::select! {
                _ = &mut ctrl_closed_rx => break,
                res = ctrl_in_rx.recv() => match res {
                    Some(Ok(m)) => m,
                    Some(Err(err)) => return Err(err),
                    None => break,
                },
                res = mic_cfg_rx.changed() => {
                    if res.is_err() {
                        continue;
                    }
                    let cfg = mic_cfg_rx.borrow().clone();
                    if let Some((handle, task)) = active_mic.take() {
                        task.abort();
                        drop(handle);
                    }
                    sync_mic(server_mic_wanted, &cfg, &mut active_mic);
                    continue;
                }
            };
            match msg {
                ControlMessage::WindowCreated {
                    window_id,
                    title,
                    app_id,
                    width,
                    height,
                } => {
                    traffic.window_named(window_id, &title, &app_id);
                    let _ = ui_tx.send(ClientUiEvent::WindowCreated {
                        window_id,
                        title,
                        app_id,
                        width,
                        height,
                    });
                }
                ControlMessage::PopupCreated {
                    window_id,
                    parent_id,
                    x,
                    y,
                    width,
                    height,
                } => {
                    traffic.popup_created(window_id, parent_id);
                    let _ = ui_tx.send(ClientUiEvent::PopupCreated {
                        window_id,
                        parent_id,
                        x,
                        y,
                        width,
                        height,
                    });
                }
                ControlMessage::PopupRepositioned {
                    window_id,
                    x,
                    y,
                    width,
                    height,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::PopupRepositioned {
                        window_id,
                        x,
                        y,
                        width,
                        height,
                    });
                }
                ControlMessage::PopupFrame {
                    window_id,
                    width,
                    height,
                    logical_width,
                    logical_height,
                    geometry,
                    input_bounds,
                    is_keyframe,
                    bgra,
                } => {
                    traffic.add_window(window_id, bgra.len());
                    let decoder = popup_decoders
                        .entry(window_id)
                        .or_insert_with(|| WindowVideoDecoder::new(selected_codec));
                    match decoder.decode(width, height, is_keyframe, &bgra) {
                        Ok(frame) => {
                            let _ = ui_tx.send(ClientUiEvent::PopupFrame {
                                window_id,
                                width: frame.width,
                                height: frame.height,
                                logical_width,
                                logical_height,
                                geometry,
                                input_bounds,
                                bgra: frame.bgra,
                                damage: frame.damage,
                            });
                        }
                        Err(err) => warn!(window_id, error = %err, "popup decode error"),
                    }
                }
                ControlMessage::StartMove { window_id } => {
                    let _ = ui_tx.send(ClientUiEvent::StartMove { window_id });
                }
                ControlMessage::StartResize { window_id, edge } => {
                    let _ = ui_tx.send(ClientUiEvent::StartResize { window_id, edge });
                }
                ControlMessage::SetMaximized {
                    window_id,
                    maximized,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::SetMaximized {
                        window_id,
                        maximized,
                    });
                }
                ControlMessage::SetMinimized { window_id } => {
                    let _ = ui_tx.send(ClientUiEvent::SetMinimized { window_id });
                }
                ControlMessage::SetFullscreen {
                    window_id,
                    fullscreen,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::SetFullscreen {
                        window_id,
                        fullscreen,
                    });
                }
                ControlMessage::ShowWindowMenu { window_id, x, y } => {
                    let _ = ui_tx.send(ClientUiEvent::ShowWindowMenu { window_id, x, y });
                }
                ControlMessage::WindowParent {
                    window_id,
                    parent_id,
                    modal,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::WindowParent {
                        window_id,
                        parent_id,
                        modal,
                    });
                }
                ControlMessage::SetDecorationMode {
                    window_id,
                    server_side,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::SetDecorationMode {
                        window_id,
                        server_side,
                    });
                }
                ControlMessage::WindowClosed { window_id } => {
                    popup_decoders.remove(&window_id);
                    traffic.window_closed(window_id);
                    let _ = ui_tx.send(ClientUiEvent::WindowClosed { window_id });
                }
                ControlMessage::CursorUpdate {
                    window_id,
                    visible,
                    shape_name,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::CursorUpdate {
                        window_id,
                        visible,
                        shape_name,
                    });
                }
                ControlMessage::CursorBitmap {
                    window_id,
                    width,
                    height,
                    logical_width,
                    logical_height,
                    hotspot_x,
                    hotspot_y,
                    bgra,
                } => {
                    let _ = ui_tx.send(ClientUiEvent::CursorBitmap {
                        window_id,
                        width,
                        height,
                        logical_width,
                        logical_height,
                        hotspot_x,
                        hotspot_y,
                        bgra,
                    });
                }
                msg @ (ControlMessage::ClipboardOffer { .. }
                | ControlMessage::RemoteDragStart { .. }) => {
                    let _ = clipboard_tx.send(msg);
                }
                ControlMessage::DragFinished { offer_id } => {
                    let _ = ui_tx.send(ClientUiEvent::DragFinished { offer_id });
                }
                ControlMessage::PrimarySelectionText { text } => {
                    let _ = ui_tx.send(ClientUiEvent::PrimarySelectionText { text });
                }
                ControlMessage::SetMicActive { active } => {
                    server_mic_wanted = active;
                    let cfg = mic_cfg_rx.borrow().clone();
                    sync_mic(server_mic_wanted, &cfg, &mut active_mic);
                }
                ControlMessage::AppList { apps } => {
                    if let Some(tx) = &launcher_tx {
                        let icon_entries: Vec<crate::icon::AppIconEntry> = apps
                            .iter()
                            .filter(|a| !a.icon_data.is_empty())
                            .map(|a| crate::icon::AppIconEntry {
                                id: a.id.clone(),
                                name: a.name.clone(),
                                exec: a.exec.clone(),
                                icon_data: a.icon_data.clone(),
                            })
                            .collect();
                        let _ = ui_tx.send(ClientUiEvent::AppIconsUpdated {
                            entries: icon_entries,
                        });
                        let _ = tx.send(LauncherNetEvent::AppList(apps));
                    } else {
                        let icon_entries: Vec<crate::icon::AppIconEntry> = apps
                            .into_iter()
                            .filter(|a| !a.icon_data.is_empty())
                            .map(|a| crate::icon::AppIconEntry {
                                id: a.id,
                                name: a.name,
                                exec: a.exec,
                                icon_data: a.icon_data,
                            })
                            .collect();
                        let _ = ui_tx.send(ClientUiEvent::AppIconsUpdated {
                            entries: icon_entries,
                        });
                    }
                }
                ControlMessage::AppWindow {
                    launch_id,
                    window_id,
                } => {
                    if let Some(tx) = &launcher_tx {
                        let _ = tx.send(LauncherNetEvent::AppWindow {
                            launch_id,
                            window_id,
                        });
                    }
                }
                ControlMessage::AppState {
                    launch_id,
                    name,
                    state,
                } => {
                    if let Some(tx) = &launcher_tx {
                        let _ = tx.send(LauncherNetEvent::AppState {
                            launch_id,
                            name,
                            state,
                        });
                    }
                }
                other => {
                    info!(?other, "received control event from server");
                }
            }
        }
        Ok(())
    }
    .await;

    let taken_over = closed_as_superseded(&conn);
    conn.close(0u32.into(), b"client disconnect");
    let _ = ui_tx.send(ClientUiEvent::Disconnected);
    if let Some(tx) = &launcher_tx {
        let _ = tx.send(LauncherNetEvent::Disconnected(None));
    }
    if let Some(t) = pty_tunnel_task {
        t.abort();
    }
    if let Some(sock_path) = pty_socket_path {
        let _ = fs::remove_file(&sock_path);
    }
    if let Some(t) = stats_task {
        t.abort();
    }
    color_scheme_task.abort();
    xfer_task.abort();
    clipboard_task.abort();
    ctrl_reader_task.abort();
    writer_task.abort();
    if let Some((handle, task)) = active_mic.take() {
        task.abort();
        drop(handle);
    }
    uni_task.abort();
    if taken_over {
        info!("server session was opened by another client");
        return Err(SessionTakenOver.into());
    }
    loop_res
}

/// Parse an XDG `org.freedesktop.appearance` `color-scheme` (`uint32 0|1|2`) from
/// `gdbus call` or `gdbus monitor` output.
fn parse_gdbus_color_scheme(text: &str) -> Option<u32> {
    if let Some(idx) = text.find("uint32 ") {
        let rest = &text[idx + "uint32 ".len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let val: u32 = digits.parse().ok()?;
        return (val <= 2).then_some(val);
    }
    if let Some(idx) = text.rfind('<') {
        let rest = &text[idx + 1..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            let val: u32 = digits.parse().ok()?;
            return (val <= 2).then_some(val);
        }
    }
    None
}

/// Parse an XDG `org.freedesktop.appearance` `contrast` (`uint32 0|1`) from `gdbus call`
/// output.
fn parse_gdbus_contrast(text: &str) -> Option<u32> {
    let idx = text.find("uint32 ")?;
    let digits: String = text[idx + "uint32 ".len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok().filter(|v| *v <= 1)
}

/// Parse an XDG `org.freedesktop.appearance` `accent-color` (`(ddd)`) from `gdbus call`
/// output such as `(<<(0.2, 0.4, 1.0)>>,)`. Returns `Some(None)` for a parsed but
/// out-of-range triple, which the spec defines as "no accent".
fn parse_gdbus_accent_color(text: &str) -> Option<Option<[f64; 3]>> {
    let start = text.match_indices('(').map(|(i, _)| i + 1).find(|&i| {
        text[i..]
            .trim_start()
            .starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '.')
    })?;
    let end = start + text[start..].find(')')?;
    let parts: Vec<f64> = text[start..end]
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    let rgb: [f64; 3] = parts.try_into().ok()?;
    Some(is_valid_accent_color(&rgb).then_some(rgb))
}

fn find_on_path(binary: &str) -> Option<std::path::PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

/// Parse a GNOME `button-layout` string from `gdbus call` (`(<<'a:b,c'>>,)`) or
/// `gsettings get` (`'a:b,c'`) output. Returns `None` unless it looks like a layout.
fn parse_button_layout(text: &str) -> Option<String> {
    let start = text.find('\'')? + 1;
    let len = text[start..].find('\'')?;
    let layout = &text[start..start + len];
    is_valid_button_layout(layout).then(|| layout.to_string())
}

/// GNOME's `org.gnome.desktop.wm.preferences` `button-layout` schema default.
const GNOME_DEFAULT_BUTTON_LAYOUT: &str = "appmenu:close";

/// Whether `XDG_CURRENT_DESKTOP` (a `:`-separated list) names GNOME.
fn is_gnome_desktop(current_desktop: Option<&str>) -> bool {
    current_desktop.is_some_and(|d| d.split(':').any(|e| e.eq_ignore_ascii_case("gnome")))
}

/// Decide whether a locally read layout reflects the user's desktop. Outside GNOME, the
/// portal and `gsettings` report GNOME's schema default when nothing set the key; that
/// value would hide minimize/maximize on the server, so treat it as "no preference" and
/// keep the server default.
fn effective_button_layout(layout: String, gnome: bool) -> Option<String> {
    (gnome || layout != GNOME_DEFAULT_BUTTON_LAYOUT).then_some(layout)
}

/// Run a short-lived command and return its stdout if it exits successfully within 5s.
async fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let fut = tokio::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let output = tokio::time::timeout(Duration::from_secs(5), fut)
        .await
        .ok()?
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn read_portal_setting(namespace: &str, key: &str) -> Option<String> {
    command_stdout(
        "gdbus",
        &[
            "call",
            "--session",
            "--dest",
            "org.freedesktop.portal.Desktop",
            "--object-path",
            "/org/freedesktop/portal/desktop",
            "--method",
            "org.freedesktop.portal.Settings.Read",
            namespace,
            key,
        ],
    )
    .await
}

async fn query_local_color_scheme() -> Option<u32> {
    let stdout = read_portal_setting("org.freedesktop.appearance", "color-scheme").await?;
    parse_gdbus_color_scheme(&stdout)
}

/// Read the local accent color. Returns `None` if it could not be read (including a
/// portal with no accent at all), and `Some(None)` for an explicitly unset accent.
async fn query_local_accent_color() -> Option<Option<[f64; 3]>> {
    let stdout = read_portal_setting("org.freedesktop.appearance", "accent-color").await?;
    parse_gdbus_accent_color(&stdout)
}

async fn query_local_contrast() -> Option<u32> {
    let stdout = read_portal_setting("org.freedesktop.appearance", "contrast").await?;
    parse_gdbus_contrast(&stdout)
}

/// Read the local titlebar button layout from the settings portal, else straight from
/// GSettings. Returns `None` if neither could be read, and `Some(None)` when the desktop
/// has no real preference (see [`effective_button_layout`]).
async fn query_local_button_layout() -> Option<Option<String>> {
    const SCHEMA: &str = "org.gnome.desktop.wm.preferences";
    let layout = match read_portal_setting(SCHEMA, "button-layout")
        .await
        .as_deref()
        .and_then(parse_button_layout)
    {
        Some(layout) => layout,
        None => command_stdout("gsettings", &["get", SCHEMA, "button-layout"])
            .await
            .as_deref()
            .and_then(parse_button_layout)?,
    };
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").ok();
    Some(effective_button_layout(
        layout,
        is_gnome_desktop(desktop.as_deref()),
    ))
}

/// Which synced desktop settings to (re-)query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SettingsMask {
    scheme: bool,
    button_layout: bool,
    accent_color: bool,
    contrast: bool,
}

impl SettingsMask {
    const ALL: Self = Self {
        scheme: true,
        button_layout: true,
        accent_color: true,
        contrast: true,
    };

    fn any(self) -> bool {
        self.scheme || self.button_layout || self.accent_color || self.contrast
    }

    fn union(self, other: Self) -> Self {
        Self {
            scheme: self.scheme || other.scheme,
            button_layout: self.button_layout || other.button_layout,
            accent_color: self.accent_color || other.accent_color,
            contrast: self.contrast || other.contrast,
        }
    }
}

/// Tracks the last desktop settings sent to the server and sends only changes.
#[derive(Default)]
struct DesktopSettingsSync {
    scheme: Option<u32>,
    /// Last layout sent: `None` until the first send, `Some(None)` for "no preference".
    button_layout: Option<Option<String>>,
    /// Last accent sent: `None` until the first send, `Some(None)` for "no accent".
    accent_color: Option<Option<[f64; 3]>>,
    contrast: Option<u32>,
}

impl DesktopSettingsSync {
    /// Send `scheme` if it changed. Returns `false` once the receiver is gone.
    fn send_scheme(&mut self, tx: &mpsc::UnboundedSender<ControlMessage>, scheme: u32) -> bool {
        if self.scheme == Some(scheme) {
            return true;
        }
        info!(scheme, "syncing local color-scheme to server");
        self.scheme = Some(scheme);
        tx.send(ControlMessage::ColorScheme { scheme }).is_ok()
    }

    /// Send `layout` (`None` = no preference) if it changed. Returns `false` once the
    /// receiver is gone.
    fn send_button_layout(
        &mut self,
        tx: &mpsc::UnboundedSender<ControlMessage>,
        layout: Option<String>,
    ) -> bool {
        if self.button_layout.as_ref() == Some(&layout) {
            return true;
        }
        match &layout {
            Some(layout) => info!(%layout, "syncing local titlebar button layout to server"),
            None => info!("no local titlebar button layout preference; using server default"),
        }
        self.button_layout = Some(layout.clone());
        tx.send(ControlMessage::ButtonLayout { layout }).is_ok()
    }

    /// Send `rgb` (`None` = no accent) if it changed. Returns `false` once the receiver
    /// is gone.
    fn send_accent_color(
        &mut self,
        tx: &mpsc::UnboundedSender<ControlMessage>,
        rgb: Option<[f64; 3]>,
    ) -> bool {
        if self.accent_color == Some(rgb) {
            return true;
        }
        info!(?rgb, "syncing local accent-color to server");
        self.accent_color = Some(rgb);
        tx.send(ControlMessage::AccentColor { rgb }).is_ok()
    }

    /// Send `contrast` if it changed. Returns `false` once the receiver is gone.
    fn send_contrast(&mut self, tx: &mpsc::UnboundedSender<ControlMessage>, contrast: u32) -> bool {
        if self.contrast == Some(contrast) {
            return true;
        }
        info!(contrast, "syncing local contrast to server");
        self.contrast = Some(contrast);
        tx.send(ControlMessage::Contrast { contrast }).is_ok()
    }

    /// Re-query the settings in `mask` (concurrently) and send those that changed.
    /// On the `initial` sync an unreadable button layout, accent or contrast counts as
    /// "no preference", so the server drops any value a previous client synced. Returns
    /// `false` once the receiver is gone.
    async fn sync(
        &mut self,
        tx: &mpsc::UnboundedSender<ControlMessage>,
        mask: SettingsMask,
        initial: bool,
    ) -> bool {
        let (new_scheme, new_layout, new_accent, new_contrast) = tokio::join!(
            async {
                if mask.scheme {
                    query_local_color_scheme().await
                } else {
                    None
                }
            },
            async {
                if mask.button_layout {
                    query_local_button_layout().await
                } else {
                    None
                }
            },
            async {
                if mask.accent_color {
                    query_local_accent_color().await
                } else {
                    None
                }
            },
            async {
                if mask.contrast {
                    query_local_contrast().await
                } else {
                    None
                }
            },
        );
        if let Some(scheme) = new_scheme {
            if !self.send_scheme(tx, scheme) {
                return false;
            }
        }
        let new_layout = if initial && mask.button_layout {
            Some(new_layout.flatten())
        } else {
            new_layout
        };
        if let Some(layout) = new_layout {
            if !self.send_button_layout(tx, layout) {
                return false;
            }
        }
        let new_accent = if initial && mask.accent_color {
            Some(new_accent.flatten())
        } else {
            new_accent
        };
        if let Some(rgb) = new_accent {
            if !self.send_accent_color(tx, rgb) {
                return false;
            }
        }
        let new_contrast = if initial && mask.contrast {
            Some(new_contrast.unwrap_or(0))
        } else {
            new_contrast
        };
        if let Some(contrast) = new_contrast {
            if !self.send_contrast(tx, contrast) {
                return false;
            }
        }
        true
    }
}

/// Which settings a line of `dbus-monitor` / `gdbus monitor` output says changed.
fn monitor_line_triggers(line: &str, using_dbus_monitor: bool) -> SettingsMask {
    let appearance = |key: &str| {
        if using_dbus_monitor {
            // dbus-monitor prints signal arguments on separate lines.
            line.contains(key)
        } else {
            line.contains("org.freedesktop.appearance") && line.contains(key)
        }
    };
    // A KDE palette change can switch both the color scheme and the accent.
    let kde_palette = using_dbus_monitor && line.contains("member=notifyChange");
    SettingsMask {
        scheme: kde_palette || appearance("color-scheme"),
        button_layout: line.contains("button-layout"),
        accent_color: kde_palette || appearance("accent-color"),
        contrast: appearance("'contrast'") || appearance("\"contrast\""),
    }
}

fn spawn_settings_monitor() -> Option<(tokio::process::Child, bool)> {
    use tokio::process::Command;

    // Watch for live setting changes on the local session D-Bus:
    // - `org.freedesktop.portal.Settings.SettingChanged` (GNOME / KDE / standard XDG portal)
    // - `org.kde.KGlobalSettings.notifyChange` (KDE Plasma palette/color-scheme switches)
    let dbus_monitor = find_on_path("dbus-monitor");
    let stdbuf = find_on_path("stdbuf");
    let using_dbus_monitor = dbus_monitor.is_some();

    let mut cmd = if let Some(ref mon_bin) = dbus_monitor {
        let args = [
            "--session",
            "type='signal',interface='org.freedesktop.portal.Settings',member='SettingChanged'",
            "type='signal',interface='org.kde.KGlobalSettings',member='notifyChange'",
        ];
        if let Some(ref sb) = stdbuf {
            let mut c = Command::new(sb);
            c.arg("-oL").arg(mon_bin).args(args);
            c
        } else {
            let mut c = Command::new(mon_bin);
            c.args(args);
            c
        }
    } else {
        let args = [
            "monitor",
            "--session",
            "--dest",
            "org.freedesktop.portal.Desktop",
            "--object-path",
            "/org/freedesktop/portal/desktop",
        ];
        if let Some(ref sb) = stdbuf {
            let mut c = Command::new(sb);
            c.args(["-oL", "gdbus"]).args(args);
            c
        } else {
            let mut c = Command::new("gdbus");
            c.args(args);
            c
        }
    };

    // `kill_on_drop` only helps when the `Child` is dropped. A client that is killed or
    // crashes drops nothing, and the monitor runs on forever: hundreds of them piled up
    // during testing, and inside the Flatpak sandbox each one also kept `bwrap` alive.
    // Have the kernel kill it when the client goes. The death signal follows the spawning
    // thread, a tokio worker here, which lives until the process exits; it survives the
    // `stdbuf` exec.
    let parent = unsafe { libc::getpid() };
    // SAFETY: pre_exec runs in the forked child before exec; prctl and getppid are
    // async-signal-safe, and no error path allocates (another thread may have held the
    // malloc lock at fork time).
    unsafe {
        cmd.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            // The client may have died before the death signal was set.
            if libc::getppid() != parent {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }

    match cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => Some((child, using_dbus_monitor)),
        Err(err) => {
            warn!(error = %err, "local desktop-settings watcher unavailable");
            None
        }
    }
}

async fn watch_local_desktop_settings(tx: mpsc::UnboundedSender<ControlMessage>) {
    use tokio::io::{AsyncBufReadExt, BufReader};

    // 1. Start the change monitor before the initial query so a change made while the
    //    query runs is still seen (its line is buffered in the pipe until we read it).
    let monitor = spawn_settings_monitor();

    // 2. Query and send the initial settings.
    let mut synced = DesktopSettingsSync::default();
    if !synced.sync(&tx, SettingsMask::ALL, true).await {
        return;
    }

    let Some((mut child, using_dbus_monitor)) = monitor else {
        return;
    };
    let Some(stdout) = child.stdout.take() else {
        return;
    };
    let mut lines = BufReader::new(stdout).lines();

    // Trailing-edge debounce: wait 80ms after the last trigger line so KDE/portal
    // finishes updating its cached settings before we re-query.
    let mut pending = SettingsMask::default();
    let debounce = tokio::time::sleep(Duration::from_secs(3600));
    tokio::pin!(debounce);

    loop {
        tokio::select! {
            line_res = lines.next_line() => {
                match line_res {
                    Ok(Some(line)) => {
                        let triggered = monitor_line_triggers(&line, using_dbus_monitor);
                        if triggered.any() {
                            pending = pending.union(triggered);
                            debounce
                                .as_mut()
                                .reset(tokio::time::Instant::now() + Duration::from_millis(80));
                        }
                    }
                    Ok(None) => {
                        tracing::debug!("local desktop-settings watcher stream closed");
                        break;
                    }
                    Err(err) => {
                        warn!(error = %err, "local desktop-settings watcher read error");
                        break;
                    }
                }
            }
            () = &mut debounce, if pending.any() => {
                let mask = std::mem::take(&mut pending);
                if !synced.sync(&tx, mask, false).await {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gdbus_color_scheme_extracts_valid_values() {
        assert_eq!(parse_gdbus_color_scheme("(<<uint32 1>>,)"), Some(1));
        assert_eq!(
            parse_gdbus_color_scheme(
                "/org/freedesktop/portal/desktop: org.freedesktop.portal.Settings.SettingChanged ('org.freedesktop.appearance', 'color-scheme', <uint32 2>)"
            ),
            Some(2)
        );
        assert_eq!(parse_gdbus_color_scheme("(<<uint32 0>>,)"), Some(0));
        assert_eq!(parse_gdbus_color_scheme("(<1>,)"), Some(1));
        assert_eq!(parse_gdbus_color_scheme("(<<2>>,)"), Some(2));
        assert_eq!(parse_gdbus_color_scheme("(<<uint32 9>>,)"), None);
        assert_eq!(parse_gdbus_color_scheme("unrelated"), None);
    }

    #[test]
    fn parse_button_layout_accepts_portal_and_gsettings_output() {
        assert_eq!(
            parse_button_layout("(<<'appmenu:minimize,maximize,close'>>,)\n").as_deref(),
            Some("appmenu:minimize,maximize,close")
        );
        assert_eq!(
            parse_button_layout("'icon:minimize,maximize,close'\n").as_deref(),
            Some("icon:minimize,maximize,close")
        );
        assert_eq!(
            parse_button_layout("'close,minimize:'").as_deref(),
            Some("close,minimize:")
        );
        assert_eq!(parse_button_layout("(<<''>>,)"), None);
        assert_eq!(parse_button_layout("'evil\ninjection:close'"), None);
        assert_eq!(parse_button_layout("unrelated"), None);
    }

    #[test]
    fn effective_button_layout_drops_gnome_default_outside_gnome() {
        assert!(is_gnome_desktop(Some("GNOME")));
        assert!(is_gnome_desktop(Some("ubuntu:GNOME")));
        assert!(!is_gnome_desktop(Some("KDE")));
        assert!(!is_gnome_desktop(Some("sway")));
        assert!(!is_gnome_desktop(None));

        let default = || GNOME_DEFAULT_BUTTON_LAYOUT.to_string();
        assert_eq!(effective_button_layout(default(), true), Some(default()));
        assert_eq!(effective_button_layout(default(), false), None);
        assert_eq!(
            effective_button_layout("icon:minimize,maximize,close".into(), false).as_deref(),
            Some("icon:minimize,maximize,close")
        );
    }

    #[test]
    fn monitor_line_triggers_only_the_changed_setting() {
        let mask = |scheme, button_layout, accent_color, contrast| SettingsMask {
            scheme,
            button_layout,
            accent_color,
            contrast,
        };
        // dbus-monitor: arguments arrive on their own lines.
        assert_eq!(
            monitor_line_triggers("   string \"color-scheme\"", true),
            mask(true, false, false, false)
        );
        assert_eq!(
            monitor_line_triggers("   string \"button-layout\"", true),
            mask(false, true, false, false)
        );
        assert_eq!(
            monitor_line_triggers("   string \"accent-color\"", true),
            mask(false, false, true, false)
        );
        assert_eq!(
            monitor_line_triggers("   string \"contrast\"", true),
            mask(false, false, false, true)
        );
        assert_eq!(
            monitor_line_triggers(
                "signal time=1 sender=:1.5 -> destination=(null destination) path=/KGlobalSettings; interface=org.kde.KGlobalSettings; member=notifyChange",
                true
            ),
            mask(true, false, true, false)
        );
        assert_eq!(
            monitor_line_triggers("   string \"gtk-theme\"", true),
            SettingsMask::default()
        );
        // gdbus monitor: one line per signal.
        assert_eq!(
            monitor_line_triggers(
                "/org/freedesktop/portal/desktop: org.freedesktop.portal.Settings.SettingChanged ('org.gnome.desktop.wm.preferences', 'button-layout', <'close:'>)",
                false
            ),
            mask(false, true, false, false)
        );
        assert_eq!(
            monitor_line_triggers(
                "/org/freedesktop/portal/desktop: org.freedesktop.portal.Settings.SettingChanged ('org.freedesktop.appearance', 'color-scheme', <uint32 1>)",
                false
            ),
            mask(true, false, false, false)
        );
        assert_eq!(
            monitor_line_triggers(
                "/org/freedesktop/portal/desktop: org.freedesktop.portal.Settings.SettingChanged ('org.freedesktop.appearance', 'accent-color', <(0.2, 0.4, 1.0)>)",
                false
            ),
            mask(false, false, true, false)
        );
        assert_eq!(
            monitor_line_triggers(
                "/org/freedesktop/portal/desktop: org.freedesktop.portal.Settings.SettingChanged ('org.freedesktop.appearance', 'contrast', <uint32 1>)",
                false
            ),
            mask(false, false, false, true)
        );
    }

    #[test]
    fn parses_gdbus_accent_color_and_contrast() {
        assert_eq!(
            parse_gdbus_accent_color("(<<(0.25, 0.5, 1.0)>>,)"),
            Some(Some([0.25, 0.5, 1.0]))
        );
        assert_eq!(
            parse_gdbus_accent_color("(<(0.2078431, 0.5176471, 0.8941177)>,)"),
            Some(Some([0.2078431, 0.5176471, 0.8941177]))
        );
        // Out-of-range components mean "no accent".
        assert_eq!(
            parse_gdbus_accent_color("(<<(-1.0, -1.0, -1.0)>>,)"),
            Some(None)
        );
        assert_eq!(parse_gdbus_accent_color("(<<uint32 1>>,)"), None);
        assert_eq!(parse_gdbus_accent_color("unrelated"), None);
        assert_eq!(parse_gdbus_contrast("(<<uint32 1>>,)"), Some(1));
        assert_eq!(parse_gdbus_contrast("(<<uint32 0>>,)"), Some(0));
        assert_eq!(parse_gdbus_contrast("(<<uint32 7>>,)"), None);
    }

    /// Run by `settings_monitor_dies_with_the_client` in a child process, which it then
    /// kills. `PATH` holds only a fake `dbus-monitor` that sleeps, so no D-Bus is needed.
    #[tokio::test]
    #[ignore = "helper for settings_monitor_dies_with_the_client"]
    async fn settings_monitor_pdeathsig_helper() {
        // Does nothing under `--include-ignored`; only the parent test sets this.
        let Some(dir) = std::env::var_os("ORIEL_PDEATHSIG_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let (child, using_dbus_monitor) = spawn_settings_monitor().unwrap();
        assert!(using_dbus_monitor);
        std::fs::write(dir.join("monitor.pid"), child.id().unwrap().to_string()).unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        drop(child);
    }

    #[test]
    fn settings_monitor_dies_with_the_client() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::{Duration, Instant};

        let dir =
            std::env::temp_dir().join(format!("oriel-monitor-pdeathsig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sleep = find_on_path("sleep").expect("sleep not on PATH");
        let fake = dir.join("dbus-monitor");
        std::fs::write(&fake, format!("#!/bin/sh\nexec {} 60\n", sleep.display())).unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let pid_file = dir.join("monitor.pid");

        let mut helper = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "quic::tests::settings_monitor_pdeathsig_helper",
            ])
            .env("ORIEL_PDEATHSIG_DIR", &dir)
            .env("PATH", &dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let pid = loop {
            if let Some(pid) = std::fs::read_to_string(&pid_file)
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
            {
                break pid;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "monitor never started"
            );
            std::thread::sleep(Duration::from_millis(20));
        };
        let gone = |pid: i32| {
            std::fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
                stat.rsplit(')')
                    .next()
                    .is_some_and(|s| s.trim_start().starts_with('Z'))
            })
        };
        // Otherwise the test would pass without the death signal doing anything.
        assert!(!gone(pid), "fake monitor {pid} was not running");

        // Like a crash or SIGKILL: no destructors run, so `kill_on_drop` can't help.
        helper.kill().unwrap();
        helper.wait().unwrap();

        let started = Instant::now();
        while !gone(pid) && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let died = gone(pid);
        if !died {
            // Still ours: it never died, so its PID can't have been reused.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(died, "settings monitor {pid} outlived the client");
    }
}
