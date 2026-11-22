//! QUIC client transport with SHA-256 certificate fingerprint pinning.

use std::fs::{self, Permissions};
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use quinn::{ClientConfig, Endpoint, TransportConfig};
use remotewayland_proto::{
    read_audio_packet, read_control_message, read_stream_header, read_video_frame,
    write_audio_packet, write_control_message, write_stream_header, AudioPacketHeader,
    ControlMessage, LaunchState, RemoteAppEntry, StreamHeader, VideoCodec, WindowId,
    PROTOCOL_VERSION,
};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as RustlsError, SignatureScheme};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::audio::{start_mic_capture, AudioPlaybackSink};
use crate::media::{VideoDecoder, WindowVideoDecoder};
use crate::window::{ClientUiEvent, UiSender};

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

/// Establish a pinned QUIC connection and perform the `AuthRequest`/`AuthResponse` handshake.
pub async fn connect_and_auth(
    server_addr: SocketAddr,
    fingerprint_sha256: String,
    token: String,
    enable_audio_bridge: bool,
    initial_scale: f64,
) -> Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    let verifier = Arc::new(PinnedCertVerifier::new(fingerprint_sha256));
    let mut rustls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    rustls_config.alpn_protocols = vec![b"remotewayland/1".to_vec()];

    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(rustls_config)
        .context("failed to create QUIC client crypto config")?;
    let mut client_config = ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(3)));
    if let Ok(idle) = Duration::from_secs(10).try_into() {
        transport.max_idle_timeout(Some(idle));
    }
    client_config.transport_config(Arc::new(transport));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
    endpoint.set_default_client_config(client_config);

    let conn = endpoint
        .connect(server_addr, "remotewayland.local")?
        .await
        .with_context(|| format!("failed to connect to {server_addr}"))?;
    info!(%server_addr, "QUIC connection established");

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .context("failed to open control stream")?;

    write_control_message(
        &mut send,
        &ControlMessage::AuthRequest {
            protocol_version: PROTOCOL_VERSION,
            token,
            supported_codecs: vec![VideoCodec::SmartTile, VideoCodec::Lz4Delta],
            enable_audio_bridge,
            initial_scale,
        },
    )
    .await?;

    match read_control_message(&mut recv).await? {
        Some(ControlMessage::AuthResponse {
            accepted: true,
            selected_codec,
            ..
        }) => {
            info!(?selected_codec, "authenticated with server");
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
    }

    Ok((conn, send, recv))
}

/// Events sent from the QUIC client to the launcher UI.
#[derive(Debug, Clone)]
#[cfg_attr(not(any(feature = "gtk4", feature = "qt")), allow(dead_code))]
pub enum LauncherNetEvent {
    AppList(Vec<RemoteAppEntry>),
    AppState {
        launch_id: u64,
        name: String,
        state: LaunchState,
    },
    NetStats {
        rtt_ms: u32,
        congested: bool,
    },
    Disconnected(Option<String>),
}

/// Connect to the RemoteWayland server, verify its TLS fingerprint, and run the session loops.
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
    let (conn, mut send, mut recv) = connect_and_auth(
        server_addr,
        fingerprint_sha256,
        token,
        enable_audio_bridge,
        initial_scale,
    )
    .await?;

    // Optional local 0600 Unix socket that tunnels `remotewayland-client pty-attach --socket <path>`
    // directly over multiplexed QUIC bidirectional streams on this authenticated connection.
    let pty_tunnel_task = if let Some(sock_path) = pty_socket_path.clone() {
        let _ = fs::remove_file(&sock_path);
        match tokio::net::UnixListener::bind(&sock_path) {
            Ok(listener) => {
                if let Err(err) = fs::set_permissions(&sock_path, Permissions::from_mode(0o600)) {
                    warn!(path = %sock_path.display(), error = %err, "failed to chmod 0600 PTY tunnel socket");
                }
                info!(path = %sock_path.display(), "local PTY Unix tunnel listening (mode 0600)");
                let tunnel_conn = conn.clone();
                Some(tokio::spawn(async move {
                    while let Ok((unix_stream, _)) = listener.accept().await {
                        let c = tunnel_conn.clone();
                        tokio::spawn(async move {
                            let Ok((mut quic_send, mut quic_recv)) = c.open_bi().await else {
                                warn!("failed to open QUIC bidi stream for tunneled PTY");
                                return;
                            };
                            let (mut unix_read, mut unix_write) = unix_stream.into_split();
                            let up = tokio::spawn(async move {
                                let _ = tokio::io::copy(&mut unix_read, &mut quic_send).await;
                                let _ = quic_send.finish();
                            });
                            let down = tokio::spawn(async move {
                                let _ = tokio::io::copy(&mut quic_recv, &mut unix_write).await;
                            });
                            let _ = tokio::join!(up, down);
                        });
                    }
                }))
            }
            Err(err) => {
                warn!(path = %sock_path.display(), error = %err, "failed to bind local PTY tunnel socket");
                None
            }
        }
    } else {
        None
    };

    let stats_task = launcher_tx.as_ref().map(|tx| {
        let stats_conn = conn.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut prev_lost = 0u64;
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let rtt_ms = stats_conn.rtt().as_millis() as u32;
                let lost = stats_conn.stats().path.lost_packets;
                let loss_delta = lost.saturating_sub(prev_lost);
                prev_lost = lost;
                let congested = rtt_ms > 150 || loss_delta > 5;
                if tx
                    .send(LauncherNetEvent::NetStats { rtt_ms, congested })
                    .is_err()
                {
                    break;
                }
            }
        })
    });

    // Task: query and monitor local XDG portal `color-scheme` and forward changes to the server.
    let (theme_tx, mut theme_rx) = mpsc::unbounded_channel::<u32>();
    let color_scheme_task = tokio::spawn(watch_local_color_scheme(theme_tx));

    // Per-window decoded video frame acknowledgments for end-to-end in-flight frame pacing.
    let (frame_ack_tx, mut frame_ack_rx) = mpsc::unbounded_channel::<(WindowId, u64)>();

    // Task: forward local input/resize/close/theme/frame-ack messages to server control stream.
    let (ctrl_closed_tx, mut ctrl_closed_rx) = tokio::sync::oneshot::channel::<()>();
    let writer_task = tokio::spawn(async move {
        loop {
            let msg = tokio::select! {
                m = ctrl_rx.recv() => match m {
                    Some(m) => m,
                    None => break,
                },
                Some(scheme) = theme_rx.recv() => ControlMessage::ColorScheme { scheme },
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
    let uni_conn = conn.clone();
    let uni_tx = ui_tx.clone();
    let uni_ack_tx = frame_ack_tx.clone();
    let uni_task = tokio::spawn(async move {
        while let Ok(mut recv_uni) = uni_conn.accept_uni().await {
            let tx = uni_tx.clone();
            let ack_tx = uni_ack_tx.clone();
            tokio::spawn(async move {
                match read_stream_header(&mut recv_uni).await {
                    Ok(StreamHeader::Video { window_id, codec }) => {
                        let mut decoder = WindowVideoDecoder::new(codec);
                        while let Ok(Some((hdr, payload))) = read_video_frame(&mut recv_uni).await {
                            match decoder.decode(hdr.width, hdr.height, hdr.is_keyframe, &payload) {
                                Ok(frame) => {
                                    tx.push_frame(
                                        window_id,
                                        frame.width,
                                        frame.height,
                                        hdr.logical_width,
                                        hdr.logical_height,
                                        frame.bgra,
                                    );
                                    let _ = ack_tx.send((window_id, hdr.pts_ns));
                                }
                                Err(err) => warn!(window_id, error = %err, "video decode error"),
                            }
                        }
                    }
                    Ok(StreamHeader::AudioPlayback) => match AudioPlaybackSink::new() {
                        Ok(sink) => {
                            while let Ok(Some((hdr, payload))) =
                                read_audio_packet(&mut recv_uni).await
                            {
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
                    }) => match AudioPlaybackSink::new_for_app(
                        &app_name,
                        &media_name,
                        &icon_name,
                        &media_role,
                    ) {
                        Ok(sink) => {
                            info!(stream_id, %app_name, %media_name, "opened per-app host audio stream");
                            while let Ok(Some((hdr, payload))) =
                                read_audio_packet(&mut recv_uni).await
                            {
                                if let Err(err) = sink.push_audio(hdr.pts_ns, &payload) {
                                    warn!(stream_id, error = %err, "audio playback push error");
                                    break;
                                }
                            }
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
            });
        }
    });

    // Read server control events and dispatch to Wayland UI thread.
    let mut popup_decoders: std::collections::HashMap<WindowId, VideoDecoder> =
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

    loop {
        let msg = tokio::select! {
            _ = &mut ctrl_closed_rx => break,
            res = read_control_message(&mut recv) => match res? {
                Some(m) => m,
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
                is_keyframe,
                bgra,
            } => {
                let decoder = popup_decoders
                    .entry(window_id)
                    .or_default();
                match decoder.decode(width, height, is_keyframe, &bgra) {
                    Ok(frame) => {
                        let _ = ui_tx.send(ClientUiEvent::PopupFrame {
                            window_id,
                            width: frame.width,
                            height: frame.height,
                            logical_width,
                            logical_height,
                            bgra: frame.bgra,
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
            ControlMessage::ClipboardText { text } => {
                let _ = ui_tx.send(ClientUiEvent::ClipboardText { text });
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
    writer_task.abort();
    if let Some((handle, task)) = active_mic.take() {
        task.abort();
        drop(handle);
    }
    uni_task.abort();
    Ok(())
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

fn find_on_path(binary: &str) -> Option<std::path::PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

async fn query_local_color_scheme() -> Option<u32> {
    use tokio::process::Command;

    let fut = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.freedesktop.portal.Desktop",
            "--object-path",
            "/org/freedesktop/portal/desktop",
            "--method",
            "org.freedesktop.portal.Settings.Read",
            "org.freedesktop.appearance",
            "color-scheme",
        ])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();

    let output = tokio::time::timeout(Duration::from_secs(5), fut)
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_gdbus_color_scheme(&stdout)
}

async fn watch_local_color_scheme(tx: mpsc::UnboundedSender<u32>) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    // 1. Query initial color-scheme from the local session portal.
    let mut last_sent_scheme: Option<u32> = None;
    if let Some(scheme) = query_local_color_scheme().await {
        info!(scheme, "syncing initial local color-scheme to server");
        if tx.send(scheme).is_err() {
            return;
        }
        last_sent_scheme = Some(scheme);
    }

    // 2. Watch for live theme/palette changes on the local session D-Bus:
    //    - `org.freedesktop.portal.Settings.SettingChanged` (GNOME / KDE / standard XDG portal)
    //    - `org.kde.KGlobalSettings.notifyChange` (KDE Plasma palette/color-scheme switches)
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

    let mut child = match cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(err) => {
            warn!(error = %err, "local color-scheme watcher unavailable");
            return;
        }
    };

    let Some(stdout) = child.stdout.take() else {
        return;
    };
    let mut lines = BufReader::new(stdout).lines();

    // Trailing-edge debounce: wait 80ms after the last trigger line so KDE/portal
    // finishes updating its cached color-scheme before we re-query.
    let mut pending_sync = false;
    let debounce = tokio::time::sleep(Duration::from_secs(3600));
    tokio::pin!(debounce);

    loop {
        tokio::select! {
            line_res = lines.next_line() => {
                match line_res {
                    Ok(Some(line)) => {
                        let triggered = if using_dbus_monitor {
                            line.contains("member=notifyChange") || line.contains("color-scheme")
                        } else {
                            line.contains("org.freedesktop.appearance")
                                && line.contains("color-scheme")
                        };
                        if triggered {
                            pending_sync = true;
                            debounce
                                .as_mut()
                                .reset(tokio::time::Instant::now() + Duration::from_millis(80));
                        }
                    }
                    Ok(None) => {
                        tracing::debug!("local color-scheme watcher stream closed");
                        break;
                    }
                    Err(err) => {
                        warn!(error = %err, "local color-scheme watcher read error");
                        break;
                    }
                }
            }
            () = &mut debounce, if pending_sync => {
                pending_sync = false;
                if let Some(scheme) = query_local_color_scheme().await {
                    if last_sent_scheme != Some(scheme) {
                        last_sent_scheme = Some(scheme);
                        info!(scheme, "local color-scheme/palette changed; syncing to server");
                        if tx.send(scheme).is_err() {
                            break;
                        }
                    }
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
}
