//! QUIC client transport with SHA-256 certificate fingerprint pinning.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use quinn::{ClientConfig, Endpoint, TransportConfig};
use remotewayland_proto::{
    read_audio_packet, read_control_message, read_stream_header, read_video_frame,
    write_audio_packet, write_control_message, write_stream_header, AudioPacketHeader,
    ControlMessage, RemoteAppEntry, StreamHeader, VideoCodec, WindowId, PROTOCOL_VERSION,
};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as RustlsError, SignatureScheme};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::audio::{start_mic_capture, AudioPlaybackSink};
use crate::media::VideoDecoder;
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
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
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
            supported_codecs: vec![VideoCodec::Lz4Delta],
            enable_audio_bridge,
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

/// Connect to the RemoteWayland server, verify its TLS fingerprint, and run the session loops.
pub async fn run_client(
    server_addr: SocketAddr,
    fingerprint_sha256: String,
    token: String,
    enable_audio_bridge: bool,
    ui_tx: UiSender,
    mut ctrl_rx: mpsc::UnboundedReceiver<ControlMessage>,
    app_tx: Option<mpsc::UnboundedSender<Vec<RemoteAppEntry>>>,
) -> Result<()> {
    let (conn, mut send, mut recv) =
        connect_and_auth(server_addr, fingerprint_sha256, token, enable_audio_bridge).await?;

    // Task: forward local winit input/resize/close messages to server control stream.
    let writer_task = tokio::spawn(async move {
        while let Some(msg) = ctrl_rx.recv().await {
            if let Err(err) = write_control_message(&mut send, &msg).await {
                warn!(error = %err, "failed to send control message");
                break;
            }
        }
    });

    // Task: capture client microphone and send Opus packets to server.
    let mic_conn = conn.clone();
    let mic_task = tokio::spawn(async move {
        match start_mic_capture() {
            Ok(mut rx) => match mic_conn.open_uni().await {
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
                        if let Err(err) = write_audio_packet(&mut stream, &hdr, &pkt.data).await {
                            warn!(error = %err, "AudioCapture stream write error");
                            break;
                        }
                    }
                }
                Err(err) => warn!(error = %err, "failed to open AudioCapture uni stream"),
            },
            Err(err) => warn!(error = %err, "client mic capture unavailable"),
        }
    });

    // Task: accept per-window video and audio playback unidirectional streams.
    let uni_conn = conn.clone();
    let uni_tx = ui_tx.clone();
    let uni_task = tokio::spawn(async move {
        while let Ok(mut recv_uni) = uni_conn.accept_uni().await {
            let tx = uni_tx.clone();
            tokio::spawn(async move {
                match read_stream_header(&mut recv_uni).await {
                    Ok(StreamHeader::Video {
                        window_id,
                        codec: _,
                    }) => {
                        let mut decoder = VideoDecoder::new();
                        while let Ok(Some((hdr, payload))) = read_video_frame(&mut recv_uni).await {
                            match decoder.decode(hdr.width, hdr.height, hdr.is_keyframe, &payload) {
                                Ok(frame) => {
                                    let _ = tx.send(ClientUiEvent::FrameDecoded {
                                        window_id,
                                        width: frame.width,
                                        height: frame.height,
                                        bgra: frame.bgra,
                                    });
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
    while let Some(msg) = read_control_message(&mut recv).await? {
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
                is_keyframe,
                bgra,
            } => {
                let decoder = popup_decoders
                    .entry(window_id)
                    .or_insert_with(VideoDecoder::new);
                match decoder.decode(width, height, is_keyframe, &bgra) {
                    Ok(frame) => {
                        let _ = ui_tx.send(ClientUiEvent::PopupFrame {
                            window_id,
                            width: frame.width,
                            height: frame.height,
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
            ControlMessage::ClipboardText { text } => {
                let _ = ui_tx.send(ClientUiEvent::ClipboardText { text });
            }
            ControlMessage::AppList { apps } => {
                if let Some(tx) = &app_tx {
                    let _ = tx.send(apps);
                }
            }
            other => {
                info!(?other, "received control event from server");
            }
        }
    }

    let _ = ui_tx.send(ClientUiEvent::Disconnected);
    writer_task.abort();
    mic_task.abort();
    uni_task.abort();
    Ok(())
}
