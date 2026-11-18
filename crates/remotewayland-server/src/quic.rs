//! QUIC server transport and ephemeral TLS/token authentication.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use quinn::{Endpoint, SendStream, ServerConfig, TransportConfig};
use remotewayland_proto::{
    read_audio_packet, read_control_message, read_stream_header, write_audio_packet,
    write_control_message, write_stream_header, write_video_frame, AudioPacketHeader,
    ControlMessage, StreamHeader, VideoCodec, VideoFrameHeader, WindowId, PROTOCOL_VERSION,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use crate::audio::{start_playback_capture, AudioMicInjector};
use crate::compositor::{CompositorCommand, CompositorEvent, CompositorHandle};
use crate::media::VideoEncoder;

/// Holds the ephemeral TLS certificate fingerprint and session token.
pub struct EphemeralCredentials {
    pub server_config: ServerConfig,
    pub cert_sha256_hex: String,
    pub token: String,
}

/// Generate a self-signed TLS 1.3 certificate and ephemeral session token.
pub fn generate_ephemeral_credentials(token: String) -> Result<EphemeralCredentials> {
    let cert = rcgen::generate_simple_self_signed(vec!["remotewayland.local".to_string()])
        .context("failed to generate self-signed TLS certificate")?;
    let cert_der = CertificateDer::from(cert.cert);
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));

    let digest = sha256_bytes(cert_der.as_ref());
    let cert_sha256_hex = hex_encode(&digest);

    let mut rustls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .context("failed to build rustls ServerConfig")?;
    rustls_config.alpn_protocols = vec![b"remotewayland/1".to_vec()];

    let quic_crypto = quinn::crypto::rustls::QuicServerConfig::try_from(rustls_config)
        .context("failed to create QUIC server crypto config")?;
    let mut server_config = ServerConfig::with_crypto(Arc::new(quic_crypto));
    let mut transport = TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    server_config.transport_config(Arc::new(transport));

    Ok(EphemeralCredentials {
        server_config,
        cert_sha256_hex,
        token,
    })
}

/// Bind the QUIC endpoint and handle incoming authenticated client sessions.
pub async fn run_server(
    listen_addr: SocketAddr,
    creds: EphemeralCredentials,
    comp_handle: CompositorHandle,
    mut event_rx: mpsc::UnboundedReceiver<CompositorEvent>,
) -> Result<()> {
    let endpoint = Endpoint::server(creds.server_config, listen_addr)
        .with_context(|| format!("failed to bind QUIC endpoint on {listen_addr}"))?;

    info!(%listen_addr, cert_sha256 = %creds.cert_sha256_hex, "RemoteWayland server listening");
    println!(
        "remotewayland://{}?fp={}&token={}",
        listen_addr, creds.cert_sha256_hex, creds.token
    );

    let (bcast_tx, _) = broadcast::channel::<CompositorEvent>(256);
    let bcast_tx_clone = bcast_tx.clone();
    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            let _ = bcast_tx_clone.send(ev);
        }
    });

    let expected_token = Arc::new(creds.token);
    while let Some(incoming) = endpoint.accept().await {
        let token = Arc::clone(&expected_token);
        let cmd_tx = comp_handle.cmd_tx.clone();
        let ev_rx = bcast_tx.subscribe();
        tokio::spawn(async move {
            if let Err(err) = handle_connection(incoming, &token, cmd_tx, ev_rx).await {
                warn!(error = %err, "client connection ended with error");
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    incoming: quinn::Incoming,
    expected_token: &str,
    cmd_tx: calloop::channel::Sender<CompositorCommand>,
    mut ev_rx: broadcast::Receiver<CompositorEvent>,
) -> Result<()> {
    let conn = incoming.await.context("QUIC handshake failed")?;
    let remote = conn.remote_address();
    info!(%remote, "QUIC peer connected");

    let (mut send, mut recv) = conn
        .accept_bi()
        .await
        .context("failed to accept control stream")?;

    let auth_msg = read_control_message(&mut recv)
        .await?
        .context("control stream closed before AuthRequest")?;

    let (selected_codec, enable_audio_bridge) = match auth_msg {
        ControlMessage::AuthRequest {
            protocol_version,
            token,
            supported_codecs,
            enable_audio_bridge,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                write_control_message(
                    &mut send,
                    &ControlMessage::AuthResponse {
                        accepted: false,
                        reason: Some(format!("unsupported protocol version {protocol_version}")),
                        selected_codec: VideoCodec::Lz4Delta,
                    },
                )
                .await?;
                return Ok(());
            }

            if token != expected_token {
                write_control_message(
                    &mut send,
                    &ControlMessage::AuthResponse {
                        accepted: false,
                        reason: Some("invalid session token".into()),
                        selected_codec: VideoCodec::Lz4Delta,
                    },
                )
                .await?;
                return Ok(());
            }

            let selected_codec = supported_codecs
                .first()
                .copied()
                .unwrap_or(VideoCodec::Lz4Delta);

            write_control_message(
                &mut send,
                &ControlMessage::AuthResponse {
                    accepted: true,
                    reason: None,
                    selected_codec,
                },
            )
            .await?;

            info!(
                %remote,
                ?selected_codec,
                enable_audio_bridge,
                "client authenticated; starting compositor/media loop"
            );
            let _ = cmd_tx.send(CompositorCommand::SyncWindows);
            (selected_codec, enable_audio_bridge)
        }
        other => {
            warn!(?other, "unexpected initial control message");
            return Ok(());
        }
    };

    // Start bidirectional PipeWire <-> Opus audio bridge if requested.
    if enable_audio_bridge {
        let playback_conn = conn.clone();
        tokio::spawn(async move {
            match start_playback_capture() {
                Ok(mut rx) => match playback_conn.open_uni().await {
                    Ok(mut stream) => {
                        if let Err(err) =
                            write_stream_header(&mut stream, &StreamHeader::AudioPlayback).await
                        {
                            warn!(error = %err, "failed to write AudioPlayback stream header");
                            return;
                        }
                        while let Some(pkt) = rx.recv().await {
                            let hdr = AudioPacketHeader {
                                pts_ns: pkt.pts_ns,
                                payload_len: pkt.data.len() as u32,
                            };
                            if let Err(err) = write_audio_packet(&mut stream, &hdr, &pkt.data).await
                            {
                                warn!(error = %err, "AudioPlayback stream write error");
                                break;
                            }
                        }
                    }
                    Err(err) => warn!(error = %err, "failed to open AudioPlayback uni stream"),
                },
                Err(err) => warn!(error = %err, "server audio capture unavailable"),
            }
        });

        let mic_conn = conn.clone();
        tokio::spawn(async move {
            while let Ok(mut recv_uni) = mic_conn.accept_uni().await {
                tokio::spawn(async move {
                    match read_stream_header(&mut recv_uni).await {
                        Ok(StreamHeader::AudioCapture) => match AudioMicInjector::new() {
                            Ok(injector) => {
                                while let Ok(Some((hdr, payload))) =
                                    read_audio_packet(&mut recv_uni).await
                                {
                                    if let Err(err) = injector.push_audio(hdr.pts_ns, &payload) {
                                        warn!(error = %err, "server mic push error");
                                        break;
                                    }
                                }
                            }
                            Err(err) => warn!(error = %err, "server mic injector unavailable"),
                        },
                        Ok(other) => warn!(?other, "unexpected uni stream from client"),
                        Err(err) => warn!(error = %err, "failed to read client uni stream header"),
                    }
                });
            }
        });
    }

    struct WindowVideoStream {
        encoder: VideoEncoder,
        stream: SendStream,
    }

    let mut video_streams: HashMap<WindowId, WindowVideoStream> = HashMap::new();
    let session_start = Instant::now();

    // Bridge compositor events -> QUIC control/video streams and QUIC control messages -> compositor.
    loop {
        tokio::select! {
            ev = ev_rx.recv() => {
                match ev {
                    Ok(CompositorEvent::WindowCreated { window_id, title, app_id, width, height }) => {
                        write_control_message(&mut send, &ControlMessage::WindowCreated {
                            window_id, title, app_id, width, height
                        }).await?;
                    }
                    Ok(CompositorEvent::PopupCreated { window_id, parent_id, x, y, width, height }) => {
                        write_control_message(&mut send, &ControlMessage::PopupCreated {
                            window_id, parent_id, x, y, width, height
                        }).await?;
                    }
                    Ok(CompositorEvent::PopupRepositioned { window_id, x, y, width, height }) => {
                        write_control_message(&mut send, &ControlMessage::PopupRepositioned {
                            window_id, x, y, width, height
                        }).await?;
                    }
                    Ok(CompositorEvent::PopupFrame { window_id, width, height, bgra }) => {
                        let compressed = lz4_flex::compress_prepend_size(&bgra);
                        write_control_message(&mut send, &ControlMessage::PopupFrame {
                            window_id, width, height, bgra: compressed
                        }).await?;
                    }
                    Ok(CompositorEvent::StartMove { window_id }) => {
                        write_control_message(&mut send, &ControlMessage::StartMove { window_id }).await?;
                    }
                    Ok(CompositorEvent::StartResize { window_id, edge }) => {
                        write_control_message(&mut send, &ControlMessage::StartResize { window_id, edge }).await?;
                    }
                    Ok(CompositorEvent::SetMaximized { window_id, maximized }) => {
                        write_control_message(&mut send, &ControlMessage::SetMaximized { window_id, maximized }).await?;
                    }
                    Ok(CompositorEvent::SetMinimized { window_id }) => {
                        write_control_message(&mut send, &ControlMessage::SetMinimized { window_id }).await?;
                    }
                    Ok(CompositorEvent::SetFullscreen { window_id, fullscreen }) => {
                        write_control_message(&mut send, &ControlMessage::SetFullscreen { window_id, fullscreen }).await?;
                    }
                    Ok(CompositorEvent::ShowWindowMenu { window_id, x, y }) => {
                        write_control_message(&mut send, &ControlMessage::ShowWindowMenu { window_id, x, y }).await?;
                    }
                    Ok(CompositorEvent::SetDecorationMode { window_id, server_side }) => {
                        write_control_message(&mut send, &ControlMessage::SetDecorationMode { window_id, server_side }).await?;
                    }
                    Ok(CompositorEvent::WindowConfigured { window_id, width, height, scale_factor }) => {
                        write_control_message(&mut send, &ControlMessage::WindowConfigured {
                            window_id,
                            width,
                            height,
                            scale_factor,
                            maximized: false,
                            fullscreen: false,
                            activated: true,
                            tiled: false,
                        }).await?;
                    }
                    Ok(CompositorEvent::WindowClosed { window_id }) => {
                        if let Some(mut vs) = video_streams.remove(&window_id) {
                            let _ = vs.stream.finish();
                        }
                        write_control_message(&mut send, &ControlMessage::WindowClosed { window_id }).await?;
                    }
                    Ok(CompositorEvent::CursorUpdate { window_id, visible, shape_name }) => {
                        write_control_message(&mut send, &ControlMessage::CursorUpdate {
                            window_id, visible, shape_name
                        }).await?;
                    }
                    Ok(CompositorEvent::ClipboardText { text }) => {
                        write_control_message(&mut send, &ControlMessage::ClipboardText { text }).await?;
                    }
                    Ok(CompositorEvent::FrameCommitted { window_id, width, height, bgra }) => {
                        if !video_streams.contains_key(&window_id) {
                            let mut stream = conn.open_uni().await.context("failed to open video uni stream")?;
                            write_stream_header(&mut stream, &StreamHeader::Video { window_id, codec: selected_codec }).await?;
                            video_streams.insert(window_id, WindowVideoStream {
                                encoder: VideoEncoder::new(),
                                stream,
                            });
                        }
                        if let Some(vs) = video_streams.get_mut(&window_id) {
                            let pts_ns = session_start.elapsed().as_nanos() as u64;
                            let frame = vs.encoder.encode_bgra(width, height, pts_ns, &bgra);
                            let hdr = VideoFrameHeader {
                                pts_ns: frame.pts_ns,
                                width: frame.width,
                                height: frame.height,
                                is_keyframe: frame.is_keyframe,
                                payload_len: frame.data.len() as u32,
                            };
                            if let Err(err) = write_video_frame(&mut vs.stream, &hdr, &frame.data).await {
                                warn!(window_id, error = %err, "failed to write video frame");
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "compositor event receiver lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            msg = read_control_message(&mut recv) => {
                match msg? {
                    Some(ControlMessage::WindowConfigured {
                        window_id,
                        width,
                        height,
                        maximized,
                        fullscreen,
                        activated,
                        tiled,
                        ..
                    }) => {
                        let _ = cmd_tx.send(CompositorCommand::ConfigureWindow {
                            window_id,
                            width,
                            height,
                            maximized,
                            fullscreen,
                            activated,
                            tiled,
                        });
                    }
                    Some(ControlMessage::WindowClosed { window_id }) => {
                        let _ = cmd_tx.send(CompositorCommand::CloseWindow { window_id });
                    }
                    Some(ControlMessage::ClipboardText { text }) => {
                        let _ = cmd_tx.send(CompositorCommand::SetClipboardText { text });
                    }
                    Some(ControlMessage::PointerMotion { window_id, x, y }) => {
                        let _ = cmd_tx.send(CompositorCommand::PointerMotion { window_id, x, y });
                    }
                    Some(ControlMessage::PointerButton { window_id, button, pressed }) => {
                        let _ = cmd_tx.send(CompositorCommand::PointerButton { window_id, button, pressed });
                    }
                    Some(ControlMessage::PointerAxis { window_id, dx, dy }) => {
                        let _ = cmd_tx.send(CompositorCommand::PointerAxis { window_id, dx, dy });
                    }
                    Some(ControlMessage::KeyEvent { window_id, keycode, pressed }) => {
                        let _ = cmd_tx.send(CompositorCommand::KeyEvent { window_id, keycode, pressed });
                    }
                    Some(other) => {
                        info!(?other, "received control message");
                    }
                    None => break,
                }
            }
        }
    }

    Ok(())
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
