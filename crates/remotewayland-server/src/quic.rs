//! QUIC server transport and ephemeral TLS/token authentication.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use quinn::{Endpoint, SendStream, ServerConfig, TransportConfig};
use remotewayland_proto::{
    read_audio_packet, read_bidi_stream_header, read_control_message, read_stream_header,
    write_audio_packet, write_control_message, write_stream_header, write_video_frame,
    AudioPacketHeader, BidiStreamHeader, ControlMessage, StreamHeader, VideoCodec,
    VideoFrameHeader, WindowId, PROTOCOL_VERSION, PTY_LAUNCH_ID_BASE,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use crate::audio::{start_per_app_playback_capture, AudioMicInjector};
use crate::compositor::{CompositorCommand, CompositorEvent, CompositorHandle};
use crate::media::VideoEncoder;

/// Holds the ephemeral TLS certificate fingerprint and single-use expiring session tokens.
pub struct EphemeralCredentials {
    pub server_config: ServerConfig,
    pub cert_sha256_hex: String,
    pub valid_tokens: Arc<Mutex<HashMap<String, Instant>>>,
}

impl EphemeralCredentials {
    /// Add a newly minted single-use session token with the given TTL.
    pub fn add_token(&self, token: String, ttl: Duration) {
        let now = Instant::now();
        if let Ok(mut map) = self.valid_tokens.lock() {
            map.retain(|_, exp| *exp > now);
            map.insert(token, now + ttl);
        }
    }
}

/// Consume a single-use session token, returning `true` only if it existed and was not expired.
fn consume_token(valid_tokens: &Mutex<HashMap<String, Instant>>, token: &str) -> bool {
    let now = Instant::now();
    let Ok(mut map) = valid_tokens.lock() else {
        return false;
    };
    map.retain(|_, exp| *exp > now);
    map.remove(token).is_some_and(|exp| exp > now)
}

/// Generate a self-signed TLS 1.3 certificate and ephemeral session token.
pub fn generate_ephemeral_credentials(
    initial_token: String,
    initial_ttl: Duration,
) -> Result<EphemeralCredentials> {
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
    transport.keep_alive_interval(Some(Duration::from_secs(3)));
    if let Ok(idle) = Duration::from_secs(10).try_into() {
        transport.max_idle_timeout(Some(idle));
    }
    server_config.transport_config(Arc::new(transport));

    let mut tokens = HashMap::new();
    tokens.insert(initial_token, Instant::now() + initial_ttl);

    Ok(EphemeralCredentials {
        server_config,
        cert_sha256_hex,
        valid_tokens: Arc::new(Mutex::new(tokens)),
    })
}

/// Bind the QUIC endpoint and handle incoming authenticated client sessions.
pub async fn run_server(
    listen_addr: SocketAddr,
    creds: EphemeralCredentials,
    comp_handle: CompositorHandle,
    mut event_rx: mpsc::UnboundedReceiver<CompositorEvent>,
    bound_addr_tx: Option<tokio::sync::oneshot::Sender<SocketAddr>>,
    quiet: bool,
) -> Result<()> {
    let endpoint = Endpoint::server(creds.server_config, listen_addr)
        .with_context(|| format!("failed to bind QUIC endpoint on {listen_addr}"))?;
    let actual_addr = endpoint.local_addr().unwrap_or(listen_addr);
    if let Some(tx) = bound_addr_tx {
        let _ = tx.send(actual_addr);
    }

    info!(%actual_addr, cert_sha256 = %creds.cert_sha256_hex, "RemoteWayland server listening");
    if !quiet {
        let first_token = creds
            .valid_tokens
            .lock()
            .ok()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_default();
        println!(
            "remotewayland://{}?fp={}&token={}",
            actual_addr, creds.cert_sha256_hex, first_token
        );
    }

    let (bcast_tx, _) = broadcast::channel::<CompositorEvent>(256);
    let bcast_tx_clone = bcast_tx.clone();
    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            let _ = bcast_tx_clone.send(ev);
        }
    });

    let pty_mgr = crate::pty::PtyManager::new(bcast_tx.clone());
    let valid_tokens = Arc::clone(&creds.valid_tokens);
    while let Some(incoming) = endpoint.accept().await {
        let tokens = Arc::clone(&valid_tokens);
        let cmd_tx = comp_handle.cmd_tx.clone();
        let ev_rx = bcast_tx.subscribe();
        let pty = Arc::clone(&pty_mgr);
        tokio::spawn(async move {
            if let Err(err) = handle_connection(incoming, tokens, cmd_tx, ev_rx, pty).await {
                warn!(error = %err, "client connection ended with error");
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    incoming: quinn::Incoming,
    valid_tokens: Arc<Mutex<HashMap<String, Instant>>>,
    cmd_tx: calloop::channel::Sender<CompositorCommand>,
    mut ev_rx: broadcast::Receiver<CompositorEvent>,
    pty_mgr: Arc<crate::pty::PtyManager>,
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

            let token_valid = consume_token(&valid_tokens, &token);
            if !token_valid {
                write_control_message(
                    &mut send,
                    &ControlMessage::AuthResponse {
                        accepted: false,
                        reason: Some("invalid or expired session token".into()),
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
            pty_mgr.sync_states();
            (selected_codec, enable_audio_bridge)
        }
        other => {
            warn!(?other, "unexpected initial control message");
            return Ok(());
        }
    };

    // Accept secondary bidirectional QUIC streams (e.g., persistent remote PTYs).
    let bidi_conn = conn.clone();
    let bidi_pty = Arc::clone(&pty_mgr);
    tokio::spawn(async move {
        while let Ok((send_bi, mut recv_bi)) = bidi_conn.accept_bi().await {
            let pty = Arc::clone(&bidi_pty);
            tokio::spawn(async move {
                match read_bidi_stream_header(&mut recv_bi).await {
                    Ok(BidiStreamHeader::Pty {
                        pty_id,
                        resume_offset,
                        cols,
                        rows,
                        command,
                    }) => {
                        if let Err(err) = pty
                            .handle_pty_stream(
                                send_bi,
                                recv_bi,
                                pty_id,
                                resume_offset,
                                cols,
                                rows,
                                command,
                            )
                            .await
                        {
                            warn!(error = %err, "PTY stream error");
                        }
                    }
                    Err(err) => warn!(error = %err, "failed to read BidiStreamHeader"),
                }
            });
        }
    });

    // Start bidirectional PipeWire <-> Opus audio bridge if requested.
    let (mut mic_active_rx, _audio_bridge_guard) = if enable_audio_bridge {
        let (mic_rx, guard) = match start_per_app_playback_capture() {
            Ok(bridge) => {
                let mut streams = bridge.streams;
                let playback_conn = conn.clone();
                tokio::spawn(async move {
                    while let Some(mut app_stream) = streams.recv().await {
                        let conn = playback_conn.clone();
                        tokio::spawn(async move {
                            let Ok(mut stream) = conn.open_uni().await else {
                                warn!(
                                    stream_id = app_stream.stream_id,
                                    app_name = %app_stream.app_name,
                                    "failed to open AudioPlaybackApp uni stream"
                                );
                                return;
                            };
                            let header = StreamHeader::AudioPlaybackApp {
                                stream_id: app_stream.stream_id,
                                app_name: app_stream.app_name.clone(),
                                media_name: app_stream.media_name.clone(),
                                icon_name: app_stream.icon_name.clone(),
                                media_role: app_stream.media_role.clone(),
                            };
                            if let Err(err) = write_stream_header(&mut stream, &header).await {
                                warn!(error = %err, "failed to write AudioPlaybackApp stream header");
                                return;
                            }
                            while let Some(pkt) = app_stream.packets.recv().await {
                                let hdr = AudioPacketHeader {
                                    pts_ns: pkt.pts_ns,
                                    payload_len: pkt.data.len() as u32,
                                };
                                if let Err(err) =
                                    write_audio_packet(&mut stream, &hdr, &pkt.data).await
                                {
                                    warn!(error = %err, "AudioPlaybackApp stream write error");
                                    break;
                                }
                            }
                            let _ = stream.finish();
                        });
                    }
                });
                (Some(bridge.mic_active), Some(bridge.guard))
            }
            Err(err) => {
                warn!(error = %err, "server audio capture unavailable");
                (None, None)
            }
        };

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
        (mic_rx, guard)
    } else {
        (None, None)
    };

    struct WindowVideoStream {
        encoder: VideoEncoder,
        stream: SendStream,
    }

    let mut video_streams: HashMap<WindowId, WindowVideoStream> = HashMap::new();
    let mut popup_encoders: HashMap<WindowId, VideoEncoder> = HashMap::new();
    let session_start = Instant::now();

    // Bridge compositor events -> QUIC control/video streams and QUIC control messages -> compositor.
    loop {
        tokio::select! {
            Some(active) = async {
                match mic_active_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                write_control_message(&mut send, &ControlMessage::SetMicActive { active }).await?;
            }
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
                        let enc = popup_encoders.entry(window_id).or_insert_with(VideoEncoder::new);
                        let pts_ns = session_start.elapsed().as_nanos() as u64;
                        let frame = enc.encode_bgra(width, height, pts_ns, &bgra);
                        write_control_message(&mut send, &ControlMessage::PopupFrame {
                            window_id,
                            width: frame.width,
                            height: frame.height,
                            is_keyframe: frame.is_keyframe,
                            bgra: frame.data,
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
                        popup_encoders.remove(&window_id);
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
                    Ok(CompositorEvent::AppState { launch_id, name, state }) => {
                        write_control_message(&mut send, &ControlMessage::AppState {
                            launch_id, name, state
                        }).await?;
                    }
                    Ok(CompositorEvent::FrameCommitted { window_id, width, height, bgra }) => {
                        if let std::collections::hash_map::Entry::Vacant(e) = video_streams.entry(window_id) {
                            let mut stream = conn.open_uni().await.context("failed to open video uni stream")?;
                            write_stream_header(&mut stream, &StreamHeader::Video { window_id, codec: selected_codec }).await?;
                            e.insert(WindowVideoStream {
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
                    Some(ControlMessage::ListApps) => {
                        let apps = tokio::task::spawn_blocking(crate::apps::discover_apps)
                            .await
                            .unwrap_or_default();
                        write_control_message(&mut send, &ControlMessage::AppList { apps }).await?;
                    }
                    Some(ControlMessage::LaunchApp { launch_id, name, exec }) => {
                        let _ = cmd_tx.send(CompositorCommand::LaunchApp { launch_id, name, exec });
                    }
                    Some(ControlMessage::TerminateApp { launch_id, force }) => {
                        if (launch_id & PTY_LAUNCH_ID_BASE) != 0 {
                            pty_mgr.terminate(launch_id & !PTY_LAUNCH_ID_BASE, force);
                        } else {
                            let _ = cmd_tx.send(CompositorCommand::TerminateApp { launch_id, force });
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_use_and_expiring_tokens() {
        let map = Mutex::new(HashMap::new());
        {
            let mut guard = map.lock().unwrap();
            guard.insert("valid".to_string(), Instant::now() + Duration::from_secs(60));
            guard.insert("expired".to_string(), Instant::now() - Duration::from_secs(1));
        }

        // First use of valid token succeeds; second use fails (single-use).
        assert!(consume_token(&map, "valid"));
        assert!(!consume_token(&map, "valid"));

        // Expired token is rejected and pruned.
        assert!(!consume_token(&map, "expired"));
        assert!(!consume_token(&map, "nonexistent"));
        assert!(map.lock().unwrap().is_empty());
    }
}
