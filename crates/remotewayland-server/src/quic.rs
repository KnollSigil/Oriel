//! QUIC server transport and ephemeral TLS/token authentication.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use quinn::{Endpoint, ServerConfig, TransportConfig};
use remotewayland_proto::{
    read_audio_packet, read_bidi_stream_header, read_control_message, read_stream_header,
    write_audio_packet, write_control_message, write_stream_header, write_video_frame,
    AudioPacketHeader, BidiStreamHeader, ControlMessage, StreamHeader, VideoCodec,
    VideoFrameHeader, WindowId, MAX_VIDEO_FRAME_BYTES, PROTOCOL_VERSION, PTY_LAUNCH_ID_BASE,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::{broadcast, mpsc, watch, Notify};
use tracing::{info, warn};

use crate::audio::{start_per_app_playback_capture, AudioMicInjector};
use crate::compositor::{CompositorCommand, CompositorEvent, CompositorHandle};
use crate::media::WindowVideoEncoder;

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
    transport.congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));
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
) -> Result<()> {
    let endpoint = Endpoint::server(creds.server_config, listen_addr)
        .with_context(|| format!("failed to bind QUIC endpoint on {listen_addr}"))?;
    let actual_addr = endpoint.local_addr().unwrap_or(listen_addr);
    if let Some(tx) = bound_addr_tx {
        let _ = tx.send(actual_addr);
    }

    // The connect URI carries a session token, so it is never written to the
    // server's own output; `remotewayland-server attach` mints one on request.
    info!(%actual_addr, cert_sha256 = %creds.cert_sha256_hex, "RemoteWayland server listening");

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
            initial_scale,
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

            let selected_codec = if supported_codecs.contains(&VideoCodec::SmartTile) {
                VideoCodec::SmartTile
            } else {
                supported_codecs
                    .first()
                    .copied()
                    .unwrap_or(VideoCodec::Lz4Delta)
            };

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
                initial_scale,
                "client authenticated; starting compositor/media loop"
            );
            let _ = cmd_tx.send(CompositorCommand::SetDisplayScale {
                scale_factor: initial_scale,
            });
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

    struct PopupEncoderState {
        encoder: WindowVideoEncoder,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        last_update: Instant,
    }

    let mut video_pipelines: HashMap<WindowId, WindowVideoPipeline> = HashMap::new();
    let mut popup_encoders: HashMap<WindowId, PopupEncoderState> = HashMap::new();
    let session_start = Instant::now();
    const POPUP_IDLE_REFRESH_DELAY: Duration = Duration::from_millis(120);

    // Serialize and coalesce client ColorScheme updates onto a single worker so bursts
    // cannot spawn unbounded blocking tasks or race on isolated portal config files.
    let (color_scheme_tx, mut color_scheme_rx) = tokio::sync::watch::channel::<Option<u32>>(None);
    tokio::spawn(async move {
        while color_scheme_rx.changed().await.is_ok() {
            let scheme_opt = *color_scheme_rx.borrow_and_update();
            if let Some(scheme) = scheme_opt {
                let _ = tokio::task::spawn_blocking(move || {
                    crate::session::apply_color_scheme(scheme);
                })
                .await;
            }
        }
    });

    // Dedicated control-stream reader task: `read_control_message` performs multiple
    // `read_exact` calls and is NOT cancel-safe inside `tokio::select!`. Running it in its
    // own task with a bounded channel prevents partial-frame drops and applies QUIC backpressure.
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

    // Bridge compositor events -> QUIC control/video streams and QUIC control messages -> compositor.
    let loop_res: Result<()> = async {
    loop {
        let next_popup_refresh = popup_encoders
            .iter()
            .filter(|(_, st)| st.encoder.has_pending_lossless_refresh())
            .map(|(&wid, st)| {
                (
                    wid,
                    POPUP_IDLE_REFRESH_DELAY.saturating_sub(st.last_update.elapsed()),
                )
            })
            .min_by_key(|&(_, wait)| wait);

        tokio::select! {
            Some(wid) = async {
                match next_popup_refresh {
                    Some((wid, wait)) => {
                        if !wait.is_zero() {
                            tokio::time::sleep(wait).await;
                        }
                        Some(wid)
                    }
                    None => std::future::pending().await,
                }
            } => {
                if let Some(st) = popup_encoders.get_mut(&wid) {
                    let pts_ns = session_start.elapsed().as_nanos() as u64;
                    if let Some(frame) = st.encoder.encode_lossless_refresh(st.width, st.height, pts_ns) {
                        st.last_update = Instant::now();
                        write_control_message(&mut send, &ControlMessage::PopupFrame {
                            window_id: wid,
                            width: frame.width,
                            height: frame.height,
                            logical_width: st.logical_width,
                            logical_height: st.logical_height,
                            is_keyframe: frame.is_keyframe,
                            bgra: frame.data,
                        }).await?;
                    }
                }
            }
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
                    Ok(CompositorEvent::PopupFrame {
                        window_id,
                        width,
                        height,
                        logical_width,
                        logical_height,
                        bgra,
                    }) => {
                        let st = popup_encoders
                            .entry(window_id)
                            .or_insert_with(|| PopupEncoderState {
                                encoder: WindowVideoEncoder::new(selected_codec),
                                width,
                                height,
                                logical_width,
                                logical_height,
                                last_update: Instant::now(),
                            });
                        st.width = width;
                        st.height = height;
                        st.logical_width = logical_width;
                        st.logical_height = logical_height;
                        st.last_update = Instant::now();
                        let pts_ns = session_start.elapsed().as_nanos() as u64;
                        let frame = st.encoder.encode_bgra(width, height, pts_ns, &bgra, 80);
                        write_control_message(&mut send, &ControlMessage::PopupFrame {
                            window_id,
                            width: frame.width,
                            height: frame.height,
                            logical_width,
                            logical_height,
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
                        video_pipelines.remove(&window_id);
                        popup_encoders.remove(&window_id);
                        write_control_message(&mut send, &ControlMessage::WindowClosed { window_id }).await?;
                    }
                    Ok(CompositorEvent::CursorUpdate { window_id, visible, shape_name }) => {
                        write_control_message(&mut send, &ControlMessage::CursorUpdate {
                            window_id, visible, shape_name
                        }).await?;
                    }
                    Ok(CompositorEvent::CursorBitmap {
                        window_id,
                        width,
                        height,
                        logical_width,
                        logical_height,
                        hotspot_x,
                        hotspot_y,
                        bgra,
                    }) => {
                        write_control_message(&mut send, &ControlMessage::CursorBitmap {
                            window_id,
                            width,
                            height,
                            logical_width,
                            logical_height,
                            hotspot_x,
                            hotspot_y,
                            bgra,
                        }).await?;
                    }
                    Ok(CompositorEvent::ClipboardText { text }) => {
                        write_control_message(&mut send, &ControlMessage::ClipboardText { text }).await?;
                    }
                    Ok(CompositorEvent::PrimarySelectionText { text }) => {
                        write_control_message(&mut send, &ControlMessage::PrimarySelectionText { text }).await?;
                    }
                    Ok(CompositorEvent::AppState { launch_id, name, state }) => {
                        write_control_message(&mut send, &ControlMessage::AppState {
                            launch_id, name, state
                        }).await?;
                    }
                    Ok(CompositorEvent::FrameCommitted {
                        window_id,
                        width,
                        height,
                        logical_width,
                        logical_height,
                        bgra,
                    }) => {
                        let pipeline = video_pipelines.entry(window_id).or_insert_with(|| {
                            spawn_window_video_pipeline(
                                window_id,
                                conn.clone(),
                                selected_codec,
                                session_start,
                            )
                        });
                        pipeline.push_frame(RawVideoFrame {
                            width,
                            height,
                            logical_width,
                            logical_height,
                            bgra,
                        });
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "compositor event receiver lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            msg = ctrl_in_rx.recv() => {
                let msg = match msg {
                    Some(Ok(m)) => Some(m),
                    Some(Err(err)) => return Err(err),
                    None => None,
                };
                match msg {
                    Some(ControlMessage::WindowConfigured {
                        window_id,
                        width,
                        height,
                        scale_factor,
                        maximized,
                        fullscreen,
                        activated,
                        tiled,
                    }) => {
                        let _ = cmd_tx.send(CompositorCommand::ConfigureWindow {
                            window_id,
                            width,
                            height,
                            scale_factor,
                            maximized,
                            fullscreen,
                            activated,
                            tiled,
                        });
                    }
                    Some(ControlMessage::DisplayScale { scale_factor }) => {
                        let _ = cmd_tx.send(CompositorCommand::SetDisplayScale { scale_factor });
                    }
                    Some(ControlMessage::WindowClosed { window_id }) => {
                        let _ = cmd_tx.send(CompositorCommand::CloseWindow { window_id });
                    }
                    Some(ControlMessage::ClipboardText { text }) => {
                        let _ = cmd_tx.send(CompositorCommand::SetClipboardText { text });
                    }
                    Some(ControlMessage::PrimarySelectionText { text }) => {
                        let _ = cmd_tx.send(CompositorCommand::SetPrimarySelectionText { text });
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
                    Some(ControlMessage::KeyboardEnter { window_id, pressed_keys }) => {
                        let _ = cmd_tx.send(CompositorCommand::KeyboardEnter { window_id, pressed_keys });
                    }
                    Some(ControlMessage::KeyboardLeave { window_id }) => {
                        let _ = cmd_tx.send(CompositorCommand::KeyboardLeave { window_id });
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
                    Some(ControlMessage::ColorScheme { scheme }) => {
                        let _ = color_scheme_tx.send(Some(scheme));
                    }
                    Some(ControlMessage::FrameAck { window_id, pts_ns }) => {
                        if let Some(pipeline) = video_pipelines.get(&window_id) {
                            pipeline.ack_frame(pts_ns);
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
    .await;

    ctrl_reader_task.abort();
    for (_, pipeline) in video_pipelines.drain() {
        pipeline.task.abort();
    }

    loop_res
}

/// Maximum number of unacknowledged video frames allowed in flight per window.
/// Allows 60 FPS pipelining over ~10-20ms Wi-Fi RTT while keeping latency bounded.
const MAX_IN_FLIGHT_FRAMES: usize = 3;

/// Safety timeout for waiting on a `FrameAck` so a stalled ack can never permanently deadlock a stream.
const FRAME_ACK_TIMEOUT: Duration = Duration::from_millis(400);

struct RawVideoFrame {
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    bgra: Arc<Vec<u8>>,
}

struct WindowVideoPipeline {
    latest_frame: Arc<Mutex<Option<RawVideoFrame>>>,
    notify: Arc<Notify>,
    ack_tx: watch::Sender<u64>,
    task: tokio::task::JoinHandle<()>,
}

impl WindowVideoPipeline {
    fn push_frame(&self, frame: RawVideoFrame) {
        *self.latest_frame.lock().unwrap() = Some(frame);
        self.notify.notify_one();
    }

    fn ack_frame(&self, pts_ns: u64) {
        self.ack_tx.send_modify(|cur| {
            if pts_ns > *cur {
                *cur = pts_ns;
            }
        });
    }
}

fn spawn_window_video_pipeline(
    window_id: WindowId,
    conn: quinn::Connection,
    selected_codec: VideoCodec,
    session_start: Instant,
) -> WindowVideoPipeline {
    let latest_frame = Arc::new(Mutex::new(None));
    let notify = Arc::new(Notify::new());
    let (ack_tx, ack_rx) = watch::channel(0u64);
    let task = tokio::spawn(run_window_video_pipeline(
        window_id,
        conn,
        selected_codec,
        session_start,
        latest_frame.clone(),
        notify.clone(),
        ack_rx,
    ));
    WindowVideoPipeline {
        latest_frame,
        notify,
        ack_tx,
        task,
    }
}

fn prune_acked_pts(in_flight_pts: &mut VecDeque<u64>, acked_pts: u64) {
    while in_flight_pts.front().is_some_and(|&pts| pts <= acked_pts) {
        in_flight_pts.pop_front();
    }
}

async fn run_window_video_pipeline(
    window_id: WindowId,
    conn: quinn::Connection,
    selected_codec: VideoCodec,
    session_start: Instant,
    latest_frame: Arc<Mutex<Option<RawVideoFrame>>>,
    notify: Arc<Notify>,
    mut ack_rx: watch::Receiver<u64>,
) {
    let mut stream = match conn.open_uni().await {
        Ok(s) => s,
        Err(err) => {
            warn!(window_id, error = %err, "failed to open video uni stream");
            return;
        }
    };
    if let Err(err) = write_stream_header(
        &mut stream,
        &StreamHeader::Video {
            window_id,
            codec: selected_codec,
        },
    )
    .await
    {
        warn!(window_id, error = %err, "failed to write video StreamHeader");
        return;
    }

    let mut encoder = Some(WindowVideoEncoder::new(selected_codec));
    let mut in_flight_pts: VecDeque<u64> = VecDeque::new();
    let mut last_pts_ns: u64 = 0;
    let mut quality: u8 = 80;
    let mut last_congestion: Option<Instant> = None;
    let mut last_dims: Option<(u32, u32, u32, u32)> = None;
    let mut refining_idle = false;
    const IDLE_REFRESH_DELAY: Duration = Duration::from_millis(120);
    const CONGESTION_COOLDOWN: Duration = Duration::from_millis(500);
    const LOSSY_CLIMB_BYTE_CAP: usize = 95_000;
    const LOSSY_NUDGE_BYTE_CAP: usize = 130_000;

    loop {
        // 1. Wait until a new frame is available in the 1-slot mailbox,
        //    or stream progressive lossless refinement chunks after 120ms idle.
        let mut do_idle_refresh = false;
        loop {
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if latest_frame.lock().unwrap().is_some() {
                refining_idle = false;
                break;
            }
            let needs_refresh = encoder
                .as_ref()
                .is_some_and(|e| e.has_pending_lossless_refresh())
                && last_dims.is_some();
            if needs_refresh && refining_idle {
                do_idle_refresh = true;
                break;
            }
            tokio::select! {
                _ = notified => {}
                res = ack_rx.changed() => {
                    if res.is_err() {
                        let _ = stream.finish();
                        return;
                    }
                    prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
                }
                _ = async {
                    if needs_refresh {
                        tokio::time::sleep(IDLE_REFRESH_DELAY).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    if latest_frame.lock().unwrap().is_none() {
                        refining_idle = true;
                        do_idle_refresh = true;
                        break;
                    }
                }
            }
        }

        // 2. Enforce end-to-end in-flight frame cap before taking/encoding the newest frame.
        //    Any newer frames arriving while waiting overwrite `latest_frame` in place.
        prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
        let mut experienced_backpressure = false;
        let mut ack_timed_out = false;
        let ack_wait_start = Instant::now();
        while in_flight_pts.len() >= MAX_IN_FLIGHT_FRAMES {
            experienced_backpressure = true;
            match tokio::time::timeout(FRAME_ACK_TIMEOUT, ack_rx.changed()).await {
                Ok(Ok(())) => {
                    prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
                }
                Ok(Err(_)) => {
                    let _ = stream.finish();
                    return;
                }
                Err(_) => {
                    ack_timed_out = true;
                    in_flight_pts.pop_front();
                }
            }
        }
        let ack_wait_elapsed = if experienced_backpressure {
            ack_wait_start.elapsed()
        } else {
            Duration::ZERO
        };

        let raw_pts = session_start.elapsed().as_nanos() as u64;
        let pts_ns = raw_pts.max(last_pts_ns.saturating_add(1));

        let encode_start = Instant::now();
        let (logical_width, logical_height, frame) = if do_idle_refresh {
            // Re-check mailbox in case a new frame arrived while waiting for credit.
            if latest_frame.lock().unwrap().is_some() {
                refining_idle = false;
                continue;
            }
            let Some((lw_phys, lh_phys, log_w, log_h)) = last_dims else {
                refining_idle = false;
                continue;
            };
            let mut enc = encoder.take().unwrap();
            let encoded_res = tokio::task::spawn_blocking(move || {
                let frame = enc.encode_lossless_refresh(lw_phys, lh_phys, pts_ns);
                (enc, frame)
            })
            .await;
            let (enc, frame_opt) = match encoded_res {
                Ok(pair) => pair,
                Err(err) => {
                    warn!(window_id, error = %err, "video lossless refresh task panicked");
                    let _ = stream.finish();
                    return;
                }
            };
            let still_has_lossy = enc.has_pending_lossless_refresh();
            encoder = Some(enc);
            if !still_has_lossy {
                refining_idle = false;
            }
            let Some(frame) = frame_opt else {
                refining_idle = false;
                continue;
            };
            last_pts_ns = pts_ns;
            (log_w, log_h, frame)
        } else {
            // 3. Take the latest frame AFTER acquiring credit so stale intermediate frames are skipped.
            let Some(raw) = latest_frame.lock().unwrap().take() else {
                continue;
            };
            last_pts_ns = pts_ns;
            let logical_width = raw.logical_width;
            let logical_height = raw.logical_height;
            last_dims = Some((raw.width, raw.height, logical_width, logical_height));

            // 4. Encode off the async runtime thread so control/input latency stays unaffected.
            let mut enc = encoder.take().unwrap();
            let q = quality;
            let encoded_res = tokio::task::spawn_blocking(move || {
                let frame = enc.encode_bgra(raw.width, raw.height, pts_ns, &raw.bgra, q);
                (enc, frame)
            })
            .await;
            let (enc, frame) = match encoded_res {
                Ok(pair) => pair,
                Err(err) => {
                    warn!(window_id, error = %err, "video encoder task panicked");
                    let _ = stream.finish();
                    return;
                }
            };
            encoder = Some(enc);
            (logical_width, logical_height, frame)
        };
        let encode_elapsed = encode_start.elapsed();

        // 5. Write encoded frame to the per-window QUIC stream and adapt DCT quality dynamically.
        let Ok(payload_len) = u32::try_from(frame.data.len()) else {
            warn!(window_id, bytes = frame.data.len(), "encoded video frame exceeds u32::MAX; dropping");
            continue;
        };
        if frame.data.len() > MAX_VIDEO_FRAME_BYTES {
            warn!(
                window_id,
                bytes = frame.data.len(),
                max = MAX_VIDEO_FRAME_BYTES,
                "encoded video frame exceeds MAX_VIDEO_FRAME_BYTES; dropping"
            );
            continue;
        }
        let hdr = VideoFrameHeader {
            pts_ns: frame.pts_ns,
            width: frame.width,
            height: frame.height,
            logical_width,
            logical_height,
            is_keyframe: frame.is_keyframe,
            payload_len,
        };
        let send_start = Instant::now();
        if let Err(err) = write_video_frame(&mut stream, &hdr, &frame.data).await {
            warn!(window_id, error = %err, "failed to write video frame");
            return;
        }
        let send_elapsed = send_start.elapsed();
        tracing::debug!(
            window_id,
            w = frame.width,
            h = frame.height,
            copy_rect = frame.had_copy_rect,
            dirty_tiles = frame.dirty_tiles,
            lossy = frame.used_lossy,
            quality,
            bytes = frame.data.len(),
            ack_wait_ms = ack_wait_elapsed.as_millis() as u64,
            encode_ms = encode_elapsed.as_millis() as u64,
            send_ms = send_elapsed.as_millis() as u64,
            "sent video frame"
        );
        if !do_idle_refresh {
            if ack_timed_out || ack_wait_elapsed > Duration::from_millis(80) {
                quality = quality.saturating_sub(12).max(26);
                last_congestion = Some(Instant::now());
            } else if ack_wait_elapsed > Duration::from_millis(30)
                || send_elapsed > Duration::from_millis(20)
            {
                quality = quality.saturating_sub(6).max(26);
                last_congestion = Some(Instant::now());
            } else if frame.used_lossy
                && frame.data.len() >= LOSSY_NUDGE_BYTE_CAP
                && ack_wait_elapsed > Duration::from_millis(15)
            {
                quality = quality.saturating_sub(4).max(26);
                last_congestion = Some(Instant::now());
            } else if frame.used_lossy
                && frame.data.len() < LOSSY_CLIMB_BYTE_CAP
                && ack_wait_elapsed < Duration::from_millis(12)
                && send_elapsed < Duration::from_millis(8)
                && last_congestion.is_none_or(|t| t.elapsed() >= CONGESTION_COOLDOWN)
            {
                let step = if quality < 44 { 1 } else { 2 };
                quality = (quality + step).min(82);
            }
        }

        in_flight_pts.push_back(pts_ns);
        prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
    }
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

    #[test]
    fn prune_acked_pts_pops_up_to_acked_timestamp() {
        let mut q = VecDeque::from([10, 20, 30, 40]);
        prune_acked_pts(&mut q, 5);
        assert_eq!(q, VecDeque::from([10, 20, 30, 40]));

        prune_acked_pts(&mut q, 20);
        assert_eq!(q, VecDeque::from([30, 40]));

        prune_acked_pts(&mut q, 100);
        assert!(q.is_empty());
    }
}
