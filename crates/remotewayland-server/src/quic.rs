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
    VideoFrameHeader, WindowId, PROTOCOL_VERSION, PTY_LAUNCH_ID_BASE,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::{broadcast, mpsc, watch, Notify};
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

    let mut video_pipelines: HashMap<WindowId, WindowVideoPipeline> = HashMap::new();
    let mut popup_encoders: HashMap<WindowId, VideoEncoder> = HashMap::new();
    let session_start = Instant::now();

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
                        video_pipelines.remove(&window_id);
                        popup_encoders.remove(&window_id);
                        write_control_message(&mut send, &ControlMessage::WindowClosed { window_id }).await?;
                    }
                    Ok(CompositorEvent::CursorUpdate { window_id, visible, shape_name }) => {
                        write_control_message(&mut send, &ControlMessage::CursorUpdate {
                            window_id, visible, shape_name
                        }).await?;
                    }
                    Ok(CompositorEvent::CursorBitmap { window_id, width, height, hotspot_x, hotspot_y, bgra }) => {
                        write_control_message(&mut send, &ControlMessage::CursorBitmap {
                            window_id, width, height, hotspot_x, hotspot_y, bgra
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
                    Ok(CompositorEvent::FrameCommitted { window_id, width, height, bgra }) => {
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
                            bgra,
                        });
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

    for (_, pipeline) in video_pipelines.drain() {
        pipeline.task.abort();
    }

    Ok(())
}

/// Maximum number of unacknowledged video frames allowed in flight per window.
/// Keeps end-to-end latency bounded when UI updates outpace network bandwidth or decoding.
const MAX_IN_FLIGHT_FRAMES: usize = 2;

/// Safety timeout for waiting on a `FrameAck` so a stalled ack can never permanently deadlock a stream.
const FRAME_ACK_TIMEOUT: Duration = Duration::from_secs(1);

struct RawVideoFrame {
    width: u32,
    height: u32,
    bgra: Vec<u8>,
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

    let mut encoder = Some(VideoEncoder::new());
    let mut in_flight_pts: VecDeque<u64> = VecDeque::new();
    let mut last_pts_ns: u64 = 0;

    loop {
        // 1. Wait until a new frame is available in the 1-slot mailbox.
        loop {
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if latest_frame.lock().unwrap().is_some() {
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
            }
        }

        // 2. Enforce end-to-end in-flight frame cap before taking/encoding the newest frame.
        //    Any newer frames arriving while waiting overwrite `latest_frame` in place.
        prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
        while in_flight_pts.len() >= MAX_IN_FLIGHT_FRAMES {
            match tokio::time::timeout(FRAME_ACK_TIMEOUT, ack_rx.changed()).await {
                Ok(Ok(())) => {
                    prune_acked_pts(&mut in_flight_pts, *ack_rx.borrow_and_update());
                }
                Ok(Err(_)) => {
                    let _ = stream.finish();
                    return;
                }
                Err(_) => {
                    in_flight_pts.pop_front();
                }
            }
        }

        // 3. Take the latest frame AFTER acquiring credit so stale intermediate frames are skipped.
        let Some(raw) = latest_frame.lock().unwrap().take() else {
            continue;
        };

        let raw_pts = session_start.elapsed().as_nanos() as u64;
        let pts_ns = raw_pts.max(last_pts_ns.saturating_add(1));
        last_pts_ns = pts_ns;

        // 4. Encode off the async runtime thread so control/input latency stays unaffected.
        let mut enc = encoder.take().unwrap();
        let encoded_res = tokio::task::spawn_blocking(move || {
            let frame = enc.encode_bgra(raw.width, raw.height, pts_ns, &raw.bgra);
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

        // 5. Write encoded frame to the per-window QUIC stream.
        let hdr = VideoFrameHeader {
            pts_ns: frame.pts_ns,
            width: frame.width,
            height: frame.height,
            is_keyframe: frame.is_keyframe,
            payload_len: frame.data.len() as u32,
        };
        if let Err(err) = write_video_frame(&mut stream, &hdr, &frame.data).await {
            warn!(window_id, error = %err, "failed to write video frame");
            return;
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
