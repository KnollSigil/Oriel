//! QUIC server transport and ephemeral TLS/token authentication.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use quinn::{Endpoint, ServerConfig, TransportConfig, VarInt};
use remotewayland_proto::{
    read_audio_packet, read_bidi_stream_header, read_control_message, read_protocol_preamble,
    read_stream_header, write_audio_packet, write_control_message, write_stream_header,
    write_video_frame, AudioPacketHeader, BidiStreamHeader, ClientRole, ControlMessage, FrameRect,
    LaunchState, PeerProtocol, PortRange, StreamHeader, VideoCodec, VideoFrameHeader, WindowId,
    CLOSE_CODE_PROTOCOL_MISMATCH, CLOSE_CODE_SUPERSEDED, MAX_VIDEO_FRAME_BYTES, PROTOCOL_VERSION,
    PTY_LAUNCH_ID_BASE,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::{broadcast, mpsc, oneshot, watch, Notify};
use tracing::{info, warn};

use crate::audio::{start_per_app_playback_capture, AudioMicInjector};
use crate::compositor::{CompositorCommand, CompositorEvent, CompositorHandle};
use crate::media::WindowVideoEncoder;

/// A single-use session token waiting to be used.
#[derive(Debug, Clone)]
pub struct TokenGrant {
    pub expires: Instant,
    /// SHA-256 hex fingerprint of the client certificate the token is bound to. A bound token
    /// is only accepted from a connection that presented that certificate, so it is useless
    /// to anyone who learns it without also holding the client's private key.
    pub client_fp: Option<String>,
}

/// Single-use session tokens waiting to be used, by token.
pub type TokenMap = Arc<Mutex<HashMap<String, TokenGrant>>>;

/// Holds the ephemeral TLS certificate fingerprint and single-use expiring session tokens.
pub struct EphemeralCredentials {
    pub server_config: ServerConfig,
    pub cert_sha256_hex: String,
    pub valid_tokens: TokenMap,
}

impl EphemeralCredentials {
    /// Add a newly minted single-use session token with the given TTL, optionally bound to
    /// a client certificate fingerprint.
    pub fn add_token(&self, token: String, ttl: Duration, client_fp: Option<String>) {
        let now = Instant::now();
        if let Ok(mut map) = self.valid_tokens.lock() {
            map.retain(|_, grant| grant.expires > now);
            map.insert(
                token,
                TokenGrant {
                    expires: now + ttl,
                    client_fp: client_fp.map(|fp| fp.to_ascii_lowercase()),
                },
            );
        }
    }
}

/// Consume a single-use session token, returning `true` only if it existed, was not expired,
/// and, when bound to a client certificate, `presented_fp` is that certificate's fingerprint.
/// A token presented with the wrong certificate stays valid, so whoever tried it cannot keep
/// the client it was minted for from using it.
fn consume_token(
    valid_tokens: &Mutex<HashMap<String, TokenGrant>>,
    token: &str,
    presented_fp: Option<&str>,
) -> bool {
    let now = Instant::now();
    let Ok(mut map) = valid_tokens.lock() else {
        return false;
    };
    map.retain(|_, grant| grant.expires > now);
    let Some(grant) = map.get(token) else {
        return false;
    };
    if let Some(bound) = &grant.client_fp {
        if presented_fp != Some(bound.as_str()) {
            return false;
        }
    }
    map.remove(token);
    true
}

/// Asks every client for a certificate without requiring one, and accepts any certificate
/// whose handshake signature checks out: it is the client's proof that it holds the key
/// whose fingerprint a token may be bound to. Which certificates are acceptable is decided
/// per token by [`consume_token`], not here.
#[derive(Debug)]
struct AnyClientCertVerifier {
    supported: rustls::crypto::WebPkiSupportedAlgorithms,
}

impl rustls::server::danger::ClientCertVerifier for AnyClientCertVerifier {
    fn client_auth_mandatory(&self) -> bool {
        false
    }

    fn root_hint_subjects(&self) -> &[rustls::DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, rustls::Error> {
        Ok(rustls::server::danger::ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.supported)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.supported)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.supported.supported_schemes()
    }
}

/// SHA-256 hex fingerprint of the certificate the peer of `conn` presented, if any.
fn peer_cert_fingerprint(conn: &quinn::Connection) -> Option<String> {
    let identity = conn.peer_identity()?;
    let certs = identity.downcast::<Vec<CertificateDer<'static>>>().ok()?;
    certs
        .first()
        .map(|cert| hex_encode(&sha256_bytes(cert.as_ref())))
}

/// Generate a self-signed TLS 1.3 certificate and ephemeral session token.
pub fn generate_ephemeral_credentials(
    initial_token: String,
    initial_ttl: Duration,
) -> Result<EphemeralCredentials> {
    let cert = rcgen::generate_simple_self_signed(vec!["remotewayland.local".to_string()])
        .context("failed to generate self-signed TLS certificate")?;
    let cert_der = CertificateDer::from(cert.cert);
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));

    let digest = sha256_bytes(cert_der.as_ref());
    let cert_sha256_hex = hex_encode(&digest);

    let client_verifier = Arc::new(AnyClientCertVerifier {
        supported: rustls::crypto::ring::default_provider().signature_verification_algorithms,
    });
    let mut rustls_config = rustls::ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
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
    tokens.insert(
        initial_token,
        TokenGrant {
            expires: Instant::now() + initial_ttl,
            client_fp: None,
        },
    );

    Ok(EphemeralCredentials {
        server_config,
        cert_sha256_hex,
        valid_tokens: Arc::new(Mutex::new(tokens)),
    })
}

/// Where the QUIC server listens: an IP, and either a port range or (`None`) any free port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListenSpec {
    pub ip: IpAddr,
    pub ports: Option<PortRange>,
}

impl std::fmt::Display for ListenSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.ip, self.ports) {
            (IpAddr::V4(ip), None) => write!(f, "{ip}"),
            (IpAddr::V6(ip), None) => write!(f, "[{ip}]"),
            (IpAddr::V4(ip), Some(ports)) => write!(f, "{ip}:{ports}"),
            (IpAddr::V6(ip), Some(ports)) => write!(f, "[{ip}]:{ports}"),
        }
    }
}

/// Parses `IP`, `IP:PORT` or `IP:START-END` (IPv6 in brackets when a port follows, e.g.
/// `[::]:7443`). Port `0` means any free port, like leaving the port out.
impl std::str::FromStr for ListenSpec {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if let Ok(ip) = s.parse::<IpAddr>() {
            return Ok(Self { ip, ports: None });
        }
        let (ip, ports) = if let Some(rest) = s.strip_prefix('[') {
            let (ip, rest) = rest
                .split_once(']')
                .with_context(|| format!("'{s}': missing ']' after the IPv6 address"))?;
            match rest {
                "" => (ip, None),
                _ => {
                    let ports = rest.strip_prefix(':').with_context(|| {
                        format!("'{s}': expected ':' and a port after the IPv6 address")
                    })?;
                    (ip, Some(ports))
                }
            }
        } else {
            let (ip, ports) = s
                .rsplit_once(':')
                .with_context(|| format!("'{s}' is not an IP address, IP:PORT or IP:START-END"))?;
            (ip, Some(ports))
        };
        let ip: IpAddr = ip
            .parse()
            .with_context(|| format!("'{ip}' is not an IP address"))?;
        let ports = match ports.map(str::trim) {
            None | Some("0") => None,
            Some(ports) => Some(
                ports
                    .parse::<PortRange>()
                    .with_context(|| format!("invalid port or port range '{ports}'"))?,
            ),
        };
        Ok(Self { ip, ports })
    }
}

/// Bind the QUIC endpoint. Without a port range the OS picks any free port; with one, the
/// first free port in it is used, and binding fails if none is.
pub fn bind_endpoint(server_config: ServerConfig, listen: ListenSpec) -> Result<Endpoint> {
    let Some(range) = listen.ports else {
        let addr = SocketAddr::new(listen.ip, 0);
        return Endpoint::server(server_config, addr)
            .with_context(|| format!("failed to bind a UDP port on {}", listen.ip));
    };
    for port in range.ports() {
        let addr = SocketAddr::new(listen.ip, port);
        match Endpoint::server(server_config.clone(), addr) {
            Ok(endpoint) => return Ok(endpoint),
            Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("failed to bind UDP {addr}"));
            }
        }
    }
    if range.start() == range.end() {
        anyhow::bail!("UDP port {range} on {} is already in use", listen.ip)
    }
    anyhow::bail!(
        "every UDP port in {range} on {} is already in use",
        listen.ip
    )
}

/// How long a newly authenticated display client waits for the client it replaces to tear
/// down (release input, stop its audio bridge) before starting anyway.
const TAKEOVER_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// State shared by every connection to this server.
struct Shared {
    valid_tokens: TokenMap,
    cmd_tx: calloop::channel::Sender<CompositorCommand>,
    events: broadcast::Sender<CompositorEvent>,
    pty_mgr: Arc<crate::pty::PtyManager>,
    display: Arc<Mutex<DisplaySlot>>,
}

/// Handle incoming authenticated client sessions on an endpoint from [`bind_endpoint`].
pub async fn run_server(
    endpoint: Endpoint,
    creds: EphemeralCredentials,
    comp_handle: CompositorHandle,
    mut event_rx: mpsc::UnboundedReceiver<CompositorEvent>,
) -> Result<()> {
    let actual_addr = endpoint
        .local_addr()
        .context("failed to read the bound QUIC address")?;

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

    let shared = Arc::new(Shared {
        valid_tokens: Arc::clone(&creds.valid_tokens),
        cmd_tx: comp_handle.cmd_tx.clone(),
        pty_mgr: crate::pty::PtyManager::new(bcast_tx.clone()),
        events: bcast_tx,
        display: Arc::new(Mutex::new(DisplaySlot::default())),
    });
    while let Some(incoming) = endpoint.accept().await {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move {
            if let Err(err) = handle_connection(incoming, shared).await {
                warn!(error = %err, "client connection ended with error");
            }
        });
    }

    Ok(())
}

async fn handle_connection(incoming: quinn::Incoming, shared: Arc<Shared>) -> Result<()> {
    let conn = incoming.await.context("QUIC handshake failed")?;
    let remote = conn.remote_address();
    info!(%remote, "QUIC peer connected");

    let (mut send, mut recv) = conn
        .accept_bi()
        .await
        .context("failed to accept control stream")?;

    // The preamble's layout is independent of the message encoding, so a client of any
    // version can be told about a mismatch through the QUIC close reason.
    let mismatch = match read_protocol_preamble(&mut recv).await? {
        PeerProtocol::Version(PROTOCOL_VERSION) => None,
        PeerProtocol::Version(v) => Some(format!(
            "unsupported protocol version {v} (server speaks {PROTOCOL_VERSION})"
        )),
        PeerProtocol::Legacy => Some(format!(
            "client uses a protocol older than version 9 (server speaks {PROTOCOL_VERSION}); \
             update the client"
        )),
    };
    if let Some(reason) = mismatch {
        warn!(%remote, %reason, "rejecting client");
        conn.close(
            VarInt::from_u32(CLOSE_CODE_PROTOCOL_MISMATCH),
            reason.as_bytes(),
        );
        return Ok(());
    }

    let auth_msg = read_control_message(&mut recv)
        .await?
        .context("control stream closed before AuthRequest")?;

    let (token, role, supported_codecs, enable_audio_bridge, initial_scale) = match auth_msg {
        ControlMessage::AuthRequest {
            token,
            role,
            supported_codecs,
            enable_audio_bridge,
            initial_scale,
        } => (
            token,
            role,
            supported_codecs,
            enable_audio_bridge,
            initial_scale,
        ),
        other => {
            warn!(?other, "unexpected initial control message");
            return Ok(());
        }
    };

    let client_fp = peer_cert_fingerprint(&conn);
    if !consume_token(&shared.valid_tokens, &token, client_fp.as_deref()) {
        reject_auth(&mut send, "invalid or expired session token".into()).await?;
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

    let pty_streams = spawn_pty_stream_acceptor(conn.clone(), Arc::clone(&shared.pty_mgr));
    let res = match role {
        ClientRole::PtyOnly => {
            info!(%remote, "PTY-only client authenticated");
            // The control stream stays open (and unused) for the connection's lifetime.
            conn.closed().await;
            Ok(())
        }
        ClientRole::Display => {
            run_display_session(
                &conn,
                send,
                recv,
                &shared,
                selected_codec,
                enable_audio_bridge,
                initial_scale,
            )
            .await
        }
    };
    pty_streams.abort();
    res
}

/// How long to wait for the client to acknowledge an authentication rejection.
const REJECT_DELIVERY_TIMEOUT: Duration = Duration::from_secs(2);

async fn reject_auth(send: &mut quinn::SendStream, reason: String) -> Result<()> {
    write_control_message(
        send,
        &ControlMessage::AuthResponse {
            accepted: false,
            reason: Some(reason),
            selected_codec: VideoCodec::Lz4Delta,
        },
    )
    .await?;
    // The caller drops the connection next, which discards unacknowledged stream data;
    // wait until the client has the rejection so it can show the reason.
    let _ = send.finish();
    let _ = tokio::time::timeout(REJECT_DELIVERY_TIMEOUT, send.stopped()).await;
    Ok(())
}

/// Accept secondary bidirectional QUIC streams (persistent remote PTYs) on `conn`.
fn spawn_pty_stream_acceptor(
    conn: quinn::Connection,
    pty_mgr: Arc<crate::pty::PtyManager>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok((send_bi, mut recv_bi)) = conn.accept_bi().await {
            let pty = Arc::clone(&pty_mgr);
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
    })
}

/// The display client currently being served. The server drives one display client at a
/// time: windows, input, display scale and audio are all session-wide, so two clients
/// would fight over them.
#[derive(Default)]
struct DisplaySlot {
    next_id: u64,
    active: Option<ActiveDisplay>,
}

struct ActiveDisplay {
    id: u64,
    remote: SocketAddr,
    /// Closes the client's connection with [`CLOSE_CODE_SUPERSEDED`].
    supersede: Box<dyn FnOnce() + Send>,
    /// Resolves once the client's session has fully torn down (its lease was dropped).
    finished: oneshot::Receiver<()>,
}

/// Held by the display session that owns the slot; dropping it frees the slot (unless a
/// newer client already took it) and wakes a client waiting to take over.
struct DisplayLease {
    slot: Arc<Mutex<DisplaySlot>>,
    id: u64,
    _finished: oneshot::Sender<()>,
}

impl DisplayLease {
    /// Whether a newer display client has taken the session from this one.
    fn superseded(&self) -> bool {
        self.slot
            .lock()
            .map(|s| s.active.as_ref().is_none_or(|a| a.id != self.id))
            .unwrap_or(true)
    }
}

impl Drop for DisplayLease {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.slot.lock() {
            if slot.active.as_ref().is_some_and(|a| a.id == self.id) {
                slot.active = None;
            }
        }
    }
}

/// Make the caller the active display client. The previous one, if any, is superseded and
/// this waits (bounded) until its session has torn down, so the two never drive input or
/// run audio bridges at the same time. Call only after the client has authenticated, so
/// unauthenticated peers cannot kick the active session.
async fn take_over_display(
    slot: &Arc<Mutex<DisplaySlot>>,
    remote: SocketAddr,
    supersede: Box<dyn FnOnce() + Send>,
) -> DisplayLease {
    let (finished_tx, finished_rx) = oneshot::channel();
    let (id, previous) = {
        let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
        s.next_id += 1;
        let id = s.next_id;
        let previous = s.active.replace(ActiveDisplay {
            id,
            remote,
            supersede,
            finished: finished_rx,
        });
        (id, previous)
    };
    if let Some(previous) = previous {
        info!(
            previous = %previous.remote,
            new = %remote,
            "display session taken over by a newly connected client"
        );
        (previous.supersede)();
        if tokio::time::timeout(TAKEOVER_TEARDOWN_TIMEOUT, previous.finished)
            .await
            .is_err()
        {
            warn!(previous = %previous.remote, "previous display client did not tear down in time");
        }
    }
    DisplayLease {
        slot: Arc::clone(slot),
        id,
        _finished: finished_tx,
    }
}

static NEXT_SYNC_ID: AtomicU64 = AtomicU64::new(1);

/// A snapshot requested from the compositor whose [`CompositorEvent::SnapshotTaken`]
/// marker has not been seen on the event stream yet.
struct PendingSnapshot {
    sync_id: u64,
    reply: oneshot::Receiver<Vec<CompositorEvent>>,
}

fn request_snapshot(
    cmd_tx: &calloop::channel::Sender<CompositorCommand>,
) -> Result<PendingSnapshot> {
    let sync_id = NEXT_SYNC_ID.fetch_add(1, Ordering::Relaxed);
    let (reply_tx, reply) = oneshot::channel();
    cmd_tx
        .send(CompositorCommand::Snapshot {
            sync_id,
            reply: reply_tx,
        })
        .map_err(|_| anyhow!("compositor is no longer running"))?;
    Ok(PendingSnapshot { sync_id, reply })
}

async fn run_display_session(
    conn: &quinn::Connection,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    shared: &Shared,
    selected_codec: VideoCodec,
    enable_audio_bridge: bool,
    initial_scale: f64,
) -> Result<()> {
    let remote = conn.remote_address();
    let supersede_conn = conn.clone();
    let lease = take_over_display(
        &shared.display,
        remote,
        Box::new(move || {
            supersede_conn.close(
                VarInt::from_u32(CLOSE_CODE_SUPERSEDED),
                b"session opened by another client",
            );
        }),
    )
    .await;
    if lease.superseded() || conn.close_reason().is_some() {
        // Superseded or gone while waiting for the previous client to tear down.
        return Ok(());
    }

    info!(
        %remote,
        ?selected_codec,
        enable_audio_bridge,
        initial_scale,
        "display client authenticated; starting compositor/media loop"
    );
    let cmd_tx = shared.cmd_tx.clone();
    let pty_mgr = &shared.pty_mgr;
    // Subscribe only now: everything before this point is covered by the initial snapshot.
    let mut ev_rx = shared.events.subscribe();
    let _ = cmd_tx.send(CompositorCommand::SetDisplayScale {
        scale_factor: initial_scale,
    });
    let mut pending_snapshot = Some(request_snapshot(&cmd_tx)?);

    // Start bidirectional PipeWire <-> Opus audio bridge if requested.
    let (mut mic_active_rx, audio_bridge_guard) = if enable_audio_bridge {
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

    // Serialize and coalesce each client-synced desktop setting onto its own worker so
    // bursts cannot spawn unbounded blocking tasks or race on isolated portal config files.
    let color_scheme_tx = spawn_latest_applier(crate::session::apply_color_scheme);
    let button_layout_tx = spawn_latest_applier(|layout: Option<String>| {
        crate::session::apply_button_layout(layout.as_deref())
    });
    let accent_color_tx = spawn_latest_applier(crate::session::apply_accent_color);
    let contrast_tx = spawn_latest_applier(crate::session::apply_contrast);

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

    // Replies produced off the event loop (e.g. the app list, whose discovery blocks), so
    // slow work never stalls event forwarding and makes the event receiver lag.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<ControlMessage>();

    let mut fwd = EventForwarder::new(conn.clone(), selected_codec);

    // Bridge compositor events -> QUIC control/video streams and QUIC control messages -> compositor.
    let loop_res: Result<()> = async {
    loop {
        let next_popup_refresh = fwd.next_popup_refresh();

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
                fwd.send_popup_lossless_refresh(&mut send, wid).await?;
            }
            Some(active) = async {
                match mic_active_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                write_control_message(&mut send, &ControlMessage::SetMicActive { active }).await?;
            }
            Some(msg) = out_rx.recv() => {
                write_control_message(&mut send, &msg).await?;
            }
            ev = ev_rx.recv() => {
                match ev {
                    Ok(CompositorEvent::SnapshotTaken { sync_id })
                        if pending_snapshot.as_ref().is_some_and(|p| p.sync_id == sync_id) =>
                    {
                        let Some(pending) = pending_snapshot.take() else {
                            continue;
                        };
                        let mut events = pending
                            .reply
                            .await
                            .context("compositor dropped a snapshot request")?;
                        // PTY events reach the stream directly rather than through the
                        // compositor, so their snapshot is taken now: any PTY event still
                        // queued behind the marker is newer than the compositor snapshot
                        // and is applied after this one.
                        events.extend(pty_mgr.state_events());
                        fwd.apply_snapshot(&mut send, events).await?;
                    }
                    // Already reflected in the pending snapshot.
                    Ok(_) if pending_snapshot.is_some() => {}
                    Ok(ev) => fwd.forward(&mut send, ev).await?,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        // Dropped events may include windows or programs that opened or
                        // closed; bring the client back in line from a fresh snapshot.
                        warn!(skipped, "compositor event receiver lagged; resyncing client");
                        pending_snapshot = Some(request_snapshot(&cmd_tx)?);
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
                        let out_tx = out_tx.clone();
                        tokio::spawn(async move {
                            let apps = tokio::task::spawn_blocking(crate::apps::discover_apps)
                                .await
                                .unwrap_or_default();
                            let _ = out_tx.send(ControlMessage::AppList { apps });
                        });
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
                    Some(ControlMessage::ButtonLayout { layout }) => {
                        let _ = button_layout_tx.send(Some(layout));
                    }
                    Some(ControlMessage::AccentColor { rgb }) => {
                        let _ = accent_color_tx.send(Some(rgb));
                    }
                    Some(ControlMessage::Contrast { contrast }) => {
                        let _ = contrast_tx.send(Some(contrast));
                    }
                    Some(ControlMessage::FrameAck { window_id, pts_ns }) => {
                        fwd.ack_frame(window_id, pts_ns);
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

    // Tear down before releasing the slot, so a client taking over never overlaps with
    // this one's input or audio bridge.
    ctrl_reader_task.abort();
    fwd.shutdown();
    let _ = cmd_tx.send(CompositorCommand::ReleaseInput);
    drop(audio_bridge_guard);
    let superseded = lease.superseded();
    if conn.close_reason().is_none() {
        conn.close(0u32.into(), b"session ended");
    }
    drop(lease);

    if superseded {
        info!(%remote, "display client disconnected: session taken over by another client");
        return Ok(());
    }
    loop_res
}

/// Run `apply` on a blocking worker for the latest value sent on the returned channel,
/// skipping values superseded while a previous one was being applied.
fn spawn_latest_applier<T>(apply: impl Fn(T) + Clone + Send + 'static) -> watch::Sender<Option<T>>
where
    T: Clone + Send + Sync + 'static,
{
    let (tx, mut rx) = watch::channel::<Option<T>>(None);
    tokio::spawn(async move {
        while rx.changed().await.is_ok() {
            let Some(value) = rx.borrow_and_update().clone() else {
                continue;
            };
            let apply = apply.clone();
            let _ = tokio::task::spawn_blocking(move || apply(value)).await;
        }
    });
    tx
}

struct PopupEncoderState {
    encoder: WindowVideoEncoder,
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    geometry: FrameRect,
    input_bounds: FrameRect,
    last_update: Instant,
}

const POPUP_IDLE_REFRESH_DELAY: Duration = Duration::from_millis(120);

/// Forwards compositor events to one display client and tracks what that client has been
/// told, so a snapshot can retract windows and programs it missed the end of.
struct EventForwarder {
    conn: quinn::Connection,
    selected_codec: VideoCodec,
    session_start: Instant,
    video_pipelines: HashMap<WindowId, WindowVideoPipeline>,
    popup_encoders: HashMap<WindowId, PopupEncoderState>,
    /// Windows and popups announced to the client and not closed since.
    announced_windows: HashSet<WindowId>,
    /// Programs (launched apps, window entries, PTYs) the client shows as live, by launch id.
    live_programs: HashMap<u64, String>,
}

impl EventForwarder {
    fn new(conn: quinn::Connection, selected_codec: VideoCodec) -> Self {
        Self {
            conn,
            selected_codec,
            session_start: Instant::now(),
            video_pipelines: HashMap::new(),
            popup_encoders: HashMap::new(),
            announced_windows: HashSet::new(),
            live_programs: HashMap::new(),
        }
    }

    /// The popup whose pending lossless refresh is due next, and how long until it is.
    fn next_popup_refresh(&self) -> Option<(WindowId, Duration)> {
        self.popup_encoders
            .iter()
            .filter(|(_, st)| st.encoder.has_pending_lossless_refresh())
            .map(|(&wid, st)| {
                (
                    wid,
                    POPUP_IDLE_REFRESH_DELAY.saturating_sub(st.last_update.elapsed()),
                )
            })
            .min_by_key(|&(_, wait)| wait)
    }

    async fn send_popup_lossless_refresh(
        &mut self,
        send: &mut quinn::SendStream,
        wid: WindowId,
    ) -> Result<()> {
        let pts_ns = self.session_start.elapsed().as_nanos() as u64;
        let Some(st) = self.popup_encoders.get_mut(&wid) else {
            return Ok(());
        };
        let Some(frame) = st
            .encoder
            .encode_lossless_refresh(st.width, st.height, pts_ns)
        else {
            return Ok(());
        };
        st.last_update = Instant::now();
        write_control_message(
            send,
            &ControlMessage::PopupFrame {
                window_id: wid,
                width: frame.width,
                height: frame.height,
                logical_width: st.logical_width,
                logical_height: st.logical_height,
                geometry: st.geometry,
                input_bounds: st.input_bounds,
                is_keyframe: frame.is_keyframe,
                bgra: frame.data,
            },
        )
        .await
    }

    fn ack_frame(&self, window_id: WindowId, pts_ns: u64) {
        if let Some(pipeline) = self.video_pipelines.get(&window_id) {
            pipeline.ack_frame(pts_ns);
        }
    }

    /// Bring the client in line with `snapshot`: retract what it shows that no longer
    /// exists, then replay the snapshot (re-announcing what already exists is idempotent).
    async fn apply_snapshot(
        &mut self,
        send: &mut quinn::SendStream,
        snapshot: Vec<CompositorEvent>,
    ) -> Result<()> {
        let (stale_windows, stale_programs) =
            stale_entries(&self.announced_windows, &self.live_programs, &snapshot);
        for window_id in stale_windows {
            self.forward(send, CompositorEvent::WindowClosed { window_id })
                .await?;
        }
        for (launch_id, name) in stale_programs {
            // The real exit code is unknown; clients only use it to drop the entry.
            self.forward(
                send,
                CompositorEvent::AppState {
                    launch_id,
                    name,
                    state: LaunchState::Exited(0),
                },
            )
            .await?;
        }
        for ev in snapshot {
            self.forward(send, ev).await?;
        }
        Ok(())
    }

    async fn forward(&mut self, send: &mut quinn::SendStream, ev: CompositorEvent) -> Result<()> {
        match ev {
            CompositorEvent::WindowCreated {
                window_id,
                title,
                app_id,
                width,
                height,
            } => {
                self.announced_windows.insert(window_id);
                write_control_message(
                    send,
                    &ControlMessage::WindowCreated {
                        window_id,
                        title,
                        app_id,
                        width,
                        height,
                    },
                )
                .await?;
            }
            CompositorEvent::PopupCreated {
                window_id,
                parent_id,
                x,
                y,
                width,
                height,
            } => {
                self.announced_windows.insert(window_id);
                write_control_message(
                    send,
                    &ControlMessage::PopupCreated {
                        window_id,
                        parent_id,
                        x,
                        y,
                        width,
                        height,
                    },
                )
                .await?;
            }
            CompositorEvent::PopupRepositioned {
                window_id,
                x,
                y,
                width,
                height,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::PopupRepositioned {
                        window_id,
                        x,
                        y,
                        width,
                        height,
                    },
                )
                .await?;
            }
            CompositorEvent::PopupFrame {
                window_id,
                width,
                height,
                logical_width,
                logical_height,
                geometry,
                input_bounds,
                bgra,
            } => {
                let selected_codec = self.selected_codec;
                let st =
                    self.popup_encoders
                        .entry(window_id)
                        .or_insert_with(|| PopupEncoderState {
                            encoder: WindowVideoEncoder::new(selected_codec),
                            width,
                            height,
                            logical_width,
                            logical_height,
                            geometry,
                            input_bounds,
                            last_update: Instant::now(),
                        });
                st.width = width;
                st.height = height;
                st.logical_width = logical_width;
                st.logical_height = logical_height;
                st.geometry = geometry;
                st.input_bounds = input_bounds;
                st.last_update = Instant::now();
                let pts_ns = self.session_start.elapsed().as_nanos() as u64;
                let frame = st.encoder.encode_bgra(width, height, pts_ns, &bgra, 80);
                write_control_message(
                    send,
                    &ControlMessage::PopupFrame {
                        window_id,
                        width: frame.width,
                        height: frame.height,
                        logical_width,
                        logical_height,
                        geometry,
                        input_bounds,
                        is_keyframe: frame.is_keyframe,
                        bgra: frame.data,
                    },
                )
                .await?;
            }
            CompositorEvent::StartMove { window_id } => {
                write_control_message(send, &ControlMessage::StartMove { window_id }).await?;
            }
            CompositorEvent::StartResize { window_id, edge } => {
                write_control_message(send, &ControlMessage::StartResize { window_id, edge })
                    .await?;
            }
            CompositorEvent::SetMaximized {
                window_id,
                maximized,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::SetMaximized {
                        window_id,
                        maximized,
                    },
                )
                .await?;
            }
            CompositorEvent::SetMinimized { window_id } => {
                write_control_message(send, &ControlMessage::SetMinimized { window_id }).await?;
            }
            CompositorEvent::SetFullscreen {
                window_id,
                fullscreen,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::SetFullscreen {
                        window_id,
                        fullscreen,
                    },
                )
                .await?;
            }
            CompositorEvent::ShowWindowMenu { window_id, x, y } => {
                write_control_message(send, &ControlMessage::ShowWindowMenu { window_id, x, y })
                    .await?;
            }
            CompositorEvent::SetDecorationMode {
                window_id,
                server_side,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::SetDecorationMode {
                        window_id,
                        server_side,
                    },
                )
                .await?;
            }
            CompositorEvent::WindowConfigured {
                window_id,
                width,
                height,
                scale_factor,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::WindowConfigured {
                        window_id,
                        width,
                        height,
                        scale_factor,
                        maximized: false,
                        fullscreen: false,
                        activated: true,
                        tiled: false,
                    },
                )
                .await?;
            }
            CompositorEvent::WindowClosed { window_id } => {
                self.announced_windows.remove(&window_id);
                if let Some(pipeline) = self.video_pipelines.remove(&window_id) {
                    pipeline.task.abort();
                }
                self.popup_encoders.remove(&window_id);
                write_control_message(send, &ControlMessage::WindowClosed { window_id }).await?;
            }
            CompositorEvent::CursorUpdate {
                window_id,
                visible,
                shape_name,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::CursorUpdate {
                        window_id,
                        visible,
                        shape_name,
                    },
                )
                .await?;
            }
            CompositorEvent::CursorBitmap {
                window_id,
                width,
                height,
                logical_width,
                logical_height,
                hotspot_x,
                hotspot_y,
                bgra,
            } => {
                write_control_message(
                    send,
                    &ControlMessage::CursorBitmap {
                        window_id,
                        width,
                        height,
                        logical_width,
                        logical_height,
                        hotspot_x,
                        hotspot_y,
                        bgra,
                    },
                )
                .await?;
            }
            CompositorEvent::ClipboardText { text } => {
                write_control_message(send, &ControlMessage::ClipboardText { text }).await?;
            }
            CompositorEvent::PrimarySelectionText { text } => {
                write_control_message(send, &ControlMessage::PrimarySelectionText { text }).await?;
            }
            CompositorEvent::AppState {
                launch_id,
                name,
                state,
            } => {
                if matches!(state, LaunchState::Exited(_)) {
                    self.live_programs.remove(&launch_id);
                } else {
                    self.live_programs.insert(launch_id, name.clone());
                }
                write_control_message(
                    send,
                    &ControlMessage::AppState {
                        launch_id,
                        name,
                        state,
                    },
                )
                .await?;
            }
            CompositorEvent::FrameCommitted {
                window_id,
                width,
                height,
                logical_width,
                logical_height,
                geometry,
                input_bounds,
                pixels,
            } => {
                let (conn, selected_codec, session_start) =
                    (&self.conn, self.selected_codec, self.session_start);
                let pipeline = self.video_pipelines.entry(window_id).or_insert_with(|| {
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
                    geometry,
                    input_bounds,
                    pixels,
                });
            }
            CompositorEvent::SnapshotTaken { .. } => {}
        }
        Ok(())
    }

    fn shutdown(&mut self) {
        for (_, pipeline) in self.video_pipelines.drain() {
            pipeline.task.abort();
        }
    }
}

/// What a client tracking `announced_windows` and `live_programs` shows that `snapshot`
/// no longer contains: windows/popups to close and programs to mark exited.
fn stale_entries(
    announced_windows: &HashSet<WindowId>,
    live_programs: &HashMap<u64, String>,
    snapshot: &[CompositorEvent],
) -> (Vec<WindowId>, Vec<(u64, String)>) {
    let mut windows = HashSet::new();
    let mut programs = HashSet::new();
    for ev in snapshot {
        match ev {
            CompositorEvent::WindowCreated { window_id, .. }
            | CompositorEvent::PopupCreated { window_id, .. } => {
                windows.insert(*window_id);
            }
            CompositorEvent::AppState {
                launch_id, state, ..
            } if !matches!(state, LaunchState::Exited(_)) => {
                programs.insert(*launch_id);
            }
            _ => {}
        }
    }
    let mut stale_windows: Vec<WindowId> = announced_windows
        .iter()
        .filter(|id| !windows.contains(*id))
        .copied()
        .collect();
    stale_windows.sort_unstable();
    let mut stale_programs: Vec<(u64, String)> = live_programs
        .iter()
        .filter(|(id, _)| !programs.contains(*id))
        .map(|(id, name)| (*id, name.clone()))
        .collect();
    stale_programs.sort_unstable();
    (stale_windows, stale_programs)
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
    geometry: FrameRect,
    input_bounds: FrameRect,
    pixels: crate::gpu::FramePixels,
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
    let mut last_dims: Option<(u32, u32, u32, u32, FrameRect, FrameRect)> = None;
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
        let (logical_width, logical_height, geometry, input_bounds, frame) = if do_idle_refresh {
            // Re-check mailbox in case a new frame arrived while waiting for credit.
            if latest_frame.lock().unwrap().is_some() {
                refining_idle = false;
                continue;
            }
            let Some((lw_phys, lh_phys, log_w, log_h, geometry, input_bounds)) = last_dims else {
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
            (log_w, log_h, geometry, input_bounds, frame)
        } else {
            // 3. Take the latest frame AFTER acquiring credit so stale intermediate frames are skipped.
            let Some(raw) = latest_frame.lock().unwrap().take() else {
                continue;
            };
            last_pts_ns = pts_ns;
            let logical_width = raw.logical_width;
            let logical_height = raw.logical_height;
            let (geometry, input_bounds) = (raw.geometry, raw.input_bounds);
            last_dims = Some((
                raw.width,
                raw.height,
                logical_width,
                logical_height,
                geometry,
                input_bounds,
            ));

            // 4. Encode off the async runtime thread so control/input latency stays unaffected.
            let mut enc = encoder.take().unwrap();
            let q = quality;
            let encoded_res = tokio::task::spawn_blocking(move || {
                let res = enc.encode_frame(raw.width, raw.height, pts_ns, &raw.pixels, q);
                (enc, res)
            })
            .await;
            let (enc, frame_res) = match encoded_res {
                Ok(pair) => pair,
                Err(err) => {
                    warn!(window_id, error = %err, "video encoder task panicked");
                    let _ = stream.finish();
                    return;
                }
            };
            encoder = Some(enc);
            let frame = match frame_res {
                Ok(f) => f,
                Err(err) => {
                    warn!(window_id, error = %format!("{err:#}"), "video frame encode failed");
                    continue;
                }
            };
            (logical_width, logical_height, geometry, input_bounds, frame)
        };
        let encode_elapsed = encode_start.elapsed();

        // 5. Write encoded frame to the per-window QUIC stream and adapt DCT quality dynamically.
        let Ok(payload_len) = u32::try_from(frame.data.len()) else {
            warn!(
                window_id,
                bytes = frame.data.len(),
                "encoded video frame exceeds u32::MAX; dropping"
            );
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
            geometry,
            input_bounds,
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

    fn grant(expires: Instant, client_fp: Option<&str>) -> TokenGrant {
        TokenGrant {
            expires,
            client_fp: client_fp.map(str::to_string),
        }
    }

    #[test]
    fn single_use_and_expiring_tokens() {
        let map = Mutex::new(HashMap::new());
        {
            let mut guard = map.lock().unwrap();
            let later = Instant::now() + Duration::from_secs(60);
            guard.insert("valid".to_string(), grant(later, None));
            guard.insert(
                "expired".to_string(),
                grant(Instant::now() - Duration::from_secs(1), None),
            );
        }

        // First use of valid token succeeds; second use fails (single-use). An unbound token
        // does not care whether a client certificate was presented.
        assert!(consume_token(&map, "valid", Some("ab")));
        assert!(!consume_token(&map, "valid", None));

        // Expired token is rejected and pruned.
        assert!(!consume_token(&map, "expired", None));
        assert!(!consume_token(&map, "nonexistent", None));
        assert!(map.lock().unwrap().is_empty());
    }

    #[test]
    fn bound_tokens_need_the_bound_certificate() {
        let map = Mutex::new(HashMap::new());
        let later = Instant::now() + Duration::from_secs(60);
        map.lock()
            .unwrap()
            .insert("bound".to_string(), grant(later, Some("aa")));

        // Wrong or missing certificate: rejected, and the token stays usable.
        assert!(!consume_token(&map, "bound", None));
        assert!(!consume_token(&map, "bound", Some("bb")));
        assert!(consume_token(&map, "bound", Some("aa")));
        assert!(!consume_token(&map, "bound", Some("aa")));
    }

    #[test]
    fn listen_specs_parse_and_round_trip() {
        let any_v4 = ListenSpec {
            ip: "0.0.0.0".parse().unwrap(),
            ports: None,
        };
        let cases = [
            ("0.0.0.0", any_v4, "0.0.0.0"),
            ("0.0.0.0:0", any_v4, "0.0.0.0"),
            (
                "127.0.0.1:7443",
                ListenSpec {
                    ip: "127.0.0.1".parse().unwrap(),
                    ports: Some("7443".parse().unwrap()),
                },
                "127.0.0.1:7443",
            ),
            (
                "0.0.0.0:7443-7462",
                ListenSpec {
                    ports: Some("7443-7462".parse().unwrap()),
                    ..any_v4
                },
                "0.0.0.0:7443-7462",
            ),
            (
                "::",
                ListenSpec {
                    ip: "::".parse().unwrap(),
                    ports: None,
                },
                "[::]",
            ),
            (
                "[::1]:7443-7444",
                ListenSpec {
                    ip: "::1".parse().unwrap(),
                    ports: Some("7443-7444".parse().unwrap()),
                },
                "[::1]:7443-7444",
            ),
        ];
        for (input, spec, display) in cases {
            let parsed: ListenSpec = input.parse().unwrap();
            assert_eq!(parsed, spec, "{input}");
            assert_eq!(parsed.to_string(), display, "{input}");
            assert_eq!(display.parse::<ListenSpec>().unwrap(), spec, "{display}");
        }
        for bad in [
            "",
            "7443",
            "host:7443",
            "0.0.0.0:",
            "0.0.0.0:9-1",
            "[::1",
            "[::1]7443",
        ] {
            assert!(
                bad.parse::<ListenSpec>().is_err(),
                "{bad:?} should not parse"
            );
        }
    }

    fn test_server_config() -> ServerConfig {
        generate_ephemeral_credentials("t".into(), Duration::from_secs(1))
            .unwrap()
            .server_config
    }

    fn listen(spec: &str) -> ListenSpec {
        spec.parse().unwrap()
    }

    #[tokio::test]
    async fn bind_endpoint_stays_inside_the_requested_ports() {
        let any = bind_endpoint(test_server_config(), listen("127.0.0.1")).unwrap();
        let taken = any.local_addr().unwrap().port();
        assert_ne!(taken, 0);

        // A single port that is taken fails instead of falling back to another port.
        let err = bind_endpoint(test_server_config(), listen(&format!("127.0.0.1:{taken}")))
            .err()
            .unwrap();
        assert!(err.to_string().contains("already in use"), "{err:#}");

        // A range skips the taken port and stays inside the range.
        let end = taken.saturating_add(8);
        let ranged = bind_endpoint(
            test_server_config(),
            listen(&format!("127.0.0.1:{taken}-{end}")),
        )
        .unwrap();
        let port = ranged.local_addr().unwrap().port();
        assert!(port > taken && port <= end, "bound {port}");
    }

    fn app_state(launch_id: u64, state: LaunchState) -> CompositorEvent {
        CompositorEvent::AppState {
            launch_id,
            name: format!("app {launch_id}"),
            state,
        }
    }

    #[test]
    fn stale_entries_are_what_the_client_shows_but_the_snapshot_lacks() {
        let announced = HashSet::from([1, 2, 3, 4]);
        let live = HashMap::from([
            (10, "app 10".to_string()),
            (11, "app 11".to_string()),
            (12, "app 12".to_string()),
        ]);
        let snapshot = vec![
            CompositorEvent::WindowCreated {
                window_id: 1,
                title: String::new(),
                app_id: String::new(),
                width: 1,
                height: 1,
            },
            CompositorEvent::PopupCreated {
                window_id: 3,
                parent_id: 1,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            // New since the client last heard: not stale, just announced by the replay.
            CompositorEvent::WindowCreated {
                window_id: 5,
                title: String::new(),
                app_id: String::new(),
                width: 1,
                height: 1,
            },
            app_state(10, LaunchState::Running),
            app_state(12, LaunchState::Exited(0)),
        ];

        let (windows, programs) = stale_entries(&announced, &live, &snapshot);
        assert_eq!(windows, vec![2, 4]);
        assert_eq!(
            programs,
            vec![(11, "app 11".to_string()), (12, "app 12".to_string())]
        );
    }

    fn superseder(flag: &Arc<std::sync::atomic::AtomicBool>) -> Box<dyn FnOnce() + Send> {
        let flag = Arc::clone(flag);
        Box::new(move || flag.store(true, Ordering::SeqCst))
    }

    #[tokio::test]
    async fn new_display_client_supersedes_and_waits_for_the_previous_one() {
        let slot = Arc::new(Mutex::new(DisplaySlot::default()));
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let first_closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let first = take_over_display(&slot, addr, superseder(&first_closed)).await;
        assert!(!first.superseded());

        let second_closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let takeover = {
            let slot = Arc::clone(&slot);
            let closed = Arc::clone(&second_closed);
            tokio::spawn(async move { take_over_display(&slot, addr, superseder(&closed)).await })
        };
        // The first client is told to go away, and the second waits for it to tear down.
        while !first_closed.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!takeover.is_finished());
        assert!(first.superseded());

        // Releasing the first lease lets the second through without freeing its slot.
        drop(first);
        let second = takeover.await.unwrap();
        assert!(!second.superseded());
        assert!(!second_closed.load(Ordering::SeqCst));

        drop(second);
        assert!(slot.lock().unwrap().active.is_none());
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
