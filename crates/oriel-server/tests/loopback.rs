//! End-to-end tests over loopback QUIC: the real server connection handling and display
//! session, driven by the real client, against a fake compositor that serves one window.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use calloop::channel::{self, Channel};
use oriel_client::window::{create_ui_channel, ClientUiEvent};
use oriel_proto::{read_control_message, ClientRole, FrameRect, WindowId, PROTOCOL_MAGIC};
use oriel_server::compositor::{CompositorCommand, CompositorEvent, CompositorHandle};
use oriel_server::damage::FrameDamage;
use oriel_server::gpu::FramePixels;
use oriel_server::quic::{
    bind_endpoint, generate_ephemeral_credentials, run_server_with_limits, AuthLimits, ListenSpec,
};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

const WINDOW: WindowId = 7;
const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;

/// A test pattern: horizontal and vertical gradients over a checkerboard.
fn pattern() -> Vec<u8> {
    let mut px = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let check = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            px.extend_from_slice(&[(x * 2) as u8, (y * 3) as u8, check, 255]);
        }
    }
    px
}

fn window_events() -> Vec<CompositorEvent> {
    let rect = FrameRect {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
    };
    vec![
        CompositorEvent::WindowCreated {
            window_id: WINDOW,
            title: "Loopback".into(),
            app_id: "test.loopback".into(),
            width: WIDTH,
            height: HEIGHT,
        },
        CompositorEvent::FrameCommitted {
            window_id: WINDOW,
            width: WIDTH,
            height: HEIGHT,
            logical_width: WIDTH,
            logical_height: HEIGHT,
            geometry: rect,
            input_bounds: rect,
            pixels: FramePixels::Cpu(Arc::new(pattern())),
            damage: FrameDamage::Full,
        },
    ]
}

/// Answer the compositor commands a display session sends: snapshots get one window with
/// one frame. Everything else is ignored.
fn run_fake_compositor(
    cmd_rx: Channel<CompositorCommand>,
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
) {
    std::thread::spawn(move || {
        let mut event_loop = calloop::EventLoop::<bool>::try_new().unwrap();
        event_loop
            .handle()
            .insert_source(cmd_rx, move |event, _, closed| match event {
                channel::Event::Msg(CompositorCommand::Snapshot { sync_id, reply }) => {
                    let _ = reply.send(window_events());
                    let _ = event_tx.send(CompositorEvent::SnapshotTaken { sync_id });
                }
                channel::Event::Msg(_) => {}
                channel::Event::Closed => *closed = true,
            })
            .unwrap();
        let mut closed = false;
        while !closed {
            event_loop
                .dispatch(Some(Duration::from_millis(50)), &mut closed)
                .unwrap();
        }
    });
}

struct TestServer {
    addr: SocketAddr,
    fingerprint: String,
    tokens: oriel_server::quic::TokenMap,
}

impl TestServer {
    /// A fresh single-use token, bound to `client_fp` if given.
    fn token(&self, client_fp: Option<&str>) -> String {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let token = format!(
            "test-{}",
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        self.tokens.lock().unwrap().insert(
            token.clone(),
            oriel_server::quic::TokenGrant {
                expires: Instant::now() + Duration::from_secs(60),
                client_fp: client_fp.map(str::to_string),
            },
        );
        token
    }
}

async fn start_server(limits: AuthLimits) -> TestServer {
    let creds = generate_ephemeral_credentials("unused".into(), Duration::from_secs(1)).unwrap();
    let listen = ListenSpec {
        ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
        ports: None,
    };
    let endpoint = bind_endpoint(creds.server_config.clone(), listen).unwrap();
    let addr = endpoint.local_addr().unwrap();
    let (cmd_tx, cmd_rx) = channel::channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    run_fake_compositor(cmd_rx, event_tx);
    let server = TestServer {
        addr,
        fingerprint: creds.cert_sha256_hex.clone(),
        tokens: Arc::clone(&creds.valid_tokens),
    };
    let handle = CompositorHandle {
        socket_name: "wayland-test".into(),
        cmd_tx,
    };
    tokio::spawn(run_server_with_limits(
        endpoint, creds, handle, event_rx, None, limits,
    ));
    server
}

async fn auth(server: &TestServer, token: String) -> anyhow::Result<quinn::Connection> {
    oriel_client::quic::connect_and_auth(
        server.addr,
        server.fingerprint.clone(),
        token,
        ClientRole::PtyOnly,
        false,
        1.0,
    )
    .await
    .map(|(conn, ..)| conn)
}

#[tokio::test(flavor = "multi_thread")]
async fn client_sees_the_window_and_its_frame() {
    let server = start_server(AuthLimits::default()).await;
    let (ui_tx, ui_rx) = create_ui_channel();
    let (events, frames) = ui_rx.into_headless();
    let (_ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
    let client = tokio::spawn(oriel_client::quic::run_client(
        server.addr,
        server.fingerprint.clone(),
        server.token(None),
        false,
        1.0,
        ui_tx,
        ctrl_rx,
        None,
        None,
    ));

    let events = tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut created = false;
        while Instant::now() < deadline {
            match events.recv_timeout(Duration::from_millis(100)) {
                Ok(ClientUiEvent::WindowCreated {
                    window_id,
                    title,
                    width,
                    height,
                    ..
                }) if window_id == WINDOW => {
                    assert_eq!((title.as_str(), width, height), ("Loopback", WIDTH, HEIGHT));
                    created = true;
                }
                Ok(ClientUiEvent::FrameReady { window_id }) if window_id == WINDOW => {
                    assert!(created, "frame arrived before its window");
                    return Some(events);
                }
                _ => {}
            }
        }
        None
    })
    .await
    .unwrap();
    assert!(events.is_some(), "no window and frame within 20s");

    // Lossy first frames settle into a lossless refresh: wait for the exact pixels.
    let expected = pattern();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut last = None;
    while Instant::now() < deadline {
        if let Some((w, h, bgra)) = frames.take(WINDOW) {
            assert_eq!((w, h), (WIDTH, HEIGHT));
            if *bgra == expected {
                client.abort();
                return;
            }
            last = Some(bgra);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!(
        "decoded frame never matched the source pixels (got a frame: {})",
        last.is_some()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tokens_are_single_use_and_wrong_ones_are_rejected() {
    let server = start_server(AuthLimits::default()).await;
    let token = server.token(None);
    auth(&server, token.clone()).await.expect("valid token");
    let err = auth(&server, token).await.expect_err("reused token");
    assert!(
        format!("{err:#}").contains("invalid or expired session token"),
        "{err:#}"
    );
    let err = auth(&server, "nonsense".into())
        .await
        .expect_err("unknown token");
    assert!(format!("{err:#}").contains("invalid or expired"), "{err:#}");
}

#[tokio::test(flavor = "multi_thread")]
async fn bound_tokens_need_the_clients_certificate() {
    let server = start_server(AuthLimits::default()).await;
    let other_fp = "ab".repeat(32);
    auth(&server, server.token(Some(&other_fp)))
        .await
        .expect_err("token bound to another certificate");
    let ours = &oriel_client::quic::client_identity().unwrap().fingerprint;
    auth(&server, server.token(Some(ours)))
        .await
        .expect("token bound to this client's certificate");
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_server_certificate_is_refused() {
    let server = start_server(AuthLimits::default()).await;
    let res = oriel_client::quic::connect_and_auth(
        server.addr,
        "00".repeat(32),
        server.token(None),
        ClientRole::PtyOnly,
        false,
        1.0,
    )
    .await;
    assert!(
        res.is_err(),
        "connected to a server with another certificate"
    );
}

/// Open a connection that does the QUIC handshake, then stalls before authenticating.
async fn stalled_connection(
    server: &TestServer,
) -> (quinn::Connection, quinn::SendStream, quinn::RecvStream) {
    let conn = oriel_client::quic::connect_raw(server.addr, server.fingerprint.clone())
        .await
        .unwrap();
    let (mut send, recv) = conn.open_bi().await.unwrap();
    send.write_all(&PROTOCOL_MAGIC).await.unwrap();
    send.write_u32(oriel_proto::PROTOCOL_VERSION).await.unwrap();
    // Announce a frame, then never send it.
    send.write_u32(1000).await.unwrap();
    send.flush().await.unwrap();
    (conn, send, recv)
}

#[tokio::test(flavor = "multi_thread")]
async fn unauthenticated_connections_time_out() {
    let server = start_server(AuthLimits {
        timeout: Duration::from_millis(500),
        max_pending: 32,
    })
    .await;
    let (conn, _send, _recv) = stalled_connection(&server).await;
    tokio::time::timeout(Duration::from_secs(5), conn.closed())
        .await
        .expect("server kept an unauthenticated connection past its deadline");
}

#[tokio::test(flavor = "multi_thread")]
async fn oversized_auth_frames_are_refused() {
    let server = start_server(AuthLimits::default()).await;
    let conn = oriel_client::quic::connect_raw(server.addr, server.fingerprint.clone())
        .await
        .unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send.write_all(&PROTOCOL_MAGIC).await.unwrap();
    send.write_u32(oriel_proto::PROTOCOL_VERSION).await.unwrap();
    send.write_u32(1024 * 1024).await.unwrap();
    send.flush().await.unwrap();
    // The server gives up on the stream instead of waiting for a megabyte.
    let res = tokio::time::timeout(Duration::from_secs(5), read_control_message(&mut recv))
        .await
        .expect("server waited for an oversized auth frame");
    assert!(!matches!(res, Ok(Some(_))), "server answered: {res:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn too_many_unauthenticated_connections_are_refused() {
    let server = start_server(AuthLimits {
        timeout: Duration::from_secs(30),
        max_pending: 2,
    })
    .await;
    let a = stalled_connection(&server).await;
    let b = stalled_connection(&server).await;
    let third = oriel_client::quic::connect_raw(server.addr, server.fingerprint.clone()).await;
    assert!(
        third.is_err(),
        "third unauthenticated connection was accepted"
    );
    // Authenticated clients are not held up by the stalled ones once they go.
    a.0.close(0u32.into(), b"done");
    b.0.close(0u32.into(), b"done");
    drop((a, b));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match auth(&server, server.token(None)).await {
            Ok(_) => break,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(err) => panic!("server stayed full: {err:#}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn version_mismatch_is_explained() {
    let server = start_server(AuthLimits::default()).await;
    let conn = oriel_client::quic::connect_raw(server.addr, server.fingerprint.clone())
        .await
        .unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send.write_all(&PROTOCOL_MAGIC).await.unwrap();
    send.write_u32(oriel_proto::PROTOCOL_VERSION + 1)
        .await
        .unwrap();
    send.flush().await.unwrap();
    let _ = read_control_message(&mut recv).await;
    let reason = match conn.closed().await {
        quinn::ConnectionError::ApplicationClosed(close) => {
            String::from_utf8_lossy(&close.reason).into_owned()
        }
        other => panic!("unexpected close: {other}"),
    };
    assert!(reason.contains("unsupported protocol version"), "{reason}");
}
