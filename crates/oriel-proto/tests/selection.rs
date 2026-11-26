//! Lazy selection transfer over a loopback QUIC connection: an owner serves a pipe the
//! selection's app writes to, a reader fetches it into a pipe a local app reads from.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr};
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::Duration;

use oriel_proto::selection::{fetch_selection_data, serve_selection_data};
use oriel_proto::{read_bidi_stream_header, BidiStreamHeader};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Connected (owner-side, reader-side) QUIC connections over loopback.
async fn quic_pair() -> (quinn::Connection, quinn::Connection) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_der = CertificateDer::from(cert.cert);
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));
    let server_cfg = quinn::ServerConfig::with_single_cert(vec![cert_der.clone()], key).unwrap();
    let server =
        quinn::Endpoint::server(server_cfg, SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).unwrap();
    let mut roots = rustls::RootCertStore::empty();
    roots.add(cert_der).unwrap();
    let mut client = quinn::Endpoint::client(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).unwrap();
    client.set_default_client_config(
        quinn::ClientConfig::with_root_certificates(Arc::new(roots)).unwrap(),
    );
    let addr = server.local_addr().unwrap();
    let accept = tokio::spawn(async move { server.accept().await.unwrap().await.unwrap() });
    let reader = client.connect(addr, "localhost").unwrap().await.unwrap();
    let owner = accept.await.unwrap();
    (owner, reader)
}

/// Serve `SelectionData` requests: offer 1 answers with `data` (written by a thread, like
/// an app would), any other offer is unavailable.
fn serve(conn: quinn::Connection, data: Arc<Vec<u8>>) {
    tokio::spawn(async move {
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            let data = Arc::clone(&data);
            tokio::spawn(async move {
                let BidiStreamHeader::SelectionData { offer_id, mime } =
                    read_bidi_stream_header(&mut recv).await.unwrap()
                else {
                    panic!("unexpected header");
                };
                assert_eq!(mime, "image/png");
                let source = (offer_id == 1).then(|| {
                    let (read, mut write) = std::io::pipe().unwrap();
                    std::thread::spawn(move || {
                        let _ = write.write_all(&data);
                    });
                    OwnedFd::from(read)
                });
                serve_selection_data(source, &mut send).await;
            });
        }
    });
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[2])
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fetches_large_data_and_reports_missing_offers() {
    let (owner, reader) = quic_pair().await;
    let data = Arc::new(pattern(8 * 1024 * 1024 + 17));
    serve(owner, Arc::clone(&data));

    let (mut app_read, app_write) = std::io::pipe().unwrap();
    let app = std::thread::spawn(move || {
        let mut got = Vec::new();
        app_read.read_to_end(&mut got).unwrap();
        got
    });
    fetch_selection_data(&reader, 1, "image/png".into(), app_write.into())
        .await
        .unwrap();
    assert_eq!(app.join().unwrap(), *data);

    // A selection that is gone: an error, and the app's pipe ends empty.
    let (mut app_read, app_write) = std::io::pipe().unwrap();
    assert!(
        fetch_selection_data(&reader, 2, "image/png".into(), app_write.into())
            .await
            .is_err()
    );
    let mut got = Vec::new();
    app_read.read_to_end(&mut got).unwrap();
    assert!(got.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stops_when_the_app_stops_reading() {
    let (owner, reader) = quic_pair().await;
    serve(owner, Arc::new(pattern(32 * 1024 * 1024)));
    let (mut app_read, app_write) = std::io::pipe().unwrap();
    let app = std::thread::spawn(move || {
        let mut first = [0u8; 1024];
        app_read.read_exact(&mut first).unwrap();
        // Closing the read end makes the next write fail with EPIPE.
    });
    let fetch = fetch_selection_data(&reader, 1, "image/png".into(), app_write.into());
    tokio::time::timeout(Duration::from_secs(10), fetch)
        .await
        .expect("fetch did not stop")
        .unwrap();
    app.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fetches_into_a_file_destination() {
    let (owner, reader) = quic_pair().await;
    let data = Arc::new(pattern(3 * 1024 * 1024 + 5));
    serve(owner, Arc::clone(&data));
    let path = std::env::temp_dir().join(format!("oriel-sel-dest-{}", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    fetch_selection_data(&reader, 1, "image/png".into(), file.into())
        .await
        .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), *data);
    let _ = std::fs::remove_file(&path);
}
