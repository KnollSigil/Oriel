//! End-to-end test of streamed file transfer: an owner serves files over a loopback QUIC
//! connection and a real FUSE mount presents them. Skipped when FUSE is unavailable.

use std::io::{Read, Seek, SeekFrom};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oriel_proto::xfer::{self, XferOwner};
use oriel_proto::xfer_fuse::XferMount;
use oriel_proto::{read_bidi_stream_header, BidiStreamHeader};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oriel-xfer-it-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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

fn serve(conn: quinn::Connection, owner: Arc<XferOwner>) {
    tokio::spawn(async move {
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            let owner = Arc::clone(&owner);
            tokio::spawn(async move {
                let res = match read_bidi_stream_header(&mut recv).await.unwrap() {
                    BidiStreamHeader::FileList { offer_id, path } => {
                        xfer::serve_list(&owner, offer_id, &path, &mut send).await
                    }
                    BidiStreamHeader::FileRead {
                        offer_id,
                        path,
                        offset,
                    } => xfer::serve_read(&owner, offer_id, &path, offset, &mut send).await,
                    other => panic!("unexpected header {other:?}"),
                };
                if res.is_ok() {
                    let _ = send.finish();
                }
            });
        }
    });
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[1] ^ seed)
        .collect()
}

fn read_all(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn files_stream_through_fuse() {
    let src = scratch("src");
    let big = pattern(40 * 1024 * 1024 + 123, 7);
    std::fs::write(src.join("big.bin"), &big).unwrap();
    std::fs::create_dir_all(src.join("tree/sub")).unwrap();
    std::fs::write(src.join("tree/a.txt"), b"alpha").unwrap();
    std::fs::write(src.join("tree/sub/b.txt"), b"beta").unwrap();

    let (owner_conn, reader_conn) = quic_pair().await;
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn, Arc::clone(&owner));

    let mnt_dir = scratch("mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };

    let (id, roots) = owner
        .offer(&[src.join("big.bin"), src.join("tree")])
        .unwrap();
    let paths = mount.add_offer(reader_conn.clone(), id, roots);
    assert_eq!(paths.len(), 2);

    let (big_path, tree_path) = (paths[0].clone(), paths[1].clone());
    tokio::task::spawn_blocking(move || {
        // stat comes from the offer, no network round trip.
        assert_eq!(
            std::fs::metadata(&big_path).unwrap().len(),
            big.len() as u64
        );
        // Sequential copy.
        assert!(read_all(&big_path) == big, "big file mismatch");
        // Random-offset reads, including backwards seeks and a read past EOF.
        let mut f = std::fs::File::open(&big_path).unwrap();
        for &off in &[
            30_000_000u64,
            5,
            39_999_000,
            1_000_000,
            big.len() as u64 - 10,
        ] {
            let mut buf = vec![0u8; 4096];
            f.seek(SeekFrom::Start(off)).unwrap();
            let n = f.read(&mut buf).unwrap();
            assert_eq!(&buf[..n], &big[off as usize..off as usize + n], "at {off}");
        }
        // Directory tree, listed lazily.
        let mut names: Vec<_> = std::fs::read_dir(&tree_path)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["a.txt", "sub"]);
        assert_eq!(read_all(&tree_path.join("sub/b.txt")), b"beta");
        assert_eq!(read_all(&tree_path.join("a.txt")), b"alpha");
        // Read-only.
        assert!(std::fs::write(tree_path.join("a.txt"), b"x").is_err());
        assert!(std::fs::metadata(tree_path.join("missing")).is_err());
    })
    .await
    .unwrap();

    // A newer offer retires the old one, but it stays readable during the grace period.
    let (id2, roots2) = owner.offer(&[src.join("tree/a.txt")]).unwrap();
    let paths2 = mount.add_offer(reader_conn.clone(), id2, roots2);
    let old_tree = paths[1].clone();
    tokio::task::spawn_blocking(move || {
        assert_eq!(read_all(&paths2[0]), b"alpha");
        assert_eq!(read_all(&old_tree.join("a.txt")), b"alpha");
    })
    .await
    .unwrap();

    // When the owner goes away, reads fail instead of hanging (use a file whose pages
    // are not in the kernel's cache yet).
    std::fs::write(src.join("fresh.bin"), pattern(100_000, 3)).unwrap();
    let (id3, roots3) = owner.offer(&[src.join("fresh.bin")]).unwrap();
    let gone = mount.add_offer(reader_conn.clone(), id3, roots3)[0].clone();
    reader_conn.close(0u32.into(), b"bye");
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || {
            let mut f = std::fs::File::open(&gone)?;
            f.seek(SeekFrom::Start(12_345))?;
            let mut buf = [0u8; 16];
            f.read(&mut buf)
        }),
    )
    .await
    .expect("read must not hang")
    .unwrap();
    assert!(res.is_err(), "read after disconnect should fail: {res:?}");

    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}
