//! End-to-end test of streamed file transfer: an owner serves files over a loopback QUIC
//! connection and a real FUSE mount presents them. Skipped when FUSE is unavailable.

use std::io::{Read, Seek, SeekFrom};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oriel_proto::transfers;
use oriel_proto::xfer::{self, OfferSlot, XferOwner};
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
        let conn_id = conn.stable_id();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            let owner = Arc::clone(&owner);
            tokio::spawn(async move {
                let header = read_bidi_stream_header(&mut recv).await.unwrap();
                assert!(
                    matches!(
                        header,
                        BidiStreamHeader::FileList { .. } | BidiStreamHeader::FileRead { .. }
                    ),
                    "unexpected header {header:?}"
                );
                xfer::serve_request(&owner, conn_id, &header, &mut send).await;
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
    let _live = live(&owner_conn, &reader_conn);
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
        .offer(
            OfferSlot::Clipboard,
            &[src.join("big.bin"), src.join("tree")],
        )
        .unwrap();
    let paths = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots);
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
    let (id2, roots2) = owner
        .offer(OfferSlot::Clipboard, &[src.join("tree/a.txt")])
        .unwrap();
    let paths2 = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id2, roots2);
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
    let (id3, roots3) = owner
        .offer(OfferSlot::Clipboard, &[src.join("fresh.bin")])
        .unwrap();
    let gone = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id3, roots3)[0].clone();
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

/// The one transfer listed for `conn`, once it has run long enough to be listed.
/// Keep both connections' cancels and pauses remembered, as their handlers do.
fn live(
    a: &quinn::Connection,
    b: &quinn::Connection,
) -> (transfers::ConnectionGuard, transfers::ConnectionGuard) {
    (
        transfers::ConnectionGuard::new(a.stable_id()),
        transfers::ConnectionGuard::new(b.stable_id()),
    )
}

/// Deliver the owner's queued pause requests to the reader, as `PauseTransfer` does.
fn deliver_requests(owner_conn: &quinn::Connection, reader_conn: &quinn::Connection) -> usize {
    let requests = transfers::take_peer_requests(owner_conn.stable_id());
    for r in &requests {
        let file = transfers::TransferFile {
            direction: transfers::TransferDirection::Incoming,
            offer_id: r.offer_id,
            path: r.path.clone(),
        };
        transfers::set_paused_file(reader_conn.stable_id(), &file, r.paused);
    }
    requests.len()
}

async fn only_transfer(conn: &quinn::Connection) -> transfers::TransferInfo {
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let listed = transfers::list(conn.stable_id());
    assert_eq!(listed.len(), 1, "{listed:?}");
    listed.into_iter().next().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transfers_are_listed_and_can_be_cancelled() {
    let src = scratch("cancel-src");
    let len = 40 * 1024 * 1024;
    std::fs::write(src.join("in.bin"), pattern(len, 1)).unwrap();
    std::fs::write(src.join("out.bin"), pattern(len, 2)).unwrap();
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("cancel-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let (id, roots) = owner
        .offer(
            OfferSlot::Clipboard,
            &[src.join("in.bin"), src.join("out.bin")],
        )
        .unwrap();
    let paths = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots);

    // Start reading a file and stop, like a paused copy; returns the open file.
    let start = |path: std::path::PathBuf| {
        tokio::task::spawn_blocking(move || {
            let mut f = std::fs::File::open(path).unwrap();
            let mut buf = vec![0u8; 1024 * 1024];
            f.read_exact(&mut buf).unwrap();
            f
        })
    };

    // Cancelled on the reading side: the next read fails.
    let mut f = start(paths[0].clone()).await.unwrap();
    let t = only_transfer(&reader_conn).await;
    assert_eq!(t.name, "in.bin");
    assert_eq!(t.direction, transfers::TransferDirection::Incoming);
    assert_eq!(t.size, len as u64);
    assert!(t.position >= 1024 * 1024, "{t:?}");
    assert!(
        transfers::list(owner_conn.stable_id()).len() == 1,
        "the owner serves it"
    );
    assert!(transfers::cancel(t.id));
    let (f, res) = tokio::task::spawn_blocking(move || {
        f.seek(SeekFrom::Start(30 * 1024 * 1024)).unwrap();
        let mut buf = [0u8; 4096];
        let res = f.read(&mut buf);
        (f, res)
    })
    .await
    .unwrap();
    // The page cache reports any failed read as EIO, whatever the filesystem said.
    assert!(res.is_err(), "read after cancel should fail: {res:?}");
    drop(f);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        transfers::list(reader_conn.stable_id()).is_empty(),
        "closed"
    );

    // Cancelled on the serving side: the reader gets an error, not a short file.
    let f = start(paths[1].clone()).await.unwrap();
    let t = only_transfer(&owner_conn).await;
    assert_eq!(t.direction, transfers::TransferDirection::Outgoing);
    assert!(transfers::cancel(t.id));
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || {
            let mut f = f;
            let mut rest = Vec::new();
            f.read_to_end(&mut rest).map(|_| rest.len())
        }),
    )
    .await
    .expect("read must not hang")
    .unwrap();
    assert!(
        res.is_err(),
        "read after the owner cancelled should fail: {res:?}"
    );

    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_transfers_hold_reads_and_stop_the_sender() {
    let src = scratch("pause-src");
    let len = 64 * 1024 * 1024;
    let data = pattern(len, 9);
    std::fs::write(src.join("p.bin"), &data).unwrap();
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("pause-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let (id, roots) = owner
        .offer(OfferSlot::Clipboard, &[src.join("p.bin")])
        .unwrap();
    let path = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots)[0].clone();
    let f = tokio::task::spawn_blocking(move || {
        let mut f = std::fs::File::open(path).unwrap();
        let mut buf = vec![0u8; 1024 * 1024];
        f.read_exact(&mut buf).unwrap();
        f
    })
    .await
    .unwrap();
    let t = only_transfer(&reader_conn).await;
    assert!(transfers::set_paused(t.id, true));
    assert!(transfers::list(reader_conn.stable_id())[0].paused);

    // A read past what is cached waits instead of failing.
    let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
    let expect = data[40 * 1024 * 1024..40 * 1024 * 1024 + 4096].to_vec();
    tokio::task::spawn_blocking(move || {
        let mut f = f;
        f.seek(SeekFrom::Start(40 * 1024 * 1024)).unwrap();
        let mut buf = vec![0u8; 4096];
        let res = f.read_exact(&mut buf).map(|()| buf);
        let _ = done_tx.send(res);
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    let sent = owner_conn.stats().udp_tx.bytes;
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        done_rx.try_recv().is_err(),
        "the read must wait while paused"
    );
    let idle = owner_conn.stats().udp_tx.bytes - sent;
    assert!(
        idle < 256 * 1024,
        "the sender kept sending while paused: {idle} bytes"
    );

    assert!(transfers::set_paused(t.id, false));
    let got = tokio::time::timeout(Duration::from_secs(10), done_rx)
        .await
        .expect("the read must finish once resumed")
        .unwrap()
        .unwrap();
    assert!(got == expect, "wrong data after resuming");

    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}

/// The owner cancels a transfer that its peer holds paused (what the launcher does for an
/// outgoing transfer): once the peer resumes, its waiting read fails instead of hanging,
/// and reopening the file fails too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_transfer_paused_by_the_reader_ends_its_reads() {
    let src = scratch("pause-cancel-src");
    let len = 64 * 1024 * 1024;
    std::fs::write(src.join("pc.bin"), pattern(len, 11)).unwrap();
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("pause-cancel-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let (id, roots) = owner
        .offer(OfferSlot::Clipboard, &[src.join("pc.bin")])
        .unwrap();
    let path = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots)[0].clone();
    let reopen = path.clone();
    let f = tokio::task::spawn_blocking(move || {
        let mut f = std::fs::File::open(path).unwrap();
        let mut buf = vec![0u8; 1024 * 1024];
        f.read_exact(&mut buf).unwrap();
        f
    })
    .await
    .unwrap();
    let outgoing = only_transfer(&owner_conn).await;
    let incoming = only_transfer(&reader_conn).await;
    // The owner pauses it: the reader is asked to hold its reads.
    assert!(transfers::set_paused(outgoing.id, true));
    assert_eq!(deliver_requests(&owner_conn, &reader_conn), 1);
    assert!(transfers::list(reader_conn.stable_id())[0].paused);
    assert_eq!(transfers::list(reader_conn.stable_id())[0].id, incoming.id);

    let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
    tokio::task::spawn_blocking(move || {
        let mut f = f;
        f.seek(SeekFrom::Start(40 * 1024 * 1024)).unwrap();
        let mut buf = vec![0u8; 4096];
        let _ = done_tx.send(f.read_exact(&mut buf).is_err());
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        done_rx.try_recv().is_err(),
        "the read must wait while paused"
    );

    // The owner cancels, which asks the reader to resume.
    assert!(transfers::cancel(outgoing.id));
    assert_eq!(deliver_requests(&owner_conn, &reader_conn), 1);
    let failed = tokio::time::timeout(Duration::from_secs(10), done_rx)
        .await
        .expect("the held read must end once resumed after the cancel")
        .unwrap();
    assert!(failed, "the read must fail after the cancel");
    let reread = tokio::task::spawn_blocking(move || {
        // Past anything the page cache holds from before the cancel.
        let mut f = std::fs::File::open(reopen)?;
        f.seek(SeekFrom::Start(50 * 1024 * 1024))?;
        let mut buf = vec![0u8; 4096];
        f.read_exact(&mut buf)
    })
    .await
    .unwrap();
    assert!(reread.is_err(), "a reopened cancelled file must not read");

    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}

/// A reader held by a paused transfer can be killed, and its held request then ends with
/// the transfer; unmounting lets held reads go.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_readers_can_be_killed() {
    let src = scratch("pause-kill-src");
    std::fs::write(src.join("k.bin"), pattern(16 * 1024 * 1024, 13)).unwrap();
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("pause-kill-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let (id, roots) = owner
        .offer(OfferSlot::Clipboard, &[src.join("k.bin")])
        .unwrap();
    let path = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots)[0].clone();
    // Pause the file before anything reads it, as the peer's request would.
    let file = transfers::TransferFile {
        direction: transfers::TransferDirection::Incoming,
        offer_id: id,
        path: vec!["k.bin".to_string()],
    };
    transfers::set_paused_file(reader_conn.stable_id(), &file, true);

    let mut dd = std::process::Command::new("dd")
        .arg(format!("if={}", path.display()))
        .args(["of=/dev/null", "bs=1M", "count=1"])
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(dd.try_wait().unwrap().is_none(), "the read must be held");
    dd.kill().unwrap();
    let start = std::time::Instant::now();
    let status = tokio::task::spawn_blocking(move || dd.wait().unwrap())
        .await
        .unwrap();
    assert!(!status.success());
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "a killed reader must not wait for the resume: {:?}",
        start.elapsed()
    );
    // Nor must its held request keep the transfer alive (listed, paused, forever).
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while transfers::list(reader_conn.stable_id())
        .iter()
        .any(|t| t.name == "k.bin")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the killed reader's transfer is still listed"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // The pause ended with that transfer: nothing was left listed to resume it from.
    assert!(transfers::list(reader_conn.stable_id()).is_empty());
    // Another held reader is let go by the unmount.
    transfers::set_paused_file(reader_conn.stable_id(), &file, true);
    let mut dd = std::process::Command::new("dd")
        .arg(format!("if={}", path.display()))
        .args(["of=/dev/null", "bs=1M", "count=1"])
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(dd.try_wait().unwrap().is_none(), "the read must be held");
    let unmounted = tokio::task::spawn_blocking(move || drop(mount));
    let status = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || dd.wait().unwrap()),
    )
    .await
    .expect("unmounting must let the held read go")
    .unwrap();
    assert!(!status.success());
    tokio::time::timeout(Duration::from_secs(5), unmounted)
        .await
        .expect("the unmount must finish")
        .unwrap();
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}

/// A read held by a pause is let go when the connection closes: nobody is left to resume it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_reads_end_when_the_connection_closes() {
    let src = scratch("pause-close-src");
    std::fs::write(src.join("c.bin"), pattern(16 * 1024 * 1024, 17)).unwrap();
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("pause-close-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let (id, roots) = owner
        .offer(OfferSlot::Clipboard, &[src.join("c.bin")])
        .unwrap();
    let path = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots)[0].clone();
    let file = transfers::TransferFile {
        direction: transfers::TransferDirection::Incoming,
        offer_id: id,
        path: vec!["c.bin".to_string()],
    };
    transfers::set_paused_file(reader_conn.stable_id(), &file, true);
    let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
    tokio::task::spawn_blocking(move || {
        let res = std::fs::File::open(path).and_then(|mut f| {
            let mut buf = vec![0u8; 4096];
            f.read_exact(&mut buf)
        });
        let _ = done_tx.send(res.is_err());
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(done_rx.try_recv().is_err(), "the read must be held");
    reader_conn.close(0u32.into(), b"gone");
    let failed = tokio::time::timeout(Duration::from_secs(5), done_rx)
        .await
        .expect("closing the connection must let the held read go")
        .unwrap();
    assert!(failed);
    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}

/// Reads held by paused transfers must not hold up other transfers on the same mount (the
/// kernel limits a mount's outstanding background reads, such as readahead).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_files_do_not_stall_other_reads() {
    const PAUSED: usize = 16;
    let src = scratch("pause-many-src");
    for i in 0..=PAUSED {
        std::fs::write(
            src.join(format!("m{i}.bin")),
            pattern(4 * 1024 * 1024, i as u8),
        )
        .unwrap();
    }
    let (owner_conn, reader_conn) = quic_pair().await;
    let _live = live(&owner_conn, &reader_conn);
    let owner = Arc::new(XferOwner::new());
    serve(owner_conn.clone(), Arc::clone(&owner));
    let mnt_dir = scratch("pause-many-mnt");
    let mount = match XferMount::mount(&mnt_dir.join("xfer"), tokio::runtime::Handle::current()) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping: FUSE unavailable: {err:#}");
            return;
        }
    };
    let files: Vec<_> = (0..=PAUSED)
        .map(|i| src.join(format!("m{i}.bin")))
        .collect();
    let (id, roots) = owner.offer(OfferSlot::Clipboard, &files).unwrap();
    let paths = mount.add_offer(reader_conn.clone(), OfferSlot::Clipboard, id, roots);
    let mut readers = Vec::new();
    for (i, path) in paths.iter().take(PAUSED).enumerate() {
        let file = transfers::TransferFile {
            direction: transfers::TransferDirection::Incoming,
            offer_id: id,
            path: vec![format!("m{i}.bin")],
        };
        transfers::set_paused_file(reader_conn.stable_id(), &file, true);
        // Buffered sequential reads: the kernel adds readahead, sent in the background.
        readers.push(
            std::process::Command::new("dd")
                .arg(format!("if={}", path.display()))
                .args(["of=/dev/null", "bs=64K"])
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    let free = paths[PAUSED].clone();
    let expect = pattern(4 * 1024 * 1024, PAUSED as u8);
    let got = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || std::fs::read(free).unwrap()),
    )
    .await
    .expect("an unpaused file must read while others are paused")
    .unwrap();
    assert!(got == expect);
    for mut r in readers {
        let _ = r.kill();
        let _ = r.wait();
    }
    drop(mount);
    let _ = std::fs::remove_dir_all(&mnt_dir);
    let _ = std::fs::remove_dir_all(&src);
}
