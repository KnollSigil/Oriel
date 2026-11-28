//! `oriel-server attach` and `oriel-server stop` against a fake daemon: a helper process
//! (this test binary, re-run as the `fake_daemon` test) that answers on the control socket
//! the way a running server, possibly of another version, does.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const SERVER: &str = env!("CARGO_BIN_EXE_oriel-server");
/// Tells the `fake_daemon` test which socket to serve and what to report.
const SOCKET_ENV: &str = "ORIEL_TEST_FAKE_DAEMON_SOCKET";
const PROTOCOL_ENV: &str = "ORIEL_TEST_FAKE_DAEMON_PROTOCOL";
const PORT: u16 = 47_011;

/// A private `XDG_RUNTIME_DIR`, removed on drop.
struct RuntimeDir(PathBuf);

impl RuntimeDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oriel-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(dir)
    }

    fn socket(&self) -> PathBuf {
        self.0.join("oriel.sock")
    }
}

impl Drop for RuntimeDir {
    fn drop(&mut self) {
        // A real daemon started here (by `attach --replace`, or by `attach` when a fake
        // daemon died early) must not outlive the test.
        let _ = oriel_server(self, &["stop"]);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The fake daemon process, killed on drop if a test left it running.
struct FakeDaemon(Child);

impl FakeDaemon {
    /// Start one serving `dir`'s control socket, reporting `protocol` (`None`: like
    /// daemons from before the field existed).
    fn start(dir: &RuntimeDir, protocol: Option<u32>) -> Self {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "fake_daemon", "--ignored", "--nocapture"])
            .env(SOCKET_ENV, dir.socket())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(v) = protocol {
            cmd.env(PROTOCOL_ENV, v.to_string());
        }
        let daemon = Self(cmd.spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::os::unix::net::UnixStream::connect(dir.socket()).is_err() {
            assert!(Instant::now() < deadline, "the fake daemon did not start");
            std::thread::sleep(Duration::from_millis(20));
        }
        daemon
    }

    fn is_running(&mut self) -> bool {
        self.0.try_wait().unwrap().is_none()
    }
}

impl Drop for FakeDaemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn oriel_server(dir: &RuntimeDir, args: &[&str]) -> Output {
    let out = run_oriel_server(dir, args);
    // `attach` gives up on a daemon slow to start but leaves it starting; let it get to
    // where `stop` (when the test is done) can find it.
    if text(&out.stderr).contains("timed out waiting for oriel-server daemon to start") {
        let deadline = Instant::now() + Duration::from_secs(20);
        while std::os::unix::net::UnixStream::connect(dir.socket()).is_err()
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    out
}

fn run_oriel_server(dir: &RuntimeDir, args: &[&str]) -> Output {
    Command::new(SERVER)
        .args(args)
        .env("XDG_RUNTIME_DIR", &dir.0)
        // Starting a daemon need not wait for the login shell.
        .env("ORIEL_SKIP_SHELL_ENV", "1")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Not a test: the fake daemon process the tests start. It answers each control-socket
/// request with a fixed URI's parts and runs until it is killed.
#[test]
#[ignore = "helper process for the other tests in this file"]
fn fake_daemon() {
    let Some(sock) = std::env::var_os(SOCKET_ENV) else {
        return;
    };
    let protocol = std::env::var(PROTOCOL_ENV).ok();
    let listener = UnixListener::bind(Path::new(&sock)).unwrap();
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut line = String::new();
        let _ = BufReader::new(&stream).read_line(&mut line);
        let protocol = protocol
            .as_ref()
            .map(|v| format!(r#","protocol":{v}"#))
            .unwrap_or_default();
        let _ = writeln!(
            stream,
            r#"{{"port":{PORT},"fp":"fakefp","token":"faketoken"{protocol}}}"#
        );
    }
}

#[test]
fn stop_without_a_daemon_says_so() {
    let dir = RuntimeDir::new("stop-none");
    let out = oriel_server(&dir, &["stop"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stderr).contains("no Oriel server is running"));

    // A socket left behind by a daemon that died is not a running daemon either.
    drop(UnixListener::bind(dir.socket()).unwrap());
    let out = oriel_server(&dir, &["stop"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stderr).contains("no Oriel server is running"));
}

#[test]
fn stop_terminates_the_daemon_and_waits_for_it() {
    let dir = RuntimeDir::new("stop");
    let mut daemon = FakeDaemon::start(&dir, Some(oriel_proto::PROTOCOL_VERSION));
    let pid = daemon.0.id();
    let out = oriel_server(&dir, &["stop"]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        text(&out.stderr).contains(&format!("stopped the Oriel server (pid {pid})")),
        "{out:?}"
    );
    // `stop` returns once the daemon is gone; it is only left for us to reap.
    let status = daemon.0.wait().unwrap();
    assert!(!status.success());
    assert!(!daemon.is_running());
}

#[test]
fn attach_reuses_a_daemon_of_this_version() {
    let dir = RuntimeDir::new("attach-same");
    let mut daemon = FakeDaemon::start(&dir, Some(oriel_proto::PROTOCOL_VERSION));
    let out = oriel_server(&dir, &["attach", "--host", "example.test"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        text(&out.stdout).trim(),
        format!("oriel://example.test:{PORT}?fp=fakefp&token=faketoken")
    );
    assert!(!text(&out.stderr).contains("protocol"), "{out:?}");
    // --replace leaves a daemon of the same version alone.
    let out = oriel_server(&dir, &["attach", "--replace", "--host", "example.test"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("token=faketoken"));
    assert!(daemon.is_running());
}

#[test]
fn attach_refuses_a_daemon_of_another_version() {
    let dir = RuntimeDir::new("attach-other");
    let mut daemon = FakeDaemon::start(&dir, Some(oriel_proto::PROTOCOL_VERSION + 1));
    let out = oriel_server(&dir, &["attach"]);
    assert!(!out.status.success(), "{out:?}");
    // No URI to a server the client could not talk to.
    assert_eq!(text(&out.stdout), "");
    let err = text(&out.stderr);
    assert!(err.contains("speaks protocol version"), "{err}");
    assert!(err.contains("attach --replace"), "{err}");
    assert!(
        daemon.is_running(),
        "attach must not stop it without --replace"
    );
}

#[test]
fn attach_warns_about_a_daemon_of_unknown_version() {
    let dir = RuntimeDir::new("attach-unknown");
    let mut daemon = FakeDaemon::start(&dir, None);
    let out = oriel_server(&dir, &["attach"]);
    // It may speak this version, so it is still used.
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("token=faketoken"));
    let err = text(&out.stderr);
    assert!(
        err.contains("does not report its protocol version"),
        "{err}"
    );
    assert!(err.contains("attach --replace"), "{err}");
    assert!(daemon.is_running());
}

#[test]
fn attach_replace_stops_a_daemon_of_another_version() {
    let dir = RuntimeDir::new("attach-replace");
    let mut daemon = FakeDaemon::start(&dir, Some(oriel_proto::PROTOCOL_VERSION + 1));
    let pid = daemon.0.id();
    let out = oriel_server(&dir, &["attach", "--replace", "--listen", "127.0.0.1"]);
    let err = text(&out.stderr);
    assert!(err.contains("stopping it to start this version"), "{err}");
    let deadline = Instant::now() + Duration::from_secs(5);
    while daemon.is_running() {
        assert!(
            Instant::now() < deadline,
            "the old daemon (pid {pid}) is still running"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    // Whether the new server starts depends on this machine (it runs a compositor and an
    // audio stack); if it does, `attach` hands out its URI, not the old daemon's.
    if out.status.success() {
        let uri = text(&out.stdout);
        assert!(uri.starts_with("oriel://127.0.0.1:"), "{uri}");
        assert!(!uri.contains("faketoken"), "{uri}");
    }
}
