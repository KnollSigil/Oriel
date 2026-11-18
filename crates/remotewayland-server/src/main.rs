//! RemoteWayland server entrypoint.
//!
//! Launches an isolated headless Wayland session + optional PipeWire audio bridge
//! and streams rootless windows, audio, and remote PTYs over QUIC.

mod apps;
mod audio;
mod compositor;
mod media;
mod pty;
mod quic;

use std::net::SocketAddr;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "remotewayland-server", about = "RemoteWayland headless app & PTY server")]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<ServerSubcommand>,

    /// Address and UDP port to bind the QUIC server on.
    #[arg(long, default_value = "0.0.0.0:7443")]
    listen: SocketAddr,

    /// Ephemeral session token (auto-generated if omitted).
    #[arg(long)]
    token: Option<String>,

    /// Run quietly as a background daemon (used internally by `attach`).
    #[arg(long, hide = true)]
    daemon: bool,

    /// Target GUI application and arguments to launch inside the isolated session.
    #[arg(last = true)]
    command: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum ServerSubcommand {
    /// Attach to an existing persistent user session (or start one) and print a fresh connect URI.
    Attach {
        /// Address and UDP port to bind if starting a new server daemon.
        #[arg(long, default_value = "0.0.0.0:7443")]
        listen: SocketAddr,

        /// Hostname or IP to embed in the printed URI (defaults to listen IP).
        #[arg(long)]
        host: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ControlSockInfo {
    port: u16,
    fp: String,
    token: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(ServerSubcommand::Attach { listen, host }) = cli.subcommand {
        return run_attach(listen, host).await;
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let token = cli.token.unwrap_or_else(generate_ephemeral_token);
    let creds = Arc::new(quic::generate_ephemeral_credentials(token)?);

    let (comp_handle, event_rx) = compositor::start_compositor(cli.command)?;
    info!(socket = ?comp_handle.socket_name, "headless Wayland compositor ready");

    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel::<SocketAddr>();
    let sock_creds = Arc::clone(&creds);
    tokio::spawn(async move {
        if let Ok(actual_addr) = bound_rx.await {
            if let Err(err) = run_control_socket(actual_addr.port(), sock_creds).await {
                warn!(error = %err, "control Unix socket failed");
            }
        }
    });

    let server_creds = quic::EphemeralCredentials {
        server_config: creds.server_config.clone(),
        cert_sha256_hex: creds.cert_sha256_hex.clone(),
        valid_tokens: Arc::clone(&creds.valid_tokens),
    };

    quic::run_server(
        cli.listen,
        server_creds,
        comp_handle,
        event_rx,
        Some(bound_tx),
        cli.daemon,
    )
    .await
}

fn control_socket_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        PathBuf::from(dir).join("remotewayland.sock")
    } else {
        let uid = unsafe { libc::geteuid() };
        PathBuf::from(format!("/tmp/remotewayland-{uid}.sock"))
    }
}

fn daemon_log_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        PathBuf::from(dir).join("remotewayland-server.log")
    } else {
        let uid = unsafe { libc::geteuid() };
        PathBuf::from(format!("/tmp/remotewayland-server-{uid}.log"))
    }
}

async fn run_control_socket(port: u16, creds: Arc<quic::EphemeralCredentials>) -> Result<()> {
    let sock_path = control_socket_path();
    if sock_path.exists() {
        if UnixStream::connect(&sock_path).await.is_ok() {
            return Ok(());
        }
        let _ = std::fs::remove_file(&sock_path);
    }

    let listener = UnixListener::bind(&sock_path)
        .with_context(|| format!("failed to bind control socket {}", sock_path.display()))?;
    info!(path = %sock_path.display(), "control Unix socket listening");

    loop {
        let (mut stream, _) = listener.accept().await?;
        let token = generate_ephemeral_token();
        creds.add_token(token.clone());
        let info = ControlSockInfo {
            port,
            fp: creds.cert_sha256_hex.clone(),
            token,
        };
        if let Ok(mut json) = serde_json::to_string(&info) {
            json.push('\n');
            let _ = stream.write_all(json.as_bytes()).await;
        }
    }
}

async fn run_attach(listen: SocketAddr, host: Option<String>) -> Result<()> {
    let sock_path = control_socket_path();
    if let Ok(info) = query_control_socket(&sock_path).await {
        print_attach_uri(&host, listen, &info);
        return Ok(());
    }

    let _ = std::fs::remove_file(&sock_path);
    let exe = std::env::current_exe().context("failed to locate remotewayland-server binary")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(daemon_log_path())
        .ok();

    let mut cmd = Command::new(exe);
    cmd.arg("--daemon")
        .arg("--listen")
        .arg(listen.to_string())
        .stdin(Stdio::null());
    if let Some(file) = log_file {
        let err_file = file.try_clone().unwrap_or(file);
        cmd.stdout(Stdio::from(err_file));
        cmd.stderr(Stdio::null());
    } else {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    cmd.spawn().context("failed to spawn remotewayland-server daemon")?;

    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if let Ok(info) = query_control_socket(&sock_path).await {
            print_attach_uri(&host, listen, &info);
            return Ok(());
        }
    }

    anyhow::bail!("timed out waiting for remotewayland-server daemon to start")
}

async fn query_control_socket(path: &Path) -> Result<ControlSockInfo> {
    let stream = UnixStream::connect(path).await?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let info: ControlSockInfo =
        serde_json::from_str(line.trim()).context("invalid control socket response")?;
    Ok(info)
}

fn print_attach_uri(host: &Option<String>, listen: SocketAddr, info: &ControlSockInfo) {
    let host_str = host
        .clone()
        .unwrap_or_else(|| listen.ip().to_string());
    println!(
        "remotewayland://{}:{}?fp={}&token={}",
        host_str, info.port, info.fp, info.token
    );
}

fn generate_ephemeral_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("rw-{nanos:x}-{pid:x}")
}
