//! RemoteWayland server entrypoint.
//!
//! Launches an isolated headless Wayland session + PipeWire audio bridge
//! and streams rootless windows + audio over QUIC.

mod audio;
mod compositor;
mod media;
mod quic;

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "remotewayland-server", about = "RemoteWayland headless app server")]
struct Cli {
    /// Address and UDP port to bind the QUIC server on.
    #[arg(long, default_value = "0.0.0.0:7443")]
    listen: SocketAddr,

    /// Ephemeral session token (auto-generated if omitted).
    #[arg(long)]
    token: Option<String>,

    /// Target GUI application and arguments to launch inside the isolated session.
    #[arg(last = true)]
    command: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();
    media::init_gstreamer()?;
    let token = cli.token.unwrap_or_else(generate_ephemeral_token);
    let creds = quic::generate_ephemeral_credentials(token)?;

    let (comp_handle, event_rx) = compositor::start_compositor(cli.command)?;
    tracing::info!(socket = ?comp_handle.socket_name, "headless Wayland compositor ready");

    quic::run_server(cli.listen, creds, comp_handle, event_rx).await
}

fn generate_ephemeral_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("rw-{nanos:x}")
}
