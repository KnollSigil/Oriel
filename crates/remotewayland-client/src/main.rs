//! RemoteWayland client entrypoint.
//!
//! Connects to a RemoteWayland server over QUIC, renders rootless remote windows
//! and native `xdg_popup`s via `smithay-client-toolkit`, decodes video via GStreamer,
//! and bridges local PipeWire audio.

mod audio;
mod media;
mod quic;
mod window;

use std::net::SocketAddr;

use anyhow::{bail, Context, Result};
use clap::Parser;
use tokio::sync::mpsc;
use tracing::error;
use tracing_subscriber::EnvFilter;

use crate::window::ClientUiEvent;

#[derive(Debug, Parser)]
#[command(name = "remotewayland-client", about = "RemoteWayland client")]
struct Cli {
    /// Connection URI printed by remotewayland-server (remotewayland://host:port?fp=...&token=...).
    #[arg(long)]
    uri: Option<String>,

    /// Server UDP address (if not using --uri).
    #[arg(long)]
    server: Option<SocketAddr>,

    /// Expected server TLS certificate SHA-256 hex fingerprint (if not using --uri).
    #[arg(long)]
    fingerprint: Option<String>,

    /// Session authentication token (if not using --uri).
    #[arg(long)]
    token: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();
    let (server, fingerprint, token) = if let Some(uri) = cli.uri {
        parse_uri(&uri)?
    } else {
        (
            cli.server.context("--server or --uri is required")?,
            cli.fingerprint
                .context("--fingerprint or --uri is required")?,
            cli.token.context("--token or --uri is required")?,
        )
    };

    let (ui_tx, ui_rx) = window::create_ui_channel();
    let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();

    std::thread::Builder::new()
        .name("rw-client-net".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(async move {
                if let Err(err) =
                    quic::run_client(server, fingerprint, token, ui_tx.clone(), ctrl_rx).await
                {
                    error!(error = %err, "QUIC client exited with error");
                    let _ = ui_tx.send(ClientUiEvent::Disconnected);
                }
            });
        })
        .context("failed to spawn client network thread")?;

    window::run_event_loop(ui_rx, ctrl_tx)
}

fn parse_uri(uri: &str) -> Result<(SocketAddr, String, String)> {
    let rest = uri
        .strip_prefix("remotewayland://")
        .context("URI must start with remotewayland://")?;
    let (addr_str, query) = rest
        .split_once('?')
        .context("URI must include ?fp=...&token=...")?;
    let addr: SocketAddr = addr_str.parse().context("invalid server socket address")?;

    let mut fp = None;
    let mut token = None;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            match k {
                "fp" => fp = Some(v.to_string()),
                "token" => token = Some(v.to_string()),
                _ => {}
            }
        }
    }

    match (fp, token) {
        (Some(fp), Some(token)) => Ok((addr, fp, token)),
        _ => bail!("URI missing fp or token query parameter"),
    }
}
