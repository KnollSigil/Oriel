//! RemoteWayland client entrypoint.
//!
//! Connects to a RemoteWayland server over QUIC (directly or SSH-bootstrapped),
//! renders rootless remote windows & native `xdg_popup`s via `smithay-client-toolkit`,
//! bridges local PipeWire audio, and supports remote PTYs + GTK4/libadwaita launcher.

#[cfg(feature = "gtk4")]
mod adw_launcher;
mod audio;
mod hosts;
mod icon;
mod launcher;
mod media;
mod pty;
mod quic;
#[cfg(any(feature = "gtk4", test))]
mod search;
#[cfg(any(feature = "gtk4", test))]
mod session;
mod window;

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use remotewayland_proto::ControlMessage;
use tokio::sync::mpsc;
use tracing::error;
use tracing_subscriber::EnvFilter;

use crate::hosts::{load_client_settings, parse_uri, ssh_bootstrap, RenderScaleCap};
use crate::window::ClientUiEvent;

#[derive(Debug, Parser)]
#[command(name = "remotewayland-client", about = "RemoteWayland client")]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<ClientSubcommand>,

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

    /// Host PipeWire microphone source name or serial to use (default: system default).
    #[arg(long)]
    mic: Option<String>,

    /// Cap remote window render scale (uncapped, 2.0, 1.5, or 1.0).
    #[arg(long)]
    max_scale: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ClientSubcommand {
    /// Connect to a remote host or local user via SSH (`ssh <target> remotewayland-server attach`).
    Connect {
        /// SSH target (`user@host`, `otheruser@localhost`, etc.).
        target: String,

        /// Open a remote terminal (PTY over QUIC, no Wayland/PipeWire needed) instead of GUI session.
        #[arg(long)]
        terminal: bool,

        /// Remote path to `remotewayland-server` binary.
        #[arg(long)]
        server_bin: Option<String>,

        /// Remote QUIC UDP port to bind if starting a new server daemon.
        #[arg(long, default_value_t = 7443)]
        port: u16,

        /// Disable PipeWire audio bridging.
        #[arg(long)]
        no_audio: bool,

        /// Host PipeWire microphone source name or serial to use (default: system default).
        #[arg(long)]
        mic: Option<String>,

        /// Cap remote window render scale (uncapped, 2.0, 1.5, or 1.0).
        #[arg(long)]
        max_scale: Option<String>,

        /// Optional remote GUI application command to launch immediately upon connecting.
        #[arg(last = true)]
        command: Vec<String>,
    },

    /// Attach current terminal's stdin/stdout to a remote PTY over a local Unix tunnel or QUIC.
    PtyAttach {
        /// Local Unix socket path exposed by a connected RemoteWayland client session.
        #[arg(long)]
        socket: Option<std::path::PathBuf>,

        /// Connection URI printed by remotewayland-server (direct QUIC mode if --socket is omitted).
        #[arg(long)]
        uri: Option<String>,

        /// Reattach to an existing persistent server PTY by ID instead of starting a new shell.
        #[arg(long)]
        pty_id: Option<u64>,

        /// Optional command to run instead of interactive `$SHELL`.
        #[arg(long)]
        command: Option<String>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.subcommand {
        Some(ClientSubcommand::PtyAttach {
            socket,
            uri,
            pty_id,
            command,
        }) => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to build tokio runtime")?;
            let code = if let Some(sock_path) = socket {
                rt.block_on(pty::run_pty_attach_socket(&sock_path, pty_id, command))?
            } else if let Some(uri) = uri {
                let (server, fingerprint, token) = parse_uri(&uri)?;
                rt.block_on(pty::run_pty_attach(
                    server,
                    fingerprint,
                    token,
                    pty_id,
                    command,
                ))?
            } else {
                anyhow::bail!("pty-attach requires either --socket <path> or --uri <uri>");
            };
            std::process::exit(code);
        }
        Some(ClientSubcommand::Connect {
            target,
            terminal,
            server_bin,
            port,
            no_audio,
            mic,
            max_scale,
            command,
        }) => {
            audio::set_mic_target(mic);
            let scale_cap = match max_scale {
                Some(s) => RenderScaleCap::parse_cli(&s)?,
                None => load_client_settings().max_scale,
            };
            let uri = ssh_bootstrap(&target, server_bin.as_deref(), port)?;
            if terminal {
                let cmd_str = if command.is_empty() {
                    None
                } else {
                    Some(command.join(" "))
                };
                pty::spawn_local_terminal_pty(&uri, cmd_str.as_deref())?;
                return Ok(());
            }
            let (server, fingerprint, token) = parse_uri(&uri)?;
            let initial_exec = if command.is_empty() {
                None
            } else {
                Some(command.join(" "))
            };
            let pty_sock = pty::pty_tunnel_socket_path(&target).ok();
            run_gui_session(
                server,
                fingerprint,
                token,
                !no_audio,
                scale_cap,
                initial_exec,
                pty_sock,
            )
        }
        None => {
            audio::set_mic_target(cli.mic);
            let cli_max_scale = cli
                .max_scale
                .as_deref()
                .map(RenderScaleCap::parse_cli)
                .transpose()?;
            let effective_scale_cap =
                cli_max_scale.unwrap_or_else(|| load_client_settings().max_scale);
            if let Some(uri) = cli.uri {
                let (server, fingerprint, token) = parse_uri(&uri)?;
                run_gui_session(
                    server,
                    fingerprint,
                    token,
                    true,
                    effective_scale_cap,
                    None,
                    None,
                )
            } else if let (Some(server), Some(fp), Some(token)) =
                (cli.server, cli.fingerprint, cli.token)
            {
                run_gui_session(
                    server,
                    fp,
                    token,
                    true,
                    effective_scale_cap,
                    None,
                    None,
                )
            } else {
                launcher::run_launcher(cli_max_scale)
            }
        }
    }
}

fn run_gui_session(
    server: SocketAddr,
    fingerprint: String,
    token: String,
    enable_audio: bool,
    max_scale: RenderScaleCap,
    initial_exec: Option<String>,
    pty_socket_path: Option<std::path::PathBuf>,
) -> Result<()> {
    let (ui_tx, ui_rx) = window::create_ui_channel();
    let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();

    if let Some(exec) = initial_exec {
        let _ = ctrl_tx.send(ControlMessage::LaunchApp {
            launch_id: 0,
            name: None,
            exec,
        });
    }

    std::thread::Builder::new()
        .name("rw-client-net".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(async move {
                if let Err(err) = quic::run_client(
                    server,
                    fingerprint,
                    token,
                    enable_audio,
                    1.0,
                    ui_tx.clone(),
                    ctrl_rx,
                    None,
                    pty_socket_path,
                )
                .await
                {
                    error!(error = %err, "QUIC client exited with error");
                    let _ = ui_tx.send(ClientUiEvent::Disconnected);
                }
            });
        })
        .context("failed to spawn client network thread")?;

    window::run_event_loop(ui_rx, ctrl_tx, max_scale)
}
