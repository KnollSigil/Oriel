//! Oriel client entrypoint.
//!
//! Connects to a Oriel server over QUIC (directly or SSH-bootstrapped),
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
#[cfg(feature = "gtk4")]
mod symbolic_icon;
mod window;
mod xfer;

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use oriel_proto::ControlMessage;
use tokio::sync::mpsc;
use tracing::error;
use tracing_subscriber::EnvFilter;

use crate::hosts::{
    load_client_settings, parse_uri, resolve_uri, ssh_bootstrap, take_env_secret, RenderScaleCap,
    TOKEN_ENV, URI_ENV,
};
use crate::window::ClientUiEvent;

#[derive(Debug, Parser)]
#[command(name = "oriel-client", about = "Oriel client")]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<ClientSubcommand>,

    /// Read the connection URI printed by `oriel-server attach`
    /// (oriel://host:port?fp=...&token=...) from stdin. Only `-` is accepted: a URI
    /// on the command line would expose its token to other local users. The URI can also be
    /// given in $ORIEL_URI.
    #[arg(long, value_name = "-")]
    uri: Option<String>,

    /// Server UDP address (if not using a URI); the token goes in $ORIEL_TOKEN.
    #[arg(long)]
    server: Option<SocketAddr>,

    /// Expected server TLS certificate SHA-256 hex fingerprint (if not using --uri).
    #[arg(long)]
    fingerprint: Option<String>,

    /// Host PipeWire microphone source name or serial to use (default: system default).
    #[arg(long)]
    mic: Option<String>,

    /// Cap remote window render scale (uncapped, 2.0, 1.5, or 1.0).
    #[arg(long)]
    max_scale: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ClientSubcommand {
    /// Connect to a remote host or local user via SSH (`ssh <target> oriel-server attach`).
    Connect {
        /// SSH target (`user@host`, `otheruser@localhost`, etc.).
        target: String,

        /// Open a remote terminal (PTY over QUIC, no Wayland/PipeWire needed) instead of GUI session.
        #[arg(long)]
        terminal: bool,

        /// Remote path to `oriel-server` binary.
        #[arg(long)]
        server_bin: Option<String>,

        /// UDP port or inclusive port range (`7443`, `7443-7462`) the remote server must
        /// listen on. Without it, the server uses any free port.
        #[arg(long, value_name = "PORT[-END]")]
        port: Option<oriel_proto::PortRange>,

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
        /// Local Unix socket path exposed by a connected Oriel client session.
        #[arg(long)]
        socket: Option<std::path::PathBuf>,

        /// Read the connection URI from stdin (direct QUIC mode); only `-` is accepted. The
        /// URI can also be given in $ORIEL_URI.
        #[arg(long, value_name = "-", conflicts_with_all = ["socket", "ssh"])]
        uri: Option<String>,

        /// Get a connection URI by running `oriel-server attach` over SSH on this
        /// target (direct QUIC mode).
        #[arg(long, value_name = "TARGET", conflicts_with = "socket")]
        ssh: Option<String>,

        /// Remote path to `oriel-server` binary (with --ssh).
        #[arg(long, requires = "ssh")]
        server_bin: Option<String>,

        /// UDP port or inclusive port range the remote server must listen on (with --ssh).
        #[arg(long, value_name = "PORT[-END]", requires = "ssh")]
        port: Option<oriel_proto::PortRange>,

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
    // Taken before any threads start, and removed so nothing this client starts inherits them.
    let env_uri = take_env_secret(URI_ENV);
    let env_token = take_env_secret(TOKEN_ENV);

    if !matches!(cli.subcommand, Some(ClientSubcommand::PtyAttach { .. })) {
        xfer::cleanup_stale_in_background();
    }
    let result = match cli.subcommand {
        Some(ClientSubcommand::PtyAttach {
            socket,
            uri,
            ssh,
            server_bin,
            port,
            pty_id,
            command,
        }) => {
            let uri = match ssh {
                Some(target) => match ssh_bootstrap(&target, server_bin.as_deref(), port) {
                    Ok(uri) => Some(uri),
                    Err(err) => {
                        // Usually running in a terminal window that closes when this exits.
                        pty::report_error_and_wait(&err);
                        std::process::exit(1);
                    }
                },
                None if socket.is_none() => resolve_uri(uri.as_deref(), env_uri)?,
                None => None,
            };
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
                anyhow::bail!(
                    "pty-attach requires --socket <path>, --ssh <target>, --uri -, or {URI_ENV}"
                );
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
            if terminal {
                let cmd_str = if command.is_empty() {
                    None
                } else {
                    Some(command.join(" "))
                };
                // The terminal bootstraps over SSH itself, so the session token never has to
                // pass through its arguments or environment.
                pty::spawn_local_terminal_pty_ssh(
                    &target,
                    server_bin.as_deref(),
                    port,
                    cmd_str.as_deref(),
                )?;
                return Ok(());
            }
            let uri = ssh_bootstrap(&target, server_bin.as_deref(), port)?;
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
            if let Some(uri) = resolve_uri(cli.uri.as_deref(), env_uri)? {
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
            } else if let (Some(server), Some(fp)) = (cli.server, cli.fingerprint) {
                let token = env_token
                    .with_context(|| format!("--server/--fingerprint need {TOKEN_ENV} set"))?;
                run_gui_session(server, fp, token, true, effective_scale_cap, None, None)
            } else {
                launcher::run_launcher(cli_max_scale)
            }
        }
    };
    // Files offered by servers vanish from the local clipboard with this process.
    xfer::unmount();
    result
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

    let _ = ctrl_tx.send(ControlMessage::ServerDownscale {
        enabled: load_client_settings().downscale_hidpi,
    });
    if let Some(exec) = initial_exec {
        let _ = ctrl_tx.send(ControlMessage::LaunchApp {
            launch_id: 0,
            name: None,
            exec,
        });
    }

    std::thread::Builder::new()
        .name("oriel-net".into())
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
