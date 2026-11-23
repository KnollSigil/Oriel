//! RemoteWayland launcher UI (native Qt Quick/QML when `feature = "qt"` is enabled,
//! native GTK4/libadwaita when `feature = "gtk4"` is enabled without `qt`,
//! with an interactive terminal fallback otherwise).

use anyhow::Result;

use crate::hosts::RenderScaleCap;

#[cfg(feature = "qt")]
pub fn run_launcher(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
    crate::qt_launcher::run(cli_max_scale)
}

#[cfg(all(feature = "gtk4", not(feature = "qt")))]
pub fn run_launcher(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
    crate::adw_launcher::run(cli_max_scale)
}

#[cfg(all(not(feature = "gtk4"), not(feature = "qt")))]
pub fn run_launcher(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
    cli_ui::run(cli_max_scale)
}

#[cfg(all(not(feature = "gtk4"), not(feature = "qt")))]
mod cli_ui {
    use std::io::{self, BufRead, Write};

    use anyhow::Result;
    use remotewayland_proto::{ControlMessage, RemoteAppEntry};
    use tokio::sync::mpsc;

    use crate::hosts::{
        load_client_settings, load_hosts, parse_uri, save_hosts, ssh_bootstrap, HostProfile,
        RenderScaleCap,
    };
    use crate::pty::{pty_tunnel_socket_path, spawn_local_terminal_pty_socket};
    use crate::quic::{self, LauncherNetEvent};
    use crate::window::{self, ClientUiEvent};

    pub fn run(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
        let global_max_scale = load_client_settings().max_scale;
        println!("RemoteWayland Launcher (built without `gtk4` feature; using interactive CLI)");
        let mut hosts = load_hosts();

        let target = loop {
            println!("\nSaved hosts:");
            if hosts.is_empty() {
                println!("  (none)");
            } else {
                for (i, h) in hosts.iter().enumerate() {
                    println!("  [{}] {} ({})", i + 1, h.label, h.ssh_target);
                }
            }
            print!("\nEnter host number or user@host to connect (q to quit): ");
            io::stdout().flush()?;
            let mut line = String::new();
            if io::stdin().lock().read_line(&mut line)? == 0 {
                return Ok(());
            }
            let input = line.trim();
            if input.eq_ignore_ascii_case("q") {
                return Ok(());
            }
            if let Ok(idx) = input.parse::<usize>() {
                if idx >= 1 && idx <= hosts.len() {
                    break hosts[idx - 1].clone();
                }
            } else if !input.is_empty() {
                let profile = HostProfile {
                    label: input.to_string(),
                    ssh_target: input.to_string(),
                    server_bin: None,
                    port: 7443,
                    audio_enabled: true,
                    max_scale: None,
                };
                hosts.push(profile.clone());
                let _ = save_hosts(&hosts);
                break profile;
            }
        };

        println!("Connecting to {} via SSH...", target.ssh_target);
        let uri = ssh_bootstrap(
            &target.ssh_target,
            target.server_bin.as_deref(),
            target.port,
        )?;
        let (server, fp, token) = parse_uri(&uri)?;
        let pty_sock = pty_tunnel_socket_path(&target.ssh_target)?;

        let (ui_tx, ui_rx) = window::create_ui_channel();
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let (net_tx, mut net_rx) = mpsc::unbounded_channel::<LauncherNetEvent>();

        let effective_max_scale = cli_max_scale.or(target.max_scale).unwrap_or(global_max_scale);
        let ui_tx_net = ui_tx.clone();
        let pty_sock_net = pty_sock.clone();
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
                        fp,
                        token,
                        target.audio_enabled,
                        1.0,
                        ui_tx_net.clone(),
                        ctrl_rx,
                        Some(net_tx),
                        Some(pty_sock_net),
                    )
                    .await
                    {
                        tracing::error!(error = %err, "QUIC client exited with error");
                        let _ = ui_tx_net.send(ClientUiEvent::Disconnected);
                    }
                });
            })?;

        std::thread::Builder::new()
            .name("rw-wayland-ui".into())
            .spawn({
                let ctrl_tx = ctrl_tx.clone();
                move || {
                    if let Err(err) =
                        window::run_event_loop(ui_rx, ctrl_tx, effective_max_scale)
                    {
                        tracing::error!(error = %err, "Wayland UI loop exited");
                    }
                }
            })?;

        let _ = ctrl_tx.send(ControlMessage::ListApps);
        let mut apps = Vec::new();
        while let Some(ev) = net_rx.blocking_recv() {
            if let LauncherNetEvent::AppList(list) = ev {
                apps = list;
                break;
            }
        }

        let mut next_id = 1u64;
        loop {
            println!(
                "\nConnected to {} ({} apps discovered)",
                target.ssh_target,
                apps.len()
            );
            println!("  [t] New remote terminal (no Wayland/PipeWire needed)");
            println!("  [l] Search & launch remote GUI app");
            println!("  [c] Run custom GUI command");
            println!("  [q] Disconnect and quit");
            print!("> ");
            io::stdout().flush()?;

            let mut cmd = String::new();
            if io::stdin().lock().read_line(&mut cmd)? == 0 {
                break;
            }
            match cmd.trim() {
                "t" => {
                    if let Err(err) = spawn_local_terminal_pty_socket(&pty_sock, None, None) {
                        eprintln!("Failed to open remote terminal: {err}");
                    }
                }
                "l" => {
                    print!("Filter (blank for all): ");
                    io::stdout().flush()?;
                    let mut filter = String::new();
                    io::stdin().lock().read_line(&mut filter)?;
                    let needle = filter.trim().to_lowercase();
                    let matches: Vec<&RemoteAppEntry> = apps
                        .iter()
                        .filter(|a| needle.is_empty() || a.name.to_lowercase().contains(&needle))
                        .take(25)
                        .collect();
                    for (i, a) in matches.iter().enumerate() {
                        println!("  [{}] {} ({})", i + 1, a.name, a.exec);
                    }
                    print!("Launch number (or Enter to cancel): ");
                    io::stdout().flush()?;
                    let mut pick = String::new();
                    io::stdin().lock().read_line(&mut pick)?;
                    if let Ok(idx) = pick.trim().parse::<usize>() {
                        if idx >= 1 && idx <= matches.len() {
                            let id = next_id;
                            next_id += 1;
                            let _ = ctrl_tx.send(ControlMessage::LaunchApp {
                                launch_id: id,
                                name: Some(matches[idx - 1].name.clone()),
                                exec: matches[idx - 1].exec.clone(),
                            });
                        }
                    }
                }
                "c" => {
                    print!("Command: ");
                    io::stdout().flush()?;
                    let mut exec = String::new();
                    io::stdin().lock().read_line(&mut exec)?;
                    let exec = exec.trim();
                    if !exec.is_empty() {
                        let id = next_id;
                        next_id += 1;
                        let _ = ctrl_tx.send(ControlMessage::LaunchApp {
                            launch_id: id,
                            name: None,
                            exec: exec.to_string(),
                        });
                    }
                }
                "q" => break,
                _ => {}
            }
        }

        let _ = ui_tx.send(ClientUiEvent::Disconnected);
        Ok(())
    }
}
