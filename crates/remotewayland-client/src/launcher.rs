//! RemoteWayland launcher UI (native GTK4 when `feature = "gtk4"` is enabled,
//! with an interactive terminal fallback otherwise).

use anyhow::Result;

#[cfg(feature = "gtk4")]
pub fn run_launcher() -> Result<()> {
    gtk4_ui::run()
}

#[cfg(not(feature = "gtk4"))]
pub fn run_launcher() -> Result<()> {
    cli_ui::run()
}

#[cfg(not(feature = "gtk4"))]
mod cli_ui {
    use std::io::{self, BufRead, Write};

    use anyhow::Result;
    use remotewayland_proto::{ControlMessage, RemoteAppEntry};
    use tokio::sync::mpsc;

    use crate::hosts::{load_hosts, parse_uri, save_hosts, ssh_bootstrap, HostProfile};
    use crate::pty::spawn_local_terminal_pty;
    use crate::quic;
    use crate::window::{self, ClientUiEvent};

    pub fn run() -> Result<()> {
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

        let (ui_tx, ui_rx) = window::create_ui_channel();
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let (app_tx, mut app_rx) = mpsc::unbounded_channel::<Vec<RemoteAppEntry>>();

        let ui_tx_net = ui_tx.clone();
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
                        ui_tx_net.clone(),
                        ctrl_rx,
                        Some(app_tx),
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
                    if let Err(err) = window::run_event_loop(ui_rx, ctrl_tx) {
                        tracing::error!(error = %err, "Wayland UI loop exited");
                    }
                }
            })?;

        let _ = ctrl_tx.send(ControlMessage::ListApps);
        let apps = app_rx.blocking_recv().unwrap_or_default();

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
                    if let Err(err) = spawn_local_terminal_pty(&uri, None) {
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
                            let _ = ctrl_tx.send(ControlMessage::LaunchApp {
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
                        let _ = ctrl_tx.send(ControlMessage::LaunchApp {
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

#[cfg(feature = "gtk4")]
mod gtk4_ui {
    use std::cell::RefCell;
    use std::rc::Rc;

    use anyhow::Result;
    use gtk4::gdk_pixbuf::PixbufLoader;
    use gtk4::glib;
    use gtk4::prelude::*;
    use remotewayland_proto::{ControlMessage, RemoteAppEntry};
    use tokio::sync::mpsc;

    use crate::hosts::{load_hosts, parse_uri, save_hosts, ssh_bootstrap, HostProfile};
    use crate::pty::spawn_local_terminal_pty;
    use crate::quic;
    use crate::window::{self, ClientUiEvent};

    struct ActiveSession {
        uri: String,
        ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
        ui_tx: window::UiSender,
    }

    pub fn run() -> Result<()> {
        let app = gtk4::Application::builder()
            .application_id("org.remotewayland.Launcher")
            .build();

        app.connect_activate(build_ui);
        app.run_with_args(&Vec::<String>::new());
        Ok(())
    }

    fn build_ui(app: &gtk4::Application) {
        let window = gtk4::ApplicationWindow::builder()
            .application(app)
            .title("RemoteWayland")
            .default_width(860)
            .default_height(600)
            .build();

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        root.set_margin_top(12);
        root.set_margin_bottom(12);
        root.set_margin_start(12);
        root.set_margin_end(12);

        let hosts = Rc::new(RefCell::new(load_hosts()));
        let session: Rc<RefCell<Option<ActiveSession>>> = Rc::new(RefCell::new(None));
        let all_apps: Rc<RefCell<Vec<RemoteAppEntry>>> = Rc::new(RefCell::new(Vec::new()));

        // Top connection bar
        let conn_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        let host_entry = gtk4::Entry::builder()
            .placeholder_text("user@host (or user@localhost)")
            .hexpand(true)
            .build();
        if let Some(first) = hosts.borrow().first() {
            host_entry.set_text(&first.ssh_target);
        }
        let connect_btn = gtk4::Button::with_label("Connect");
        let term_btn = gtk4::Button::with_label("New Terminal");
        term_btn.set_sensitive(false);
        let status_label = gtk4::Label::new(Some("Disconnected"));

        conn_bar.append(&host_entry);
        conn_bar.append(&connect_btn);
        conn_bar.append(&term_btn);
        conn_bar.append(&status_label);
        root.append(&conn_bar);

        // Search + custom command bar
        let action_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        let search_entry = gtk4::SearchEntry::builder()
            .placeholder_text("Filter remote apps...")
            .hexpand(true)
            .build();
        let cmd_entry = gtk4::Entry::builder()
            .placeholder_text("Run custom GUI command...")
            .hexpand(true)
            .build();
        let run_btn = gtk4::Button::with_label("Run");
        run_btn.set_sensitive(false);
        action_bar.append(&search_entry);
        action_bar.append(&cmd_entry);
        action_bar.append(&run_btn);
        root.append(&action_bar);

        // App icon grid
        let flow = gtk4::FlowBox::builder()
            .valign(gtk4::Align::Start)
            .max_children_per_line(8)
            .min_children_per_line(3)
            .selection_mode(gtk4::SelectionMode::None)
            .row_spacing(10)
            .column_spacing(10)
            .build();
        let scroll = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .child(&flow)
            .build();
        root.append(&scroll);

        let populate_grid = {
            let flow = flow.clone();
            let all_apps = Rc::clone(&all_apps);
            let session = Rc::clone(&session);
            move |filter: &str| {
                while let Some(child) = flow.first_child() {
                    flow.remove(&child);
                }
                let needle = filter.trim().to_lowercase();
                for entry in all_apps.borrow().iter() {
                    if !needle.is_empty() && !entry.name.to_lowercase().contains(&needle) {
                        continue;
                    }
                    let btn = gtk4::Button::new();
                    let vbox = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
                    vbox.set_size_request(96, 84);
                    if !entry.icon_data.is_empty() {
                        if let Some(pb) = load_pixbuf(&entry.icon_data) {
                            let tex = gtk4::gdk::Texture::for_pixbuf(&pb);
                            let img = gtk4::Image::from_paintable(Some(&tex));
                            img.set_pixel_size(48);
                            vbox.append(&img);
                        }
                    }
                    let lbl = gtk4::Label::new(Some(&entry.name));
                    lbl.set_ellipsize(gtk4::pango::EllipsizeMode::End);
                    lbl.set_max_width_chars(14);
                    vbox.append(&lbl);
                    btn.set_child(Some(&vbox));

                    let exec = entry.exec.clone();
                    let session = Rc::clone(&session);
                    btn.connect_clicked(move |_| {
                        if let Some(s) = session.borrow().as_ref() {
                            let _ = s.ctrl_tx.send(ControlMessage::LaunchApp { exec: exec.clone() });
                        }
                    });
                    flow.insert(&btn, -1);
                }
            }
        };

        search_entry.connect_search_changed({
            let populate_grid = populate_grid.clone();
            move |e| populate_grid(&e.text())
        });

        run_btn.connect_clicked({
            let cmd_entry = cmd_entry.clone();
            let session = Rc::clone(&session);
            move |_| {
                let exec = cmd_entry.text().trim().to_string();
                if !exec.is_empty() {
                    if let Some(s) = session.borrow().as_ref() {
                        let _ = s.ctrl_tx.send(ControlMessage::LaunchApp { exec });
                    }
                }
            }
        });

        term_btn.connect_clicked({
            let session = Rc::clone(&session);
            move |_| {
                if let Some(s) = session.borrow().as_ref() {
                    let _ = spawn_local_terminal_pty(&s.uri, None);
                }
            }
        });

        connect_btn.connect_clicked({
            let host_entry = host_entry.clone();
            let status_label = status_label.clone();
            let term_btn = term_btn.clone();
            let run_btn = run_btn.clone();
            let hosts = Rc::clone(&hosts);
            let session = Rc::clone(&session);
            let all_apps = Rc::clone(&all_apps);
            let populate_grid = populate_grid.clone();
            let search_entry = search_entry.clone();

            move |_| {
                let target = host_entry.text().trim().to_string();
                if target.is_empty() {
                    return;
                }
                status_label.set_text("Connecting via SSH...");

                {
                    let mut list = hosts.borrow_mut();
                    if !list.iter().any(|h| h.ssh_target == target) {
                        list.push(HostProfile {
                            label: target.clone(),
                            ssh_target: target.clone(),
                            server_bin: None,
                            port: 7443,
                            audio_enabled: true,
                        });
                        let _ = save_hosts(&list);
                    }
                }

                let (tx, rx) = std::sync::mpsc::channel::<Result<(Vec<RemoteAppEntry>, ActiveSession), String>>();

                let target_bg = target.clone();
                std::thread::spawn(move || {
                    let res = (|| -> anyhow::Result<_> {
                        let uri = ssh_bootstrap(&target_bg, None, 7443)?;
                        let (server, fp, token) = parse_uri(&uri)?;
                        let (ui_tx, ui_rx) = window::create_ui_channel();
                        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
                        let (app_tx, mut app_rx) = mpsc::unbounded_channel();

                        let ui_tx_net = ui_tx.clone();
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Builder::new_multi_thread()
                                .enable_all()
                                .build()
                                .expect("tokio rt");
                            rt.block_on(async move {
                                let _ = quic::run_client(
                                    server,
                                    fp,
                                    token,
                                    true,
                                    ui_tx_net,
                                    ctrl_rx,
                                    Some(app_tx),
                                )
                                .await;
                            });
                        });

                        let ctrl_tx_ui = ctrl_tx.clone();
                        std::thread::spawn(move || {
                            let _ = window::run_event_loop(ui_rx, ctrl_tx_ui);
                        });

                        let _ = ctrl_tx.send(ControlMessage::ListApps);
                        let apps = app_rx.blocking_recv().unwrap_or_default();

                        Ok((apps, ActiveSession { uri, ctrl_tx, ui_tx }))
                    })();

                    let _ = tx.send(res.map_err(|e| e.to_string()));
                });

                let status_label = status_label.clone();
                let term_btn = term_btn.clone();
                let run_btn = run_btn.clone();
                let session = Rc::clone(&session);
                let all_apps = Rc::clone(&all_apps);
                let populate_grid = populate_grid.clone();
                let search_entry = search_entry.clone();
                glib::timeout_add_local(std::time::Duration::from_millis(25), move || {
                    match rx.try_recv() {
                        Ok(Ok((apps, active))) => {
                            if let Some(old) = session.borrow_mut().take() {
                                let _ = old.ui_tx.send(ClientUiEvent::Disconnected);
                            }
                            *session.borrow_mut() = Some(active);
                            *all_apps.borrow_mut() = apps;
                            term_btn.set_sensitive(true);
                            run_btn.set_sensitive(true);
                            status_label.set_text("Connected");
                            populate_grid(&search_entry.text());
                            glib::ControlFlow::Break
                        }
                        Ok(Err(err)) => {
                            status_label.set_text(&format!("Error: {err}"));
                            glib::ControlFlow::Break
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            glib::ControlFlow::Break
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            glib::ControlFlow::Continue
                        }
                    }
                });
            }
        });

        window.set_child(Some(&root));
        window.present();
    }

    fn load_pixbuf(data: &[u8]) -> Option<gtk4::gdk_pixbuf::Pixbuf> {
        let loader = PixbufLoader::new();
        loader.write(data).ok()?;
        loader.close().ok()?;
        loader.pixbuf()
    }
}
