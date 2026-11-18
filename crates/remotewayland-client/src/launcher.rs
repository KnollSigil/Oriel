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
    use crate::quic::{self, LauncherNetEvent};
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
        let (net_tx, mut net_rx) = mpsc::unbounded_channel::<LauncherNetEvent>();

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
                        Some(net_tx),
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

#[cfg(feature = "gtk4")]
mod gtk4_ui {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::process::Child;
    use std::rc::Rc;

    use anyhow::Result;
    use gtk4::gdk_pixbuf::PixbufLoader;
    use gtk4::glib;
    use gtk4::prelude::*;
    use remotewayland_proto::{ControlMessage, LaunchState, RemoteAppEntry};
    use tokio::sync::mpsc;

    use crate::hosts::{load_hosts, parse_uri, save_hosts, ssh_bootstrap, HostProfile};
    use crate::pty::spawn_local_terminal_pty;
    use crate::quic::{self, LauncherNetEvent};
    use crate::window::{self, ClientUiEvent};

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ConnStatus {
        Disconnected,
        Connecting,
        Connected,
    }

    struct TrackedProgram {
        launch_id: u64,
        name: String,
        state: LaunchState,
        term_sent: bool,
        pty_child: Option<Child>,
    }

    struct HostSession {
        status: ConnStatus,
        error: Option<String>,
        uri: Option<String>,
        ctrl_tx: Option<mpsc::UnboundedSender<ControlMessage>>,
        ui_tx: Option<window::UiSender>,
        net_rx: Option<mpsc::UnboundedReceiver<LauncherNetEvent>>,
        connect_rx: Option<std::sync::mpsc::Receiver<Result<ConnectResult, String>>>,
        apps: Vec<RemoteAppEntry>,
        programs: Vec<TrackedProgram>,
        rtt_ms: Option<u32>,
        congested: bool,
    }

    struct ConnectResult {
        uri: String,
        ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
        ui_tx: window::UiSender,
        net_rx: mpsc::UnboundedReceiver<LauncherNetEvent>,
    }

    struct AppUi {
        hosts: Vec<HostProfile>,
        sessions: HashMap<String, HostSession>,
        active_target: Option<String>,
        next_launch_id: u64,
        grid_apps: Vec<RemoteAppEntry>,
        icon_cache: HashMap<String, Option<gtk4::gdk::Texture>>,
    }

    type CallbackSlot = Rc<RefCell<Option<Rc<dyn Fn()>>>>;

    impl AppUi {
        fn active_session_mut(&mut self) -> Option<&mut HostSession> {
            let key = self.active_target.as_ref()?;
            self.sessions.get_mut(key)
        }

        fn active_session(&self) -> Option<&HostSession> {
            let key = self.active_target.as_ref()?;
            self.sessions.get(key)
        }
    }

    pub fn run() -> Result<()> {
        let app = gtk4::Application::builder()
            .application_id("org.remotewayland.Launcher")
            .build();

        app.connect_activate(build_ui);
        app.run_with_args(&Vec::<String>::new());
        Ok(())
    }

    fn install_css() {
        let provider = gtk4::CssProvider::new();
        provider.load_from_string(
            "
            flowboxchild {
                border-radius: 8px;
                padding: 4px;
                border: 2px solid transparent;
                color: inherit;
            }
            flowboxchild:hover {
                background-color: alpha(currentColor, 0.10);
            }
            flowboxchild:selected {
                background-color: rgba(53, 132, 228, 0.18);
                border: 2px solid rgba(53, 132, 228, 0.55);
                color: inherit;
            }
            flowboxchild:focus {
                background-color: rgba(53, 132, 228, 0.30);
                border: 2px solid #3584e4;
                color: inherit;
            }
            flowboxchild:selected label,
            flowboxchild:focus label {
                color: inherit;
            }
            .server-list row {
                padding: 1px 3px;
                min-height: 22px;
                border-radius: 5px;
            }
            .server-list row:selected {
                background-color: rgba(53, 132, 228, 0.20);
                color: inherit;
            }
            .server-list row:selected label {
                color: inherit;
            }
            .server-list row:focus {
                outline: 2px solid #3584e4;
                outline-offset: -2px;
            }
            searchentry:focus-within {
                outline: 2px solid rgba(53, 132, 228, 0.80);
                outline-offset: 1px;
            }
            .compact-btn {
                min-height: 18px;
                min-width: 18px;
                padding: 1px 4px;
            }
            ",
        );
        if let Some(display) = gtk4::gdk::Display::default() {
            gtk4::style_context_add_provider_for_display(
                &display,
                &provider,
                gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
            );
        }
    }

    fn build_ui(app: &gtk4::Application) {
        install_css();

        let window = gtk4::ApplicationWindow::builder()
            .application(app)
            .title("RemoteWayland")
            .default_width(960)
            .default_height(640)
            .build();

        let initial_hosts = load_hosts();
        let initial_target = initial_hosts.first().map(|h| h.ssh_target.clone());
        let seed_id = ((std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64)
            & 0xFFFF_FFFF)
            << 16
            | 1;
        let state = Rc::new(RefCell::new(AppUi {
            hosts: initial_hosts,
            sessions: HashMap::new(),
            active_target: initial_target,
            next_launch_id: seed_id,
            grid_apps: Vec::new(),
            icon_cache: HashMap::new(),
        }));
        let current_filter = Rc::new(RefCell::new(String::new()));

        let outer = gtk4::Box::new(gtk4::Orientation::Vertical, 0);

        // Main split: draggable left server sidebar + right app launcher
        let paned = gtk4::Paned::new(gtk4::Orientation::Horizontal);
        paned.set_vexpand(true);
        paned.set_position(171);
        paned.set_resize_start_child(false);
        paned.set_shrink_start_child(true);

        // --- Left Sidebar ---
        let sidebar = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        sidebar.set_margin_top(6);
        sidebar.set_margin_bottom(6);
        sidebar.set_margin_start(4);
        sidebar.set_margin_end(4);

        let side_hdr = gtk4::Box::new(gtk4::Orientation::Horizontal, 2);
        let side_title = gtk4::Label::new(Some("Servers"));
        side_title.set_halign(gtk4::Align::Start);
        side_title.set_hexpand(true);
        side_title.set_ellipsize(gtk4::pango::EllipsizeMode::End);
        side_title.add_css_class("heading");
        let manage_btn = gtk4::Button::with_label("⚙");
        manage_btn.set_tooltip_text(Some("Manage connections"));
        manage_btn.add_css_class("compact-btn");
        side_hdr.append(&side_title);
        side_hdr.append(&manage_btn);
        sidebar.append(&side_hdr);

        let server_list = gtk4::ListBox::new();
        server_list.add_css_class("server-list");
        server_list.set_selection_mode(gtk4::SelectionMode::Single);
        let server_scroll = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .child(&server_list)
            .build();
        sidebar.append(&server_scroll);

        // --- Right Main Pane ---
        let right_pane = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        right_pane.set_hexpand(true);
        right_pane.set_vexpand(true);
        right_pane.set_margin_top(10);
        right_pane.set_margin_bottom(8);
        right_pane.set_margin_start(10);
        right_pane.set_margin_end(10);

        paned.set_start_child(Some(&sidebar));
        paned.set_end_child(Some(&right_pane));

        // Search + Terminal bar (KDE launcher style)
        let top_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        let search_entry = gtk4::SearchEntry::builder()
            .placeholder_text("Type to search apps or enter command, then press Enter…")
            .hexpand(true)
            .build();
        let term_btn = gtk4::Button::with_label("New Terminal");
        term_btn.set_sensitive(false);

        // Host microphone selector + mute control (live-switches PipeWire source).
        let mic_sources: Rc<RefCell<Vec<crate::audio::HostMicSource>>> =
            Rc::new(RefCell::new(Vec::new()));
        let updating_mics = Rc::new(std::cell::Cell::new(false));
        let mic_dropdown = gtk4::DropDown::from_strings(&["Mic: Off (Muted)", "Mic: System Default"]);
        mic_dropdown.set_tooltip_text(Some("Select or mute host microphone for remote apps"));

        let populate_mics = {
            let mic_sources = Rc::clone(&mic_sources);
            let updating_mics = Rc::clone(&updating_mics);
            move |dropdown: &gtk4::DropDown| {
                updating_mics.set(true);
                let cfg = crate::audio::subscribe_mic_config().borrow().clone();
                let list = crate::audio::list_host_mic_sources();
                let mut labels: Vec<String> = Vec::with_capacity(list.len() + 2);
                labels.push("Mic: Off (Muted)".to_string());
                labels.push("Mic: System Default".to_string());
                let mut selected_idx = if cfg.muted { 0u32 } else { 1u32 };
                for (i, src) in list.iter().enumerate() {
                    labels.push(format!("Mic: {}", src.description));
                    if !cfg.muted && cfg.target.as_deref() == Some(src.node_name.as_str()) {
                        selected_idx = (i + 2) as u32;
                    }
                }
                *mic_sources.borrow_mut() = list;
                let str_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                let model = gtk4::StringList::new(&str_refs);
                dropdown.set_model(Some(&model));
                dropdown.set_selected(selected_idx);
                updating_mics.set(false);
            }
        };
        populate_mics(&mic_dropdown);

        mic_dropdown.connect_selected_notify({
            let mic_sources = Rc::clone(&mic_sources);
            let updating_mics = Rc::clone(&updating_mics);
            move |dd| {
                if updating_mics.get() {
                    return;
                }
                let idx = dd.selected();
                if idx == 0 {
                    crate::audio::set_mic_muted(true);
                } else if idx == 1 {
                    crate::audio::set_mic_muted(false);
                    crate::audio::set_mic_target(None);
                } else if let Some(src) = mic_sources.borrow().get((idx - 2) as usize) {
                    crate::audio::set_mic_muted(false);
                    crate::audio::set_mic_target(Some(src.node_name.clone()));
                }
            }
        });

        let mic_refresh_btn = gtk4::Button::with_label("↻");
        mic_refresh_btn.set_tooltip_text(Some("Refresh host microphones"));
        mic_refresh_btn.connect_clicked({
            let mic_dropdown = mic_dropdown.clone();
            move |_| {
                populate_mics(&mic_dropdown);
            }
        });

        top_bar.append(&search_entry);
        top_bar.append(&mic_dropdown);
        top_bar.append(&mic_refresh_btn);
        top_bar.append(&term_btn);
        right_pane.append(&top_bar);

        // App icon grid (filtered in-place via set_filter_func for instant typing/backspace)
        let flow = gtk4::FlowBox::builder()
            .valign(gtk4::Align::Start)
            .max_children_per_line(8)
            .min_children_per_line(3)
            .selection_mode(gtk4::SelectionMode::Single)
            .activate_on_single_click(true)
            .row_spacing(8)
            .column_spacing(8)
            .build();
        flow.set_filter_func({
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            move |child| {
                let needle = current_filter.borrow();
                if needle.is_empty() {
                    return true;
                }
                let idx = child.index() as usize;
                state
                    .borrow()
                    .grid_apps
                    .get(idx)
                    .is_some_and(|a| a.name.to_lowercase().contains(needle.as_str()))
            }
        });

        let grid_scroll = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .child(&flow)
            .build();
        right_pane.append(&grid_scroll);

        outer.append(&paned);
        outer.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        // --- Compact Bottom Status Bar + Popover ---
        let status_bar_btn = gtk4::MenuButton::new();
        status_bar_btn.set_has_frame(false);
        status_bar_btn.set_hexpand(true);
        status_bar_btn.set_direction(gtk4::ArrowType::Up);

        let status_bar_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        let status_dot = gtk4::Label::new(Some("●"));
        let status_net_lbl = gtk4::Label::new(Some("Disconnected"));
        status_net_lbl.set_hexpand(true);
        status_net_lbl.set_halign(gtk4::Align::Start);
        status_net_lbl.set_ellipsize(gtk4::pango::EllipsizeMode::End);
        status_net_lbl.set_single_line_mode(true);
        let status_pills_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        status_pills_box.set_halign(gtk4::Align::End);
        let status_count_lbl = gtk4::Label::new(Some("0 active"));
        status_bar_box.append(&status_dot);
        status_bar_box.append(&status_net_lbl);
        status_bar_box.append(&status_pills_box);
        status_bar_box.append(&status_count_lbl);
        status_bar_btn.set_child(Some(&status_bar_box));

        let programs_popover = gtk4::Popover::new();
        let programs_box = gtk4::Box::new(gtk4::Orientation::Vertical, 6);
        programs_box.set_margin_top(8);
        programs_box.set_margin_bottom(8);
        programs_box.set_margin_start(8);
        programs_box.set_margin_end(8);
        programs_box.set_width_request(340);
        let programs_list = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        programs_box.append(&programs_list);
        programs_popover.set_child(Some(&programs_box));
        status_bar_btn.set_popover(Some(&programs_popover));

        let collect_kill_buttons = |list: &gtk4::Box| -> Vec<gtk4::Button> {
            let mut btns = Vec::new();
            let mut row = list.first_child();
            while let Some(r) = row {
                let mut child = r.first_child();
                while let Some(c) = child {
                    if let Ok(btn) = c.clone().downcast::<gtk4::Button>() {
                        btns.push(btn);
                    }
                    child = c.next_sibling();
                }
                row = r.next_sibling();
            }
            btns
        };

        let pop_key_ctrl = gtk4::EventControllerKey::new();
        pop_key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        pop_key_ctrl.connect_key_pressed({
            let programs_list = programs_list.clone();
            let programs_popover = programs_popover.clone();
            let status_bar_btn = status_bar_btn.clone();
            let window_ref = window.clone();
            move |_, key, _, mods| {
                if key == gtk4::gdk::Key::Escape {
                    programs_popover.popdown();
                    status_bar_btn.grab_focus();
                    return glib::Propagation::Stop;
                }
                let is_next = key == gtk4::gdk::Key::Down
                    || (key == gtk4::gdk::Key::Tab
                        && !mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK));
                let is_prev = key == gtk4::gdk::Key::Up
                    || key == gtk4::gdk::Key::ISO_Left_Tab
                    || (key == gtk4::gdk::Key::Tab
                        && mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK));
                if is_next || is_prev {
                    let btns = collect_kill_buttons(&programs_list);
                    if btns.is_empty() {
                        return glib::Propagation::Stop;
                    }
                    let focused = gtk4::prelude::GtkWindowExt::focus(&window_ref)
                        .or_else(|| programs_popover.focus_child());
                    let cur = focused.as_ref().and_then(|f| {
                        btns.iter()
                            .position(|b| b.upcast_ref::<gtk4::Widget>() == f || f.is_ancestor(b))
                    });
                    let next_idx = match cur {
                        Some(idx) if is_next => (idx + 1) % btns.len(),
                        Some(0) => btns.len() - 1,
                        Some(idx) => idx - 1,
                        None if is_prev => btns.len() - 1,
                        None => 0,
                    };
                    btns[next_idx].grab_focus();
                    return glib::Propagation::Stop;
                }
                glib::Propagation::Proceed
            }
        });
        programs_popover.add_controller(pop_key_ctrl);

        outer.append(&status_bar_btn);

        // --- Helpers ---
        let refresh_net_status: Rc<dyn Fn()> = {
            let state = Rc::clone(&state);
            let status_net_lbl = status_net_lbl.clone();
            let status_dot = status_dot.clone();
            let term_btn = term_btn.clone();

            Rc::new(move || {
                let st = state.borrow();
                let Some(target) = st.active_target.as_ref() else {
                    status_dot.set_markup("<span foreground='#888888'>●</span>");
                    status_net_lbl.set_text("No server selected");
                    term_btn.set_sensitive(false);
                    return;
                };

                let Some(sess) = st.sessions.get(target) else {
                    status_dot.set_markup("<span foreground='#888888'>●</span>");
                    status_net_lbl.set_text(&format!("{target} · Disconnected"));
                    term_btn.set_sensitive(false);
                    return;
                };

                match sess.status {
                    ConnStatus::Disconnected => {
                        status_dot.set_markup("<span foreground='#cc4444'>●</span>");
                        let msg = sess
                            .error
                            .as_deref()
                            .map(|e| format!("{target} · Disconnected ({e})"))
                            .unwrap_or_else(|| format!("{target} · Disconnected"));
                        status_net_lbl.set_text(&msg);
                        status_net_lbl.set_tooltip_text(Some(&msg));
                        term_btn.set_sensitive(false);
                    }
                    ConnStatus::Connecting => {
                        status_dot.set_markup("<span foreground='#ddaa22'>●</span>");
                        status_net_lbl.set_text(&format!("{target} · Connecting via SSH…"));
                        term_btn.set_sensitive(false);
                    }
                    ConnStatus::Connected => {
                        term_btn.set_sensitive(true);
                        if sess.congested {
                            status_dot.set_markup("<span foreground='#dd8822'>●</span>");
                            let rtt = sess.rtt_ms.unwrap_or(0);
                            status_net_lbl.set_text(&format!("{target} · Congested · {rtt}ms"));
                        } else {
                            status_dot.set_markup("<span foreground='#44bb55'>●</span>");
                            let rtt_str = sess
                                .rtt_ms
                                .map(|r| format!(" · {r}ms"))
                                .unwrap_or_default();
                            status_net_lbl.set_text(&format!("{target} · Connected{rtt_str}"));
                        }
                    }
                }
            })
        };

        let refresh_programs_ui: Rc<dyn Fn()> = {
            let state = Rc::clone(&state);
            let status_pills_box = status_pills_box.clone();
            let status_count_lbl = status_count_lbl.clone();
            let programs_list = programs_list.clone();
            let programs_popover = programs_popover.clone();

            Rc::new(move || {
                while let Some(c) = status_pills_box.first_child() {
                    status_pills_box.remove(&c);
                }
                while let Some(c) = programs_list.first_child() {
                    programs_list.remove(&c);
                }

                let st = state.borrow();
                let Some(target) = st.active_target.as_ref() else {
                    status_count_lbl.set_text("0 active");
                    return;
                };
                let Some(sess) = st.sessions.get(target) else {
                    status_count_lbl.set_text("0 active");
                    return;
                };

                let active_count = sess.programs.len();
                status_count_lbl.set_text(&format!("{active_count} active"));

                for prog in &sess.programs {
                    if prog.state == LaunchState::Launching {
                        let pill = gtk4::Label::new(Some(&format!("Launching {}…", prog.name)));
                        status_pills_box.append(&pill);
                    }

                    let row = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
                    let state_str = match prog.state {
                        LaunchState::Launching => "Launching",
                        LaunchState::Running => "Running",
                        LaunchState::Exited(_) => "Exited",
                    };
                    let lbl = gtk4::Label::new(Some(&format!("{} — {}", prog.name, state_str)));
                    lbl.set_halign(gtk4::Align::Start);
                    lbl.set_hexpand(true);
                    row.append(&lbl);

                    let kill_label = if prog.term_sent { "Kill" } else { "×" };
                    let kill_btn = gtk4::Button::with_label(kill_label);
                    let target_c = target.clone();
                    let launch_id = prog.launch_id;
                    let state_c = Rc::clone(&state);
                    kill_btn.connect_clicked(move |btn| {
                        let mut st = state_c.borrow_mut();
                        if let Some(sess) = st.sessions.get_mut(&target_c) {
                            if let Some(p) =
                                sess.programs.iter_mut().find(|p| p.launch_id == launch_id)
                            {
                                if let Some(child) = p.pty_child.as_mut() {
                                    let _ = child.kill();
                                } else if let Some(tx) = &sess.ctrl_tx {
                                    let force = p.term_sent;
                                    p.term_sent = true;
                                    btn.set_label("Kill");
                                    let _ = tx.send(ControlMessage::TerminateApp {
                                        launch_id,
                                        force,
                                    });
                                }
                            }
                        }
                    });
                    row.append(&kill_btn);
                    programs_list.append(&row);
                }

                if sess.programs.is_empty() {
                    programs_list.append(&gtk4::Label::new(Some("No active programs")));
                } else if programs_popover.is_visible() && programs_popover.focus_child().is_some()
                {
                    if let Some(first) = collect_kill_buttons(&programs_list).into_iter().next() {
                        first.grab_focus();
                    }
                }
            })
        };

        let refresh_status_bar: Rc<dyn Fn()> = {
            let refresh_net_status = Rc::clone(&refresh_net_status);
            let refresh_programs_ui = Rc::clone(&refresh_programs_ui);
            Rc::new(move || {
                refresh_net_status();
                refresh_programs_ui();
            })
        };

        let launch_exec_or_app: Rc<dyn Fn(String, String)> = {
            let state = Rc::clone(&state);
            let search_entry = search_entry.clone();
            let refresh_status_bar = Rc::clone(&refresh_status_bar);
            Rc::new(move |name: String, exec: String| {
                {
                    let mut st = state.borrow_mut();
                    let launch_id = st.next_launch_id;
                    st.next_launch_id += 1;
                    if let Some(sess) = st.active_session_mut() {
                        if let Some(tx) = &sess.ctrl_tx {
                            sess.programs.push(TrackedProgram {
                                launch_id,
                                name: name.clone(),
                                state: LaunchState::Launching,
                                term_sent: false,
                                pty_child: None,
                            });
                            let _ = tx.send(ControlMessage::LaunchApp {
                                launch_id,
                                name: Some(name),
                                exec,
                            });
                        }
                    }
                }
                search_entry.set_text("");
                search_entry.grab_focus();
                refresh_status_bar();
            })
        };

        let select_first_visible: Rc<dyn Fn()> = {
            let flow = flow.clone();
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            Rc::new(move || {
                let needle = current_filter.borrow();
                let first_idx = state
                    .borrow()
                    .grid_apps
                    .iter()
                    .position(|a| needle.is_empty() || a.name.to_lowercase().contains(needle.as_str()));
                if let Some(idx) = first_idx {
                    if let Some(child) = flow.child_at_index(idx as i32) {
                        flow.select_child(&child);
                    }
                } else {
                    flow.unselect_all();
                }
            })
        };

        let rebuild_grid: Rc<dyn Fn()> = {
            let flow = flow.clone();
            let state = Rc::clone(&state);
            let select_first_visible = Rc::clone(&select_first_visible);
            Rc::new(move || {
                while let Some(child) = flow.first_child() {
                    flow.remove(&child);
                }
                let apps = {
                    let mut st = state.borrow_mut();
                    let apps = st
                        .active_session()
                        .map(|s| s.apps.clone())
                        .unwrap_or_default();
                    for entry in &apps {
                        if !st.icon_cache.contains_key(&entry.id) && !entry.icon_data.is_empty() {
                            let tex = load_pixbuf(&entry.icon_data)
                                .map(|pb| gtk4::gdk::Texture::for_pixbuf(&pb));
                            st.icon_cache.insert(entry.id.clone(), tex);
                        }
                    }
                    st.grid_apps = apps.clone();
                    apps
                };

                for entry in &apps {
                    let vbox = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
                    vbox.set_size_request(96, 84);
                    vbox.set_margin_top(4);
                    vbox.set_margin_bottom(4);
                    vbox.set_margin_start(4);
                    vbox.set_margin_end(4);
                    if let Some(Some(tex)) = state.borrow().icon_cache.get(&entry.id) {
                        let img = gtk4::Image::from_paintable(Some(tex));
                        img.set_pixel_size(48);
                        vbox.append(&img);
                    }
                    let lbl = gtk4::Label::new(Some(&entry.name));
                    lbl.set_ellipsize(gtk4::pango::EllipsizeMode::End);
                    lbl.set_max_width_chars(14);
                    vbox.append(&lbl);

                    let child = gtk4::FlowBoxChild::new();
                    child.set_tooltip_text(Some(&entry.name));
                    child.set_child(Some(&vbox));
                    flow.insert(&child, -1);
                }

                flow.invalidate_filter();
                select_first_visible();
            })
        };

        let start_connect: Rc<dyn Fn(HostProfile)> = {
            let state = Rc::clone(&state);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);
            Rc::new(move |profile: HostProfile| {
                let target = profile.ssh_target.clone();
                {
                    let mut st = state.borrow_mut();
                    st.active_target = Some(target.clone());
                    if let Some(existing) = st.sessions.get(&target) {
                        if existing.status != ConnStatus::Disconnected {
                            return;
                        }
                    }
                    let (tx, rx) = std::sync::mpsc::channel::<Result<ConnectResult, String>>();
                    st.sessions.insert(
                        target.clone(),
                        HostSession {
                            status: ConnStatus::Connecting,
                            error: None,
                            uri: None,
                            ctrl_tx: None,
                            ui_tx: None,
                            net_rx: None,
                            connect_rx: Some(rx),
                            apps: Vec::new(),
                            programs: Vec::new(),
                            rtt_ms: None,
                            congested: false,
                        },
                    );

                    std::thread::spawn(move || {
                        let res = (|| -> anyhow::Result<ConnectResult> {
                            let uri = ssh_bootstrap(
                                &profile.ssh_target,
                                profile.server_bin.as_deref(),
                                profile.port,
                            )?;
                            let (server, fp, token) = parse_uri(&uri)?;
                            let (ui_tx, ui_rx) = window::create_ui_channel();
                            let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
                            let (net_tx, net_rx) = mpsc::unbounded_channel();

                            let ui_tx_net = ui_tx.clone();
                            std::thread::spawn(move || {
                                let rt = tokio::runtime::Builder::new_multi_thread()
                                    .enable_all()
                                    .build()
                                    .expect("tokio rt");
                                rt.block_on(async move {
                                    if let Err(err) = quic::run_client(
                                        server,
                                        fp,
                                        token,
                                        profile.audio_enabled,
                                        ui_tx_net.clone(),
                                        ctrl_rx,
                                        Some(net_tx.clone()),
                                    )
                                    .await
                                    {
                                        let _ =
                                            net_tx.send(LauncherNetEvent::Disconnected(Some(
                                                err.to_string(),
                                            )));
                                        let _ = ui_tx_net.send(ClientUiEvent::Disconnected);
                                    }
                                });
                            });

                            let ctrl_tx_ui = ctrl_tx.clone();
                            std::thread::spawn(move || {
                                let _ = window::run_event_loop(ui_rx, ctrl_tx_ui);
                            });

                            let _ = ctrl_tx.send(ControlMessage::ListApps);
                            Ok(ConnectResult {
                                uri,
                                ctrl_tx,
                                ui_tx,
                                net_rx,
                            })
                        })();
                        let _ = tx.send(res.map_err(|e| e.to_string()));
                    });
                }
                refresh_status_bar();
            })
        };

        let rebuild_sidebar: Rc<dyn Fn()> = {
            let state_cb = Rc::clone(&state);
            let server_list_cb = server_list.clone();
            let rebuild_grid_cb = Rc::clone(&rebuild_grid);
            let refresh_status_bar_cb = Rc::clone(&refresh_status_bar);

            let self_ref: CallbackSlot = Rc::new(RefCell::new(None));
            let self_ref_clone = Rc::clone(&self_ref);

            let cb: Rc<dyn Fn()> = Rc::new(move || {
                while let Some(c) = server_list_cb.first_child() {
                    server_list_cb.remove(&c);
                }
                let (hosts_snap, active_snap, statuses) = {
                    let st = state_cb.borrow();
                    let statuses: HashMap<String, ConnStatus> = st
                        .sessions
                        .iter()
                        .map(|(k, s)| (k.clone(), s.status))
                        .collect();
                    (st.hosts.clone(), st.active_target.clone(), statuses)
                };

                for profile in hosts_snap {
                    let row_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
                    row_box.set_margin_top(1);
                    row_box.set_margin_bottom(1);
                    row_box.set_margin_start(3);
                    row_box.set_margin_end(2);

                    let status = statuses
                        .get(&profile.ssh_target)
                        .copied()
                        .unwrap_or(ConnStatus::Disconnected);
                    let dot = gtk4::Label::new(None);
                    let color = match status {
                        ConnStatus::Connected => "#44bb55",
                        ConnStatus::Connecting => "#ddaa22",
                        ConnStatus::Disconnected => "#888888",
                    };
                    dot.set_markup(&format!("<span foreground='{color}'>●</span>"));

                    let lbl = gtk4::Label::new(Some(&profile.label));
                    lbl.set_halign(gtk4::Align::Start);
                    lbl.set_hexpand(true);
                    lbl.set_ellipsize(gtk4::pango::EllipsizeMode::End);

                    let dc_btn = gtk4::Button::with_label("⏹");
                    dc_btn.set_has_frame(false);
                    dc_btn.set_can_focus(false);
                    dc_btn.set_focus_on_click(false);
                    dc_btn.add_css_class("compact-btn");
                    dc_btn.set_tooltip_text(Some("Disconnect"));
                    dc_btn.set_sensitive(status != ConnStatus::Disconnected);
                    let dc_target = profile.ssh_target.clone();
                    let state_dc = Rc::clone(&state_cb);
                    let self_ref_dc = Rc::clone(&self_ref_clone);
                    let refresh_dc = Rc::clone(&refresh_status_bar_cb);
                    let grid_dc = Rc::clone(&rebuild_grid_cb);
                    dc_btn.connect_clicked(move |_| {
                        {
                            let mut st = state_dc.borrow_mut();
                            if let Some(sess) = st.sessions.get_mut(&dc_target) {
                                if let Some(tx) = sess.ui_tx.take() {
                                    let _ = tx.send(ClientUiEvent::Disconnected);
                                }
                                sess.status = ConnStatus::Disconnected;
                                sess.ctrl_tx = None;
                                sess.net_rx = None;
                                sess.connect_rx = None;
                                sess.apps.clear();
                                sess.programs.clear();
                                sess.rtt_ms = None;
                                sess.congested = false;
                            }
                        }
                        if let Some(f) = self_ref_dc.borrow().as_ref() {
                            f();
                        }
                        grid_dc();
                        refresh_dc();
                    });

                    row_box.append(&dot);
                    row_box.append(&lbl);
                    row_box.append(&dc_btn);

                    let row = gtk4::ListBoxRow::new();
                    row.set_child(Some(&row_box));
                    row.set_widget_name(&profile.ssh_target);
                    server_list_cb.append(&row);

                    if active_snap.as_deref() == Some(&profile.ssh_target) {
                        server_list_cb.select_row(Some(&row));
                    }
                }
            });

            *self_ref.borrow_mut() = Some(Rc::clone(&cb));

            server_list.connect_row_activated({
                let state = Rc::clone(&state);
                let start_connect = Rc::clone(&start_connect);
                let rebuild_grid = Rc::clone(&rebuild_grid);
                let search_entry = search_entry.clone();
                let refresh_status_bar = Rc::clone(&refresh_status_bar);
                let cb = Rc::clone(&cb);
                move |_, row| {
                    let target = row.widget_name().to_string();
                    let profile = state
                        .borrow()
                        .hosts
                        .iter()
                        .find(|h| h.ssh_target == target)
                        .cloned();
                    if let Some(p) = profile {
                        start_connect(p);
                        cb();
                        rebuild_grid();
                        refresh_status_bar();
                        search_entry.grab_focus();
                    }
                }
            });

            cb
        };

        // Manage Connections dialog
        manage_btn.connect_clicked({
            let window = window.clone();
            let state = Rc::clone(&state);
            let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
            let rebuild_grid = Rc::clone(&rebuild_grid);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);

            move |_| {
                open_manage_connections_dialog(
                    &window,
                    Rc::clone(&state),
                    Rc::clone(&rebuild_sidebar),
                    Rc::clone(&rebuild_grid),
                    Rc::clone(&refresh_status_bar),
                );
            }
        });

        // Instant in-place search filter + KDE-style Enter to launch & reset
        search_entry.connect_search_changed({
            let current_filter = Rc::clone(&current_filter);
            let flow = flow.clone();
            let select_first_visible = Rc::clone(&select_first_visible);
            move |e| {
                *current_filter.borrow_mut() = e.text().trim().to_lowercase();
                flow.invalidate_filter();
                select_first_visible();
            }
        });

        search_entry.connect_activate({
            let state = Rc::clone(&state);
            let flow = flow.clone();
            let current_filter = Rc::clone(&current_filter);
            let launch_exec_or_app = Rc::clone(&launch_exec_or_app);
            move |e| {
                let typed = e.text().trim().to_string();
                let selected_idx = flow
                    .selected_children()
                    .first()
                    .map(|c| c.index() as usize);
                let picked = {
                    let st = state.borrow();
                    let needle = current_filter.borrow();
                    selected_idx
                        .and_then(|idx| st.grid_apps.get(idx))
                        .filter(|a| {
                            needle.is_empty() || a.name.to_lowercase().contains(needle.as_str())
                        })
                        .or_else(|| {
                            st.grid_apps.iter().find(|a| {
                                needle.is_empty()
                                    || a.name.to_lowercase().contains(needle.as_str())
                            })
                        })
                        .cloned()
                };
                if let Some(app) = picked {
                    launch_exec_or_app(app.name, app.exec);
                } else if !typed.is_empty() {
                    launch_exec_or_app(typed.clone(), typed);
                }
            }
        });

        // Arrow Down from search_entry focuses FlowBox; Escape clears search
        let key_ctrl = gtk4::EventControllerKey::new();
        key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        key_ctrl.connect_key_pressed({
            let flow = flow.clone();
            let search_entry = search_entry.clone();
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            move |_, key, _, _| {
                if key == gtk4::gdk::Key::Down {
                    let target = flow.selected_children().into_iter().next().or_else(|| {
                        let needle = current_filter.borrow();
                        let idx = state.borrow().grid_apps.iter().position(|a| {
                            needle.is_empty() || a.name.to_lowercase().contains(needle.as_str())
                        })?;
                        flow.child_at_index(idx as i32)
                    });
                    if let Some(child) = target {
                        flow.select_child(&child);
                        child.grab_focus();
                        return glib::Propagation::Stop;
                    }
                } else if key == gtk4::gdk::Key::Escape {
                    search_entry.set_text("");
                    return glib::Propagation::Stop;
                }
                glib::Propagation::Proceed
            }
        });
        search_entry.add_controller(key_ctrl);

        // FlowBox key controller: deterministic Left/Right/Up/Down navigation, plus typing/Backspace/Escape -> search_entry
        let flow_key_ctrl = gtk4::EventControllerKey::new();
        flow_key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        flow_key_ctrl.connect_key_pressed({
            let flow = flow.clone();
            let search_entry = search_entry.clone();
            let window_ref = window.clone();
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            move |_, key, _, mods| {
                let is_arrow = matches!(
                    key,
                    gtk4::gdk::Key::Left
                        | gtk4::gdk::Key::Right
                        | gtk4::gdk::Key::Up
                        | gtk4::gdk::Key::Down
                );
                if is_arrow {
                    let needle = current_filter.borrow();
                    let apps = &state.borrow().grid_apps;
                    let mut visible: Vec<(gtk4::FlowBoxChild, f32, f32)> = Vec::new();
                    for (i, a) in apps.iter().enumerate() {
                        if needle.is_empty() || a.name.to_lowercase().contains(needle.as_str()) {
                            if let Some(ch) = flow.child_at_index(i as i32) {
                                if let Some(b) = ch.compute_bounds(&flow) {
                                    visible.push((ch, b.x(), b.y()));
                                }
                            }
                        }
                    }
                    drop(needle);

                    if visible.is_empty() {
                        return glib::Propagation::Proceed;
                    }

                    let focused = gtk4::prelude::GtkWindowExt::focus(&window_ref);
                    let cur_pos = focused
                        .as_ref()
                        .and_then(|f| {
                            visible
                                .iter()
                                .position(|(c, _, _)| c.upcast_ref::<gtk4::Widget>() == f || f.is_ancestor(c))
                        })
                        .or_else(|| {
                            flow.selected_children().first().and_then(|sel| {
                                visible.iter().position(|(c, _, _)| c == sel)
                            })
                        })
                        .unwrap_or(0);

                    let (_, cur_x, cur_y) = visible[cur_pos];
                    let next_child = match key {
                        gtk4::gdk::Key::Left => {
                            if cur_pos > 0 {
                                Some(visible[cur_pos - 1].0.clone())
                            } else {
                                None
                            }
                        }
                        gtk4::gdk::Key::Right => {
                            if cur_pos + 1 < visible.len() {
                                Some(visible[cur_pos + 1].0.clone())
                            } else {
                                None
                            }
                        }
                        gtk4::gdk::Key::Up => {
                            let min_y = visible.iter().map(|(_, _, y)| *y).fold(f32::INFINITY, f32::min);
                            if (cur_y - min_y).abs() < 8.0 {
                                search_entry.grab_focus();
                                search_entry.set_position(-1);
                                return glib::Propagation::Stop;
                            }
                            let row_above_y = visible
                                .iter()
                                .filter(|(_, _, y)| *y < cur_y - 8.0)
                                .map(|(_, _, y)| *y)
                                .fold(f32::NEG_INFINITY, f32::max);
                            visible
                                .iter()
                                .filter(|(_, _, y)| (*y - row_above_y).abs() < 8.0)
                                .min_by(|(_, x1, _), (_, x2, _)| {
                                    (x1 - cur_x).abs().total_cmp(&(x2 - cur_x).abs())
                                })
                                .map(|(c, _, _)| c.clone())
                        }
                        gtk4::gdk::Key::Down => {
                            let row_below_y = visible
                                .iter()
                                .filter(|(_, _, y)| *y > cur_y + 8.0)
                                .map(|(_, _, y)| *y)
                                .fold(f32::INFINITY, f32::min);
                            if row_below_y.is_finite() {
                                visible
                                    .iter()
                                    .filter(|(_, _, y)| (*y - row_below_y).abs() < 8.0)
                                    .min_by(|(_, x1, _), (_, x2, _)| {
                                        (x1 - cur_x).abs().total_cmp(&(x2 - cur_x).abs())
                                    })
                                    .map(|(c, _, _)| c.clone())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let Some(target) = next_child {
                        flow.select_child(&target);
                        target.grab_focus();
                    }
                    return glib::Propagation::Stop;
                }
                if key == gtk4::gdk::Key::BackSpace || key == gtk4::gdk::Key::Delete {
                    let mut s = search_entry.text().to_string();
                    s.pop();
                    search_entry.set_text(&s);
                    search_entry.grab_focus();
                    search_entry.set_position(-1);
                    return glib::Propagation::Stop;
                }
                if key == gtk4::gdk::Key::Escape {
                    search_entry.set_text("");
                    search_entry.grab_focus();
                    return glib::Propagation::Stop;
                }
                if !mods.intersects(
                    gtk4::gdk::ModifierType::CONTROL_MASK | gtk4::gdk::ModifierType::ALT_MASK,
                ) {
                    if let Some(ch) = key.to_unicode() {
                        if !ch.is_control() {
                            let mut s = search_entry.text().to_string();
                            s.push(ch);
                            search_entry.set_text(&s);
                            search_entry.grab_focus();
                            search_entry.set_position(-1);
                            return glib::Propagation::Stop;
                        }
                    }
                }
                glib::Propagation::Proceed
            }
        });
        flow.add_controller(flow_key_ctrl);

        // Clicking or activating a FlowBox item launches & resets
        flow.connect_child_activated({
            let state = Rc::clone(&state);
            let launch_exec_or_app = Rc::clone(&launch_exec_or_app);
            move |_, child| {
                let idx = child.index() as usize;
                let picked = state.borrow().grid_apps.get(idx).cloned();
                if let Some(app) = picked {
                    launch_exec_or_app(app.name, app.exec);
                }
            }
        });

        // New Terminal button
        term_btn.connect_clicked({
            let state = Rc::clone(&state);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);
            let search_entry = search_entry.clone();
            move |_| {
                {
                    let mut st = state.borrow_mut();
                    let launch_id = st.next_launch_id;
                    st.next_launch_id += 1;
                    if let Some(sess) = st.active_session_mut() {
                        if let Some(uri) = sess.uri.clone() {
                            if let Ok(child) = spawn_local_terminal_pty(&uri, None) {
                                sess.programs.push(TrackedProgram {
                                    launch_id,
                                    name: "Terminal (PTY)".to_string(),
                                    state: LaunchState::Running,
                                    term_sent: false,
                                    pty_child: Some(child),
                                });
                            }
                        }
                    }
                }
                search_entry.grab_focus();
                refresh_status_bar();
            }
        });

        // Clean Tab / Shift+Tab focus ring:
        // manage_btn <-> server_list <-> (search_entry + flow) <-> term_btn <-> status_bar_btn
        let win_key_ctrl = gtk4::EventControllerKey::new();
        win_key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        win_key_ctrl.connect_key_pressed({
            let window_ref = window.clone();
            let manage_btn = manage_btn.clone();
            let server_list = server_list.clone();
            let search_entry = search_entry.clone();
            let flow = flow.clone();
            let term_btn = term_btn.clone();
            let status_bar_btn = status_bar_btn.clone();
            move |_, key, _, mods| {
                let is_shift_tab = key == gtk4::gdk::Key::ISO_Left_Tab
                    || (key == gtk4::gdk::Key::Tab
                        && mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK));
                let is_tab = key == gtk4::gdk::Key::Tab
                    && !mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK);
                if !is_tab && !is_shift_tab {
                    return glib::Propagation::Proceed;
                }

                let Some(focused) = gtk4::prelude::GtkWindowExt::focus(&window_ref) else {
                    return glib::Propagation::Proceed;
                };
                let focus_servers = || {
                    if let Some(row) = server_list
                        .selected_row()
                        .or_else(|| server_list.row_at_index(0))
                    {
                        row.grab_focus();
                    } else {
                        server_list.grab_focus();
                    }
                };

                let in_manage = focused == manage_btn || focused.is_ancestor(&manage_btn);
                let in_servers = focused == server_list || focused.is_ancestor(&server_list);
                let in_launcher = focused == search_entry
                    || focused.is_ancestor(&search_entry)
                    || focused == flow
                    || focused.is_ancestor(&flow);
                let in_term = focused == term_btn || focused.is_ancestor(&term_btn);
                let in_status = focused == status_bar_btn || focused.is_ancestor(&status_bar_btn);

                if is_tab {
                    if in_manage {
                        focus_servers();
                    } else if in_servers {
                        search_entry.grab_focus();
                    } else if in_launcher {
                        if term_btn.is_sensitive() {
                            term_btn.grab_focus();
                        } else {
                            status_bar_btn.grab_focus();
                        }
                    } else if in_term {
                        status_bar_btn.grab_focus();
                    } else if in_status {
                        manage_btn.grab_focus();
                    } else {
                        return glib::Propagation::Proceed;
                    }
                    return glib::Propagation::Stop;
                }

                // Shift+Tab
                if in_manage {
                    status_bar_btn.grab_focus();
                } else if in_servers {
                    manage_btn.grab_focus();
                } else if in_launcher {
                    focus_servers();
                } else if in_term {
                    search_entry.grab_focus();
                } else if in_status {
                    if term_btn.is_sensitive() {
                        term_btn.grab_focus();
                    } else {
                        search_entry.grab_focus();
                    }
                } else {
                    return glib::Propagation::Proceed;
                }
                glib::Propagation::Stop
            }
        });
        window.add_controller(win_key_ctrl);

        // Auto-focus search box on Alt-Tab if active server is connected, else focus server list
        window.connect_is_active_notify({
            let state = Rc::clone(&state);
            let search_entry = search_entry.clone();
            let server_list = server_list.clone();
            move |win| {
                if win.is_active() {
                    let connected = state
                        .borrow()
                        .active_session()
                        .is_some_and(|s| s.status == ConnStatus::Connected);
                    if connected {
                        search_entry.grab_focus();
                    } else if let Some(row) = server_list
                        .selected_row()
                        .or_else(|| server_list.row_at_index(0))
                    {
                        row.grab_focus();
                    } else {
                        server_list.grab_focus();
                    }
                }
            }
        });

        // Periodic 50ms poll for SSH connect results, QUIC LauncherNetEvents, and PTY exits
        glib::timeout_add_local(std::time::Duration::from_millis(50), {
            let state = Rc::clone(&state);
            let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
            let rebuild_grid = Rc::clone(&rebuild_grid);
            let refresh_net_status = Rc::clone(&refresh_net_status);
            let refresh_programs_ui = Rc::clone(&refresh_programs_ui);

            move || {
                let mut sidebar_dirty = false;
                let mut grid_dirty = false;
                let mut net_dirty = false;
                let mut programs_dirty = false;

                {
                    let mut st = state.borrow_mut();
                    let active_key = st.active_target.clone();
                    for (key, sess) in st.sessions.iter_mut() {
                        let is_active = active_key.as_deref() == Some(key.as_str());

                        if let Some(rx) = sess.connect_rx.take() {
                            match rx.try_recv() {
                                Ok(Ok(res)) => {
                                    sess.status = ConnStatus::Connected;
                                    sess.uri = Some(res.uri);
                                    sess.ctrl_tx = Some(res.ctrl_tx);
                                    sess.ui_tx = Some(res.ui_tx);
                                    sess.net_rx = Some(res.net_rx);
                                    sidebar_dirty = true;
                                    if is_active {
                                        net_dirty = true;
                                        programs_dirty = true;
                                    }
                                }
                                Ok(Err(err)) => {
                                    sess.status = ConnStatus::Disconnected;
                                    sess.error = Some(err);
                                    sidebar_dirty = true;
                                    if is_active {
                                        net_dirty = true;
                                    }
                                }
                                Err(std::sync::mpsc::TryRecvError::Empty) => {
                                    sess.connect_rx = Some(rx);
                                }
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                    sess.status = ConnStatus::Disconnected;
                                    sidebar_dirty = true;
                                    if is_active {
                                        net_dirty = true;
                                    }
                                }
                            }
                        }

                        if let Some(net_rx) = sess.net_rx.as_mut() {
                            while let Ok(ev) = net_rx.try_recv() {
                                match ev {
                                    LauncherNetEvent::AppList(apps) => {
                                        sess.apps = apps;
                                        if is_active {
                                            grid_dirty = true;
                                        }
                                    }
                                    LauncherNetEvent::AppState {
                                        launch_id,
                                        name,
                                        state: new_state,
                                    } => {
                                        if matches!(new_state, LaunchState::Exited(_)) {
                                            sess.programs.retain(|p| p.launch_id != launch_id);
                                        } else if let Some(p) = sess
                                            .programs
                                            .iter_mut()
                                            .find(|p| p.launch_id == launch_id)
                                        {
                                            p.state = new_state;
                                        } else {
                                            sess.programs.push(TrackedProgram {
                                                launch_id,
                                                name,
                                                state: new_state,
                                                term_sent: false,
                                                pty_child: None,
                                            });
                                        }
                                        if is_active {
                                            programs_dirty = true;
                                        }
                                    }
                                    LauncherNetEvent::NetStats { rtt_ms, congested } => {
                                        sess.rtt_ms = Some(rtt_ms);
                                        sess.congested = congested;
                                        if is_active {
                                            net_dirty = true;
                                        }
                                    }
                                    LauncherNetEvent::Disconnected(err) => {
                                        sess.status = ConnStatus::Disconnected;
                                        sess.error = err;
                                        sess.ctrl_tx = None;
                                        sess.rtt_ms = None;
                                        sess.congested = false;
                                        sess.programs.clear();
                                        sidebar_dirty = true;
                                        if is_active {
                                            net_dirty = true;
                                            programs_dirty = true;
                                        }
                                    }
                                }
                            }
                        }

                        let before = sess.programs.len();
                        sess.programs.retain_mut(|p| {
                            if let Some(child) = p.pty_child.as_mut() {
                                !matches!(child.try_wait(), Ok(Some(_)))
                            } else {
                                true
                            }
                        });
                        if sess.programs.len() != before && is_active {
                            programs_dirty = true;
                        }
                    }
                }

                if sidebar_dirty {
                    rebuild_sidebar();
                }
                if grid_dirty {
                    rebuild_grid();
                }
                if net_dirty {
                    refresh_net_status();
                }
                if programs_dirty {
                    refresh_programs_ui();
                }

                glib::ControlFlow::Continue
            }
        });

        rebuild_sidebar();
        refresh_status_bar();
        window.set_child(Some(&outer));
        window.present();
        if let Some(row) = server_list
            .selected_row()
            .or_else(|| server_list.row_at_index(0))
        {
            row.grab_focus();
        } else {
            server_list.grab_focus();
        }
    }

    fn open_manage_connections_dialog(
        window: &gtk4::ApplicationWindow,
        state: Rc<RefCell<AppUi>>,
        rebuild_sidebar: Rc<dyn Fn()>,
        rebuild_grid: Rc<dyn Fn()>,
        refresh_status_bar: Rc<dyn Fn()>,
    ) {
        let dlg = gtk4::Window::builder()
            .title("Manage Connections")
            .transient_for(window)
            .modal(true)
            .default_width(560)
            .default_height(360)
            .build();

        let content = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        content.set_margin_top(12);
        content.set_margin_bottom(12);
        content.set_margin_start(12);
        content.set_margin_end(12);

        let list_box = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        let scroll = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .child(&list_box)
            .build();
        content.append(&scroll);

        let populate_manage: Rc<dyn Fn()> = {
            let list_box = list_box.clone();
            let state = Rc::clone(&state);
            let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
            let rebuild_grid = Rc::clone(&rebuild_grid);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);

            let self_ref: CallbackSlot = Rc::new(RefCell::new(None));
            let self_ref_c = Rc::clone(&self_ref);

            let f: Rc<dyn Fn()> = Rc::new(move || {
                while let Some(c) = list_box.first_child() {
                    list_box.remove(&c);
                }
                let hosts = state.borrow().hosts.clone();
                if hosts.is_empty() {
                    list_box.append(&gtk4::Label::new(Some("No saved connections.")));
                    return;
                }
                for profile in hosts {
                    let row = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
                    let e_label = gtk4::Entry::builder()
                        .text(&profile.label)
                        .placeholder_text("Label")
                        .width_chars(12)
                        .build();
                    let e_target = gtk4::Entry::builder()
                        .text(&profile.ssh_target)
                        .placeholder_text("user@host")
                        .hexpand(true)
                        .build();
                    let e_port = gtk4::Entry::builder()
                        .text(profile.port.to_string())
                        .width_chars(5)
                        .build();
                    let e_bin = gtk4::Entry::builder()
                        .text(profile.server_bin.as_deref().unwrap_or(""))
                        .placeholder_text("server_bin")
                        .width_chars(12)
                        .build();

                    let save_btn = gtk4::Button::with_label("Save");
                    let rm_btn = gtk4::Button::with_label("Remove");

                    let orig_target = profile.ssh_target.clone();
                    save_btn.connect_clicked({
                        let state = Rc::clone(&state);
                        let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
                        let rebuild_grid = Rc::clone(&rebuild_grid);
                        let refresh_status_bar = Rc::clone(&refresh_status_bar);
                        let self_ref_c = Rc::clone(&self_ref_c);
                        let orig_target = orig_target.clone();
                        let e_label = e_label.clone();
                        let e_target = e_target.clone();
                        let e_port = e_port.clone();
                        let e_bin = e_bin.clone();
                        move |_| {
                            let new_target = e_target.text().trim().to_string();
                            if new_target.is_empty() {
                                return;
                            }
                            let new_label = e_label.text().trim().to_string();
                            let new_port = e_port.text().trim().parse::<u16>().unwrap_or(7443);
                            let bin_raw = e_bin.text().trim().to_string();
                            let new_bin = if bin_raw.is_empty() {
                                None
                            } else {
                                Some(bin_raw)
                            };
                            let target_changed = new_target != orig_target;
                            {
                                let mut st = state.borrow_mut();
                                if let Some(h) =
                                    st.hosts.iter_mut().find(|h| h.ssh_target == orig_target)
                                {
                                    h.label = if new_label.is_empty() {
                                        new_target.clone()
                                    } else {
                                        new_label
                                    };
                                    h.ssh_target = new_target.clone();
                                    h.port = new_port;
                                    h.server_bin = new_bin;
                                }
                                if target_changed {
                                    if let Some(sess) = st.sessions.remove(&orig_target) {
                                        if let Some(tx) = sess.ui_tx {
                                            let _ = tx.send(ClientUiEvent::Disconnected);
                                        }
                                    }
                                    if st.active_target.as_deref() == Some(&orig_target) {
                                        st.active_target = Some(new_target);
                                    }
                                }
                                let _ = save_hosts(&st.hosts);
                            }
                            rebuild_sidebar();
                            if target_changed {
                                rebuild_grid();
                                refresh_status_bar();
                            }
                            if let Some(cb) = self_ref_c.borrow().as_ref() {
                                cb();
                            }
                        }
                    });

                    rm_btn.connect_clicked({
                        let state = Rc::clone(&state);
                        let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
                        let rebuild_grid = Rc::clone(&rebuild_grid);
                        let refresh_status_bar = Rc::clone(&refresh_status_bar);
                        let self_ref_c = Rc::clone(&self_ref_c);
                        let orig_target = orig_target.clone();
                        move |_| {
                            {
                                let mut st = state.borrow_mut();
                                st.hosts.retain(|h| h.ssh_target != orig_target);
                                let _ = save_hosts(&st.hosts);
                                if let Some(sess) = st.sessions.remove(&orig_target) {
                                    if let Some(tx) = sess.ui_tx {
                                        let _ = tx.send(ClientUiEvent::Disconnected);
                                    }
                                }
                                if st.active_target.as_deref() == Some(&orig_target) {
                                    st.active_target =
                                        st.hosts.first().map(|h| h.ssh_target.clone());
                                }
                            }
                            rebuild_sidebar();
                            rebuild_grid();
                            refresh_status_bar();
                            if let Some(cb) = self_ref_c.borrow().as_ref() {
                                cb();
                            }
                        }
                    });

                    row.append(&e_label);
                    row.append(&e_target);
                    row.append(&e_port);
                    row.append(&e_bin);
                    row.append(&save_btn);
                    row.append(&rm_btn);
                    list_box.append(&row);
                }
            });
            *self_ref.borrow_mut() = Some(Rc::clone(&f));
            f
        };

        populate_manage();

        // Add Connection bar at the bottom of Manage Connections
        content.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        let add_hdr = gtk4::Label::new(Some("Add Connection"));
        add_hdr.set_halign(gtk4::Align::Start);
        add_hdr.add_css_class("heading");
        content.append(&add_hdr);

        let add_row = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        let add_label = gtk4::Entry::builder()
            .placeholder_text("Label")
            .width_chars(12)
            .build();
        let add_target = gtk4::Entry::builder()
            .placeholder_text("user@host")
            .hexpand(true)
            .build();
        let add_port = gtk4::Entry::builder()
            .placeholder_text("Port")
            .text("7443")
            .width_chars(5)
            .build();
        let add_bin = gtk4::Entry::builder()
            .placeholder_text("server_bin")
            .width_chars(12)
            .build();
        let add_commit_btn = gtk4::Button::with_label("Add");

        add_commit_btn.connect_clicked({
            let state = Rc::clone(&state);
            let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
            let populate_manage = Rc::clone(&populate_manage);
            let add_label = add_label.clone();
            let add_target = add_target.clone();
            let add_port = add_port.clone();
            let add_bin = add_bin.clone();
            move |_| {
                let target = add_target.text().trim().to_string();
                if target.is_empty() {
                    return;
                }
                let label_raw = add_label.text().trim().to_string();
                let label = if label_raw.is_empty() {
                    target.clone()
                } else {
                    label_raw
                };
                let port = add_port.text().trim().parse::<u16>().unwrap_or(7443);
                let bin_raw = add_bin.text().trim().to_string();
                let server_bin = if bin_raw.is_empty() {
                    None
                } else {
                    Some(bin_raw)
                };
                let profile = HostProfile {
                    label,
                    ssh_target: target.clone(),
                    server_bin,
                    port,
                    audio_enabled: true,
                };
                {
                    let mut st = state.borrow_mut();
                    st.hosts.retain(|h| h.ssh_target != target);
                    st.hosts.push(profile);
                    let _ = save_hosts(&st.hosts);
                }
                add_label.set_text("");
                add_target.set_text("");
                add_bin.set_text("");
                rebuild_sidebar();
                populate_manage();
            }
        });

        add_row.append(&add_label);
        add_row.append(&add_target);
        add_row.append(&add_port);
        add_row.append(&add_bin);
        add_row.append(&add_commit_btn);
        content.append(&add_row);

        dlg.set_child(Some(&content));
        dlg.present();
    }

    fn load_pixbuf(data: &[u8]) -> Option<gtk4::gdk_pixbuf::Pixbuf> {
        let loader = PixbufLoader::new();
        loader.write(data).ok()?;
        loader.close().ok()?;
        loader.pixbuf()
    }
}
