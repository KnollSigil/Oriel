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
    use crate::pty::{pty_tunnel_socket_path, spawn_local_terminal_pty_socket};
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
        let pty_sock = pty_tunnel_socket_path(&target.ssh_target)?;

        let (ui_tx, ui_rx) = window::create_ui_channel();
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let (net_tx, mut net_rx) = mpsc::unbounded_channel::<LauncherNetEvent>();

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

#[cfg(feature = "gtk4")]
mod gtk4_ui {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::process::Child;
    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use anyhow::Result;
    use gtk4::gdk_pixbuf::PixbufLoader;
    use gtk4::gio;
    use gtk4::glib;
    use gtk4::prelude::*;
    use remotewayland_proto::{ControlMessage, LaunchState, RemoteAppEntry, PTY_LAUNCH_ID_BASE};
    use tokio::sync::mpsc;

    use crate::hosts::{load_hosts, parse_uri, save_hosts, ssh_bootstrap, HostProfile};
    use crate::pty::{
        available_local_terminals, pty_tunnel_socket_path, spawn_local_terminal_pty_socket,
        spawn_local_terminal_pty_socket_with_terminal, terminal_basename,
    };
    use crate::quic::{self, LauncherNetEvent};
    use crate::search::{self, MatchTier, SearchIndexEntry};
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
        pty_socket: Option<PathBuf>,
        ctrl_tx: Option<mpsc::UnboundedSender<ControlMessage>>,
        ui_tx: Option<window::UiSender>,
        net_rx: Option<mpsc::UnboundedReceiver<LauncherNetEvent>>,
        connect_rx: Option<std::sync::mpsc::Receiver<Result<ConnectResult, String>>>,
        apps: Vec<RemoteAppEntry>,
        programs: Vec<TrackedProgram>,
        rtt_ms: Option<u32>,
        congested: bool,
        auto_reconnect: bool,
        reconnect_at: Option<Instant>,
        reconnect_backoff_secs: u64,
    }

    struct ConnectResult {
        pty_socket: PathBuf,
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
        grid_search: Vec<SearchIndexEntry>,
        grid_tiers: Vec<Option<MatchTier>>,
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
                background-color: alpha(@theme_selected_bg_color, 0.18);
                border: 2px solid alpha(@theme_selected_bg_color, 0.55);
                color: inherit;
            }
            flowboxchild:focus {
                background-color: alpha(@theme_selected_bg_color, 0.30);
                border: 2px solid @theme_selected_bg_color;
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
                background-color: alpha(@theme_selected_bg_color, 0.20);
                color: inherit;
            }
            .server-list row:selected label {
                color: inherit;
            }
            .server-list row:focus {
                outline: 2px solid @theme_selected_bg_color;
                outline-offset: -2px;
            }
            searchentry:focus-within {
                outline: 2px solid alpha(@theme_selected_bg_color, 0.80);
                outline-offset: 1px;
            }
            .compact-btn {
                min-height: 18px;
                min-width: 18px;
                padding: 1px 4px;
            }
            .rw-split-btn > button {
                border-top-right-radius: 0;
                border-bottom-right-radius: 0;
                border-right: none;
                margin-right: 0;
            }
            .rw-split-btn > menubutton > button {
                border-top-left-radius: 0;
                border-bottom-left-radius: 0;
                border-left: 1px solid alpha(currentColor, 0.18);
                margin-left: 0;
                padding-left: 5px;
                padding-right: 5px;
                min-width: 22px;
            }
            /* Breeze GTK4 doesn't highlight keyboard-focused PopoverMenu items with :focus-visible,
               so target :focus/:selected explicitly on our terminal selector popover. */
            popover.rw-term-popover modelbutton:focus,
            popover.rw-term-popover modelbutton:selected {
                background-color: alpha(@theme_selected_bg_color, 0.25);
                outline: 2px solid @theme_selected_bg_color;
                outline-offset: -2px;
            }
            /* Match terminal popover highlight inside Settings and use inset outline
               so focus rings aren't clipped by the ScrolledWindow viewport. */
            popover.rw-settings-popover checkbutton {
                padding: 3px 6px;
                border-radius: 5px;
            }
            popover.rw-settings-popover checkbutton:focus {
                background-color: alpha(@theme_selected_bg_color, 0.22);
                outline: 2px solid @theme_selected_bg_color;
                outline-offset: -2px;
            }
            popover.rw-settings-popover button:focus-visible {
                outline: 2px solid @theme_selected_bg_color;
                outline-offset: -2px;
                border-radius: 4px;
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

    #[derive(Debug, Default, PartialEq, Eq)]
    struct GtkSettingsIni {
        prefer_dark: Option<bool>,
        theme_name: Option<String>,
        icon_theme_name: Option<String>,
    }

    fn parse_gtk_settings_ini(content: &str) -> GtkSettingsIni {
        let mut parsed = GtkSettingsIni::default();
        let mut in_settings = false;
        for raw_line in content.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                in_settings = line.eq_ignore_ascii_case("[Settings]");
                continue;
            }
            if !in_settings {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let val = val.trim();
            match key {
                "gtk-application-prefer-dark-theme" => {
                    if val.eq_ignore_ascii_case("true") || val == "1" {
                        parsed.prefer_dark = Some(true);
                    } else if val.eq_ignore_ascii_case("false") || val == "0" {
                        parsed.prefer_dark = Some(false);
                    }
                }
                "gtk-theme-name" if !val.is_empty() => {
                    parsed.theme_name = Some(val.to_string());
                }
                "gtk-icon-theme-name" if !val.is_empty() => {
                    parsed.icon_theme_name = Some(val.to_string());
                }
                _ => {}
            }
        }
        parsed
    }

    fn is_gtk4_theme_file(path: &Path) -> bool {
        matches!(
            path.file_name().and_then(|n| n.to_str()),
            Some("colors.css" | "gtk.css" | "settings.ini")
        )
    }

    fn gtk4_config_dir() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(xdg).join("gtk-4.0"));
        }
        std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .map(|h| PathBuf::from(h).join(".config").join("gtk-4.0"))
    }

    /// Watch `$XDG_CONFIG_HOME/gtk-4.0` (`colors.css`, `gtk.css`, `settings.ini`) and
    /// live-reload GTK4 user CSS + settings when the desktop color scheme changes.
    ///
    /// KDE Plasma's `kde-gtk-config` updates `gtk-4.0/colors.css` and `settings.ini`
    /// atomically on every theme/palette change, but GTK4 only loads `gtk.css` once at
    /// startup (and its portal `color-scheme` signal can fire before `colors.css` is
    /// rewritten, causing stale-theme flashes or missed same-polarity theme switches).
    fn install_live_theme_reload(window: &gtk4::ApplicationWindow) {
        let Some(gtk4_dir) = gtk4_config_dir() else {
            return;
        };
        let Some(display) = gtk4::gdk::Display::default() else {
            return;
        };

        let user_css_provider = gtk4::CssProvider::new();
        // Registered after GTK4's startup user provider at the same priority so reloaded
        // `@define-color` rules from `gtk.css` / `colors.css` take precedence immediately.
        gtk4::style_context_add_provider_for_display(
            &display,
            &user_css_provider,
            gtk4::STYLE_PROVIDER_PRIORITY_USER,
        );

        let gtk_css_path = gtk4_dir.join("gtk.css");
        let settings_ini_path = gtk4_dir.join("settings.ini");
        let reloading = Rc::new(Cell::new(false));

        let reload_now: Rc<dyn Fn(bool)> = Rc::new({
            let user_css_provider = user_css_provider.clone();
            let gtk_css_path = gtk_css_path.clone();
            let settings_ini_path = settings_ini_path.clone();
            let reloading = Rc::clone(&reloading);
            move |sync_settings: bool| {
                reloading.set(true);
                if sync_settings {
                    if let Ok(content) = std::fs::read_to_string(&settings_ini_path) {
                        let parsed = parse_gtk_settings_ini(&content);
                        if let Some(settings) = gtk4::Settings::default() {
                            if let Some(dark) = parsed.prefer_dark {
                                if settings.is_gtk_application_prefer_dark_theme() != dark {
                                    settings.set_gtk_application_prefer_dark_theme(dark);
                                }
                            }
                            if let Some(theme) = parsed.theme_name {
                                if settings.gtk_theme_name().as_deref() != Some(theme.as_str()) {
                                    settings.set_gtk_theme_name(Some(&theme));
                                }
                            }
                            if let Some(icon_theme) = parsed.icon_theme_name {
                                if settings.gtk_icon_theme_name().as_deref()
                                    != Some(icon_theme.as_str())
                                {
                                    settings.set_gtk_icon_theme_name(Some(&icon_theme));
                                }
                            }
                        }
                    }
                }

                if gtk_css_path.is_file() {
                    user_css_provider.load_from_path(&gtk_css_path);
                } else {
                    user_css_provider.load_from_string("");
                }
                reloading.set(false);
            }
        });

        // Debounce bursts (e.g. `kde-gtk-config` writing `colors.css` + `settings.ini`
        // right after `xdg-desktop-portal-kde` emits `SettingChanged`).
        let pending_reload: Rc<Cell<Option<glib::SourceId>>> = Rc::new(Cell::new(None));
        let pending_sync_settings: Rc<Cell<bool>> = Rc::new(Cell::new(false));
        let schedule_reload: Rc<dyn Fn(bool)> = Rc::new({
            let pending_reload = Rc::clone(&pending_reload);
            let pending_sync_settings = Rc::clone(&pending_sync_settings);
            let reload_now = Rc::clone(&reload_now);
            move |sync_settings: bool| {
                if sync_settings {
                    pending_sync_settings.set(true);
                }
                if let Some(id) = pending_reload.take() {
                    id.remove();
                }
                let pending_inner = Rc::clone(&pending_reload);
                let sync_inner = Rc::clone(&pending_sync_settings);
                let reload_inner = Rc::clone(&reload_now);
                let id = glib::timeout_add_local_once(Duration::from_millis(80), move || {
                    pending_inner.set(None);
                    let do_sync = sync_inner.replace(false);
                    reload_inner(do_sync);
                });
                pending_reload.set(Some(id));
            }
        });

        let dir_file = gio::File::for_path(&gtk4_dir);
        let monitor = match dir_file
            .monitor_directory(gio::FileMonitorFlags::WATCH_MOVES, gio::Cancellable::NONE)
        {
            Ok(mon) => {
                let schedule = Rc::clone(&schedule_reload);
                mon.connect_changed(move |_mon, file, other_file, _event| {
                    let paths = [file.path(), other_file.and_then(|f| f.path())];
                    let mut relevant = false;
                    let mut touched_settings = false;
                    for path in paths.into_iter().flatten() {
                        if is_gtk4_theme_file(&path) {
                            relevant = true;
                            if path.file_name().and_then(|n| n.to_str()) == Some("settings.ini") {
                                touched_settings = true;
                            }
                        }
                    }
                    if relevant {
                        schedule(touched_settings);
                    }
                });
                Some(mon)
            }
            Err(err) => {
                tracing::debug!(
                    dir = %gtk4_dir.display(),
                    error = %err,
                    "unable to watch gtk-4.0 config directory for live theme reload"
                );
                None
            }
        };

        // Also re-sync CSS (without overwriting `prefer_dark` from a potentially stale
        // `settings.ini`) shortly after GTK4's own portal listener flips dark-mode so
        // light<->dark transitions never settle on a stale `colors.css`.
        let dark_notify = gtk4::Settings::default().map(|settings| {
            let schedule = Rc::clone(&schedule_reload);
            let reloading = Rc::clone(&reloading);
            let id = settings.connect_gtk_application_prefer_dark_theme_notify(move |_| {
                if !reloading.get() {
                    schedule(false);
                }
            });
            (settings, Cell::new(Some(id)))
        });

        window.connect_destroy(move |_| {
            if let Some(id) = pending_reload.take() {
                id.remove();
            }
            if let Some(ref mon) = monitor {
                mon.cancel();
            }
            if let Some((ref settings, ref id_cell)) = dark_notify {
                if let Some(id) = id_cell.take() {
                    settings.disconnect(id);
                }
            }
            gtk4::style_context_remove_provider_for_display(&display, &user_css_provider);
        });
    }

    fn build_ui(app: &gtk4::Application) {
        install_css();

        let window = gtk4::ApplicationWindow::builder()
            .application(app)
            .title("RemoteWayland")
            .default_width(960)
            .default_height(640)
            .build();
        install_live_theme_reload(&window);

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
            grid_search: Vec::new(),
            grid_tiers: Vec::new(),
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
        // Split "New Terminal" button (AdwSplitButton-style): primary click uses the default
        // terminal emulator; narrow arrow half opens a native PopoverMenu to pick any installed emulator.
        let term_split = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        term_split.add_css_class("linked");
        term_split.add_css_class("rw-split-btn");
        term_split.set_sensitive(false);

        let term_btn = gtk4::Button::with_label("New Terminal");
        term_btn.set_tooltip_text(Some("Open remote terminal in the default terminal emulator"));

        let term_menu = gio::Menu::new();
        let term_popover = gtk4::PopoverMenu::from_model(Some(&term_menu));
        term_popover.add_css_class("rw-term-popover");

        let term_menu_btn = gtk4::MenuButton::builder()
            .icon_name("pan-down-symbolic")
            .tooltip_text("Choose terminal emulator for remote PTY")
            .popover(&term_popover)
            .build();

        term_split.append(&term_btn);
        term_split.append(&term_menu_btn);

        // Host microphone selector + mute control inside the Settings popover
        // (live-switches PipeWire source). Uses an inline radio list (`CheckButton` group)
        // instead of a nested combobox/dropdown popup so GTK4 Wayland autohide grabs are
        // never disrupted and switching microphones takes a single click or keypress.
        let mic_list_box = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        let updating_mics = Rc::new(std::cell::Cell::new(false));
        let active_mic_btn: Rc<RefCell<Option<gtk4::CheckButton>>> = Rc::new(RefCell::new(None));

        let populate_mics: Rc<dyn Fn()> = Rc::new({
            let mic_list_box = mic_list_box.clone();
            let updating_mics = Rc::clone(&updating_mics);
            let active_mic_btn = Rc::clone(&active_mic_btn);
            move || {
                updating_mics.set(true);
                while let Some(child) = mic_list_box.first_child() {
                    mic_list_box.remove(&child);
                }
                *active_mic_btn.borrow_mut() = None;

                let cfg = crate::audio::subscribe_mic_config().borrow().clone();
                let list = crate::audio::list_host_mic_sources();

                let mut entries: Vec<(Option<String>, bool, String)> =
                    Vec::with_capacity(list.len() + 2);
                // (target_node_name, is_muted, display_label)
                entries.push((None, true, "Off (Muted)".to_string()));
                entries.push((None, false, "System Default".to_string()));
                for src in list {
                    entries.push((Some(src.node_name), false, src.description));
                }

                let mut first_btn: Option<gtk4::CheckButton> = None;
                let mut default_btn: Option<gtk4::CheckButton> = None;
                let mut selected_btn: Option<gtk4::CheckButton> = None;

                for (target, is_muted, label) in entries {
                    let is_selected = if is_muted {
                        cfg.muted
                    } else if target.is_none() {
                        !cfg.muted && cfg.target.is_none()
                    } else {
                        !cfg.muted && cfg.target == target
                    };

                    let btn = gtk4::CheckButton::with_label(&label);
                    if let Some(ref group) = first_btn {
                        btn.set_group(Some(group));
                    } else {
                        first_btn = Some(btn.clone());
                    }
                    if !is_muted && target.is_none() {
                        default_btn = Some(btn.clone());
                    }
                    if is_selected {
                        btn.set_active(true);
                        selected_btn = Some(btn.clone());
                    }

                    let updating = Rc::clone(&updating_mics);
                    let active_mic_btn_c = Rc::clone(&active_mic_btn);
                    btn.connect_toggled(move |b| {
                        if updating.get() || !b.is_active() {
                            return;
                        }
                        *active_mic_btn_c.borrow_mut() = Some(b.clone());
                        if is_muted {
                            crate::audio::set_mic_muted(true);
                        } else {
                            // Set target before unmuting so we never briefly unmute the previous mic.
                            crate::audio::set_mic_target(target.clone());
                            crate::audio::set_mic_muted(false);
                        }
                    });
                    mic_list_box.append(&btn);
                }

                // If the configured target disappeared, fall back to System Default and
                // clear the stale target so audio::MicConfig stays in sync with the UI.
                if selected_btn.is_none() {
                    if let Some(ref fallback) = default_btn {
                        fallback.set_active(true);
                        selected_btn = Some(fallback.clone());
                        if !cfg.muted {
                            crate::audio::set_mic_target(None);
                        }
                    } else {
                        selected_btn = first_btn.clone();
                    }
                }
                *active_mic_btn.borrow_mut() = selected_btn;
                updating_mics.set(false);
            }
        });

        let mic_refresh_btn = gtk4::Button::with_label("↻");
        mic_refresh_btn.add_css_class("compact-btn");
        mic_refresh_btn.set_tooltip_text(Some("Refresh host microphones"));
        mic_refresh_btn.connect_clicked({
            let populate_mics = Rc::clone(&populate_mics);
            let active_mic_btn = Rc::clone(&active_mic_btn);
            move |_| {
                populate_mics();
                if let Some(btn) = active_mic_btn.borrow().as_ref() {
                    btn.grab_focus();
                }
            }
        });

        let settings_popover = gtk4::Popover::new();
        settings_popover.add_css_class("rw-settings-popover");
        let settings_box = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        settings_box.set_margin_top(10);
        settings_box.set_margin_bottom(10);
        settings_box.set_margin_start(10);
        settings_box.set_margin_end(10);
        settings_box.set_width_request(300);

        let settings_title = gtk4::Label::new(None);
        settings_title.set_markup("<b>Settings</b>");
        settings_title.set_halign(gtk4::Align::Start);

        let mic_header = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        let mic_lbl = gtk4::Label::new(Some("Microphone"));
        mic_lbl.set_halign(gtk4::Align::Start);
        mic_lbl.set_hexpand(true);
        mic_lbl.add_css_class("dim-label");
        mic_header.append(&mic_lbl);
        mic_header.append(&mic_refresh_btn);

        let mic_scroll = gtk4::ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .propagate_natural_height(true)
            .max_content_height(300)
            .child(&mic_list_box)
            .build();

        settings_box.append(&settings_title);
        settings_box.append(&mic_header);
        settings_box.append(&mic_scroll);
        settings_popover.set_child(Some(&settings_box));
        settings_popover.connect_show({
            let populate_mics = Rc::clone(&populate_mics);
            move |_| {
                populate_mics();
            }
        });
        settings_popover.connect_map({
            let active_mic_btn = Rc::clone(&active_mic_btn);
            move |pop| {
                let pop = pop.clone();
                let active_mic_btn = Rc::clone(&active_mic_btn);
                glib::idle_add_local_once(move || {
                    if pop.is_visible() {
                        if let Some(btn) = active_mic_btn.borrow().as_ref() {
                            btn.grab_focus();
                        }
                    }
                });
            }
        });

        let settings_btn = gtk4::MenuButton::builder()
            .icon_name("emblem-system-symbolic")
            .tooltip_text("Settings")
            .popover(&settings_popover)
            .build();

        top_bar.append(&search_entry);
        top_bar.append(&term_split);
        top_bar.append(&settings_btn);
        right_pane.append(&top_bar);

        // App icon grid (filtered + ranked in-place via set_filter_func / set_sort_func)
        let flow = gtk4::FlowBox::builder()
            .valign(gtk4::Align::Start)
            .max_children_per_line(8)
            .min_children_per_line(3)
            .selection_mode(gtk4::SelectionMode::Single)
            .activate_on_single_click(true)
            .row_spacing(8)
            .column_spacing(8)
            .build();

        /// Recover the original `grid_apps`/`grid_search` index from a `FlowBoxChild`.
        /// `child.index()` reflects sorted visual order once `set_sort_func` runs, so we
        /// tag each child's `widget_name` with its stable source index in `rebuild_grid`.
        fn flow_child_app_idx(child: &gtk4::FlowBoxChild) -> Option<usize> {
            child.widget_name().parse::<usize>().ok()
        }
        fn recompute_grid_tiers(st: &mut AppUi, query: &str) {
            st.grid_tiers.clear();
            if query.is_empty() {
                return;
            }
            st.grid_tiers.extend(
                st.grid_search
                    .iter()
                    .map(|entry| search::match_tier(entry, query)),
            );
        }
        fn child_matches_query(st: &AppUi, child: &gtk4::FlowBoxChild, query: &str) -> bool {
            if query.is_empty() {
                return true;
            }
            flow_child_app_idx(child)
                .and_then(|idx| st.grid_tiers.get(idx).copied().flatten())
                .is_some()
        }
        fn first_matching_child(
            flow: &gtk4::FlowBox,
            st: &AppUi,
            query: &str,
        ) -> Option<gtk4::FlowBoxChild> {
            let mut cur = flow.first_child();
            while let Some(widget) = cur {
                if let Ok(child) = widget.clone().downcast::<gtk4::FlowBoxChild>() {
                    if child_matches_query(st, &child, query) {
                        return Some(child);
                    }
                }
                cur = widget.next_sibling();
            }
            None
        }

        flow.set_filter_func({
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            move |child| {
                let needle = current_filter.borrow();
                child_matches_query(&state.borrow(), child, needle.as_str())
            }
        });
        flow.set_sort_func({
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            move |a, b| {
                let st = state.borrow();
                let needle = current_filter.borrow();
                let ia = flow_child_app_idx(a);
                let ib = flow_child_app_idx(b);
                let ea = ia.and_then(|i| st.grid_search.get(i));
                let eb = ib.and_then(|i| st.grid_search.get(i));
                let ord = match (ea, eb) {
                    (Some(ea), Some(eb)) if !needle.is_empty() => {
                        let ta = ia
                            .and_then(|i| st.grid_tiers.get(i).copied().flatten())
                            .map(|t| t as u8)
                            .unwrap_or(u8::MAX);
                        let tb = ib
                            .and_then(|i| st.grid_tiers.get(i).copied().flatten())
                            .map(|t| t as u8)
                            .unwrap_or(u8::MAX);
                        (ta, ea.name_lower.as_str()).cmp(&(tb, eb.name_lower.as_str()))
                    }
                    (Some(ea), Some(eb)) => ea.name_lower.cmp(&eb.name_lower),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                ord.into()
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
            let term_split = term_split.clone();
            let term_popover = term_popover.clone();

            Rc::new(move || {
                let st = state.borrow();
                let Some(target) = st.active_target.as_ref() else {
                    status_dot.set_markup("<span foreground='#888888'>●</span>");
                    status_net_lbl.set_text("No server selected");
                    term_popover.popdown();
                    term_split.set_sensitive(false);
                    return;
                };

                let Some(sess) = st.sessions.get(target) else {
                    status_dot.set_markup("<span foreground='#888888'>●</span>");
                    status_net_lbl.set_text(&format!("{target} · Disconnected"));
                    term_popover.popdown();
                    term_split.set_sensitive(false);
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
                        term_popover.popdown();
                        term_split.set_sensitive(false);
                    }
                    ConnStatus::Connecting => {
                        status_dot.set_markup("<span foreground='#ddaa22'>●</span>");
                        status_net_lbl.set_text(&format!("{target} · Connecting via SSH…"));
                        term_popover.popdown();
                        term_split.set_sensitive(false);
                    }
                    ConnStatus::Connected => {
                        term_split.set_sensitive(true);
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

                    if (prog.launch_id & PTY_LAUNCH_ID_BASE) != 0
                        && prog.name.ends_with("(detached)")
                    {
                        if let Some(sock_path) = sess.pty_socket.clone() {
                            let pty_id = prog.launch_id & !PTY_LAUNCH_ID_BASE;
                            let reattach_btn = gtk4::Button::with_label("↩");
                            reattach_btn.set_tooltip_text(Some("Reattach detached terminal"));
                            reattach_btn.connect_clicked(move |_| {
                                let _ = spawn_local_terminal_pty_socket(
                                    &sock_path,
                                    Some(pty_id),
                                    None,
                                );
                            });
                            row.append(&reattach_btn);
                        }
                    }

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
                let st = state.borrow();
                if let Some(child) = first_matching_child(&flow, &st, needle.as_str()) {
                    flow.select_child(&child);
                } else {
                    flow.unselect_all();
                }
            })
        };

        let rebuild_grid: Rc<dyn Fn()> = {
            let flow = flow.clone();
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
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
                    st.grid_search = apps.iter().map(SearchIndexEntry::from_app).collect();
                    st.grid_apps = apps.clone();
                    let needle = current_filter.borrow();
                    recompute_grid_tiers(&mut st, needle.as_str());
                    apps
                };

                for (idx, entry) in apps.iter().enumerate() {
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
                    child.set_widget_name(&idx.to_string());
                    child.set_tooltip_text(Some(&entry.name));
                    child.set_child(Some(&vbox));
                    flow.insert(&child, -1);
                }

                flow.invalidate_filter();
                flow.invalidate_sort();
                select_first_visible();
            })
        };

        let start_connect_inner: Rc<dyn Fn(HostProfile, bool)> = {
            let state = Rc::clone(&state);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);
            Rc::new(move |profile: HostProfile, is_auto_reconnect: bool| {
                let target = profile.ssh_target.clone();
                {
                    let mut st = state.borrow_mut();
                    if !is_auto_reconnect {
                        st.active_target = Some(target.clone());
                    }
                    let (prev_backoff, prev_auto) = if let Some(existing) = st.sessions.get(&target)
                    {
                        if existing.status != ConnStatus::Disconnected {
                            return;
                        }
                        (existing.reconnect_backoff_secs, existing.auto_reconnect)
                    } else {
                        (2, false)
                    };
                    let (tx, rx) = std::sync::mpsc::channel::<Result<ConnectResult, String>>();
                    st.sessions.insert(
                        target.clone(),
                        HostSession {
                            status: ConnStatus::Connecting,
                            error: if is_auto_reconnect {
                                Some("Reconnecting…".to_string())
                            } else {
                                None
                            },
                            pty_socket: None,
                            ctrl_tx: None,
                            ui_tx: None,
                            net_rx: None,
                            connect_rx: Some(rx),
                            apps: Vec::new(),
                            programs: Vec::new(),
                            rtt_ms: None,
                            congested: false,
                            auto_reconnect: if is_auto_reconnect { prev_auto } else { false },
                            reconnect_at: None,
                            reconnect_backoff_secs: if is_auto_reconnect { prev_backoff } else { 2 },
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
                            let pty_socket = pty_tunnel_socket_path(&profile.ssh_target)?;
                            let (ui_tx, ui_rx) = window::create_ui_channel();
                            let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
                            let (net_tx, net_rx) = mpsc::unbounded_channel();

                            let ui_tx_net = ui_tx.clone();
                            let pty_sock_net = pty_socket.clone();
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
                                        Some(pty_sock_net),
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
                                pty_socket,
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

        let start_connect: Rc<dyn Fn(HostProfile)> = {
            let inner = Rc::clone(&start_connect_inner);
            Rc::new(move |profile: HostProfile| inner(profile, false))
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
                    let dc_enabled = {
                        let st = state_cb.borrow();
                        st.sessions
                            .get(&profile.ssh_target)
                            .is_some_and(|s| s.status != ConnStatus::Disconnected || s.auto_reconnect)
                    };
                    dc_btn.set_sensitive(dc_enabled);
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
                                sess.auto_reconnect = false;
                                sess.reconnect_at = None;
                                sess.reconnect_backoff_secs = 2;
                                sess.error = None;
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

        // Instant in-place ranked search filter + KDE-style Enter to launch & reset
        let apply_filter: Rc<dyn Fn(String)> = Rc::new({
            let state = Rc::clone(&state);
            let current_filter = Rc::clone(&current_filter);
            let flow = flow.clone();
            let select_first_visible = Rc::clone(&select_first_visible);
            move |query: String| {
                *current_filter.borrow_mut() = query.clone();
                recompute_grid_tiers(&mut state.borrow_mut(), &query);
                flow.invalidate_filter();
                flow.invalidate_sort();
                select_first_visible();
            }
        });

        search_entry.connect_search_changed({
            let apply_filter = Rc::clone(&apply_filter);
            move |e| {
                apply_filter(e.text().trim().to_lowercase());
            }
        });

        search_entry.connect_activate({
            let state = Rc::clone(&state);
            let flow = flow.clone();
            let current_filter = Rc::clone(&current_filter);
            let apply_filter = Rc::clone(&apply_filter);
            let launch_exec_or_app = Rc::clone(&launch_exec_or_app);
            move |e| {
                let typed = e.text().trim().to_string();
                if let Some(cmd) = search::explicit_command(&typed) {
                    if !cmd.is_empty() {
                        launch_exec_or_app(cmd.to_string(), cmd.to_string());
                    }
                    return;
                }
                // Sync immediately if Enter was pressed before GtkSearchEntry's 150ms debounce fired.
                let normalized = typed.to_lowercase();
                if *current_filter.borrow() != normalized {
                    apply_filter(normalized);
                }
                let selected_child = flow.selected_children().into_iter().next();
                let picked = {
                    let st = state.borrow();
                    let needle = current_filter.borrow();
                    selected_child
                        .as_ref()
                        .filter(|c| child_matches_query(&st, c, needle.as_str()))
                        .and_then(flow_child_app_idx)
                        .and_then(|idx| st.grid_apps.get(idx))
                        .or_else(|| {
                            first_matching_child(&flow, &st, needle.as_str())
                                .and_then(|c| flow_child_app_idx(&c))
                                .and_then(|idx| st.grid_apps.get(idx))
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
                    let target = {
                        let needle = current_filter.borrow();
                        let st = state.borrow();
                        flow.selected_children()
                            .into_iter()
                            .next()
                            .filter(|c| child_matches_query(&st, c, needle.as_str()))
                            .or_else(|| first_matching_child(&flow, &st, needle.as_str()))
                    };
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
                    let st = state.borrow();
                    let mut visible: Vec<(gtk4::FlowBoxChild, f32, f32)> = Vec::new();
                    let mut cur = flow.first_child();
                    while let Some(widget) = cur {
                        if let Ok(ch) = widget.clone().downcast::<gtk4::FlowBoxChild>() {
                            if child_matches_query(&st, &ch, needle.as_str()) {
                                if let Some(b) = ch.compute_bounds(&flow) {
                                    visible.push((ch, b.x(), b.y()));
                                }
                            }
                        }
                        cur = widget.next_sibling();
                    }
                    drop(st);
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
                let picked = flow_child_app_idx(child)
                    .and_then(|idx| state.borrow().grid_apps.get(idx).cloned());
                if let Some(app) = picked {
                    launch_exec_or_app(app.name, app.exec);
                }
            }
        });

        // New Terminal split button (primary = default emulator, dropdown = pick emulator)
        let launch_terminal_with: Rc<dyn Fn(Option<&str>)> = Rc::new({
            let state = Rc::clone(&state);
            let refresh_status_bar = Rc::clone(&refresh_status_bar);
            let search_entry = search_entry.clone();
            move |terminal: Option<&str>| {
                let sock_path = state
                    .borrow()
                    .active_session()
                    .and_then(|sess| sess.pty_socket.clone());
                if let Some(sock_path) = sock_path {
                    // Server PtyManager announces this persistent PTY via AppState.
                    if let Err(err) = spawn_local_terminal_pty_socket_with_terminal(
                        &sock_path,
                        None,
                        None,
                        terminal,
                    ) {
                        tracing::warn!(
                            error = %err,
                            ?terminal,
                            "failed to launch local terminal for remote PTY"
                        );
                    }
                }
                search_entry.grab_focus();
                refresh_status_bar();
            }
        });

        term_btn.connect_clicked({
            let launch_terminal_with = Rc::clone(&launch_terminal_with);
            move |_| {
                launch_terminal_with(None);
            }
        });

        let launch_term_action =
            gio::SimpleAction::new("launch-terminal", Some(glib::VariantTy::STRING));
        launch_term_action.connect_activate({
            let launch_terminal_with = Rc::clone(&launch_terminal_with);
            move |_, param| {
                if let Some(term) = param.and_then(|v| v.get::<String>()) {
                    launch_terminal_with(Some(&term));
                }
            }
        });
        window.add_action(&launch_term_action);

        let last_term_list: Rc<RefCell<Option<Vec<String>>>> = Rc::new(RefCell::new(None));
        let populate_term_menu: Rc<dyn Fn()> = Rc::new({
            let term_menu = term_menu.clone();
            let last_term_list = Rc::clone(&last_term_list);
            move || {
                let terminals = available_local_terminals();
                if last_term_list.borrow().as_deref() == Some(terminals.as_slice()) {
                    return;
                }
                *last_term_list.borrow_mut() = Some(terminals.clone());
                term_menu.remove_all();
                if terminals.is_empty() {
                    let item = gio::MenuItem::new(Some("No terminal emulators found"), None);
                    term_menu.append_item(&item);
                    return;
                }
                for (idx, term) in terminals.into_iter().enumerate() {
                    let display_name = terminal_basename(&term);
                    let raw_label = if idx == 0 {
                        format!("{display_name} (default)")
                    } else {
                        display_name.to_string()
                    };
                    // Escape underscores so GMenu doesn't interpret them as mnemonic prefixes.
                    let label_text = raw_label.replace('_', "__");
                    let item = gio::MenuItem::new(Some(&label_text), None);
                    item.set_action_and_target_value(
                        Some("win.launch-terminal"),
                        Some(&term.to_variant()),
                    );
                    term_menu.append_item(&item);
                }
            }
        });
        // Populate before first popup so PopoverMenu widgets already exist when
        // keyboard activation grabs focus, and refresh before each subsequent popup.
        populate_term_menu();
        term_menu_btn.set_create_popup_func({
            let populate_term_menu = Rc::clone(&populate_term_menu);
            move |_| {
                populate_term_menu();
            }
        });
        term_popover.connect_map(|pop| {
            let pop = pop.clone();
            glib::idle_add_local_once(move || {
                if pop.is_visible() {
                    pop.child_focus(gtk4::DirectionType::TabForward);
                }
            });
        });

        // Clean Tab / Shift+Tab focus ring:
        // manage_btn <-> server_list <-> (search_entry + flow) <-> term_btn <-> term_menu_btn <-> settings_btn <-> status_bar_btn
        let win_key_ctrl = gtk4::EventControllerKey::new();
        win_key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        win_key_ctrl.connect_key_pressed({
            let window_ref = window.clone();
            let manage_btn = manage_btn.clone();
            let server_list = server_list.clone();
            let search_entry = search_entry.clone();
            let flow = flow.clone();
            let term_split = term_split.clone();
            let term_btn = term_btn.clone();
            let term_menu_btn = term_menu_btn.clone();
            let term_popover = term_popover.clone();
            let settings_btn = settings_btn.clone();
            let settings_popover = settings_popover.clone();
            let programs_popover = programs_popover.clone();
            let status_bar_btn = status_bar_btn.clone();
            move |_, key, _, mods| {
                let is_shift_tab = key == gtk4::gdk::Key::ISO_Left_Tab
                    || (key == gtk4::gdk::Key::Tab
                        && mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK));
                let is_tab = key == gtk4::gdk::Key::Tab
                    && !mods.contains(gtk4::gdk::ModifierType::SHIFT_MASK);
                if (!is_tab && !is_shift_tab)
                    || settings_popover.is_visible()
                    || term_popover.is_visible()
                    || programs_popover.is_visible()
                {
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
                let in_term_btn = focused == term_btn || focused.is_ancestor(&term_btn);
                let in_term_menu =
                    focused == term_menu_btn || focused.is_ancestor(&term_menu_btn);
                let in_settings = focused == settings_btn || focused.is_ancestor(&settings_btn);
                let in_status = focused == status_bar_btn || focused.is_ancestor(&status_bar_btn);

                if is_tab {
                    if in_manage {
                        focus_servers();
                    } else if in_servers {
                        search_entry.grab_focus();
                    } else if in_launcher {
                        if term_split.is_sensitive() {
                            term_btn.grab_focus();
                        } else {
                            settings_btn.grab_focus();
                        }
                    } else if in_term_btn {
                        term_menu_btn.grab_focus();
                    } else if in_term_menu {
                        settings_btn.grab_focus();
                    } else if in_settings {
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
                } else if in_term_btn {
                    search_entry.grab_focus();
                } else if in_term_menu {
                    term_btn.grab_focus();
                } else if in_settings {
                    if term_split.is_sensitive() {
                        term_menu_btn.grab_focus();
                    } else {
                        search_entry.grab_focus();
                    }
                } else if in_status {
                    settings_btn.grab_focus();
                } else {
                    return glib::Propagation::Proceed;
                }
                glib::Propagation::Stop
            }
        });
        window.add_controller(win_key_ctrl);

        // Auto-focus search box on Alt-Tab if active server is connected, else focus server list.
        // Skip when any popover is open so opening an xdg_popup doesn't steal focus back to search.
        window.connect_is_active_notify({
            let state = Rc::clone(&state);
            let search_entry = search_entry.clone();
            let server_list = server_list.clone();
            let term_popover = term_popover.clone();
            let settings_popover = settings_popover.clone();
            let programs_popover = programs_popover.clone();
            move |win| {
                if win.is_active() {
                    if term_popover.is_visible()
                        || settings_popover.is_visible()
                        || programs_popover.is_visible()
                    {
                        return;
                    }
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

        // Periodic 50ms poll for SSH connect results, QUIC LauncherNetEvents, auto-reconnect, and PTY exits
        glib::timeout_add_local(std::time::Duration::from_millis(50), {
            let state = Rc::clone(&state);
            let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
            let rebuild_grid = Rc::clone(&rebuild_grid);
            let refresh_net_status = Rc::clone(&refresh_net_status);
            let refresh_programs_ui = Rc::clone(&refresh_programs_ui);
            let start_connect_inner = Rc::clone(&start_connect_inner);

            move || {
                let mut sidebar_dirty = false;
                let mut grid_dirty = false;
                let mut net_dirty = false;
                let mut programs_dirty = false;
                let mut profiles_to_reconnect: Vec<HostProfile> = Vec::new();

                {
                    let mut st = state.borrow_mut();
                    let active_key = st.active_target.clone();
                    let hosts_copy = st.hosts.clone();
                    let now = Instant::now();
                    for (key, sess) in st.sessions.iter_mut() {
                        let is_active = active_key.as_deref() == Some(key.as_str());

                        if let Some(rx) = sess.connect_rx.take() {
                            match rx.try_recv() {
                                Ok(Ok(res)) => {
                                    sess.status = ConnStatus::Connected;
                                    sess.error = None;
                                    sess.auto_reconnect = true;
                                    sess.reconnect_at = None;
                                    sess.reconnect_backoff_secs = 2;
                                    sess.pty_socket = Some(res.pty_socket);
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
                                    if sess.auto_reconnect {
                                        let delay = sess.reconnect_backoff_secs;
                                        sess.reconnect_at = Some(now + Duration::from_secs(delay));
                                        sess.reconnect_backoff_secs = (delay * 2).min(15);
                                        sess.error = Some(format!("Reconnecting in {delay}s…"));
                                    } else {
                                        sess.error = Some(err);
                                    }
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
                                    if sess.auto_reconnect {
                                        let delay = sess.reconnect_backoff_secs;
                                        sess.reconnect_at = Some(now + Duration::from_secs(delay));
                                        sess.reconnect_backoff_secs = (delay * 2).min(15);
                                        sess.error = Some(format!("Reconnecting in {delay}s…"));
                                    }
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
                                            p.name = name;
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
                                        sess.ctrl_tx = None;
                                        sess.rtt_ms = None;
                                        sess.congested = false;
                                        sess.programs.clear();
                                        if sess.auto_reconnect {
                                            let delay = sess.reconnect_backoff_secs;
                                            sess.reconnect_at =
                                                Some(now + Duration::from_secs(delay));
                                            sess.reconnect_backoff_secs = (delay * 2).min(15);
                                            sess.error =
                                                Some(format!("Reconnecting in {delay}s…"));
                                        } else {
                                            sess.error = err;
                                        }
                                        sidebar_dirty = true;
                                        if is_active {
                                            net_dirty = true;
                                            programs_dirty = true;
                                        }
                                    }
                                }
                            }
                        }

                        if sess.auto_reconnect
                            && sess.status == ConnStatus::Disconnected
                            && sess.connect_rx.is_none()
                            && sess.reconnect_at.is_some_and(|t| now >= t)
                        {
                            sess.reconnect_at = None;
                            if let Some(profile) =
                                hosts_copy.iter().find(|h| h.ssh_target == *key).cloned()
                            {
                                profiles_to_reconnect.push(profile);
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

                for profile in profiles_to_reconnect {
                    start_connect_inner(profile, true);
                    sidebar_dirty = true;
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_gtk_settings_ini_reads_settings_section() {
            let sample = "\
                # KDE generated
                [Other]
                gtk-application-prefer-dark-theme=true

                [Settings]
                gtk-application-prefer-dark-theme=false
                gtk-icon-theme-name=breeze
                gtk-theme-name=Breeze
                gtk-font-name=Noto Sans, 10
            ";
            assert_eq!(
                parse_gtk_settings_ini(sample),
                GtkSettingsIni {
                    prefer_dark: Some(false),
                    theme_name: Some("Breeze".into()),
                    icon_theme_name: Some("breeze".into()),
                }
            );
        }

        #[test]
        fn parse_gtk_settings_ini_handles_numeric_bools_and_missing_keys() {
            let sample = "[Settings]\ngtk-application-prefer-dark-theme=1\n";
            assert_eq!(
                parse_gtk_settings_ini(sample),
                GtkSettingsIni {
                    prefer_dark: Some(true),
                    theme_name: None,
                    icon_theme_name: None,
                }
            );
        }

        #[test]
        fn is_gtk4_theme_file_matches_expected_names() {
            assert!(is_gtk4_theme_file(Path::new("/home/u/.config/gtk-4.0/colors.css")));
            assert!(is_gtk4_theme_file(Path::new("/home/u/.config/gtk-4.0/gtk.css")));
            assert!(is_gtk4_theme_file(Path::new("/home/u/.config/gtk-4.0/settings.ini")));
            assert!(!is_gtk4_theme_file(Path::new("/home/u/.config/gtk-4.0/bookmarks")));
            assert!(!is_gtk4_theme_file(Path::new("/home/u/.config/gtk-4.0/colors.css.tmp")));
        }
    }
}
