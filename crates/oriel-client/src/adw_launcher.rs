//! Native GTK4 + libadwaita launcher.
//!
//! Layout: an `AdwNavigationSplitView` with the server list in the sidebar and the
//! searchable app grid as content (collapsing to a single pane on narrow windows),
//! a bottom status bar listing running programs, and `AdwPreferencesDialog`s for
//! preferences and connection management.
//!
//! Connection state lives in the toolkit-independent [`SessionManager`]; this module only
//! renders it. Background session events are awaited on the GTK main loop and each one
//! refreshes just the views its [`Changes`] name.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use adw::prelude::*;
use anyhow::Result;
use gtk::gdk_pixbuf::PixbufLoader;
use gtk::{gdk, gio, glib};
use gtk4 as gtk;
use oriel_proto::transfers::TransferDirection;
use oriel_proto::{LaunchState, PortRange, RemoteAppEntry};

use crate::hosts::{HostProfile, RenderScaleCap};
use crate::pty::{available_local_terminals, terminal_basename};
use crate::quic::TransferStatus;
use crate::search::{self, MatchTier, SearchIndexEntry};
use crate::session::{Changes, ConnStatus, HostEditError, SessionEvents, SessionManager};
use crate::symbolic_icon::SymbolicIcon;

/// Must match the Flatpak app ID: the sandbox only lets the app own D-Bus names under it,
/// and compositors match the Wayland app ID to the installed `.desktop` file.
const APP_ID: &str = "dev.knoll.Oriel";
/// Labels for [`RenderScaleCap::to_index`] / [`RenderScaleCap::from_index`].
const SCALE_CAP_LABELS: [&str; 4] = ["Uncapped", "2.0x", "1.5x", "1.0x"];
/// Labels for [`RenderScaleCap::option_to_index`] / [`RenderScaleCap::option_from_index`].
const HOST_SCALE_CAP_LABELS: [&str; 5] = ["Global", "Uncapped", "2.0x", "1.5x", "1.0x"];

struct AppUi {
    sessions: SessionManager,
    grid_apps: Vec<RemoteAppEntry>,
    grid_search: Vec<SearchIndexEntry>,
    grid_tiers: Vec<Option<MatchTier>>,
    icon_cache: HashMap<String, Option<gdk::Paintable>>,
}

/// Refreshes the views named by a [`Changes`]. Bound once all views exist, so callbacks
/// created earlier can still reach it.
type ApplyChanges = Rc<dyn Fn(Changes)>;
type ApplyChangesSlot = Rc<RefCell<Option<ApplyChanges>>>;

pub fn run(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
    let app = adw::Application::builder().application_id(APP_ID).build();
    app.connect_startup(|app| {
        install_css();
        install_icon_theme_fallback();
        app.set_accels_for_action("win.preferences", &["<Control>comma"]);
    });
    app.connect_activate(move |app| {
        if let Some(win) = app.active_window() {
            win.present();
            return;
        }
        build_ui(app, cli_max_scale);
    });
    app.run_with_args(&Vec::<String>::new());
    Ok(())
}

const FALLBACK_ICON_THEME: &str = "Adwaita";

/// Uses Adwaita whenever the configured icon theme isn't installed, now and on every change.
///
/// GTK4 doesn't fall back to Adwaita, so a missing theme leaves only hicolor and GTK's few
/// built-in icons. This happens in a Flatpak when the settings portal doesn't provide
/// `icon-theme`: GTK then uses its hardcoded default "gnome", which nothing ships anymore.
///
/// The display's own `IconTheme` can't be pinned, so this sets `gtk-icon-theme-name`. An
/// app-set value shadows the session's, so the configured theme is read from the display
/// (unaffected by the override) and the override is reset once that theme is installed.
fn install_icon_theme_fallback() {
    let (Some(display), Some(settings)) = (gdk::Display::default(), gtk::Settings::default())
    else {
        return;
    };
    apply_icon_theme_fallback(&display, &settings);
    display.connect_setting_changed(move |display, name| {
        if name == "gtk-icon-theme-name" {
            apply_icon_theme_fallback(display, &settings);
        }
    });
}

fn apply_icon_theme_fallback(display: &gdk::Display, settings: &gtk::Settings) {
    const PROPERTY: &str = "gtk-icon-theme-name";
    let configured = display_string_setting(display, PROPERTY).unwrap_or_default();
    let installed = !configured.is_empty()
        && gtk::IconTheme::for_display(display)
            .search_path()
            .iter()
            .any(|dir| dir.join(&configured).join("index.theme").is_file());
    if installed {
        // Follow the session's theme again; harmless if nothing was overridden.
        settings.reset_property(PROPERTY);
    } else if settings.gtk_icon_theme_name().as_deref() != Some(FALLBACK_ICON_THEME) {
        tracing::info!(
            "icon theme {configured:?} is not installed; using {FALLBACK_ICON_THEME} instead"
        );
        settings.set_gtk_icon_theme_name(Some(FALLBACK_ICON_THEME));
    }
}

/// The session's value of a string setting, before any app override. Not
/// `DisplayExt::get_setting`, which passes GDK an untyped `GValue` that it can't fill.
fn display_string_setting(display: &gdk::Display, name: &str) -> Option<String> {
    use glib::translate::{ToGlibPtr, ToGlibPtrMut};

    let mut value = glib::Value::from_type(glib::Type::STRING);
    // SAFETY: `display` and the NUL-terminated `name` outlive the call, and `value` is an
    // initialized string `GValue`, the type GDK stores for string settings.
    let found = unsafe {
        gdk::ffi::gdk_display_get_setting(
            display.to_glib_none().0,
            name.to_glib_none().0,
            value.to_glib_none_mut().0,
        )
    };
    if found == glib::ffi::GFALSE {
        return None;
    }
    value.get::<Option<String>>().ok().flatten()
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(
        "
        flowbox.app-grid > flowboxchild {
            border-radius: 12px;
            padding: 6px;
        }
        flowbox.app-grid > flowboxchild:hover {
            background-color: alpha(currentColor, 0.07);
        }
        flowbox.app-grid > flowboxchild:selected {
            background-color: alpha(@accent_bg_color, 0.18);
            color: inherit;
        }
        .status-dot {
            font-size: 0.8em;
        }
        ",
    );
    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

/// Color a status indicator with one of Adwaita's semantic text style classes.
fn set_status_class(widget: &impl IsA<gtk::Widget>, class: &str) {
    for c in ["success", "warning", "error", "accent", "dim-label"] {
        if c != class {
            widget.remove_css_class(c);
        }
    }
    widget.add_css_class(class);
}

/// A byte count for people ("1.2 GB").
fn format_size(bytes: u64) -> String {
    glib::format_size(bytes).to_string()
}

/// A transfer rate for people, in kB/s at least ("0 kB/s", "1.2 MB/s"): an idle connection
/// still sends a few bytes of keepalives.
fn format_rate(bytes_per_sec: u64) -> String {
    if bytes_per_sec < 1000 {
        "0 kB/s".to_string()
    } else {
        format!("{}/s", format_size(bytes_per_sec))
    }
}

/// One file transfer's row in the programs popover, updated in place every second.
struct TransferRow {
    row: gtk::Box,
    bar: gtk::ProgressBar,
    detail: gtk::Label,
    cancel: gtk::Button,
}

impl TransferRow {
    fn new(id: u64, direction: TransferDirection, name: &str, target: &str) -> Self {
        // The same arrows as the status bar's received/sent rates.
        let (arrow, tip) = match direction {
            TransferDirection::Incoming => ("↓", format!("From {target}")),
            TransferDirection::Outgoing => ("↑", format!("To {target}")),
        };
        let top = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        top.append(
            &gtk::Label::builder()
                .label(arrow)
                .tooltip_text(&tip)
                .css_classes(["dim-label"])
                .build(),
        );
        top.append(
            &gtk::Label::builder()
                .label(name)
                .tooltip_text(name)
                .halign(gtk::Align::Start)
                .hexpand(true)
                .ellipsize(gtk::pango::EllipsizeMode::Middle)
                .build(),
        );
        let cancel = gtk::Button::builder()
            .icon_name("window-close-symbolic")
            .tooltip_text("Cancel transfer")
            .css_classes(["flat", "circular"])
            .build();
        cancel.connect_clicked(move |btn| {
            oriel_proto::transfers::cancel(id);
            btn.set_sensitive(false);
        });
        top.append(&cancel);
        let bar = gtk::ProgressBar::new();
        let detail = gtk::Label::builder()
            .halign(gtk::Align::Start)
            .css_classes(["dim-label", "caption", "numeric"])
            .build();
        let row = gtk::Box::new(gtk::Orientation::Vertical, 4);
        row.append(&top);
        row.append(&bar);
        row.append(&detail);
        Self {
            row,
            bar,
            detail,
            cancel,
        }
    }

    fn update(&self, t: &TransferStatus) {
        let info = &t.info;
        self.bar.set_fraction(if info.size == 0 {
            1.0
        } else {
            info.position as f64 / info.size as f64
        });
        if info.cancelled {
            self.cancel.set_sensitive(false);
        }
        let speed = if info.cancelled {
            "Cancelled".to_string()
        } else if t.rate == 0 {
            // Nothing is reading it: the app paused or is busy writing.
            "Idle".to_string()
        } else {
            format_rate(t.rate)
        };
        self.detail.set_text(&format!(
            "{} of {} · {speed}",
            format_size(info.position),
            format_size(info.size)
        ));
    }
}

fn status_dot() -> gtk::Label {
    let dot = gtk::Label::new(Some("●"));
    dot.add_css_class("status-dot");
    dot
}

fn show_toast(overlay: &adw::ToastOverlay, title: &str) {
    overlay.add_toast(adw::Toast::builder().title(title).use_markup(false).build());
}

fn show_dialog_toast(dialog: &adw::PreferencesDialog, title: &str) {
    dialog.add_toast(adw::Toast::builder().title(title).use_markup(false).build());
}

fn escape_markup(s: &str) -> String {
    glib::markup_escape_text(s).to_string()
}

/// `true` if `focused` is `widget` or one of its descendants.
fn focus_within(focused: &gtk::Widget, widget: &impl IsA<gtk::Widget>) -> bool {
    let widget = widget.upcast_ref::<gtk::Widget>();
    focused == widget || focused.is_ancestor(widget)
}

/// Grab focus only if `widget` is currently shown and usable (e.g. not on the hidden
/// pane of a collapsed split view).
fn try_focus(widget: &impl IsA<gtk::Widget>) -> bool {
    widget.is_mapped() && widget.is_sensitive() && widget.grab_focus()
}

fn focus_server_list(list: &gtk::ListBox) -> bool {
    if !list.is_mapped() {
        return false;
    }
    match list.selected_row().or_else(|| list.row_at_index(0)) {
        Some(row) => row.grab_focus(),
        None => list.grab_focus(),
    }
}

/// The primary button and dropdown arrow inside an `AdwSplitButton`, so the custom
/// Tab ring can stop on each.
fn split_button_parts(split: &adw::SplitButton) -> (gtk::Widget, gtk::Widget) {
    let mut primary = None;
    let mut arrow = None;
    let mut child = split.first_child();
    while let Some(c) = child {
        if c.is::<gtk::MenuButton>() {
            arrow = Some(c.clone());
        } else if c.is::<gtk::Button>() {
            primary = Some(c.clone());
        }
        child = c.next_sibling();
    }
    let fallback = split.clone().upcast::<gtk::Widget>();
    (
        primary.unwrap_or_else(|| fallback.clone()),
        arrow.unwrap_or(fallback),
    )
}

fn build_ui(app: &adw::Application, cli_max_scale: Option<RenderScaleCap>) {
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Oriel")
        .default_width(960)
        .default_height(640)
        .width_request(360)
        .height_request(320)
        .build();

    let (sessions, session_events) = SessionManager::new(cli_max_scale);
    let state = Rc::new(RefCell::new(AppUi {
        sessions,
        grid_apps: Vec::new(),
        grid_search: Vec::new(),
        grid_tiers: Vec::new(),
        icon_cache: HashMap::new(),
    }));
    let current_filter = Rc::new(RefCell::new(String::new()));
    let apply_slot: ApplyChangesSlot = Rc::new(RefCell::new(None));
    let apply_changes: ApplyChanges = {
        let apply_slot = Rc::clone(&apply_slot);
        Rc::new(move |changes| {
            let apply = apply_slot.borrow().clone();
            if let Some(apply) = apply {
                apply(changes);
            }
        })
    };

    // --- Sidebar: saved servers ---
    let manage_btn = gtk::Button::builder()
        .icon_name("document-edit-symbolic")
        .tooltip_text("Manage Connections")
        .action_name("win.manage-connections")
        .build();
    let sidebar_header = adw::HeaderBar::new();
    sidebar_header.pack_end(&manage_btn);

    let server_list = gtk::ListBox::new();
    server_list.add_css_class("navigation-sidebar");
    server_list.set_selection_mode(gtk::SelectionMode::Single);
    let no_servers_lbl = gtk::Label::builder()
        .label("No saved servers")
        .wrap(true)
        .justify(gtk::Justification::Center)
        .margin_top(24)
        .margin_start(12)
        .margin_end(12)
        .css_classes(["dim-label"])
        .build();
    server_list.set_placeholder(Some(&no_servers_lbl));
    let server_scroll = gtk::ScrolledWindow::builder()
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .child(&server_list)
        .build();

    let sidebar_view = adw::ToolbarView::new();
    sidebar_view.add_top_bar(&sidebar_header);
    sidebar_view.set_content(Some(&server_scroll));
    let sidebar_page = adw::NavigationPage::new(&sidebar_view, "Servers");

    // --- Content header: search, New Terminal split button, preferences ---
    let search_entry = gtk::SearchEntry::builder()
        .placeholder_text("Search apps or type a command…")
        .hexpand(true)
        .build();
    let search_clamp = adw::Clamp::builder()
        .maximum_size(520)
        .hexpand(true)
        .child(&search_entry)
        .build();

    // Primary click opens the default terminal emulator; the dropdown picks any installed one.
    let term_menu = gio::Menu::new();
    let term_popover = gtk::PopoverMenu::from_model(Some(&term_menu));
    let term_split = adw::SplitButton::builder()
        .label("New Terminal")
        .tooltip_text("Open a remote terminal in the default terminal emulator")
        .dropdown_tooltip("Choose terminal emulator")
        .popover(&term_popover)
        .sensitive(false)
        .build();
    let (term_btn, term_menu_btn) = split_button_parts(&term_split);

    let settings_btn = gtk::Button::builder()
        .icon_name("emblem-system-symbolic")
        .tooltip_text("Preferences")
        .action_name("win.preferences")
        .build();

    let content_header = adw::HeaderBar::new();
    content_header.set_title_widget(Some(&search_clamp));
    content_header.pack_end(&settings_btn);
    content_header.pack_end(&term_split);

    // --- Content: app grid, or a status page when there is nothing to show ---
    let flow = gtk::FlowBox::builder()
        .valign(gtk::Align::Start)
        .max_children_per_line(8)
        .min_children_per_line(3)
        .selection_mode(gtk::SelectionMode::Single)
        .activate_on_single_click(true)
        .row_spacing(8)
        .column_spacing(8)
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .css_classes(["app-grid"])
        .build();
    let grid_scroll = gtk::ScrolledWindow::builder()
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .child(&flow)
        .build();

    let empty_add_btn = gtk::Button::builder()
        .label("Add Server")
        .halign(gtk::Align::Center)
        .action_name("win.manage-connections")
        .css_classes(["pill", "suggested-action"])
        .build();
    let empty_spinner = gtk::Spinner::builder()
        .width_request(32)
        .height_request(32)
        .halign(gtk::Align::Center)
        .build();
    let empty_extra = gtk::Box::new(gtk::Orientation::Vertical, 0);
    empty_extra.append(&empty_add_btn);
    empty_extra.append(&empty_spinner);
    let empty_page = adw::StatusPage::builder().child(&empty_extra).build();

    let content_stack = gtk::Stack::new();
    content_stack.add_named(&grid_scroll, Some("grid"));
    content_stack.add_named(&empty_page, Some("empty"));

    let toasts = adw::ToastOverlay::new();
    toasts.set_child(Some(&content_stack));

    // --- Bottom status bar; click to list running programs ---
    let status_bar_btn = gtk::MenuButton::builder()
        .has_frame(false)
        .hexpand(true)
        .direction(gtk::ArrowType::Up)
        .tooltip_text("Running programs")
        .build();
    let status_bar_box = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let status_dot_lbl = status_dot();
    let status_net_lbl = gtk::Label::builder()
        .label("Disconnected")
        .hexpand(true)
        .halign(gtk::Align::Start)
        .ellipsize(gtk::pango::EllipsizeMode::End)
        .single_line_mode(true)
        .build();
    let status_pills_box = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    status_pills_box.set_halign(gtk::Align::End);
    let status_count_lbl = gtk::Label::builder()
        .label("0 active")
        .css_classes(["dim-label", "numeric"])
        .build();
    let status_xfer_lbl = gtk::Label::builder()
        .css_classes(["numeric"])
        .visible(false)
        .build();
    let status_rate_lbl = gtk::Label::builder()
        .css_classes(["dim-label", "numeric"])
        .tooltip_text("Network traffic: received ↓, sent ↑")
        .visible(false)
        .build();
    status_bar_box.append(&status_dot_lbl);
    status_bar_box.append(&status_net_lbl);
    status_bar_box.append(&status_pills_box);
    status_bar_box.append(&status_xfer_lbl);
    status_bar_box.append(&status_rate_lbl);
    status_bar_box.append(&status_count_lbl);
    status_bar_btn.set_child(Some(&status_bar_box));

    let programs_popover = gtk::Popover::new();
    let popover_content = gtk::Box::new(gtk::Orientation::Vertical, 6);
    popover_content.set_margin_top(6);
    popover_content.set_margin_bottom(6);
    popover_content.set_margin_start(6);
    popover_content.set_margin_end(6);
    popover_content.set_width_request(340);
    // File transfers, shown above the programs while there are any.
    let transfers_section = gtk::Box::new(gtk::Orientation::Vertical, 6);
    transfers_section.set_visible(false);
    transfers_section.append(
        &gtk::Label::builder()
            .label("File Transfers")
            .halign(gtk::Align::Start)
            .css_classes(["heading"])
            .build(),
    );
    let transfers_list = gtk::Box::new(gtk::Orientation::Vertical, 8);
    transfers_section.append(&transfers_list);
    transfers_section.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    let programs_list = gtk::Box::new(gtk::Orientation::Vertical, 4);
    popover_content.append(&transfers_section);
    popover_content.append(&programs_list);
    programs_popover.set_child(Some(&popover_content));
    status_bar_btn.set_popover(Some(&programs_popover));

    let bottom_bar = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    bottom_bar.add_css_class("toolbar");
    bottom_bar.append(&status_bar_btn);

    let content_view = adw::ToolbarView::new();
    content_view.add_top_bar(&content_header);
    content_view.set_content(Some(&toasts));
    content_view.add_bottom_bar(&bottom_bar);
    let content_page = adw::NavigationPage::new(&content_view, "Apps");

    let split_view = adw::NavigationSplitView::builder()
        .sidebar(&sidebar_page)
        .content(&content_page)
        .min_sidebar_width(160.0)
        .max_sidebar_width(260.0)
        .build();
    let narrow = adw::Breakpoint::new(
        adw::BreakpointCondition::parse("max-width: 560sp").expect("valid breakpoint condition"),
    );
    narrow.add_setter(&split_view, "collapsed", Some(&true.to_value()));
    window.add_breakpoint(narrow);
    window.set_content(Some(&split_view));

    // --- Grid search helpers ---
    /// Recover the original `grid_apps`/`grid_search` index from a `FlowBoxChild`.
    /// `child.index()` reflects sorted visual order once `set_sort_func` runs, so we
    /// tag each child's `widget_name` with its stable source index in `rebuild_grid`.
    fn flow_child_app_idx(child: &gtk::FlowBoxChild) -> Option<usize> {
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
    fn child_matches_query(st: &AppUi, child: &gtk::FlowBoxChild, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }
        flow_child_app_idx(child)
            .and_then(|idx| st.grid_tiers.get(idx).copied().flatten())
            .is_some()
    }
    fn first_matching_child(
        flow: &gtk::FlowBox,
        st: &AppUi,
        query: &str,
    ) -> Option<gtk::FlowBoxChild> {
        let mut cur = flow.first_child();
        while let Some(widget) = cur {
            if let Ok(child) = widget.clone().downcast::<gtk::FlowBoxChild>() {
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

    // --- Programs popover keyboard navigation ---
    // Every visible button in `root`, in order: transfer cancel buttons, then each
    // program's reattach and close buttons.
    fn collect_kill_buttons(root: &impl IsA<gtk::Widget>) -> Vec<gtk::Button> {
        fn walk(w: &gtk::Widget, out: &mut Vec<gtk::Button>) {
            if !w.is_visible() {
                return;
            }
            if let Some(btn) = w.downcast_ref::<gtk::Button>() {
                out.push(btn.clone());
                return;
            }
            let mut child = w.first_child();
            while let Some(c) = child {
                walk(&c, out);
                child = c.next_sibling();
            }
        }
        let mut btns = Vec::new();
        walk(root.upcast_ref(), &mut btns);
        btns
    }

    let pop_key_ctrl = gtk::EventControllerKey::new();
    pop_key_ctrl.set_propagation_phase(gtk::PropagationPhase::Capture);
    pop_key_ctrl.connect_key_pressed({
        let popover_content = popover_content.clone();
        let programs_popover = programs_popover.clone();
        let status_bar_btn = status_bar_btn.clone();
        let window_ref = window.clone();
        move |_, key, _, mods| {
            if key == gdk::Key::Escape {
                programs_popover.popdown();
                status_bar_btn.grab_focus();
                return glib::Propagation::Stop;
            }
            let is_next = key == gdk::Key::Down
                || (key == gdk::Key::Tab && !mods.contains(gdk::ModifierType::SHIFT_MASK));
            let is_prev = key == gdk::Key::Up
                || key == gdk::Key::ISO_Left_Tab
                || (key == gdk::Key::Tab && mods.contains(gdk::ModifierType::SHIFT_MASK));
            if is_next || is_prev {
                let btns = collect_kill_buttons(&popover_content);
                if btns.is_empty() {
                    return glib::Propagation::Stop;
                }
                let focused = gtk::prelude::GtkWindowExt::focus(&window_ref)
                    .or_else(|| programs_popover.focus_child());
                let cur = focused
                    .as_ref()
                    .and_then(|f| btns.iter().position(|b| focus_within(f, b)));
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

    // --- View refresh helpers ---
    let refresh_content_view: Rc<dyn Fn()> = {
        let state = Rc::clone(&state);
        let current_filter = Rc::clone(&current_filter);
        let search_entry = search_entry.clone();
        let content_stack = content_stack.clone();
        let content_page = content_page.clone();
        let empty_page = empty_page.clone();
        let empty_add_btn = empty_add_btn.clone();
        let empty_spinner = empty_spinner.clone();
        Rc::new(move || {
            let st = state.borrow();
            let sessions = &st.sessions;
            let active_label = sessions.active_target().map(|t| sessions.host_label(t));
            content_page.set_title(active_label.as_deref().unwrap_or("Apps"));

            let mut busy = false;
            let mut offer_add = false;
            let (icon, title, description): (&str, &str, String) =
                match (sessions.active_target(), sessions.active_session()) {
                    (None, _) if sessions.hosts().is_empty() => {
                        offer_add = true;
                        (
                            "network-server-symbolic",
                            "No Servers",
                            "Add a server to launch its applications here.".to_string(),
                        )
                    }
                    (None, _) => (
                        "network-server-symbolic",
                        "No Server Selected",
                        "Choose a server from the sidebar.".to_string(),
                    ),
                    (Some(_), None) => (
                        "network-offline-symbolic",
                        "Not Connected",
                        format!(
                            "Activate <b>{}</b> in the sidebar to connect.",
                            escape_markup(active_label.as_deref().unwrap_or_default())
                        ),
                    ),
                    (Some(target), Some(sess)) => match sess.status {
                        ConnStatus::Connecting => {
                            busy = true;
                            let detail = match &sess.error {
                                Some(e) => escape_markup(e),
                                None => format!(
                                    "Starting Oriel on <b>{}</b> over SSH.",
                                    escape_markup(target)
                                ),
                            };
                            ("network-transmit-receive-symbolic", "Connecting…", detail)
                        }
                        ConnStatus::Disconnected => (
                            "network-offline-symbolic",
                            "Disconnected",
                            match &sess.error {
                                Some(e) => escape_markup(e),
                                None => {
                                    "Activate the server in the sidebar to reconnect.".to_string()
                                }
                            },
                        ),
                        ConnStatus::Connected => {
                            let query = current_filter.borrow();
                            let has_match = if query.is_empty() {
                                !st.grid_apps.is_empty()
                            } else {
                                st.grid_tiers.iter().any(Option::is_some)
                            };
                            if has_match {
                                content_stack.set_visible_child_name("grid");
                                return;
                            }
                            if !sess.apps_loaded {
                                busy = true;
                                ("view-app-grid-symbolic", "Loading Apps…", String::new())
                            } else if query.is_empty() {
                                (
                                    "view-app-grid-symbolic",
                                    "No Apps Found",
                                    "The server reported no desktop applications. \
                                     Type a command and press Enter to run it."
                                        .to_string(),
                                )
                            } else {
                                let typed = search_entry.text();
                                let cmd = search::explicit_command(&typed)
                                    .unwrap_or_else(|| typed.trim());
                                (
                                    "system-search-symbolic",
                                    "No Matching Apps",
                                    format!(
                                        "Press Enter to run <b>{}</b> as a command.",
                                        escape_markup(cmd)
                                    ),
                                )
                            }
                        }
                    },
                };

            empty_page.set_icon_name(Some(icon));
            empty_page.set_title(title);
            empty_page.set_description((!description.is_empty()).then_some(description.as_str()));
            empty_add_btn.set_visible(offer_add);
            empty_spinner.set_visible(busy);
            empty_spinner.set_spinning(busy);
            content_stack.set_visible_child_name("empty");
        })
    };

    let refresh_net_status: Rc<dyn Fn()> = {
        let state = Rc::clone(&state);
        let status_net_lbl = status_net_lbl.clone();
        let status_rate_lbl = status_rate_lbl.clone();
        let status_dot_lbl = status_dot_lbl.clone();
        let term_split = term_split.clone();
        let term_popover = term_popover.clone();
        let refresh_content_view = Rc::clone(&refresh_content_view);

        Rc::new(move || {
            {
                let st = state.borrow();
                let (class, msg, connected) = match st.sessions.active_target() {
                    None => ("dim-label", "No server selected".to_string(), false),
                    Some(target) => match st.sessions.session(target) {
                        None => ("dim-label", format!("{target} · Disconnected"), false),
                        Some(sess) => match sess.status {
                            ConnStatus::Disconnected => {
                                let msg = sess
                                    .error
                                    .as_deref()
                                    .map(|e| format!("{target} · Disconnected ({e})"))
                                    .unwrap_or_else(|| format!("{target} · Disconnected"));
                                ("error", msg, false)
                            }
                            ConnStatus::Connecting => {
                                ("accent", format!("{target} · Connecting…"), false)
                            }
                            ConnStatus::Connected if sess.congested => {
                                let rtt = sess.rtt_ms.unwrap_or(0);
                                ("warning", format!("{target} · Congested · {rtt}ms"), true)
                            }
                            ConnStatus::Connected => {
                                let rtt =
                                    sess.rtt_ms.map(|r| format!(" · {r}ms")).unwrap_or_default();
                                ("success", format!("{target} · Connected{rtt}"), true)
                            }
                        },
                    },
                };
                set_status_class(&status_dot_lbl, class);
                let rates = st
                    .sessions
                    .active_target()
                    .and_then(|t| st.sessions.session(t))
                    .and_then(|s| s.rates)
                    .filter(|_| connected);
                if let Some((rx, tx)) = rates {
                    status_rate_lbl.set_text(&format!(
                        "↓ {}  ↑ {}",
                        format_rate(rx),
                        format_rate(tx)
                    ));
                }
                status_rate_lbl.set_visible(rates.is_some());
                status_net_lbl.set_text(&msg);
                status_net_lbl.set_tooltip_text(Some(&msg));
                if !connected {
                    term_popover.popdown();
                }
                term_split.set_sensitive(connected);
            }
            refresh_content_view();
        })
    };

    let refresh_programs_ui: Rc<dyn Fn()> = {
        let state = Rc::clone(&state);
        let status_pills_box = status_pills_box.clone();
        let status_count_lbl = status_count_lbl.clone();
        let programs_list = programs_list.clone();
        let programs_popover = programs_popover.clone();
        let toasts = toasts.clone();

        Rc::new(move || {
            while let Some(c) = status_pills_box.first_child() {
                status_pills_box.remove(&c);
            }
            while let Some(c) = programs_list.first_child() {
                programs_list.remove(&c);
            }

            let st = state.borrow();
            let Some((target, sess)) = st
                .sessions
                .active_target()
                .and_then(|t| st.sessions.session(t).map(|s| (t, s)))
            else {
                status_count_lbl.set_text("0 active");
                programs_list.append(
                    &gtk::Label::builder()
                        .label("No active programs")
                        .css_classes(["dim-label"])
                        .build(),
                );
                return;
            };

            status_count_lbl.set_text(&format!("{} active", sess.programs.len()));

            for prog in &sess.programs {
                if prog.state == LaunchState::Launching {
                    let pill = gtk::Label::builder()
                        .label(format!("Launching {}…", prog.name))
                        .css_classes(["dim-label"])
                        .ellipsize(gtk::pango::EllipsizeMode::End)
                        .max_width_chars(24)
                        .build();
                    status_pills_box.append(&pill);
                }

                let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
                let name_lbl = gtk::Label::builder()
                    .label(&prog.name)
                    .halign(gtk::Align::Start)
                    .hexpand(true)
                    .ellipsize(gtk::pango::EllipsizeMode::End)
                    .build();
                let state_lbl = gtk::Label::builder()
                    .label(match prog.state {
                        LaunchState::Launching => "Launching",
                        LaunchState::Running => "Running",
                        LaunchState::Exited(_) => "Exited",
                    })
                    .css_classes(["dim-label", "caption"])
                    .build();
                row.append(&name_lbl);
                row.append(&state_lbl);

                if prog.is_detached_pty() && sess.status == ConnStatus::Connected {
                    let reattach_btn = gtk::Button::builder()
                        .icon_name("utilities-terminal-symbolic")
                        .tooltip_text("Reattach detached terminal")
                        .css_classes(["flat"])
                        .build();
                    let state_c = Rc::clone(&state);
                    let toasts = toasts.clone();
                    let target_c = target.to_string();
                    let launch_id = prog.launch_id;
                    reattach_btn.connect_clicked(move |_| {
                        let res = state_c
                            .borrow()
                            .sessions
                            .reattach_terminal(&target_c, launch_id);
                        if let Err(err) = res {
                            show_toast(&toasts, &format!("Couldn't reattach the terminal: {err}"));
                        }
                    });
                    row.append(&reattach_btn);
                }

                let kill_btn = if prog.term_sent {
                    gtk::Button::builder()
                        .label("Kill")
                        .tooltip_text("Force kill")
                        .css_classes(["destructive-action"])
                        .build()
                } else {
                    gtk::Button::builder()
                        .icon_name("window-close-symbolic")
                        .tooltip_text("Close")
                        .css_classes(["flat", "circular"])
                        .build()
                };
                let target_c = target.to_string();
                let launch_id = prog.launch_id;
                let state_c = Rc::clone(&state);
                // Restyle in place rather than rebuilding the list, so focus stays put.
                kill_btn.connect_clicked(move |btn| {
                    let sent = state_c
                        .borrow_mut()
                        .sessions
                        .terminate(&target_c, launch_id);
                    if sent {
                        btn.set_label("Kill");
                        btn.set_tooltip_text(Some("Force kill"));
                        btn.remove_css_class("flat");
                        btn.remove_css_class("circular");
                        btn.add_css_class("destructive-action");
                    }
                });
                row.append(&kill_btn);
                programs_list.append(&row);
            }

            if sess.programs.is_empty() {
                programs_list.append(
                    &gtk::Label::builder()
                        .label("No active programs")
                        .css_classes(["dim-label"])
                        .build(),
                );
            } else if programs_popover.is_visible() && programs_popover.focus_child().is_some() {
                if let Some(first) = collect_kill_buttons(&programs_list).into_iter().next() {
                    first.grab_focus();
                }
            }
        })
    };

    let refresh_transfers_ui: Rc<dyn Fn()> = {
        let state = Rc::clone(&state);
        let transfers_section = transfers_section.clone();
        let transfers_list = transfers_list.clone();
        let status_xfer_lbl = status_xfer_lbl.clone();
        let rows: RefCell<HashMap<(String, u64), TransferRow>> = RefCell::default();
        Rc::new(move || {
            let st = state.borrow();
            let active = st
                .sessions
                .active_target()
                .and_then(|t| st.sessions.session(t).map(|s| (t, s)));
            let transfers = active.map(|(_, s)| s.transfers.as_slice()).unwrap_or(&[]);
            let target = active.map(|(t, _)| t).unwrap_or_default();
            let mut rows = rows.borrow_mut();
            // Rows are keyed by host too: switching hosts must not reuse another's rows.
            rows.retain(|(host, id), row| {
                let keep = host == target && transfers.iter().any(|t| t.info.id == *id);
                if !keep {
                    transfers_list.remove(&row.row);
                }
                keep
            });
            for t in transfers {
                let row = rows
                    .entry((target.to_string(), t.info.id))
                    .or_insert_with(|| {
                        let row =
                            TransferRow::new(t.info.id, t.info.direction, &t.info.name, target);
                        transfers_list.append(&row.row);
                        row
                    });
                row.update(t);
            }
            transfers_section.set_visible(!transfers.is_empty());
            status_xfer_lbl.set_visible(!transfers.is_empty());
            if !transfers.is_empty() {
                let size: u64 = transfers.iter().map(|t| t.info.size).sum();
                let done: u64 = transfers.iter().map(|t| t.info.position).sum();
                let percent = (done * 100).checked_div(size).unwrap_or(100);
                let count = match transfers.len() {
                    1 => "1 transfer".to_string(),
                    n => format!("{n} transfers"),
                };
                status_xfer_lbl.set_text(&format!("{count} · {percent}%"));
            }
        })
    };

    let refresh_status_bar: Rc<dyn Fn()> = {
        let refresh_net_status = Rc::clone(&refresh_net_status);
        let refresh_programs_ui = Rc::clone(&refresh_programs_ui);
        let refresh_transfers_ui = Rc::clone(&refresh_transfers_ui);
        Rc::new(move || {
            refresh_net_status();
            refresh_programs_ui();
            refresh_transfers_ui();
        })
    };

    let launch_exec_or_app: Rc<dyn Fn(String, String)> = {
        let state = Rc::clone(&state);
        let search_entry = search_entry.clone();
        let apply_changes = Rc::clone(&apply_changes);
        Rc::new(move |name: String, exec: String| {
            let changes = state.borrow_mut().sessions.launch(name, exec);
            search_entry.set_text("");
            search_entry.grab_focus();
            apply_changes(changes);
        })
    };

    // Grid apps: `Terminal=true` entries open in a local terminal over a remote PTY.
    let launch_app: Rc<dyn Fn(RemoteAppEntry)> = {
        let state = Rc::clone(&state);
        let search_entry = search_entry.clone();
        let toasts = toasts.clone();
        let launch_exec_or_app = Rc::clone(&launch_exec_or_app);
        Rc::new(move |app: RemoteAppEntry| {
            if !app.terminal {
                launch_exec_or_app(app.name, app.exec);
                return;
            }
            if let Err(err) = state.borrow().sessions.launch_in_terminal(&app.exec) {
                tracing::warn!(error = %err, exec = %app.exec, "failed to open terminal app");
                show_toast(&toasts, &format!("Couldn't open {}: {err}", app.name));
            }
            search_entry.set_text("");
            search_entry.grab_focus();
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
        let refresh_content_view = Rc::clone(&refresh_content_view);
        Rc::new(move || {
            while let Some(child) = flow.first_child() {
                flow.remove(&child);
            }
            let apps = {
                let mut st = state.borrow_mut();
                let apps = st
                    .sessions
                    .active_session()
                    .map(|s| s.apps.clone())
                    .unwrap_or_default();
                for entry in &apps {
                    if !st.icon_cache.contains_key(&entry.id) && !entry.icon_data.is_empty() {
                        st.icon_cache
                            .insert(entry.id.clone(), app_icon_paintable(entry));
                    }
                }
                st.grid_search = apps.iter().map(SearchIndexEntry::from_app).collect();
                st.grid_apps = apps.clone();
                let needle = current_filter.borrow();
                recompute_grid_tiers(&mut st, needle.as_str());
                apps
            };

            for (idx, entry) in apps.iter().enumerate() {
                let vbox = gtk::Box::new(gtk::Orientation::Vertical, 6);
                vbox.set_size_request(96, 84);
                let img = match state.borrow().icon_cache.get(&entry.id) {
                    Some(Some(icon)) => gtk::Image::from_paintable(Some(icon)),
                    _ => gtk::Image::from_icon_name("application-x-executable-symbolic"),
                };
                img.set_pixel_size(48);
                img.add_css_class("icon-dropshadow");
                vbox.append(&img);
                let lbl = gtk::Label::builder()
                    .label(&entry.name)
                    .ellipsize(gtk::pango::EllipsizeMode::End)
                    .max_width_chars(14)
                    .build();
                vbox.append(&lbl);

                let child = gtk::FlowBoxChild::new();
                child.set_widget_name(&idx.to_string());
                child.set_tooltip_text(Some(&entry.name));
                child.set_child(Some(&vbox));
                flow.insert(&child, -1);
            }

            flow.invalidate_filter();
            flow.invalidate_sort();
            select_first_visible();
            refresh_content_view();
        })
    };

    let rebuild_sidebar: Rc<dyn Fn()> = {
        let state = Rc::clone(&state);
        let server_list = server_list.clone();
        let apply_changes = Rc::clone(&apply_changes);
        Rc::new(move || {
            // `remove_all` keeps the placeholder (removing every child would drop it).
            server_list.remove_all();
            let st = state.borrow();
            let sessions = &st.sessions;
            for profile in sessions.hosts() {
                let (status, wants_connection) = sessions
                    .session(&profile.ssh_target)
                    .map(|s| (s.status, s.wants_connection()))
                    .unwrap_or((ConnStatus::Disconnected, false));

                let row_box = gtk::Box::new(gtk::Orientation::Horizontal, 8);
                let dot = status_dot();
                set_status_class(
                    &dot,
                    match status {
                        ConnStatus::Connected => "success",
                        ConnStatus::Connecting => "accent",
                        ConnStatus::Disconnected => "dim-label",
                    },
                );

                let lbl = gtk::Label::builder()
                    .label(&profile.label)
                    .halign(gtk::Align::Start)
                    .hexpand(true)
                    .ellipsize(gtk::pango::EllipsizeMode::End)
                    .build();

                let dc_btn = gtk::Button::builder()
                    .icon_name("media-playback-stop-symbolic")
                    .tooltip_text("Disconnect")
                    .can_focus(false)
                    .focus_on_click(false)
                    .valign(gtk::Align::Center)
                    .css_classes(["flat", "circular"])
                    .visible(wants_connection)
                    .build();
                let dc_target = profile.ssh_target.clone();
                let state_dc = Rc::clone(&state);
                let apply_dc = Rc::clone(&apply_changes);
                dc_btn.connect_clicked(move |_| {
                    let changes = state_dc.borrow_mut().sessions.disconnect(&dc_target);
                    apply_dc(changes);
                });

                row_box.append(&dot);
                row_box.append(&lbl);
                row_box.append(&dc_btn);

                let row = gtk::ListBoxRow::new();
                row.set_child(Some(&row_box));
                row.set_widget_name(&profile.ssh_target);
                row.set_tooltip_text(Some(&profile.ssh_target));
                server_list.append(&row);

                if sessions.active_target() == Some(profile.ssh_target.as_str()) {
                    server_list.select_row(Some(&row));
                }
            }
        })
    };

    server_list.connect_row_activated({
        let state = Rc::clone(&state);
        let apply_changes = Rc::clone(&apply_changes);
        let search_entry = search_entry.clone();
        let split_view = split_view.clone();
        move |_, row| {
            let target = row.widget_name();
            let changes = state.borrow_mut().sessions.connect(&target);
            if changes != Changes::NONE {
                apply_changes(changes);
                split_view.set_show_content(true);
                search_entry.grab_focus();
            }
        }
    });

    // Every view now exists; route `Changes` to the refreshers they name.
    *apply_slot.borrow_mut() = Some({
        let rebuild_sidebar = Rc::clone(&rebuild_sidebar);
        let rebuild_grid = Rc::clone(&rebuild_grid);
        let refresh_net_status = Rc::clone(&refresh_net_status);
        let refresh_programs_ui = Rc::clone(&refresh_programs_ui);
        let refresh_transfers_ui = Rc::clone(&refresh_transfers_ui);
        Rc::new(move |changes: Changes| {
            if changes.hosts {
                rebuild_sidebar();
            }
            if changes.apps {
                rebuild_grid();
            }
            if changes.status {
                refresh_net_status();
            }
            if changes.programs {
                refresh_programs_ui();
            }
            if changes.transfers {
                refresh_transfers_ui();
            }
        })
    });

    // --- Window actions ---
    let manage_action = gio::SimpleAction::new("manage-connections", None);
    manage_action.connect_activate({
        let window = window.clone();
        let state = Rc::clone(&state);
        let apply_changes = Rc::clone(&apply_changes);
        move |_, _| {
            open_manage_connections_dialog(ManageCtx {
                window: window.clone(),
                dialog: adw::PreferencesDialog::builder()
                    .title("Connections")
                    .search_enabled(false)
                    .build(),
                saved_group: adw::PreferencesGroup::builder()
                    .title("Saved Connections")
                    .build(),
                saved_rows: Rc::new(RefCell::new(Vec::new())),
                state: Rc::clone(&state),
                apply_changes: Rc::clone(&apply_changes),
            });
        }
    });
    window.add_action(&manage_action);

    let prefs_action = gio::SimpleAction::new("preferences", None);
    prefs_action.connect_activate({
        let window = window.clone();
        let state = Rc::clone(&state);
        move |_, _| open_preferences_dialog(&window, Rc::clone(&state))
    });
    window.add_action(&prefs_action);

    // --- Search: instant in-place ranked filter + Enter to launch & reset ---
    let apply_filter: Rc<dyn Fn(String)> = Rc::new({
        let state = Rc::clone(&state);
        let current_filter = Rc::clone(&current_filter);
        let flow = flow.clone();
        let select_first_visible = Rc::clone(&select_first_visible);
        let refresh_content_view = Rc::clone(&refresh_content_view);
        move |query: String| {
            *current_filter.borrow_mut() = query.clone();
            recompute_grid_tiers(&mut state.borrow_mut(), &query);
            flow.invalidate_filter();
            flow.invalidate_sort();
            select_first_visible();
            refresh_content_view();
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
        let launch_app = Rc::clone(&launch_app);
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
                launch_app(app);
            } else if !typed.is_empty() {
                launch_exec_or_app(typed.clone(), typed);
            }
        }
    });

    // Arrow Down from the search entry focuses the grid; Escape clears the search.
    let key_ctrl = gtk::EventControllerKey::new();
    key_ctrl.set_propagation_phase(gtk::PropagationPhase::Capture);
    key_ctrl.connect_key_pressed({
        let flow = flow.clone();
        let search_entry = search_entry.clone();
        let state = Rc::clone(&state);
        let current_filter = Rc::clone(&current_filter);
        move |_, key, _, _| {
            if key == gdk::Key::Down {
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
            } else if key == gdk::Key::Escape {
                search_entry.set_text("");
                return glib::Propagation::Stop;
            }
            glib::Propagation::Proceed
        }
    });
    search_entry.add_controller(key_ctrl);

    // Grid: deterministic spatial arrow navigation, plus typing/Backspace/Escape -> search.
    let flow_key_ctrl = gtk::EventControllerKey::new();
    flow_key_ctrl.set_propagation_phase(gtk::PropagationPhase::Capture);
    flow_key_ctrl.connect_key_pressed({
        let flow = flow.clone();
        let search_entry = search_entry.clone();
        let window_ref = window.clone();
        let state = Rc::clone(&state);
        let current_filter = Rc::clone(&current_filter);
        move |_, key, _, mods| {
            let is_arrow = matches!(
                key,
                gdk::Key::Left | gdk::Key::Right | gdk::Key::Up | gdk::Key::Down
            );
            if is_arrow {
                let needle = current_filter.borrow();
                let st = state.borrow();
                let mut visible: Vec<(gtk::FlowBoxChild, f32, f32)> = Vec::new();
                let mut cur = flow.first_child();
                while let Some(widget) = cur {
                    if let Ok(ch) = widget.clone().downcast::<gtk::FlowBoxChild>() {
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

                let focused = gtk::prelude::GtkWindowExt::focus(&window_ref);
                let cur_pos = focused
                    .as_ref()
                    .and_then(|f| visible.iter().position(|(c, _, _)| focus_within(f, c)))
                    .or_else(|| {
                        flow.selected_children()
                            .first()
                            .and_then(|sel| visible.iter().position(|(c, _, _)| c == sel))
                    })
                    .unwrap_or(0);

                let (_, cur_x, cur_y) = visible[cur_pos];
                let nearest_in_row = |row_y: f32| {
                    visible
                        .iter()
                        .filter(|(_, _, y)| (*y - row_y).abs() < 8.0)
                        .min_by(|(_, x1, _), (_, x2, _)| {
                            (x1 - cur_x).abs().total_cmp(&(x2 - cur_x).abs())
                        })
                        .map(|(c, _, _)| c.clone())
                };
                let next_child = match key {
                    gdk::Key::Left => cur_pos.checked_sub(1).map(|i| visible[i].0.clone()),
                    gdk::Key::Right => visible.get(cur_pos + 1).map(|v| v.0.clone()),
                    gdk::Key::Up => {
                        let min_y = visible
                            .iter()
                            .map(|(_, _, y)| *y)
                            .fold(f32::INFINITY, f32::min);
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
                        nearest_in_row(row_above_y)
                    }
                    gdk::Key::Down => {
                        let row_below_y = visible
                            .iter()
                            .filter(|(_, _, y)| *y > cur_y + 8.0)
                            .map(|(_, _, y)| *y)
                            .fold(f32::INFINITY, f32::min);
                        row_below_y
                            .is_finite()
                            .then(|| nearest_in_row(row_below_y))
                            .flatten()
                    }
                    _ => None,
                };

                if let Some(target) = next_child {
                    flow.select_child(&target);
                    target.grab_focus();
                }
                return glib::Propagation::Stop;
            }
            if key == gdk::Key::BackSpace || key == gdk::Key::Delete {
                let mut s = search_entry.text().to_string();
                s.pop();
                search_entry.set_text(&s);
                search_entry.grab_focus();
                search_entry.set_position(-1);
                return glib::Propagation::Stop;
            }
            if key == gdk::Key::Escape {
                search_entry.set_text("");
                search_entry.grab_focus();
                return glib::Propagation::Stop;
            }
            if !mods.intersects(gdk::ModifierType::CONTROL_MASK | gdk::ModifierType::ALT_MASK) {
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

    // Clicking or activating a grid item launches & resets.
    flow.connect_child_activated({
        let state = Rc::clone(&state);
        let launch_app = Rc::clone(&launch_app);
        move |_, child| {
            let picked = flow_child_app_idx(child)
                .and_then(|idx| state.borrow().grid_apps.get(idx).cloned());
            if let Some(app) = picked {
                launch_app(app);
            }
        }
    });

    // --- New Terminal split button ---
    let launch_terminal_with = Rc::new({
        let state = Rc::clone(&state);
        let search_entry = search_entry.clone();
        let toasts = toasts.clone();
        move |terminal: Option<&str>| {
            // The server announces the new persistent PTY back as a program.
            if let Err(err) = state.borrow().sessions.open_terminal(terminal) {
                tracing::warn!(
                    error = %err,
                    ?terminal,
                    "failed to launch local terminal for remote PTY"
                );
                show_toast(&toasts, &format!("Couldn't open a terminal: {err}"));
            }
            search_entry.grab_focus();
        }
    });

    term_split.connect_clicked({
        let launch_terminal_with = Rc::clone(&launch_terminal_with);
        move |_| launch_terminal_with(None)
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
                term_menu.append_item(&gio::MenuItem::new(
                    Some("No terminal emulators found"),
                    None,
                ));
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
                let item = gio::MenuItem::new(Some(&raw_label.replace('_', "__")), None);
                item.set_action_and_target_value(
                    Some("win.launch-terminal"),
                    Some(&term.to_variant()),
                );
                term_menu.append_item(&item);
            }
        }
    });
    // Populate up front so the PopoverMenu's items exist when keyboard activation grabs
    // focus; re-check on map and whenever the window regains focus (both are no-ops
    // unless the installed terminals changed).
    populate_term_menu();
    term_popover.connect_map({
        let populate_term_menu = Rc::clone(&populate_term_menu);
        move |pop| {
            populate_term_menu();
            let pop = pop.clone();
            glib::idle_add_local_once(move || {
                if pop.is_visible() {
                    pop.child_focus(gtk::DirectionType::TabForward);
                }
            });
        }
    });

    // --- Keyboard focus ring ---
    // Tab / Shift+Tab cycle: manage <-> servers <-> (search + grid) <-> New Terminal <->
    // terminal chooser <-> preferences <-> status bar. Stops that are hidden (collapsed
    // split view) or insensitive (no connection) are skipped.
    type FocusStop = (Box<dyn Fn(&gtk::Widget) -> bool>, Box<dyn Fn() -> bool>);
    fn widget_stop(w: impl IsA<gtk::Widget> + Clone + 'static) -> FocusStop {
        let w2 = w.clone();
        (
            Box::new(move |f| focus_within(f, &w)),
            Box::new(move || try_focus(&w2)),
        )
    }
    let focus_ring: Rc<Vec<FocusStop>> = Rc::new(vec![
        widget_stop(manage_btn.clone()),
        {
            let (a, b) = (server_list.clone(), server_list.clone());
            (
                Box::new(move |f| focus_within(f, &a)),
                Box::new(move || focus_server_list(&b)),
            )
        },
        {
            let (entry, grid) = (search_entry.clone(), flow.clone());
            let entry2 = search_entry.clone();
            (
                Box::new(move |f| focus_within(f, &entry) || focus_within(f, &grid)),
                Box::new(move || try_focus(&entry2)),
            )
        },
        widget_stop(term_btn.clone()),
        widget_stop(term_menu_btn.clone()),
        widget_stop(settings_btn.clone()),
        widget_stop(status_bar_btn.clone()),
    ]);

    let win_key_ctrl = gtk::EventControllerKey::new();
    win_key_ctrl.set_propagation_phase(gtk::PropagationPhase::Capture);
    win_key_ctrl.connect_key_pressed({
        let window_ref = window.clone();
        let focus_ring = Rc::clone(&focus_ring);
        let term_popover = term_popover.clone();
        let programs_popover = programs_popover.clone();
        move |_, key, _, mods| {
            let shift = mods.contains(gdk::ModifierType::SHIFT_MASK);
            let backward = key == gdk::Key::ISO_Left_Tab || (key == gdk::Key::Tab && shift);
            let forward = key == gdk::Key::Tab && !shift;
            if (!forward && !backward)
                || window_ref.visible_dialog().is_some()
                || term_popover.is_visible()
                || programs_popover.is_visible()
            {
                return glib::Propagation::Proceed;
            }
            let Some(focused) = gtk::prelude::GtkWindowExt::focus(&window_ref) else {
                return glib::Propagation::Proceed;
            };
            let Some(cur) = focus_ring
                .iter()
                .position(|(contains, _)| contains(&focused))
            else {
                return glib::Propagation::Proceed;
            };
            let n = focus_ring.len();
            for step in 1..n {
                let idx = if forward {
                    (cur + step) % n
                } else {
                    (cur + n - step) % n
                };
                if (focus_ring[idx].1)() {
                    break;
                }
            }
            glib::Propagation::Stop
        }
    });
    window.add_controller(win_key_ctrl);

    // Auto-focus the search box on Alt-Tab if the active server is connected, else the
    // server list. Skip while a popover or dialog is open so it doesn't steal focus back.
    window.connect_is_active_notify({
        let state = Rc::clone(&state);
        let search_entry = search_entry.clone();
        let server_list = server_list.clone();
        let term_popover = term_popover.clone();
        let programs_popover = programs_popover.clone();
        let populate_term_menu = Rc::clone(&populate_term_menu);
        move |win| {
            if !win.is_active()
                || win.visible_dialog().is_some()
                || term_popover.is_visible()
                || programs_popover.is_visible()
            {
                return;
            }
            populate_term_menu();
            let connected = state.borrow().sessions.is_active_connected();
            if !(connected && try_focus(&search_entry)) {
                focus_server_list(&server_list);
            }
        }
    });

    // --- Session events: applied on the main loop as they arrive (no polling) ---
    glib::spawn_future_local(pump_session_events(
        session_events,
        Rc::clone(&state),
        apply_changes,
    ));

    rebuild_sidebar();
    refresh_status_bar();
    window.present();
    focus_server_list(&server_list);
}

/// Feed background session events to the manager and refresh the affected views.
async fn pump_session_events(
    mut events: SessionEvents,
    state: Rc<RefCell<AppUi>>,
    apply_changes: ApplyChanges,
) {
    while let Some(event) = events.recv().await {
        let changes = state.borrow_mut().sessions.handle_event(event);
        apply_changes(changes);
    }
}

// --- Preferences dialog ---------------------------------------------------------------

fn open_preferences_dialog(window: &adw::ApplicationWindow, state: Rc<RefCell<AppUi>>) {
    let dialog = adw::PreferencesDialog::builder()
        .title("Preferences")
        .search_enabled(false)
        .build();
    let page = adw::PreferencesPage::new();

    let display_group = adw::PreferencesGroup::builder().title("Display").build();
    let scale_row = adw::ComboRow::builder()
        .title("Render Scale Cap")
        .subtitle("Upper limit for HiDPI rendering of remote windows")
        .model(&gtk::StringList::new(&SCALE_CAP_LABELS))
        .selected(state.borrow().sessions.global_max_scale().to_index() as u32)
        .build();
    scale_row.connect_selected_notify({
        let state = Rc::clone(&state);
        move |row| {
            let cap = RenderScaleCap::from_index(row.selected() as usize);
            state.borrow_mut().sessions.set_global_max_scale(cap);
        }
    });
    display_group.add(&scale_row);
    let downscale_row = adw::SwitchRow::builder()
        .title("Downscale HiDPI Frames on Server")
        .subtitle(
            "Shrink windows of apps without fractional scaling to this display's scale before \
             streaming them, saving bandwidth",
        )
        .active(state.borrow().sessions.downscale_hidpi())
        .build();
    downscale_row.connect_active_notify({
        let state = Rc::clone(&state);
        move |row| {
            state
                .borrow_mut()
                .sessions
                .set_downscale_hidpi(row.is_active())
        }
    });
    display_group.add(&downscale_row);

    // Inline radio rows (rather than a nested dropdown) so switching microphones is a
    // single click or keypress; the PipeWire source switches live.
    let mic_group = adw::PreferencesGroup::builder()
        .title("Microphone")
        .description("Local microphone that remote applications hear")
        .build();
    let mic_refresh_btn = gtk::Button::builder()
        .icon_name("view-refresh-symbolic")
        .tooltip_text("Refresh microphones")
        .valign(gtk::Align::Center)
        .css_classes(["flat"])
        .build();
    mic_group.set_header_suffix(Some(&mic_refresh_btn));

    let mic_rows: Rc<RefCell<Vec<adw::ActionRow>>> = Rc::new(RefCell::new(Vec::new()));
    let active_mic_row: Rc<RefCell<Option<adw::ActionRow>>> = Rc::new(RefCell::new(None));
    let updating_mics = Rc::new(Cell::new(false));

    let populate_mics: Rc<dyn Fn()> = Rc::new({
        let mic_group = mic_group.clone();
        let mic_rows = Rc::clone(&mic_rows);
        let active_mic_row = Rc::clone(&active_mic_row);
        let updating_mics = Rc::clone(&updating_mics);
        move || {
            updating_mics.set(true);
            for row in mic_rows.borrow_mut().drain(..) {
                mic_group.remove(&row);
            }
            *active_mic_row.borrow_mut() = None;

            let cfg = crate::audio::subscribe_mic_config().borrow().clone();
            let sources = crate::audio::list_host_mic_sources();

            // (target_node_name, is_muted, display_label)
            let mut entries: Vec<(Option<String>, bool, String)> =
                Vec::with_capacity(sources.len() + 2);
            entries.push((None, true, "Off (Muted)".to_string()));
            entries.push((None, false, "System Default".to_string()));
            for src in sources {
                entries.push((Some(src.node_name), false, src.description));
            }

            let mut group_leader: Option<gtk::CheckButton> = None;
            let mut default_row: Option<(adw::ActionRow, gtk::CheckButton)> = None;
            let mut selected_row: Option<adw::ActionRow> = None;

            for (target, is_muted, label) in entries {
                let is_selected = if is_muted {
                    cfg.muted
                } else {
                    !cfg.muted && cfg.target == target
                };

                let check = gtk::CheckButton::new();
                match &group_leader {
                    Some(leader) => check.set_group(Some(leader)),
                    None => group_leader = Some(check.clone()),
                }
                let row = adw::ActionRow::builder()
                    .title(&label)
                    .use_markup(false)
                    .activatable_widget(&check)
                    .build();
                row.add_prefix(&check);

                if !is_muted && target.is_none() {
                    default_row = Some((row.clone(), check.clone()));
                }
                if is_selected {
                    check.set_active(true);
                    selected_row = Some(row.clone());
                }

                let updating = Rc::clone(&updating_mics);
                let active_mic_row_c = Rc::clone(&active_mic_row);
                let row_c = row.clone();
                check.connect_toggled(move |b| {
                    if updating.get() || !b.is_active() {
                        return;
                    }
                    *active_mic_row_c.borrow_mut() = Some(row_c.clone());
                    if is_muted {
                        crate::audio::set_mic_muted(true);
                    } else {
                        // Set target before unmuting so we never briefly unmute the previous mic.
                        crate::audio::set_mic_target(target.clone());
                        crate::audio::set_mic_muted(false);
                    }
                });

                mic_group.add(&row);
                mic_rows.borrow_mut().push(row);
            }

            // If the configured source disappeared, fall back to System Default and clear
            // the stale target so audio::MicConfig stays in sync with the UI.
            if selected_row.is_none() {
                if let Some((row, check)) = default_row {
                    check.set_active(true);
                    if !cfg.muted {
                        crate::audio::set_mic_target(None);
                    }
                    selected_row = Some(row);
                }
            }
            *active_mic_row.borrow_mut() = selected_row;
            updating_mics.set(false);
        }
    });

    mic_refresh_btn.connect_clicked({
        let populate_mics = Rc::clone(&populate_mics);
        let active_mic_row = Rc::clone(&active_mic_row);
        move |_| {
            populate_mics();
            if let Some(row) = active_mic_row.borrow().as_ref() {
                row.grab_focus();
            }
        }
    });

    populate_mics();
    page.add(&display_group);
    page.add(&mic_group);
    dialog.add(&page);
    dialog.present(Some(window));

    // Microphone switching is the common reason to open this; start focus there.
    glib::idle_add_local_once(move || {
        if let Some(row) = active_mic_row.borrow().as_ref() {
            row.grab_focus();
        }
    });
}

// --- Manage Connections dialog ----------------------------------------------------------

#[derive(Clone)]
struct ManageCtx {
    window: adw::ApplicationWindow,
    dialog: adw::PreferencesDialog,
    saved_group: adw::PreferencesGroup,
    saved_rows: Rc<RefCell<Vec<adw::ExpanderRow>>>,
    state: Rc<RefCell<AppUi>>,
    apply_changes: ApplyChanges,
}

/// The server's UDP port or port range. Left empty, the server uses any free port.
fn ports_row(ports: Option<PortRange>, show_apply_button: bool) -> adw::EntryRow {
    let row = adw::EntryRow::builder()
        .title("UDP Port or Range (optional)")
        .text(ports.map(|p| p.to_string()).unwrap_or_default())
        .show_apply_button(show_apply_button)
        .build();
    row.connect_changed(|row| row.remove_css_class("error"));
    row
}

/// The ports entered in `row` (`None` when empty). An invalid entry is marked and
/// explained in the returned message.
fn ports_from_row(row: &adw::EntryRow) -> Result<Option<PortRange>, String> {
    let Some(text) = optional_text(row) else {
        return Ok(None);
    };
    text.parse().map(Some).map_err(|err: anyhow::Error| {
        row.add_css_class("error");
        format!("Invalid port: {err}. Use a port like 7443 or a range like 7443-7462.")
    })
}

fn host_scale_row(selected: Option<RenderScaleCap>) -> adw::ComboRow {
    adw::ComboRow::builder()
        .title("Render Scale Cap")
        .model(&gtk::StringList::new(&HOST_SCALE_CAP_LABELS))
        .selected(RenderScaleCap::option_to_index(selected) as u32)
        .build()
}

fn optional_text(row: &adw::EntryRow) -> Option<String> {
    let text = row.text().trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn open_manage_connections_dialog(ctx: ManageCtx) {
    let page = adw::PreferencesPage::new();

    // Add Connection group
    let add_group = adw::PreferencesGroup::builder()
        .title("Add Connection")
        .build();
    let add_btn = gtk::Button::builder()
        .label("Add")
        .valign(gtk::Align::Center)
        .sensitive(false)
        .css_classes(["suggested-action"])
        .build();
    add_group.set_header_suffix(Some(&add_btn));

    let add_label = adw::EntryRow::builder().title("Name (optional)").build();
    let add_target = adw::EntryRow::builder()
        .title("SSH Target (user@host)")
        .build();
    let add_port = ports_row(None, false);
    let add_bin = adw::EntryRow::builder()
        .title("Server Binary (optional)")
        .build();
    let add_scale = host_scale_row(None);
    add_group.add(&add_label);
    add_group.add(&add_target);
    add_group.add(&add_port);
    add_group.add(&add_bin);
    add_group.add(&add_scale);

    add_target.connect_changed({
        let add_btn = add_btn.clone();
        move |row| add_btn.set_sensitive(!row.text().trim().is_empty())
    });

    let add_connection: Rc<dyn Fn()> = Rc::new({
        let ctx = ctx.clone();
        let add_label = add_label.clone();
        let add_target = add_target.clone();
        let add_port = add_port.clone();
        let add_bin = add_bin.clone();
        let add_scale = add_scale.clone();
        move || {
            let target = add_target.text().trim().to_string();
            if target.is_empty() {
                return;
            }
            let ports = match ports_from_row(&add_port) {
                Ok(ports) => ports,
                Err(msg) => {
                    show_dialog_toast(&ctx.dialog, &msg);
                    return;
                }
            };
            let label = optional_text(&add_label).unwrap_or_else(|| target.clone());
            let profile = HostProfile {
                label: label.clone(),
                ssh_target: target.clone(),
                server_bin: optional_text(&add_bin),
                ports,
                audio_enabled: true,
                max_scale: RenderScaleCap::option_from_index(add_scale.selected() as usize),
            };
            let added = ctx.state.borrow_mut().sessions.add_host(profile);
            let changes = match added {
                Ok(changes) => changes,
                Err(err) => {
                    show_dialog_toast(&ctx.dialog, &err.to_string());
                    return;
                }
            };
            add_label.set_text("");
            add_target.set_text("");
            add_bin.set_text("");
            add_port.set_text("");
            add_scale.set_selected(0);
            populate_saved_connections(&ctx);
            (ctx.apply_changes)(changes);
            ctx.dialog.add_toast(
                adw::Toast::builder()
                    .title(format!("Added “{label}”"))
                    .use_markup(false)
                    .build(),
            );
            add_target.grab_focus();
        }
    });
    add_btn.connect_clicked({
        let add_connection = Rc::clone(&add_connection);
        move |_| add_connection()
    });
    add_target.connect_entry_activated({
        let add_connection = Rc::clone(&add_connection);
        move |_| add_connection()
    });

    populate_saved_connections(&ctx);
    page.add(&ctx.saved_group);
    page.add(&add_group);
    ctx.dialog.add(&page);
    ctx.dialog.present(Some(&ctx.window));

    if ctx.state.borrow().sessions.hosts().is_empty() {
        let add_target = add_target.clone();
        glib::idle_add_local_once(move || {
            add_target.grab_focus();
        });
    }
}

fn populate_saved_connections(ctx: &ManageCtx) {
    for row in ctx.saved_rows.borrow_mut().drain(..) {
        ctx.saved_group.remove(&row);
    }
    let hosts = ctx.state.borrow().sessions.hosts().to_vec();
    ctx.saved_group
        .set_description(hosts.is_empty().then_some("No saved connections yet."));
    for profile in &hosts {
        let row = saved_connection_row(ctx, profile);
        ctx.saved_group.add(&row);
        ctx.saved_rows.borrow_mut().push(row);
    }
}

/// One expandable row per saved host. Edits are saved as they are applied (Enter or
/// the entry's apply button, or a scale change).
fn saved_connection_row(ctx: &ManageCtx, profile: &HostProfile) -> adw::ExpanderRow {
    let expander = adw::ExpanderRow::builder()
        .title(&profile.label)
        .subtitle(&profile.ssh_target)
        .use_markup(false)
        .build();
    let remove_btn = gtk::Button::builder()
        .icon_name("user-trash-symbolic")
        .tooltip_text("Remove Connection")
        .valign(gtk::Align::Center)
        .css_classes(["flat"])
        .build();
    expander.add_suffix(&remove_btn);

    let label_row = adw::EntryRow::builder()
        .title("Name")
        .text(&profile.label)
        .show_apply_button(true)
        .build();
    let target_row = adw::EntryRow::builder()
        .title("SSH Target (user@host)")
        .text(&profile.ssh_target)
        .show_apply_button(true)
        .build();
    let port = ports_row(profile.ports, true);
    let bin_row = adw::EntryRow::builder()
        .title("Server Binary (optional)")
        .text(profile.server_bin.as_deref().unwrap_or(""))
        .show_apply_button(true)
        .build();
    let scale_row = host_scale_row(profile.max_scale);
    expander.add_row(&label_row);
    expander.add_row(&target_row);
    expander.add_row(&port);
    expander.add_row(&bin_row);
    expander.add_row(&scale_row);

    // The host's key in the session manager; follows SSH target edits.
    let current_target = Rc::new(RefCell::new(profile.ssh_target.clone()));

    let save: Rc<dyn Fn()> = Rc::new({
        let ctx = ctx.clone();
        let expander = expander.clone();
        let label_row = label_row.clone();
        let target_row = target_row.clone();
        let port = port.clone();
        let bin_row = bin_row.clone();
        let scale_row = scale_row.clone();
        let current_target = Rc::clone(&current_target);
        move || {
            let orig_target = current_target.borrow().clone();
            let Some(mut profile) = ctx.state.borrow().sessions.host(&orig_target).cloned() else {
                return;
            };
            profile.ssh_target = target_row.text().trim().to_string();
            profile.label = optional_text(&label_row).unwrap_or_else(|| profile.ssh_target.clone());
            profile.ports = match ports_from_row(&port) {
                Ok(ports) => ports,
                Err(msg) => {
                    show_dialog_toast(&ctx.dialog, &msg);
                    return;
                }
            };
            profile.server_bin = optional_text(&bin_row);
            profile.max_scale = RenderScaleCap::option_from_index(scale_row.selected() as usize);
            let (new_label, new_target) = (profile.label.clone(), profile.ssh_target.clone());

            let updated = ctx
                .state
                .borrow_mut()
                .sessions
                .update_host(&orig_target, profile);
            let changes = match updated {
                Ok(changes) => changes,
                Err(err) => {
                    target_row.add_css_class("error");
                    if err == HostEditError::DuplicateTarget {
                        show_dialog_toast(&ctx.dialog, &err.to_string());
                    }
                    return;
                }
            };
            target_row.remove_css_class("error");
            *current_target.borrow_mut() = new_target.clone();
            expander.set_title(&new_label);
            expander.set_subtitle(&new_target);
            (ctx.apply_changes)(changes);
        }
    });
    for entry in [&label_row, &target_row, &port, &bin_row] {
        let save = Rc::clone(&save);
        entry.connect_apply(move |_| save());
    }
    scale_row.connect_selected_notify({
        let save = Rc::clone(&save);
        move |_| save()
    });

    remove_btn.connect_clicked({
        let ctx = ctx.clone();
        let expander = expander.clone();
        let current_target = Rc::clone(&current_target);
        move |_| {
            let confirm = adw::AlertDialog::new(
                Some("Remove Connection?"),
                Some(&format!(
                    "“{}” will be disconnected and removed from your saved connections.",
                    expander.title()
                )),
            );
            confirm.add_response("cancel", "_Cancel");
            confirm.add_response("remove", "_Remove");
            confirm.set_response_appearance("remove", adw::ResponseAppearance::Destructive);
            confirm.set_default_response(Some("cancel"));
            confirm.set_close_response("cancel");
            confirm.connect_response(None, {
                let ctx = ctx.clone();
                let current_target = Rc::clone(&current_target);
                move |_, response| {
                    if response != "remove" {
                        return;
                    }
                    let target = current_target.borrow().clone();
                    let changes = ctx.state.borrow_mut().sessions.remove_host(&target);
                    populate_saved_connections(&ctx);
                    (ctx.apply_changes)(changes);
                }
            });
            confirm.present(Some(&ctx.dialog));
        }
    });

    expander
}

/// Size symbolic icons are rasterised at: 2x the 48 px grid icon, since their SVGs are
/// often drawn at 16-24 px and would blur if rendered at their own size and scaled up.
const SYMBOLIC_ICON_PX: i32 = 96;

/// Launcher grid icon for `entry`; symbolic icons are drawn in the foreground colour so
/// they stay readable on both light and dark backgrounds.
fn app_icon_paintable(entry: &RemoteAppEntry) -> Option<gdk::Paintable> {
    if entry.icon_symbolic {
        let pixbuf = load_pixbuf_sized(&entry.icon_data, SYMBOLIC_ICON_PX)?;
        let mask = gdk::Texture::for_pixbuf(&pixbuf);
        Some(SymbolicIcon::new(mask).upcast())
    } else {
        let pixbuf = load_pixbuf(&entry.icon_data)?;
        Some(gdk::Texture::for_pixbuf(&pixbuf).upcast())
    }
}

/// Decode `data`, scaling it to fit `max_px` square (keeping its aspect ratio); vector
/// formats render at that size directly.
fn load_pixbuf_sized(data: &[u8], max_px: i32) -> Option<gtk::gdk_pixbuf::Pixbuf> {
    let loader = PixbufLoader::new();
    loader.connect_size_prepared(move |loader, w, h| {
        if w > 0 && h > 0 {
            let scale = f64::from(max_px) / f64::from(w.max(h));
            let sw = ((f64::from(w) * scale).round() as i32).max(1);
            let sh = ((f64::from(h) * scale).round() as i32).max(1);
            loader.set_size(sw, sh);
        }
    });
    loader.write(data).ok()?;
    loader.close().ok()?;
    loader.pixbuf()
}

fn load_pixbuf(data: &[u8]) -> Option<gtk::gdk_pixbuf::Pixbuf> {
    let loader = PixbufLoader::new();
    loader.write(data).ok()?;
    loader.close().ok()?;
    loader.pixbuf()
}
