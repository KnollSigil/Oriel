//! Native Qt Quick / QML launcher UI powered by Qt Bridge for Rust (`qtbridge`).
//!
//! Uses system Qt Quick Controls styling (e.g., KDE Plasma's `qqc2-desktop-style` / Breeze)
//! for native palette, live dark/light theme switching, keyboard-first navigation, and
//! event-driven Qt main-thread dispatch via [`QThreadHandle`] (0 Hz idle wakeups).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use qtbridge::{qobject, QApp, QListModel, QListModelBase, QModelItem, QThreadHandle, QmlObject};
use remotewayland_proto::{ControlMessage, LaunchState, RemoteAppEntry, PTY_LAUNCH_ID_BASE};
use tokio::sync::mpsc;

use crate::hosts::{
    load_client_settings, load_hosts, parse_uri, save_client_settings, save_hosts, ssh_bootstrap,
    ClientSettings, HostProfile, RenderScaleCap,
};
use crate::pty::{
    available_local_terminals, pty_tunnel_socket_path, spawn_local_terminal_pty_socket,
    spawn_local_terminal_pty_socket_with_terminal, terminal_basename,
};
use crate::quic::{self, LauncherNetEvent};
use crate::search::{self, MatchTier, SearchIndexEntry};
use crate::window::{self, ClientUiEvent};

const DEFAULT_QUIC_PORT: u16 = 7443;
const INITIAL_RECONNECT_BACKOFF_SECS: u64 = 2;
const MAX_RECONNECT_BACKOFF_SECS: u64 = 30;
const MAX_ICON_BYTES: usize = 512 * 1024;

static SHARED_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

fn shared_runtime() -> std::result::Result<&'static tokio::runtime::Runtime, String> {
    SHARED_RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("rw-qt-net")
                .enable_all()
                .build()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

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
}

struct ConnectResult {
    pty_socket: PathBuf,
    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    ui_tx: window::UiSender,
}

struct HostSession {
    attempt_id: u64,
    cancel_connect: Arc<AtomicBool>,
    status: ConnStatus,
    error: Option<String>,
    pty_socket: Option<PathBuf>,
    ctrl_tx: Option<mpsc::UnboundedSender<ControlMessage>>,
    ui_tx: Option<window::UiSender>,
    apps: Vec<RemoteAppEntry>,
    icon_uri_cache: HashMap<String, String>,
    programs: Vec<TrackedProgram>,
    rtt_ms: Option<u32>,
    congested: bool,
    auto_reconnect: bool,
    reconnect_backoff_secs: u64,
}

// ---------------------------------------------------------------------------
// QML List Models
// ---------------------------------------------------------------------------

macro_rules! define_list_model {
    ($model:ident, $item:ident, key = $key:ident) => {
        #[derive(Default)]
        pub struct $model {
            items: Vec<$item>,
            pending: Option<Vec<$item>>,
        }

        #[qobject(Base = QListModel, NoQmlElement)]
        impl $model {}

        impl $model {
            fn update_items(&mut self, next: Vec<$item>) {
                if self.items == next {
                    return;
                }
                if self.items.len() == next.len()
                    && self
                        .items
                        .iter()
                        .zip(&next)
                        .all(|(a, b)| a.$key == b.$key)
                {
                    for (i, item) in next.into_iter().enumerate() {
                        if self.items[i] != item {
                            self.set(i, item);
                        }
                    }
                } else {
                    self.pending = Some(next);
                    self.reset();
                }
            }
        }

        impl QListModel for $model {
            type Item = $item;
            fn len(&self) -> usize {
                self.items.len()
            }
            fn get(&self, index: usize) -> Option<&Self::Item> {
                self.items.get(index)
            }
            fn set_unnotified(&mut self, index: usize, value: Self::Item) -> bool {
                if let Some(slot) = self.items.get_mut(index) {
                    *slot = value;
                    true
                } else {
                    false
                }
            }
            fn reset_unnotified(&mut self) {
                if let Some(next) = self.pending.take() {
                    self.items = next;
                }
            }
        }
    };
    ($model:ident, $item:ident) => {
        #[derive(Default)]
        pub struct $model {
            items: Vec<$item>,
            pending: Option<Vec<$item>>,
        }

        #[qobject(Base = QListModel, NoQmlElement)]
        impl $model {}

        impl $model {
            fn replace_all(&mut self, next: Vec<$item>) {
                self.pending = Some(next);
                self.reset();
            }
        }

        impl QListModel for $model {
            type Item = $item;
            fn len(&self) -> usize {
                self.items.len()
            }
            fn get(&self, index: usize) -> Option<&Self::Item> {
                self.items.get(index)
            }
            fn reset_unnotified(&mut self) {
                if let Some(next) = self.pending.take() {
                    self.items = next;
                }
            }
        }
    };
}

#[derive(QModelItem, Default, Clone, PartialEq, Eq)]
pub struct HostItem {
    pub label: String,
    pub ssh_target: String,
    pub status: String,
    pub is_active: bool,
    pub rtt_text: String,
    pub congested: bool,
    pub program_count: i32,
    pub port: i32,
    pub server_bin: String,
    pub audio_enabled: bool,
    pub max_scale_index: i32,
}
define_list_model!(HostListModel, HostItem, key = ssh_target);

#[derive(QModelItem, Default, Clone, PartialEq, Eq)]
pub struct AppGridItem {
    pub app_id: String,
    pub name: String,
    pub generic_name: String,
    pub exec: String,
    pub icon_source: String,
}
define_list_model!(AppGridModel, AppGridItem);

#[derive(QModelItem, Default, Clone, PartialEq, Eq)]
pub struct ProgramItem {
    pub launch_id: String,
    pub name: String,
    pub state: String,
    pub can_reattach: bool,
    pub term_sent: bool,
}
define_list_model!(ProgramListModel, ProgramItem, key = launch_id);

#[derive(QModelItem, Default, Clone, PartialEq, Eq)]
pub struct MicItem {
    pub label: String,
    pub target: String,
    pub is_muted: bool,
    pub is_selected: bool,
}
define_list_model!(MicListModel, MicItem);

#[derive(QModelItem, Default, Clone, PartialEq, Eq)]
pub struct TerminalItem {
    pub label: String,
    pub binary: String,
}
define_list_model!(TerminalListModel, TerminalItem);

// ---------------------------------------------------------------------------
// Main Launcher Backend QObject
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LauncherBackend {
    // Exposed QML properties
    active_target: String,
    active_label: String,
    status_text: String,
    is_connected: bool,
    is_connecting: bool,
    has_error: bool,
    error_text: String,
    rtt_text: String,
    is_congested: bool,
    search_query: String,
    is_explicit_command: bool,
    explicit_command_text: String,
    app_count: i32,
    program_count: i32,
    mic_summary: String,
    selected_mic_index: i32,
    global_max_scale_index: i32,

    // Internal Rust state
    hosts: Vec<HostProfile>,
    sessions: HashMap<String, HostSession>,
    global_max_scale: RenderScaleCap,
    cli_max_scale: Option<RenderScaleCap>,
    next_launch_id: u64,
    next_attempt_id: u64,
    grid_apps: Vec<RemoteAppEntry>,
    grid_search: Vec<SearchIndexEntry>,
    filtered_indices: Vec<usize>,
    mic_entries: Vec<(Option<String>, bool, String)>,

    // Thread-safe handle to post background events onto the Qt main thread
    qt_handle: Option<Arc<QThreadHandle<LauncherBackend>>>,

    // Shared model handles
    hosts_model: Rc<RefCell<HostListModel>>,
    apps_model: Rc<RefCell<AppGridModel>>,
    programs_model: Rc<RefCell<ProgramListModel>>,
    mics_model: Rc<RefCell<MicListModel>>,
    terminals_model: Rc<RefCell<TerminalListModel>>,
}

#[qobject(NoQmlElement)]
impl LauncherBackend {
    qproperty!("activeTarget", Member = active_target, Notify = state_changed);
    qproperty!("activeLabel", Member = active_label, Notify = state_changed);
    qproperty!("statusText", Member = status_text, Notify = state_changed);
    qproperty!("isConnected", Member = is_connected, Notify = state_changed);
    qproperty!("isConnecting", Member = is_connecting, Notify = state_changed);
    qproperty!("hasError", Member = has_error, Notify = state_changed);
    qproperty!("errorText", Member = error_text, Notify = state_changed);
    qproperty!("rttText", Member = rtt_text, Notify = net_stats_changed);
    qproperty!("isCongested", Member = is_congested, Notify = net_stats_changed);
    qproperty!("searchQuery", Member = search_query, Notify = search_changed);
    qproperty!("isExplicitCommand", Member = is_explicit_command, Notify = search_changed);
    qproperty!("explicitCommandText", Member = explicit_command_text, Notify = search_changed);
    qproperty!("appCount", Member = app_count, Notify = state_changed);
    qproperty!("programCount", Member = program_count, Notify = state_changed);
    qproperty!("micSummary", Member = mic_summary, Notify = mic_changed);
    qproperty!("selectedMicIndex", Member = selected_mic_index, Notify = mic_changed);
    qproperty!("globalMaxScaleIndex", Member = global_max_scale_index, Notify = scale_changed);

    #[qsignal]
    fn state_changed(&mut self);

    #[qsignal]
    fn net_stats_changed(&mut self);

    #[qsignal]
    fn search_changed(&mut self);

    #[qsignal]
    fn mic_changed(&mut self);

    #[qsignal]
    fn scale_changed(&mut self);

    #[qslot]
    fn select_host(&mut self, ssh_target: String) {
        if ssh_target.is_empty() {
            return;
        }
        self.active_target = ssh_target.clone();
        let should_connect = self
            .sessions
            .get(&ssh_target)
            .map(|s| s.status == ConnStatus::Disconnected)
            .unwrap_or(true);
        if should_connect {
            if let Some(profile) = self.hosts.iter().find(|h| h.ssh_target == ssh_target).cloned() {
                self.start_connect_inner(profile, false);
            }
        }
        self.rebuild_app_grid_from_active();
        self.sync_all_views();
    }

    #[qslot]
    fn select_host_at(&mut self, index: i32) {
        if let Ok(idx) = usize::try_from(index) {
            if let Some(target) = self.hosts.get(idx).map(|h| h.ssh_target.clone()) {
                self.select_host(target);
            }
        }
    }

    #[qslot]
    fn connect_host(&mut self, ssh_target: String) {
        let target = if ssh_target.is_empty() {
            self.active_target.clone()
        } else {
            ssh_target
        };
        if let Some(profile) = self.hosts.iter().find(|h| h.ssh_target == target).cloned() {
            self.active_target = target;
            self.start_connect_inner(profile, false);
            self.sync_all_views();
        }
    }

    #[qslot]
    fn disconnect_host(&mut self, ssh_target: String) {
        let target = if ssh_target.is_empty() {
            self.active_target.clone()
        } else {
            ssh_target
        };
        self.teardown_session(&target, false);
        if self.active_target == target {
            self.rebuild_app_grid_from_active();
        }
        self.sync_all_views();
    }

    #[qslot]
    fn add_or_update_host(
        &mut self,
        old_target: String,
        label: String,
        ssh_target: String,
        server_bin: String,
        port: i32,
        audio_enabled: bool,
        max_scale_index: i32,
    ) {
        let new_target = ssh_target.trim().to_string();
        if new_target.is_empty() {
            return;
        }
        if !old_target.is_empty() && old_target != new_target {
            self.teardown_session(&old_target, true);
        }
        let max_scale =
            RenderScaleCap::option_from_index(usize::try_from(max_scale_index).unwrap_or(0));
        let changed = upsert_host_profile(
            &mut self.hosts,
            &mut self.active_target,
            &old_target,
            &label,
            &new_target,
            &server_bin,
            port,
            audio_enabled,
            max_scale,
        );
        if changed {
            if let Err(err) = save_hosts(&self.hosts) {
                tracing::warn!(error = %err, "failed to save hosts.json");
            }
            if let Some(sess) = self.sessions.get(&new_target) {
                if let Some(ui_tx) = &sess.ui_tx {
                    let cap = self.effective_max_scale(max_scale);
                    let _ = ui_tx.send(ClientUiEvent::SetMaxScale { cap });
                }
            }
            self.rebuild_app_grid_from_active();
            self.sync_all_views();
        }
    }

    #[qslot]
    fn select_global_max_scale(&mut self, index: i32) {
        let idx = usize::try_from(index).unwrap_or(0);
        let cap = RenderScaleCap::from_index(idx);
        if self.global_max_scale != cap {
            self.global_max_scale = cap;
            self.global_max_scale_index = cap.to_index() as i32;
            if let Err(err) = save_client_settings(&ClientSettings { max_scale: cap }) {
                tracing::warn!(error = %err, "failed to save settings.json");
            }
            self.broadcast_max_scale();
            self.scale_changed();
        }
    }

    #[qslot]
    fn delete_host(&mut self, ssh_target: String) {
        if ssh_target.is_empty() {
            return;
        }
        self.teardown_session(&ssh_target, true);
        if remove_host_profile(&mut self.hosts, &mut self.active_target, &ssh_target) {
            self.rebuild_app_grid_from_active();
            if let Err(err) = save_hosts(&self.hosts) {
                tracing::warn!(error = %err, "failed to save hosts.json");
            }
            self.sync_all_views();
        }
    }

    #[qslot]
    fn set_search_query(&mut self, query: String) {
        self.search_query = query;
        if let Some(cmd) = search::explicit_command(&self.search_query) {
            self.is_explicit_command = true;
            self.explicit_command_text = cmd.to_string();
        } else {
            self.is_explicit_command = false;
            self.explicit_command_text.clear();
        }
        self.apply_app_filter();
        self.search_changed();
        self.state_changed();
    }

    #[qslot]
    fn launch_app_at(&mut self, filtered_index: i32) {
        let Ok(idx) = usize::try_from(filtered_index) else {
            return;
        };
        let Some(&app_idx) = self.filtered_indices.get(idx) else {
            return;
        };
        let Some(app) = self.grid_apps.get(app_idx).cloned() else {
            return;
        };
        self.launch_command(app.exec, app.name);
        if !self.search_query.is_empty() {
            self.set_search_query(String::new());
        }
    }

    #[qslot]
    fn launch_primary_action(&mut self) {
        if let Some(cmd) = search::explicit_command(&self.search_query) {
            let cmd = cmd.to_string();
            if !cmd.is_empty() {
                let label = cmd.split_whitespace().next().unwrap_or(&cmd).to_string();
                self.launch_command(cmd, label);
                self.set_search_query(String::new());
            }
            return;
        }
        if !self.filtered_indices.is_empty() {
            self.launch_app_at(0);
            return;
        }
        let custom = self.search_query.trim().to_string();
        if !custom.is_empty() {
            let label = custom
                .split_whitespace()
                .next()
                .unwrap_or(&custom)
                .to_string();
            self.launch_command(custom, label);
            self.set_search_query(String::new());
        }
    }

    #[qslot]
    fn open_terminal(&mut self, binary: String) {
        self.spawn_terminal_inner(None, &binary, "Terminal error");
    }

    #[qslot]
    fn stop_program(&mut self, launch_id_str: String) {
        let Ok(launch_id) = launch_id_str.parse::<u64>() else {
            return;
        };
        let Some(session) = self.active_session_mut() else {
            return;
        };
        if let Some(prog) = session.programs.iter_mut().find(|p| p.launch_id == launch_id) {
            if let Some(ctrl_tx) = &session.ctrl_tx {
                let force = prog.term_sent;
                if ctrl_tx
                    .send(ControlMessage::TerminateApp { launch_id, force })
                    .is_ok()
                {
                    prog.term_sent = true;
                } else {
                    session.error = Some("Connection closed while stopping application".to_string());
                }
            }
        }
        session
            .programs
            .retain(|p| !matches!(p.state, LaunchState::Exited(_)));
        self.sync_all_views();
    }

    #[qslot]
    fn stop_program_at(&mut self, index: i32) {
        let launch_id = usize::try_from(index)
            .ok()
            .and_then(|idx| self.programs_model.borrow().get(idx).map(|p| p.launch_id.clone()));
        if let Some(id) = launch_id {
            self.stop_program(id);
        }
    }

    #[qslot]
    fn reattach_pty_at(&mut self, index: i32) {
        let launch_id = usize::try_from(index).ok().and_then(|idx| {
            self.programs_model
                .borrow()
                .get(idx)
                .filter(|p| p.can_reattach)
                .map(|p| p.launch_id.clone())
        });
        if let Some(id) = launch_id {
            self.reattach_pty(id, String::new());
        }
    }

    #[qslot]
    fn reattach_pty(&mut self, launch_id_str: String, binary: String) {
        let Ok(launch_id) = launch_id_str.parse::<u64>() else {
            return;
        };
        if (launch_id & PTY_LAUNCH_ID_BASE) == 0 {
            return;
        }
        let pty_id = launch_id & !PTY_LAUNCH_ID_BASE;
        self.spawn_terminal_inner(Some(pty_id), &binary, "Terminal attach error");
    }

    #[qslot]
    fn refresh_mics(&mut self) {
        self.rebuild_mics_model();
    }

    #[qslot]
    fn select_mic(&mut self, index: i32) {
        let Ok(idx) = usize::try_from(index) else {
            return;
        };
        let Some((target, is_muted, _)) = self.mic_entries.get(idx).cloned() else {
            return;
        };
        if is_muted {
            crate::audio::set_mic_muted(true);
        } else {
            crate::audio::set_mic_target(target);
            crate::audio::set_mic_muted(false);
        }
        self.rebuild_mics_model();
    }
}

impl LauncherBackend {
    fn initialize(
        &mut self,
        qt_handle: Arc<QThreadHandle<LauncherBackend>>,
        cli_max_scale: Option<RenderScaleCap>,
        hosts_model: Rc<RefCell<HostListModel>>,
        apps_model: Rc<RefCell<AppGridModel>>,
        programs_model: Rc<RefCell<ProgramListModel>>,
        mics_model: Rc<RefCell<MicListModel>>,
        terminals_model: Rc<RefCell<TerminalListModel>>,
    ) {
        self.qt_handle = Some(qt_handle);
        self.cli_max_scale = cli_max_scale;
        self.global_max_scale = load_client_settings().max_scale;
        self.global_max_scale_index = self.global_max_scale.to_index() as i32;
        self.hosts_model = hosts_model;
        self.apps_model = apps_model;
        self.programs_model = programs_model;
        self.mics_model = mics_model;
        self.terminals_model = terminals_model;

        let seed_id = ((std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64)
            & 0xFFFF_FFFF)
            << 16
            | 1;
        self.next_launch_id = seed_id;
        self.next_attempt_id = 1;

        self.hosts = load_hosts();
        self.active_target = self
            .hosts
            .first()
            .map(|h| h.ssh_target.clone())
            .unwrap_or_default();

        let terms: Vec<TerminalItem> = available_local_terminals()
            .into_iter()
            .map(|bin| TerminalItem {
                label: terminal_basename(&bin).to_string(),
                binary: bin,
            })
            .collect();
        self.terminals_model.borrow_mut().replace_all(terms);

        self.rebuild_mics_model();
        self.scale_changed();
        self.sync_all_views();
    }

    fn effective_max_scale(&self, host_override: Option<RenderScaleCap>) -> RenderScaleCap {
        self.cli_max_scale
            .or(host_override)
            .unwrap_or(self.global_max_scale)
    }

    fn broadcast_max_scale(&self) {
        for host in &self.hosts {
            if let Some(sess) = self.sessions.get(&host.ssh_target) {
                if let Some(ui_tx) = &sess.ui_tx {
                    let cap = self.effective_max_scale(host.max_scale);
                    let _ = ui_tx.send(ClientUiEvent::SetMaxScale { cap });
                }
            }
        }
    }

    fn disconnect_all(&mut self) {
        let targets: Vec<String> = self.sessions.keys().cloned().collect();
        for target in targets {
            self.teardown_session(&target, true);
        }
    }

    fn teardown_session(&mut self, target: &str, remove_entry: bool) {
        if let Some(session) = self.sessions.get_mut(target) {
            session.cancel_connect.store(true, Ordering::Release);
            session.attempt_id = session.attempt_id.wrapping_add(1);
            session.auto_reconnect = false;
            session.status = ConnStatus::Disconnected;
            session.error = None;
            if let Some(ui_tx) = session.ui_tx.take() {
                let _ = ui_tx.send(ClientUiEvent::Disconnected);
            }
            session.ctrl_tx = None;
            session.apps.clear();
            session.icon_uri_cache.clear();
            session.programs.clear();
            session.rtt_ms = None;
            session.congested = false;
        }
        if remove_entry {
            self.sessions.remove(target);
        }
    }

    fn alloc_launch_id(&mut self) -> u64 {
        let id = self.next_launch_id & !PTY_LAUNCH_ID_BASE;
        self.next_launch_id = self.next_launch_id.wrapping_add(1).max(1) & !PTY_LAUNCH_ID_BASE;
        id.max(1)
    }

    fn alloc_attempt_id(&mut self) -> u64 {
        let id = self.next_attempt_id;
        self.next_attempt_id = self.next_attempt_id.wrapping_add(1).max(1);
        id
    }

    fn active_session(&self) -> Option<&HostSession> {
        if self.active_target.is_empty() {
            return None;
        }
        self.sessions.get(&self.active_target)
    }

    fn active_session_mut(&mut self) -> Option<&mut HostSession> {
        if self.active_target.is_empty() {
            return None;
        }
        self.sessions.get_mut(&self.active_target)
    }

    fn spawn_terminal_inner(&mut self, pty_id: Option<u64>, binary: &str, err_prefix: &str) {
        let Some(socket) = self.active_session().and_then(|s| s.pty_socket.clone()) else {
            return;
        };
        let term_opt = {
            let trimmed = binary.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        };
        let spawn_res = match term_opt {
            Some(bin) => {
                spawn_local_terminal_pty_socket_with_terminal(&socket, pty_id, None, Some(bin))
            }
            None => spawn_local_terminal_pty_socket(&socket, pty_id, None),
        };
        match spawn_res {
            Ok(mut child) => {
                // Reap the terminal launcher child on a lightweight thread so it never becomes
                // a zombie and requires zero main-thread polling.
                let _ = std::thread::Builder::new()
                    .name("rw-pty-reap".into())
                    .spawn(move || {
                        let _ = child.wait();
                    });
            }
            Err(err) => {
                tracing::error!(error = %err, ?pty_id, "{err_prefix}");
                if let Some(session) = self.active_session_mut() {
                    session.error = Some(format!("{err_prefix}: {err}"));
                }
                self.sync_all_views();
            }
        }
    }

    fn launch_command(&mut self, command: String, name: String) {
        let launch_id = self.alloc_launch_id();
        let Some(session) = self.active_session_mut() else {
            return;
        };
        let Some(ctrl_tx) = &session.ctrl_tx else {
            return;
        };
        if ctrl_tx
            .send(ControlMessage::LaunchApp {
                launch_id,
                exec: command,
                name: Some(name.clone()),
            })
            .is_ok()
        {
            session.programs.push(TrackedProgram {
                launch_id,
                name,
                state: LaunchState::Launching,
                term_sent: false,
            });
        } else {
            session.error = Some("Connection closed while launching application".to_string());
        }
        self.sync_all_views();
    }

    fn start_connect_inner(&mut self, profile: HostProfile, is_auto_reconnect: bool) {
        let Some(qt_handle) = self.qt_handle.clone() else {
            return;
        };
        let target = profile.ssh_target.clone();
        if !is_auto_reconnect {
            self.active_target = target.clone();
        }
        let (prev_backoff, prev_auto) = if let Some(existing) = self.sessions.get(&target) {
            if existing.status != ConnStatus::Disconnected {
                return;
            }
            (existing.reconnect_backoff_secs, existing.auto_reconnect)
        } else {
            (INITIAL_RECONNECT_BACKOFF_SECS, false)
        };

        let attempt_id = self.alloc_attempt_id();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.sessions.insert(
            target.clone(),
            HostSession {
                attempt_id,
                cancel_connect: Arc::clone(&cancel_flag),
                status: ConnStatus::Connecting,
                error: if is_auto_reconnect {
                    Some("Reconnecting…".to_string())
                } else {
                    None
                },
                pty_socket: None,
                ctrl_tx: None,
                ui_tx: None,
                apps: Vec::new(),
                icon_uri_cache: HashMap::new(),
                programs: Vec::new(),
                rtt_ms: None,
                congested: false,
                auto_reconnect: if is_auto_reconnect { prev_auto } else { false },
                reconnect_backoff_secs: if is_auto_reconnect {
                    prev_backoff
                } else {
                    INITIAL_RECONNECT_BACKOFF_SECS
                },
            },
        );

        let effective_max_scale = self.effective_max_scale(profile.max_scale);
        let target_bg = target.clone();
        let handle_done = Arc::clone(&qt_handle);
        std::thread::spawn(move || {
            let res = (|| -> std::result::Result<ConnectResult, String> {
                let uri = ssh_bootstrap(
                    &profile.ssh_target,
                    profile.server_bin.as_deref(),
                    profile.port,
                )
                .map_err(|e| e.to_string())?;
                if cancel_flag.load(Ordering::Acquire) {
                    return Err("Connection cancelled".to_string());
                }
                let (server, fp, token) = parse_uri(&uri).map_err(|e| e.to_string())?;
                let pty_socket =
                    pty_tunnel_socket_path(&profile.ssh_target).map_err(|e| e.to_string())?;
                let rt = shared_runtime()?;

                let (ui_tx, ui_rx) = window::create_ui_channel();
                let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
                let (net_tx, mut net_rx) = mpsc::unbounded_channel();

                if cancel_flag.load(Ordering::Acquire) {
                    return Err("Connection cancelled".to_string());
                }

                // Forward LauncherNetEvent directly onto the Qt main thread (0 Hz idle polling)
                let handle_ev = Arc::clone(&qt_handle);
                let target_ev = target_bg.clone();
                rt.spawn(async move {
                    let mut saw_disconnect = false;
                    while let Some(ev) = net_rx.recv().await {
                        if matches!(ev, LauncherNetEvent::Disconnected(_)) {
                            saw_disconnect = true;
                        }
                        let t = target_ev.clone();
                        handle_ev.invoke_mut(move |b| b.on_net_event(&t, attempt_id, ev));
                    }
                    if !saw_disconnect {
                        let t = target_ev;
                        handle_ev.invoke_mut(move |b| {
                            b.on_net_event(&t, attempt_id, LauncherNetEvent::Disconnected(None))
                        });
                    }
                });

                let ui_tx_net = ui_tx.clone();
                let pty_sock_net = pty_socket.clone();
                rt.spawn(async move {
                    if let Err(err) = quic::run_client(
                        server,
                        fp,
                        token,
                        profile.audio_enabled,
                        1.0,
                        ui_tx_net.clone(),
                        ctrl_rx,
                        Some(net_tx.clone()),
                        Some(pty_sock_net),
                    )
                    .await
                    {
                        let _ = net_tx.send(LauncherNetEvent::Disconnected(Some(err.to_string())));
                        let _ = ui_tx_net.send(ClientUiEvent::Disconnected);
                    }
                });

                let ctrl_tx_ui = ctrl_tx.clone();
                std::thread::Builder::new()
                    .name("rw-wayland-ui".into())
                    .spawn(move || {
                        let _ = window::run_event_loop(ui_rx, ctrl_tx_ui, effective_max_scale);
                    })
                    .map_err(|e| format!("failed to spawn Wayland UI thread: {e}"))?;

                let _ = ctrl_tx.send(ControlMessage::ListApps);
                Ok(ConnectResult {
                    pty_socket,
                    ctrl_tx,
                    ui_tx,
                })
            })();

            handle_done.invoke_mut(move |b| b.on_connect_finished(&target_bg, attempt_id, res));
        });
    }

    fn on_connect_finished(
        &mut self,
        target: &str,
        attempt_id: u64,
        res: std::result::Result<ConnectResult, String>,
    ) {
        let Some(session) = self.sessions.get_mut(target) else {
            if let Ok(conn) = res {
                let _ = conn.ui_tx.send(ClientUiEvent::Disconnected);
            }
            return;
        };
        if session.attempt_id != attempt_id {
            if let Ok(conn) = res {
                let _ = conn.ui_tx.send(ClientUiEvent::Disconnected);
            }
            return;
        }

        let mut schedule_delay = None;
        match res {
            Ok(conn) => {
                session.status = ConnStatus::Connected;
                session.error = None;
                session.pty_socket = Some(conn.pty_socket);
                session.ctrl_tx = Some(conn.ctrl_tx);
                session.ui_tx = Some(conn.ui_tx);
                session.auto_reconnect = true;
                session.reconnect_backoff_secs = INITIAL_RECONNECT_BACKOFF_SECS;
            }
            Err(err) => {
                session.status = ConnStatus::Disconnected;
                session.error = Some(err);
                if session.auto_reconnect {
                    let (delay, next_backoff) =
                        next_reconnect_backoff(session.reconnect_backoff_secs);
                    session.reconnect_backoff_secs = next_backoff;
                    schedule_delay = Some(delay);
                }
            }
        }

        if let Some(delay) = schedule_delay {
            self.schedule_reconnect(target, attempt_id, delay);
        }
        self.sync_all_views();
    }

    fn on_net_event(&mut self, target: &str, attempt_id: u64, ev: LauncherNetEvent) {
        let Some(session) = self.sessions.get_mut(target) else {
            return;
        };
        if session.attempt_id != attempt_id {
            return;
        }

        match ev {
            LauncherNetEvent::AppList(mut list) => {
                list.sort_by_cached_key(|a| a.name.to_lowercase());
                session.apps = list;
                if self.active_target == target {
                    self.rebuild_app_grid_from_active();
                }
                self.sync_all_views();
            }
            LauncherNetEvent::AppState {
                launch_id,
                name,
                state,
            } => {
                if matches!(state, LaunchState::Exited(_)) {
                    session.programs.retain(|p| p.launch_id != launch_id);
                } else if let Some(p) = session
                    .programs
                    .iter_mut()
                    .find(|p| p.launch_id == launch_id)
                {
                    p.state = state;
                    if !name.is_empty() {
                        p.name = name;
                    }
                } else {
                    session.programs.push(TrackedProgram {
                        launch_id,
                        name,
                        state,
                        term_sent: false,
                    });
                }
                self.sync_all_views();
            }
            LauncherNetEvent::NetStats { rtt_ms, congested } => {
                let changed = session.rtt_ms != Some(rtt_ms) || session.congested != congested;
                session.rtt_ms = Some(rtt_ms);
                session.congested = congested;
                if changed {
                    self.sync_net_stats_only();
                }
            }
            LauncherNetEvent::Disconnected(err) => {
                session.status = ConnStatus::Disconnected;
                session.error = err.or_else(|| Some("Disconnected".to_string()));
                if let Some(ui_tx) = session.ui_tx.take() {
                    let _ = ui_tx.send(ClientUiEvent::Disconnected);
                }
                session.ctrl_tx = None;
                session.rtt_ms = None;
                session.congested = false;
                session.apps.clear();
                session.icon_uri_cache.clear();
                session.programs.clear();
                let schedule_delay = if session.auto_reconnect {
                    let (delay, next_backoff) =
                        next_reconnect_backoff(session.reconnect_backoff_secs);
                    session.reconnect_backoff_secs = next_backoff;
                    Some(delay)
                } else {
                    None
                };
                if let Some(delay) = schedule_delay {
                    self.schedule_reconnect(target, attempt_id, delay);
                }
                if self.active_target == target {
                    self.rebuild_app_grid_from_active();
                }
                self.sync_all_views();
            }
        }
    }

    fn schedule_reconnect(&self, target: &str, attempt_id: u64, delay_secs: u64) {
        let Some(qt_handle) = self.qt_handle.clone() else {
            return;
        };
        let Ok(rt) = shared_runtime() else {
            return;
        };
        let target = target.to_string();
        rt.spawn(async move {
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            qt_handle.invoke_mut(move |b| b.on_reconnect_timer(&target, attempt_id));
        });
    }

    fn on_reconnect_timer(&mut self, target: &str, attempt_id: u64) {
        let should_reconnect = self
            .sessions
            .get(target)
            .is_some_and(|s| {
                s.attempt_id == attempt_id
                    && s.status == ConnStatus::Disconnected
                    && s.auto_reconnect
            });
        if !should_reconnect {
            return;
        }
        if let Some(profile) = self.hosts.iter().find(|h| h.ssh_target == target).cloned() {
            self.start_connect_inner(profile, true);
            self.sync_all_views();
        }
    }

    fn rebuild_app_grid_from_active(&mut self) {
        let apps = if let Some(sess) = self.active_session_mut() {
            for entry in &sess.apps {
                if !sess.icon_uri_cache.contains_key(&entry.id) && !entry.icon_data.is_empty() {
                    if let Some(uri) =
                        format_icon_data_uri(entry.icon_mime.as_deref(), &entry.icon_data)
                    {
                        sess.icon_uri_cache.insert(entry.id.clone(), uri);
                    }
                }
            }
            sess.apps.clone()
        } else {
            Vec::new()
        };
        self.grid_search = apps.iter().map(SearchIndexEntry::from_app).collect();
        self.grid_apps = apps;
        self.apply_app_filter();
    }

    fn apply_app_filter(&mut self) {
        let q = self.search_query.trim().to_lowercase();
        let mut ranked: Vec<(Option<MatchTier>, usize)> = if q.is_empty() {
            (0..self.grid_apps.len()).map(|i| (None, i)).collect()
        } else if search::explicit_command(&q).is_some() {
            Vec::new()
        } else {
            self.grid_search
                .iter()
                .enumerate()
                .filter_map(|(i, entry)| Some((Some(search::match_tier(entry, &q)?), i)))
                .collect()
        };

        if !q.is_empty() {
            ranked.sort_by(|(ta, ia), (tb, ib)| {
                ta.cmp(tb).then_with(|| {
                    self.grid_search[*ia]
                        .name_lower
                        .cmp(&self.grid_search[*ib].name_lower)
                })
            });
        }

        self.filtered_indices = ranked.iter().map(|(_, i)| *i).collect();
        let icon_cache = self.active_session().map(|s| &s.icon_uri_cache);
        let items: Vec<AppGridItem> = self
            .filtered_indices
            .iter()
            .map(|&i| {
                let app = &self.grid_apps[i];
                AppGridItem {
                    app_id: app.id.clone(),
                    name: app.name.clone(),
                    generic_name: app.generic_name.clone().unwrap_or_default(),
                    exec: app.exec.clone(),
                    icon_source: icon_cache
                        .and_then(|c| c.get(&app.id))
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect();
        self.app_count = i32::try_from(items.len()).unwrap_or(i32::MAX);
        self.apps_model.borrow_mut().replace_all(items);
    }

    fn rebuild_mics_model(&mut self) {
        let cfg = crate::audio::subscribe_mic_config().borrow().clone();
        let list = crate::audio::list_host_mic_sources();

        let mut entries: Vec<(Option<String>, bool, String)> = Vec::with_capacity(list.len() + 2);
        entries.push((None, true, "Off (Muted)".to_string()));
        entries.push((None, false, "System Default".to_string()));
        for src in list {
            entries.push((Some(src.node_name), false, src.description));
        }

        let mut summary = "System Default".to_string();
        let mut selected_idx: i32 = 1;
        let mut items = Vec::with_capacity(entries.len());
        for (idx, (target, is_muted, label)) in entries.iter().enumerate() {
            let is_selected = if *is_muted {
                cfg.muted
            } else if target.is_none() {
                !cfg.muted && cfg.target.is_none()
            } else {
                !cfg.muted && cfg.target == *target
            };
            if is_selected {
                summary = label.clone();
                selected_idx = i32::try_from(idx).unwrap_or(1);
            }
            items.push(MicItem {
                label: label.clone(),
                target: target.clone().unwrap_or_default(),
                is_muted: *is_muted,
                is_selected,
            });
        }

        self.mic_entries = entries;
        self.mic_summary = summary;
        self.selected_mic_index = selected_idx;
        self.mics_model.borrow_mut().replace_all(items);
        self.mic_changed();
    }

    fn build_host_items(&self) -> Vec<HostItem> {
        self.hosts
            .iter()
            .map(|h| {
                let sess = self.sessions.get(&h.ssh_target);
                let status = match sess.map(|s| s.status).unwrap_or(ConnStatus::Disconnected) {
                    ConnStatus::Connected => "connected",
                    ConnStatus::Connecting => "connecting",
                    ConnStatus::Disconnected => "disconnected",
                };
                let rtt_text = sess
                    .and_then(|s| s.rtt_ms)
                    .map(|ms| format!("{ms} ms"))
                    .unwrap_or_default();
                let congested = sess.map(|s| s.congested).unwrap_or(false);
                let program_count = sess
                    .map(|s| i32::try_from(s.programs.len()).unwrap_or(i32::MAX))
                    .unwrap_or(0);
                HostItem {
                    label: h.label.clone(),
                    ssh_target: h.ssh_target.clone(),
                    status: status.to_string(),
                    is_active: h.ssh_target == self.active_target,
                    rtt_text,
                    congested,
                    program_count,
                    port: i32::from(h.port),
                    server_bin: h.server_bin.clone().unwrap_or_default(),
                    audio_enabled: h.audio_enabled,
                    max_scale_index: RenderScaleCap::option_to_index(h.max_scale) as i32,
                }
            })
            .collect()
    }

    /// Lightweight update path for ~1 Hz RTT/congestion ticks: updates host row in-place
    /// and emits only `net_stats_changed` without re-evaluating all 12 `state_changed` bindings.
    fn sync_net_stats_only(&mut self) {
        let host_items = self.build_host_items();
        self.hosts_model.borrow_mut().update_items(host_items);

        let (rtt_text, is_congested) = self
            .active_session()
            .map(|sess| {
                (
                    sess.rtt_ms.map(|ms| format!("{ms} ms")).unwrap_or_default(),
                    sess.congested,
                )
            })
            .unwrap_or_default();

        if self.rtt_text != rtt_text || self.is_congested != is_congested {
            self.rtt_text = rtt_text;
            self.is_congested = is_congested;
            self.net_stats_changed();
        }
    }

    fn sync_all_views(&mut self) {
        let host_items = self.build_host_items();
        self.hosts_model.borrow_mut().update_items(host_items);

        self.active_label = self
            .hosts
            .iter()
            .find(|h| h.ssh_target == self.active_target)
            .map(|h| h.label.clone())
            .unwrap_or_else(|| self.active_target.clone());

        let mut progs = Vec::new();
        let mut new_rtt = String::new();
        let mut new_congested = false;

        if let Some(sess) = self.sessions.get(&self.active_target) {
            match sess.status {
                ConnStatus::Connected => {
                    self.is_connected = true;
                    self.is_connecting = false;
                    self.status_text = "Connected".to_string();
                }
                ConnStatus::Connecting => {
                    self.is_connected = false;
                    self.is_connecting = true;
                    self.status_text = "Connecting…".to_string();
                }
                ConnStatus::Disconnected => {
                    self.is_connected = false;
                    self.is_connecting = false;
                    self.status_text = "Disconnected".to_string();
                }
            }
            self.error_text = sess.error.clone().unwrap_or_default();
            new_rtt = sess.rtt_ms.map(|ms| format!("{ms} ms")).unwrap_or_default();
            new_congested = sess.congested;
            progs = sess
                .programs
                .iter()
                .map(|p| ProgramItem {
                    launch_id: p.launch_id.to_string(),
                    name: p.name.clone(),
                    state: match p.state {
                        LaunchState::Launching => "Launching".to_string(),
                        LaunchState::Running => {
                            if p.term_sent {
                                "Stopping…".to_string()
                            } else {
                                "Running".to_string()
                            }
                        }
                        LaunchState::Exited(_) => "Exited".to_string(),
                    },
                    can_reattach: is_detached_pty(p.launch_id, &p.name),
                    term_sent: p.term_sent,
                })
                .collect();
        } else {
            self.is_connected = false;
            self.is_connecting = false;
            self.status_text = if self.hosts.is_empty() {
                "No servers configured".to_string()
            } else {
                "Disconnected".to_string()
            };
            self.error_text.clear();
        }

        self.has_error = !self.error_text.is_empty();
        self.program_count = i32::try_from(progs.len()).unwrap_or(i32::MAX);
        self.programs_model.borrow_mut().update_items(progs);

        let net_changed = self.rtt_text != new_rtt || self.is_congested != new_congested;
        self.rtt_text = new_rtt;
        self.is_congested = new_congested;

        self.state_changed();
        if net_changed {
            self.net_stats_changed();
        }
    }
}

// ---------------------------------------------------------------------------
// Pure, unit-testable helper functions
// ---------------------------------------------------------------------------

fn is_detached_pty(launch_id: u64, name: &str) -> bool {
    (launch_id & PTY_LAUNCH_ID_BASE) != 0 && name.ends_with("(detached)")
}

fn next_reconnect_backoff(current_secs: u64) -> (u64, u64) {
    let delay = current_secs.clamp(INITIAL_RECONNECT_BACKOFF_SECS, MAX_RECONNECT_BACKOFF_SECS);
    let next = delay.saturating_mul(2).min(MAX_RECONNECT_BACKOFF_SECS);
    (delay, next)
}

#[allow(clippy::too_many_arguments)]
fn upsert_host_profile(
    hosts: &mut Vec<HostProfile>,
    active_target: &mut String,
    old_target: &str,
    label: &str,
    ssh_target: &str,
    server_bin: &str,
    port: i32,
    audio_enabled: bool,
    max_scale: Option<RenderScaleCap>,
) -> bool {
    let ssh_target = ssh_target.trim().to_string();
    if ssh_target.is_empty() {
        return false;
    }
    let label = {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            ssh_target.clone()
        } else {
            trimmed.to_string()
        }
    };
    let server_bin = {
        let trimmed = server_bin.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    };
    let port = u16::try_from(port)
        .ok()
        .filter(|&p| p >= 1)
        .unwrap_or(DEFAULT_QUIC_PORT);

    let profile = HostProfile {
        label,
        ssh_target: ssh_target.clone(),
        server_bin,
        port,
        audio_enabled,
        max_scale,
    };

    if !old_target.is_empty() {
        if old_target != ssh_target {
            // Avoid duplicate entries if renaming onto an already-existing target.
            hosts.retain(|h| h.ssh_target != ssh_target);
        }
        if let Some(existing) = hosts.iter_mut().find(|h| h.ssh_target == old_target) {
            *existing = profile;
        } else {
            hosts.push(profile);
        }
        if active_target == old_target {
            *active_target = ssh_target;
        }
    } else if let Some(existing) = hosts.iter_mut().find(|h| h.ssh_target == ssh_target) {
        *existing = profile;
    } else {
        hosts.push(profile);
        if active_target.is_empty() {
            *active_target = ssh_target;
        }
    }
    true
}

fn remove_host_profile(
    hosts: &mut Vec<HostProfile>,
    active_target: &mut String,
    ssh_target: &str,
) -> bool {
    let before = hosts.len();
    hosts.retain(|h| h.ssh_target != ssh_target);
    if hosts.len() == before {
        return false;
    }
    if active_target == ssh_target {
        *active_target = hosts
            .first()
            .map(|h| h.ssh_target.clone())
            .unwrap_or_default();
    }
    true
}

/// Validate remote icon MIME type and payload size before formatting a QML `data:` URI.
fn format_icon_data_uri(mime: Option<&str>, data: &[u8]) -> Option<String> {
    if data.is_empty() || data.len() > MAX_ICON_BYTES {
        return None;
    }
    let safe_mime = match mime.unwrap_or("image/png").trim() {
        "image/png" => "image/png",
        "image/svg+xml" => "image/svg+xml",
        _ => return None,
    };
    Some(format!("data:{safe_mime};base64,{}", base64_encode(data)))
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

pub fn run(cli_max_scale: Option<RenderScaleCap>) -> Result<()> {
    // Prefer KDE's native desktop style on Plasma if the user hasn't explicitly overridden it.
    if std::env::var_os("QT_QUICK_CONTROLS_STYLE").is_none()
        && std::env::var("XDG_CURRENT_DESKTOP")
            .map(|d| d.to_ascii_lowercase().contains("kde"))
            .unwrap_or(false)
    {
        std::env::set_var("QT_QUICK_CONTROLS_STYLE", "org.kde.desktop");
    }

    let mut app = QApp::new();
    app.application_name("RemoteWayland");

    let hosts_model = HostListModel::default_with_attached_qobject();
    let apps_model = AppGridModel::default_with_attached_qobject();
    let programs_model = ProgramListModel::default_with_attached_qobject();
    let mics_model = MicListModel::default_with_attached_qobject();
    let terminals_model = TerminalListModel::default_with_attached_qobject();
    let backend = LauncherBackend::default_with_attached_qobject();
    let qt_handle = Arc::new(backend.borrow().get_thread_handle());

    backend.borrow_mut().initialize(
        qt_handle,
        cli_max_scale,
        Rc::clone(&hosts_model),
        Rc::clone(&apps_model),
        Rc::clone(&programs_model),
        Rc::clone(&mics_model),
        Rc::clone(&terminals_model),
    );

    let backend_cleanup = Rc::clone(&backend);
    app.set_initial_object("backend", backend)
        .set_initial_object("hostsModel", hosts_model)
        .set_initial_object("appsModel", apps_model)
        .set_initial_object("programsModel", programs_model)
        .set_initial_object("micsModel", mics_model)
        .set_initial_object("terminalsModel", terminals_model)
        .load_qml(include_bytes!("qml/Main.qml"))
        .run();

    backend_cleanup.borrow_mut().disconnect_all();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encode_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(&[0x00, 0xFF, 0x80]), "AP+A");
    }

    #[test]
    fn format_icon_data_uri_enforces_allowlist_and_size_cap() {
        assert_eq!(
            format_icon_data_uri(None, b"abc").as_deref(),
            Some("data:image/png;base64,YWJj")
        );
        assert_eq!(
            format_icon_data_uri(Some("image/svg+xml"), b"<svg/>").as_deref(),
            Some("data:image/svg+xml;base64,PHN2Zy8+")
        );
        assert_eq!(format_icon_data_uri(Some("image/png"), b""), None);
        assert_eq!(format_icon_data_uri(Some("text/html"), b"<b>x</b>"), None);
        assert_eq!(
            format_icon_data_uri(Some("image/png;charset=utf-8"), b"abc"),
            None
        );
        let oversized = vec![0u8; MAX_ICON_BYTES + 1];
        assert_eq!(format_icon_data_uri(Some("image/png"), &oversized), None);
    }

    #[test]
    fn next_reconnect_backoff_doubles_and_caps() {
        assert_eq!(next_reconnect_backoff(0), (2, 4));
        assert_eq!(next_reconnect_backoff(2), (2, 4));
        assert_eq!(next_reconnect_backoff(4), (4, 8));
        assert_eq!(next_reconnect_backoff(16), (16, 30));
        assert_eq!(next_reconnect_backoff(30), (30, 30));
    }

    #[test]
    fn is_detached_pty_requires_bit_flag_and_suffix() {
        let pty_id = PTY_LAUNCH_ID_BASE | 42;
        assert!(is_detached_pty(pty_id, "bash (detached)"));
        assert!(!is_detached_pty(pty_id, "bash"));
        assert!(!is_detached_pty(42, "bash (detached)"));
    }

    #[test]
    fn host_profile_upsert_and_remove_handle_renames_and_dedup() {
        let mut hosts = Vec::new();
        let mut active = String::new();

        assert!(!upsert_host_profile(
            &mut hosts, &mut active, "", "Box", "   ", "", 7443, true, None
        ));
        assert!(upsert_host_profile(
            &mut hosts, &mut active, "", "", "alice@host1", "  ", 0, true, None
        ));
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].label, "alice@host1");
        assert_eq!(hosts[0].port, DEFAULT_QUIC_PORT);
        assert_eq!(active, "alice@host1");

        assert!(upsert_host_profile(
            &mut hosts,
            &mut active,
            "",
            "Box 2",
            "bob@host2",
            "/opt/rw",
            9000,
            false,
            Some(RenderScaleCap::Scale150),
        ));
        assert_eq!(hosts.len(), 2);
        assert_eq!(hosts[1].max_scale, Some(RenderScaleCap::Scale150));

        // Renaming host1 onto host2 deduplicates instead of creating two bob@host2 entries.
        assert!(upsert_host_profile(
            &mut hosts,
            &mut active,
            "alice@host1",
            "Merged",
            "bob@host2",
            "",
            8443,
            true,
            Some(RenderScaleCap::Scale200),
        ));
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].label, "Merged");
        assert_eq!(hosts[0].ssh_target, "bob@host2");
        assert_eq!(hosts[0].port, 8443);
        assert_eq!(hosts[0].max_scale, Some(RenderScaleCap::Scale200));
        assert_eq!(active, "bob@host2");

        assert!(remove_host_profile(&mut hosts, &mut active, "bob@host2"));
        assert!(hosts.is_empty());
        assert!(active.is_empty());
    }
}
