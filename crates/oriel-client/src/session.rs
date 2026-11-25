//! Toolkit-independent launcher session state.
//!
//! [`SessionManager`] owns the saved hosts and one [`HostSession`] per host: SSH bootstrap,
//! the QUIC client and Wayland window threads, launched-program tracking, network stats,
//! and auto-reconnect with exponential backoff. A GUI front-end only renders this state
//! and forwards user intents to it.
//!
//! Background work never touches the manager directly. It posts [`SessionEvent`]s to the
//! [`SessionEvents`] stream returned by [`SessionManager::new`]; the front-end awaits that
//! stream on its main thread (e.g. `glib::spawn_future_local`) and feeds each event to
//! [`SessionManager::handle_event`]. Every mutating call returns [`Changes`] describing
//! which parts of the UI need refreshing, so there is no polling.
//!
//! Each connection attempt carries an attempt id. Disconnecting, editing or removing a host
//! invalidates the current attempt, so results and events that arrive late from an
//! abandoned attempt are discarded instead of resurrecting it. The blocking SSH bootstrap
//! runs on a worker thread, but the QUIC client and window loop are only started here,
//! once the manager has accepted the worker's result, so an abandoned attempt never
//! touches the network or the host's PTY tunnel socket.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::ops::{BitOr, BitOrAssign};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, Result};
use oriel_proto::{ControlMessage, LaunchState, RemoteAppEntry, PTY_LAUNCH_ID_BASE};
use tokio::sync::mpsc;

use crate::hosts::{
    load_client_settings, load_hosts, parse_uri, save_client_settings, save_hosts, ssh_bootstrap,
    ClientSettings, HostProfile, RenderScaleCap,
};
use crate::pty::{pty_tunnel_socket_path, spawn_local_terminal_pty_socket_with_terminal};
use crate::quic::{self, LauncherNetEvent};
use crate::window::{self, ClientUiEvent};

const INITIAL_RECONNECT_BACKOFF_SECS: u64 = 2;
const MAX_RECONNECT_BACKOFF_SECS: u64 = 15;
/// Shown when another client took over the host's session.
const TAKEN_OVER_MESSAGE: &str = "Session opened by another client";

/// Tokio runtime shared by every launcher connection (QUIC clients, event forwarding,
/// reconnect timers). `quic::run_client` aborts all of its tasks, including per-stream
/// ones, when it returns.
fn shared_runtime() -> std::result::Result<&'static tokio::runtime::Runtime, String> {
    static RUNTIME: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> =
        OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("oriel-ui-net")
                .enable_all()
                .build()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnStatus {
    Disconnected,
    /// SSH bootstrap or the QUIC handshake is in progress.
    Connecting,
    /// The QUIC session is authenticated.
    Connected,
}

/// A remote program launched from (or announced to) this session.
#[derive(Debug, Clone)]
pub struct TrackedProgram {
    pub launch_id: u64,
    pub name: String,
    pub state: LaunchState,
    /// A graceful terminate was already requested; the next request force-kills.
    pub term_sent: bool,
}

impl TrackedProgram {
    /// A persistent remote PTY with no attached local terminal, which can be reattached.
    pub fn is_detached_pty(&self) -> bool {
        is_detached_pty(self.launch_id, &self.name)
    }
}

/// Live state for one saved host.
pub struct HostSession {
    pub status: ConnStatus,
    pub error: Option<String>,
    pub apps: Vec<RemoteAppEntry>,
    /// Set once the server has answered `ListApps`, so an empty list means "no apps".
    pub apps_loaded: bool,
    pub programs: Vec<TrackedProgram>,
    pub rtt_ms: Option<u32>,
    pub congested: bool,
    /// Reconnect automatically after an unexpected disconnect. Armed once a QUIC session
    /// is established; cleared by an explicit disconnect.
    pub auto_reconnect: bool,
    attempt_id: u64,
    reconnect_backoff_secs: u64,
    pty_socket: Option<PathBuf>,
    ctrl_tx: Option<mpsc::UnboundedSender<ControlMessage>>,
    ui_tx: Option<window::UiSender>,
}

impl HostSession {
    /// Whether this host can show a disconnect control (connected, connecting, or
    /// waiting to reconnect).
    pub fn wants_connection(&self) -> bool {
        self.status != ConnStatus::Disconnected || self.auto_reconnect
    }

    /// Close the connection's window loop and QUIC client (if running) and clear
    /// everything learned from it. Leaves status, error and reconnect state alone.
    fn close_connection(&mut self) {
        if let Some(ui_tx) = self.ui_tx.take() {
            let _ = ui_tx.send(ClientUiEvent::Disconnected);
        }
        self.ctrl_tx = None;
        self.pty_socket = None;
        self.apps.clear();
        self.apps_loaded = false;
        self.programs.clear();
        self.rtt_ms = None;
        self.congested = false;
    }

    /// Stop everything belonging to the current attempt and invalidate it, so late
    /// results and events from it are ignored.
    fn teardown(&mut self) {
        self.attempt_id = self.attempt_id.wrapping_add(1);
        self.close_connection();
        self.status = ConnStatus::Disconnected;
        self.error = None;
        self.auto_reconnect = false;
        self.reconnect_backoff_secs = INITIAL_RECONNECT_BACKOFF_SECS;
    }

    /// The PTY tunnel socket, once the QUIC session is up and listening on it.
    fn connected_pty_socket(&self) -> Result<&Path> {
        self.pty_socket
            .as_deref()
            .filter(|_| self.status == ConnStatus::Connected)
            .ok_or_else(|| anyhow!("not connected"))
    }
}

/// Which parts of the UI need refreshing after a [`SessionManager`] call.
#[must_use]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Changes {
    /// The host list or any host's connection status.
    pub hosts: bool,
    /// The active host's app catalog (or which host is active).
    pub apps: bool,
    /// The active host's connection status, error, or network stats.
    pub status: bool,
    /// The active host's running programs.
    pub programs: bool,
}

impl Changes {
    pub const NONE: Self = Self {
        hosts: false,
        apps: false,
        status: false,
        programs: false,
    };
    pub const ALL: Self = Self {
        hosts: true,
        apps: true,
        status: true,
        programs: true,
    };
    const HOSTS: Self = Self {
        hosts: true,
        ..Self::NONE
    };
    const APPS: Self = Self {
        apps: true,
        ..Self::NONE
    };
    const STATUS: Self = Self {
        status: true,
        ..Self::NONE
    };
    const PROGRAMS: Self = Self {
        programs: true,
        ..Self::NONE
    };

    /// `self` if it concerns the active host, otherwise nothing.
    fn if_active(self, is_active: bool) -> Self {
        if is_active {
            self
        } else {
            Self::NONE
        }
    }
}

impl BitOr for Changes {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self {
            hosts: self.hosts | rhs.hosts,
            apps: self.apps | rhs.apps,
            status: self.status | rhs.status,
            programs: self.programs | rhs.programs,
        }
    }
}

impl BitOrAssign for Changes {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// Why a host edit was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostEditError {
    EmptyTarget,
    DuplicateTarget,
}

impl std::fmt::Display for HostEditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::EmptyTarget => "The SSH target can't be empty",
            Self::DuplicateTarget => "Another connection already uses that SSH target",
        })
    }
}

impl std::error::Error for HostEditError {}

/// What the SSH bootstrap produced: everything needed to start the QUIC client.
struct Prepared {
    server: SocketAddr,
    fingerprint: String,
    token: String,
    pty_socket: PathBuf,
}

/// Handles to a started connection (QUIC client and window loop running).
struct Connection {
    pty_socket: PathBuf,
    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    ui_tx: window::UiSender,
}

/// Background notification for [`SessionManager::handle_event`]. Opaque to front-ends.
pub struct SessionEvent(EventKind);

enum EventKind {
    /// The SSH bootstrap worker finished (successfully, with an error, or by panicking).
    ConnectFinished {
        target: String,
        attempt_id: u64,
        result: std::result::Result<Prepared, String>,
    },
    Net {
        target: String,
        attempt_id: u64,
        event: LauncherNetEvent,
    },
    ReconnectDue {
        target: String,
        attempt_id: u64,
    },
}

/// Stream of [`SessionEvent`]s to await on the UI thread. Executor-agnostic.
pub struct SessionEvents(mpsc::UnboundedReceiver<SessionEvent>);

impl SessionEvents {
    pub async fn recv(&mut self) -> Option<SessionEvent> {
        self.0.recv().await
    }
}

/// Where hosts and settings are saved (swapped out in tests).
struct Persistence {
    save_hosts: fn(&[HostProfile]) -> Result<()>,
    save_settings: fn(&ClientSettings) -> Result<()>,
}

pub struct SessionManager {
    hosts: Vec<HostProfile>,
    sessions: HashMap<String, HostSession>,
    active_target: Option<String>,
    settings: ClientSettings,
    cli_max_scale: Option<RenderScaleCap>,
    next_launch_id: u64,
    next_attempt_id: u64,
    events_tx: mpsc::UnboundedSender<SessionEvent>,
    persistence: Persistence,
}

impl SessionManager {
    /// Load saved hosts and settings. The first host starts out active (not connected).
    pub fn new(cli_max_scale: Option<RenderScaleCap>) -> (Self, SessionEvents) {
        Self::with_state(
            load_hosts(),
            load_client_settings(),
            cli_max_scale,
            Persistence {
                save_hosts,
                save_settings: save_client_settings,
            },
        )
    }

    fn with_state(
        hosts: Vec<HostProfile>,
        settings: ClientSettings,
        cli_max_scale: Option<RenderScaleCap>,
        persistence: Persistence,
    ) -> (Self, SessionEvents) {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let active_target = hosts.first().map(|h| h.ssh_target.clone());
        // Time-seeded so launch ids from a restarted launcher don't collide with programs
        // the server still tracks from a previous run.
        let seed = ((std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64)
            & 0xFFFF_FFFF)
            << 16
            | 1;
        let manager = Self {
            hosts,
            sessions: HashMap::new(),
            active_target,
            settings,
            cli_max_scale,
            next_launch_id: seed,
            next_attempt_id: 1,
            events_tx,
            persistence,
        };
        (manager, SessionEvents(events_rx))
    }

    // --- Queries -----------------------------------------------------------------------

    pub fn hosts(&self) -> &[HostProfile] {
        &self.hosts
    }

    pub fn host(&self, target: &str) -> Option<&HostProfile> {
        self.hosts.iter().find(|h| h.ssh_target == target)
    }

    /// Display label for `target`, falling back to the target itself.
    pub fn host_label(&self, target: &str) -> String {
        self.host(target)
            .map(|h| h.label.clone())
            .unwrap_or_else(|| target.to_string())
    }

    pub fn active_target(&self) -> Option<&str> {
        self.active_target.as_deref()
    }

    pub fn session(&self, target: &str) -> Option<&HostSession> {
        self.sessions.get(target)
    }

    pub fn active_session(&self) -> Option<&HostSession> {
        self.sessions.get(self.active_target.as_deref()?)
    }

    pub fn is_active_connected(&self) -> bool {
        self.active_session()
            .is_some_and(|s| s.status == ConnStatus::Connected)
    }

    pub fn global_max_scale(&self) -> RenderScaleCap {
        self.settings.max_scale
    }

    pub fn downscale_hidpi(&self) -> bool {
        self.settings.downscale_hidpi
    }

    fn effective_max_scale(&self, host_override: Option<RenderScaleCap>) -> RenderScaleCap {
        self.cli_max_scale
            .or(host_override)
            .unwrap_or(self.settings.max_scale)
    }

    fn is_active(&self, target: &str) -> bool {
        self.active_target.as_deref() == Some(target)
    }

    // --- Connections -------------------------------------------------------------------

    /// Make `target` the active host and connect to it unless it is already connected or
    /// connecting. A pending auto-reconnect is replaced by an immediate attempt.
    pub fn connect(&mut self, target: &str) -> Changes {
        let Some(profile) = self.host(target).cloned() else {
            return Changes::NONE;
        };
        self.active_target = Some(target.to_string());
        let _ = self.start_connect(profile, false);
        Changes::ALL
    }

    /// Disconnect `target` and stop auto-reconnecting it.
    pub fn disconnect(&mut self, target: &str) -> Changes {
        let Some(sess) = self.sessions.get_mut(target) else {
            return Changes::NONE;
        };
        sess.teardown();
        Changes::HOSTS | Changes::ALL.if_active(self.is_active(target))
    }

    /// Start an SSH bootstrap attempt for `profile` on a worker thread.
    fn start_connect(&mut self, profile: HostProfile, is_auto_reconnect: bool) -> Changes {
        let target = profile.ssh_target.clone();
        let Some(attempt_id) = self.begin_attempt(&target, is_auto_reconnect) else {
            return Changes::NONE;
        };
        let changes = Changes::HOSTS | Changes::ALL.if_active(self.is_active(&target));
        let events_tx = self.events_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("oriel-ssh".into())
            .spawn(move || connect_worker(profile, attempt_id, events_tx));
        match spawned {
            Ok(_) => changes,
            Err(err) => {
                // The closure (and its events sender) is gone; fail the attempt directly,
                // which also keeps auto-reconnect going.
                let err = format!("failed to spawn connection thread: {err}");
                changes | self.fail_attempt(&target, attempt_id, Some(err))
            }
        }
    }

    /// Replace `target`'s session with a fresh `Connecting` one and return its attempt id.
    /// Returns `None` if it is already connected or connecting.
    fn begin_attempt(&mut self, target: &str, is_auto_reconnect: bool) -> Option<u64> {
        let (prev_backoff, prev_auto) = match self.sessions.get(target) {
            Some(existing) if existing.status != ConnStatus::Disconnected => return None,
            Some(existing) => (existing.reconnect_backoff_secs, existing.auto_reconnect),
            None => (INITIAL_RECONNECT_BACKOFF_SECS, false),
        };
        let attempt_id = self.next_attempt_id;
        self.next_attempt_id = self.next_attempt_id.wrapping_add(1).max(1);
        self.sessions.insert(
            target.to_string(),
            HostSession {
                status: ConnStatus::Connecting,
                error: is_auto_reconnect.then(|| "Reconnecting…".to_string()),
                apps: Vec::new(),
                apps_loaded: false,
                programs: Vec::new(),
                rtt_ms: None,
                congested: false,
                auto_reconnect: is_auto_reconnect && prev_auto,
                attempt_id,
                reconnect_backoff_secs: if is_auto_reconnect {
                    prev_backoff
                } else {
                    INITIAL_RECONNECT_BACKOFF_SECS
                },
                pty_socket: None,
                ctrl_tx: None,
                ui_tx: None,
            },
        );
        Some(attempt_id)
    }

    /// Apply a background event. Events from superseded attempts are discarded.
    pub fn handle_event(&mut self, event: SessionEvent) -> Changes {
        match event.0 {
            EventKind::ConnectFinished {
                target,
                attempt_id,
                result,
            } => self.on_connect_finished(&target, attempt_id, result),
            EventKind::Net {
                target,
                attempt_id,
                event,
            } => self.on_net_event(&target, attempt_id, event),
            EventKind::ReconnectDue { target, attempt_id } => {
                self.on_reconnect_due(&target, attempt_id)
            }
        }
    }

    fn current_session_mut(&mut self, target: &str, attempt_id: u64) -> Option<&mut HostSession> {
        self.sessions
            .get_mut(target)
            .filter(|s| s.attempt_id == attempt_id)
    }

    /// The SSH bootstrap finished. A stale result is dropped before anything is started;
    /// otherwise the QUIC client and window loop start and the session stays `Connecting`
    /// until the handshake succeeds ([`LauncherNetEvent::Connected`]).
    fn on_connect_finished(
        &mut self,
        target: &str,
        attempt_id: u64,
        result: std::result::Result<Prepared, String>,
    ) -> Changes {
        if self.current_session_mut(target, attempt_id).is_none() {
            return Changes::NONE;
        }
        match result.and_then(|prepared| self.start_connection(target, attempt_id, prepared)) {
            Ok(conn) => self.adopt_connection(target, attempt_id, conn),
            Err(err) => self.fail_attempt(target, attempt_id, Some(err)),
        }
    }

    /// Attach a started connection to its (still `Connecting`) session.
    fn adopt_connection(&mut self, target: &str, attempt_id: u64, conn: Connection) -> Changes {
        if let Some(sess) = self.current_session_mut(target, attempt_id) {
            sess.pty_socket = Some(conn.pty_socket);
            sess.ctrl_tx = Some(conn.ctrl_tx);
            sess.ui_tx = Some(conn.ui_tx);
        }
        Changes::NONE
    }

    /// Start the window loop and QUIC client for an accepted SSH bootstrap.
    fn start_connection(
        &self,
        target: &str,
        attempt_id: u64,
        prepared: Prepared,
    ) -> std::result::Result<Connection, String> {
        let rt = shared_runtime()?;
        let host = self.host(target);
        let max_scale = self.effective_max_scale(host.and_then(|h| h.max_scale));
        let audio_enabled = host.is_none_or(|h| h.audio_enabled);

        let (ui_tx, ui_rx) = window::create_ui_channel();
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let (net_tx, net_rx) = mpsc::unbounded_channel();

        let ctrl_tx_ui = ctrl_tx.clone();
        std::thread::Builder::new()
            .name("oriel-ui".into())
            .spawn(move || {
                if let Err(err) = window::run_event_loop(ui_rx, ctrl_tx_ui, max_scale) {
                    tracing::error!(error = %err, "Wayland UI loop exited");
                }
            })
            .map_err(|err| format!("failed to spawn Wayland UI thread: {err}"))?;

        let _ = ctrl_tx.send(ControlMessage::ServerDownscale {
            enabled: self.settings.downscale_hidpi,
        });
        let _ = ctrl_tx.send(ControlMessage::ListApps);
        rt.spawn(forward_net_events(
            target.to_string(),
            attempt_id,
            net_rx,
            self.events_tx.clone(),
        ));
        let Prepared {
            server,
            fingerprint,
            token,
            pty_socket,
        } = prepared;
        let ui_tx_net = ui_tx.clone();
        let pty_socket_net = pty_socket.clone();
        rt.spawn(async move {
            if let Err(err) = quic::run_client(
                server,
                fingerprint,
                token,
                audio_enabled,
                1.0,
                ui_tx_net.clone(),
                ctrl_rx,
                Some(net_tx.clone()),
                Some(pty_socket_net),
            )
            .await
            {
                let event = if err.is::<quic::SessionTakenOver>() {
                    LauncherNetEvent::TakenOver
                } else {
                    LauncherNetEvent::Disconnected(Some(err.to_string()))
                };
                let _ = net_tx.send(event);
                let _ = ui_tx_net.send(ClientUiEvent::Disconnected);
            }
        });
        Ok(Connection {
            pty_socket,
            ctrl_tx,
            ui_tx,
        })
    }

    /// The current attempt failed or its connection was lost: close it, then either
    /// schedule a reconnect (keeping `err` visible) or show `err`.
    fn fail_attempt(&mut self, target: &str, attempt_id: u64, err: Option<String>) -> Changes {
        let is_active = self.is_active(target);
        let Some(sess) = self.current_session_mut(target, attempt_id) else {
            return Changes::NONE;
        };
        sess.close_connection();
        sess.status = ConnStatus::Disconnected;
        if sess.auto_reconnect {
            let delay = arm_reconnect(sess, err.as_deref());
            self.schedule_reconnect(target, attempt_id, delay);
        } else {
            sess.error = err;
        }
        Changes::HOSTS | Changes::ALL.if_active(is_active)
    }

    fn on_net_event(&mut self, target: &str, attempt_id: u64, event: LauncherNetEvent) -> Changes {
        let is_active = self.is_active(target);
        let Some(sess) = self.current_session_mut(target, attempt_id) else {
            return Changes::NONE;
        };
        match event {
            LauncherNetEvent::Connected => {
                sess.status = ConnStatus::Connected;
                sess.error = None;
                sess.auto_reconnect = true;
                sess.reconnect_backoff_secs = INITIAL_RECONNECT_BACKOFF_SECS;
                return Changes::HOSTS | Changes::ALL.if_active(is_active);
            }
            LauncherNetEvent::Disconnected(err) => {
                return self.fail_attempt(target, attempt_id, err);
            }
            LauncherNetEvent::TakenOver => {
                // Reconnecting would take the session back and start a tug of war.
                sess.auto_reconnect = false;
                return self.fail_attempt(target, attempt_id, Some(TAKEN_OVER_MESSAGE.into()));
            }
            _ if sess.status != ConnStatus::Connected => return Changes::NONE,
            _ => {}
        }
        match event {
            LauncherNetEvent::AppList(apps) => {
                sess.apps = apps;
                sess.apps_loaded = true;
                Changes::APPS.if_active(is_active)
            }
            LauncherNetEvent::AppState {
                launch_id,
                name,
                state,
            } => {
                track_program_state(&mut sess.programs, launch_id, name, state);
                Changes::PROGRAMS.if_active(is_active)
            }
            LauncherNetEvent::NetStats { rtt_ms, congested } => {
                let changed = sess.rtt_ms != Some(rtt_ms) || sess.congested != congested;
                sess.rtt_ms = Some(rtt_ms);
                sess.congested = congested;
                Changes::STATUS.if_active(is_active && changed)
            }
            LauncherNetEvent::Connected
            | LauncherNetEvent::Disconnected(_)
            | LauncherNetEvent::TakenOver => Changes::NONE,
        }
    }

    fn schedule_reconnect(&self, target: &str, attempt_id: u64, delay_secs: u64) {
        let rt = match shared_runtime() {
            Ok(rt) => rt,
            Err(err) => {
                tracing::warn!(%err, "cannot schedule reconnect");
                return;
            }
        };
        let events_tx = self.events_tx.clone();
        let target = target.to_string();
        rt.spawn(async move {
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            let _ = events_tx.send(SessionEvent(EventKind::ReconnectDue { target, attempt_id }));
        });
    }

    fn on_reconnect_due(&mut self, target: &str, attempt_id: u64) -> Changes {
        let due = self
            .current_session_mut(target, attempt_id)
            .is_some_and(|s| s.status == ConnStatus::Disconnected && s.auto_reconnect);
        let Some(profile) = due.then(|| self.host(target).cloned()).flatten() else {
            return Changes::NONE;
        };
        self.start_connect(profile, true)
    }

    // --- Programs ----------------------------------------------------------------------

    fn alloc_launch_id(&mut self) -> u64 {
        // Keep the PTY bit clear so GUI launch ids never look like persistent PTYs.
        let id = (self.next_launch_id & !PTY_LAUNCH_ID_BASE).max(1);
        self.next_launch_id = self.next_launch_id.wrapping_add(1);
        id
    }

    /// Launch `exec` on the active host, tracked under the display `name`.
    pub fn launch(&mut self, name: String, exec: String) -> Changes {
        let launch_id = self.alloc_launch_id();
        let is_active_connected = self.is_active_connected();
        let Some(sess) = self
            .active_target
            .as_deref()
            .and_then(|t| self.sessions.get_mut(t))
        else {
            return Changes::NONE;
        };
        let Some(ctrl_tx) = sess.ctrl_tx.as_ref().filter(|_| is_active_connected) else {
            return Changes::NONE;
        };
        let sent = ctrl_tx.send(ControlMessage::LaunchApp {
            launch_id,
            name: Some(name.clone()),
            exec,
        });
        if sent.is_err() {
            return Changes::NONE;
        }
        sess.programs.push(TrackedProgram {
            launch_id,
            name,
            state: LaunchState::Launching,
            term_sent: false,
        });
        Changes::PROGRAMS
    }

    /// Ask `target` to close a program: gracefully the first time, then force-kill.
    /// Returns `true` if a request was sent.
    pub fn terminate(&mut self, target: &str, launch_id: u64) -> bool {
        let Some(sess) = self.sessions.get_mut(target) else {
            return false;
        };
        let Some(ctrl_tx) = &sess.ctrl_tx else {
            return false;
        };
        let Some(prog) = sess.programs.iter_mut().find(|p| p.launch_id == launch_id) else {
            return false;
        };
        let force = prog.term_sent;
        if ctrl_tx
            .send(ControlMessage::TerminateApp { launch_id, force })
            .is_err()
        {
            return false;
        }
        prog.term_sent = true;
        true
    }

    /// Open a new remote shell on the active host in a local terminal emulator (`None`
    /// picks the default). The server announces the PTY back as a program.
    pub fn open_terminal(&self, terminal: Option<&str>) -> Result<()> {
        let sess = self
            .active_session()
            .ok_or_else(|| anyhow!("not connected"))?;
        spawn_terminal(sess.connected_pty_socket()?, None, None, terminal)
    }

    /// Run `exec` (a `Terminal=true` app) on the active host in a new remote PTY shown in
    /// the default local terminal emulator. The server announces the PTY back as a program.
    pub fn launch_in_terminal(&self, exec: &str) -> Result<()> {
        let sess = self
            .active_session()
            .ok_or_else(|| anyhow!("not connected"))?;
        spawn_terminal(sess.connected_pty_socket()?, None, Some(exec), None)
    }

    /// Reattach a detached persistent PTY of `target` in the default terminal emulator.
    pub fn reattach_terminal(&self, target: &str, launch_id: u64) -> Result<()> {
        let sess = self
            .session(target)
            .ok_or_else(|| anyhow!("not connected"))?;
        spawn_terminal(
            sess.connected_pty_socket()?,
            Some(launch_id & !PTY_LAUNCH_ID_BASE),
            None,
            None,
        )
    }

    // --- Settings & saved hosts --------------------------------------------------------

    /// Change the global render scale cap, save it, and apply it to open connections.
    pub fn set_global_max_scale(&mut self, cap: RenderScaleCap) {
        if self.settings.max_scale == cap {
            return;
        }
        self.settings.max_scale = cap;
        self.save_settings();
        for host in &self.hosts {
            self.send_max_scale(host);
        }
    }

    /// Turn server-side downscaling of HiDPI frames on or off, save it, and apply it to open
    /// connections.
    pub fn set_downscale_hidpi(&mut self, enabled: bool) {
        if self.settings.downscale_hidpi == enabled {
            return;
        }
        self.settings.downscale_hidpi = enabled;
        self.save_settings();
        for sess in self.sessions.values() {
            if let Some(ctrl_tx) = &sess.ctrl_tx {
                let _ = ctrl_tx.send(ControlMessage::ServerDownscale { enabled });
            }
        }
    }

    fn save_settings(&self) {
        if let Err(err) = (self.persistence.save_settings)(&self.settings) {
            tracing::warn!(error = %err, "failed to save client settings");
        }
    }

    fn send_max_scale(&self, host: &HostProfile) {
        let ui_tx = self
            .sessions
            .get(&host.ssh_target)
            .and_then(|s| s.ui_tx.as_ref());
        if let Some(ui_tx) = ui_tx {
            let cap = self.effective_max_scale(host.max_scale);
            let _ = ui_tx.send(ClientUiEvent::SetMaxScale { cap });
        }
    }

    fn save_hosts(&self) {
        if let Err(err) = (self.persistence.save_hosts)(&self.hosts) {
            tracing::warn!(error = %err, "failed to save hosts");
        }
    }

    /// Save a new host; an existing host with the same SSH target is replaced.
    pub fn add_host(
        &mut self,
        profile: HostProfile,
    ) -> std::result::Result<Changes, HostEditError> {
        if profile.ssh_target.trim().is_empty() {
            return Err(HostEditError::EmptyTarget);
        }
        self.hosts.retain(|h| h.ssh_target != profile.ssh_target);
        self.hosts.push(profile);
        self.save_hosts();
        Ok(Changes::HOSTS | Changes::STATUS)
    }

    /// Replace the saved host `orig_target` with `profile`. Changing the SSH target
    /// disconnects the old session; other edits apply to the live session.
    pub fn update_host(
        &mut self,
        orig_target: &str,
        profile: HostProfile,
    ) -> std::result::Result<Changes, HostEditError> {
        let new_target = profile.ssh_target.clone();
        if new_target.trim().is_empty() {
            return Err(HostEditError::EmptyTarget);
        }
        let target_changed = new_target != orig_target;
        if target_changed && self.host(&new_target).is_some() {
            return Err(HostEditError::DuplicateTarget);
        }
        let Some(slot) = self.hosts.iter_mut().find(|h| h.ssh_target == orig_target) else {
            return Ok(Changes::NONE);
        };
        *slot = profile;

        let mut changes = Changes::HOSTS | Changes::STATUS;
        if target_changed {
            if let Some(mut sess) = self.sessions.remove(orig_target) {
                sess.teardown();
            }
            if self.is_active(orig_target) {
                self.active_target = Some(new_target);
                changes = Changes::ALL;
            }
        } else if let Some(host) = self.host(&new_target) {
            self.send_max_scale(host);
        }
        self.save_hosts();
        Ok(changes)
    }

    /// Disconnect and forget `target`. If it was active, the first remaining host becomes
    /// active.
    pub fn remove_host(&mut self, target: &str) -> Changes {
        let before = self.hosts.len();
        self.hosts.retain(|h| h.ssh_target != target);
        if self.hosts.len() == before {
            return Changes::NONE;
        }
        self.save_hosts();
        if let Some(mut sess) = self.sessions.remove(target) {
            sess.teardown();
        }
        if self.is_active(target) {
            self.active_target = self.hosts.first().map(|h| h.ssh_target.clone());
        }
        Changes::ALL
    }
}

/// SSH-bootstrap the server and report what is needed to connect. Runs on its own thread
/// because `ssh_bootstrap` blocks. Always reports back, even if bootstrapping panics, so
/// the session can't be left `Connecting` forever.
fn connect_worker(
    profile: HostProfile,
    attempt_id: u64,
    events_tx: mpsc::UnboundedSender<SessionEvent>,
) {
    let result = catch_panic(|| prepare_connection(&profile));
    let _ = events_tx.send(SessionEvent(EventKind::ConnectFinished {
        target: profile.ssh_target,
        attempt_id,
        result,
    }));
}

fn prepare_connection(profile: &HostProfile) -> std::result::Result<Prepared, String> {
    let uri = ssh_bootstrap(
        &profile.ssh_target,
        profile.server_bin.as_deref(),
        profile.ports,
    )
    .map_err(|e| e.to_string())?;
    let (server, fingerprint, token) = parse_uri(&uri).map_err(|e| e.to_string())?;
    let pty_socket = pty_tunnel_socket_path(&profile.ssh_target).map_err(|e| e.to_string())?;
    Ok(Prepared {
        server,
        fingerprint,
        token,
        pty_socket,
    })
}

/// Run `f`, turning a panic into an error.
fn catch_panic<T>(
    f: impl FnOnce() -> std::result::Result<T, String>,
) -> std::result::Result<T, String> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        Err(format!("internal error while connecting: {msg}"))
    })
}

/// Tag a connection's network events with its attempt and forward them to the manager.
/// `run_client` may report several `Disconnected`s while shutting down (a bare one, then
/// the error); they are coalesced into one, sent once the connection is fully gone. A
/// `TakenOver` among them wins, so the manager knows not to reconnect.
async fn forward_net_events(
    target: String,
    attempt_id: u64,
    mut net_rx: mpsc::UnboundedReceiver<LauncherNetEvent>,
    events_tx: mpsc::UnboundedSender<SessionEvent>,
) {
    let forward = |event| {
        let _ = events_tx.send(SessionEvent(EventKind::Net {
            target: target.clone(),
            attempt_id,
            event,
        }));
    };
    let mut disconnected: Option<Option<String>> = None;
    let mut taken_over = false;
    while let Some(event) = net_rx.recv().await {
        match event {
            LauncherNetEvent::Disconnected(err) => {
                let prev = disconnected.take().flatten();
                disconnected = Some(err.or(prev));
            }
            LauncherNetEvent::TakenOver => {
                taken_over = true;
                disconnected.get_or_insert(None);
            }
            event if disconnected.is_none() => forward(event),
            _ => {}
        }
    }
    if taken_over {
        forward(LauncherNetEvent::TakenOver);
    } else {
        forward(LauncherNetEvent::Disconnected(disconnected.flatten()));
    }
}

/// Mark `sess` as waiting to reconnect after `cause`, and return the delay in seconds.
fn arm_reconnect(sess: &mut HostSession, cause: Option<&str>) -> u64 {
    let (delay, next) = next_reconnect_backoff(sess.reconnect_backoff_secs);
    sess.reconnect_backoff_secs = next;
    sess.error = Some(reconnect_message(delay, cause));
    delay
}

fn reconnect_message(delay_secs: u64, cause: Option<&str>) -> String {
    match cause.map(str::trim).filter(|c| !c.is_empty()) {
        Some(cause) => format!("Reconnecting in {delay_secs}s… Last error: {cause}"),
        None => format!("Reconnecting in {delay_secs}s…"),
    }
}

/// `(delay for this attempt, backoff for the next)`: doubling, capped.
fn next_reconnect_backoff(current_secs: u64) -> (u64, u64) {
    let delay = current_secs.clamp(INITIAL_RECONNECT_BACKOFF_SECS, MAX_RECONNECT_BACKOFF_SECS);
    let next = delay.saturating_mul(2).min(MAX_RECONNECT_BACKOFF_SECS);
    (delay, next)
}

fn is_detached_pty(launch_id: u64, name: &str) -> bool {
    (launch_id & PTY_LAUNCH_ID_BASE) != 0 && name.ends_with("(detached)")
}

fn track_program_state(
    programs: &mut Vec<TrackedProgram>,
    launch_id: u64,
    name: String,
    state: LaunchState,
) {
    if matches!(state, LaunchState::Exited(_)) {
        programs.retain(|p| p.launch_id != launch_id);
    } else if let Some(p) = programs.iter_mut().find(|p| p.launch_id == launch_id) {
        p.state = state;
        if !name.is_empty() {
            p.name = name;
        }
    } else {
        programs.push(TrackedProgram {
            launch_id,
            name,
            state,
            term_sent: false,
        });
    }
}

/// Spawn a local terminal attached to a remote PTY and reap it in the background.
fn spawn_terminal(
    socket: &Path,
    pty_id: Option<u64>,
    command: Option<&str>,
    terminal: Option<&str>,
) -> Result<()> {
    let mut child =
        spawn_local_terminal_pty_socket_with_terminal(socket, pty_id, command, terminal)?;
    std::thread::Builder::new()
        .name("oriel-pty-reap".into())
        .spawn(move || {
            let _ = child.wait();
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(target: &str) -> HostProfile {
        HostProfile {
            label: target.to_string(),
            ssh_target: target.to_string(),
            server_bin: None,
            ports: None,
            audio_enabled: true,
            max_scale: None,
        }
    }

    fn manager(targets: &[&str]) -> (SessionManager, SessionEvents) {
        SessionManager::with_state(
            targets.iter().map(|t| profile(t)).collect(),
            ClientSettings::default(),
            None,
            Persistence {
                save_hosts: |_| Ok(()),
                save_settings: |_| Ok(()),
            },
        )
    }

    fn connection() -> (Connection, mpsc::UnboundedReceiver<ControlMessage>) {
        let (ui_tx, _ui_rx) = window::create_ui_channel();
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let conn = Connection {
            pty_socket: PathBuf::from("/nonexistent/pty.sock"),
            ctrl_tx,
            ui_tx,
        };
        (conn, ctrl_rx)
    }

    fn prepared() -> Prepared {
        Prepared {
            server: "127.0.0.1:7443".parse().unwrap(),
            fingerprint: "00".into(),
            token: "t".into(),
            pty_socket: PathBuf::from("/nonexistent/pty.sock"),
        }
    }

    fn finished(
        target: &str,
        attempt_id: u64,
        result: std::result::Result<Prepared, String>,
    ) -> SessionEvent {
        SessionEvent(EventKind::ConnectFinished {
            target: target.to_string(),
            attempt_id,
            result,
        })
    }

    fn net(target: &str, attempt_id: u64, event: LauncherNetEvent) -> SessionEvent {
        SessionEvent(EventKind::Net {
            target: target.to_string(),
            attempt_id,
            event,
        })
    }

    /// Start an attempt and attach fake connection handles, without SSH, QUIC or a window
    /// thread. The session stays `Connecting` until the handshake event.
    fn started(
        mgr: &mut SessionManager,
        target: &str,
        is_auto_reconnect: bool,
    ) -> (u64, mpsc::UnboundedReceiver<ControlMessage>) {
        let attempt = mgr.begin_attempt(target, is_auto_reconnect).unwrap();
        let (conn, ctrl_rx) = connection();
        let _ = mgr.adopt_connection(target, attempt, conn);
        (attempt, ctrl_rx)
    }

    /// [`started`], then a successful QUIC handshake.
    fn connected(
        mgr: &mut SessionManager,
        target: &str,
    ) -> (u64, mpsc::UnboundedReceiver<ControlMessage>) {
        let (attempt, ctrl_rx) = started(mgr, target, false);
        let _ = mgr.handle_event(net(target, attempt, LauncherNetEvent::Connected));
        (attempt, ctrl_rx)
    }

    #[test]
    fn downscale_setting_applies_to_open_connections() {
        let (mut mgr, _events) = manager(&["a", "b"]);
        assert!(mgr.downscale_hidpi());
        let (_, mut rx_a) = connected(&mut mgr, "a");
        let (_, mut rx_b) = connected(&mut mgr, "b");
        while rx_a.try_recv().is_ok() {}
        while rx_b.try_recv().is_ok() {}

        mgr.set_downscale_hidpi(false);
        assert!(!mgr.downscale_hidpi());
        for rx in [&mut rx_a, &mut rx_b] {
            assert_eq!(
                rx.try_recv().ok(),
                Some(ControlMessage::ServerDownscale { enabled: false })
            );
        }
        // Unchanged: nothing more is sent.
        mgr.set_downscale_hidpi(false);
        assert!(rx_a.try_recv().is_err());
    }

    #[test]
    fn reconnect_backoff_doubles_and_caps() {
        assert_eq!(next_reconnect_backoff(0), (2, 4));
        assert_eq!(next_reconnect_backoff(2), (2, 4));
        assert_eq!(next_reconnect_backoff(4), (4, 8));
        assert_eq!(next_reconnect_backoff(8), (8, 15));
        assert_eq!(next_reconnect_backoff(15), (15, 15));
    }

    #[test]
    fn detached_pty_requires_bit_flag_and_suffix() {
        let pty_id = PTY_LAUNCH_ID_BASE | 42;
        assert!(is_detached_pty(pty_id, "bash (detached)"));
        assert!(!is_detached_pty(pty_id, "bash"));
        assert!(!is_detached_pty(42, "bash (detached)"));
    }

    #[test]
    fn launch_ids_never_set_the_pty_bit() {
        let (mut mgr, _events) = manager(&[]);
        mgr.next_launch_id = PTY_LAUNCH_ID_BASE | 7;
        assert_eq!(mgr.alloc_launch_id(), 7);
        assert_eq!(mgr.alloc_launch_id(), 8);
    }

    #[test]
    fn quic_handshake_arms_auto_reconnect() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, _ctrl_rx) = started(&mut mgr, "a@h", false);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Connecting);
        assert!(!sess.auto_reconnect);
        // The PTY tunnel socket isn't bound before the handshake.
        assert!(mgr.open_terminal(None).is_err());

        let changes = mgr.handle_event(net("a@h", attempt, LauncherNetEvent::Connected));
        assert_eq!(changes, Changes::ALL);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Connected);
        assert!(sess.auto_reconnect);
        // A second attempt is refused while connected.
        assert!(mgr.begin_attempt("a@h", false).is_none());
    }

    #[test]
    fn failed_manual_connect_reports_error_without_reconnecting() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let attempt = mgr.begin_attempt("a@h", false).unwrap();
        let changes = mgr.handle_event(finished("a@h", attempt, Err("ssh: no route".into())));
        assert_eq!(changes, Changes::ALL);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert_eq!(sess.error.as_deref(), Some("ssh: no route"));
        assert!(!sess.wants_connection());
    }

    #[test]
    fn failed_quic_handshake_reports_error_without_reconnecting() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, mut ctrl_rx) = started(&mut mgr, "a@h", false);
        let err = LauncherNetEvent::Disconnected(Some("handshake timed out".into()));
        assert_eq!(mgr.handle_event(net("a@h", attempt, err)), Changes::ALL);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert_eq!(sess.error.as_deref(), Some("handshake timed out"));
        assert!(!sess.wants_connection());
        // The connection handles were released.
        assert!(matches!(
            ctrl_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn failed_reconnect_keeps_retrying_and_shows_the_cause() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, _ctrl_rx) = connected(&mut mgr, "a@h");
        let _ = mgr.handle_event(net("a@h", attempt, LauncherNetEvent::Disconnected(None)));
        assert_eq!(
            mgr.session("a@h").unwrap().error.as_deref(),
            Some("Reconnecting in 2s…")
        );

        let retry = mgr.begin_attempt("a@h", true).unwrap();
        let err = Err("ssh: Permission denied\n".into());
        let _ = mgr.handle_event(finished("a@h", retry, err));
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert!(sess.wants_connection());
        assert_eq!(
            sess.error.as_deref(),
            Some("Reconnecting in 4s… Last error: ssh: Permission denied")
        );
    }

    #[test]
    fn stale_connect_result_is_dropped_before_starting() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let attempt = mgr.begin_attempt("a@h", false).unwrap();
        let _ = mgr.disconnect("a@h");

        assert_eq!(
            mgr.handle_event(finished("a@h", attempt, Ok(prepared()))),
            Changes::NONE
        );
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert!(sess.ctrl_tx.is_none() && sess.ui_tx.is_none() && sess.pty_socket.is_none());
    }

    #[test]
    fn worker_panics_become_errors() {
        let ok: std::result::Result<u8, String> = catch_panic(|| Ok(1));
        assert_eq!(ok, Ok(1));
        let panicked: std::result::Result<u8, String> = catch_panic(|| panic!("boom"));
        assert_eq!(
            panicked,
            Err("internal error while connecting: boom".to_string())
        );
    }

    #[test]
    fn unexpected_disconnect_schedules_reconnect_and_explicit_disconnect_cancels_it() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, _ctrl_rx) = connected(&mut mgr, "a@h");
        let _ = mgr.handle_event(net("a@h", attempt, LauncherNetEvent::AppList(Vec::new())));
        assert!(mgr.session("a@h").unwrap().apps_loaded);

        let changes = mgr.handle_event(net(
            "a@h",
            attempt,
            LauncherNetEvent::Disconnected(Some("timeout".into())),
        ));
        assert_eq!(changes, Changes::ALL);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert_eq!(
            sess.error.as_deref(),
            Some("Reconnecting in 2s… Last error: timeout")
        );
        assert!(sess.wants_connection());
        assert!(!sess.apps_loaded);

        let _ = mgr.disconnect("a@h");
        assert!(!mgr.session("a@h").unwrap().wants_connection());
        // The timer for the old attempt fires but is ignored.
        let due = SessionEvent(EventKind::ReconnectDue {
            target: "a@h".into(),
            attempt_id: attempt,
        });
        assert_eq!(mgr.handle_event(due), Changes::NONE);
        assert_eq!(mgr.session("a@h").unwrap().status, ConnStatus::Disconnected);
    }

    #[test]
    fn taken_over_session_disconnects_without_reconnecting() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, _ctrl_rx) = connected(&mut mgr, "a@h");
        assert!(mgr.session("a@h").unwrap().auto_reconnect);

        let changes = mgr.handle_event(net("a@h", attempt, LauncherNetEvent::TakenOver));
        assert_eq!(changes, Changes::ALL);
        let sess = mgr.session("a@h").unwrap();
        assert_eq!(sess.status, ConnStatus::Disconnected);
        assert_eq!(sess.error.as_deref(), Some(TAKEN_OVER_MESSAGE));
        assert!(!sess.wants_connection());
    }

    #[test]
    fn taken_over_wins_when_coalescing_disconnects() {
        let (net_tx, net_rx) = mpsc::unbounded_channel();
        let (events_tx, mut events_rx) = mpsc::unbounded_channel();
        net_tx.send(LauncherNetEvent::Connected).unwrap();
        net_tx.send(LauncherNetEvent::Disconnected(None)).unwrap();
        net_tx.send(LauncherNetEvent::TakenOver).unwrap();
        net_tx
            .send(LauncherNetEvent::Disconnected(Some("closed".into())))
            .unwrap();
        drop(net_tx);
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(forward_net_events("a@h".into(), 1, net_rx, events_tx));

        let mut forwarded = Vec::new();
        while let Ok(SessionEvent(EventKind::Net { event, .. })) = events_rx.try_recv() {
            forwarded.push(event);
        }
        assert!(matches!(
            forwarded.as_slice(),
            [LauncherNetEvent::Connected, LauncherNetEvent::TakenOver]
        ));
    }

    #[test]
    fn events_for_inactive_hosts_only_touch_the_host_list() {
        let (mut mgr, _events) = manager(&["a@h", "b@h"]);
        let (attempt, _ctrl_rx) = connected(&mut mgr, "b@h");
        assert_eq!(mgr.active_target(), Some("a@h"));
        let stats = LauncherNetEvent::NetStats {
            rtt_ms: 10,
            congested: false,
        };
        assert_eq!(mgr.handle_event(net("b@h", attempt, stats)), Changes::NONE);
        let dc = LauncherNetEvent::Disconnected(None);
        assert_eq!(mgr.handle_event(net("b@h", attempt, dc)), Changes::HOSTS);
    }

    #[test]
    fn programs_track_launch_state_and_terminate_escalates() {
        let (mut mgr, _events) = manager(&["a@h"]);
        let (attempt, mut ctrl_rx) = connected(&mut mgr, "a@h");

        assert_eq!(
            mgr.launch("Editor".into(), "gedit".into()),
            Changes::PROGRAMS
        );
        let launch_id = match ctrl_rx.try_recv() {
            Ok(ControlMessage::LaunchApp { launch_id, .. }) => launch_id,
            other => panic!("expected LaunchApp, got {other:?}"),
        };
        let running = LauncherNetEvent::AppState {
            launch_id,
            name: String::new(),
            state: LaunchState::Running,
        };
        let _ = mgr.handle_event(net("a@h", attempt, running));
        let prog = &mgr.session("a@h").unwrap().programs[0];
        assert_eq!(prog.name, "Editor");
        assert_eq!(prog.state, LaunchState::Running);

        assert!(mgr.terminate("a@h", launch_id));
        assert!(mgr.terminate("a@h", launch_id));
        assert!(matches!(
            ctrl_rx.try_recv(),
            Ok(ControlMessage::TerminateApp { force: false, .. })
        ));
        assert!(matches!(
            ctrl_rx.try_recv(),
            Ok(ControlMessage::TerminateApp { force: true, .. })
        ));

        let exited = LauncherNetEvent::AppState {
            launch_id,
            name: "Editor".into(),
            state: LaunchState::Exited(0),
        };
        let _ = mgr.handle_event(net("a@h", attempt, exited));
        assert!(mgr.session("a@h").unwrap().programs.is_empty());
    }

    #[test]
    fn launch_requires_a_connected_active_host() {
        let (mut mgr, _events) = manager(&["a@h"]);
        assert_eq!(mgr.launch("x".into(), "x".into()), Changes::NONE);
        let (_, _ctrl_rx) = started(&mut mgr, "a@h", false);
        assert_eq!(mgr.launch("x".into(), "x".into()), Changes::NONE);
    }

    #[test]
    fn host_edits_validate_rename_and_remove() {
        let (mut mgr, _events) = manager(&["a@h", "b@h"]);
        let (_, mut ctrl_rx) = connected(&mut mgr, "a@h");

        assert_eq!(
            mgr.update_host("a@h", profile("  ")),
            Err(HostEditError::EmptyTarget)
        );
        assert_eq!(
            mgr.update_host("a@h", profile("b@h")),
            Err(HostEditError::DuplicateTarget)
        );

        // Renaming the active host follows it and drops the old connection.
        assert_eq!(mgr.update_host("a@h", profile("c@h")), Ok(Changes::ALL));
        assert_eq!(mgr.active_target(), Some("c@h"));
        assert!(mgr.session("a@h").is_none());
        assert!(matches!(
            ctrl_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
        let targets: Vec<_> = mgr.hosts().iter().map(|h| h.ssh_target.as_str()).collect();
        assert_eq!(targets, ["c@h", "b@h"]);

        // Re-adding an existing target replaces it instead of duplicating it.
        let mut relabeled = profile("b@h");
        relabeled.label = "Box".into();
        assert!(mgr.add_host(relabeled).is_ok());
        assert_eq!(mgr.hosts().len(), 2);
        assert_eq!(mgr.host_label("b@h"), "Box");

        assert_eq!(mgr.remove_host("c@h"), Changes::ALL);
        assert_eq!(mgr.active_target(), Some("b@h"));
        assert_eq!(mgr.remove_host("c@h"), Changes::NONE);
    }
}
