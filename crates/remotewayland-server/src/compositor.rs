//! Headless Smithay Wayland compositor for RemoteWayland.
//!
//! Runs on a dedicated `calloop` thread, accepts Wayland client connections on an
//! isolated `WAYLAND_DISPLAY` socket, tracks `xdg_toplevel` and `xdg_popup`
//! windows, extracts committed SHM frames, and forwards input/configure events.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::os::unix::io::{AsRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::channel::{self, Channel, Sender as CalloopSender};
use calloop::generic::Generic;
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction};
use remotewayland_proto::{
    LaunchState, WindowId, MAX_CURSOR_DIM, MAX_SELECTION_BYTES, SUPPORTED_TEXT_MIMES,
};
use smithay::backend::input::KeyState;
use smithay::desktop::utils::send_frames_surface_tree;
use smithay::desktop::{PopupKind, PopupManager, Space, Window};
use smithay::input::keyboard::{
    FilterResult, KeyboardTarget, KeysymHandle, ModifiersState, XkbConfig,
};
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorImageStatus, CursorImageSurfaceData, MotionEvent,
};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as DecorationMode;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::{
    Mode as KdeDecorationMode, OrgKdeKwinServerDecoration,
};
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration_manager::Mode as KdeDefaultDecorationMode;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_output;
use smithay::reexports::wayland_server::protocol::wl_seat;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    backend::ObjectId, Client, Display, DisplayHandle, Resource, WEnum,
};
use smithay::utils::{IsAlive, Logical, Point, Rectangle, Serial, SERIAL_COUNTER};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    get_parent, with_states, with_surface_tree_downward, CompositorClientState, CompositorHandler,
    CompositorState, SubsurfaceCachedState, TraversalAction,
};
use smithay::wayland::cursor_shape::CursorShapeManagerState;
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::selection::data_device::{
    request_data_device_client_selection, set_data_device_focus, set_data_device_selection,
    ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
};
use smithay::wayland::selection::primary_selection::{
    request_primary_client_selection, set_primary_focus, set_primary_selection,
    PrimarySelectionHandler, PrimarySelectionState,
};
use smithay::wayland::selection::{SelectionHandler, SelectionSource, SelectionTarget};
use smithay::wayland::shell::kde::decoration::{KdeDecorationHandler, KdeDecorationState};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, SurfaceCachedState, ToplevelSurface, XdgPopupSurfaceData,
    XdgShellHandler, XdgShellState, XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{with_buffer_contents, ShmHandler, ShmState};
use smithay::wayland::socket::ListeningSocketSource;
use smithay::wayland::tablet_manager::TabletSeatHandler;
use smithay::wayland::xwayland_shell::{XWaylandShellHandler, XWaylandShellState};
use smithay::xwayland::xwm::{
    Reorder, ResizeEdge as X11ResizeEdge, WmWindowProperty, WmWindowType, X11Window, XwmId,
};
use smithay::xwayland::{X11Surface, X11Wm, XWayland, XWaylandClientData, XWaylandEvent, XwmHandler};
use smithay::{
    delegate_compositor, delegate_cursor_shape, delegate_data_device, delegate_kde_decoration,
    delegate_output, delegate_primary_selection, delegate_seat, delegate_shm,
    delegate_xdg_decoration, delegate_xdg_shell, delegate_xwayland_shell,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Events emitted by the headless compositor to the QUIC server loop.
#[derive(Debug, Clone)]
pub enum CompositorEvent {
    WindowCreated {
        window_id: WindowId,
        title: String,
        app_id: String,
        width: u32,
        height: u32,
    },
    PopupCreated {
        window_id: WindowId,
        parent_id: WindowId,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    PopupRepositioned {
        window_id: WindowId,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    PopupFrame {
        window_id: WindowId,
        width: u32,
        height: u32,
        bgra: Vec<u8>,
    },
    StartMove {
        window_id: WindowId,
    },
    StartResize {
        window_id: WindowId,
        edge: u32,
    },
    SetMaximized {
        window_id: WindowId,
        maximized: bool,
    },
    SetMinimized {
        window_id: WindowId,
    },
    SetFullscreen {
        window_id: WindowId,
        fullscreen: bool,
    },
    ShowWindowMenu {
        window_id: WindowId,
        x: i32,
        y: i32,
    },
    SetDecorationMode {
        window_id: WindowId,
        server_side: bool,
    },
    WindowConfigured {
        window_id: WindowId,
        width: u32,
        height: u32,
        scale_factor: f64,
    },
    WindowClosed {
        window_id: WindowId,
    },
    FrameCommitted {
        window_id: WindowId,
        width: u32,
        height: u32,
        bgra: Vec<u8>,
    },
    CursorUpdate {
        window_id: WindowId,
        visible: bool,
        shape_name: String,
    },
    CursorBitmap {
        window_id: WindowId,
        width: u32,
        height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        bgra: Vec<u8>,
    },
    ClipboardText {
        text: String,
    },
    PrimarySelectionText {
        text: String,
    },
    AppState {
        launch_id: u64,
        name: String,
        state: LaunchState,
    },
}

/// Commands sent from the QUIC server loop to the headless compositor.
#[derive(Debug, Clone)]
pub enum CompositorCommand {
    ConfigureWindow {
        window_id: WindowId,
        width: u32,
        height: u32,
        maximized: bool,
        fullscreen: bool,
        activated: bool,
        tiled: bool,
    },
    CloseWindow {
        window_id: WindowId,
    },
    PointerMotion {
        window_id: WindowId,
        x: f64,
        y: f64,
    },
    PointerButton {
        window_id: WindowId,
        button: u32,
        pressed: bool,
    },
    PointerAxis {
        window_id: WindowId,
        dx: f64,
        dy: f64,
    },
    KeyEvent {
        window_id: WindowId,
        keycode: u32,
        pressed: bool,
    },
    SetClipboardText {
        text: String,
    },
    SetPrimarySelectionText {
        text: String,
    },
    LaunchApp {
        launch_id: u64,
        name: Option<String>,
        exec: String,
    },
    TerminateApp {
        launch_id: u64,
        force: bool,
    },
    AppExited {
        launch_id: u64,
        code: i32,
    },
    SyncWindows,
}

/// Handle to interact with the running compositor thread.
#[derive(Clone)]
pub struct CompositorHandle {
    pub socket_name: OsString,
    pub cmd_tx: CalloopSender<CompositorCommand>,
}

#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl smithay::reexports::wayland_server::backend::ClientData for ClientState {
    fn initialized(&self, _client_id: smithay::reexports::wayland_server::backend::ClientId) {}
    fn disconnected(
        &self,
        _client_id: smithay::reexports::wayland_server::backend::ClientId,
        _reason: smithay::reexports::wayland_server::backend::DisconnectReason,
    ) {
    }
}

const WINDOW_LAUNCH_ID_BASE: u64 = 1u64 << 62;

struct TrackedWindow {
    window_id: WindowId,
    window: Window,
    last_title: String,
    last_app_id: String,
    last_width: u32,
    last_height: u32,
    geo_offset: (i32, i32),
    server_side_decoration: bool,
    announced: bool,
    has_window_entry: bool,
    latest_bgra: Option<(u32, u32, Vec<u8>)>,
    last_sub_info: Vec<(bool, u32, u32, i32, i32)>,
}

struct TrackedPopup {
    window_id: WindowId,
    parent_id: WindowId,
    surface: PopupSurface,
    geo: Rectangle<i32, Logical>,
    geo_offset: (i32, i32),
    announced: bool,
    latest_bgra: Option<(u32, u32, Vec<u8>)>,
}

struct TrackedX11Window {
    window_id: WindowId,
    surface: X11Surface,
    geo: Rectangle<i32, Logical>,
    last_title: String,
    last_app_id: String,
    last_width: u32,
    last_height: u32,
    server_side_decoration: bool,
    announced: bool,
    has_window_entry: bool,
    latest_bgra: Option<(u32, u32, Vec<u8>)>,
}

struct TrackedX11Popup {
    window_id: WindowId,
    parent_id: WindowId,
    surface: X11Surface,
    geo: Rectangle<i32, Logical>,
    rel_pos: (i32, i32),
    announced: bool,
    latest_bgra: Option<(u32, u32, Vec<u8>)>,
}

/// Keyboard focus target supporting both native Wayland surfaces and XWayland `X11Surface`s.
/// Using `X11Surface` as a `KeyboardTarget` ensures Smithay sends `XSetInputFocus` and
/// `WM_TAKE_FOCUS` to X11 clients (required for X11 menus/popups in Chromium, GTK, etc.).
#[derive(Debug, Clone, PartialEq)]
enum KeyboardFocusTarget {
    Wayland(WlSurface),
    X11(X11Surface),
}

impl IsAlive for KeyboardFocusTarget {
    fn alive(&self) -> bool {
        match self {
            Self::Wayland(s) => s.alive(),
            Self::X11(s) => s.alive(),
        }
    }
}

impl WaylandFocus for KeyboardFocusTarget {
    fn wl_surface(&self) -> Option<std::borrow::Cow<'_, WlSurface>> {
        match self {
            Self::Wayland(s) => Some(std::borrow::Cow::Borrowed(s)),
            Self::X11(s) => s.wl_surface().map(std::borrow::Cow::Owned),
        }
    }
}

impl KeyboardTarget<AppState> for KeyboardFocusTarget {
    fn enter(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(s) => KeyboardTarget::enter(s, seat, data, keys, serial),
            Self::X11(s) => KeyboardTarget::enter(s, seat, data, keys, serial),
        }
    }

    fn leave(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        match self {
            Self::Wayland(s) => KeyboardTarget::leave(s, seat, data, serial),
            Self::X11(s) => KeyboardTarget::leave(s, seat, data, serial),
        }
    }

    fn key(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        key: KeysymHandle<'_>,
        state: KeyState,
        serial: Serial,
        time: u32,
    ) {
        match self {
            Self::Wayland(s) => KeyboardTarget::key(s, seat, data, key, state, serial, time),
            Self::X11(s) => KeyboardTarget::key(s, seat, data, key, state, serial, time),
        }
    }

    fn modifiers(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(s) => KeyboardTarget::modifiers(s, seat, data, modifiers, serial),
            Self::X11(s) => KeyboardTarget::modifiers(s, seat, data, modifiers, serial),
        }
    }
}

/// Tracks which source currently owns a selection (Wayland client, X11 client, or remote host).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionOwner {
    None,
    RemoteHost,
    WaylandClient,
    X11Client,
}

struct LaunchedApp {
    launch_id: u64,
    name: String,
    pgid: i32,
    state: LaunchState,
    linked_window_id: Option<WindowId>,
}

#[allow(dead_code)]
struct AppState {
    display_handle: DisplayHandle,
    loop_handle: LoopHandle<'static, Self>,
    socket_name: OsString,
    cmd_tx: CalloopSender<CompositorCommand>,
    compositor_state: CompositorState,
    shm_state: ShmState,
    xdg_shell_state: XdgShellState,
    xdg_decoration_state: XdgDecorationState,
    kde_decoration_state: KdeDecorationState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,
    primary_selection_state: PrimarySelectionState,
    cursor_shape_state: CursorShapeManagerState,
    output_manager_state: OutputManagerState,
    xwayland_shell_state: XWaylandShellState,
    xwm: Option<X11Wm>,
    xwayland_client: Option<Client>,
    output: Output,
    output_size: (i32, i32),
    seat: Seat<Self>,
    space: Space<Window>,
    popups: PopupManager,
    next_window_id: WindowId,
    focused_window_id: WindowId,
    last_focused_x11_window_id: Option<WindowId>,
    next_x11_pos: (i32, i32),
    cursor_surface: Option<WlSurface>,
    last_cursor_bitmap: Option<(u32, u32, i32, i32, Vec<u8>)>,
    clipboard_owner: SelectionOwner,
    clipboard_text: Option<String>,
    pending_clipboard_mime: Option<String>,
    primary_owner: SelectionOwner,
    primary_text: Option<String>,
    pending_primary_mime: Option<String>,
    pending_initial_command: Vec<String>,
    start_time: Instant,
    /// Map from `WlSurface::id()` (`ObjectId`, unique across clients) -> WindowId (for toplevels & popups).
    surface_to_window: HashMap<ObjectId, WindowId>,
    /// Cached RGBA buffer per `WlSurface::id()` for surface-tree compositing.
    surface_buffers: HashMap<ObjectId, (u32, u32, Vec<u8>)>,
    /// KDE server-decoration modes set on a `WlSurface` before `xdg_surface.get_toplevel`.
    pending_kde_decorations: HashMap<ObjectId, bool>,
    windows: HashMap<WindowId, TrackedWindow>,
    popup_windows: HashMap<WindowId, TrackedPopup>,
    x11_windows: HashMap<WindowId, TrackedX11Window>,
    x11_popups: HashMap<WindowId, TrackedX11Popup>,
    x11_id_by_win: HashMap<X11Window, WindowId>,
    launched_apps: BTreeMap<u64, LaunchedApp>,
    next_server_launch_id: u64,
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
}

impl AppState {
    fn alloc_window_id(&mut self) -> WindowId {
        let id = self.next_window_id;
        self.next_window_id += 1;
        id
    }

    /// Dynamically expand the headless virtual output if a window or X11 geometry
    /// exceeds its current bounds so X11 windows and menus are not clamped on 4K/ultrawide displays.
    fn ensure_virtual_output_size(&mut self, min_w: i32, min_h: i32) {
        let new_w = self.output_size.0.max(min_w);
        let new_h = self.output_size.1.max(min_h);
        if (new_w, new_h) != self.output_size {
            self.output_size = (new_w, new_h);
            let mode = OutputMode {
                size: (new_w, new_h).into(),
                refresh: 60_000,
            };
            self.output
                .change_current_state(Some(mode), None, None, Some((0, 0).into()));
            self.output.set_preferred(mode);
            debug!(width = new_w, height = new_h, "expanded headless virtual output");
        }
    }

    fn alloc_x11_position(&mut self, w: i32, h: i32) -> (i32, i32) {
        let (x, y) = self.next_x11_pos;
        let next_x = if x + 40 + w.min(800) > self.output_size.0 {
            100
        } else {
            x + 40
        };
        let next_y = if y + 40 + h.min(600) > self.output_size.1 {
            100
        } else {
            y + 40
        };
        self.next_x11_pos = (next_x, next_y);
        self.ensure_virtual_output_size(x + w + 64, y + h + 64);
        (x, y)
    }

    fn emit_cursor_surface_bitmap(&mut self, surf: &WlSurface) {
        let Some((w, h, bgra)) = self.surface_buffers.get(&surf.id()) else {
            return;
        };
        if *w == 0 || *h == 0 || *w > MAX_CURSOR_DIM || *h > MAX_CURSOR_DIM {
            return;
        }
        let (hotspot_x, hotspot_y) = with_states(surf, |states| {
            states
                .data_map
                .get::<CursorImageSurfaceData>()
                .and_then(|d| d.lock().ok())
                .map(|attrs| (attrs.hotspot.x, attrs.hotspot.y))
                .unwrap_or((0, 0))
        });
        if self
            .last_cursor_bitmap
            .as_ref()
            .is_some_and(|(lw, lh, lhx, lhy, lbgra)| {
                *lw == *w && *lh == *h && *lhx == hotspot_x && *lhy == hotspot_y && lbgra == bgra
            })
        {
            return;
        }
        let bgra_cloned = bgra.clone();
        self.last_cursor_bitmap = Some((*w, *h, hotspot_x, hotspot_y, bgra_cloned.clone()));
        let _ = self.event_tx.send(CompositorEvent::CursorBitmap {
            window_id: self.focused_window_id,
            width: *w,
            height: *h,
            hotspot_x,
            hotspot_y,
            bgra: bgra_cloned,
        });
    }

    fn cleanup_xwayland_state(&mut self) {
        crate::session::clear_x11_env();
        self.xwm = None;
        let x11_wins: Vec<X11Window> = self.x11_id_by_win.keys().copied().collect();
        for win in x11_wins {
            self.remove_x11_window(win);
        }
    }

    fn hit_test_tree(
        &self,
        tree_root: &WlSurface,
        base: (i32, i32),
        x: f64,
        y: f64,
    ) -> Option<(WlSurface, Point<f64, Logical>, Point<f64, Logical>)> {
        let ix = x as i32;
        let iy = y as i32;
        let mut hit = None;
        with_surface_tree_downward(
            tree_root,
            base,
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == tree_root {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, _)) = self.surface_buffers.get(&surf.id()) {
                    let lx = ix - nx;
                    let ly = iy - ny;
                    if lx >= 0 && lx < *sw as i32 && ly >= 0 && ly < *sh as i32 {
                        let mut attrs = states
                            .cached_state
                            .get::<smithay::wayland::compositor::SurfaceAttributes>();
                        let accepts = match attrs.current().input_region.as_ref() {
                            None => true,
                            Some(reg) => reg.contains((lx, ly)),
                        };
                        if accepts {
                            hit = Some((
                                surf.clone(),
                                Point::<f64, Logical>::from((nx as f64, ny as f64)),
                                Point::<f64, Logical>::from((x, y)),
                            ));
                        }
                    }
                }
                TraversalAction::DoChildren((nx, ny))
            },
            |_, _, _| {},
            |_, _, _| true,
        );
        hit
    }

    fn handle_command(&mut self, cmd: CompositorCommand) {
        let serial = SERIAL_COUNTER.next_serial();
        let time_ms = self.start_time.elapsed().as_millis() as u32;
        match cmd {
            CompositorCommand::ConfigureWindow {
                window_id,
                width,
                height,
                maximized,
                fullscreen,
                activated,
                tiled,
            } => {
                if let Some(tracked) = self.windows.get(&window_id) {
                    if let Some(toplevel) = tracked.window.toplevel() {
                        debug!(
                            window_id,
                            width, height, maximized, fullscreen, activated, tiled,
                            "ConfigureWindow -> remote toplevel"
                        );
                        toplevel.with_pending_state(|state| {
                            state.size = Some((width as i32, height as i32).into());
                            if maximized {
                                state.states.set(xdg_toplevel::State::Maximized);
                            } else {
                                state.states.unset(xdg_toplevel::State::Maximized);
                            }
                            if fullscreen {
                                state.states.set(xdg_toplevel::State::Fullscreen);
                            } else {
                                state.states.unset(xdg_toplevel::State::Fullscreen);
                            }
                            if activated {
                                state.states.set(xdg_toplevel::State::Activated);
                            } else {
                                state.states.unset(xdg_toplevel::State::Activated);
                            }
                            for edge in [
                                xdg_toplevel::State::TiledLeft,
                                xdg_toplevel::State::TiledRight,
                                xdg_toplevel::State::TiledTop,
                                xdg_toplevel::State::TiledBottom,
                            ] {
                                if tiled {
                                    state.states.set(edge);
                                } else {
                                    state.states.unset(edge);
                                }
                            }
                        });
                        toplevel.send_pending_configure();
                    }
                } else if let Some(xwin) = self.x11_windows.get_mut(&window_id) {
                    let w = width.max(1) as i32;
                    let h = height.max(1) as i32;
                    let mut geo = xwin.geo;
                    geo.size = (w, h).into();
                    let min_w = geo.loc.x + geo.size.w + 64;
                    let min_h = geo.loc.y + geo.size.h + 64;
                    xwin.geo = geo;
                    let surf = xwin.surface.clone();
                    let has_child_popup = self
                        .x11_popups
                        .values()
                        .any(|p| p.parent_id == window_id);
                    let _ = surf.set_maximized(maximized);
                    let _ = surf.set_fullscreen(fullscreen);
                    if activated || !has_child_popup {
                        let _ = surf.set_activated(activated);
                    }
                    let _ = surf.configure(geo);
                    if activated && !has_child_popup {
                        if let Some(xwm) = self.xwm.as_mut() {
                            let _ = xwm.raise_window(&surf);
                        }
                        if let Some(keyboard) = self.seat.get_keyboard() {
                            let target = KeyboardFocusTarget::X11(surf.clone());
                            if keyboard.current_focus().as_ref() != Some(&target) {
                                keyboard.set_focus(self, Some(target), serial);
                            }
                        }
                    }
                    self.ensure_virtual_output_size(min_w, min_h);
                }
            }
            CompositorCommand::CloseWindow { window_id } => {
                debug!(window_id, "CloseWindow command received from client");
                if let Some(tracked) = self.windows.get(&window_id) {
                    if let Some(toplevel) = tracked.window.toplevel() {
                        toplevel.send_close();
                    }
                } else if let Some(popup) = self.popup_windows.get(&window_id) {
                    popup.surface.send_popup_done();
                } else if let Some(xwin) = self.x11_windows.get(&window_id) {
                    let _ = xwin.surface.close();
                } else if let Some(xpop) = self.x11_popups.get(&window_id) {
                    let _ = xpop.surface.close();
                }
            }
            CompositorCommand::PointerMotion { window_id, x, y } => {
                self.focused_window_id = window_id;
                if let Some(pointer) = self.seat.get_pointer() {
                    let focus_and_loc = if let Some(tw) = self.windows.get(&window_id) {
                        let (gx, gy) = tw.geo_offset;
                        tw.window.toplevel().and_then(|t| {
                            self.hit_test_tree(t.wl_surface(), (-gx, -gy), x, y)
                        })
                    } else if let Some(tp) = self.popup_windows.get(&window_id) {
                        let (gx, gy) = tp.geo_offset;
                        self.hit_test_tree(tp.surface.wl_surface(), (-gx, -gy), x, y)
                    } else if let Some(xw) = self.x11_windows.get(&window_id) {
                        let ox = xw.geo.loc.x as f64;
                        let oy = xw.geo.loc.y as f64;
                        xw.surface.wl_surface().map(|s| {
                            (
                                s,
                                Point::<f64, Logical>::from((ox, oy)),
                                Point::<f64, Logical>::from((ox + x, oy + y)),
                            )
                        })
                    } else if let Some(xp) = self.x11_popups.get(&window_id) {
                        let ox = xp.geo.loc.x as f64;
                        let oy = xp.geo.loc.y as f64;
                        xp.surface.wl_surface().map(|s| {
                            (
                                s,
                                Point::<f64, Logical>::from((ox, oy)),
                                Point::<f64, Logical>::from((ox + x, oy + y)),
                            )
                        })
                    } else {
                        None
                    };

                    let (focus, location) = match focus_and_loc {
                        Some((surf, origin, loc)) => (Some((surf, origin)), loc),
                        None => (None, (x, y).into()),
                    };
                    pointer.motion(
                        self,
                        focus,
                        &MotionEvent {
                            location,
                            serial,
                            time: time_ms,
                        },
                    );
                    pointer.frame(self);
                }
            }
            CompositorCommand::PointerButton {
                window_id,
                button,
                pressed,
            } => {
                self.focused_window_id = window_id;
                if pressed {
                    if let Some(keyboard) = self.seat.get_keyboard() {
                        if let Some(tracked) = self.windows.get(&window_id) {
                            if let Some(toplevel) = tracked.window.toplevel() {
                                let root_surface = toplevel.wl_surface().clone();
                                let target = KeyboardFocusTarget::Wayland(root_surface.clone());
                                if keyboard.current_focus().as_ref() != Some(&target) {
                                    keyboard.set_focus(self, Some(target), serial);
                                }
                                for (popup, _) in PopupManager::popups_for_surface(&root_surface) {
                                    let _ = PopupManager::dismiss_popup(&root_surface, &popup);
                                }
                            }
                        } else if let Some(xw) = self.x11_windows.get(&window_id) {
                            self.last_focused_x11_window_id = Some(window_id);
                            let x11_surf = xw.surface.clone();
                            let has_child_popup = self
                                .x11_popups
                                .values()
                                .any(|p| p.parent_id == window_id);
                            let target = KeyboardFocusTarget::X11(x11_surf.clone());
                            if keyboard.current_focus().as_ref() != Some(&target) {
                                keyboard.set_focus(self, Some(target), serial);
                            }
                            if !x11_surf.is_activated() {
                                let _ = x11_surf.set_activated(true);
                                if !has_child_popup {
                                    if let Some(xwm) = self.xwm.as_mut() {
                                        let _ = xwm.raise_window(&x11_surf);
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(pointer) = self.seat.get_pointer() {
                    let state = if pressed {
                        smithay::backend::input::ButtonState::Pressed
                    } else {
                        smithay::backend::input::ButtonState::Released
                    };
                    pointer.button(
                        self,
                        &ButtonEvent {
                            button,
                            state,
                            serial,
                            time: time_ms,
                        },
                    );
                    pointer.frame(self);
                }
            }
            CompositorCommand::PointerAxis {
                window_id,
                dx,
                dy,
            } => {
                self.focused_window_id = window_id;
                if let Some(pointer) = self.seat.get_pointer() {
                    let mut frame = AxisFrame::new(time_ms);
                    if dx != 0.0 {
                        frame = frame.value(smithay::backend::input::Axis::Horizontal, dx);
                    }
                    if dy != 0.0 {
                        frame = frame.value(smithay::backend::input::Axis::Vertical, dy);
                    }
                    pointer.axis(self, frame);
                    pointer.frame(self);
                }
            }
            CompositorCommand::KeyEvent {
                window_id,
                keycode,
                pressed,
            } => {
                self.focused_window_id = window_id;
                if let Some(keyboard) = self.seat.get_keyboard() {
                    if let Some(tracked) = self.windows.get(&window_id) {
                        if let Some(toplevel) = tracked.window.toplevel() {
                            let target = KeyboardFocusTarget::Wayland(toplevel.wl_surface().clone());
                            if keyboard.current_focus().as_ref() != Some(&target) {
                                keyboard.set_focus(self, Some(target), serial);
                            }
                        }
                    } else if let Some(xw) = self.x11_windows.get(&window_id) {
                        self.last_focused_x11_window_id = Some(window_id);
                        let target = KeyboardFocusTarget::X11(xw.surface.clone());
                        if keyboard.current_focus().as_ref() != Some(&target) {
                            keyboard.set_focus(self, Some(target), serial);
                        }
                    } else if let Some(xp) = self.x11_popups.get(&window_id) {
                        let target = KeyboardFocusTarget::X11(xp.surface.clone());
                        if keyboard.current_focus().as_ref() != Some(&target) {
                            keyboard.set_focus(self, Some(target), serial);
                        }
                    }
                    let state = if pressed {
                        KeyState::Pressed
                    } else {
                        KeyState::Released
                    };
                    let xkb_keycode = keycode.saturating_add(8);
                    keyboard.input::<(), _>(
                        self,
                        xkb_keycode.into(),
                        state,
                        serial,
                        time_ms,
                        |_, _, _| FilterResult::Forward,
                    );
                }
            }
            CompositorCommand::SetClipboardText { text } => {
                if self.clipboard_owner == SelectionOwner::RemoteHost
                    && self.clipboard_text.as_deref() == Some(text.as_str())
                {
                    return;
                }
                self.clipboard_owner = SelectionOwner::RemoteHost;
                self.clipboard_text = Some(text);
                let mimes = text_selection_mimes();
                set_data_device_selection(&self.display_handle, &self.seat, mimes.clone(), ());
                if let Some(xwm) = self.xwm.as_mut() {
                    let _ = xwm.new_selection(SelectionTarget::Clipboard, Some(mimes));
                }
            }
            CompositorCommand::SetPrimarySelectionText { text } => {
                if self.primary_owner == SelectionOwner::RemoteHost
                    && self.primary_text.as_deref() == Some(text.as_str())
                {
                    return;
                }
                self.primary_owner = SelectionOwner::RemoteHost;
                self.primary_text = Some(text);
                let mimes = text_selection_mimes();
                set_primary_selection(&self.display_handle, &self.seat, mimes.clone(), ());
                if let Some(xwm) = self.xwm.as_mut() {
                    let _ = xwm.new_selection(SelectionTarget::Primary, Some(mimes));
                }
            }
            CompositorCommand::LaunchApp {
                launch_id,
                name,
                exec,
            } => {
                let id = if launch_id == 0 {
                    let v = self.next_server_launch_id;
                    self.next_server_launch_id += 1;
                    v
                } else {
                    launch_id
                };
                let display_name = name
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| derive_exec_display_name(&exec));
                match spawn_gui_shell(&self.socket_name, &exec, id, self.cmd_tx.clone()) {
                    Ok(pgid) => {
                        self.launched_apps.insert(
                            id,
                            LaunchedApp {
                                launch_id: id,
                                name: display_name.clone(),
                                pgid,
                                state: LaunchState::Launching,
                                linked_window_id: None,
                            },
                        );
                        let _ = self.event_tx.send(CompositorEvent::AppState {
                            launch_id: id,
                            name: display_name,
                            state: LaunchState::Launching,
                        });
                    }
                    Err(err) => {
                        error!(error = %err, exec, "failed to launch remote application");
                        let _ = self.event_tx.send(CompositorEvent::AppState {
                            launch_id: id,
                            name: display_name,
                            state: LaunchState::Exited(1),
                        });
                    }
                }
            }
            CompositorCommand::TerminateApp { launch_id, force } => {
                if launch_id >= WINDOW_LAUNCH_ID_BASE {
                    let wid = launch_id & !WINDOW_LAUNCH_ID_BASE;
                    if let Some(tracked) = self.windows.get(&wid) {
                        if let Some(toplevel) = tracked.window.toplevel() {
                            toplevel.send_close();
                        }
                    } else if let Some(xwin) = self.x11_windows.get(&wid) {
                        let _ = xwin.surface.close();
                    }
                } else if let Some(app) = self.launched_apps.get(&launch_id) {
                    if !matches!(app.state, LaunchState::Exited(_)) && app.pgid > 0 {
                        let sig = if force { libc::SIGKILL } else { libc::SIGTERM };
                        unsafe {
                            libc::kill(-app.pgid, sig);
                        }
                    }
                }
            }
            CompositorCommand::AppExited { launch_id, code } => {
                if let Some(app) = self.launched_apps.remove(&launch_id) {
                    let _ = self.event_tx.send(CompositorEvent::AppState {
                        launch_id,
                        name: app.name.clone(),
                        state: LaunchState::Exited(code),
                    });
                    if let Some(wid) = app.linked_window_id {
                        if let Some(tracked) = self.windows.get_mut(&wid) {
                            if !tracked.has_window_entry {
                                tracked.has_window_entry = true;
                                let win_name = if !app.name.is_empty() {
                                    app.name
                                } else if !tracked.last_title.is_empty() {
                                    tracked.last_title.clone()
                                } else {
                                    tracked.last_app_id.clone()
                                };
                                let _ = self.event_tx.send(CompositorEvent::AppState {
                                    launch_id: WINDOW_LAUNCH_ID_BASE | wid,
                                    name: win_name,
                                    state: LaunchState::Running,
                                });
                            }
                        } else if let Some(xwin) = self.x11_windows.get_mut(&wid) {
                            if !xwin.has_window_entry {
                                xwin.has_window_entry = true;
                                let win_name = if !app.name.is_empty() {
                                    app.name
                                } else if !xwin.last_title.is_empty() {
                                    xwin.last_title.clone()
                                } else {
                                    xwin.last_app_id.clone()
                                };
                                let _ = self.event_tx.send(CompositorEvent::AppState {
                                    launch_id: WINDOW_LAUNCH_ID_BASE | wid,
                                    name: win_name,
                                    state: LaunchState::Running,
                                });
                            }
                        }
                    }
                }
            }
            CompositorCommand::SyncWindows => {
                for app in self.launched_apps.values() {
                    if !matches!(app.state, LaunchState::Exited(_)) {
                        let _ = self.event_tx.send(CompositorEvent::AppState {
                            launch_id: app.launch_id,
                            name: app.name.clone(),
                            state: app.state,
                        });
                    }
                }
                for tracked in self.windows.values() {
                    if tracked.announced && tracked.has_window_entry {
                        let win_name = if !tracked.last_title.is_empty() {
                            tracked.last_title.clone()
                        } else if !tracked.last_app_id.is_empty() {
                            tracked.last_app_id.clone()
                        } else {
                            format!("Window #{}", tracked.window_id)
                        };
                        let _ = self.event_tx.send(CompositorEvent::AppState {
                            launch_id: WINDOW_LAUNCH_ID_BASE | tracked.window_id,
                            name: win_name,
                            state: LaunchState::Running,
                        });
                    }
                }
                for tracked in self.windows.values() {
                    if tracked.announced {
                        let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                            window_id: tracked.window_id,
                            title: tracked.last_title.clone(),
                            app_id: tracked.last_app_id.clone(),
                            width: tracked.last_width,
                            height: tracked.last_height,
                        });
                        let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                            window_id: tracked.window_id,
                            server_side: tracked.server_side_decoration,
                        });
                        if let Some((w, h, bgra)) = &tracked.latest_bgra {
                            let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
                                window_id: tracked.window_id,
                                width: *w,
                                height: *h,
                                bgra: bgra.clone(),
                            });
                        }
                    }
                }
                for popup in self.popup_windows.values() {
                    if popup.announced {
                        if let Some((w, h, bgra)) = &popup.latest_bgra {
                            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                                window_id: popup.window_id,
                                parent_id: popup.parent_id,
                                x: popup.geo.loc.x,
                                y: popup.geo.loc.y,
                                width: *w,
                                height: *h,
                            });
                            let _ = self.event_tx.send(CompositorEvent::PopupFrame {
                                window_id: popup.window_id,
                                width: *w,
                                height: *h,
                                bgra: bgra.clone(),
                            });
                        }
                    }
                }
                for xwin in self.x11_windows.values() {
                    if xwin.announced {
                        if xwin.has_window_entry {
                            let win_name = if !xwin.last_title.is_empty() {
                                xwin.last_title.clone()
                            } else if !xwin.last_app_id.is_empty() {
                                xwin.last_app_id.clone()
                            } else {
                                format!("X11 Window #{}", xwin.window_id)
                            };
                            let _ = self.event_tx.send(CompositorEvent::AppState {
                                launch_id: WINDOW_LAUNCH_ID_BASE | xwin.window_id,
                                name: win_name,
                                state: LaunchState::Running,
                            });
                        }
                        let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                            window_id: xwin.window_id,
                            title: xwin.last_title.clone(),
                            app_id: xwin.last_app_id.clone(),
                            width: xwin.last_width,
                            height: xwin.last_height,
                        });
                        let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                            window_id: xwin.window_id,
                            server_side: xwin.server_side_decoration,
                        });
                        if let Some((w, h, bgra)) = &xwin.latest_bgra {
                            let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
                                window_id: xwin.window_id,
                                width: *w,
                                height: *h,
                                bgra: bgra.clone(),
                            });
                        }
                    }
                }
                for xpop in self.x11_popups.values() {
                    if xpop.announced {
                        if let Some((w, h, bgra)) = &xpop.latest_bgra {
                            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                                window_id: xpop.window_id,
                                parent_id: xpop.parent_id,
                                x: xpop.rel_pos.0,
                                y: xpop.rel_pos.1,
                                width: *w,
                                height: *h,
                            });
                            let _ = self.event_tx.send(CompositorEvent::PopupFrame {
                                window_id: xpop.window_id,
                                width: *w,
                                height: *h,
                                bgra: bgra.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    fn recomposite_window(&mut self, window_id: WindowId, root_surface: &WlSurface) {
        let root_id = root_surface.id();
        let Some((surf_w, surf_h, _)) = self.surface_buffers.get(&root_id) else {
            return;
        };
        let surf_w = *surf_w;
        let surf_h = *surf_h;

        let (title, app_id) = with_states(root_surface, |states| {
            let guard = states
                .data_map
                .get::<XdgToplevelSurfaceData>()
                .and_then(|d| d.lock().ok());
            match guard {
                Some(data) => (
                    data.title.clone().unwrap_or_default(),
                    data.app_id.clone().unwrap_or_default(),
                ),
                None => (String::new(), String::new()),
            }
        });

        let xdg_geo = with_states(root_surface, |states| {
            states
                .cached_state
                .get::<SurfaceCachedState>()
                .current()
                .geometry
        });
        let (crop_x, crop_y, win_w, win_h) = match xdg_geo {
            Some(g) if g.size.w > 0 && g.size.h > 0 => {
                let cx = g.loc.x.max(0);
                let cy = g.loc.y.max(0);
                let cw = (g.size.w as u32).min(surf_w.saturating_sub(cx as u32));
                let ch = (g.size.h as u32).min(surf_h.saturating_sub(cy as u32));
                (cx, cy, cw.max(1), ch.max(1))
            }
            _ => (0, 0, surf_w, surf_h),
        };

        let mut composite = vec![0u8; (win_w * win_h * 4) as usize];
        let mut sub_info = Vec::new();

        with_surface_tree_downward(
            root_surface,
            (-crop_x, -crop_y),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == root_surface {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, srgba)) = self.surface_buffers.get(&surf.id()) {
                    sub_info.push((surf == root_surface, *sw, *sh, nx, ny));
                    blit_rgba_over(&mut composite, win_w, win_h, srgba, *sw, *sh, nx, ny);
                }
                TraversalAction::DoChildren((nx, ny))
            },
            |_, _, _| {},
            |_, _, _| true,
        );

        let Some(tracked) = self.windows.get_mut(&window_id) else {
            return;
        };
        tracked.geo_offset = (crop_x, crop_y);
        if sub_info != tracked.last_sub_info {
            debug!(
                window_id,
                win_w,
                win_h,
                crop_x,
                crop_y,
                ?sub_info,
                "window surface tree changed"
            );
            tracked.last_sub_info = sub_info;
        }

        if tracked
            .latest_bgra
            .as_ref()
            .is_some_and(|(lw, lh, lbgra)| *lw == win_w && *lh == win_h && lbgra == &composite)
        {
            return;
        }

        if !tracked.announced && win_w > 0 && win_h > 0 {
            tracked.announced = true;
            tracked.last_width = win_w;
            tracked.last_height = win_h;
            tracked.last_title = title.clone();
            tracked.last_app_id = app_id.clone();
            let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                window_id,
                title: title.clone(),
                app_id: app_id.clone(),
                width: win_w,
                height: win_h,
            });
            let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                window_id,
                server_side: tracked.server_side_decoration,
            });
            let mut linked = false;
            for app in self.launched_apps.values_mut() {
                if app.state == LaunchState::Launching {
                    app.state = LaunchState::Running;
                    app.linked_window_id = Some(window_id);
                    linked = true;
                    let _ = self.event_tx.send(CompositorEvent::AppState {
                        launch_id: app.launch_id,
                        name: app.name.clone(),
                        state: LaunchState::Running,
                    });
                    break;
                }
            }
            if !linked {
                tracked.has_window_entry = true;
                let win_name = if !title.is_empty() {
                    title
                } else if !app_id.is_empty() {
                    app_id
                } else {
                    format!("Window #{window_id}")
                };
                let _ = self.event_tx.send(CompositorEvent::AppState {
                    launch_id: WINDOW_LAUNCH_ID_BASE | window_id,
                    name: win_name,
                    state: LaunchState::Running,
                });
            }
        } else if win_w != tracked.last_width || win_h != tracked.last_height {
            tracked.last_width = win_w;
            tracked.last_height = win_h;
            let _ = self.event_tx.send(CompositorEvent::WindowConfigured {
                window_id,
                width: win_w,
                height: win_h,
                scale_factor: 1.0,
            });
        }

        tracked.latest_bgra = Some((win_w, win_h, composite.clone()));
        let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
            window_id,
            width: win_w,
            height: win_h,
            bgra: composite,
        });
    }

    fn recomposite_popup(&mut self, window_id: WindowId) {
        let Some(popup) = self.popup_windows.get(&window_id) else {
            return;
        };
        let root_surface = popup.surface.wl_surface().clone();
        let root_id = root_surface.id();
        let Some((surf_w, surf_h, _)) = self.surface_buffers.get(&root_id) else {
            return;
        };
        let surf_w = *surf_w;
        let surf_h = *surf_h;

        let acked_geo = with_states(&root_surface, |states| {
            states
                .data_map
                .get::<XdgPopupSurfaceData>()
                .and_then(|d| d.lock().ok())
                .map(|attr| attr.last_acked.unwrap_or(attr.current).geometry)
        });

        let xdg_geo = with_states(&root_surface, |states| {
            states
                .cached_state
                .get::<SurfaceCachedState>()
                .current()
                .geometry
        });
        let (crop_x, crop_y, pop_w, pop_h) = match xdg_geo {
            Some(g) if g.size.w > 0 && g.size.h > 0 => {
                let cx = g.loc.x.max(0);
                let cy = g.loc.y.max(0);
                let cw = (g.size.w as u32).min(surf_w.saturating_sub(cx as u32));
                let ch = (g.size.h as u32).min(surf_h.saturating_sub(cy as u32));
                (cx, cy, cw.max(1), ch.max(1))
            }
            _ => (0, 0, surf_w, surf_h),
        };

        let mut bgra = vec![0u8; (pop_w * pop_h * 4) as usize];
        with_surface_tree_downward(
            &root_surface,
            (-crop_x, -crop_y),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == &root_surface {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, sbgra)) = self.surface_buffers.get(&surf.id()) {
                    blit_rgba_over(&mut bgra, pop_w, pop_h, sbgra, *sw, *sh, nx, ny);
                }
                TraversalAction::DoChildren((nx, ny))
            },
            |_, _, _| {},
            |_, _, _| true,
        );

        let Some(popup) = self.popup_windows.get_mut(&window_id) else {
            return;
        };
        if let Some(g) = acked_geo {
            if g.size.w > 0 && g.size.h > 0 {
                popup.geo = g;
            }
        }
        popup.geo_offset = (crop_x, crop_y);

        if popup
            .latest_bgra
            .as_ref()
            .is_some_and(|(lw, lh, lbgra)| *lw == pop_w && *lh == pop_h && lbgra == &bgra)
        {
            return;
        }

        if !popup.announced {
            popup.announced = true;
            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                window_id,
                parent_id: popup.parent_id,
                x: popup.geo.loc.x,
                y: popup.geo.loc.y,
                width: pop_w,
                height: pop_h,
            });
        }

        popup.latest_bgra = Some((pop_w, pop_h, bgra.clone()));
        let _ = self.event_tx.send(CompositorEvent::PopupFrame {
            window_id,
            width: pop_w,
            height: pop_h,
            bgra,
        });
    }
}

impl CompositorHandler for AppState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        if let Some(state) = client.get_data::<ClientState>() {
            return &state.compositor_state;
        }
        if let Some(state) = client.get_data::<XWaylandClientData>() {
            return &state.compositor_state;
        }
        panic!("unknown Wayland client data type");
    }

    fn commit(&mut self, surface: &WlSurface) {
        self.popups.commit(surface);

        let is_cursor_surface = self
            .cursor_surface
            .as_ref()
            .is_some_and(|cs| cs == surface);

        let mut cur = surface.clone();
        let window_id_opt = loop {
            let key = cur.id();
            if let Some(&wid) = self.surface_to_window.get(&key) {
                break Some(wid);
            }
            match get_parent(&cur) {
                Some(parent) => cur = parent,
                None => break None,
            }
        };

        let buffer_opt: Option<WlBuffer> = with_states(surface, |states| {
            let mut attrs = states
                .cached_state
                .get::<smithay::wayland::compositor::SurfaceAttributes>();
            match attrs.current().buffer.take() {
                Some(smithay::wayland::compositor::BufferAssignment::NewBuffer(buf)) => Some(buf),
                _ => None,
            }
        });

        if let Some(buffer) = buffer_opt {
            let extracted = with_buffer_contents(&buffer, |ptr, len, spec| {
                let width = spec.width.max(0) as u32;
                let height = spec.height.max(0) as u32;
                let stride = spec.stride.max(0) as usize;
                let opaque = spec.format == wl_shm::Format::Xrgb8888;
                let mut bgra = vec![0u8; (width * height * 4) as usize];
                let src = unsafe { std::slice::from_raw_parts(ptr, len) };
                for y in 0..(height as usize) {
                    let row_start = spec.offset.max(0) as usize + y * stride;
                    let row_bytes = (width as usize) * 4;
                    let dst_start = y * row_bytes;
                    if row_start + row_bytes <= src.len() {
                        bgra[dst_start..dst_start + row_bytes]
                            .copy_from_slice(&src[row_start..row_start + row_bytes]);
                        if opaque {
                            for px in bgra[dst_start..dst_start + row_bytes].chunks_exact_mut(4) {
                                px[3] = 0xFF;
                            }
                        }
                    }
                }
                (width, height, bgra)
            });

            if let Ok((w, h, bgra)) = extracted {
                if w > 0 && h > 0 {
                    self.surface_buffers.insert(surface.id(), (w, h, bgra));
                }
            }
            buffer.release();
        }

        if is_cursor_surface {
            self.emit_cursor_surface_bitmap(surface);
        }

        let Some(window_id) = window_id_opt else {
            return;
        };

        if let Some(tracked) = self.windows.get_mut(&window_id) {
            tracked.window.on_commit();
            if let Some(toplevel) = tracked.window.toplevel() {
                let root_surface = toplevel.wl_surface().clone();
                self.recomposite_window(window_id, &root_surface);
            }
        } else if self.popup_windows.contains_key(&window_id) {
            self.recomposite_popup(window_id);
        } else if self.x11_windows.contains_key(&window_id) {
            self.recomposite_x11_window(window_id, surface);
        } else if self.x11_popups.contains_key(&window_id) {
            self.recomposite_x11_popup(window_id, surface);
        }
    }
}

impl BufferHandler for AppState {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}

impl ShmHandler for AppState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl XdgShellHandler for AppState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let window_id = self.alloc_window_id();
        let surf_id = surface.wl_surface().id();

        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Activated);
        });
        surface.send_configure();

        let wl_surface = surface.wl_surface().clone();
        let window = Window::new_wayland_window(surface);
        self.space.map_element(window.clone(), (0, 0), false);
        let initial_ssd = self
            .pending_kde_decorations
            .remove(&surf_id)
            .unwrap_or(false);
        self.surface_to_window.insert(surf_id.clone(), window_id);
        self.windows.insert(
            window_id,
            TrackedWindow {
                window_id,
                window,
                last_title: String::new(),
                last_app_id: String::new(),
                last_width: 0,
                last_height: 0,
                geo_offset: (0, 0),
                server_side_decoration: initial_ssd,
                announced: false,
                has_window_entry: false,
                latest_bgra: None,
                last_sub_info: Vec::new(),
            },
        );
        self.focused_window_id = window_id;
        if let Some(keyboard) = self.seat.get_keyboard() {
            let serial = SERIAL_COUNTER.next_serial();
            keyboard.set_focus(self, Some(KeyboardFocusTarget::Wayland(wl_surface)), serial);
        }
        debug!(window_id, ?surf_id, "new xdg_toplevel registered");
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        self.surface_buffers.remove(&surf_id);
        self.pending_kde_decorations.remove(&surf_id);
        if let Some(window_id) = self.surface_to_window.remove(&surf_id) {
            if let Some(tracked) = self.windows.remove(&window_id) {
                self.space.unmap_elem(&tracked.window);
                if tracked.announced {
                    let _ = self
                        .event_tx
                        .send(CompositorEvent::WindowClosed { window_id });
                    if tracked.has_window_entry {
                        let _ = self.event_tx.send(CompositorEvent::AppState {
                            launch_id: WINDOW_LAUNCH_ID_BASE | window_id,
                            name: tracked.last_title,
                            state: LaunchState::Exited(0),
                        });
                    }
                }
            }
            debug!(window_id, "xdg_toplevel destroyed");
        }
    }

    fn move_request(&mut self, surface: ToplevelSurface, _seat: wl_seat::WlSeat, _serial: Serial) {
        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::StartMove { window_id });
        }
    }

    fn resize_request(
        &mut self,
        surface: ToplevelSurface,
        _seat: wl_seat::WlSeat,
        _serial: Serial,
        edges: xdg_toplevel::ResizeEdge,
    ) {
        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::StartResize {
                window_id,
                edge: edges as u32,
            });
        }
    }

    fn maximize_request(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            debug!(window_id, "remote app requested maximize");
            let _ = self.event_tx.send(CompositorEvent::SetMaximized {
                window_id,
                maximized: true,
            });
        }
    }

    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        surface.with_pending_state(|state| {
            state.states.unset(xdg_toplevel::State::Maximized);
        });
        surface.send_pending_configure();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::SetMaximized {
                window_id,
                maximized: false,
            });
        }
    }

    fn minimize_request(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self
                .event_tx
                .send(CompositorEvent::SetMinimized { window_id });
        }
    }

    fn fullscreen_request(
        &mut self,
        surface: ToplevelSurface,
        _output: Option<wl_output::WlOutput>,
    ) {
        let surf_id = surface.wl_surface().id();
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Fullscreen);
        });
        surface.send_pending_configure();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::SetFullscreen {
                window_id,
                fullscreen: true,
            });
        }
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        surface.with_pending_state(|state| {
            state.states.unset(xdg_toplevel::State::Fullscreen);
        });
        surface.send_pending_configure();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::SetFullscreen {
                window_id,
                fullscreen: false,
            });
        }
    }

    fn show_window_menu(
        &mut self,
        surface: ToplevelSurface,
        _seat: wl_seat::WlSeat,
        _serial: Serial,
        location: Point<i32, Logical>,
    ) {
        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let (gx, gy) = self
                .windows
                .get(&window_id)
                .map(|tw| tw.geo_offset)
                .unwrap_or((0, 0));
            let _ = self.event_tx.send(CompositorEvent::ShowWindowMenu {
                window_id,
                x: location.x - gx,
                y: location.y - gy,
            });
        }
    }

    fn new_popup(&mut self, surface: PopupSurface, positioner: PositionerState) {
        let window_id = self.alloc_window_id();
        let surf_id = surface.wl_surface().id();

        let parent_id = surface
            .get_parent_surface()
            .and_then(|p| self.surface_to_window.get(&p.id()).copied())
            .unwrap_or(1);

        let geo = positioner.get_geometry();
        surface.with_pending_state(|state| {
            state.geometry = geo;
            state.positioner = positioner;
        });
        let _ = self.popups.track_popup(PopupKind::Xdg(surface.clone()));
        let _ = surface.send_configure();
        debug!(window_id, parent_id, ?geo, "Wayland xdg_popup created");

        self.surface_to_window.insert(surf_id.clone(), window_id);
        self.popup_windows.insert(
            window_id,
            TrackedPopup {
                window_id,
                parent_id,
                surface,
                geo,
                geo_offset: (0, 0),
                announced: false,
                latest_bgra: None,
            },
        );
    }

    fn popup_destroyed(&mut self, surface: PopupSurface) {
        let surf_id = surface.wl_surface().id();
        self.surface_buffers.remove(&surf_id);
        if let Some(window_id) = self.surface_to_window.remove(&surf_id) {
            if let Some(popup) = self.popup_windows.remove(&window_id) {
                if popup.announced {
                    let _ = self
                        .event_tx
                        .send(CompositorEvent::WindowClosed { window_id });
                }
            }
        }
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: wl_seat::WlSeat, _serial: Serial) {}

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        let geo = positioner.get_geometry();
        surface.with_pending_state(|state| {
            state.geometry = geo;
        });
        surface.send_repositioned(token);

        let surf_id = surface.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            if let Some(popup) = self.popup_windows.get_mut(&window_id) {
                popup.geo = geo;
                if popup.announced {
                    let _ = self.event_tx.send(CompositorEvent::PopupRepositioned {
                        window_id,
                        x: geo.loc.x,
                        y: geo.loc.y,
                        width: geo.size.w.max(1) as u32,
                        height: geo.size.h.max(1) as u32,
                    });
                }
            }
        }
    }
}

impl SeatHandler for AppState {
    type KeyboardFocus = KeyboardFocusTarget;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&KeyboardFocusTarget>) {
        let dh = &self.display_handle;
        let client = focused
            .and_then(|f| f.wl_surface())
            .and_then(|s| dh.get_client(s.id()).ok());
        set_data_device_focus(dh, seat, client.clone());
        set_primary_focus(dh, seat, client);
    }
    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        match image {
            CursorImageStatus::Hidden => {
                self.cursor_surface = None;
                self.last_cursor_bitmap = None;
                let _ = self.event_tx.send(CompositorEvent::CursorUpdate {
                    window_id: self.focused_window_id,
                    visible: false,
                    shape_name: String::new(),
                });
            }
            CursorImageStatus::Named(icon) => {
                self.cursor_surface = None;
                self.last_cursor_bitmap = None;
                let _ = self.event_tx.send(CompositorEvent::CursorUpdate {
                    window_id: self.focused_window_id,
                    visible: true,
                    shape_name: icon.name().to_string(),
                });
            }
            CursorImageStatus::Surface(surf) => {
                self.cursor_surface = Some(surf.clone());
                if self.surface_buffers.contains_key(&surf.id()) {
                    self.emit_cursor_surface_bitmap(&surf);
                } else {
                    self.last_cursor_bitmap = None;
                    let _ = self.event_tx.send(CompositorEvent::CursorUpdate {
                        window_id: self.focused_window_id,
                        visible: true,
                        shape_name: "default".to_string(),
                    });
                }
            }
        }
    }
}

const SELECTION_IO_TIMEOUT: Duration = Duration::from_secs(3);

fn text_selection_mimes() -> Vec<String> {
    SUPPORTED_TEXT_MIMES.iter().map(|s| (*s).to_string()).collect()
}

fn pick_preferred_text_mime(mime_types: &[String]) -> Option<String> {
    SUPPORTED_TEXT_MIMES
        .iter()
        .find(|&&m| mime_types.iter().any(|t| t == m))
        .map(|&m| m.to_string())
}

/// Read up to `MAX_SELECTION_BYTES` of UTF-8 text from a selection pipe with a
/// bounded `poll(2)` deadline so a misbehaving client cannot leak threads or RAM.
fn read_selection_fd_bounded(fd: OwnedFd) -> Option<String> {
    let raw = fd.as_raw_fd();
    let deadline = Instant::now() + SELECTION_IO_TIMEOUT;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    while buf.len() < MAX_SELECTION_BYTES {
        let rem = deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            break;
        }
        let mut pfd = libc::pollfd {
            fd: raw,
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis().min(i32::MAX as u128) as i32) };
        if rc < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        if rc == 0 {
            break;
        }
        let max_read = chunk.len().min(MAX_SELECTION_BYTES - buf.len());
        let n = unsafe { libc::read(raw, chunk.as_mut_ptr().cast::<libc::c_void>(), max_read) };
        if n <= 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n as usize]);
    }
    String::from_utf8(buf).ok().filter(|s| !s.is_empty())
}

/// Write selection bytes to `fd` with a bounded `poll(2)` deadline so a stalled
/// reader cannot pin a worker thread indefinitely.
fn write_selection_fd_timeout(fd: OwnedFd, bytes: &[u8]) {
    let raw = fd.as_raw_fd();
    let deadline = Instant::now() + SELECTION_IO_TIMEOUT;
    let mut written = 0;
    while written < bytes.len() {
        let rem = deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            break;
        }
        let mut pfd = libc::pollfd {
            fd: raw,
            events: libc::POLLOUT,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, rem.as_millis().min(i32::MAX as u128) as i32) };
        if rc < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        if rc == 0 || (pfd.revents & (libc::POLLERR | libc::POLLHUP)) != 0 {
            break;
        }
        let n = unsafe {
            libc::write(
                raw,
                bytes[written..].as_ptr().cast::<libc::c_void>(),
                bytes.len() - written,
            )
        };
        if n <= 0 {
            break;
        }
        written += n as usize;
    }
}

impl SelectionHandler for AppState {
    type SelectionUserData = ();

    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        _seat: Seat<Self>,
    ) {
        let is_clipboard = ty == SelectionTarget::Clipboard;
        if is_clipboard {
            self.clipboard_text = None;
            self.clipboard_owner = if source.is_some() {
                SelectionOwner::WaylandClient
            } else {
                SelectionOwner::None
            };
        } else {
            self.primary_text = None;
            self.primary_owner = if source.is_some() {
                SelectionOwner::WaylandClient
            } else {
                SelectionOwner::None
            };
        }

        if let Some(src) = source {
            let mime_types = src.mime_types();
            if let Some(xwm) = self.xwm.as_mut() {
                let _ = xwm.new_selection(ty, Some(mime_types.clone()));
            }
            if let Some(mime) = pick_preferred_text_mime(&mime_types) {
                // Note: Smithay invokes `SelectionHandler::new_selection` BEFORE storing the new
                // `OfferReplySource::Client(source)` into `SeatData`.
                // Defer requesting client selection until right after dispatch finishes.
                if is_clipboard {
                    self.pending_clipboard_mime = Some(mime);
                } else {
                    self.pending_primary_mime = Some(mime);
                }
            }
        } else if let Some(xwm) = self.xwm.as_mut() {
            let _ = xwm.new_selection(ty, None);
        }
    }

    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        _user_data: &Self::SelectionUserData,
    ) {
        let (owner, cached) = match ty {
            SelectionTarget::Clipboard => (self.clipboard_owner, self.clipboard_text.clone()),
            SelectionTarget::Primary => (self.primary_owner, self.primary_text.clone()),
        };
        match owner {
            SelectionOwner::RemoteHost => {
                if let Some(text) = cached {
                    std::thread::spawn(move || {
                        write_selection_fd_timeout(fd, text.as_bytes());
                    });
                }
            }
            SelectionOwner::X11Client => {
                if let Some(xwm) = self.xwm.as_mut() {
                    if let Err(err) =
                        xwm.send_selection(ty, mime_type, fd, self.loop_handle.clone())
                    {
                        debug!(error = ?err, ?ty, "X11Wm::send_selection failed for Wayland client");
                    }
                }
            }
            SelectionOwner::WaylandClient | SelectionOwner::None => {}
        }
    }
}

impl AppState {
    fn poll_pending_selections(&mut self) {
        if let Some(mime) = self.pending_clipboard_mime.take() {
            if let Ok((read_pipe, write_pipe)) = std::io::pipe() {
                let write_fd: OwnedFd = write_pipe.into();
                if request_data_device_client_selection(&self.seat, mime, write_fd).is_ok() {
                    let _ = self.display_handle.flush_clients();
                    let event_tx = self.event_tx.clone();
                    let read_fd: OwnedFd = read_pipe.into();
                    std::thread::spawn(move || {
                        if let Some(text) = read_selection_fd_bounded(read_fd) {
                            let _ = event_tx.send(CompositorEvent::ClipboardText { text });
                        }
                    });
                }
            }
        }
        if let Some(mime) = self.pending_primary_mime.take() {
            if let Ok((read_pipe, write_pipe)) = std::io::pipe() {
                let write_fd: OwnedFd = write_pipe.into();
                if request_primary_client_selection(&self.seat, mime, write_fd).is_ok() {
                    let _ = self.display_handle.flush_clients();
                    let event_tx = self.event_tx.clone();
                    let read_fd: OwnedFd = read_pipe.into();
                    std::thread::spawn(move || {
                        if let Some(text) = read_selection_fd_bounded(read_fd) {
                            let _ = event_tx.send(CompositorEvent::PrimarySelectionText { text });
                        }
                    });
                }
            }
        }
    }

    fn recomposite_x11_window(&mut self, window_id: WindowId, wl_surface: &WlSurface) {
        let Some((w, h, bgra_ref)) = self.surface_buffers.get(&wl_surface.id()) else {
            return;
        };
        let (w, h) = (*w, *h);
        if w == 0 || h == 0 {
            return;
        }
        let Some(xwin) = self.x11_windows.get_mut(&window_id) else {
            return;
        };
        let mut title = xwin.surface.title();
        let app_id = x11_app_id(&xwin.surface.class(), &xwin.surface.instance());
        if title.is_empty() {
            title = if !app_id.is_empty() {
                app_id.clone()
            } else {
                format!("X11 Window #{window_id}")
            };
        }
        let title_or_app_changed = title != xwin.last_title || app_id != xwin.last_app_id;
        let size_changed = w != xwin.last_width || h != xwin.last_height;
        let frame_unchanged = xwin.announced
            && !title_or_app_changed
            && xwin
                .latest_bgra
                .as_ref()
                .is_some_and(|(lw, lh, lbgra)| *lw == w && *lh == h && lbgra == bgra_ref);
        if frame_unchanged {
            return;
        }

        let bgra = bgra_ref.clone();
        xwin.last_title = title.clone();
        xwin.last_app_id = app_id.clone();
        xwin.last_width = w;
        xwin.last_height = h;
        xwin.latest_bgra = Some((w, h, bgra.clone()));

        if !xwin.announced {
            xwin.announced = true;
            let ssd = xwin.server_side_decoration;
            let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                window_id,
                title: title.clone(),
                app_id: app_id.clone(),
                width: w,
                height: h,
            });
            let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                window_id,
                server_side: ssd,
            });

            // Link to an in-flight launched app if any, or emit a standalone window AppState.
            let mut matched_launch = None;
            for app in self.launched_apps.values_mut() {
                if app.state == LaunchState::Launching && app.linked_window_id.is_none() {
                    app.state = LaunchState::Running;
                    app.linked_window_id = Some(window_id);
                    matched_launch = Some((app.launch_id, app.name.clone()));
                    break;
                }
            }
            if let Some((lid, lname)) = matched_launch {
                let _ = self.event_tx.send(CompositorEvent::AppState {
                    launch_id: lid,
                    name: lname,
                    state: LaunchState::Running,
                });
            } else if let Some(xwin_mut) = self.x11_windows.get_mut(&window_id) {
                xwin_mut.has_window_entry = true;
                let _ = self.event_tx.send(CompositorEvent::AppState {
                    launch_id: WINDOW_LAUNCH_ID_BASE | window_id,
                    name: title,
                    state: LaunchState::Running,
                });
            }
        } else {
            if title_or_app_changed {
                let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                    window_id,
                    title,
                    app_id,
                    width: w,
                    height: h,
                });
            }
            if size_changed {
                let _ = self.event_tx.send(CompositorEvent::WindowConfigured {
                    window_id,
                    width: w,
                    height: h,
                    scale_factor: 1.0,
                });
            }
        }

        let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
            window_id,
            width: w,
            height: h,
            bgra,
        });
    }

    fn recomposite_x11_popup(&mut self, window_id: WindowId, wl_surface: &WlSurface) {
        let Some((w, h, bgra_ref)) = self.surface_buffers.get(&wl_surface.id()) else {
            return;
        };
        let (w, h) = (*w, *h);
        if w == 0 || h == 0 {
            return;
        }
        let Some(xpop) = self.x11_popups.get_mut(&window_id) else {
            return;
        };
        if xpop.announced
            && xpop
                .latest_bgra
                .as_ref()
                .is_some_and(|(lw, lh, lbgra)| *lw == w && *lh == h && lbgra == bgra_ref)
        {
            return;
        }
        let bgra = bgra_ref.clone();
        xpop.latest_bgra = Some((w, h, bgra.clone()));
        if !xpop.announced {
            xpop.announced = true;
            debug!(
                window_id,
                parent_id = xpop.parent_id,
                x = xpop.rel_pos.0,
                y = xpop.rel_pos.1,
                w,
                h,
                "emitting PopupCreated for X11 popup"
            );
            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                window_id,
                parent_id: xpop.parent_id,
                x: xpop.rel_pos.0,
                y: xpop.rel_pos.1,
                width: w,
                height: h,
            });
        }
        let _ = self.event_tx.send(CompositorEvent::PopupFrame {
            window_id,
            width: w,
            height: h,
            bgra,
        });
    }

    fn find_x11_parent(&self, window: &X11Surface) -> Option<(WindowId, Rectangle<i32, Logical>)> {
        if let Some(transient_x11) = window.is_transient_for() {
            if let Some(&pid) = self.x11_id_by_win.get(&transient_x11) {
                if let Some(parent) = self.x11_windows.get(&pid).filter(|p| p.announced) {
                    return Some((pid, parent.geo));
                }
            }
        }
        if let Some(pid) = window.pid() {
            if let Some(focused_id) = self.last_focused_x11_window_id {
                if let Some(parent) = self.x11_windows.get(&focused_id).filter(|p| p.announced) {
                    if parent.surface.pid() == Some(pid) {
                        return Some((focused_id, parent.geo));
                    }
                }
            }
            for (&wid, parent) in &self.x11_windows {
                if parent.announced && parent.surface.pid() == Some(pid) {
                    return Some((wid, parent.geo));
                }
            }
        }
        if let Some(focused_id) = self.last_focused_x11_window_id {
            if let Some(parent) = self.x11_windows.get(&focused_id).filter(|p| p.announced) {
                return Some((focused_id, parent.geo));
            }
        }
        self.x11_windows
            .iter()
            .find(|(_, parent)| parent.announced)
            .map(|(&wid, parent)| (wid, parent.geo))
    }

    fn remove_x11_window(&mut self, x11_win: X11Window) {
        let Some(window_id) = self.x11_id_by_win.remove(&x11_win) else {
            return;
        };
        if let Some(xwin) = self.x11_windows.remove(&window_id) {
            if let Some(wl) = xwin.surface.wl_surface() {
                self.surface_to_window.remove(&wl.id());
                self.surface_buffers.remove(&wl.id());
            }
            if self.last_focused_x11_window_id == Some(window_id) {
                self.last_focused_x11_window_id = self.x11_windows.keys().next().copied();
            }
            if xwin.has_window_entry {
                let _ = self.event_tx.send(CompositorEvent::AppState {
                    launch_id: WINDOW_LAUNCH_ID_BASE | window_id,
                    name: xwin.last_title,
                    state: LaunchState::Exited(0),
                });
            }
            if xwin.announced {
                let _ = self
                    .event_tx
                    .send(CompositorEvent::WindowClosed { window_id });
            }
        } else if let Some(xpop) = self.x11_popups.remove(&window_id) {
            debug!(
                window_id,
                x11_win,
                announced = xpop.announced,
                "removing X11 popup"
            );
            if let Some(wl) = xpop.surface.wl_surface() {
                self.surface_to_window.remove(&wl.id());
                self.surface_buffers.remove(&wl.id());
            }
            if xpop.announced {
                let _ = self
                    .event_tx
                    .send(CompositorEvent::WindowClosed { window_id });
            }
        }
    }
}

impl PrimarySelectionHandler for AppState {
    fn primary_selection_state(&self) -> &PrimarySelectionState {
        &self.primary_selection_state
    }
}

impl XWaylandShellHandler for AppState {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm: XwmId, wl_surface: WlSurface, surface: X11Surface) {
        let x11_win = surface.window_id();
        debug!(
            x11_win,
            wl_id = ?wl_surface.id(),
            tracked = self.x11_id_by_win.contains_key(&x11_win),
            "XWayland surface_associated"
        );
        if let Some(&window_id) = self.x11_id_by_win.get(&x11_win) {
            self.surface_to_window.insert(wl_surface.id(), window_id);
            if self.x11_windows.contains_key(&window_id) {
                if let Some(keyboard) = self.seat.get_keyboard() {
                    let target = KeyboardFocusTarget::X11(surface.clone());
                    if keyboard.current_focus().as_ref() != Some(&target) {
                        keyboard.set_focus(self, Some(target), SERIAL_COUNTER.next_serial());
                    }
                }
                self.recomposite_x11_window(window_id, &wl_surface);
            } else if self.x11_popups.contains_key(&window_id) {
                self.recomposite_x11_popup(window_id, &wl_surface);
            }
        }
    }
}

fn x11_app_id(class: &str, instance: &str) -> String {
    if !class.is_empty() && !instance.is_empty() && class.eq_ignore_ascii_case(instance) {
        instance.to_string()
    } else if !class.is_empty() {
        class.to_string()
    } else {
        instance.to_string()
    }
}

fn x11_resize_edge_to_xdg(edge: X11ResizeEdge) -> u32 {
    match edge {
        X11ResizeEdge::Top => xdg_toplevel::ResizeEdge::Top as u32,
        X11ResizeEdge::Bottom => xdg_toplevel::ResizeEdge::Bottom as u32,
        X11ResizeEdge::Left => xdg_toplevel::ResizeEdge::Left as u32,
        X11ResizeEdge::Right => xdg_toplevel::ResizeEdge::Right as u32,
        X11ResizeEdge::TopLeft => xdg_toplevel::ResizeEdge::TopLeft as u32,
        X11ResizeEdge::TopRight => xdg_toplevel::ResizeEdge::TopRight as u32,
        X11ResizeEdge::BottomLeft => xdg_toplevel::ResizeEdge::BottomLeft as u32,
        X11ResizeEdge::BottomRight => xdg_toplevel::ResizeEdge::BottomRight as u32,
    }
}

impl XwmHandler for AppState {
    fn xwm_state(&mut self, _xwm: XwmId) -> &mut X11Wm {
        self.xwm.as_mut().expect("X11Wm not initialized")
    }

    fn new_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn new_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn map_window_request(&mut self, _xwm: XwmId, window: X11Surface) {
        let x11_win = window.window_id();
        if self.x11_id_by_win.contains_key(&x11_win) {
            let _ = window.set_mapped(true);
            return;
        }
        let mut geo = window.geometry();
        let is_popup_type = matches!(
            window.window_type(),
            Some(
                WmWindowType::PopupMenu
                    | WmWindowType::DropdownMenu
                    | WmWindowType::Menu
                    | WmWindowType::Tooltip
                    | WmWindowType::Notification
            )
        );
        if is_popup_type {
            if let Some((parent_id, parent_geo)) = self.find_x11_parent(&window) {
                if let Err(err) = window.set_mapped(true) {
                    warn!(error = ?err, x11_win, "failed to map X11 popup window");
                    return;
                }
                if geo.size.w > 0 && geo.size.h > 0 {
                    let _ = window.configure(geo);
                }
                let window_id = self.alloc_window_id();
                let wl_opt = window.wl_surface();
                if let Some(ref wl) = wl_opt {
                    self.surface_to_window.insert(wl.id(), window_id);
                }
                self.x11_id_by_win.insert(x11_win, window_id);
                let rel_pos = (geo.loc.x - parent_geo.loc.x, geo.loc.y - parent_geo.loc.y);
                debug!(
                    window_id,
                    parent_id,
                    x11_win,
                    ?rel_pos,
                    w = geo.size.w,
                    h = geo.size.h,
                    "mapped managed X11 popup window"
                );
                self.x11_popups.insert(
                    window_id,
                    TrackedX11Popup {
                        window_id,
                        parent_id,
                        surface: window,
                        geo,
                        rel_pos,
                        announced: false,
                        latest_bgra: None,
                    },
                );
                if let Some(wl) = wl_opt {
                    self.recomposite_x11_popup(window_id, &wl);
                }
                return;
            }
        }

        if geo.size.w <= 0 {
            geo.size.w = 800;
        }
        if geo.size.h <= 0 {
            geo.size.h = 600;
        }
        let (px, py) = self.alloc_x11_position(geo.size.w, geo.size.h);
        geo.loc = (px, py).into();

        if let Err(err) = window.set_mapped(true) {
            warn!(error = ?err, x11_win, "failed to map X11 window");
            return;
        }
        let _ = window.configure(geo);
        let _ = window.set_activated(true);
        if let Some(xwm) = self.xwm.as_mut() {
            let _ = xwm.raise_window(&window);
        }
        if let Some(keyboard) = self.seat.get_keyboard() {
            keyboard.set_focus(
                self,
                Some(KeyboardFocusTarget::X11(window.clone())),
                SERIAL_COUNTER.next_serial(),
            );
        }

        let window_id = self.alloc_window_id();
        // Smithay's `X11Surface::is_decorated()` returns `true` when the X11 client
        // draws its own decorations (`_MOTIF_WM_HINTS` decorations == 0), so SSD is the inverse.
        let is_csd = window.is_decorated();
        let ssd = !is_csd;
        debug!(
            window_id,
            x11_win,
            class = %window.class(),
            instance = %window.instance(),
            is_csd,
            ssd,
            "X11 map_window_request decoration mode"
        );
        let wl_opt = window.wl_surface();
        if let Some(ref wl) = wl_opt {
            self.surface_to_window.insert(wl.id(), window_id);
        }
        self.x11_id_by_win.insert(x11_win, window_id);
        self.last_focused_x11_window_id = Some(window_id);
        self.x11_windows.insert(
            window_id,
            TrackedX11Window {
                window_id,
                surface: window,
                geo,
                last_title: String::new(),
                last_app_id: String::new(),
                last_width: geo.size.w as u32,
                last_height: geo.size.h as u32,
                server_side_decoration: ssd,
                announced: false,
                has_window_entry: false,
                latest_bgra: None,
            },
        );
        if let Some(wl) = wl_opt {
            self.recomposite_x11_window(window_id, &wl);
        }
    }

    fn mapped_override_redirect_window(&mut self, _xwm: XwmId, window: X11Surface) {
        let x11_win = window.window_id();
        if self.x11_id_by_win.contains_key(&x11_win) {
            return;
        }
        let geo = window.geometry();
        let window_id = self.alloc_window_id();
        let wl_opt = window.wl_surface();
        if let Some(ref wl) = wl_opt {
            self.surface_to_window.insert(wl.id(), window_id);
        }
        self.x11_id_by_win.insert(x11_win, window_id);

        if let Some((parent_id, parent_geo)) = self.find_x11_parent(&window) {
            let rel_pos = (geo.loc.x - parent_geo.loc.x, geo.loc.y - parent_geo.loc.y);
            debug!(
                window_id,
                parent_id,
                x11_win,
                win_type = ?window.window_type(),
                ?geo,
                ?parent_geo,
                ?rel_pos,
                has_wl = wl_opt.is_some(),
                "mapped X11 override-redirect popup"
            );
            self.x11_popups.insert(
                window_id,
                TrackedX11Popup {
                    window_id,
                    parent_id,
                    surface: window,
                    geo,
                    rel_pos,
                    announced: false,
                    latest_bgra: None,
                },
            );
            if let Some(wl) = wl_opt {
                self.recomposite_x11_popup(window_id, &wl);
            }
        } else {
            // Orphaned override-redirect window (e.g., splash screen or dmenu) -> borderless toplevel.
            self.x11_windows.insert(
                window_id,
                TrackedX11Window {
                    window_id,
                    surface: window,
                    geo,
                    last_title: String::new(),
                    last_app_id: String::new(),
                    last_width: geo.size.w.max(1) as u32,
                    last_height: geo.size.h.max(1) as u32,
                    server_side_decoration: false,
                    announced: false,
                    has_window_entry: false,
                    latest_bgra: None,
                },
            );
            if let Some(wl) = wl_opt {
                self.recomposite_x11_window(window_id, &wl);
            }
        }
    }

    fn unmapped_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.remove_x11_window(window.window_id());
    }

    fn destroyed_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.remove_x11_window(window.window_id());
    }

    fn configure_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        x: Option<i32>,
        y: Option<i32>,
        w: Option<u32>,
        h: Option<u32>,
        _reorder: Option<Reorder>,
    ) {
        let mut geo = window.geometry();
        if let Some(x) = x {
            geo.loc.x = x;
        }
        if let Some(y) = y {
            geo.loc.y = y;
        }
        if let Some(w) = w {
            geo.size.w = w.max(1) as i32;
        }
        if let Some(h) = h {
            geo.size.h = h.max(1) as i32;
        }
        self.ensure_virtual_output_size(geo.loc.x + geo.size.w + 64, geo.loc.y + geo.size.h + 64);
        let _ = window.configure(geo);
        if let Some(&wid) = self.x11_id_by_win.get(&window.window_id()) {
            if let Some(xwin) = self.x11_windows.get_mut(&wid) {
                xwin.geo = geo;
            }
        }
    }

    fn configure_notify(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        geometry: Rectangle<i32, Logical>,
        _above: Option<X11Window>,
    ) {
        let Some(&wid) = self.x11_id_by_win.get(&window.window_id()) else {
            return;
        };
        if let Some(xwin) = self.x11_windows.get_mut(&wid) {
            xwin.geo = geometry;
        } else if let Some(xpop) = self.x11_popups.get(&wid) {
            let parent_id = xpop.parent_id;
            let parent_loc = self
                .x11_windows
                .get(&parent_id)
                .map(|p| p.geo.loc)
                .unwrap_or_else(|| (0, 0).into());
            let new_rel = (geometry.loc.x - parent_loc.x, geometry.loc.y - parent_loc.y);
            if let Some(xpop_mut) = self.x11_popups.get_mut(&wid) {
                let changed = xpop_mut.rel_pos != new_rel || xpop_mut.geo.size != geometry.size;
                xpop_mut.geo = geometry;
                xpop_mut.rel_pos = new_rel;
                if xpop_mut.announced && changed {
                    let _ = self.event_tx.send(CompositorEvent::PopupRepositioned {
                        window_id: wid,
                        x: new_rel.0,
                        y: new_rel.1,
                        width: geometry.size.w.max(1) as u32,
                        height: geometry.size.h.max(1) as u32,
                    });
                }
            }
        }
    }

    fn property_notify(&mut self, _xwm: XwmId, window: X11Surface, property: WmWindowProperty) {
        if !matches!(
            property,
            WmWindowProperty::Title | WmWindowProperty::Class | WmWindowProperty::MotifHints
        ) {
            return;
        }
        let Some(&wid) = self.x11_id_by_win.get(&window.window_id()) else {
            return;
        };
        let Some(xwin) = self.x11_windows.get_mut(&wid) else {
            return;
        };
        if property == WmWindowProperty::MotifHints {
            let is_csd = window.is_decorated();
            let ssd = !is_csd;
            debug!(
                window_id = wid,
                class = %window.class(),
                instance = %window.instance(),
                is_csd,
                ssd,
                prev_ssd = xwin.server_side_decoration,
                "X11 MotifHints updated"
            );
            if ssd != xwin.server_side_decoration {
                xwin.server_side_decoration = ssd;
                if xwin.announced {
                    let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                        window_id: wid,
                        server_side: ssd,
                    });
                }
            }
            return;
        }
        let mut title = window.title();
        let app_id = x11_app_id(&window.class(), &window.instance());
        if title.is_empty() {
            title = if !app_id.is_empty() {
                app_id.clone()
            } else {
                format!("X11 Window #{wid}")
            };
        }
        if title != xwin.last_title || app_id != xwin.last_app_id {
            xwin.last_title = title.clone();
            xwin.last_app_id = app_id.clone();
            if xwin.announced {
                let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                    window_id: wid,
                    title,
                    app_id,
                    width: xwin.last_width,
                    height: xwin.last_height,
                });
            }
        }
    }

    fn maximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self.event_tx.send(CompositorEvent::SetMaximized {
                window_id,
                maximized: true,
            });
        }
    }

    fn unmaximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self.event_tx.send(CompositorEvent::SetMaximized {
                window_id,
                maximized: false,
            });
        }
    }

    fn fullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self.event_tx.send(CompositorEvent::SetFullscreen {
                window_id,
                fullscreen: true,
            });
        }
    }

    fn unfullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self.event_tx.send(CompositorEvent::SetFullscreen {
                window_id,
                fullscreen: false,
            });
        }
    }

    fn minimize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self
                .event_tx
                .send(CompositorEvent::SetMinimized { window_id });
        }
    }

    fn move_request(&mut self, _xwm: XwmId, window: X11Surface, _button: u32) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self
                .event_tx
                .send(CompositorEvent::StartMove { window_id });
        }
    }

    fn resize_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        _button: u32,
        resize_edge: X11ResizeEdge,
    ) {
        if let Some(&window_id) = self.x11_id_by_win.get(&window.window_id()) {
            let _ = self.event_tx.send(CompositorEvent::StartResize {
                window_id,
                edge: x11_resize_edge_to_xdg(resize_edge),
            });
        }
    }

    fn allow_selection_access(&mut self, _xwm: XwmId, _selection: SelectionTarget) -> bool {
        true
    }

    fn send_selection(
        &mut self,
        _xwm: XwmId,
        selection: SelectionTarget,
        mime_type: String,
        fd: OwnedFd,
    ) {
        let (owner, cached) = match selection {
            SelectionTarget::Clipboard => (self.clipboard_owner, self.clipboard_text.clone()),
            SelectionTarget::Primary => (self.primary_owner, self.primary_text.clone()),
        };
        match owner {
            SelectionOwner::RemoteHost => {
                if let Some(text) = cached {
                    std::thread::spawn(move || {
                        write_selection_fd_timeout(fd, text.as_bytes());
                    });
                }
            }
            SelectionOwner::WaylandClient => match selection {
                SelectionTarget::Clipboard => {
                    let _ = request_data_device_client_selection(&self.seat, mime_type, fd);
                    let _ = self.display_handle.flush_clients();
                }
                SelectionTarget::Primary => {
                    let _ = request_primary_client_selection(&self.seat, mime_type, fd);
                    let _ = self.display_handle.flush_clients();
                }
            },
            SelectionOwner::X11Client | SelectionOwner::None => {}
        }
    }

    fn new_selection(
        &mut self,
        _xwm: XwmId,
        selection: SelectionTarget,
        mime_types: Vec<String>,
    ) {
        match selection {
            SelectionTarget::Clipboard => {
                self.clipboard_owner = SelectionOwner::X11Client;
                self.clipboard_text = None;
                set_data_device_selection(
                    &self.display_handle,
                    &self.seat,
                    mime_types.clone(),
                    (),
                );
            }
            SelectionTarget::Primary => {
                self.primary_owner = SelectionOwner::X11Client;
                self.primary_text = None;
                set_primary_selection(&self.display_handle, &self.seat, mime_types.clone(), ());
            }
        }

        if let Some(mime) = pick_preferred_text_mime(&mime_types) {
            if let Ok((read_pipe, write_pipe)) = std::io::pipe() {
                let write_fd: OwnedFd = write_pipe.into();
                if let Some(xwm) = self.xwm.as_mut() {
                    if xwm
                        .send_selection(selection, mime, write_fd, self.loop_handle.clone())
                        .is_ok()
                    {
                        let event_tx = self.event_tx.clone();
                        let read_fd: OwnedFd = read_pipe.into();
                        std::thread::spawn(move || {
                            if let Some(text) = read_selection_fd_bounded(read_fd) {
                                let ev = match selection {
                                    SelectionTarget::Clipboard => {
                                        CompositorEvent::ClipboardText { text }
                                    }
                                    SelectionTarget::Primary => {
                                        CompositorEvent::PrimarySelectionText { text }
                                    }
                                };
                                let _ = event_tx.send(ev);
                            }
                        });
                    }
                }
            }
        }
    }

    fn cleared_selection(&mut self, _xwm: XwmId, selection: SelectionTarget) {
        match selection {
            SelectionTarget::Clipboard => {
                if self.clipboard_owner == SelectionOwner::X11Client {
                    self.clipboard_owner = SelectionOwner::None;
                }
            }
            SelectionTarget::Primary => {
                if self.primary_owner == SelectionOwner::X11Client {
                    self.primary_owner = SelectionOwner::None;
                }
            }
        }
    }

    fn disconnected(&mut self, _xwm: XwmId) {
        warn!("X11Wm lost connection to XWayland; clearing X11 session state");
        self.cleanup_xwayland_state();
    }
}

impl DataDeviceHandler for AppState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

impl ClientDndGrabHandler for AppState {}
impl ServerDndGrabHandler for AppState {
    fn send(&mut self, _mime_type: String, _fd: OwnedFd, _seat: Seat<Self>) {}
}

impl OutputHandler for AppState {}
impl TabletSeatHandler for AppState {}

impl AppState {
    fn update_window_decoration_mode(
        &mut self,
        surface: &WlSurface,
        server_side: bool,
        record_pending: bool,
    ) {
        let surf_id = surface.id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            if let Some(tracked) = self.windows.get_mut(&window_id) {
                if tracked.server_side_decoration != server_side {
                    tracked.server_side_decoration = server_side;
                    if tracked.announced {
                        let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                            window_id,
                            server_side,
                        });
                    }
                }
            }
        } else if record_pending {
            self.pending_kde_decorations.insert(surf_id, server_side);
        }
    }
}

impl XdgDecorationHandler for AppState {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        // Default to CSD until the client explicitly requests SSD via `request_mode`.
        // Qt/KDE apps (konsole, dolphin, kcalc, kwrite) explicitly call `set_mode(ServerSide)`,
        // whereas GTK3 apps (e.g., lxterminal) draw their own CSD header/titlebar if not told CSD.
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ClientSide);
        });
        toplevel.send_pending_configure();
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: DecorationMode) {
        let server_side = mode == DecorationMode::ServerSide;
        debug!(?mode, server_side, "xdg-decoration request_mode");
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(mode);
        });
        toplevel.send_pending_configure();
        self.update_window_decoration_mode(toplevel.wl_surface(), server_side, false);
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ClientSide);
        });
        toplevel.send_pending_configure();
        self.update_window_decoration_mode(toplevel.wl_surface(), false, false);
    }
}

impl KdeDecorationHandler for AppState {
    fn kde_decoration_state(&self) -> &KdeDecorationState {
        &self.kde_decoration_state
    }

    fn new_decoration(&mut self, surface: &WlSurface, decoration: &OrgKdeKwinServerDecoration) {
        // Advertise Mode::Server by default (matching KWin) so GTK3 apps loaded with
        // `window-decorations-gtk-module` (e.g. lxterminal) drop their Adwaita CSD headerbar.
        debug!(surf_id = ?surface.id(), "kde-decoration new_decoration -> Mode::Server");
        decoration.mode(KdeDecorationMode::Server);
        self.update_window_decoration_mode(surface, true, true);
    }

    fn request_mode(
        &mut self,
        surface: &WlSurface,
        decoration: &OrgKdeKwinServerDecoration,
        mode: WEnum<KdeDecorationMode>,
    ) {
        let WEnum::Value(mode) = mode else {
            return;
        };
        debug!(surf_id = ?surface.id(), ?mode, "kde-decoration request_mode");
        decoration.mode(mode);
        self.update_window_decoration_mode(surface, mode == KdeDecorationMode::Server, true);
    }

    fn release(&mut self, _decoration: &OrgKdeKwinServerDecoration, surface: &WlSurface) {
        self.pending_kde_decorations.remove(&surface.id());
        self.update_window_decoration_mode(surface, false, false);
    }
}

delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_xdg_shell!(AppState);
delegate_xdg_decoration!(AppState);
delegate_kde_decoration!(AppState);
delegate_seat!(AppState);
delegate_data_device!(AppState);
delegate_primary_selection!(AppState);
delegate_cursor_shape!(AppState);
delegate_output!(AppState);
delegate_xwayland_shell!(AppState);

/// Start the headless Smithay Wayland compositor on a background OS thread
/// and optionally launch `command` inside the isolated session.
pub fn start_compositor(
    command: Vec<String>,
    enable_xwayland: bool,
) -> Result<(CompositorHandle, mpsc::UnboundedReceiver<CompositorEvent>)> {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx): (CalloopSender<CompositorCommand>, Channel<CompositorCommand>) =
        channel::channel();

    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<OsString>>(1);
    let cmd_tx_loop = cmd_tx.clone();

    std::thread::Builder::new()
        .name("rw-compositor".into())
        .spawn(move || {
            if let Err(err) = run_compositor_loop(
                event_tx,
                cmd_rx,
                cmd_tx_loop,
                ready_tx.clone(),
                command,
                enable_xwayland,
            ) {
                error!(error = %err, "headless compositor exited with error");
                let _ = ready_tx.send(Err(err));
            }
        })
        .context("failed to spawn compositor thread")?;

    let socket_name = ready_rx
        .recv()
        .context("compositor thread terminated before signaling ready")??;

    Ok((
        CompositorHandle {
            socket_name,
            cmd_tx,
        },
        event_rx,
    ))
}

fn spawn_xwayland_with_auth(
    dh: &DisplayHandle,
) -> Result<Option<(XWayland, Client, Option<PathBuf>)>> {
    let runtime_dir = crate::session::session_runtime_dir();
    let shim_dir = runtime_dir.as_ref().map(|d| d.join("bin"));
    let Some(real_xwayland) = crate::session::find_xwayland_binary(shim_dir.as_deref()) else {
        warn!("Xwayland binary not found in PATH; continuing in Wayland-only mode");
        return Ok(None);
    };

    let xauth_and_path = if let Some(ref rdir) = runtime_dir {
        let xauth = rdir.join("Xauthority");
        let bin_dir = crate::session::write_xwayland_auth_shim(rdir, &real_xwayland, &xauth)?;
        let mut new_path = OsString::from(&bin_dir);
        if let Some(old_path) = std::env::var_os("PATH") {
            new_path.push(":");
            new_path.push(old_path);
        }
        Some((xauth, new_path))
    } else {
        None
    };

    // Scan free X11 display slots so we can pre-generate an exact MIT-MAGIC-COOKIE-1
    // Xauthority file for the chosen display number before launching Xwayland.
    for display_num in 0..64u32 {
        let mut envs: Vec<(OsString, OsString)> = Vec::new();
        let xauth_path = if let Some((ref xauth, ref new_path)) = xauth_and_path {
            crate::session::write_xauthority_file(xauth, display_num)?;
            envs.push((OsString::from("PATH"), new_path.clone()));
            Some(xauth.clone())
        } else {
            None
        };

        match XWayland::spawn(
            dh,
            Some(display_num),
            envs,
            false, // disable abstract Unix sockets for multi-user isolation
            Stdio::null(),
            Stdio::null(),
            |_| {},
        ) {
            Ok((xwayland, client)) => {
                // Defense-in-depth: restrict filesystem X11 socket inode to owner-only (0600).
                let sock_path = PathBuf::from(format!("/tmp/.X11-unix/X{display_num}"));
                let _ = crate::session::set_mode(&sock_path, 0o600);
                return Ok(Some((xwayland, client, xauth_path)));
            }
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::AddrInUse
                        | std::io::ErrorKind::AlreadyExists
                        | std::io::ErrorKind::PermissionDenied
                ) =>
            {
                continue;
            }
            Err(err) => {
                return Err(err).context("failed to spawn XWayland");
            }
        }
    }

    anyhow::bail!("no free X11 display slot available in range :0..:63")
}

fn run_compositor_loop(
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
    cmd_rx: Channel<CompositorCommand>,
    cmd_tx: CalloopSender<CompositorCommand>,
    ready_tx: std::sync::mpsc::SyncSender<Result<OsString>>,
    command: Vec<String>,
    enable_xwayland: bool,
) -> Result<()> {
    let mut event_loop: EventLoop<AppState> =
        EventLoop::try_new().context("failed to create calloop EventLoop")?;
    let loop_handle = event_loop.handle();
    let display: Display<AppState> =
        Display::new().context("failed to create Wayland Display")?;
    let dh = display.handle();

    let compositor_state = CompositorState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    let xdg_shell_state = XdgShellState::new::<AppState>(&dh);
    let xdg_decoration_state = XdgDecorationState::new::<AppState>(&dh);
    let kde_decoration_state =
        KdeDecorationState::new::<AppState>(&dh, KdeDefaultDecorationMode::Server);
    let mut seat_state = SeatState::new();
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let primary_selection_state = PrimarySelectionState::new::<AppState>(&dh);
    let cursor_shape_state = CursorShapeManagerState::new::<AppState>(&dh);
    let output_manager_state = OutputManagerState::new_with_xdg_output::<AppState>(&dh);
    let xwayland_shell_state = XWaylandShellState::new::<AppState>(&dh);

    let mut seat = seat_state.new_wl_seat(&dh, "remotewayland-seat");
    seat.add_keyboard(XkbConfig::default(), 200, 25)
        .context("failed to add virtual keyboard to seat")?;
    seat.add_pointer();

    let output = Output::new(
        "RW-HEADLESS-1".into(),
        PhysicalProperties {
            size: (0, 0).into(),
            subpixel: Subpixel::Unknown,
            make: "RemoteWayland".into(),
            model: "Headless".into(),
        },
    );
    let mode = OutputMode {
        size: (1920, 1080).into(),
        refresh: 60_000,
    };
    output.change_current_state(Some(mode), None, None, Some((0, 0).into()));
    output.set_preferred(mode);
    output.create_global::<AppState>(&dh);

    let mut space = Space::default();
    space.map_output(&output, (0, 0));

    let socket_source =
        ListeningSocketSource::new_auto().context("failed to bind Wayland listening socket")?;
    let socket_name = socket_source.socket_name().to_os_string();
    info!(?socket_name, "headless Wayland compositor listening");

    let mut xwayland_spawned = false;
    let mut xwayland_client: Option<Client> = None;
    if enable_xwayland {
        match spawn_xwayland_with_auth(&dh) {
            Ok(Some((xwayland, client, xauth_path))) => {
                xwayland_spawned = true;
                xwayland_client = Some(client.clone());
                let xwm_loop_handle = loop_handle.clone();
                if let Err(err) = loop_handle.insert_source(xwayland, move |event, _, state| {
                    match event {
                        XWaylandEvent::Ready {
                            x11_socket,
                            display_number,
                        } => {
                            let disp = format!(":{display_number}");
                            let sock_path =
                                PathBuf::from(format!("/tmp/.X11-unix/X{display_number}"));
                            let _ = crate::session::set_mode(&sock_path, 0o600);
                            crate::session::set_x11_env(disp.clone(), xauth_path.clone());
                            match X11Wm::start_wm(
                                xwm_loop_handle.clone(),
                                x11_socket,
                                client.clone(),
                            ) {
                                Ok(wm) => {
                                    state.xwm = Some(wm);
                                    info!(
                                        display = %disp,
                                        xauthority = ?xauth_path,
                                        "embedded XWayland window manager ready"
                                    );
                                }
                                Err(err) => {
                                    warn!(
                                        error = ?err,
                                        display = %disp,
                                        "failed to attach X11 window manager to XWayland"
                                    );
                                }
                            }
                            if !state.pending_initial_command.is_empty() {
                                let cmd = std::mem::take(&mut state.pending_initial_command);
                                spawn_gui_command(&state.socket_name, &cmd);
                            }
                        }
                        XWaylandEvent::Error => {
                            warn!("XWayland exited unexpectedly; clearing X11 state and continuing Wayland-only");
                            state.cleanup_xwayland_state();
                            if !state.pending_initial_command.is_empty() {
                                let cmd = std::mem::take(&mut state.pending_initial_command);
                                spawn_gui_command(&state.socket_name, &cmd);
                            }
                        }
                    }
                }) {
                    warn!(error = ?err, "failed to insert XWayland event source; continuing Wayland-only");
                    xwayland_spawned = false;
                }
            }
            Ok(None) => {}
            Err(err) => {
                warn!(error = %err, "failed to start XWayland; continuing in Wayland-only mode");
            }
        }
    }

    let pending_initial_command = if xwayland_spawned {
        command
    } else {
        if !command.is_empty() {
            spawn_gui_command(&socket_name, &command);
        }
        Vec::new()
    };

    let mut state = AppState {
        display_handle: dh.clone(),
        loop_handle: loop_handle.clone(),
        socket_name: socket_name.clone(),
        cmd_tx,
        compositor_state,
        shm_state,
        xdg_shell_state,
        xdg_decoration_state,
        kde_decoration_state,
        seat_state,
        data_device_state,
        primary_selection_state,
        cursor_shape_state,
        output_manager_state,
        xwayland_shell_state,
        xwm: None,
        xwayland_client,
        output,
        output_size: (1920, 1080),
        seat,
        space,
        popups: PopupManager::default(),
        next_window_id: 1,
        focused_window_id: 1,
        last_focused_x11_window_id: None,
        next_x11_pos: (100, 100),
        cursor_surface: None,
        last_cursor_bitmap: None,
        clipboard_owner: SelectionOwner::None,
        clipboard_text: None,
        pending_clipboard_mime: None,
        primary_owner: SelectionOwner::None,
        primary_text: None,
        pending_primary_mime: None,
        pending_initial_command,
        start_time: Instant::now(),
        surface_to_window: HashMap::new(),
        surface_buffers: HashMap::new(),
        pending_kde_decorations: HashMap::new(),
        windows: HashMap::new(),
        popup_windows: HashMap::new(),
        x11_windows: HashMap::new(),
        x11_popups: HashMap::new(),
        x11_id_by_win: HashMap::new(),
        launched_apps: BTreeMap::new(),
        next_server_launch_id: 1_000_000,
        event_tx,
    };

    event_loop
        .handle()
        .insert_source(socket_source, |client_stream, _, state| {
            if let Err(err) = state
                .display_handle
                .insert_client(client_stream, Arc::new(ClientState::default()))
            {
                warn!(error = %err, "failed to insert Wayland client");
            }
        })
        .context("failed to insert Wayland socket source")?;

    event_loop
        .handle()
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            |_, display, state| {
                unsafe {
                    if let Err(err) = display.get_mut().dispatch_clients(state) {
                        warn!(error = %err, "Wayland dispatch_clients error");
                    }
                }
                state.poll_pending_selections();
                Ok(PostAction::Continue)
            },
        )
        .context("failed to insert Wayland display source")?;

    event_loop
        .handle()
        .insert_source(cmd_rx, |event, _, state| {
            if let channel::Event::Msg(cmd) = event {
                state.handle_command(cmd);
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert compositor command channel: {e:?}"))?;

    let _ = ready_tx.send(Ok(socket_name.clone()));

    let initial_cmd_deadline = Instant::now() + Duration::from_secs(3);

    loop {
        event_loop.dispatch(Duration::from_millis(16), &mut state)?;
        if !state.pending_initial_command.is_empty() && Instant::now() >= initial_cmd_deadline {
            warn!("XWayland startup timed out; launching initial command without waiting");
            let cmd = std::mem::take(&mut state.pending_initial_command);
            spawn_gui_command(&state.socket_name, &cmd);
        }
        let elapsed = state.start_time.elapsed();
        for tracked in state.windows.values() {
            tracked.window.send_frame(
                &state.output,
                elapsed,
                Some(Duration::ZERO),
                |_, _| Some(state.output.clone()),
            );
        }
        for popup in state.popup_windows.values() {
            send_frames_surface_tree(
                popup.surface.wl_surface(),
                &state.output,
                elapsed,
                Some(Duration::ZERO),
                |_, _| Some(state.output.clone()),
            );
        }
        for xwin in state.x11_windows.values() {
            if let Some(wl) = xwin.surface.wl_surface() {
                send_frames_surface_tree(
                    &wl,
                    &state.output,
                    elapsed,
                    Some(Duration::ZERO),
                    |_, _| Some(state.output.clone()),
                );
            }
        }
        for xpop in state.x11_popups.values() {
            if let Some(wl) = xpop.surface.wl_surface() {
                send_frames_surface_tree(
                    &wl,
                    &state.output,
                    elapsed,
                    Some(Duration::ZERO),
                    |_, _| Some(state.output.clone()),
                );
            }
        }
        if let Some(ref cs) = state.cursor_surface {
            if cs.alive() {
                send_frames_surface_tree(
                    cs,
                    &state.output,
                    elapsed,
                    Some(Duration::ZERO),
                    |_, _| Some(state.output.clone()),
                );
            } else {
                let id = cs.id();
                state.surface_buffers.remove(&id);
                state.cursor_surface = None;
                state.last_cursor_bitmap = None;
            }
        }
        state.display_handle.flush_clients()?;
        state.popups.cleanup();
    }
}

fn spawn_gui_command(socket_name: &OsString, command: &[String]) {
    if let Err(err) = crate::audio::ensure_virtual_audio_devices() {
        tracing::warn!(error = %err, "failed to initialize isolated PipeWire audio session");
    }
    let mut proc = Command::new(&command[0]);
    proc.args(&command[1..]);
    apply_gui_env(&mut proc, socket_name);
    match proc.spawn() {
        Ok(child) => info!(pid = child.id(), cmd = ?command, "launched target application"),
        Err(err) => error!(error = %err, cmd = ?command, "failed to launch target application"),
    }
}

fn derive_exec_display_name(exec: &str) -> String {
    let tokens: Vec<&str> = exec.split_whitespace().collect();
    if let Some(first) = tokens.first() {
        let base = std::path::Path::new(first)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(first);
        if base == "flatpak" {
            if let Some(app_id) = tokens
                .iter()
                .skip(1)
                .find(|t| !t.starts_with('-') && **t != "run")
            {
                return app_id.rsplit('.').next().unwrap_or(app_id).to_string();
            }
        }
        return base.to_string();
    }
    exec.to_string()
}

fn spawn_gui_shell(
    socket_name: &OsString,
    exec: &str,
    launch_id: u64,
    cmd_tx: CalloopSender<CompositorCommand>,
) -> Result<i32> {
    if let Err(err) = crate::audio::ensure_virtual_audio_devices() {
        tracing::warn!(error = %err, "failed to initialize isolated PipeWire audio session");
    }
    let mut proc = Command::new("/bin/sh");
    proc.arg("-c").arg(exec);
    apply_gui_env(&mut proc, socket_name);
    unsafe {
        proc.pre_exec(|| {
            libc::setpgid(0, 0);
            Ok(())
        });
    }
    let mut child = proc.spawn()?;
    let pid = child.id() as i32;
    info!(pid, launch_id, exec, "launched remote application");
    std::thread::Builder::new()
        .name(format!("rw-app-wait-{launch_id}"))
        .spawn(move || {
            let code = child
                .wait()
                .ok()
                .and_then(|s| s.code())
                .unwrap_or(1);
            let _ = cmd_tx.send(CompositorCommand::AppExited { launch_id, code });
        })
        .ok();
    Ok(pid)
}

fn apply_gui_env(proc: &mut Command, socket_name: &OsString) {
    crate::session::apply_session_env(proc, Some(socket_name.as_os_str()));
}

fn blit_rgba_over(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    ox: i32,
    oy: i32,
) {
    for sy in 0..(src_h as i32) {
        let dy = sy + oy;
        if dy < 0 || dy >= dst_h as i32 {
            continue;
        }
        for sx in 0..(src_w as i32) {
            let dx = sx + ox;
            if dx < 0 || dx >= dst_w as i32 {
                continue;
            }
            let s_idx = ((sy as u32 * src_w + sx as u32) * 4) as usize;
            let d_idx = ((dy as u32 * dst_w + dx as u32) * 4) as usize;
            let sa = src[s_idx + 3] as u32;
            if sa == 255 {
                dst[d_idx..d_idx + 4].copy_from_slice(&src[s_idx..s_idx + 4]);
            } else if sa > 0 {
                let inv = 255 - sa;
                dst[d_idx] = ((src[s_idx] as u32 * sa + dst[d_idx] as u32 * inv) / 255) as u8;
                dst[d_idx + 1] =
                    ((src[s_idx + 1] as u32 * sa + dst[d_idx + 1] as u32 * inv) / 255) as u8;
                dst[d_idx + 2] =
                    ((src[s_idx + 2] as u32 * sa + dst[d_idx + 2] as u32 * inv) / 255) as u8;
                dst[d_idx + 3] = (sa + (dst[d_idx + 3] as u32 * inv) / 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_preferred_text_mime_prefers_utf8_over_legacy() {
        let mimes = vec![
            "STRING".to_string(),
            "text/plain".to_string(),
            "text/plain;charset=utf-8".to_string(),
        ];
        assert_eq!(
            pick_preferred_text_mime(&mimes).as_deref(),
            Some("text/plain;charset=utf-8")
        );
        assert_eq!(
            pick_preferred_text_mime(&["image/png".to_string()]),
            None
        );
    }

    #[test]
    fn x11_resize_edge_maps_to_xdg_edges() {
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::Top),
            xdg_toplevel::ResizeEdge::Top as u32
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::BottomRight),
            xdg_toplevel::ResizeEdge::BottomRight as u32
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::TopLeft),
            xdg_toplevel::ResizeEdge::TopLeft as u32
        );
    }

    #[test]
    fn selection_fd_helpers_roundtrip_text() {
        let (read_pipe, write_pipe) = std::io::pipe().unwrap();
        let write_fd: OwnedFd = write_pipe.into();
        let read_fd: OwnedFd = read_pipe.into();
        write_selection_fd_timeout(write_fd, b"hello xwayland");
        assert_eq!(
            read_selection_fd_bounded(read_fd).as_deref(),
            Some("hello xwayland")
        );
    }

    #[test]
    fn x11_app_id_prefers_instance_when_case_variant_of_class() {
        assert_eq!(x11_app_id("XTerm", "xterm"), "xterm");
        assert_eq!(x11_app_id("firefox", "Navigator"), "firefox");
        assert_eq!(x11_app_id("", "xterm"), "xterm");
    }
}
