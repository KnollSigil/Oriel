//! Headless Smithay Wayland compositor for RemoteWayland.
//!
//! Runs on a dedicated `calloop` thread, accepts Wayland client connections on an
//! isolated `WAYLAND_DISPLAY` socket, tracks `xdg_toplevel` and `xdg_popup`
//! windows, extracts committed SHM and GPU (dmabuf) frames, and forwards input/configure events.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::os::unix::io::{AsRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::channel::{self, Channel, Sender as CalloopSender};
use calloop::generic::Generic;
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction};
use remotewayland_proto::{
    LaunchState, FrameRect, WindowId, MAX_CURSOR_DIM, MAX_SELECTION_BYTES,
    SUPPORTED_TEXT_MIMES,
};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Buffer as _;
use smithay::backend::input::{Axis, AxisSource, KeyState};
use smithay::desktop::utils::send_frames_surface_tree;
use smithay::desktop::{PopupKind, PopupManager, Space, Window};
use smithay::input::keyboard::{
    FilterResult, KeyboardHandle, KeyboardTarget, Keycode, KeysymHandle, ModifiersState, XkbConfig,
};
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorImageStatus, CursorImageSurfaceData, MotionEvent,
};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale as OutputScale, Subpixel};
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
use smithay::utils::{IsAlive, Logical, Point, Rectangle, Serial, Transform, SERIAL_COUNTER};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    add_blocker, add_pre_commit_hook, get_parent, send_surface_state, Blocker, BlockerState,
    Cacheable, with_states, with_surface_tree_downward,
    CompositorClientState, CompositorHandler, CompositorState, RectangleKind, RegionAttributes,
    SubsurfaceCachedState, SurfaceAttributes,
    TraversalAction,
};
use smithay::wayland::cursor_shape::CursorShapeManagerState;
use smithay::wayland::drm_syncobj::{
    DrmSyncPoint, DrmSyncobjCachedState, DrmSyncobjHandler, DrmSyncobjState,
};
use smithay::wayland::dmabuf::{
    get_dmabuf, DmabufFeedbackBuilder, DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier,
};
use smithay::wayland::fractional_scale::{
    self, FractionalScaleHandler, FractionalScaleManagerState,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::viewporter::{ViewportCachedState, ViewporterState};
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
    delegate_compositor, delegate_cursor_shape, delegate_data_device, delegate_dmabuf,
    delegate_drm_syncobj,
    delegate_fractional_scale,
    delegate_kde_decoration, delegate_output, delegate_primary_selection, delegate_seat,
    delegate_shm, delegate_viewporter, delegate_xdg_decoration, delegate_xdg_shell,
    delegate_xwayland_shell,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::gpu::{
    BgraFrame, FramePixels, GpuImporter, GpuMode, ImportedFrame, PendingReadback, Readback,
};
use calloop::timer::{TimeoutAction, Timer};
use smithay::backend::egl::fence::EGLFence;
use smithay::backend::renderer::sync::Fence as _;

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
        logical_width: u32,
        logical_height: u32,
        geometry: FrameRect,
        input_bounds: FrameRect,
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
        logical_width: u32,
        logical_height: u32,
        geometry: FrameRect,
        input_bounds: FrameRect,
        pixels: FramePixels,
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
        logical_width: u32,
        logical_height: u32,
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
    /// Emitted at the exact point a [`CompositorCommand::Snapshot`] with this id was taken.
    /// Every event before it is already reflected in the snapshot.
    SnapshotTaken {
        sync_id: u64,
    },
}

/// Commands sent from the QUIC server loop to the headless compositor.
#[derive(Debug)]
pub enum CompositorCommand {
    /// The isolated session (D-Bus, settings portal, xdg-desktop-portal) is up, so the
    /// initial command can launch without racing the portals it reads settings from.
    SessionReady,
    ConfigureWindow {
        window_id: WindowId,
        width: u32,
        height: u32,
        scale_factor: f64,
        maximized: bool,
        fullscreen: bool,
        activated: bool,
        tiled: bool,
    },
    SetDisplayScale {
        scale_factor: f64,
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
    KeyboardEnter {
        window_id: WindowId,
        pressed_keys: Vec<u32>,
    },
    KeyboardLeave {
        window_id: WindowId,
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
    /// Reply with the events that describe the current session state (see
    /// `snapshot_events`) and emit [`CompositorEvent::SnapshotTaken`] on the event stream.
    Snapshot {
        sync_id: u64,
        reply: tokio::sync::oneshot::Sender<Vec<CompositorEvent>>,
    },
    /// Release every held pointer button and key and clear keyboard focus, e.g. when the
    /// client that was driving input goes away mid-press.
    ReleaseInput,
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

#[derive(Clone)]
struct CachedFrame {
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    /// Window rectangle inside the frame (logical); the rest is the app's drop shadow.
    geometry: FrameRect,
    /// Bounding box of the app's input region inside the frame (logical).
    input_bounds: FrameRect,
    pixels: FramePixels,
}

impl PartialEq for CachedFrame {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.logical_width == other.logical_width
            && self.logical_height == other.logical_height
            && self.geometry == other.geometry
            && self.input_bounds == other.input_bounds
            && self.pixels == other.pixels
    }
}
impl Eq for CachedFrame {}

struct TrackedWindow {
    window_id: WindowId,
    window: Window,
    last_title: String,
    last_app_id: String,
    last_width: u32,
    last_height: u32,
    scale_factor: f64,
    server_side_decoration: bool,
    announced: bool,
    has_window_entry: bool,
    latest_bgra: Option<CachedFrame>,
    last_sub_info: Vec<(bool, u32, u32, i32, i32)>,
}

struct TrackedPopup {
    window_id: WindowId,
    parent_id: WindowId,
    surface: PopupSurface,
    geo: Rectangle<i32, Logical>,
    announced: bool,
    latest_bgra: Option<CachedFrame>,
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
    latest_bgra: Option<CachedFrame>,
}

struct TrackedX11Popup {
    window_id: WindowId,
    parent_id: WindowId,
    surface: X11Surface,
    geo: Rectangle<i32, Logical>,
    rel_pos: (i32, i32),
    announced: bool,
    latest_bgra: Option<CachedFrame>,
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
    dmabuf_state: DmabufState,
    /// `zwp_linux_dmabuf_v1` global, present only when a GPU is available.
    dmabuf_global: Option<DmabufGlobal>,
    /// Offscreen GPU renderer reading client dmabufs back into CPU frames.
    gpu: Option<GpuImporter>,
    /// `wp_linux_drm_syncobj_v1` (explicit sync), when the GPU's kernel driver supports it.
    syncobj_state: Option<DrmSyncobjState>,
    /// Dmabuf readbacks for commits not yet applied, by job id.
    readback_jobs: HashMap<u64, ReadbackJob>,
    next_readback_job: u64,
    /// Readback job ids per surface, in commit order.
    surface_readbacks: HashMap<ObjectId, VecDeque<u64>>,
    xdg_shell_state: XdgShellState,
    xdg_decoration_state: XdgDecorationState,
    kde_decoration_state: KdeDecorationState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,
    primary_selection_state: PrimarySelectionState,
    cursor_shape_state: CursorShapeManagerState,
    fractional_scale_manager_state: FractionalScaleManagerState,
    viewporter_state: ViewporterState,
    output_manager_state: OutputManagerState,
    xwayland_shell_state: XWaylandShellState,
    xwm: Option<X11Wm>,
    xwayland_client: Option<Client>,
    output: Output,
    output_size: (i32, i32),
    output_scale: i32,
    display_scale: f64,
    xwayland_scale: f64,
    seat: Seat<Self>,
    space: Space<Window>,
    popups: PopupManager,
    next_window_id: WindowId,
    focused_window_id: WindowId,
    last_focused_x11_window_id: Option<WindowId>,
    /// Pointer buttons a client has pressed and not yet released.
    pressed_buttons: HashSet<u32>,
    next_x11_pos: (i32, i32),
    cursor_surface: Option<WlSurface>,
    last_cursor_bitmap: Option<(u32, u32, u32, u32, i32, i32, Vec<u8>)>,
    clipboard_owner: SelectionOwner,
    clipboard_text: Option<String>,
    pending_clipboard_mime: Option<String>,
    primary_owner: SelectionOwner,
    primary_text: Option<String>,
    pending_primary_mime: Option<String>,
    /// The command to launch once [`Self::initial_x11_settled`] and
    /// [`Self::initial_session_ready`] (see [`AppState::maybe_launch_initial_command`]).
    pending_initial_command: Vec<String>,
    /// XWayland is ready, failed, or not used, so X11 apps find their `DISPLAY`.
    initial_x11_settled: bool,
    /// [`CompositorCommand::SessionReady`] arrived.
    initial_session_ready: bool,
    start_time: Instant,
    /// Map from `WlSurface::id()` (`ObjectId`, unique across clients) -> WindowId (for toplevels & popups).
    surface_to_window: HashMap<ObjectId, WindowId>,
    /// Cached CPU or VRAM-resident frame per `WlSurface::id()`.
    surface_buffers: HashMap<ObjectId, (u32, u32, FramePixels)>,
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
            self.apply_virtual_output_mode();
            debug!(
                width = new_w,
                height = new_h,
                scale = self.output_scale,
                "expanded headless virtual output"
            );
        }
    }

    fn apply_virtual_output_mode(&self) {
        let scale = self.output_scale.max(1);
        let mode = OutputMode {
            size: (self.output_size.0 * scale, self.output_size.1 * scale).into(),
            refresh: 60_000,
        };
        self.output.change_current_state(
            Some(mode),
            None,
            Some(OutputScale::Integer(scale)),
            Some((0, 0).into()),
        );
        self.output.set_preferred(mode);
    }

    fn apply_surface_tree_scale(root: &WlSurface, scale: f64) {
        let scale = sanitize_scale(scale);
        let int_scale = scale.ceil().max(1.0) as i32;
        with_surface_tree_downward(
            root,
            (),
            |surf, states, _| {
                fractional_scale::with_fractional_scale(states, |fs| {
                    fs.set_preferred_scale(scale);
                });
                send_surface_state(surf, states, int_scale, Transform::Normal);
                TraversalAction::DoChildren(())
            },
            |_, _, _| {},
            |_, _, _| true,
        );
    }

    fn window_scale_for_surface(&self, surface: &WlSurface) -> f64 {
        let mut cur = surface.clone();
        loop {
            if let Some(&wid) = self.surface_to_window.get(&cur.id()) {
                if let Some(tw) = self.windows.get(&wid) {
                    return tw.scale_factor;
                }
                if let Some(tp) = self.popup_windows.get(&wid) {
                    if let Some(parent) = self.windows.get(&tp.parent_id) {
                        return parent.scale_factor;
                    }
                }
            }
            match get_parent(&cur) {
                Some(parent) => cur = parent,
                None => break,
            }
        }
        if let Some(tw) = self.windows.get(&self.focused_window_id) {
            return tw.scale_factor;
        }
        self.display_scale
    }

    fn update_output_and_xwayland_scale(&mut self) {
        let max_scale = self
            .windows
            .values()
            .map(|w| w.scale_factor)
            .fold(self.display_scale, f64::max);
        let max_scale = sanitize_scale(max_scale);
        let new_output_scale = max_scale.ceil().max(1.0) as i32;
        if new_output_scale != self.output_scale {
            self.output_scale = new_output_scale;
            self.apply_virtual_output_mode();
            debug!(
                output_scale = new_output_scale,
                "updated virtual output integer scale"
            );
        }

        // Global XWayland HiDPI scaling: keep Smithay XWayland client_scale and Xft.dpi in sync.
        let target_x11_scale = sanitize_scale(self.display_scale);
        if (target_x11_scale - self.xwayland_scale).abs() > 0.005 {
            self.xwayland_scale = target_x11_scale;
            self.apply_xwayland_scale();
        }
    }

    fn apply_xwayland_scale(&self) {
        let scale = sanitize_scale(self.xwayland_scale);
        if let Some(client) = &self.xwayland_client {
            if let Some(data) = client.get_data::<XWaylandClientData>() {
                data.compositor_state.set_client_scale(scale);
            }
        }
        let dpi = (96.0 * scale).round() as u32;
        debug!(
            scale,
            dpi, "applying global XWayland HiDPI scale and Xft.dpi"
        );
        std::thread::spawn(move || {
            apply_xft_dpi(dpi);
        });
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
        let Some((w, h, surf_pixels)) = self.surface_buffers.get(&surf.id()) else {
            return;
        };
        if *w == 0 || *h == 0 || *w > MAX_CURSOR_DIM || *h > MAX_CURSOR_DIM {
            return;
        }
        let bgra = surf_pixels.to_cpu_bgra();
        let (logical_w, logical_h) = surface_logical_size(surf, *w, *h);
        let (hotspot_x, hotspot_y) = with_states(surf, |states| {
            states
                .data_map
                .get::<CursorImageSurfaceData>()
                .and_then(|d| d.lock().ok())
                .map(|attrs| (attrs.hotspot.x, attrs.hotspot.y))
                .unwrap_or((0, 0))
        });
        if self.last_cursor_bitmap.as_ref().is_some_and(
            |(lw, lh, llogw, llogh, lhx, lhy, lbgra)| {
                *lw == *w
                    && *lh == *h
                    && *llogw == logical_w
                    && *llogh == logical_h
                    && *lhx == hotspot_x
                    && *lhy == hotspot_y
                    && lbgra == bgra.as_ref()
            },
        ) {
            return;
        }
        let bgra_cloned = (*bgra).clone();
        self.last_cursor_bitmap = Some((
            *w,
            *h,
            logical_w,
            logical_h,
            hotspot_x,
            hotspot_y,
            bgra_cloned.clone(),
        ));
        let _ = self.event_tx.send(CompositorEvent::CursorBitmap {
            window_id: self.focused_window_id,
            width: *w,
            height: *h,
            logical_width: logical_w,
            logical_height: logical_h,
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

    /// Bounding box of the input regions of every mapped surface in the tree under
    /// `tree_root` (each surface's whole area when it set no input region), relative to the
    /// root and clipped to a `log_w` x `log_h` frame: [`FrameRect::NONE`] when no surface
    /// takes input. Subsurfaces have input regions of their own, so the root's alone would
    /// miss input on a subsurface reaching past it.
    fn tree_input_bounds(&self, tree_root: &WlSurface, log_w: u32, log_h: u32) -> FrameRect {
        let mut bounds: Option<(i32, i32, i32, i32)> = None;
        with_surface_tree_downward(
            tree_root,
            (0, 0),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == tree_root {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, _)) = self.surface_buffers.get(&surf.id()) {
                    let (sub_w, sub_h) = surface_logical_size_from_states(states, *sw, *sh);
                    let mut attrs = states.cached_state.get::<SurfaceAttributes>();
                    let b =
                        input_region_bounds(attrs.current().input_region.as_ref(), sub_w, sub_h);
                    if !b.is_empty() {
                        let (x0, y0) = (nx + b.x, ny + b.y);
                        let (x1, y1) = (x0 + b.width as i32, y0 + b.height as i32);
                        bounds = Some(match bounds {
                            Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
                            None => (x0, y0, x1, y1),
                        });
                    }
                }
                TraversalAction::DoChildren((nx, ny))
            },
            |_, _, _| {},
            |_, _, _| true,
        );
        let Some((x0, y0, x1, y1)) = bounds else {
            return FrameRect::NONE;
        };
        let (x0, y0) = (x0.max(0), y0.max(0));
        let (x1, y1) = (x1.min(log_w as i32), y1.min(log_h as i32));
        if x1 <= x0 || y1 <= y0 {
            return FrameRect::NONE;
        }
        FrameRect {
            x: x0,
            y: y0,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
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
                    let (log_w, log_h) = surface_logical_size_from_states(states, *sw, *sh);
                    let lx = ix - nx;
                    let ly = iy - ny;
                    if lx >= 0 && lx < log_w as i32 && ly >= 0 && ly < log_h as i32 {
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

    fn keyboard_focus_target_for_window(
        &mut self,
        window_id: WindowId,
    ) -> Option<KeyboardFocusTarget> {
        if let Some(tracked) = self.windows.get(&window_id) {
            tracked
                .window
                .toplevel()
                .map(|t| KeyboardFocusTarget::Wayland(t.wl_surface().clone()))
        } else if let Some(popup) = self.popup_windows.get(&window_id) {
            Some(KeyboardFocusTarget::Wayland(
                popup.surface.wl_surface().clone(),
            ))
        } else if let Some(xw) = self.x11_windows.get(&window_id) {
            self.last_focused_x11_window_id = Some(window_id);
            Some(KeyboardFocusTarget::X11(xw.surface.clone()))
        } else {
            self.x11_popups
                .get(&window_id)
                .map(|xp| KeyboardFocusTarget::X11(xp.surface.clone()))
        }
    }

    fn handle_command(&mut self, cmd: CompositorCommand) {
        let serial = SERIAL_COUNTER.next_serial();
        let time_ms = self.start_time.elapsed().as_millis() as u32;
        match cmd {
            CompositorCommand::ConfigureWindow {
                window_id,
                width,
                height,
                scale_factor,
                maximized,
                fullscreen,
                activated,
                tiled,
            } => {
                let scale = sanitize_scale(scale_factor);
                let mut scale_updated = false;
                let mut toplevel_surface: Option<WlSurface> = None;
                if let Some(tracked) = self.windows.get_mut(&window_id) {
                    if (tracked.scale_factor - scale).abs() > 0.005 {
                        tracked.scale_factor = scale;
                        scale_updated = true;
                    }
                    if let Some(toplevel) = tracked.window.toplevel() {
                        toplevel_surface = Some(toplevel.wl_surface().clone());
                        debug!(
                            window_id,
                            width,
                            height,
                            scale,
                            maximized,
                            fullscreen,
                            activated,
                            tiled,
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
                    let has_child_popup =
                        self.x11_popups.values().any(|p| p.parent_id == window_id);
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
                if scale_updated {
                    if let Some(root) = toplevel_surface {
                        Self::apply_surface_tree_scale(&root, scale);
                    }
                    let child_popups: Vec<WlSurface> = self
                        .popup_windows
                        .values()
                        .filter(|p| p.parent_id == window_id)
                        .map(|p| p.surface.wl_surface().clone())
                        .collect();
                    for pop_surf in child_popups {
                        Self::apply_surface_tree_scale(&pop_surf, scale);
                    }
                    if self.focused_window_id == window_id {
                        if let Some(cs) = self.cursor_surface.clone() {
                            Self::apply_surface_tree_scale(&cs, scale);
                        }
                    }
                    self.update_output_and_xwayland_scale();
                }
            }
            CompositorCommand::SessionReady => {
                self.initial_session_ready = true;
                self.maybe_launch_initial_command();
            }
            CompositorCommand::SetDisplayScale { scale_factor } => {
                let scale = sanitize_scale(scale_factor);
                if (self.display_scale - scale).abs() > 0.005 {
                    debug!(
                        old = self.display_scale,
                        new = scale,
                        "updating session display scale"
                    );
                    self.display_scale = scale;
                    self.update_output_and_xwayland_scale();
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
                        tw.window
                            .toplevel()
                            .and_then(|t| self.hit_test_tree(t.wl_surface(), (0, 0), x, y))
                    } else if let Some(tp) = self.popup_windows.get(&window_id) {
                        self.hit_test_tree(tp.surface.wl_surface(), (0, 0), x, y)
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
                            let has_child_popup =
                                self.x11_popups.values().any(|p| p.parent_id == window_id);
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
                if pressed {
                    self.pressed_buttons.insert(button);
                } else {
                    self.pressed_buttons.remove(&button);
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
            CompositorCommand::PointerAxis { window_id, dx, dy } => {
                self.focused_window_id = window_id;
                if let Some(pointer) = self.seat.get_pointer() {
                    let is_wheel = dx.abs() >= 10.0 || dy.abs() >= 10.0;
                    let mut frame = AxisFrame::new(time_ms).source(if is_wheel {
                        AxisSource::Wheel
                    } else {
                        AxisSource::Continuous
                    });
                    if dx != 0.0 {
                        frame = frame.value(Axis::Horizontal, dx);
                        if is_wheel {
                            frame = frame.v120(Axis::Horizontal, (dx * 8.0).round() as i32);
                        }
                    }
                    if dy != 0.0 {
                        frame = frame.value(Axis::Vertical, dy);
                        if is_wheel {
                            frame = frame.v120(Axis::Vertical, (dy * 8.0).round() as i32);
                        }
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
                    if let Some(target) = self.keyboard_focus_target_for_window(window_id) {
                        if keyboard.current_focus().as_ref() != Some(&target) {
                            keyboard.set_focus(self, Some(target), serial);
                        }
                    }
                    let state = if pressed {
                        KeyState::Pressed
                    } else {
                        KeyState::Released
                    };
                    let xkb_keycode = Keycode::from(keycode.saturating_add(8));
                    // Guard against duplicate presses that would double-increment XKB modifier refcounts.
                    if pressed && keyboard.pressed_keys().contains(&xkb_keycode) {
                        return;
                    }
                    keyboard.input::<(), _>(
                        self,
                        xkb_keycode,
                        state,
                        serial,
                        time_ms,
                        |_, _, _| FilterResult::Forward,
                    );
                }
            }
            CompositorCommand::KeyboardEnter {
                window_id,
                pressed_keys,
            } => {
                self.focused_window_id = window_id;
                if let Some(keyboard) = self.seat.get_keyboard() {
                    let target = self.keyboard_focus_target_for_window(window_id);
                    sync_keyboard_enter(self, &keyboard, target, &pressed_keys, serial, time_ms);
                }
            }
            CompositorCommand::KeyboardLeave { window_id } => {
                if let Some(keyboard) = self.seat.get_keyboard() {
                    let target = self.keyboard_focus_target_for_window(window_id);
                    let focus_matches = match (keyboard.current_focus(), target.as_ref()) {
                        (Some(curr), Some(t)) => curr == *t,
                        (_, None) => self.focused_window_id == window_id,
                        (None, Some(_)) => true,
                    };
                    if focus_matches {
                        clear_keyboard_focus_and_keys(self, &keyboard, serial, time_ms);
                    }
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
            CompositorCommand::Snapshot { sync_id, reply } => {
                // The marker goes out on the shared event stream at the exact point the
                // snapshot is taken, so the requesting connection can tell which streamed
                // events are older than the snapshot.
                let events = self.snapshot_events();
                let _ = self
                    .event_tx
                    .send(CompositorEvent::SnapshotTaken { sync_id });
                let _ = reply.send(events);
            }
            CompositorCommand::ReleaseInput => {
                if let Some(pointer) = self.seat.get_pointer() {
                    let held: Vec<u32> = self.pressed_buttons.drain().collect();
                    for button in held {
                        pointer.button(
                            self,
                            &ButtonEvent {
                                button,
                                state: smithay::backend::input::ButtonState::Released,
                                serial,
                                time: time_ms,
                            },
                        );
                    }
                    pointer.frame(self);
                }
                if let Some(keyboard) = self.seat.get_keyboard() {
                    clear_keyboard_focus_and_keys(self, &keyboard, serial, time_ms);
                }
            }
        }
    }

    /// Events that bring a freshly connected (or resyncing) client up to date: every
    /// running app and window entry, and every announced window and popup with its
    /// decoration mode and latest frame.
    fn snapshot_events(&self) -> Vec<CompositorEvent> {
        let mut events = Vec::new();
        for app in self.launched_apps.values() {
            if !matches!(app.state, LaunchState::Exited(_)) {
                events.push(CompositorEvent::AppState {
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
                events.push(CompositorEvent::AppState {
                    launch_id: WINDOW_LAUNCH_ID_BASE | tracked.window_id,
                    name: win_name,
                    state: LaunchState::Running,
                });
            }
        }
        for tracked in self.windows.values() {
            if tracked.announced {
                events.push(CompositorEvent::WindowCreated {
                    window_id: tracked.window_id,
                    title: tracked.last_title.clone(),
                    app_id: tracked.last_app_id.clone(),
                    width: tracked.last_width,
                    height: tracked.last_height,
                });
                events.push(CompositorEvent::SetDecorationMode {
                    window_id: tracked.window_id,
                    server_side: tracked.server_side_decoration,
                });
                if let Some(frame) = &tracked.latest_bgra {
                    events.push(CompositorEvent::FrameCommitted {
                        window_id: tracked.window_id,
                        width: frame.width,
                        height: frame.height,
                        logical_width: frame.logical_width,
                        logical_height: frame.logical_height,
                        geometry: frame.geometry,
                        input_bounds: frame.input_bounds,
                        pixels: frame.pixels.clone(),
                    });
                }
            }
        }
        for popup in self.popup_windows.values() {
            if popup.announced {
                if let Some(frame) = &popup.latest_bgra {
                    events.push(CompositorEvent::PopupCreated {
                        window_id: popup.window_id,
                        parent_id: popup.parent_id,
                        x: popup.geo.loc.x,
                        y: popup.geo.loc.y,
                        width: frame.logical_width,
                        height: frame.logical_height,
                    });
                    events.push(CompositorEvent::PopupFrame {
                        window_id: popup.window_id,
                        width: frame.width,
                        height: frame.height,
                        logical_width: frame.logical_width,
                        logical_height: frame.logical_height,
                        geometry: frame.geometry,
                        input_bounds: frame.input_bounds,
                        bgra: (*frame.pixels.to_cpu_bgra()).clone(),
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
                    events.push(CompositorEvent::AppState {
                        launch_id: WINDOW_LAUNCH_ID_BASE | xwin.window_id,
                        name: win_name,
                        state: LaunchState::Running,
                    });
                }
                events.push(CompositorEvent::WindowCreated {
                    window_id: xwin.window_id,
                    title: xwin.last_title.clone(),
                    app_id: xwin.last_app_id.clone(),
                    width: xwin.last_width,
                    height: xwin.last_height,
                });
                events.push(CompositorEvent::SetDecorationMode {
                    window_id: xwin.window_id,
                    server_side: xwin.server_side_decoration,
                });
                if let Some(frame) = &xwin.latest_bgra {
                    events.push(CompositorEvent::FrameCommitted {
                        window_id: xwin.window_id,
                        width: frame.width,
                        height: frame.height,
                        logical_width: frame.logical_width,
                        logical_height: frame.logical_height,
                        geometry: frame.geometry,
                        input_bounds: frame.input_bounds,
                        pixels: frame.pixels.clone(),
                    });
                }
            }
        }
        for xpop in self.x11_popups.values() {
            if xpop.announced {
                if let Some(frame) = &xpop.latest_bgra {
                    events.push(CompositorEvent::PopupCreated {
                        window_id: xpop.window_id,
                        parent_id: xpop.parent_id,
                        x: xpop.rel_pos.0,
                        y: xpop.rel_pos.1,
                        width: frame.logical_width,
                        height: frame.logical_height,
                    });
                    events.push(CompositorEvent::PopupFrame {
                        window_id: xpop.window_id,
                        width: frame.width,
                        height: frame.height,
                        logical_width: frame.logical_width,
                        logical_height: frame.logical_height,
                        geometry: frame.geometry,
                        input_bounds: frame.input_bounds,
                        bgra: (*frame.pixels.to_cpu_bgra()).clone(),
                    });
                }
            }
        }
        events
    }

    /// Launch the initial command once XWayland has settled and the session is ready.
    fn maybe_launch_initial_command(&mut self) {
        if self.initial_x11_settled
            && self.initial_session_ready
            && !self.pending_initial_command.is_empty()
        {
            let cmd = std::mem::take(&mut self.pending_initial_command);
            spawn_gui_command(&self.socket_name, &cmd);
        }
    }

    fn recomposite_window(&mut self, window_id: WindowId, root_surface: &WlSurface) {
        let root_id = root_surface.id();
        let Some((surf_w, surf_h, _)) = self.surface_buffers.get(&root_id) else {
            return;
        };
        let surf_w = *surf_w;
        let surf_h = *surf_h;
        let (root_log_w, root_log_h) = surface_logical_size(root_surface, surf_w, surf_h);
        let root_sx = surf_w as f64 / root_log_w as f64;
        let root_sy = surf_h as f64 / root_log_h as f64;

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
        // Stream the whole surface, drop shadow included, and tell the client where the
        // window sits inside it so its own compositor sees the same window geometry.
        let geometry = clamp_window_geometry(xdg_geo, root_log_w, root_log_h);
        let input_bounds = self.tree_input_bounds(root_surface, root_log_w, root_log_h);
        let (win_log_w, win_log_h) = (geometry.width, geometry.height);

        let mut sub_info = Vec::new();
        let mut has_subsurfaces = false;
        // Whether some subsurface renders at another scale than the root surface (Firefox:
        // an integer-scale root surface for the shadow, fractional-scale content on top).
        let mut mixed_scale = false;
        with_surface_tree_downward(
            root_surface,
            (0, 0),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == root_surface {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, _)) = self.surface_buffers.get(&surf.id()) {
                    let dst_x = (nx as f64 * root_sx).round() as i32;
                    let dst_y = (ny as f64 * root_sy).round() as i32;
                    if surf != root_surface {
                        has_subsurfaces = true;
                        let (log_w, log_h) = surface_logical_size_from_states(states, *sw, *sh);
                        let sub_sx = *sw as f64 / log_w as f64;
                        let sub_sy = *sh as f64 / log_h as f64;
                        if (sub_sx - root_sx).abs() > 0.01 || (sub_sy - root_sy).abs() > 0.01 {
                            mixed_scale = true;
                        }
                    }
                    sub_info.push((surf == root_surface, *sw, *sh, dst_x, dst_y));
                }
                TraversalAction::DoChildren((nx, ny))
            },
            |_, _, _| {},
            |_, _, _| true,
        );

        // Surfaces at mixed scales are composited at the client's display scale, so each
        // is resampled at most once: the client shows the frame at that scale. Otherwise
        // the root's own scale needs no resampling at all.
        let (comp_sx, comp_sy) = if mixed_scale {
            let scale = self
                .windows
                .get(&window_id)
                .map(|w| w.scale_factor)
                .unwrap_or(self.display_scale);
            (scale, scale)
        } else {
            (root_sx, root_sy)
        };
        let (win_phys_w, win_phys_h) = if mixed_scale {
            (
                ((root_log_w as f64 * comp_sx).round() as u32).max(1),
                ((root_log_h as f64 * comp_sy).round() as u32).max(1),
            )
        } else {
            (surf_w, surf_h)
        };

        let composite: FramePixels = if !has_subsurfaces {
            // Zero-copy fast path for single-surface windows (both VRAM Dmabuf and SHM).
            self.surface_buffers.get(&root_id).unwrap().2.clone()
        } else {
            let mut buf = vec![0u8; (win_phys_w * win_phys_h * 4) as usize];
            with_surface_tree_downward(
                root_surface,
                (0, 0),
                |surf, states, (ox, oy)| {
                    let (nx, ny) = if surf == root_surface {
                        (*ox, *oy)
                    } else {
                        let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                        let loc = sub_state.current().location;
                        (ox + loc.x, oy + loc.y)
                    };
                    if let Some((sw, sh, spixels)) = self.surface_buffers.get(&surf.id()) {
                        let srgba = spixels.to_cpu_bgra();
                        let (sub_log_w, sub_log_h) =
                            surface_logical_size_from_states(states, *sw, *sh);
                        let dst_x = (nx as f64 * comp_sx).round() as i32;
                        let dst_y = (ny as f64 * comp_sy).round() as i32;
                        let (dst_w, dst_h) = if surf == root_surface {
                            (win_phys_w, win_phys_h)
                        } else {
                            (
                                ((sub_log_w as f64 * comp_sx).round() as u32).max(1),
                                ((sub_log_h as f64 * comp_sy).round() as u32).max(1),
                            )
                        };
                        if dst_w == *sw && dst_h == *sh {
                            blit_rgba_over(
                                &mut buf, win_phys_w, win_phys_h, &srgba, *sw, *sh, dst_x, dst_y,
                            );
                        } else {
                            blit_rgba_over_scaled(
                                &mut buf, win_phys_w, win_phys_h, &srgba, *sw, *sh, dst_x, dst_y,
                                dst_w, dst_h,
                            );
                        }
                    }
                    TraversalAction::DoChildren((nx, ny))
                },
                |_, _, _| {},
                |_, _, _| true,
            );
            FramePixels::Cpu(Arc::new(buf))
        };

        let Some(tracked) = self.windows.get_mut(&window_id) else {
            return;
        };
        if sub_info != tracked.last_sub_info {
            debug!(
                window_id,
                win_phys_w,
                win_phys_h,
                win_log_w,
                win_log_h,
                ?geometry,
                ?sub_info,
                "window surface tree changed"
            );
            tracked.last_sub_info = sub_info;
        }

        let new_frame = CachedFrame {
            width: win_phys_w,
            height: win_phys_h,
            logical_width: root_log_w,
            logical_height: root_log_h,
            geometry,
            input_bounds,
            pixels: composite,
        };

        if tracked.latest_bgra.as_ref() == Some(&new_frame) {
            return;
        }

        if !tracked.announced && win_log_w > 0 && win_log_h > 0 {
            tracked.announced = true;
            tracked.last_width = win_log_w;
            tracked.last_height = win_log_h;
            tracked.last_title = title.clone();
            tracked.last_app_id = app_id.clone();
            let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                window_id,
                title: title.clone(),
                app_id: app_id.clone(),
                width: win_log_w,
                height: win_log_h,
            });
            let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                window_id,
                server_side: tracked.server_side_decoration,
            });
            let client_pid = root_surface
                .client()
                .and_then(|c| c.get_credentials(&self.display_handle).ok())
                .map(|creds| creds.pid);
            let claimed = claim_launch_for_window(&mut self.launched_apps, client_pid, window_id);
            if let Some((launch_id, name)) = claimed {
                let _ = self.event_tx.send(CompositorEvent::AppState {
                    launch_id,
                    name,
                    state: LaunchState::Running,
                });
            } else {
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
        } else if win_log_w != tracked.last_width || win_log_h != tracked.last_height {
            tracked.last_width = win_log_w;
            tracked.last_height = win_log_h;
            let _ = self.event_tx.send(CompositorEvent::WindowConfigured {
                window_id,
                width: win_log_w,
                height: win_log_h,
                scale_factor: tracked.scale_factor,
            });
        }

        tracked.latest_bgra = Some(new_frame.clone());
        let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
            window_id,
            width: new_frame.width,
            height: new_frame.height,
            logical_width: new_frame.logical_width,
            logical_height: new_frame.logical_height,
            geometry: new_frame.geometry,
            input_bounds: new_frame.input_bounds,
            pixels: new_frame.pixels,
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
        let (root_log_w, root_log_h) = surface_logical_size(&root_surface, surf_w, surf_h);
        let root_sx = surf_w as f64 / root_log_w as f64;
        let root_sy = surf_h as f64 / root_log_h as f64;

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
        // Like toplevels, stream the whole popup surface (popovers draw their own shadow)
        // along with where the popup's window rectangle sits inside it.
        let geometry = clamp_window_geometry(xdg_geo, root_log_w, root_log_h);
        let input_bounds = self.tree_input_bounds(&root_surface, root_log_w, root_log_h);
        let (pop_log_w, pop_log_h) = (geometry.width, geometry.height);
        let (pop_phys_w, pop_phys_h) = (surf_w, surf_h);

        let mut bgra = vec![0u8; (pop_phys_w * pop_phys_h * 4) as usize];
        with_surface_tree_downward(
            &root_surface,
            (0, 0),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == &root_surface {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.current().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, spixels)) = self.surface_buffers.get(&surf.id()) {
                    let sbgra = spixels.to_cpu_bgra();
                    let (sub_log_w, sub_log_h) = surface_logical_size_from_states(states, *sw, *sh);
                    let dst_x = (nx as f64 * root_sx).round() as i32;
                    let dst_y = (ny as f64 * root_sy).round() as i32;
                    let (dst_w, dst_h) = if surf == &root_surface {
                        (*sw, *sh)
                    } else {
                        (
                            ((sub_log_w as f64 * root_sx).round() as u32).max(1),
                            ((sub_log_h as f64 * root_sy).round() as u32).max(1),
                        )
                    };
                    if dst_w == *sw && dst_h == *sh {
                        blit_rgba_over(
                            &mut bgra, pop_phys_w, pop_phys_h, &sbgra, *sw, *sh, dst_x, dst_y,
                        );
                    } else {
                        blit_rgba_over_scaled(
                            &mut bgra, pop_phys_w, pop_phys_h, &sbgra, *sw, *sh, dst_x, dst_y,
                            dst_w, dst_h,
                        );
                    }
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

        let bgra_arc = Arc::new(bgra);
        let new_frame = CachedFrame {
            width: pop_phys_w,
            height: pop_phys_h,
            logical_width: root_log_w,
            logical_height: root_log_h,
            geometry,
            input_bounds,
            pixels: FramePixels::Cpu(Arc::clone(&bgra_arc)),
        };

        if popup.latest_bgra.as_ref() == Some(&new_frame) {
            return;
        }

        if !popup.announced {
            popup.announced = true;
            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                window_id,
                parent_id: popup.parent_id,
                x: popup.geo.loc.x,
                y: popup.geo.loc.y,
                width: pop_log_w,
                height: pop_log_h,
            });
        }

        popup.latest_bgra = Some(new_frame.clone());
        let _ = self.event_tx.send(CompositorEvent::PopupFrame {
            window_id,
            width: new_frame.width,
            height: new_frame.height,
            logical_width: new_frame.logical_width,
            logical_height: new_frame.logical_height,
            geometry: new_frame.geometry,
            input_bounds: new_frame.input_bounds,
            bgra: (*bgra_arc).clone(),
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

    fn new_surface(&mut self, surface: &WlSurface) {
        add_pre_commit_hook::<Self, _>(surface, |state, _dh, surface| {
            state.queue_buffer_readback(surface);
        });
    }

    fn commit(&mut self, surface: &WlSurface) {
        self.popups.commit(surface);

        let buffer_opt = with_states(surface, |states| {
            let mut attrs = states
                .cached_state
                .get::<smithay::wayland::compositor::SurfaceAttributes>();
            let buffer = match attrs.current().buffer.take() {
                Some(smithay::wayland::compositor::BufferAssignment::NewBuffer(buf)) => Some(buf),
                _ => None,
            };
            let release_point = states
                .cached_state
                .get::<DrmSyncobjCachedState>()
                .current()
                .release_point
                .take();
            // Always consumed, so a job never outlives the commit it was merged into.
            let readback_job = states
                .cached_state
                .get::<ReadbackJobState>()
                .current()
                .job
                .take();
            (buffer, release_point, readback_job)
        });
        let (buffer_opt, release_point, readback_job) = buffer_opt;
        let readback_frame = readback_job.and_then(|id| self.take_readback(surface, id));

        if let Some(buffer) = buffer_opt {
            let frame: Option<ImportedFrame> = if get_dmabuf(&buffer).is_ok() {
                // Imported/blitted before this commit was applied (see `queue_buffer_readback`).
                readback_frame
            } else {
                read_shm_buffer(&buffer).map(|frame| {
                    let crop = with_states(surface, |states| {
                        let (scale, transform) = {
                            let mut attrs = states
                                .cached_state
                                .get::<smithay::wayland::compositor::SurfaceAttributes>(
                            );
                            let current = attrs.current();
                            (current.buffer_scale, current.buffer_transform)
                        };
                        let src = states
                            .cached_state
                            .get::<ViewportCachedState>()
                            .current()
                            .src;
                        let size = (frame.0 as i32, frame.1 as i32).into();
                        buffer_crop(src, size, scale, transform.into())
                    });
                    let (w, h, bgra) = match crop {
                        Some(rect) => crop_frame(&frame, rect),
                        None => frame,
                    };
                    (w, h, FramePixels::Cpu(Arc::new(bgra)))
                })
            };
            if let Some(frame) = frame {
                self.store_surface_buffer(surface, frame);
            }
            release_buffer(buffer, release_point);
        }

        self.surface_contents_updated(surface);
    }

    fn destroyed(&mut self, surface: &WlSurface) {
        self.surface_buffers.remove(&surface.id());
        if let Some(jobs) = self.surface_readbacks.remove(&surface.id()) {
            for id in jobs {
                self.readback_jobs.remove(&id);
            }
        }
    }
}

/// The readback job for a commit's new dmabuf, double-buffered with the rest of the surface
/// state. smithay's commit and merge (for synchronized subsurfaces) then decide which job
/// belongs to the buffer that is finally applied.
#[derive(Debug, Default)]
struct ReadbackJobState {
    job: Option<u64>,
}

impl Cacheable for ReadbackJobState {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        Self {
            job: self.job.take(),
        }
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        if self.job.is_some() {
            into.job = self.job;
        }
    }
}

/// A client dmabuf imported/blitted before the commit that attached it is applied.
struct ReadbackJob {
    /// The frame, once the GPU blit or readback succeeded.
    frame: Option<ImportedFrame>,
}

/// State carried through the stages of one readback.
struct ReadbackContext {
    id: u64,
    dmabuf: Dmabuf,
    crop: Option<Rectangle<f64, smithay::utils::Buffer>>,
    client: Client,
    released: Arc<AtomicBool>,
}

/// Holds a surface commit until its buffer has been read back.
struct ReadbackBlocker(Arc<AtomicBool>);

impl Blocker for ReadbackBlocker {
    fn state(&self) -> BlockerState {
        if self.0.load(Ordering::Acquire) {
            BlockerState::Released
        } else {
            BlockerState::Pending
        }
    }
}

impl AppState {
    /// On `wl_surface.commit` with a new dmabuf: hold the commit, wait for the client's GPU work
    /// (explicit-sync acquire point, else the dmabuf's implicit fences), copy the buffer on the
    /// GPU, and release the commit once the copy lands. Each commit is then applied together with
    /// its own buffer contents, and the event loop never waits on the GPU.
    fn queue_buffer_readback(&mut self, surface: &WlSurface) {
        if self.gpu.is_none() {
            return;
        }
        let Some(client) = surface.client() else {
            return;
        };
        let pending = with_states(surface, |states| {
            let (buffer, scale, transform) = {
                let mut attrs = states
                    .cached_state
                    .get::<smithay::wayland::compositor::SurfaceAttributes>();
                let pending = attrs.pending();
                let Some(smithay::wayland::compositor::BufferAssignment::NewBuffer(buffer)) =
                    pending.buffer.as_ref()
                else {
                    return None;
                };
                (
                    buffer.clone(),
                    pending.buffer_scale,
                    pending.buffer_transform,
                )
            };
            let dmabuf = get_dmabuf(&buffer).ok()?.clone();
            let src = states
                .cached_state
                .get::<ViewportCachedState>()
                .pending()
                .src;
            let crop = buffer_crop(src, dmabuf.size(), scale, transform.into());
            let acquire_point = states
                .cached_state
                .get::<DrmSyncobjCachedState>()
                .pending()
                .acquire_point
                .clone();
            Some((dmabuf, crop, acquire_point))
        });
        let Some((dmabuf, crop, acquire_point)) = pending else {
            return;
        };

        self.next_readback_job += 1;
        let id = self.next_readback_job;
        with_states(surface, |states| {
            states.cached_state.get::<ReadbackJobState>().pending().job = Some(id);
        });
        self.readback_jobs.insert(id, ReadbackJob { frame: None });
        self.surface_readbacks
            .entry(surface.id())
            .or_default()
            .push_back(id);
        let released = Arc::new(AtomicBool::new(false));
        add_blocker(surface, ReadbackBlocker(released.clone()));
        let ctx = ReadbackContext {
            id,
            dmabuf: dmabuf.clone(),
            crop,
            client,
            released,
        };

        // Stage 1: wait until the client's rendering into the buffer has finished. Shared so it
        // can still run here if the wait can't be set up.
        let ctx = Rc::new(Cell::new(Some(ctx)));
        let waiting_ctx = Rc::clone(&ctx);
        let waiting = if let Some(acquire_point) = acquire_point {
            // GL doesn't wait on syncobj timelines, so reading before the acquire point signals
            // would copy a half-rendered frame. If it can't be watched, skip this frame.
            let fd = acquire_point
                .eventfd()
                .or_else(|_| acquire_point.export_sync_file().map(Arc::new));
            let inserted = match fd {
                Ok(fd) => self
                    .loop_handle
                    .insert_source(
                        Generic::new(fd, Interest::READ, Mode::OneShot),
                        move |_, _, state| {
                            if let Some(ctx) = waiting_ctx.take() {
                                state.run_readback(ctx);
                            }
                            Ok(PostAction::Remove)
                        },
                    )
                    .map_err(|err| anyhow::anyhow!("{err}")),
                Err(err) => Err(err.into()),
            };
            if let Err(err) = inserted {
                warn!(
                    error = %err,
                    "cannot wait for explicit-sync acquire point; skipping frame"
                );
                if let Some(ctx) = ctx.take() {
                    self.complete_readback(ctx, None);
                }
                return;
            }
            true
        } else {
            match dmabuf.generate_blocker(Interest::READ) {
                Ok((_blocker, source)) => self
                    .loop_handle
                    .insert_source(source, move |_, _, state| {
                        if let Some(ctx) = waiting_ctx.take() {
                            state.run_readback(ctx);
                        }
                        Ok(())
                    })
                    .is_ok(),
                // The client's rendering already finished.
                Err(_) => false,
            }
        };
        if !waiting {
            if let Some(ctx) = ctx.take() {
                self.run_readback(ctx);
            }
        }
    }

    /// Stage 2: queue the GPU copy, then (stage 3) map it once its fence signals.
    fn run_readback(&mut self, ctx: ReadbackContext) {
        let Some(gpu) = self.gpu.as_mut() else {
            self.complete_readback(ctx, None);
            return;
        };
        let (readback, fence) = match gpu.start_readback(&ctx.dmabuf, ctx.crop) {
            Ok(Readback::Ready(frame)) => {
                self.complete_readback(ctx, Some(frame));
                return;
            }
            Ok(Readback::Pending { readback, fence }) => (readback, fence),
            Ok(Readback::Polled { readback, fence }) => {
                self.poll_readback(ctx, readback, fence);
                return;
            }
            Err(err) => {
                warn!(error = %format!("{err:#}"), "failed to read client dmabuf");
                self.complete_readback(ctx, None);
                return;
            }
        };

        let job = Rc::new(Cell::new(Some((ctx, readback))));
        let fenced_job = Rc::clone(&job);
        let inserted = self.loop_handle.insert_source(
            Generic::new(fence, Interest::READ, Mode::OneShot),
            move |_, _, state| {
                if let Some((ctx, readback)) = fenced_job.take() {
                    state.finish_readback(ctx, readback);
                }
                Ok(PostAction::Remove)
            },
        );
        if let Err(err) = inserted {
            warn!(error = %err, "failed to watch readback fence; mapping synchronously");
            if let Some((ctx, readback)) = job.take() {
                self.finish_readback(ctx, readback);
            }
        }
    }

    /// Without an exportable fence, check it on a short timer instead of blocking the loop.
    fn poll_readback(&mut self, ctx: ReadbackContext, readback: PendingReadback, fence: EGLFence) {
        const POLL_INTERVAL: Duration = Duration::from_millis(1);
        let job = Rc::new(Cell::new(Some((ctx, readback))));
        let polled_job = Rc::clone(&job);
        let inserted = self.loop_handle.insert_source(
            Timer::from_duration(POLL_INTERVAL),
            move |_, _, state| {
                if !fence.is_signaled() {
                    return TimeoutAction::ToDuration(POLL_INTERVAL);
                }
                if let Some((ctx, readback)) = polled_job.take() {
                    state.finish_readback(ctx, readback);
                }
                TimeoutAction::Drop
            },
        );
        if let Err(err) = inserted {
            warn!(error = %err, "failed to poll readback fence; mapping synchronously");
            if let Some((ctx, readback)) = job.take() {
                self.finish_readback(ctx, readback);
            }
        }
    }

    fn finish_readback(&mut self, ctx: ReadbackContext, readback: PendingReadback) {
        let frame = match self.gpu.as_mut().map(|gpu| gpu.finish_readback(readback)) {
            Some(Ok(frame)) => Some(frame),
            Some(Err(err)) => {
                warn!(error = %format!("{err:#}"), "failed to read client dmabuf");
                None
            }
            None => None,
        };
        self.complete_readback(ctx, frame);
    }

    /// Record the frame and let the held commit apply.
    fn complete_readback(&mut self, ctx: ReadbackContext, frame: Option<ImportedFrame>) {
        if let Some(job) = self.readback_jobs.get_mut(&ctx.id) {
            job.frame = frame;
        }
        ctx.released.store(true, Ordering::Release);
        let dh = self.display_handle.clone();
        self.client_compositor_state(&ctx.client)
            .blocker_cleared(self, &dh);
    }

    /// Take the readback for the commit being applied. Earlier jobs of the surface belong to
    /// commits that were applied or merged away, so they are dropped too.
    fn take_readback(&mut self, surface: &WlSurface, id: u64) -> Option<ImportedFrame> {
        if let Some(queue) = self.surface_readbacks.get_mut(&surface.id()) {
            while queue.front().is_some_and(|&queued| queued < id) {
                if let Some(stale) = queue.pop_front() {
                    self.readback_jobs.remove(&stale);
                }
            }
            if queue.front() == Some(&id) {
                queue.pop_front();
            }
        }
        self.readback_jobs.remove(&id)?.frame
    }

    fn store_surface_buffer(&mut self, surface: &WlSurface, (w, h, pixels): ImportedFrame) {
        if w > 0 && h > 0 {
            self.surface_buffers.insert(surface.id(), (w, h, pixels));
        }
    }

    /// Re-emit the cursor or recomposite the window owning `surface` after its contents changed.
    fn surface_contents_updated(&mut self, surface: &WlSurface) {
        if self.cursor_surface.as_ref().is_some_and(|cs| cs == surface) {
            self.emit_cursor_surface_bitmap(surface);
        }

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

/// Hand a buffer back to its client, signalling its explicit-sync release point if any.
fn release_buffer(buffer: WlBuffer, release_point: Option<DrmSyncPoint>) {
    if let Some(point) = release_point {
        if let Err(err) = point.signal() {
            warn!(error = %err, "failed to signal explicit-sync release point");
        }
    }
    buffer.release();
}

/// The region of a buffer selected by a `wp_viewport` source rectangle, in buffer pixels, or
/// `None` when the whole buffer is used. Clients such as Chromium allocate padded buffers and
/// crop them this way.
fn buffer_crop(
    src: Option<Rectangle<f64, Logical>>,
    buffer_size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    buffer_scale: i32,
    transform: Transform,
) -> Option<Rectangle<f64, smithay::utils::Buffer>> {
    let src = src?;
    let scale = buffer_scale.max(1);
    let surface_size = buffer_size.to_logical(scale, transform).to_f64();
    let full = Rectangle::from_size(buffer_size.to_f64());
    let rect = src
        .to_buffer(f64::from(scale), transform, &surface_size)
        .intersection(full)?;
    (rect != full).then_some(rect)
}

/// Crop a CPU frame to `rect` (buffer pixels, rounded to whole pixels).
fn crop_frame((w, h, bgra): &BgraFrame, rect: Rectangle<f64, smithay::utils::Buffer>) -> BgraFrame {
    let x = (rect.loc.x.round().max(0.0) as u32).min(*w);
    let y = (rect.loc.y.round().max(0.0) as u32).min(*h);
    let cw = (rect.size.w.round().max(1.0) as u32).min(w - x);
    let ch = (rect.size.h.round().max(1.0) as u32).min(h - y);
    let mut out = Vec::with_capacity((cw * ch * 4) as usize);
    for row in y..y + ch {
        let start = ((row * w + x) * 4) as usize;
        out.extend_from_slice(&bgra[start..start + (cw * 4) as usize]);
    }
    (cw, ch, out)
}

/// Copy a `wl_shm` buffer into tightly packed BGRA rows.
fn read_shm_buffer(buffer: &WlBuffer) -> Option<(u32, u32, Vec<u8>)> {
    with_buffer_contents(buffer, |ptr, len, spec| {
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
                    for px in bgra[dst_start..dst_start + row_bytes]
                        .as_chunks_mut::<4>()
                        .0
                    {
                        px[3] = 0xFF;
                    }
                }
            }
        }
        (width, height, bgra)
    })
    .ok()
}

impl BufferHandler for AppState {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}

impl DrmSyncobjHandler for AppState {
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
        self.syncobj_state.as_mut()
    }
}

impl DmabufHandler for AppState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        if self.gpu.as_mut().is_some_and(|gpu| gpu.can_import(&dmabuf)) {
            let _ = notifier.successful::<AppState>();
        } else {
            notifier.failed();
        }
    }
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

        let initial_scale = sanitize_scale(self.display_scale);
        let wl_surface = surface.wl_surface().clone();
        Self::apply_surface_tree_scale(&wl_surface, initial_scale);
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
                scale_factor: initial_scale,
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
            // Frames carry the whole surface, so surface-local coordinates match the client's.
            let _ = self.event_tx.send(CompositorEvent::ShowWindowMenu {
                window_id,
                x: location.x,
                y: location.y,
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

        let parent_scale = self
            .windows
            .get(&parent_id)
            .map(|w| w.scale_factor)
            .unwrap_or(self.display_scale);
        Self::apply_surface_tree_scale(surface.wl_surface(), parent_scale);

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
                let scale = self
                    .windows
                    .get(&self.focused_window_id)
                    .map(|w| w.scale_factor)
                    .unwrap_or(self.display_scale);
                Self::apply_surface_tree_scale(&surf, scale);
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
    SUPPORTED_TEXT_MIMES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
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
        let Some((w, h, pixels_ref)) = self.surface_buffers.get(&wl_surface.id()) else {
            return;
        };
        let (w, h) = (*w, *h);
        if w == 0 || h == 0 {
            return;
        }
        let x11_scale = sanitize_scale(self.xwayland_scale);
        let log_w = ((w as f64 / x11_scale).round() as u32).max(1);
        let log_h = ((h as f64 / x11_scale).round() as u32).max(1);
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
        let size_changed = log_w != xwin.last_width || log_h != xwin.last_height;
        let new_frame = CachedFrame {
            width: w,
            height: h,
            logical_width: log_w,
            logical_height: log_h,
            geometry: FrameRect::full(log_w, log_h),
            input_bounds: FrameRect::full(log_w, log_h),
            pixels: pixels_ref.clone(),
        };
        let frame_unchanged = xwin.announced
            && !title_or_app_changed
            && xwin.latest_bgra.as_ref() == Some(&new_frame);
        if frame_unchanged {
            return;
        }

        xwin.last_title = title.clone();
        xwin.last_app_id = app_id.clone();
        xwin.last_width = log_w;
        xwin.last_height = log_h;
        xwin.latest_bgra = Some(new_frame.clone());

        if !xwin.announced {
            xwin.announced = true;
            let ssd = xwin.server_side_decoration;
            let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                window_id,
                title: title.clone(),
                app_id: app_id.clone(),
                width: log_w,
                height: log_h,
            });
            let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                window_id,
                server_side: ssd,
            });

            // Link to the launched app that owns this window's process, or emit a standalone
            // window AppState. (The Wayland client here is Xwayland, so use the X11 pid.)
            let x11_pid = xwin.surface.pid().and_then(|p| i32::try_from(p).ok());
            let matched_launch =
                claim_launch_for_window(&mut self.launched_apps, x11_pid, window_id);
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
                    width: log_w,
                    height: log_h,
                });
            }
            if size_changed {
                let _ = self.event_tx.send(CompositorEvent::WindowConfigured {
                    window_id,
                    width: log_w,
                    height: log_h,
                    scale_factor: x11_scale,
                });
            }
        }

        let _ = self.event_tx.send(CompositorEvent::FrameCommitted {
            window_id,
            width: new_frame.width,
            height: new_frame.height,
            logical_width: new_frame.logical_width,
            logical_height: new_frame.logical_height,
            geometry: new_frame.geometry,
            input_bounds: new_frame.input_bounds,
            pixels: new_frame.pixels,
        });
    }

    fn recomposite_x11_popup(&mut self, window_id: WindowId, wl_surface: &WlSurface) {
        let Some((w, h, pixels_ref)) = self.surface_buffers.get(&wl_surface.id()) else {
            return;
        };
        let (w, h) = (*w, *h);
        if w == 0 || h == 0 {
            return;
        }
        let x11_scale = sanitize_scale(self.xwayland_scale);
        let log_w = ((w as f64 / x11_scale).round() as u32).max(1);
        let log_h = ((h as f64 / x11_scale).round() as u32).max(1);
        let Some(xpop) = self.x11_popups.get_mut(&window_id) else {
            return;
        };
        let new_frame = CachedFrame {
            width: w,
            height: h,
            logical_width: log_w,
            logical_height: log_h,
            geometry: FrameRect::full(log_w, log_h),
            input_bounds: FrameRect::full(log_w, log_h),
            pixels: pixels_ref.clone(),
        };
        if xpop.announced && xpop.latest_bgra.as_ref() == Some(&new_frame) {
            return;
        }
        let popup_bgra = (*new_frame.pixels.to_cpu_bgra()).clone();
        xpop.latest_bgra = Some(new_frame.clone());
        if !xpop.announced {
            xpop.announced = true;
            debug!(
                window_id,
                parent_id = xpop.parent_id,
                x = xpop.rel_pos.0,
                y = xpop.rel_pos.1,
                w,
                h,
                log_w,
                log_h,
                "emitting PopupCreated for X11 popup"
            );
            let _ = self.event_tx.send(CompositorEvent::PopupCreated {
                window_id,
                parent_id: xpop.parent_id,
                x: xpop.rel_pos.0,
                y: xpop.rel_pos.1,
                width: log_w,
                height: log_h,
            });
        }
        let _ = self.event_tx.send(CompositorEvent::PopupFrame {
            window_id,
            width: new_frame.width,
            height: new_frame.height,
            logical_width: new_frame.logical_width,
            logical_height: new_frame.logical_height,
            geometry: new_frame.geometry,
            input_bounds: new_frame.input_bounds,
            bgra: popup_bgra,
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
            let _ = self.event_tx.send(CompositorEvent::StartMove { window_id });
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

    fn new_selection(&mut self, _xwm: XwmId, selection: SelectionTarget, mime_types: Vec<String>) {
        match selection {
            SelectionTarget::Clipboard => {
                self.clipboard_owner = SelectionOwner::X11Client;
                self.clipboard_text = None;
                set_data_device_selection(&self.display_handle, &self.seat, mime_types.clone(), ());
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

impl FractionalScaleHandler for AppState {
    fn new_fractional_scale(&mut self, surface: WlSurface) {
        let scale = self.window_scale_for_surface(&surface);
        Self::apply_surface_tree_scale(&surface, scale);
    }
}

delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_dmabuf!(AppState);
delegate_drm_syncobj!(AppState);
delegate_xdg_shell!(AppState);
delegate_xdg_decoration!(AppState);
delegate_kde_decoration!(AppState);
delegate_seat!(AppState);
delegate_data_device!(AppState);
delegate_primary_selection!(AppState);
delegate_cursor_shape!(AppState);
delegate_fractional_scale!(AppState);
delegate_viewporter!(AppState);
delegate_output!(AppState);
delegate_xwayland_shell!(AppState);

/// Start the headless Smithay Wayland compositor on a background OS thread
/// and optionally launch `command` inside the isolated session.
pub fn start_compositor(
    command: Vec<String>,
    enable_xwayland: bool,
    gpu_mode: GpuMode,
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
                gpu_mode,
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

/// Open the GPU selected by `mode` and advertise its dmabuf formats to clients.
fn init_gpu(
    dh: &DisplayHandle,
    dmabuf_state: &mut DmabufState,
    mode: &GpuMode,
) -> Result<Option<(GpuImporter, DmabufGlobal)>> {
    let Some(gpu) = GpuImporter::new(mode)? else {
        return Ok(None);
    };
    let feedback = DmabufFeedbackBuilder::new(gpu.node().dev_id(), gpu.formats())
        .build()
        .context("failed to build dmabuf feedback")?;
    let global = dmabuf_state.create_global_with_default_feedback::<AppState>(dh, &feedback);
    Ok(Some((gpu, global)))
}

fn run_compositor_loop(
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
    cmd_rx: Channel<CompositorCommand>,
    cmd_tx: CalloopSender<CompositorCommand>,
    ready_tx: std::sync::mpsc::SyncSender<Result<OsString>>,
    command: Vec<String>,
    enable_xwayland: bool,
    gpu_mode: GpuMode,
) -> Result<()> {
    let mut event_loop: EventLoop<AppState> =
        EventLoop::try_new().context("failed to create calloop EventLoop")?;
    let loop_handle = event_loop.handle();
    let display: Display<AppState> = Display::new().context("failed to create Wayland Display")?;
    let dh = display.handle();

    let compositor_state = CompositorState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    // Create the dmabuf global before Xwayland starts so glamor finds the render node.
    let mut dmabuf_state = DmabufState::new();
    let (gpu, dmabuf_global) = match init_gpu(&dh, &mut dmabuf_state, &gpu_mode) {
        Ok(Some((gpu, global))) => (Some(gpu), Some(global)),
        Ok(None) => {
            info!("GPU acceleration disabled; apps render in software");
            (None, None)
        }
        // The user asked for this GPU, so don't silently fall back to software.
        Err(err) if matches!(gpu_mode, GpuMode::Node(_)) => {
            return Err(err.context("the GPU given with --render-node is unusable"));
        }
        Err(err) => {
            warn!(
                error = %format!("{err:#}"),
                "GPU acceleration unavailable; apps render in software"
            );
            (None, None)
        }
    };
    let syncobj_state = gpu
        .as_ref()
        .and_then(GpuImporter::syncobj_device)
        .map(|device| DrmSyncobjState::new::<AppState>(&dh, device));
    let xdg_shell_state = XdgShellState::new::<AppState>(&dh);
    let xdg_decoration_state = XdgDecorationState::new::<AppState>(&dh);
    let kde_decoration_state =
        KdeDecorationState::new::<AppState>(&dh, KdeDefaultDecorationMode::Server);
    let mut seat_state = SeatState::new();
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let primary_selection_state = PrimarySelectionState::new::<AppState>(&dh);
    let cursor_shape_state = CursorShapeManagerState::new::<AppState>(&dh);
    let fractional_scale_manager_state = FractionalScaleManagerState::new::<AppState>(&dh);
    let viewporter_state = ViewporterState::new::<AppState>(&dh);
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
                                    state.xwayland_scale = sanitize_scale(state.display_scale);
                                    state.apply_xwayland_scale();
                                    info!(
                                        display = %disp,
                                        xauthority = ?xauth_path,
                                        scale = state.xwayland_scale,
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
                            state.initial_x11_settled = true;
                            state.maybe_launch_initial_command();
                        }
                        XWaylandEvent::Error => {
                            warn!("XWayland exited unexpectedly; clearing X11 state and continuing Wayland-only");
                            state.cleanup_xwayland_state();
                            state.initial_x11_settled = true;
                            state.maybe_launch_initial_command();
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

    let pending_initial_command = command;

    let mut state = AppState {
        display_handle: dh.clone(),
        loop_handle: loop_handle.clone(),
        socket_name: socket_name.clone(),
        cmd_tx,
        compositor_state,
        shm_state,
        dmabuf_state,
        dmabuf_global,
        gpu,
        syncobj_state,
        readback_jobs: HashMap::new(),
        next_readback_job: 0,
        surface_readbacks: HashMap::new(),
        xdg_shell_state,
        xdg_decoration_state,
        kde_decoration_state,
        seat_state,
        data_device_state,
        primary_selection_state,
        cursor_shape_state,
        fractional_scale_manager_state,
        viewporter_state,
        output_manager_state,
        xwayland_shell_state,
        xwm: None,
        xwayland_client,
        output,
        output_size: (1920, 1080),
        output_scale: 1,
        display_scale: 1.0,
        xwayland_scale: 1.0,
        seat,
        space,
        popups: PopupManager::default(),
        next_window_id: 1,
        focused_window_id: 1,
        last_focused_x11_window_id: None,
        pressed_buttons: HashSet::new(),
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
        initial_x11_settled: !xwayland_spawned,
        initial_session_ready: false,
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

    const FRAME_INTERVAL: Duration = Duration::from_millis(16);
    // Fallback if XWayland or the session never reports ready (portal startup waits up to
    // a few seconds for its D-Bus names).
    let initial_cmd_deadline = Instant::now() + Duration::from_secs(5);
    let mut last_frame_tick = Instant::now() - FRAME_INTERVAL;

    loop {
        let timeout = FRAME_INTERVAL.saturating_sub(last_frame_tick.elapsed());
        event_loop.dispatch(timeout, &mut state)?;
        if !state.pending_initial_command.is_empty() && Instant::now() >= initial_cmd_deadline {
            warn!(
                xwayland_ready = state.initial_x11_settled,
                session_ready = state.initial_session_ready,
                "timed out waiting for XWayland/session startup; launching initial command anyway"
            );
            let cmd = std::mem::take(&mut state.pending_initial_command);
            spawn_gui_command(&state.socket_name, &cmd);
        }
        if last_frame_tick.elapsed() >= FRAME_INTERVAL {
            last_frame_tick = Instant::now();
            let elapsed = state.start_time.elapsed();
            for tracked in state.windows.values() {
                tracked
                    .window
                    .send_frame(&state.output, elapsed, Some(Duration::ZERO), |_, _| {
                        Some(state.output.clone())
                    });
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
        }
        state.display_handle.flush_clients()?;
        state.popups.cleanup();
    }
}

fn spawn_gui_command(socket_name: &OsString, command: &[String]) {
    if let Err(err) = crate::audio::ensure_virtual_audio_devices() {
        tracing::warn!(error = %err, "failed to initialize isolated PipeWire audio session");
    }
    let command =
        crate::session::with_flatpak_run_options(command, crate::session::flatpak_run_options);
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
    let exec =
        crate::session::shell_with_flatpak_run_options(exec, crate::session::flatpak_run_options);
    let exec = exec.as_str();
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
            let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(1);
            let _ = cmd_tx.send(CompositorCommand::AppExited { launch_id, code });
        })
        .ok();
    Ok(pid)
}

/// Link a newly announced window to the launch it belongs to and mark that launch
/// `Running`. Returns the launch's id and name, or `None` if the window should get its own
/// entry.
///
/// Launches run in their own process group (`spawn_gui_shell`), so a window belongs to the
/// still-launching app whose group contains, or whose group leader is an ancestor of, the
/// window's client process. Without a client pid this only guesses when exactly one launch
/// is pending, so a launch that never opens a window can't swallow another app's window.
fn claim_launch_for_window(
    apps: &mut BTreeMap<u64, LaunchedApp>,
    client_pid: Option<i32>,
    window_id: WindowId,
) -> Option<(u64, String)> {
    let is_pending =
        |app: &LaunchedApp| app.state == LaunchState::Launching && app.linked_window_id.is_none();
    let launch_id = match client_pid.filter(|&pid| pid > 0) {
        Some(pid) => {
            let owner = launch_owning_process(apps, pid)?;
            apps.get(&owner).filter(|app| is_pending(app))?.launch_id
        }
        None => {
            let mut pending = apps.values().filter(|app| is_pending(app));
            match (pending.next(), pending.next()) {
                (Some(app), None) => app.launch_id,
                _ => return None,
            }
        }
    };
    let app = apps.get_mut(&launch_id)?;
    app.state = LaunchState::Running;
    app.linked_window_id = Some(window_id);
    Some((app.launch_id, app.name.clone()))
}

/// The launch whose process group contains `pid`, or whose group leader is an ancestor of
/// `pid` (for children that moved to a new process group).
fn launch_owning_process(apps: &BTreeMap<u64, LaunchedApp>, pid: i32) -> Option<u64> {
    let by_group = |pgid: i32| {
        apps.values()
            .find(|app| app.pgid > 0 && app.pgid == pgid)
            .map(|app| app.launch_id)
    };
    // SAFETY: getpgid only reads process state.
    let pgid = unsafe { libc::getpgid(pid) };
    if let Some(id) = (pgid > 0).then(|| by_group(pgid)).flatten() {
        return Some(id);
    }
    let mut cur = pid;
    for _ in 0..64 {
        if let Some(id) = by_group(cur) {
            return Some(id);
        }
        cur = parent_pid(cur).filter(|&ppid| ppid > 1)?;
    }
    None
}

fn parent_pid(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    parse_stat_ppid(&stat)
}

/// Parse the parent pid from `/proc/<pid>/stat` (`pid (comm) state ppid ...`; `comm` may
/// contain spaces and parentheses).
fn parse_stat_ppid(stat: &str) -> Option<i32> {
    let after_comm = &stat[stat.rfind(')')? + 1..];
    after_comm.split_whitespace().nth(1)?.parse().ok()
}

fn apply_gui_env(proc: &mut Command, socket_name: &OsString) {
    crate::session::apply_session_env(proc, Some(socket_name.as_os_str()));
}

fn sync_keyboard_enter<D: SeatHandler + 'static>(
    state: &mut D,
    keyboard: &KeyboardHandle<D>,
    target: Option<D::KeyboardFocus>,
    evdev_pressed_keys: &[u32],
    serial: Serial,
    time_ms: u32,
) {
    let desired_keys: HashSet<Keycode> = evdev_pressed_keys
        .iter()
        .copied()
        .map(|kc| Keycode::from(kc.saturating_add(8)))
        .collect();
    let current_keys = keyboard.pressed_keys();

    if current_keys != desired_keys {
        // Temporarily unset focus while syncing held keys so Smithay updates its
        // internal pressed_keys, forwarded_pressed_keys, and XKB modifier state
        // without emitting spurious wl_keyboard::key press/release events to any client.
        if keyboard.current_focus().is_some() {
            keyboard.set_focus(state, None, serial);
        }
        for kc in current_keys.difference(&desired_keys) {
            keyboard.input::<(), _>(
                state,
                *kc,
                KeyState::Released,
                serial,
                time_ms,
                |_, _, _| FilterResult::Forward,
            );
        }
        for kc in desired_keys.difference(&current_keys) {
            keyboard.input::<(), _>(state, *kc, KeyState::Pressed, serial, time_ms, |_, _, _| {
                FilterResult::Forward
            });
        }
    }

    if let Some(target) = target {
        if keyboard.current_focus().as_ref() != Some(&target) {
            keyboard.set_focus(state, Some(target), serial);
        }
    }
}

fn clear_keyboard_focus_and_keys<D: SeatHandler + 'static>(
    state: &mut D,
    keyboard: &KeyboardHandle<D>,
    serial: Serial,
    time_ms: u32,
) {
    keyboard.set_focus(state, None, serial);
    let held: Vec<Keycode> = keyboard.pressed_keys().into_iter().collect();
    for kc in held {
        keyboard.input::<(), _>(state, kc, KeyState::Released, serial, time_ms, |_, _, _| {
            FilterResult::Forward
        });
    }
}

/// The `xdg_surface` window geometry clamped to a `log_w` x `log_h` surface, or the whole
/// surface when the app set none.
fn clamp_window_geometry(
    geo: Option<Rectangle<i32, Logical>>,
    log_w: u32,
    log_h: u32,
) -> FrameRect {
    // Clip each axis to the surface: a part outside it is cut off, not shifted inside.
    let clip = |start: i32, len: i32, max: u32| -> Option<(i32, u32)> {
        let max = max.min(i32::MAX as u32) as i32;
        let lo = start.clamp(0, max);
        let hi = start.saturating_add(len).clamp(0, max);
        (hi > lo).then(|| (lo, (hi - lo) as u32))
    };
    match geo {
        Some(g) if g.size.w > 0 && g.size.h > 0 => {
            match (
                clip(g.loc.x, g.size.w, log_w),
                clip(g.loc.y, g.size.h, log_h),
            ) {
                (Some((x, width)), Some((y, height))) => FrameRect {
                    x,
                    y,
                    width,
                    height,
                },
                // Entirely outside the surface: treat the whole surface as the window.
                _ => FrameRect::full(log_w, log_h),
            }
        }
        _ => FrameRect::full(log_w, log_h),
    }
}

/// Bounding box of the added rectangles of a surface's input region, clamped to a
/// `log_w` x `log_h` surface: the whole surface when the app set no input region, and a
/// zero-size rectangle (no input, as for a click-through surface) when it set an empty one.
fn input_region_bounds(region: Option<&RegionAttributes>, log_w: u32, log_h: u32) -> FrameRect {
    let Some(region) = region else {
        return FrameRect::full(log_w, log_h);
    };
    let mut bounds: Option<(i32, i32, i32, i32)> = None;
    for (kind, r) in &region.rects {
        if !matches!(kind, RectangleKind::Add) || r.size.w <= 0 || r.size.h <= 0 {
            continue;
        }
        let x0 = r.loc.x.max(0);
        let y0 = r.loc.y.max(0);
        let x1 = r.loc.x.saturating_add(r.size.w).min(log_w as i32);
        let y1 = r.loc.y.saturating_add(r.size.h).min(log_h as i32);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        bounds = Some(match bounds {
            Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
            None => (x0, y0, x1, y1),
        });
    }
    match bounds {
        Some((x0, y0, x1, y1)) => FrameRect {
            x: x0,
            y: y0,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        },
        None => FrameRect::NONE,
    }
}

fn sanitize_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale >= 1.0 {
        scale.min(4.0)
    } else {
        1.0
    }
}

fn surface_logical_size_from_states(
    states: &smithay::wayland::compositor::SurfaceData,
    buf_w: u32,
    buf_h: u32,
) -> (u32, u32) {
    let vp_size = {
        let mut vp = states.cached_state.get::<ViewportCachedState>();
        let cur = vp.current();
        if let Some(dst) = cur.dst {
            if dst.w > 0 && dst.h > 0 {
                Some((dst.w as u32, dst.h as u32))
            } else {
                None
            }
        } else if let Some(src) = cur.src {
            if src.size.w > 0.0 && src.size.h > 0.0 {
                Some((
                    src.size.w.round().max(1.0) as u32,
                    src.size.h.round().max(1.0) as u32,
                ))
            } else {
                None
            }
        } else {
            None
        }
    };
    if let Some(size) = vp_size {
        return size;
    }
    let buf_scale = {
        let mut attrs = states
            .cached_state
            .get::<smithay::wayland::compositor::SurfaceAttributes>();
        attrs.current().buffer_scale.max(1) as u32
    };
    ((buf_w / buf_scale).max(1), (buf_h / buf_scale).max(1))
}

fn surface_logical_size(surface: &WlSurface, buf_w: u32, buf_h: u32) -> (u32, u32) {
    with_states(surface, |states| {
        surface_logical_size_from_states(states, buf_w, buf_h)
    })
}

fn apply_xft_dpi(dpi: u32) {
    let mut cmd = Command::new("xrdb");
    if !crate::session::apply_x11_env(&mut cmd) {
        return;
    }
    cmd.arg("-merge")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Ok(mut child) = cmd.spawn() {
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = writeln!(stdin, "Xft.dpi: {dpi}");
        }
        let _ = child.wait();
    }
}

fn blit_rgba_over_scaled(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    ox: i32,
    oy: i32,
    target_w: u32,
    target_h: u32,
) {
    if src_w == 0 || src_h == 0 || target_w == 0 || target_h == 0 {
        return;
    }
    let x_scale = src_w as f64 / target_w as f64;
    let y_scale = src_h as f64 / target_h as f64;
    for ty in 0..(target_h as i32) {
        let dy = ty + oy;
        if dy < 0 || dy >= dst_h as i32 {
            continue;
        }
        let sy = ((ty as f64 + 0.5) * y_scale)
            .floor()
            .clamp(0.0, (src_h - 1) as f64) as u32;
        for tx in 0..(target_w as i32) {
            let dx = tx + ox;
            if dx < 0 || dx >= dst_w as i32 {
                continue;
            }
            let sx = ((tx as f64 + 0.5) * x_scale)
                .floor()
                .clamp(0.0, (src_w - 1) as f64) as u32;
            let s_idx = ((sy * src_w + sx) * 4) as usize;
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

    /// A pgid no process can have (above Linux's maximum pid).
    const UNRELATED_PGID: i32 = 1 << 30;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new((x, y).into(), (w, h).into())
    }

    #[test]
    fn window_geometry_excludes_shadow_and_stays_inside_surface() {
        // libadwaita-style: 1000x700 window inside a 1060x760 surface with shadow.
        assert_eq!(
            clamp_window_geometry(Some(rect(30, 30, 1000, 700)), 1060, 760),
            FrameRect {
                x: 30,
                y: 30,
                width: 1000,
                height: 700
            }
        );
        // Geometry hanging past the surface is clipped to it.
        assert_eq!(
            clamp_window_geometry(Some(rect(-5, 10, 2000, 2000)), 800, 600),
            FrameRect {
                x: 0,
                y: 10,
                width: 800,
                height: 590
            }
        );
        // A negative offset is clipped, not shifted: only (0,0)-(50,100) is visible.
        assert_eq!(
            clamp_window_geometry(Some(rect(-50, 0, 100, 100)), 800, 600),
            FrameRect {
                x: 0,
                y: 0,
                width: 50,
                height: 100
            }
        );
        // No (or empty) geometry means the whole surface is the window.
        assert_eq!(
            clamp_window_geometry(None, 640, 480),
            FrameRect::full(640, 480)
        );
        assert_eq!(
            clamp_window_geometry(Some(rect(0, 0, 0, 10)), 640, 480),
            FrameRect::full(640, 480)
        );
    }

    #[test]
    fn input_bounds_cover_added_rects_only() {
        assert_eq!(input_region_bounds(None, 100, 80), FrameRect::full(100, 80));
        let region = RegionAttributes {
            rects: vec![
                (RectangleKind::Add, rect(18, 18, 1024, 724)),
                (RectangleKind::Subtract, rect(0, 0, 40, 40)),
                (RectangleKind::Add, rect(1040, 18, 100, 10)),
            ],
        };
        assert_eq!(
            input_region_bounds(Some(&region), 1060, 760),
            FrameRect {
                x: 18,
                y: 18,
                width: 1042,
                height: 724
            }
        );
        // An empty region (or one that only subtracts) takes no input at all.
        let empty = RegionAttributes { rects: vec![] };
        assert_eq!(input_region_bounds(Some(&empty), 100, 80), FrameRect::NONE);
        let subtract_only = RegionAttributes {
            rects: vec![(RectangleKind::Subtract, rect(0, 0, 10, 10))],
        };
        assert_eq!(
            input_region_bounds(Some(&subtract_only), 100, 80),
            FrameRect::NONE
        );
    }

    fn launch(launch_id: u64, pgid: i32) -> (u64, LaunchedApp) {
        let app = LaunchedApp {
            launch_id,
            name: format!("app {launch_id}"),
            pgid,
            state: LaunchState::Launching,
            linked_window_id: None,
        };
        (launch_id, app)
    }

    /// A `sleep` in its own process group, like `spawn_gui_shell` starts apps.
    fn spawn_group_leader() -> std::process::Child {
        use std::os::unix::process::CommandExt;
        Command::new("sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .expect("spawn sleep")
    }

    #[test]
    fn window_links_to_the_launch_owning_its_process_not_the_oldest() {
        let mut child = spawn_group_leader();
        let pid = child.id() as i32;
        // Launch 1 never opens a window; launch 2's process opens one.
        let mut apps = BTreeMap::from([launch(1, UNRELATED_PGID), launch(2, pid)]);

        let claimed = claim_launch_for_window(&mut apps, Some(pid), 7);
        assert_eq!(claimed, Some((2, "app 2".to_string())));
        assert_eq!(apps[&2].state, LaunchState::Running);
        assert_eq!(apps[&2].linked_window_id, Some(7));
        assert_eq!(apps[&1].state, LaunchState::Launching);

        // A second window from the same app gets its own entry.
        assert_eq!(claim_launch_for_window(&mut apps, Some(pid), 8), None);
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn window_from_an_unrelated_process_claims_nothing() {
        let mut apps = BTreeMap::from([launch(1, UNRELATED_PGID)]);
        let own_pid = std::process::id() as i32;
        assert_eq!(claim_launch_for_window(&mut apps, Some(own_pid), 7), None);
        assert_eq!(apps[&1].state, LaunchState::Launching);
    }

    #[test]
    fn window_from_a_descendant_process_links_to_its_launch() {
        // This test process descends from its parent; treat the parent as the launch.
        let parent = parent_pid(std::process::id() as i32).expect("parent pid");
        let mut apps = BTreeMap::from([launch(1, UNRELATED_PGID), launch(2, parent)]);
        let own_pid = std::process::id() as i32;
        assert_eq!(launch_owning_process(&apps, own_pid), Some(2));
        assert!(claim_launch_for_window(&mut apps, Some(own_pid), 7).is_some());
    }

    #[test]
    fn window_without_a_pid_only_links_when_one_launch_is_pending() {
        let mut apps = BTreeMap::from([launch(1, UNRELATED_PGID), launch(2, UNRELATED_PGID)]);
        assert_eq!(claim_launch_for_window(&mut apps, None, 7), None);
        apps.remove(&1);
        assert_eq!(
            claim_launch_for_window(&mut apps, None, 7),
            Some((2, "app 2".to_string()))
        );
    }

    #[test]
    fn parses_ppid_from_proc_stat_with_odd_comm() {
        let stat = "1234 (we (ird) name) S 42 1234 1234 0";
        assert_eq!(parse_stat_ppid(stat), Some(42));
        assert_eq!(parse_stat_ppid("garbage"), None);
    }

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
        assert_eq!(pick_preferred_text_mime(&["image/png".to_string()]), None);
    }

    #[test]
    fn buffer_crop_maps_viewport_source_to_buffer_pixels() {
        let size = (2226, 2459).into();
        // Chromium-style padded buffer at buffer scale 1 with a source crop.
        let src = Rectangle::new((0.0, 0.0).into(), (1826.0, 2036.0).into());
        let crop = buffer_crop(Some(src), size, 1, Transform::Normal).unwrap();
        assert_eq!(
            crop,
            Rectangle::new((0.0, 0.0).into(), (1826.0, 2036.0).into())
        );
        // Source coordinates are surface-local, so buffer scale 2 doubles them.
        let src = Rectangle::new((10.0, 20.0).into(), (500.0, 600.0).into());
        let crop = buffer_crop(Some(src), size, 2, Transform::Normal).unwrap();
        assert_eq!(
            crop,
            Rectangle::new((20.0, 40.0).into(), (1000.0, 1200.0).into())
        );
        // No source, or a source covering the whole buffer, needs no crop.
        assert_eq!(buffer_crop(None, size, 1, Transform::Normal), None);
        let whole = Rectangle::new((0.0, 0.0).into(), (2226.0, 2459.0).into());
        assert_eq!(buffer_crop(Some(whole), size, 1, Transform::Normal), None);
    }

    #[test]
    fn crop_frame_copies_selected_rows() {
        // 3x2 frame whose pixel bytes encode their index.
        let bgra: Vec<u8> = (0..6u8).flat_map(|i| [i; 4]).collect();
        let rect = Rectangle::new((1.0, 0.0).into(), (2.0, 2.0).into());
        let (w, h, out) = crop_frame(&(3, 2, bgra), rect);
        assert_eq!((w, h), (2, 2));
        let firsts: Vec<u8> = out.chunks(4).map(|p| p[0]).collect();
        assert_eq!(firsts, vec![1, 2, 4, 5]);
    }

    #[test]
    fn sanitize_scale_clamps_invalid_and_out_of_range_values() {
        assert_eq!(sanitize_scale(1.5), 1.5);
        assert_eq!(sanitize_scale(0.5), 1.0);
        assert_eq!(sanitize_scale(-2.0), 1.0);
        assert_eq!(sanitize_scale(f64::NAN), 1.0);
        assert_eq!(sanitize_scale(f64::INFINITY), 1.0);
        assert_eq!(sanitize_scale(6.0), 4.0);
    }

    #[test]
    fn blit_rgba_over_scaled_upscales_1x_subsurface_into_2x_canvas() {
        // 2x1 opaque source: left pixel blue, right pixel red.
        let src = [
            255u8, 0, 0, 255, // (0,0)
            0, 0, 255, 255, // (1,0)
        ];
        let mut dst = vec![0u8; 4 * 2 * 4];
        blit_rgba_over_scaled(&mut dst, 4, 2, &src, 2, 1, 0, 0, 4, 2);
        // Columns 0,1 should be blue; columns 2,3 should be red on both rows.
        for y in 0..2 {
            for x in 0..2 {
                let idx = (y * 4 + x) * 4;
                assert_eq!(&dst[idx..idx + 4], &[255, 0, 0, 255]);
            }
            for x in 2..4 {
                let idx = (y * 4 + x) * 4;
                assert_eq!(&dst[idx..idx + 4], &[0, 0, 255, 255]);
            }
        }
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockKeyboardTarget(u32);

    impl IsAlive for MockKeyboardTarget {
        fn alive(&self) -> bool {
            true
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum RecordedKeyboardEvent {
        Enter(u32, Vec<u32>),
        Leave(u32),
        Key(u32, u32, KeyState),
        Modifiers(u32, bool, bool, bool, bool),
    }

    struct MockSeatState {
        seat_state: SeatState<Self>,
        events: Vec<RecordedKeyboardEvent>,
    }

    impl SeatHandler for MockSeatState {
        type KeyboardFocus = MockKeyboardTarget;
        type PointerFocus = WlSurface;
        type TouchFocus = WlSurface;

        fn seat_state(&mut self) -> &mut SeatState<Self> {
            &mut self.seat_state
        }
    }

    impl KeyboardTarget<MockSeatState> for MockKeyboardTarget {
        fn enter(
            &self,
            _seat: &Seat<MockSeatState>,
            data: &mut MockSeatState,
            keys: Vec<KeysymHandle<'_>>,
            _serial: Serial,
        ) {
            let mut raw: Vec<u32> = keys.into_iter().map(|k| k.raw_code().raw()).collect();
            raw.sort_unstable();
            data.events.push(RecordedKeyboardEvent::Enter(self.0, raw));
        }

        fn leave(&self, _seat: &Seat<MockSeatState>, data: &mut MockSeatState, _serial: Serial) {
            data.events.push(RecordedKeyboardEvent::Leave(self.0));
        }

        fn key(
            &self,
            _seat: &Seat<MockSeatState>,
            data: &mut MockSeatState,
            key: KeysymHandle<'_>,
            state: KeyState,
            _serial: Serial,
            _time: u32,
        ) {
            data.events.push(RecordedKeyboardEvent::Key(
                self.0,
                key.raw_code().raw(),
                state,
            ));
        }

        fn modifiers(
            &self,
            _seat: &Seat<MockSeatState>,
            data: &mut MockSeatState,
            modifiers: ModifiersState,
            _serial: Serial,
        ) {
            data.events.push(RecordedKeyboardEvent::Modifiers(
                self.0,
                modifiers.ctrl,
                modifiers.shift,
                modifiers.alt,
                modifiers.logo,
            ));
        }
    }

    #[test]
    fn keyboard_enter_syncs_held_modifiers_without_spurious_key_presses() {
        // evdev keycodes: KEY_LEFTCTRL = 29, KEY_LEFTSHIFT = 42, KEY_LEFTALT = 56
        const KEY_LEFTCTRL: u32 = 29;
        const KEY_LEFTSHIFT: u32 = 42;
        const KEY_LEFTALT: u32 = 56;

        let mut seat_state = SeatState::new();
        let mut seat = seat_state.new_seat("seat0");
        let keyboard = seat.add_keyboard(XkbConfig::default(), 200, 25).unwrap();
        let mut state = MockSeatState {
            seat_state,
            events: Vec::new(),
        };

        // 1. Enter window 1 with Ctrl + Shift already held (including duplicate Ctrl to test dedup).
        sync_keyboard_enter(
            &mut state,
            &keyboard,
            Some(MockKeyboardTarget(1)),
            &[KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_LEFTCTRL],
            SERIAL_COUNTER.next_serial(),
            10,
        );

        let mods = keyboard.modifier_state();
        assert!(mods.ctrl && mods.shift && !mods.alt && !mods.logo);
        assert_eq!(
            state.events,
            vec![
                RecordedKeyboardEvent::Enter(1, vec![KEY_LEFTCTRL + 8, KEY_LEFTSHIFT + 8]),
                RecordedKeyboardEvent::Modifiers(1, true, true, false, false),
            ]
        );
        state.events.clear();

        // 2. Single release of Ctrl immediately clears ctrl (proving no XKB refcount drift).
        keyboard.input::<(), _>(
            &mut state,
            Keycode::from(KEY_LEFTCTRL + 8),
            KeyState::Released,
            SERIAL_COUNTER.next_serial(),
            20,
            |_, _, _| FilterResult::Forward,
        );
        let mods = keyboard.modifier_state();
        assert!(!mods.ctrl && mods.shift);
        state.events.clear();

        // 3. Leave window 1 while Shift is still held: no spurious key release forwarded,
        // and keyboard state is reset cleanly.
        clear_keyboard_focus_and_keys(&mut state, &keyboard, SERIAL_COUNTER.next_serial(), 30);
        assert_eq!(state.events, vec![RecordedKeyboardEvent::Leave(1)]);
        assert!(keyboard.pressed_keys().is_empty());
        assert!(!keyboard.modifier_state().shift);
        state.events.clear();

        // 4. Re-enter window 2 with Alt held (e.g., Alt+Tab): receives Enter + Modifiers(alt=true),
        // with zero spurious Key(Alt, Pressed) events.
        sync_keyboard_enter(
            &mut state,
            &keyboard,
            Some(MockKeyboardTarget(2)),
            &[KEY_LEFTALT],
            SERIAL_COUNTER.next_serial(),
            40,
        );
        assert!(keyboard.modifier_state().alt);
        assert_eq!(
            state.events,
            vec![
                RecordedKeyboardEvent::Enter(2, vec![KEY_LEFTALT + 8]),
                RecordedKeyboardEvent::Modifiers(2, false, false, true, false),
            ]
        );
    }
}
