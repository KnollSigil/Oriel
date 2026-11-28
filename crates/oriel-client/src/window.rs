//! Native Wayland rootless multi-window & unbounded `xdg_popup` presenter
//! built on `smithay-client-toolkit` (`sctk`).

use std::collections::{HashMap, HashSet};
use std::os::unix::io::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use oriel_proto::selection::{ReceivedClipboard, Serve};
use oriel_proto::xfer::{
    parse_file_uri_list, FileOffer, OfferSlot, PORTAL_FILE_TRANSFER_MIME, URI_LIST_MIME,
};
use oriel_proto::{ControlMessage, FrameRect, WindowId, MAX_CURSOR_DIM, SUPPORTED_TEXT_MIMES};
use smithay_client_toolkit::reexports::calloop::channel::{self, Channel, Sender as CalloopSender};
use smithay_client_toolkit::reexports::calloop::EventLoop;
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    data_device_manager::{
        data_device::{DataDevice, DataDeviceHandler},
        data_offer::{DataOfferHandler, DragOffer},
        data_source::{CopyPasteSource, DataSourceHandler, DragSource},
        DataDeviceManagerState, WritePipe,
    },
    delegate_dispatch2, delegate_registry,
    dispatch2::Dispatch2,
    output::{OutputHandler, OutputInfo, OutputState},
    primary_selection::{
        device::{PrimarySelectionDevice, PrimarySelectionDeviceHandler},
        selection::{PrimarySelectionSource, PrimarySelectionSourceHandler},
        PrimarySelectionManagerState,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers},
        pointer::{
            CursorIcon, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec, ThemedPointer,
            BTN_LEFT,
        },
        Capability, SeatHandler, SeatState,
    },
    shell::{
        xdg::{
            popup::{Popup, PopupConfigure, PopupHandler},
            window::{DecorationMode, Window, WindowConfigure, WindowDecorations, WindowHandler},
            XdgPositioner, XdgShell, XdgSurface,
        },
        WaylandSurface,
    },
    shm::{
        slot::{Buffer, SlotPool},
        Shm, ShmHandler,
    },
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{
        wl_data_device::WlDataDevice, wl_data_device_manager::DndAction,
        wl_data_offer::WlDataOffer, wl_data_source::WlDataSource, wl_keyboard, wl_output,
        wl_pointer, wl_seat, wl_shm, wl_surface,
    },
    Connection, Proxy, QueueHandle, WEnum,
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1::{self, WpFractionalScaleManagerV1},
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_protocols::wp::primary_selection::zv1::client::{
    zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
    zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1,
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::{self, WpViewport},
    wp_viewporter::{self, WpViewporter},
};
use wayland_protocols::xdg::dialog::v1::client::{
    xdg_dialog_v1::{self, XdgDialogV1},
    xdg_wm_dialog_v1::{self, XdgWmDialogV1},
};
use wayland_protocols::xdg::shell::client::{
    xdg_positioner::{Anchor, ConstraintAdjustment, Gravity},
    xdg_surface,
    xdg_toplevel::ResizeEdge,
};
use wayland_protocols::xdg::toplevel_icon::v1::client::{
    xdg_toplevel_icon_manager_v1::{self, XdgToplevelIconManagerV1},
    xdg_toplevel_icon_v1::XdgToplevelIconV1,
};

use crate::hosts::RenderScaleCap;
use crate::icon::{decode_icon_argb8888, match_app_icon, AppIconEntry};

/// Events sent from the QUIC/GStreamer receiver tasks to the Wayland UI thread.
#[derive(Debug)]
pub enum ClientUiEvent {
    WindowCreated {
        window_id: WindowId,
        title: String,
        app_id: String,
        width: u32,
        height: u32,
    },
    WindowParent {
        window_id: WindowId,
        parent_id: Option<WindowId>,
        modal: bool,
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
        bgra: Arc<Vec<u8>>,
        damage: Vec<(i32, i32, i32, i32)>,
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
    WindowClosed {
        window_id: WindowId,
    },
    /// Signals that a new decoded frame is available in the per-window latest-frame mailbox.
    FrameReady {
        window_id: WindowId,
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
    /// The server's clipboard changed (files already presented locally).
    ClipboardOffer {
        clip: ReceivedClipboard,
        fetch: crate::xfer::SelectionFetcher,
    },
    /// An app in remote window `window_id` started a drag (files already presented
    /// locally): continue it on the local desktop.
    RemoteDragStart {
        window_id: WindowId,
        clip: ReceivedClipboard,
        fetch: crate::xfer::SelectionFetcher,
    },
    /// The remote app the local drag with offer `offer_id` was dropped on is done with it.
    DragFinished {
        offer_id: u64,
    },
    PrimarySelectionText {
        text: String,
    },
    AppIconsUpdated {
        entries: Vec<AppIconEntry>,
    },
    SetMaxScale {
        cap: RenderScaleCap,
    },
    Disconnected,
}

#[derive(Clone)]
enum ActiveCursor {
    Hidden,
    Named(CursorIcon),
    Bitmap {
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        bgra: Vec<u8>,
    },
}

struct LatestDecodedFrame {
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    geometry: FrameRect,
    input_bounds: FrameRect,
    bgra: Arc<Vec<u8>>,
    damage: Vec<(i32, i32, i32, i32)>,
}

type LatestFrameMap = Arc<Mutex<HashMap<WindowId, LatestDecodedFrame>>>;

/// Sender handle used by background tasks to wake and message the Wayland event loop.
/// Video frames are coalesced per window into a 1-slot latest-frame mailbox so stale
/// decoded frames never accumulate in the UI channel.
#[derive(Clone)]
pub struct UiSender {
    tx: CalloopSender<ClientUiEvent>,
    latest_frames: LatestFrameMap,
}

impl UiSender {
    pub fn send(
        &self,
        event: ClientUiEvent,
    ) -> Result<(), std::sync::mpsc::SendError<ClientUiEvent>> {
        self.tx.send(event)
    }

    pub fn push_frame(
        &self,
        window_id: WindowId,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        geometry: FrameRect,
        input_bounds: FrameRect,
        bgra: Arc<Vec<u8>>,
        mut damage: Vec<(i32, i32, i32, i32)>,
    ) {
        let should_notify = {
            let mut map = self.latest_frames.lock().unwrap();
            let prev = map.remove(&window_id);
            if let Some(old) = &prev {
                if old.width != width
                    || old.height != height
                    || old.damage.is_empty()
                    || damage.is_empty()
                {
                    damage.clear(); // full-frame damage
                } else if damage.len() + old.damage.len() <= 32 {
                    damage.extend_from_slice(&old.damage);
                } else {
                    // Collapse accumulated damage into a single tight bounding box to stay O(1) memory.
                    let mut min_x = i32::MAX;
                    let mut min_y = i32::MAX;
                    let mut max_x = 0i32;
                    let mut max_y = 0i32;
                    for &(dx, dy, dw, dh) in damage.iter().chain(old.damage.iter()) {
                        min_x = min_x.min(dx);
                        min_y = min_y.min(dy);
                        max_x = max_x.max(dx.saturating_add(dw));
                        max_y = max_y.max(dy.saturating_add(dh));
                    }
                    damage.clear();
                    damage.push((min_x, min_y, (max_x - min_x).max(1), (max_y - min_y).max(1)));
                }
            }
            map.insert(
                window_id,
                LatestDecodedFrame {
                    width,
                    height,
                    logical_width,
                    logical_height,
                    geometry,
                    input_bounds,
                    bgra,
                    damage,
                },
            );
            prev.is_none()
        };
        if should_notify {
            let _ = self.tx.send(ClientUiEvent::FrameReady { window_id });
        }
    }
}

/// Receiver pair passed into `run_event_loop`.
pub struct UiReceiver {
    rx: Channel<ClientUiEvent>,
    latest_frames: LatestFrameMap,
}

/// Holds an `xdg_toplevel_icon_v1` and its backing `wl_shm` buffers.
/// `Drop` destroys the icon object before releasing the backing buffers, as required
/// by the `xdg-toplevel-icon-v1` protocol (`no_buffer` error).
struct AppliedWindowIcon {
    icon: XdgToplevelIconV1,
    _buffers: Vec<Buffer>,
}

impl Drop for AppliedWindowIcon {
    fn drop(&mut self) {
        self.icon.destroy();
    }
}

struct PendingFrame {
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    geometry: FrameRect,
    input_bounds: FrameRect,
    bgra: Arc<Vec<u8>>,
    damage: Vec<(i32, i32, i32, i32)>,
}

struct RemoteWindow {
    window: Window,
    viewport: Option<WpViewport>,
    fractional_scale: Option<WpFractionalScaleV1>,
    raw_scale: f64,
    effective_scale: f64,
    has_fractional_scale: bool,
    app_id: String,
    applied_icon: Option<AppliedWindowIcon>,
    has_pixel_icon: bool,
    configured: bool,
    width: u32,
    height: u32,
    maximized: bool,
    fullscreen: bool,
    activated: bool,
    tiled: bool,
    pending_frame: Option<PendingFrame>,
    pending_resize_prev: Option<((u32, u32), Instant)>,
    /// Logical size of the last frame shown, and the window and input rectangles inside
    /// it. Frames include the app's drop shadow around the window rectangle.
    frame_size: (u32, u32),
    geometry: FrameRect,
    input_bounds: Option<FrameRect>,
    /// The remote window this one is a dialog of, kept while the link can't be made yet
    /// (see [`ClientApp::dialog_link_target`]) so it is made once it can.
    dialog_parent: Option<WindowId>,
    /// The parent the host compositor has for this window: the one last set with
    /// `xdg_toplevel.set_parent`, moved up when that parent closes as xdg-shell does.
    host_parent: Option<WindowId>,
    modal: bool,
    /// The `xdg_dialog_v1` marking this window a dialog, while it has a parent.
    xdg_dialog: Option<XdgDialogV1>,
}

impl RemoteWindow {
    /// Whether the app takes pointer input outside its window rectangle: the resize margin
    /// apps that draw their own shadow (GTK, libadwaita, Firefox, Chromium) keep there, to
    /// start edge resizes themselves.
    fn app_has_resize_margin(&self) -> bool {
        let input = self
            .input_bounds
            .unwrap_or(FrameRect::full(self.frame_size.0, self.frame_size.1));
        input_reaches_past(input, self.geometry)
    }

    /// The edge to resize from when the pointer is at `(x, y)` (surface-local), for windows
    /// without a resize margin of their own: a band along the window rectangle's edges.
    fn border_resize_edge(&self, x: f64, y: f64) -> Option<(ResizeEdge, CursorIcon)> {
        if self.maximized || self.fullscreen || self.tiled || self.app_has_resize_margin() {
            return None;
        }
        let g = self.geometry;
        if g.is_empty() {
            return border_resize_edge(x, y, self.width, self.height);
        }
        border_resize_edge(x - g.x as f64, y - g.y as f64, g.width, g.height)
    }

    /// See [`apply_frame_rects`]; also remembers the frame's size and window rectangle.
    fn apply_frame_rects(
        &mut self,
        compositor: &CompositorState,
        frame_w: u32,
        frame_h: u32,
        geometry: FrameRect,
        input_bounds: FrameRect,
    ) {
        apply_frame_rects(
            self.window.xdg_surface(),
            self.window.wl_surface(),
            compositor,
            (frame_w, frame_h),
            geometry,
            input_bounds,
            &mut self.input_bounds,
        );
        self.frame_size = (frame_w, frame_h);
        self.geometry = geometry;
    }
}

/// Point the host compositor at the window rectangle inside a frame of logical size
/// `frame`, and limit pointer input to `input_bounds` so clicks on the app's outer shadow
/// reach whatever is below. `applied` tracks the input region last set on `surface`.
fn apply_frame_rects(
    xdg_surface: &xdg_surface::XdgSurface,
    surface: &wl_surface::WlSurface,
    compositor: &CompositorState,
    frame: (u32, u32),
    geometry: FrameRect,
    input_bounds: FrameRect,
    applied: &mut Option<FrameRect>,
) {
    xdg_surface.set_window_geometry(
        geometry.x,
        geometry.y,
        geometry.width.max(1) as i32,
        geometry.height.max(1) as i32,
    );
    let full = FrameRect::full(frame.0, frame.1);
    let input_bounds = if input_bounds == full {
        None
    } else {
        Some(input_bounds)
    };
    if input_bounds != *applied {
        match input_bounds {
            Some(b) => match Region::new(compositor) {
                Ok(region) => {
                    // An empty region: the surface takes no input (click-through).
                    if !b.is_empty() {
                        region.add(b.x, b.y, b.width as i32, b.height as i32);
                    }
                    surface.set_input_region(Some(region.wl_region()));
                }
                Err(err) => warn!(error = %err, "failed to create input region"),
            },
            None => surface.set_input_region(None),
        }
        *applied = input_bounds;
    }
}

impl Drop for RemoteWindow {
    fn drop(&mut self) {
        if let Some(fs) = self.fractional_scale.take() {
            fs.destroy();
        }
        if let Some(vp) = self.viewport.take() {
            vp.destroy();
        }
        if let Some(dialog) = self.xdg_dialog.take() {
            dialog.destroy();
        }
    }
}

struct RemotePopup {
    popup: Popup,
    viewport: Option<WpViewport>,
    configured: bool,
    pending_frame: Option<PendingFrame>,
    next_token: u32,
    /// Input region last set on the popup's surface (`None`: the whole surface).
    input_bounds: Option<FrameRect>,
}

impl Drop for RemotePopup {
    fn drop(&mut self) {
        if let Some(vp) = self.viewport.take() {
            vp.destroy();
        }
    }
}

struct ClientApp {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor_state: CompositorState,
    xdg_shell: XdgShell,
    shm: Shm,
    data_device_manager_state: Option<DataDeviceManagerState>,
    primary_selection_manager_state: Option<PrimarySelectionManagerState>,
    toplevel_icon_manager: Option<XdgToplevelIconManagerV1>,
    dialog_manager: Option<XdgWmDialogV1>,
    fractional_scale_manager: Option<WpFractionalScaleManagerV1>,
    viewporter: Option<WpViewporter>,
    max_scale: RenderScaleCap,
    last_sent_display_scale: f64,
    pool: SlotPool,
    icon_pool: SlotPool,
    app_icons: Vec<AppIconEntry>,
    decoded_icon_cache: HashMap<String, Vec<(u32, i32, Vec<u8>)>>,
    conn: Connection,
    qh: QueueHandle<Self>,

    seat: Option<wl_seat::WlSeat>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    themed_pointer: Option<ThemedPointer>,
    custom_cursor_surface: Option<wl_surface::WlSurface>,
    custom_cursor_viewport: Option<WpViewport>,
    data_device: Option<DataDevice>,
    copy_paste_source: Option<CopyPasteSource>,
    primary_selection_device: Option<PrimarySelectionDevice>,
    primary_selection_source: Option<PrimarySelectionSource>,
    focused_window_id: WindowId,
    pointer_window_id: WindowId,
    last_button_serial: u32,
    last_input_serial: u32,
    last_pointer_enter_serial: u32,
    pressed_buttons: HashSet<u32>,
    last_cursor: ActiveCursor,
    border_resize_active: bool,

    windows: HashMap<WindowId, RemoteWindow>,
    popups: HashMap<WindowId, RemotePopup>,
    surface_to_remote: HashMap<u32, WindowId>,
    latest_frames: LatestFrameMap,

    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    /// The server's clipboard our `copy_paste_source` offers.
    clipboard: Option<(ReceivedClipboard, crate::xfer::SelectionFetcher)>,
    /// The compositor echoed `copy_paste_source` back as the selection, so it took it. A
    /// compositor ignores a selection set while none of our windows has keyboard focus;
    /// an unconfirmed one is set again on the next keyboard enter.
    clipboard_confirmed: bool,
    /// A type only our own clipboard sources offer, to recognise their echo exactly.
    source_marker: String,
    /// A local drag over one of our windows.
    incoming_drag: Option<IncomingDrag>,
    /// Relays of local drags dropped on a remote window, by offer id, waiting for the
    /// remote app to finish reading them.
    finishing_drags: HashMap<u64, std::sync::mpsc::Sender<IncomingDragEvent>>,
    /// A remote app's drag of files, continued on the local desktop.
    outgoing_drag: Option<OutgoingDrag>,
    last_primary: String,
    exit: bool,
}

impl ClientApp {
    fn apply_active_cursor(&mut self) {
        let Some(tp) = &self.themed_pointer else {
            return;
        };
        match &self.last_cursor {
            ActiveCursor::Hidden => {
                let _ = tp.hide_cursor();
            }
            ActiveCursor::Named(icon) => {
                let _ = tp.set_cursor(&self.conn, *icon);
            }
            ActiveCursor::Bitmap {
                width,
                height,
                logical_width,
                logical_height,
                hotspot_x,
                hotspot_y,
                bgra,
            } => {
                let w = *width;
                let h = *height;
                let lw = (*logical_width).max(1);
                let lh = (*logical_height).max(1);
                let byte_len = w
                    .checked_mul(h)
                    .and_then(|px| px.checked_mul(4))
                    .map(|n| n as usize);
                let Some(byte_len) = byte_len else {
                    let _ = tp.set_cursor(&self.conn, CursorIcon::Default);
                    return;
                };
                if w == 0
                    || h == 0
                    || w > MAX_CURSOR_DIM
                    || h > MAX_CURSOR_DIM
                    || bgra.len() < byte_len
                {
                    let _ = tp.set_cursor(&self.conn, CursorIcon::Default);
                    return;
                }
                let surf = self
                    .custom_cursor_surface
                    .get_or_insert_with(|| self.compositor_state.create_surface(&self.qh))
                    .clone();
                if self.custom_cursor_viewport.is_none() {
                    if let Some(vp_mgr) = &self.viewporter {
                        self.custom_cursor_viewport =
                            Some(vp_mgr.get_viewport(&surf, &self.qh, ()));
                    }
                }
                let stride = (w * 4) as i32;
                if let Ok((buffer, canvas)) =
                    self.pool
                        .create_buffer(w as i32, h as i32, stride, wl_shm::Format::Argb8888)
                {
                    canvas[..byte_len].copy_from_slice(&bgra[..byte_len]);
                    if let Some(vp) = &self.custom_cursor_viewport {
                        surf.set_buffer_scale(1);
                        vp.set_destination(lw as i32, lh as i32);
                    } else if w % lw == 0 && h % lh == 0 && (w / lw) == (h / lh) {
                        surf.set_buffer_scale((w / lw).max(1) as i32);
                    } else {
                        surf.set_buffer_scale(1);
                    }
                    tp.pointer().set_cursor(
                        self.last_pointer_enter_serial,
                        Some(&surf),
                        *hotspot_x,
                        *hotspot_y,
                    );
                    if buffer.attach_to(&surf).is_ok() {
                        surf.damage_buffer(0, 0, w as i32, h as i32);
                        surf.commit();
                    }
                }
            }
        }
    }
    fn release_pressed_buttons(&mut self, window_id: WindowId) {
        for button in self.pressed_buttons.drain() {
            let _ = self.ctrl_tx.send(ControlMessage::PointerButton {
                window_id,
                button,
                pressed: false,
            });
        }
    }

    /// The parent window `window_id` can be given on the host compositor now: its remote
    /// parent, once that is mapped (xdg-shell treats an unmapped parent as none) and if
    /// the link makes no loop (a protocol error that would end the connection).
    fn dialog_link_target(&self, window_id: WindowId) -> Option<WindowId> {
        dialog_link_target(
            window_id,
            self.windows.get(&window_id)?.dialog_parent,
            |id| self.windows.get(&id).map(|w| (w.configured, w.host_parent)),
        )
    }

    /// Make window `window_id` a dialog of its remote parent on the host compositor
    /// (`xdg_toplevel.set_parent`, plus `xdg_dialog_v1` for the modal hint), or an
    /// ordinary window while [`Self::dialog_link_target`] has no parent for it.
    /// `true` when its parent on the host changed.
    fn apply_dialog_parent(&mut self, window_id: WindowId) -> bool {
        let target = self.dialog_link_target(window_id);
        let Some(win) = self.windows.get(&window_id) else {
            return false;
        };
        let changed = win.host_parent != target;
        if changed {
            let parent = target.and_then(|p| self.windows.get(&p)).map(|p| &p.window);
            win.window.set_parent(parent);
        }
        let dialog_manager = self.dialog_manager.clone();
        let Some(win) = self.windows.get_mut(&window_id) else {
            return false;
        };
        win.host_parent = target;
        if target.is_none() {
            if let Some(dialog) = win.xdg_dialog.take() {
                dialog.destroy();
            }
        } else if let Some(mgr) = dialog_manager {
            let dialog = win
                .xdg_dialog
                .get_or_insert_with(|| mgr.get_xdg_dialog(win.window.xdg_toplevel(), &self.qh, ()));
            if win.modal {
                dialog.set_modal();
            } else {
                dialog.unset_modal();
            }
        }
        debug!(
            window_id,
            parent = ?win.dialog_parent,
            host_parent = ?target,
            modal = win.modal,
            "applied remote dialog relationship"
        );
        changed
    }

    /// Make the dialog links that are waiting and can be made now: after a window mapped,
    /// closed, or changed its own link.
    fn apply_waiting_dialog_parents(&mut self) {
        // Each pass that changes a link may unblock another; there are at most as many
        // links as windows.
        for _ in 0..=self.windows.len() {
            let waiting: Vec<WindowId> = self
                .windows
                .iter()
                .filter(|(_, w)| w.dialog_parent.is_some() && w.dialog_parent != w.host_parent)
                .map(|(&id, _)| id)
                .collect();
            let mut changed = false;
            for id in waiting {
                changed |= self.apply_dialog_parent(id);
            }
            if !changed {
                break;
            }
        }
    }

    fn apply_window_icon(&mut self, window_id: WindowId, commit: bool) {
        let Some(mgr) = self.toplevel_icon_manager.clone() else {
            return;
        };
        let Some(win) = self.windows.get(&window_id) else {
            return;
        };
        if win.app_id.is_empty() {
            return;
        }
        let app_id = win.app_id.clone();
        let already_has_pixels = win.has_pixel_icon;
        let already_has_icon = win.applied_icon.is_some();

        // Try matching a remote `.desktop` icon from `AppList` and uploading ARGB8888 pixel
        // buffers at 1x (64px) and 2x (128px) so compositors show the exact remote app icon
        // even when the app is not installed locally.
        let mut buffers: Vec<Buffer> = Vec::new();
        let mut icon_obj: Option<XdgToplevelIconV1> = None;

        if let Some(entry_id) = match_app_icon(&app_id, &self.app_icons).map(|e| e.id.clone()) {
            if !self.decoded_icon_cache.contains_key(&entry_id) {
                if let Some(entry) = self.app_icons.iter().find(|e| e.id == entry_id) {
                    let mut decoded: Vec<(u32, i32, Vec<u8>)> = Vec::new();
                    for (size, scale) in [(64u32, 1i32), (128u32, 2i32)] {
                        if let Some(argb) = decode_icon_argb8888(&entry.icon_data, size) {
                            decoded.push((size, scale, argb));
                        }
                    }
                    self.decoded_icon_cache.insert(entry_id.clone(), decoded);
                }
            }
            if let Some(decoded) = self.decoded_icon_cache.get(&entry_id) {
                if !decoded.is_empty() {
                    let icon = mgr.create_icon(&self.qh, ());
                    for (size, scale, argb) in decoded {
                        let stride = (*size * 4) as i32;
                        if let Ok((buffer, canvas)) = self.icon_pool.create_buffer(
                            *size as i32,
                            *size as i32,
                            stride,
                            wl_shm::Format::Argb8888,
                        ) {
                            canvas[..argb.len()].copy_from_slice(argb);
                            icon.add_buffer(buffer.wl_buffer(), *scale);
                            buffers.push(buffer);
                        }
                    }
                    if !buffers.is_empty() {
                        icon_obj = Some(icon);
                    } else {
                        icon.destroy();
                    }
                }
            }
        }

        let has_pixel_icon = !buffers.is_empty();
        if icon_obj.is_none() {
            if already_has_pixels || already_has_icon {
                return;
            }
            // Fall back to icon theme name lookup via `xdg_toplevel_icon_v1.set_name`.
            let icon = mgr.create_icon(&self.qh, ());
            icon.set_name(app_id);
            icon_obj = Some(icon);
        }

        if let (Some(icon), Some(win)) = (icon_obj, self.windows.get_mut(&window_id)) {
            mgr.set_icon(win.window.xdg_toplevel(), Some(&icon));
            if commit {
                win.window.wl_surface().commit();
            }
            win.applied_icon = Some(AppliedWindowIcon {
                icon,
                _buffers: buffers,
            });
            win.has_pixel_icon = has_pixel_icon;
        }
    }

    fn handle_ui_event(&mut self, event: ClientUiEvent) {
        match event {
            ClientUiEvent::WindowCreated {
                window_id,
                title,
                app_id,
                width,
                height,
            } => {
                if let Some(existing) = self.windows.get_mut(&window_id) {
                    if !title.is_empty() {
                        existing.window.set_title(title);
                    }
                    let app_id_changed = !app_id.is_empty() && existing.app_id != app_id;
                    if app_id_changed {
                        existing.window.set_app_id(app_id.clone());
                        existing.app_id = app_id;
                        existing.applied_icon = None;
                        existing.has_pixel_icon = false;
                    }
                    if app_id_changed {
                        self.apply_window_icon(window_id, true);
                    }
                    return;
                }
                let wl_surf = self.compositor_state.create_surface(&self.qh);
                let surf_id = wl_surf.id().protocol_id();
                let viewport = self
                    .viewporter
                    .as_ref()
                    .map(|vp| vp.get_viewport(&wl_surf, &self.qh, ()));
                let fractional_scale = self
                    .fractional_scale_manager
                    .as_ref()
                    .map(|fsm| fsm.get_fractional_scale(&wl_surf, &self.qh, window_id));
                let initial_raw = self.current_max_output_scale();
                let initial_effective = self.max_scale.apply(initial_raw);
                let window = self.xdg_shell.create_window(
                    wl_surf,
                    WindowDecorations::RequestClient,
                    &self.qh,
                );
                window.set_title(if title.is_empty() {
                    format!("Oriel #{window_id}")
                } else {
                    title
                });
                if !app_id.is_empty() {
                    window.set_app_id(app_id.clone());
                }

                self.surface_to_remote.insert(surf_id, window_id);
                let has_app_id = !app_id.is_empty();
                self.windows.insert(
                    window_id,
                    RemoteWindow {
                        window,
                        viewport,
                        fractional_scale,
                        raw_scale: initial_raw,
                        effective_scale: initial_effective,
                        has_fractional_scale: false,
                        app_id,
                        applied_icon: None,
                        has_pixel_icon: false,
                        configured: false,
                        width: width.max(64),
                        height: height.max(64),
                        maximized: false,
                        fullscreen: false,
                        activated: true,
                        tiled: false,
                        pending_frame: None,
                        pending_resize_prev: None,
                        frame_size: (0, 0),
                        geometry: FrameRect::full(0, 0),
                        input_bounds: None,
                        dialog_parent: None,
                        host_parent: None,
                        modal: false,
                        xdg_dialog: None,
                    },
                );
                if has_app_id {
                    // Apply icon before the initial commit so both `app_id` and `icon`
                    // take effect in a single initial surface commit.
                    self.apply_window_icon(window_id, false);
                }
                if let Some(win) = self.windows.get(&window_id) {
                    win.window.commit();
                }
            }
            ClientUiEvent::WindowParent {
                window_id,
                parent_id,
                modal,
            } => {
                let Some(win) = self.windows.get_mut(&window_id) else {
                    return;
                };
                win.dialog_parent = parent_id;
                win.modal = modal;
                self.apply_dialog_parent(window_id);
                // A link changing can unblock others waiting on a loop it would have made.
                self.apply_waiting_dialog_parents();
            }
            ClientUiEvent::PopupCreated {
                window_id,
                parent_id,
                x,
                y,
                width,
                height,
            } => {
                if self.popups.contains_key(&window_id) {
                    return;
                }
                let parent_xdg = self
                    .windows
                    .get(&parent_id)
                    .map(|w| w.window.xdg_surface().clone())
                    .or_else(|| {
                        self.popups
                            .get(&parent_id)
                            .map(|p| p.popup.xdg_surface().clone())
                    });
                let Some(parent_xdg) = parent_xdg else {
                    warn!(window_id, parent_id, "PopupCreated for unknown parent_id");
                    return;
                };

                let Ok(positioner) = XdgPositioner::new(&self.xdg_shell) else {
                    return;
                };
                positioner.set_size(width.max(1) as i32, height.max(1) as i32);
                positioner.set_anchor_rect(x, y, 1, 1);
                positioner.set_anchor(Anchor::TopLeft);
                positioner.set_gravity(Gravity::BottomRight);
                positioner.set_constraint_adjustment(
                    ConstraintAdjustment::FlipX
                        | ConstraintAdjustment::FlipY
                        | ConstraintAdjustment::SlideX
                        | ConstraintAdjustment::SlideY,
                );
                positioner.set_reactive();

                match Popup::new(
                    &parent_xdg,
                    &positioner,
                    &self.qh,
                    &self.compositor_state,
                    &self.xdg_shell,
                ) {
                    Ok(popup) => {
                        let surf_id = popup.wl_surface().id().protocol_id();
                        let viewport = self
                            .viewporter
                            .as_ref()
                            .map(|vp| vp.get_viewport(popup.wl_surface(), &self.qh, ()));
                        self.surface_to_remote.insert(surf_id, window_id);
                        self.popups.insert(
                            window_id,
                            RemotePopup {
                                popup,
                                viewport,
                                configured: false,
                                pending_frame: None,
                                next_token: 1,
                                input_bounds: None,
                            },
                        );
                    }
                    Err(err) => warn!(window_id, error = %err, "failed to create sctk Popup"),
                }
            }
            ClientUiEvent::PopupRepositioned {
                window_id,
                x,
                y,
                width,
                height,
            } => {
                if let Some(rp) = self.popups.get_mut(&window_id) {
                    if let Ok(positioner) = XdgPositioner::new(&self.xdg_shell) {
                        positioner.set_size(width.max(1) as i32, height.max(1) as i32);
                        positioner.set_anchor_rect(x, y, 1, 1);
                        positioner.set_anchor(Anchor::TopLeft);
                        positioner.set_gravity(Gravity::BottomRight);
                        positioner.set_constraint_adjustment(
                            ConstraintAdjustment::FlipX
                                | ConstraintAdjustment::FlipY
                                | ConstraintAdjustment::SlideX
                                | ConstraintAdjustment::SlideY,
                        );
                        positioner.set_reactive();
                        let token = rp.next_token;
                        rp.next_token = rp.next_token.wrapping_add(1);
                        rp.popup.reposition(&positioner, token);
                    }
                }
            }
            ClientUiEvent::PopupFrame {
                window_id,
                width,
                height,
                logical_width,
                logical_height,
                geometry,
                input_bounds,
                bgra,
                damage,
            } => {
                if let Some(rp) = self.popups.get_mut(&window_id) {
                    if !rp.configured {
                        rp.pending_frame = Some(PendingFrame {
                            width,
                            height,
                            logical_width,
                            logical_height,
                            geometry,
                            input_bounds,
                            bgra,
                            damage,
                        });
                    } else {
                        apply_frame_rects(
                            rp.popup.xdg_surface(),
                            rp.popup.wl_surface(),
                            &self.compositor_state,
                            (logical_width, logical_height),
                            geometry,
                            input_bounds,
                            &mut rp.input_bounds,
                        );
                        let surf = rp.popup.wl_surface().clone();
                        draw_shm_surface(
                            &mut self.pool,
                            &surf,
                            rp.viewport.as_ref(),
                            width,
                            height,
                            logical_width,
                            logical_height,
                            &bgra,
                            &damage,
                            wl_shm::Format::Argb8888,
                        );
                    }
                }
            }
            ClientUiEvent::StartMove { window_id } => {
                self.release_pressed_buttons(window_id);
                if let (Some(seat), Some(win)) = (&self.seat, self.windows.get(&window_id)) {
                    win.window.move_(seat, self.last_button_serial);
                }
            }
            ClientUiEvent::StartResize { window_id, edge } => {
                self.release_pressed_buttons(window_id);
                if let (Some(seat), Some(win)) = (&self.seat, self.windows.get(&window_id)) {
                    if let WEnum::Value(resize_edge) = WEnum::from(edge) {
                        win.window
                            .resize(seat, self.last_button_serial, resize_edge);
                    } else {
                        win.window
                            .resize(seat, self.last_button_serial, ResizeEdge::BottomRight);
                    }
                }
            }
            ClientUiEvent::SetMaximized {
                window_id,
                maximized,
            } => {
                if let Some(win) = self.windows.get(&window_id) {
                    if maximized {
                        win.window.set_maximized();
                    } else {
                        win.window.unset_maximized();
                    }
                }
            }
            ClientUiEvent::SetMinimized { window_id } => {
                if let Some(win) = self.windows.get(&window_id) {
                    win.window.set_minimized();
                }
            }
            ClientUiEvent::SetFullscreen {
                window_id,
                fullscreen,
            } => {
                if let Some(win) = self.windows.get(&window_id) {
                    if fullscreen {
                        win.window.set_fullscreen(None);
                    } else {
                        win.window.unset_fullscreen();
                    }
                }
            }
            ClientUiEvent::ShowWindowMenu { window_id, x, y } => {
                if let (Some(seat), Some(win)) = (&self.seat, self.windows.get(&window_id)) {
                    win.window
                        .show_window_menu(seat, self.last_button_serial, (x, y));
                }
            }
            ClientUiEvent::SetDecorationMode {
                window_id,
                server_side,
            } => {
                if let Some(win) = self.windows.get(&window_id) {
                    win.window.request_decoration_mode(Some(if server_side {
                        DecorationMode::Server
                    } else {
                        DecorationMode::Client
                    }));
                    win.window.commit();
                }
            }
            ClientUiEvent::WindowClosed { window_id } => {
                if let Some(win) = self.windows.remove(&window_id) {
                    self.surface_to_remote
                        .remove(&win.window.wl_surface().id().protocol_id());
                    // xdg-shell gives the closed window's children its own parent.
                    for child in self.windows.values_mut() {
                        if child.host_parent == Some(window_id) {
                            child.host_parent = win.host_parent;
                        }
                    }
                }
                if let Some(rp) = self.popups.remove(&window_id) {
                    self.surface_to_remote
                        .remove(&rp.popup.wl_surface().id().protocol_id());
                }
                self.latest_frames.lock().unwrap().remove(&window_id);
                self.apply_waiting_dialog_parents();
            }
            ClientUiEvent::FrameReady { window_id } => {
                let Some(LatestDecodedFrame {
                    width,
                    height,
                    logical_width,
                    logical_height,
                    geometry,
                    input_bounds,
                    bgra,
                    damage,
                }) = self.latest_frames.lock().unwrap().remove(&window_id)
                else {
                    return;
                };
                if let Some(win) = self.windows.get_mut(&window_id) {
                    if !win.configured {
                        win.pending_frame = Some(PendingFrame {
                            width,
                            height,
                            logical_width,
                            logical_height,
                            geometry,
                            input_bounds,
                            bgra,
                            damage,
                        });
                    } else {
                        let lw = logical_width.max(1);
                        let lh = logical_height.max(1);
                        // The configured size is the window geometry, without the shadow.
                        let win_size = (geometry.width, geometry.height);
                        if let Some((prev_size, since)) = win.pending_resize_prev {
                            if win_size == prev_size
                                && win_size != (win.width, win.height)
                                && since.elapsed() < Duration::from_millis(250)
                            {
                                return;
                            }
                            win.pending_resize_prev = None;
                        }
                        win.apply_frame_rects(
                            &self.compositor_state,
                            lw,
                            lh,
                            geometry,
                            input_bounds,
                        );
                        let surf = win.window.wl_surface().clone();
                        draw_shm_surface(
                            &mut self.pool,
                            &surf,
                            win.viewport.as_ref(),
                            width,
                            height,
                            lw,
                            lh,
                            &bgra,
                            &damage,
                            wl_shm::Format::Argb8888,
                        );
                    }
                }
            }
            ClientUiEvent::CursorUpdate {
                window_id,
                visible,
                shape_name,
            } => {
                if window_id == self.pointer_window_id || window_id == self.focused_window_id {
                    let icon = shape_name
                        .parse::<CursorIcon>()
                        .unwrap_or(CursorIcon::Default);
                    self.last_cursor = if visible {
                        ActiveCursor::Named(icon)
                    } else {
                        ActiveCursor::Hidden
                    };
                    if !self.border_resize_active {
                        self.apply_active_cursor();
                    }
                }
            }
            ClientUiEvent::CursorBitmap {
                window_id,
                width,
                height,
                logical_width,
                logical_height,
                hotspot_x,
                hotspot_y,
                bgra,
            } => {
                if window_id == self.pointer_window_id || window_id == self.focused_window_id {
                    self.last_cursor = ActiveCursor::Bitmap {
                        width,
                        height,
                        logical_width,
                        logical_height,
                        hotspot_x,
                        hotspot_y,
                        bgra,
                    };
                    if !self.border_resize_active {
                        self.apply_active_cursor();
                    }
                }
            }
            ClientUiEvent::ClipboardOffer { clip, fetch } => {
                debug!(
                    mimes = ?clip.mimes,
                    text = clip.text.as_ref().map(String::len),
                    files = clip.files.as_ref().map(Vec::len),
                    "server clipboard changed"
                );
                // `send_request` answers from `self.clipboard`, so it changes only along
                // with the source.
                if let (Some(mgr), Some(data_device)) =
                    (&self.data_device_manager_state, &self.data_device)
                {
                    let mut mimes = clip.mime_types();
                    mimes.push(self.source_marker.clone());
                    let source = mgr.create_copy_paste_source(&self.qh, mimes);
                    source.set_selection(data_device, self.last_input_serial);
                    self.copy_paste_source = Some(source);
                    self.clipboard = Some((clip, fetch));
                    self.clipboard_confirmed = false;
                }
            }
            ClientUiEvent::RemoteDragStart {
                window_id,
                clip,
                fetch,
            } => {
                self.start_outgoing_drag(window_id, clip, fetch);
            }
            ClientUiEvent::DragFinished { offer_id } => {
                // Forget relays that already gave up (their receiver is gone).
                self.finishing_drags
                    .retain(|_, tx| tx.send(IncomingDragEvent::Ping).is_ok());
                if let Some(events) = self.finishing_drags.remove(&offer_id) {
                    let _ = events.send(IncomingDragEvent::Finished);
                } else if let Some(drag) = self
                    .incoming_drag
                    .as_ref()
                    .filter(|d| d.offer_id == offer_id)
                {
                    let _ = drag.events.send(IncomingDragEvent::Finished);
                }
            }
            ClientUiEvent::PrimarySelectionText { text } => {
                if !text.is_empty()
                    && (self.primary_selection_source.is_none() || text != self.last_primary)
                {
                    self.last_primary = text;
                    if let (Some(mgr), Some(primary_device)) = (
                        &self.primary_selection_manager_state,
                        &self.primary_selection_device,
                    ) {
                        let source =
                            mgr.create_selection_source(&self.qh, SUPPORTED_TEXT_MIMES.to_vec());
                        source.set_selection(primary_device, self.last_input_serial);
                        self.primary_selection_source = Some(source);
                    }
                }
            }
            ClientUiEvent::AppIconsUpdated { entries } => {
                self.app_icons = entries;
                self.decoded_icon_cache.clear();
                let pending: Vec<WindowId> = self
                    .windows
                    .iter()
                    .filter(|(_, w)| !w.has_pixel_icon && !w.app_id.is_empty())
                    .map(|(&id, _)| id)
                    .collect();
                for window_id in pending {
                    self.apply_window_icon(window_id, true);
                }
            }
            ClientUiEvent::SetMaxScale { cap } => {
                if self.max_scale != cap {
                    self.max_scale = cap;
                    self.reapply_scale_cap();
                }
            }
            ClientUiEvent::Disconnected => {
                debug!("server disconnected; exiting Wayland client loop");
                self.exit = true;
            }
        }
        let _ = self.conn.flush();
    }

    fn current_max_output_scale(&self) -> f64 {
        self.output_state
            .outputs()
            .filter_map(|o| self.output_state.info(&o))
            .map(|info| output_scale(&info))
            .fold(1.0, f64::max)
    }

    fn maybe_send_display_scale(&mut self) {
        let raw_max = self
            .windows
            .values()
            .map(|w| w.raw_scale)
            .fold(self.current_max_output_scale(), f64::max);
        let effective = self.max_scale.apply(raw_max);
        if (effective - self.last_sent_display_scale).abs() > 0.005 {
            self.last_sent_display_scale = effective;
            debug!(scale_factor = effective, "sending DisplayScale to server");
            let _ = self.ctrl_tx.send(ControlMessage::DisplayScale {
                scale_factor: effective,
            });
        }
    }

    fn update_window_scale(&mut self, window_id: WindowId, raw_scale: f64, from_fractional: bool) {
        let Some(win) = self.windows.get_mut(&window_id) else {
            return;
        };
        if !from_fractional && win.has_fractional_scale {
            return;
        }
        if from_fractional {
            win.has_fractional_scale = true;
        }
        win.raw_scale = raw_scale.max(1.0);
        let new_effective = self.max_scale.apply(win.raw_scale);
        if (new_effective - win.effective_scale).abs() > 0.005 {
            win.effective_scale = new_effective;
            if win.configured {
                debug!(
                    window_id,
                    raw_scale = win.raw_scale,
                    effective_scale = new_effective,
                    "window scale changed -> WindowConfigured"
                );
                let _ = self.ctrl_tx.send(ControlMessage::WindowConfigured {
                    window_id,
                    width: win.width,
                    height: win.height,
                    scale_factor: new_effective,
                    maximized: win.maximized,
                    fullscreen: win.fullscreen,
                    activated: win.activated,
                    tiled: win.tiled,
                });
            }
        }
        self.maybe_send_display_scale();
    }

    fn reapply_scale_cap(&mut self) {
        let cap = self.max_scale;
        for (&window_id, win) in &mut self.windows {
            let new_effective = cap.apply(win.raw_scale);
            if (new_effective - win.effective_scale).abs() > 0.005 {
                win.effective_scale = new_effective;
                if win.configured {
                    debug!(
                        window_id,
                        effective_scale = new_effective,
                        "scale cap changed -> WindowConfigured"
                    );
                    let _ = self.ctrl_tx.send(ControlMessage::WindowConfigured {
                        window_id,
                        width: win.width,
                        height: win.height,
                        scale_factor: new_effective,
                        maximized: win.maximized,
                        fullscreen: win.fullscreen,
                        activated: win.activated,
                        tiled: win.tiled,
                    });
                }
            }
        }
        self.maybe_send_display_scale();
    }
}

/// Whether the `input` rectangle extends past the window rectangle `geometry` on any side.
fn input_reaches_past(input: FrameRect, geometry: FrameRect) -> bool {
    let g = geometry;
    !input.is_empty()
        && !g.is_empty()
        && (input.x < g.x
            || input.y < g.y
            || input.x + input.width as i32 > g.x + g.width as i32
            || input.y + input.height as i32 > g.y + g.height as i32)
}

fn border_resize_edge(x: f64, y: f64, width: u32, height: u32) -> Option<(ResizeEdge, CursorIcon)> {
    const BORDER: f64 = 5.0;
    let w = width as f64;
    let h = height as f64;
    let left = x < BORDER;
    let right = x >= (w - BORDER).max(0.0);
    let top = y < BORDER;
    let bottom = y >= (h - BORDER).max(0.0);

    match (left, right, top, bottom) {
        (true, _, true, _) => Some((ResizeEdge::TopLeft, CursorIcon::NwResize)),
        (_, true, true, _) => Some((ResizeEdge::TopRight, CursorIcon::NeResize)),
        (true, _, _, true) => Some((ResizeEdge::BottomLeft, CursorIcon::SwResize)),
        (_, true, _, true) => Some((ResizeEdge::BottomRight, CursorIcon::SeResize)),
        (true, _, _, _) => Some((ResizeEdge::Left, CursorIcon::WResize)),
        (_, true, _, _) => Some((ResizeEdge::Right, CursorIcon::EResize)),
        (_, _, true, _) => Some((ResizeEdge::Top, CursorIcon::NResize)),
        (_, _, _, true) => Some((ResizeEdge::Bottom, CursorIcon::SResize)),
        _ => None,
    }
}

/// The parent `child` can be given on the host compositor now (see
/// [`ClientApp::dialog_link_target`]), from its remote parent `parent` and `window`, which
/// gives a shown window's `(mapped, parent on the host)`.
fn dialog_link_target(
    child: WindowId,
    parent: Option<WindowId>,
    window: impl Fn(WindowId) -> Option<(bool, Option<WindowId>)>,
) -> Option<WindowId> {
    let parent = parent?;
    if !window(parent).is_some_and(|(mapped, _)| mapped) {
        return None;
    }
    // Walk up from the parent through the host's links; reaching the child is a loop. Any
    // walk longer than this has looped already.
    let mut cur = Some(parent);
    for _ in 0..1024 {
        match cur {
            None => return Some(parent),
            Some(id) if id == child => return None,
            Some(id) => cur = window(id).and_then(|(_, host_parent)| host_parent),
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn draw_shm_surface(
    pool: &mut SlotPool,
    surface: &wl_surface::WlSurface,
    viewport: Option<&WpViewport>,
    width: u32,
    height: u32,
    logical_width: u32,
    logical_height: u32,
    bgra: &[u8],
    damage: &[(i32, i32, i32, i32)],
    format: wl_shm::Format,
) {
    if width == 0 || height == 0 {
        return;
    }
    let lw = logical_width.max(1);
    let lh = logical_height.max(1);
    let stride = (width * 4) as i32;
    let needed = (width * height * 4) as usize;
    if bgra.len() < needed {
        return;
    }
    if let Ok((buffer, canvas)) = pool.create_buffer(width as i32, height as i32, stride, format) {
        canvas[..needed].copy_from_slice(&bgra[..needed]);
        if let Some(vp) = viewport {
            surface.set_buffer_scale(1);
            vp.set_destination(lw as i32, lh as i32);
        } else if width.is_multiple_of(lw)
            && height.is_multiple_of(lh)
            && (width / lw) == (height / lh)
        {
            surface.set_buffer_scale((width / lw).max(1) as i32);
        } else {
            surface.set_buffer_scale(1);
        }
        if buffer.attach_to(surface).is_ok() {
            if damage.is_empty() {
                surface.damage_buffer(0, 0, width as i32, height as i32);
            } else if damage.len() <= 32 {
                for &(dx, dy, dw, dh) in damage {
                    surface.damage_buffer(dx, dy, dw, dh);
                }
            } else {
                // Coalesce many dirty tiles into a single tight bounding box.
                let mut min_x = i32::MAX;
                let mut min_y = i32::MAX;
                let mut max_x = 0i32;
                let mut max_y = 0i32;
                for &(dx, dy, dw, dh) in damage {
                    min_x = min_x.min(dx);
                    min_y = min_y.min(dy);
                    max_x = max_x.max(dx + dw);
                    max_y = max_y.max(dy + dh);
                }
                surface.damage_buffer(min_x, min_y, (max_x - min_x).max(1), (max_y - min_y).max(1));
            }
            surface.commit();
        }
    }
}

/// An output's scale including fractional scaling: its current mode size over its xdg-output
/// logical size. `wl_output.scale` alone is only the integer ceiling (2 at 150%).
fn output_scale(info: &OutputInfo) -> f64 {
    let mode = info.modes.iter().find(|m| m.current).map(|m| m.dimensions);
    let rotated = matches!(
        info.transform,
        wl_output::Transform::_90
            | wl_output::Transform::_270
            | wl_output::Transform::Flipped90
            | wl_output::Transform::Flipped270
    );
    fractional_output_scale(info.scale_factor, info.logical_size, mode, rotated)
}

fn fractional_output_scale(
    integer_scale: i32,
    logical_size: Option<(i32, i32)>,
    mode_size: Option<(i32, i32)>,
    rotated: bool,
) -> f64 {
    let integer = integer_scale.max(1) as f64;
    let (Some((logical_w, _)), Some((mode_w, mode_h))) = (logical_size, mode_size) else {
        return integer;
    };
    let physical_w = if rotated { mode_h } else { mode_w };
    if logical_w <= 0 || physical_w <= 0 {
        return integer;
    }
    // Quantize to the fractional-scale protocol's 1/120 steps.
    let scale = (physical_w as f64 / logical_w as f64 * 120.0).round() / 120.0;
    // wl_output.scale is the ratio rounded up. If they disagree (e.g. Mutter's physical layout
    // mode reports xdg-output sizes in physical pixels), trust the integer scale.
    if scale > integer - 1.0 && scale <= integer {
        scale.max(1.0)
    } else {
        integer
    }
}

impl CompositorHandler for ClientApp {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        let surf_id = surface.id().protocol_id();
        if let Some(&window_id) = self.surface_to_remote.get(&surf_id) {
            self.update_window_scale(window_id, new_factor.max(1) as f64, false);
        }
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for ClientApp {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.maybe_send_display_scale();
    }
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.maybe_send_display_scale();
    }
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.maybe_send_display_scale();
    }
}

impl WindowHandler for ClientApp {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, window: &Window) {
        let surf_id = window.wl_surface().id().protocol_id();
        if let Some(&window_id) = self.surface_to_remote.get(&surf_id) {
            let _ = self
                .ctrl_tx
                .send(ControlMessage::WindowClosed { window_id });
        }
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        window: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        let surf_id = window.wl_surface().id().protocol_id();
        let Some(&window_id) = self.surface_to_remote.get(&surf_id) else {
            return;
        };
        let Some(win) = self.windows.get_mut(&window_id) else {
            return;
        };

        let new_w = configure.new_size.0.map(|v| v.get()).unwrap_or(win.width);
        let new_h = configure.new_size.1.map(|v| v.get()).unwrap_or(win.height);
        let maximized = configure.is_maximized();
        let fullscreen = configure.is_fullscreen();
        let activated = configure.is_activated();
        let tiled = configure.is_tiled();

        let old_size = (win.width, win.height);
        let size_changed = new_w != win.width || new_h != win.height;
        let state_changed = size_changed
            || maximized != win.maximized
            || fullscreen != win.fullscreen
            || activated != win.activated
            || tiled != win.tiled;

        win.width = new_w;
        win.height = new_h;
        win.maximized = maximized;
        win.fullscreen = fullscreen;
        win.activated = activated;
        win.tiled = tiled;

        let first = !win.configured;
        win.configured = true;
        if !first && size_changed {
            win.pending_resize_prev = Some((old_size, Instant::now()));
        }

        if let Some(frame) = win.pending_frame.take() {
            let lw = frame.logical_width.max(1);
            let lh = frame.logical_height.max(1);
            win.apply_frame_rects(
                &self.compositor_state,
                lw,
                lh,
                frame.geometry,
                frame.input_bounds,
            );
            let surf = win.window.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                win.viewport.as_ref(),
                frame.width,
                frame.height,
                lw,
                lh,
                &frame.bgra,
                &frame.damage,
                wl_shm::Format::Argb8888,
            );
        } else if first {
            let blank = vec![0u8; (new_w * new_h * 4) as usize];
            let surf = win.window.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                win.viewport.as_ref(),
                new_w,
                new_h,
                new_w,
                new_h,
                &blank,
                &[],
                wl_shm::Format::Argb8888,
            );
        }

        let scale_factor = win.effective_scale;
        if state_changed || first {
            debug!(
                window_id,
                new_w,
                new_h,
                scale_factor,
                maximized,
                fullscreen,
                activated,
                tiled,
                "host configure -> WindowConfigured"
            );
            let _ = self.ctrl_tx.send(ControlMessage::WindowConfigured {
                window_id,
                width: new_w,
                height: new_h,
                scale_factor,
                maximized,
                fullscreen,
                activated,
                tiled,
            });
        }
        if first {
            // Mapped now: dialogs waiting for this window can be attached to it.
            self.apply_waiting_dialog_parents();
        }
    }
}

impl PopupHandler for ClientApp {
    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        popup: &Popup,
        _config: PopupConfigure,
    ) {
        let surf_id = popup.wl_surface().id().protocol_id();
        let Some(&window_id) = self.surface_to_remote.get(&surf_id) else {
            return;
        };
        let Some(rp) = self.popups.get_mut(&window_id) else {
            return;
        };
        rp.configured = true;
        if let Some(frame) = rp.pending_frame.take() {
            apply_frame_rects(
                rp.popup.xdg_surface(),
                rp.popup.wl_surface(),
                &self.compositor_state,
                (frame.logical_width, frame.logical_height),
                frame.geometry,
                frame.input_bounds,
                &mut rp.input_bounds,
            );
            let surf = rp.popup.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                rp.viewport.as_ref(),
                frame.width,
                frame.height,
                frame.logical_width,
                frame.logical_height,
                &frame.bgra,
                &frame.damage,
                wl_shm::Format::Argb8888,
            );
        }
    }

    fn done(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, popup: &Popup) {
        let surf_id = popup.wl_surface().id().protocol_id();
        if let Some(window_id) = self.surface_to_remote.remove(&surf_id) {
            self.popups.remove(&window_id);
            let _ = self
                .ctrl_tx
                .send(ControlMessage::WindowClosed { window_id });
        }
    }
}

impl SeatHandler for ClientApp {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        if self.seat.is_none() {
            self.seat = Some(seat);
        }
    }

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if self.seat.is_none() {
            self.seat = Some(seat.clone());
        }
        if self.data_device.is_none() {
            if let Some(mgr) = &self.data_device_manager_state {
                self.data_device = Some(mgr.get_data_device(qh, &seat));
            }
        }
        if self.primary_selection_device.is_none() {
            if let Some(mgr) = &self.primary_selection_manager_state {
                self.primary_selection_device = Some(mgr.get_selection_device(qh, &seat));
            }
        }
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            if let Ok(kbd) = self.seat_state.get_keyboard(qh, &seat, None) {
                self.keyboard = Some(kbd);
            }
        }
        if capability == Capability::Pointer && self.themed_pointer.is_none() {
            let cursor_surf = self.compositor_state.create_surface(qh);
            if let Ok(tp) = self.seat_state.get_pointer_with_theme::<_, ()>(
                qh,
                &seat,
                self.shm.wl_shm(),
                cursor_surf,
                ThemeSpec::System,
            ) {
                self.themed_pointer = Some(tp);
            }
        }
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard {
            if let Some(kbd) = self.keyboard.take() {
                kbd.release();
            }
        }
        if capability == Capability::Pointer {
            self.themed_pointer = None;
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for ClientApp {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        serial: u32,
        raw: &[u32],
        _: &[Keysym],
    ) {
        self.last_input_serial = serial;
        self.reassert_clipboard(serial);
        let surf_id = surface.id().protocol_id();
        if let Some(&window_id) = self.surface_to_remote.get(&surf_id) {
            self.focused_window_id = window_id;
            let _ = self.ctrl_tx.send(ControlMessage::KeyboardEnter {
                window_id,
                pressed_keys: raw.to_vec(),
            });
        }
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
    ) {
        let window_id = self
            .surface_to_remote
            .get(&surface.id().protocol_id())
            .copied()
            .unwrap_or(self.focused_window_id);
        let _ = self
            .ctrl_tx
            .send(ControlMessage::KeyboardLeave { window_id });
    }

    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        self.last_input_serial = serial;
        let _ = self.ctrl_tx.send(ControlMessage::KeyEvent {
            window_id: self.focused_window_id,
            keycode: event.raw_code,
            pressed: true,
        });
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        self.last_input_serial = serial;
        let _ = self.ctrl_tx.send(ControlMessage::KeyEvent {
            window_id: self.focused_window_id,
            keycode: event.raw_code,
            pressed: false,
        });
    }

    // Key repeat is generated by the remote compositor, not locally.
    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: RawModifiers,
        _: u32,
    ) {
    }
}

impl PointerHandler for ClientApp {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            let surf_id = event.surface.id().protocol_id();
            let Some(&window_id) = self.surface_to_remote.get(&surf_id) else {
                continue;
            };
            match event.kind {
                PointerEventKind::Enter { serial } => {
                    self.last_input_serial = serial;
                    self.last_pointer_enter_serial = serial;
                    self.pointer_window_id = window_id;
                    self.last_cursor = ActiveCursor::Named(CursorIcon::Default);
                    let resize_icon = self.windows.get(&window_id).and_then(|win| {
                        win.border_resize_edge(event.position.0, event.position.1)
                            .map(|(_, icon)| icon)
                    });
                    if let Some(icon) = resize_icon {
                        self.border_resize_active = true;
                        if let Some(tp) = &self.themed_pointer {
                            let _ = tp.set_cursor(&self.conn, icon);
                        }
                    } else {
                        self.border_resize_active = false;
                        self.apply_active_cursor();
                    }
                    let _ = self.ctrl_tx.send(ControlMessage::PointerMotion {
                        window_id,
                        x: event.position.0,
                        y: event.position.1,
                    });
                }
                PointerEventKind::Motion { .. } => {
                    self.pointer_window_id = window_id;
                    // With a button held (a drag in the remote app, which reports positions
                    // outside the window too) the edges are not for resizing.
                    let resize_icon = self
                        .windows
                        .get(&window_id)
                        .filter(|_| self.pressed_buttons.is_empty())
                        .and_then(|win| {
                            win.border_resize_edge(event.position.0, event.position.1)
                                .map(|(_, icon)| icon)
                        });
                    if let Some(icon) = resize_icon {
                        self.border_resize_active = true;
                        if let Some(tp) = &self.themed_pointer {
                            let _ = tp.set_cursor(&self.conn, icon);
                        }
                    } else if self.border_resize_active {
                        self.border_resize_active = false;
                        self.apply_active_cursor();
                    }
                    let _ = self.ctrl_tx.send(ControlMessage::PointerMotion {
                        window_id,
                        x: event.position.0,
                        y: event.position.1,
                    });
                }
                PointerEventKind::Press { button, serial, .. } => {
                    self.last_button_serial = serial;
                    self.last_input_serial = serial;
                    self.focused_window_id = window_id;
                    if button == BTN_LEFT {
                        if let (Some(seat), Some(win)) = (&self.seat, self.windows.get(&window_id))
                        {
                            if let Some((edge, _)) =
                                win.border_resize_edge(event.position.0, event.position.1)
                            {
                                win.window.resize(seat, serial, edge);
                                continue;
                            }
                        }
                    }
                    self.pressed_buttons.insert(button);
                    let _ = self.ctrl_tx.send(ControlMessage::PointerMotion {
                        window_id,
                        x: event.position.0,
                        y: event.position.1,
                    });
                    let _ = self.ctrl_tx.send(ControlMessage::PointerButton {
                        window_id,
                        button,
                        pressed: true,
                    });
                }
                PointerEventKind::Release { button, serial, .. } => {
                    self.last_input_serial = serial;
                    self.pressed_buttons.remove(&button);
                    let _ = self.ctrl_tx.send(ControlMessage::PointerButton {
                        window_id,
                        button,
                        pressed: false,
                    });
                }
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    ..
                } => {
                    let dx = if horizontal.absolute != 0.0 {
                        horizontal.absolute
                    } else {
                        horizontal.discrete as f64 * 15.0
                    };
                    let dy = if vertical.absolute != 0.0 {
                        vertical.absolute
                    } else {
                        vertical.discrete as f64 * 15.0
                    };
                    if dx != 0.0 || dy != 0.0 {
                        let _ =
                            self.ctrl_tx
                                .send(ControlMessage::PointerAxis { window_id, dx, dy });
                    }
                }
                PointerEventKind::Leave { .. } => {
                    self.border_resize_active = false;
                    self.last_cursor = ActiveCursor::Named(CursorIcon::Default);
                    if self.outgoing_drag.is_some() {
                        // Starting our drag took the pointer; the button stays held in the
                        // session's drag, which the data device events now move.
                        continue;
                    }
                    self.release_pressed_buttons(window_id);
                    let _ = self
                        .ctrl_tx
                        .send(ControlMessage::PointerLeave { window_id });
                }
            }
        }
    }
}

impl ShmHandler for ClientApp {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

const SELECTION_IO_TIMEOUT: Duration = Duration::from_secs(3);

/// Read up to `MAX_SELECTION_BYTES` of UTF-8 text from a selection pipe with a
/// bounded `poll(2)` deadline so a misbehaving peer cannot leak threads or RAM.
/// Text cut short by the limits is still returned.
fn read_selection_fd_bounded(fd: OwnedFd) -> Option<String> {
    let (buf, _complete) = oriel_proto::read_selection_fd(fd, SELECTION_IO_TIMEOUT);
    String::from_utf8(buf).ok().filter(|s| !s.is_empty())
}

/// Like [`read_selection_fd_bounded`], but `None` unless the owner sent everything: a
/// file list cut mid-path could name a different file.
fn read_selection_fd_complete(fd: OwnedFd) -> Option<String> {
    let (buf, complete) = oriel_proto::read_selection_fd(fd, SELECTION_IO_TIMEOUT);
    complete
        .then(|| String::from_utf8(buf).ok())
        .flatten()
        .filter(|s| !s.is_empty())
}

/// Read a file selection (`text/uri-list` or a FileTransfer portal key) and register its
/// readable files for transfer in `slot`. `None` when it holds no readable local files.
fn read_file_offer(
    mime: &str,
    fd: OwnedFd,
    slot: OfferSlot,
) -> Option<(u64, Vec<oriel_proto::xfer::FileEntry>)> {
    let Some(body) = read_selection_fd_complete(fd) else {
        debug!(mime, "file offer: no complete data from the source");
        return None;
    };
    let paths = if mime == PORTAL_FILE_TRANSFER_MIME {
        crate::xfer::retrieve_portal_files(&body)
    } else {
        parse_file_uri_list(&body).and_then(crate::xfer::visible_host_paths)
    };
    let Some(paths) = paths else {
        debug!(mime, body = %body.escape_debug(), "file offer: no usable local paths");
        return None;
    };
    let offer = crate::xfer::OWNER.offer(slot, &paths);
    if offer.is_none() {
        debug!(?paths, "file offer: none of the paths is readable");
    }
    offer
}

/// The format to accept a drag carrying files as, if `mimes` carries files.
fn file_offer_mime(mimes: &[String]) -> Option<&'static str> {
    let has = |want: &str| mimes.iter().any(|m| m == want);
    if has(URI_LIST_MIME) {
        Some(URI_LIST_MIME)
    } else if crate::pty::flatpak_app_id().is_some() && has(PORTAL_FILE_TRANSFER_MIME) {
        Some(PORTAL_FILE_TRANSFER_MIME)
    } else {
        None
    }
}

/// Where to read the files of a local selection or drag from.
///
/// The `text/uri-list` paths come first: outside Flatpak they are always readable, inside
/// only some are (see `xfer::visible_host_paths`). The FileTransfer portal key is asked for
/// only when they are not: the portal's paths are not always readable either (a desktop
/// session with its own document portal, such as Atrium, hands out paths under its own
/// mount point, which the sandbox does not have).
struct FileSource {
    uri_list: Option<OwnedFd>,
    /// The offer, to ask for the portal key from the reading thread (wayland-client
    /// objects can send requests from any thread).
    portal: Option<(WlDataOffer, Connection)>,
}

impl FileSource {
    /// Start reading the files of `offer` with `mimes`; `None` when it carries none.
    fn new(offer: &WlDataOffer, mimes: &[String], conn: &Connection) -> Option<Self> {
        let has = |want: &str| mimes.iter().any(|m| m == want);
        let uri_list = if has(URI_LIST_MIME) {
            smithay_client_toolkit::data_device_manager::data_offer::receive(
                offer,
                URI_LIST_MIME.to_string(),
            )
            .ok()
            .map(OwnedFd::from)
        } else {
            None
        };
        let portal = (crate::pty::flatpak_app_id().is_some() && has(PORTAL_FILE_TRANSFER_MIME))
            .then(|| (offer.clone(), conn.clone()));
        (uri_list.is_some() || portal.is_some()).then_some(Self { uri_list, portal })
    }

    /// Register the readable files for transfer in `slot` (blocking).
    /// The uri-list is used when all of its paths are readable; otherwise the portal is
    /// asked (it can give all of them), and if that fails too, the readable ones are used.
    fn read(self, slot: OfferSlot) -> Option<(u64, Vec<oriel_proto::xfer::FileEntry>)> {
        let mut readable_part = None;
        if let Some(listed) = self
            .uri_list
            .and_then(read_selection_fd_complete)
            .and_then(|body| parse_file_uri_list(&body))
        {
            let visible = crate::xfer::visible_host_paths(listed.clone()).unwrap_or_default();
            if visible.len() == listed.len() {
                if let Some(files) = crate::xfer::OWNER.offer(slot, &visible) {
                    return Some(files);
                }
            } else if !visible.is_empty() {
                debug!(?listed, "file offer: some paths are not readable here");
                readable_part = Some(visible);
            }
        }
        if let Some((offer, conn)) = self.portal {
            if let Ok(pipe) = smithay_client_toolkit::data_device_manager::data_offer::receive(
                &offer,
                PORTAL_FILE_TRANSFER_MIME.to_string(),
            ) {
                let _ = conn.flush();
                if let Some(files) = read_file_offer(PORTAL_FILE_TRANSFER_MIME, pipe.into(), slot) {
                    return Some(files);
                }
            }
        }
        crate::xfer::OWNER.offer(slot, &readable_part?)
    }
}

/// What happens to a local drag over a remote window after it entered, relayed in order by
/// the thread that reads its files (see `DataDeviceHandler::enter`).
enum IncomingDragEvent {
    Motion {
        x: f64,
        y: f64,
    },
    Leave,
    Drop,
    /// The remote app it was dropped on is done reading (see `ControlMessage::DragFinished`).
    Finished,
    /// Liveness probe; ignored.
    Ping,
}

/// How long a dropped local drag stays readable for the remote app it was dropped on,
/// when the server never says it finished.
const DROP_FINISH_TIMEOUT: Duration = Duration::from_secs(30);

/// A remote app's drag continued on the local desktop. Over our own windows the local drag
/// is relayed back as plain pointer input, so the session's drag follows it.
struct OutgoingDrag {
    source: DragSource,
    clip: ReceivedClipboard,
    fetch: crate::xfer::SelectionFetcher,
    /// The remote window the drag is over, if any.
    over: Option<WindowId>,
    /// Dropped on a remote window: the session's drag ended with that drop.
    dropped_on_remote: bool,
}

/// A local drag over one of our remote windows.
struct IncomingDrag {
    offer_id: u64,
    events: std::sync::mpsc::Sender<IncomingDragEvent>,
    dropped: bool,
}

/// A local drag that entered remote window `window_id` at `x`/`y`.
struct DragEntered {
    offer: DragOffer,
    conn: Connection,
    /// Our id for the drag's data, readable by the server while the drag lasts.
    offer_id: u64,
    /// The drag's types the server can fetch (when it carries no files).
    mimes: Vec<String>,
    /// Where its files are read from, if it carries any.
    files: Option<FileSource>,
    window_id: WindowId,
    x: f64,
    y: f64,
    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
}

/// Read the files of a drag that entered a remote window, then announce the drag and
/// relay `events` to the server until it leaves or drops. After a drop the offer stays
/// readable until the remote app finished with it, then it is finished here.
fn relay_incoming_drag(drag: DragEntered, events: std::sync::mpsc::Receiver<IncomingDragEvent>) {
    let DragEntered {
        offer,
        conn,
        offer_id,
        mut mimes,
        files,
        window_id,
        x,
        y,
        ctrl_tx,
    } = drag;
    let files = match files {
        Some(source) => {
            let files = source.read(OfferSlot::Drag);
            if files.is_some() {
                mimes.clear();
            } else {
                // A file list that is unreadable here.
                mimes.retain(|m| !oriel_proto::xfer::FILE_SELECTION_MIMES.contains(&m.as_str()));
            }
            files.map(|(offer_id, roots)| FileOffer { offer_id, roots })
        }
        None => None,
    };
    let announced = files.is_some() || !mimes.is_empty();
    debug!(
        window_id,
        ?mimes,
        files = files.as_ref().map(|f| f.roots.len()),
        "local drag entered"
    );
    if announced {
        let _ = ctrl_tx.send(ControlMessage::DragEnter {
            window_id,
            x,
            y,
            offer_id,
            mimes,
            files,
        });
    }
    // The server can report the drag over before it is dropped (it refused the drag).
    let mut finished_early = false;
    while let Ok(event) = events.recv() {
        match event {
            IncomingDragEvent::Motion { x, y } => {
                if announced {
                    let _ = ctrl_tx.send(ControlMessage::DragMotion { window_id, x, y });
                }
            }
            IncomingDragEvent::Leave => {
                if announced {
                    let _ = ctrl_tx.send(ControlMessage::DragLeave);
                }
                return;
            }
            IncomingDragEvent::Drop => {
                if announced && !finished_early {
                    let _ = ctrl_tx.send(ControlMessage::DragDrop);
                    // The remote app reads the data after the drop; finishing the offer
                    // would let the source drop it.
                    let deadline = Instant::now() + DROP_FINISH_TIMEOUT;
                    loop {
                        let rem = deadline.saturating_duration_since(Instant::now());
                        match events.recv_timeout(rem) {
                            Ok(IncomingDragEvent::Finished) | Err(_) => break,
                            Ok(_) => {}
                        }
                    }
                }
                offer.finish();
                let _ = conn.flush();
                return;
            }
            IncomingDragEvent::Finished => finished_early = true,
            IncomingDragEvent::Ping => {}
        }
    }
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

impl ClientApp {
    /// Continue a remote app's drag of files on the local desktop. Needs the button that
    /// started it still held: the local drag reuses that press's implicit grab.
    fn start_outgoing_drag(
        &mut self,
        window_id: WindowId,
        clip: ReceivedClipboard,
        fetch: crate::xfer::SelectionFetcher,
    ) {
        if !self.pressed_buttons.contains(&BTN_LEFT) {
            // Dropped (inside the session) before the file list arrived.
            info!(
                window_id,
                "a remote drag's offer came after its button was released"
            );
            return;
        }
        let origin = match (self.windows.get(&window_id), self.popups.get(&window_id)) {
            (Some(win), _) => win.window.wl_surface().clone(),
            (None, Some(popup)) => popup.popup.wl_surface().clone(),
            (None, None) => return,
        };
        self.end_outgoing_drag();
        let (Some(mgr), Some(data_device)) = (&self.data_device_manager_state, &self.data_device)
        else {
            return;
        };
        // Copy only: a move would ask the session app to delete what it dragged.
        let source = mgr.create_drag_and_drop_source(&self.qh, clip.mime_types(), DndAction::Copy);
        source.start_drag(data_device, &origin, None, self.last_button_serial);
        let _ = self.conn.flush();
        debug!(
            window_id,
            mimes = ?clip.mimes,
            files = clip.files.as_ref().map(Vec::len),
            "continuing a remote drag locally"
        );
        self.outgoing_drag = Some(OutgoingDrag {
            source,
            clip,
            fetch,
            over: Some(window_id),
            dropped_on_remote: false,
        });
    }

    /// The local drag continuing a remote one is over: unless it was dropped on a remote
    /// window (the session's drag already ended there), end the session's drag without a
    /// drop. The held button is released by then.
    fn end_outgoing_drag(&mut self) {
        let Some(drag) = self.outgoing_drag.take() else {
            return;
        };
        if !drag.dropped_on_remote {
            let _ = self.ctrl_tx.send(ControlMessage::RemoteDragEnd);
        }
        self.pressed_buttons.remove(&BTN_LEFT);
        // Dropping the source destroys it.
        drop(drag);
    }

    /// Set the server's clipboard again if the compositor never took it (see
    /// `clipboard_confirmed`), now that a window has focus (`serial` from its enter).
    fn reassert_clipboard(&mut self, serial: u32) {
        if self.clipboard_confirmed {
            return;
        }
        let (Some(mgr), Some(data_device), Some((clip, _))) = (
            &self.data_device_manager_state,
            &self.data_device,
            &self.clipboard,
        ) else {
            return;
        };
        let mut mimes = clip.mime_types();
        mimes.push(self.source_marker.clone());
        let source = mgr.create_copy_paste_source(&self.qh, mimes);
        source.set_selection(data_device, serial);
        self.copy_paste_source = Some(source);
    }

    /// Tell the relay thread of the incoming drag, if any, that it ended with `event`. A
    /// dropped drag instead waits for `ControlMessage::DragFinished` in `finishing_drag`.
    fn end_incoming_drag(&mut self, event: IncomingDragEvent) {
        if let Some(drag) = self.incoming_drag.take() {
            if drag.dropped {
                // Forget relays that already gave up (their receiver is gone).
                self.finishing_drags
                    .retain(|_, tx| tx.send(IncomingDragEvent::Ping).is_ok());
                self.finishing_drags.insert(drag.offer_id, drag.events);
            } else {
                let _ = drag.events.send(event);
            }
        }
    }
}

impl DataDeviceHandler for ClientApp {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        wl_data_device: &WlDataDevice,
        x: f64,
        y: f64,
        surface: &wl_surface::WlSurface,
    ) {
        // A drag that left another window without a leave event (not expected).
        self.end_incoming_drag(IncomingDragEvent::Leave);
        let Some(data_device) = &self.data_device else {
            return;
        };
        if data_device.inner() != wl_data_device {
            return;
        }
        let Some(offer) = data_device.data().drag_offer() else {
            return;
        };
        let Some(&window_id) = self.surface_to_remote.get(&surface.id().protocol_id()) else {
            offer.accept_mime_type(offer.serial, None);
            return;
        };
        if let Some(drag) = &mut self.outgoing_drag {
            // Our own drag of a remote app's files is back over a remote window: hand the
            // pointer back to the session's drag.
            drag.over = Some(window_id);
            offer.accept_mime_type(offer.serial, drag.clip.mime_types().into_iter().next());
            offer.set_actions(DndAction::Copy, DndAction::Copy);
            let _ = self
                .ctrl_tx
                .send(ControlMessage::PointerMotion { window_id, x, y });
            return;
        }
        let all_mimes = offer.with_mime_types(|m| m.to_vec());
        let file_mime = file_offer_mime(&all_mimes);
        let mimes = oriel_proto::selection::proxyable_mimes(&all_mimes);
        let Some(accept) = file_mime
            .map(str::to_string)
            .or_else(|| mimes.first().cloned())
        else {
            debug!(window_id, mimes = ?all_mimes, "refusing a local drag with nothing to relay");
            offer.accept_mime_type(offer.serial, None);
            offer.set_actions(DndAction::empty(), DndAction::empty());
            return;
        };
        let files = FileSource::new(offer.inner(), &all_mimes, &self.conn);
        // Accepted before the remote app is known to want it: the round trip to ask would
        // lag every motion. Always a copy: a move would make the source delete the data.
        offer.accept_mime_type(offer.serial, Some(accept));
        offer.set_actions(DndAction::Copy, DndAction::Copy);
        let _ = self.conn.flush();
        let offer_id = oriel_proto::xfer::random_u64();
        crate::xfer::register_live_source(
            OfferSlot::Drag,
            offer_id,
            offer.inner().clone(),
            self.conn.clone(),
        );
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        self.incoming_drag = Some(IncomingDrag {
            offer_id,
            events: events_tx,
            dropped: false,
        });
        let conn = self.conn.clone();
        let ctrl_tx = self.ctrl_tx.clone();
        let drag = DragEntered {
            offer,
            conn,
            offer_id,
            mimes,
            files,
            window_id,
            x,
            y,
            ctrl_tx,
        };
        std::thread::spawn(move || relay_incoming_drag(drag, events_rx));
    }

    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {
        if let Some(drag) = &mut self.outgoing_drag {
            if drag.over.take().is_some() && !drag.dropped_on_remote {
                let _ = self.ctrl_tx.send(ControlMessage::RemoteDragLeave);
            }
            return;
        }
        // After a drop (always followed by a leave) this keeps the drag readable until the
        // remote app finished with it.
        self.end_incoming_drag(IncomingDragEvent::Leave);
    }

    fn motion(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, x: f64, y: f64) {
        if let Some(drag) = &self.outgoing_drag {
            if let Some(window_id) = drag.over {
                let _ = self
                    .ctrl_tx
                    .send(ControlMessage::PointerMotion { window_id, x, y });
            }
            return;
        }
        if let Some(drag) = &self.incoming_drag {
            let _ = drag.events.send(IncomingDragEvent::Motion { x, y });
        }
    }

    fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, wl_data_device: &WlDataDevice) {
        let Some(data_device) = &self.data_device else {
            return;
        };
        if data_device.inner() != wl_data_device {
            return;
        }
        let Some(offer) = data_device.data().selection_offer() else {
            return;
        };
        if offer.with_mime_types(|mimes| mimes.contains(&self.source_marker)) {
            // The echo of our own source: the compositor took it.
            self.clipboard_confirmed = true;
            return;
        }
        if self.clipboard.is_some() && !self.clipboard_confirmed {
            // The compositor never took the server's clipboard (set while unfocused) and
            // holds another selection. Which is newer is unknown, so neither side
            // overwrites the other: stop offering the server's here, and keep this one
            // local. A copy made from now on is sent as usual.
            self.copy_paste_source = None;
            self.clipboard = None;
            self.clipboard_confirmed = true;
            return;
        }
        // Another app took the clipboard after ours.
        self.copy_paste_source = None;
        self.clipboard = None;
        let all_mimes = offer.with_mime_types(|mimes| mimes.to_vec());
        let text_mime = SUPPORTED_TEXT_MIMES
            .iter()
            .find(|&&want| all_mimes.iter().any(|m| m == want))
            .map(|&s| s.to_string());
        let file_source = FileSource::new(offer.inner(), &all_mimes, &self.conn);
        let had_files = file_source.is_some();
        let text_pipe = text_mime.and_then(|m| offer.receive(m).ok());
        let _ = self.conn.flush();
        // Every other type is read only when a server pastes it.
        let offer_id = oriel_proto::xfer::random_u64();
        crate::xfer::register_live_source(
            OfferSlot::Clipboard,
            offer_id,
            offer.inner().clone(),
            self.conn.clone(),
        );
        let ctrl_tx = self.ctrl_tx.clone();
        std::thread::spawn(move || {
            let files = file_source.and_then(|source| source.read(OfferSlot::Clipboard));
            let msg = match files {
                // Files are offered as files only: their text form (paths here) and
                // file-list types mean nothing to the server.
                Some((file_offer_id, roots)) => ControlMessage::ClipboardOffer {
                    offer_id,
                    mimes: Vec::new(),
                    text: None,
                    files: Some(FileOffer {
                        offer_id: file_offer_id,
                        roots,
                    }),
                },
                None => {
                    let text = text_pipe.and_then(|pipe| {
                        let (buf, complete) = oriel_proto::read_selection_fd_max(
                            OwnedFd::from(pipe),
                            SELECTION_IO_TIMEOUT,
                            oriel_proto::selection::INLINE_TEXT_MAX,
                        );
                        complete.then(|| String::from_utf8(buf).ok()).flatten()
                    });
                    let mut mimes = oriel_proto::selection::proxyable_mimes(&all_mimes);
                    if had_files {
                        // A file list that is unreadable here.
                        mimes.retain(|m| {
                            !oriel_proto::xfer::FILE_SELECTION_MIMES.contains(&m.as_str())
                        });
                    }
                    if text.is_none() && mimes.is_empty() {
                        return;
                    }
                    ControlMessage::ClipboardOffer {
                        offer_id,
                        mimes,
                        text,
                        files: None,
                    }
                }
            };
            if let ControlMessage::ClipboardOffer {
                mimes, text, files, ..
            } = &msg
            {
                debug!(
                    ?mimes,
                    text = text.as_ref().map(String::len),
                    files = files.as_ref().map(|f| f.roots.len()),
                    "offering the local clipboard to the server"
                );
            }
            let _ = ctrl_tx.send(msg);
        });
    }

    fn drop_performed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {
        if let Some(drag) = &mut self.outgoing_drag {
            // Dropped on a remote window: release the button there, and the session's drag
            // drops on whatever is under it.
            if let Some(window_id) = drag.over {
                drag.dropped_on_remote = true;
                let _ = self.ctrl_tx.send(ControlMessage::PointerButton {
                    window_id,
                    button: BTN_LEFT,
                    pressed: false,
                });
                if let Some(offer) = self
                    .data_device
                    .as_ref()
                    .and_then(|d| d.data().drag_offer())
                {
                    offer.finish();
                }
            }
            return;
        }
        if let Some(drag) = &mut self.incoming_drag {
            if !drag.dropped {
                drag.dropped = true;
                let _ = drag.events.send(IncomingDragEvent::Drop);
            }
        }
    }
}

impl DataOfferHandler for ClientApp {
    fn source_actions(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &mut DragOffer,
        _: DndAction,
    ) {
    }

    fn selected_action(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &mut DragOffer,
        _: DndAction,
    ) {
    }
}

impl DataSourceHandler for ClientApp {
    fn accept_mime(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlDataSource,
        _: Option<String>,
    ) {
    }

    fn send_request(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        source: &WlDataSource,
        mime: String,
        write_pipe: WritePipe,
    ) {
        if let Some(drag) = self
            .outgoing_drag
            .as_ref()
            .filter(|d| d.source.inner() == source)
        {
            match drag.clip.serve(&mime) {
                Serve::Bytes(body) => {
                    std::thread::spawn(move || {
                        write_selection_fd_timeout(OwnedFd::from(write_pipe), &body);
                    });
                }
                Serve::Fetch(owner_mime) => {
                    drag.fetch
                        .fetch(drag.clip.offer_id, owner_mime, OwnedFd::from(write_pipe));
                }
                Serve::PortalFiles(paths) => {
                    crate::xfer::write_portal_key(paths, OwnedFd::from(write_pipe));
                }
                Serve::Nothing => {}
            }
            return;
        }
        if !self
            .copy_paste_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            return;
        }
        let Some((clip, fetch)) = &self.clipboard else {
            return;
        };
        match clip.serve(&mime) {
            Serve::Bytes(body) => {
                std::thread::spawn(move || {
                    write_selection_fd_timeout(OwnedFd::from(write_pipe), &body);
                });
            }
            Serve::Fetch(owner_mime) => {
                fetch.fetch(clip.offer_id, owner_mime, OwnedFd::from(write_pipe));
            }
            Serve::PortalFiles(paths) => {
                crate::xfer::write_portal_key(paths, OwnedFd::from(write_pipe));
            }
            Serve::Nothing => {}
        }
    }

    fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
        if self
            .outgoing_drag
            .as_ref()
            .is_some_and(|d| d.source.inner() == source)
        {
            self.end_outgoing_drag();
            return;
        }
        if self
            .copy_paste_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            self.copy_paste_source = None;
            // A source the compositor rejected (set while unfocused; some compositors
            // cancel it) is set again on the next keyboard enter.
            if self.clipboard_confirmed {
                self.clipboard = None;
            }
        } else {
            source.destroy();
        }
    }

    fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
    fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
        if self
            .outgoing_drag
            .as_ref()
            .is_some_and(|d| d.source.inner() == source)
        {
            self.end_outgoing_drag();
        }
    }
    fn action(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource, _: DndAction) {}
}

impl PrimarySelectionDeviceHandler for ClientApp {
    fn selection(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        primary_selection_device: &ZwpPrimarySelectionDeviceV1,
    ) {
        // Ignore selection echo when our own PrimarySelectionSource is active.
        if self.primary_selection_source.is_some() {
            return;
        }
        let Some(dev) = &self.primary_selection_device else {
            return;
        };
        if dev.inner() != primary_selection_device {
            return;
        }
        let Some(offer) = dev.data().selection_offer() else {
            return;
        };
        let mime = offer.with_mime_types(|mimes| {
            SUPPORTED_TEXT_MIMES
                .iter()
                .find(|&&want| mimes.iter().any(|m| m == want))
                .map(|&s| s.to_string())
        });
        let Some(mime) = mime else {
            return;
        };
        let Ok(read_pipe) = offer.receive(mime) else {
            return;
        };
        let _ = self.conn.flush();
        let ctrl_tx = self.ctrl_tx.clone();
        std::thread::spawn(move || {
            if let Some(text) = read_selection_fd_bounded(OwnedFd::from(read_pipe)) {
                let _ = ctrl_tx.send(ControlMessage::PrimarySelectionText { text });
            }
        });
    }
}

impl PrimarySelectionSourceHandler for ClientApp {
    fn send_request(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
        mime: String,
        write_pipe: WritePipe,
    ) {
        if self
            .primary_selection_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
            && SUPPORTED_TEXT_MIMES.contains(&mime.as_str())
        {
            let text = self.last_primary.clone();
            std::thread::spawn(move || {
                write_selection_fd_timeout(OwnedFd::from(write_pipe), text.as_bytes());
            });
        }
    }

    fn cancelled(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
    ) {
        if self
            .primary_selection_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            self.primary_selection_source = None;
        } else {
            source.destroy();
        }
    }
}

impl Dispatch2<XdgToplevelIconManagerV1, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &XdgToplevelIconManagerV1,
        _event: xdg_toplevel_icon_manager_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<XdgWmDialogV1, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &XdgWmDialogV1,
        _event: xdg_wm_dialog_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<XdgDialogV1, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &XdgDialogV1,
        _event: xdg_dialog_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<XdgToplevelIconV1, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &XdgToplevelIconV1,
        _event: wayland_protocols::xdg::toplevel_icon::v1::client::xdg_toplevel_icon_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<WpFractionalScaleManagerV1, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &WpFractionalScaleManagerV1,
        _event: wp_fractional_scale_manager_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<WpFractionalScaleV1, ClientApp> for WindowId {
    fn event(
        &self,
        state: &mut ClientApp,
        _proxy: &WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            let raw_scale = (scale as f64 / 120.0).max(1.0);
            state.update_window_scale(*self, raw_scale, true);
        }
    }
}

impl Dispatch2<WpViewporter, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &WpViewporter,
        _event: wp_viewporter::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl Dispatch2<WpViewport, ClientApp> for () {
    fn event(
        &self,
        _state: &mut ClientApp,
        _proxy: &WpViewport,
        _event: wp_viewport::Event,
        _conn: &Connection,
        _qh: &QueueHandle<ClientApp>,
    ) {
    }
}

impl ProvidesRegistryState for ClientApp {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

delegate_dispatch2!(ClientApp);
delegate_registry!(ClientApp);

/// Create a `calloop` channel pair for sending `ClientUiEvent`s to the Wayland event loop.
pub fn create_ui_channel() -> (UiSender, UiReceiver) {
    let (tx, rx) = channel::channel();
    let latest_frames = Arc::new(Mutex::new(HashMap::new()));
    (
        UiSender {
            tx,
            latest_frames: latest_frames.clone(),
        },
        UiReceiver { rx, latest_frames },
    )
}

/// The latest decoded frame of each window, as a [`UiReceiver`] would present it.
#[derive(Clone)]
pub struct LatestFrames(LatestFrameMap);

impl LatestFrames {
    /// `(width, height, bgra)` of `window_id`'s latest frame, taking it.
    pub fn take(&self, window_id: WindowId) -> Option<(u32, u32, Arc<Vec<u8>>)> {
        let frame = self.0.lock().unwrap().remove(&window_id)?;
        Some((frame.width, frame.height, frame.bgra))
    }
}

impl UiReceiver {
    /// Deliver the events to a plain channel instead of a Wayland event loop, from a
    /// thread of their own, for running a client without a display (integration tests).
    pub fn into_headless(self) -> (std::sync::mpsc::Receiver<ClientUiEvent>, LatestFrames) {
        let (tx, rx) = std::sync::mpsc::channel();
        let frames = LatestFrames(Arc::clone(&self.latest_frames));
        let ui_rx = self.rx;
        std::thread::Builder::new()
            .name("oriel-ui-headless".into())
            .spawn(move || {
                let Ok(mut event_loop) = EventLoop::<bool>::try_new() else {
                    return;
                };
                let inserted = event_loop
                    .handle()
                    .insert_source(ui_rx, move |event, _, closed| match event {
                        channel::Event::Msg(event) => {
                            if tx.send(event).is_err() {
                                *closed = true;
                            }
                        }
                        channel::Event::Closed => *closed = true,
                    });
                if inserted.is_err() {
                    return;
                }
                let mut closed = false;
                while !closed {
                    if event_loop
                        .dispatch(Some(Duration::from_millis(100)), &mut closed)
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("failed to spawn the headless UI thread");
        (rx, frames)
    }
}

/// Run the native `smithay-client-toolkit` Wayland event loop on the current thread.
pub fn run_event_loop(
    ui_rx: UiReceiver,
    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    max_scale: RenderScaleCap,
) -> Result<()> {
    let UiReceiver {
        rx: ui_rx,
        latest_frames,
    } = ui_rx;
    let conn =
        Connection::connect_to_env().context("failed to connect to local Wayland display")?;
    let (globals, event_queue) =
        registry_queue_init(&conn).context("failed to initialize Wayland registry queue")?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<ClientApp> =
        EventLoop::try_new().context("failed to create calloop EventLoop")?;
    let loop_handle = event_loop.handle();

    WaylandSource::new(conn.clone(), event_queue)
        .insert(loop_handle.clone())
        .map_err(|e| anyhow::anyhow!("failed to insert WaylandSource: {e:?}"))?;

    loop_handle
        .insert_source(ui_rx, |event, _, app| {
            if let channel::Event::Msg(ui_ev) = event {
                app.handle_ui_event(ui_ev);
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert UI event channel: {e:?}"))?;

    let compositor_state =
        CompositorState::bind(&globals, &qh).context("wl_compositor not available")?;
    let xdg_shell = XdgShell::bind(&globals, &qh).context("xdg_wm_base not available")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm not available")?;
    let data_device_manager_state = DataDeviceManagerState::bind(&globals, &qh).ok();
    let primary_selection_manager_state = PrimarySelectionManagerState::bind(&globals, &qh).ok();
    let toplevel_icon_manager: Option<XdgToplevelIconManagerV1> = globals.bind(&qh, 1..=1, ()).ok();
    let dialog_manager: Option<XdgWmDialogV1> = globals.bind(&qh, 1..=1, ()).ok();
    let fractional_scale_manager: Option<WpFractionalScaleManagerV1> =
        globals.bind(&qh, 1..=1, ()).ok();
    let viewporter: Option<WpViewporter> = globals.bind(&qh, 1..=1, ()).ok();
    let pool = SlotPool::new(1920 * 1080 * 4, &shm).context("failed to create SlotPool")?;
    let icon_pool =
        SlotPool::new(128 * 128 * 4 * 4, &shm).context("failed to create icon SlotPool")?;

    let mut app = ClientApp {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor_state,
        xdg_shell,
        shm,
        data_device_manager_state,
        primary_selection_manager_state,
        toplevel_icon_manager,
        dialog_manager,
        fractional_scale_manager,
        viewporter,
        max_scale,
        last_sent_display_scale: 0.0,
        pool,
        icon_pool,
        app_icons: Vec::new(),
        decoded_icon_cache: HashMap::new(),
        conn,
        qh,
        seat: None,
        keyboard: None,
        themed_pointer: None,
        custom_cursor_surface: None,
        custom_cursor_viewport: None,
        data_device: None,
        copy_paste_source: None,
        primary_selection_device: None,
        primary_selection_source: None,
        focused_window_id: 1,
        pointer_window_id: 1,
        last_button_serial: 0,
        last_input_serial: 0,
        last_pointer_enter_serial: 0,
        pressed_buttons: HashSet::new(),
        last_cursor: ActiveCursor::Named(CursorIcon::Default),
        border_resize_active: false,
        windows: HashMap::new(),
        popups: HashMap::new(),
        surface_to_remote: HashMap::new(),
        latest_frames,
        ctrl_tx,
        clipboard: None,
        clipboard_confirmed: false,
        source_marker: format!(
            "{}{:016x}",
            oriel_proto::selection::SOURCE_MARKER_PREFIX,
            oriel_proto::xfer::random_u64()
        ),
        incoming_drag: None,
        finishing_drags: HashMap::new(),
        outgoing_drag: None,
        last_primary: String::new(),
        exit: false,
    };

    while !app.exit {
        event_loop.dispatch(Duration::from_millis(16), &mut app)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dialog_link_target` over windows given as `(id, mapped, parent on the host)`.
    fn link_target(
        windows: &[(WindowId, bool, Option<WindowId>)],
        child: WindowId,
        parent: Option<WindowId>,
    ) -> Option<WindowId> {
        dialog_link_target(child, parent, |id| {
            windows
                .iter()
                .find(|w| w.0 == id)
                .map(|&(_, mapped, host)| (mapped, host))
        })
    }

    #[test]
    fn dialog_links_to_a_mapped_parent() {
        let windows = [(1, true, None), (2, true, None)];
        assert_eq!(link_target(&windows, 2, Some(1)), Some(1));
        assert_eq!(link_target(&windows, 2, None), None);
    }

    #[test]
    fn dialog_waits_for_its_parent_to_map() {
        // Unmapped or not shown yet: xdg-shell would treat it as no parent.
        assert_eq!(
            link_target(&[(1, false, None), (2, true, None)], 2, Some(1)),
            None
        );
        assert_eq!(link_target(&[(2, true, None)], 2, Some(1)), None);
    }

    #[test]
    fn dialog_link_never_makes_a_loop() {
        // 1 is already a dialog of 2: 2 -> 1 would make a loop.
        let windows = [(1, true, Some(2)), (2, true, None)];
        assert_eq!(link_target(&windows, 2, Some(1)), None);
        // Nor through a grandchild: 3 -> 2 -> 1, then 1 -> 3.
        let windows = [(1, true, None), (2, true, Some(1)), (3, true, Some(2))];
        assert_eq!(link_target(&windows, 1, Some(3)), None);
        // A window can't be its own parent.
        assert_eq!(link_target(&[(1, true, None)], 1, Some(1)), None);
        // A sibling is fine: 3 -> 1 while 2 -> 1.
        let windows = [(1, true, None), (2, true, Some(1)), (3, true, None)];
        assert_eq!(link_target(&windows, 3, Some(1)), Some(1));
    }

    #[test]
    fn fractional_output_scale_uses_logical_size() {
        // 2560x1440 at 150%: wl_output reports 2, xdg-output 1707x960 logical.
        assert_eq!(
            fractional_output_scale(2, Some((1707, 960)), Some((2560, 1440)), false),
            1.5
        );
        // Rotated outputs compare the mode height against the logical width.
        assert_eq!(
            fractional_output_scale(2, Some((960, 1707)), Some((2560, 1440)), true),
            1.5
        );
        assert_eq!(
            fractional_output_scale(2, Some((1920, 1080)), Some((3840, 2160)), false),
            2.0
        );
        // Without xdg-output, fall back to the integer scale; never go below 1.
        assert_eq!(
            fractional_output_scale(2, None, Some((2560, 1440)), false),
            2.0
        );
        assert_eq!(
            fractional_output_scale(1, Some((2560, 1440)), Some((1920, 1080)), false),
            1.0
        );
        // xdg-output in physical pixels (ratio 1) contradicts wl_output.scale 2: keep 2.
        assert_eq!(
            fractional_output_scale(2, Some((2560, 1440)), Some((2560, 1440)), false),
            2.0
        );
    }

    #[test]
    fn resize_margin_means_input_past_the_window_rectangle() {
        let rect = |x, y, width, height| FrameRect {
            x,
            y,
            width,
            height,
        };
        let window = rect(25, 25, 700, 616);
        // GTK-style: input covers a resize margin inside the shadow.
        assert!(input_reaches_past(rect(13, 13, 724, 640), window));
        // Shadow drawn, but input limited to the window: no margin of its own.
        assert!(!input_reaches_past(window, window));
        // No shadow at all.
        let full = FrameRect::full(700, 616);
        assert!(!input_reaches_past(full, full));
        // Click-through.
        assert!(!input_reaches_past(FrameRect::NONE, window));
    }

    #[test]
    fn ui_sender_push_frame_coalesces_to_latest_without_duplicate_wakeups() {
        let (tx, rx) = create_ui_channel();
        tx.push_frame(
            7,
            4,
            4,
            2,
            2,
            FrameRect::full(2, 2),
            FrameRect::full(2, 2),
            Arc::new(vec![1; 64]),
            vec![(0, 0, 2, 2)],
        );
        tx.push_frame(
            7,
            4,
            4,
            2,
            2,
            FrameRect::full(2, 2),
            FrameRect::full(2, 2),
            Arc::new(vec![2; 64]),
            vec![(2, 2, 2, 2)],
        );
        tx.push_frame(
            7,
            8,
            8,
            4,
            4,
            FrameRect::full(4, 4),
            FrameRect::full(4, 4),
            Arc::new(vec![3; 256]),
            vec![(0, 0, 8, 8)],
        );

        // Only the newest frame should be stored in the 1-slot mailbox.
        let latest = rx.latest_frames.lock().unwrap().remove(&7).unwrap();
        assert_eq!(
            (
                latest.width,
                latest.height,
                latest.logical_width,
                latest.logical_height
            ),
            (8, 8, 4, 4)
        );
        assert_eq!(latest.bgra[0], 3);
        // Dimension change resets coalesced damage to full-frame (empty vec).
        assert!(latest.damage.is_empty());

        // Next push after removal re-arms the mailbox.
        tx.push_frame(
            7,
            2,
            2,
            1,
            1,
            FrameRect::full(1, 1),
            FrameRect::full(1, 1),
            Arc::new(vec![9; 16]),
            vec![(0, 0, 2, 2)],
        );
        let next = rx.latest_frames.lock().unwrap().remove(&7).unwrap();
        assert_eq!(
            (
                next.width,
                next.height,
                next.logical_width,
                next.logical_height
            ),
            (2, 2, 1, 1)
        );
        assert_eq!(next.bgra[0], 9);
        assert_eq!(next.damage, vec![(0, 0, 2, 2)]);
    }
}
