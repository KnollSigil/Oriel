//! Native Wayland rootless multi-window & unbounded `xdg_popup` presenter
//! built on `smithay-client-toolkit` (`sctk`).

use std::collections::{HashMap, HashSet};
use std::os::unix::io::{AsRawFd, OwnedFd};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use remotewayland_proto::{
    ControlMessage, WindowId, MAX_CURSOR_DIM, MAX_SELECTION_BYTES, SUPPORTED_TEXT_MIMES,
};
use smithay_client_toolkit::reexports::calloop::channel::{self, Channel, Sender as CalloopSender};
use smithay_client_toolkit::reexports::calloop::EventLoop;
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    data_device_manager::{
        data_device::{DataDevice, DataDeviceHandler},
        data_offer::{DataOfferHandler, DragOffer},
        data_source::{CopyPasteSource, DataSourceHandler},
        DataDeviceManagerState, WritePipe,
    },
    delegate_compositor, delegate_data_device, delegate_keyboard, delegate_output,
    delegate_pointer, delegate_primary_selection, delegate_registry, delegate_seat, delegate_shm,
    delegate_xdg_popup, delegate_xdg_shell, delegate_xdg_window,
    output::{OutputHandler, OutputState},
    primary_selection::{
        device::{PrimarySelectionDevice, PrimarySelectionDeviceHandler},
        selection::{PrimarySelectionSource, PrimarySelectionSourceHandler},
        PrimarySelectionManagerState,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers},
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
use tracing::{debug, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{
        wl_data_device::WlDataDevice, wl_data_device_manager::DndAction,
        wl_data_source::WlDataSource, wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm,
        wl_surface,
    },
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
};
use wayland_protocols::wp::primary_selection::zv1::client::{
    zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
    zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1,
};
use wayland_protocols::xdg::shell::client::{
    xdg_positioner::{Anchor, ConstraintAdjustment, Gravity},
    xdg_toplevel::ResizeEdge,
};
use wayland_protocols::xdg::toplevel_icon::v1::client::{
    xdg_toplevel_icon_manager_v1::{self, XdgToplevelIconManagerV1},
    xdg_toplevel_icon_v1::XdgToplevelIconV1,
};

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
    WindowClosed {
        window_id: WindowId,
    },
    FrameDecoded {
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
    AppIconsUpdated {
        entries: Vec<AppIconEntry>,
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
        hotspot_x: i32,
        hotspot_y: i32,
        bgra: Vec<u8>,
    },
}

/// Sender handle used by background tasks to wake and message the Wayland event loop.
pub type UiSender = CalloopSender<ClientUiEvent>;

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

struct RemoteWindow {
    window: Window,
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
    pending_frame: Option<(u32, u32, Vec<u8>)>,
    pending_resize_prev: Option<((u32, u32), Instant)>,
}

struct RemotePopup {
    popup: Popup,
    configured: bool,
    pending_frame: Option<(u32, u32, Vec<u8>)>,
    next_token: u32,
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

    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    last_clipboard: String,
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
                hotspot_x,
                hotspot_y,
                bgra,
            } => {
                let w = *width;
                let h = *height;
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
                let stride = (w * 4) as i32;
                if let Ok((buffer, canvas)) = self.pool.create_buffer(
                    w as i32,
                    h as i32,
                    stride,
                    wl_shm::Format::Argb8888,
                ) {
                    canvas[..byte_len].copy_from_slice(&bgra[..byte_len]);
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
                let window = self.xdg_shell.create_window(
                    wl_surf,
                    WindowDecorations::RequestClient,
                    &self.qh,
                );
                window.set_title(if title.is_empty() {
                    format!("RemoteWayland #{window_id}")
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
                        self.surface_to_remote.insert(surf_id, window_id);
                        self.popups.insert(
                            window_id,
                            RemotePopup {
                                popup,
                                configured: false,
                                pending_frame: None,
                                next_token: 1,
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
                bgra,
            } => {
                if let Some(rp) = self.popups.get_mut(&window_id) {
                    if !rp.configured {
                        rp.pending_frame = Some((width, height, bgra));
                    } else {
                        let surf = rp.popup.wl_surface().clone();
                        draw_shm_surface(
                            &mut self.pool,
                            &surf,
                            width,
                            height,
                            &bgra,
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
                }
                if let Some(rp) = self.popups.remove(&window_id) {
                    self.surface_to_remote
                        .remove(&rp.popup.wl_surface().id().protocol_id());
                }
            }
            ClientUiEvent::FrameDecoded {
                window_id,
                width,
                height,
                bgra,
            } => {
                if let Some(win) = self.windows.get_mut(&window_id) {
                    if !win.configured {
                        win.pending_frame = Some((width, height, bgra));
                    } else {
                        if let Some((prev_size, since)) = win.pending_resize_prev {
                            if (width, height) == prev_size
                                && (width, height) != (win.width, win.height)
                                && since.elapsed() < Duration::from_millis(250)
                            {
                                return;
                            }
                            win.pending_resize_prev = None;
                        }
                        win.window
                            .xdg_surface()
                            .set_window_geometry(0, 0, width as i32, height as i32);
                        let surf = win.window.wl_surface().clone();
                        draw_shm_surface(
                            &mut self.pool,
                            &surf,
                            width,
                            height,
                            &bgra,
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
                    let icon = shape_name.parse::<CursorIcon>().unwrap_or(CursorIcon::Default);
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
                hotspot_x,
                hotspot_y,
                bgra,
            } => {
                if window_id == self.pointer_window_id || window_id == self.focused_window_id {
                    self.last_cursor = ActiveCursor::Bitmap {
                        width,
                        height,
                        hotspot_x,
                        hotspot_y,
                        bgra,
                    };
                    if !self.border_resize_active {
                        self.apply_active_cursor();
                    }
                }
            }
            ClientUiEvent::ClipboardText { text } => {
                if !text.is_empty()
                    && (self.copy_paste_source.is_none() || text != self.last_clipboard)
                {
                    self.last_clipboard = text;
                    if let (Some(mgr), Some(data_device)) =
                        (&self.data_device_manager_state, &self.data_device)
                    {
                        let source =
                            mgr.create_copy_paste_source(&self.qh, SUPPORTED_TEXT_MIMES.to_vec());
                        source.set_selection(data_device, self.last_input_serial);
                        self.copy_paste_source = Some(source);
                    }
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
            ClientUiEvent::Disconnected => {
                debug!("server disconnected; exiting Wayland client loop");
                self.exit = true;
            }
        }
        let _ = self.conn.flush();
    }
}

fn border_resize_edge(
    x: f64,
    y: f64,
    width: u32,
    height: u32,
) -> Option<(ResizeEdge, CursorIcon)> {
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

fn draw_shm_surface(
    pool: &mut SlotPool,
    surface: &wl_surface::WlSurface,
    width: u32,
    height: u32,
    bgra: &[u8],
    format: wl_shm::Format,
) {
    if width == 0 || height == 0 {
        return;
    }
    let stride = (width * 4) as i32;
    let needed = (width * height * 4) as usize;
    if bgra.len() < needed {
        return;
    }
    if let Ok((buffer, canvas)) = pool.create_buffer(width as i32, height as i32, stride, format) {
        canvas[..needed].copy_from_slice(&bgra[..needed]);
        if buffer.attach_to(surface).is_ok() {
            surface.damage_buffer(0, 0, width as i32, height as i32);
            surface.commit();
        }
    }
}

impl CompositorHandler for ClientApp {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
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
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
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

        if let Some((fw, fh, bgra)) = win.pending_frame.take() {
            win.window
                .xdg_surface()
                .set_window_geometry(0, 0, fw as i32, fh as i32);
            let surf = win.window.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                fw,
                fh,
                &bgra,
                wl_shm::Format::Argb8888,
            );
        } else if first {
            let blank = vec![0u8; (new_w * new_h * 4) as usize];
            let surf = win.window.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                new_w,
                new_h,
                &blank,
                wl_shm::Format::Argb8888,
            );
        }

        if state_changed {
            debug!(
                window_id,
                new_w,
                new_h,
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
                scale_factor: 1.0,
                maximized,
                fullscreen,
                activated,
                tiled,
            });
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
        if let Some((fw, fh, bgra)) = rp.pending_frame.take() {
            let surf = rp.popup.wl_surface().clone();
            draw_shm_surface(
                &mut self.pool,
                &surf,
                fw,
                fh,
                &bgra,
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
            if let Ok(tp) = self.seat_state.get_pointer_with_theme(
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
        let _ = self.ctrl_tx.send(ControlMessage::KeyboardLeave { window_id });
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

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
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
                        (!win.maximized && !win.fullscreen && !win.tiled)
                            .then(|| {
                                border_resize_edge(
                                    event.position.0,
                                    event.position.1,
                                    win.width,
                                    win.height,
                                )
                                .map(|(_, icon)| icon)
                            })
                            .flatten()
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
                    let resize_icon = self.windows.get(&window_id).and_then(|win| {
                        (!win.maximized && !win.fullscreen && !win.tiled)
                            .then(|| {
                                border_resize_edge(
                                    event.position.0,
                                    event.position.1,
                                    win.width,
                                    win.height,
                                )
                                .map(|(_, icon)| icon)
                            })
                            .flatten()
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
                            if !win.maximized && !win.fullscreen && !win.tiled {
                                if let Some((edge, _)) = border_resize_edge(
                                    event.position.0,
                                    event.position.1,
                                    win.width,
                                    win.height,
                                ) {
                                    win.window.resize(seat, serial, edge);
                                    continue;
                                }
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
                        let _ = self
                            .ctrl_tx
                            .send(ControlMessage::PointerAxis { window_id, dx, dy });
                    }
                }
                PointerEventKind::Leave { .. } => {
                    self.border_resize_active = false;
                    self.last_cursor = ActiveCursor::Named(CursorIcon::Default);
                    self.release_pressed_buttons(window_id);
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

impl DataDeviceHandler for ClientApp {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlDataDevice,
        _: f64,
        _: f64,
        _: &wl_surface::WlSurface,
    ) {
    }

    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}

    fn motion(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, _: f64, _: f64) {}

    fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, wl_data_device: &WlDataDevice) {
        // Ignore selection echo when our own CopyPasteSource is active.
        if self.copy_paste_source.is_some() {
            return;
        }
        let Some(data_device) = &self.data_device else {
            return;
        };
        if data_device.inner() != wl_data_device {
            return;
        }
        let Some(offer) = data_device.data().selection_offer() else {
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
                let _ = ctrl_tx.send(ControlMessage::ClipboardText { text });
            }
        });
    }

    fn drop_performed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
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
        if self
            .copy_paste_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
            && SUPPORTED_TEXT_MIMES.contains(&mime.as_str())
        {
            let text = self.last_clipboard.clone();
            std::thread::spawn(move || {
                write_selection_fd_timeout(OwnedFd::from(write_pipe), text.as_bytes());
            });
        }
    }

    fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
        if self
            .copy_paste_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            self.copy_paste_source = None;
        } else {
            source.destroy();
        }
    }

    fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
    fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
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

impl Dispatch<XdgToplevelIconManagerV1, ()> for ClientApp {
    fn event(
        _state: &mut Self,
        _proxy: &XdgToplevelIconManagerV1,
        _event: xdg_toplevel_icon_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<XdgToplevelIconV1, ()> for ClientApp {
    fn event(
        _state: &mut Self,
        _proxy: &XdgToplevelIconV1,
        _event: wayland_protocols::xdg::toplevel_icon::v1::client::xdg_toplevel_icon_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl ProvidesRegistryState for ClientApp {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

delegate_compositor!(ClientApp);
delegate_output!(ClientApp);
delegate_shm!(ClientApp);
delegate_seat!(ClientApp);
delegate_keyboard!(ClientApp);
delegate_pointer!(ClientApp);
delegate_data_device!(ClientApp);
delegate_primary_selection!(ClientApp);
delegate_xdg_shell!(ClientApp);
delegate_xdg_window!(ClientApp);
delegate_xdg_popup!(ClientApp);
delegate_registry!(ClientApp);

/// Create a `calloop` channel pair for sending `ClientUiEvent`s to the Wayland event loop.
pub fn create_ui_channel() -> (UiSender, Channel<ClientUiEvent>) {
    channel::channel()
}

/// Run the native `smithay-client-toolkit` Wayland event loop on the current thread.
pub fn run_event_loop(
    ui_rx: Channel<ClientUiEvent>,
    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
) -> Result<()> {
    let conn = Connection::connect_to_env().context("failed to connect to local Wayland display")?;
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
        ctrl_tx,
        last_clipboard: String::new(),
        last_primary: String::new(),
        exit: false,
    };

    while !app.exit {
        event_loop.dispatch(Duration::from_millis(16), &mut app)?;
    }

    Ok(())
}
