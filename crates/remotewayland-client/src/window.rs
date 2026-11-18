//! Native Wayland rootless multi-window & unbounded `xdg_popup` presenter
//! built on `smithay-client-toolkit` (`sctk`).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::OwnedFd;
use std::time::Duration;

use anyhow::{Context, Result};
use remotewayland_proto::{ControlMessage, WindowId};
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
    delegate_pointer, delegate_registry, delegate_seat, delegate_shm, delegate_xdg_popup,
    delegate_xdg_shell, delegate_xdg_window,
    output::{OutputHandler, OutputState},
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
    shm::{slot::SlotPool, Shm, ShmHandler},
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
    Connection, Proxy, QueueHandle, WEnum,
};
use wayland_protocols::xdg::shell::client::{
    xdg_positioner::{Anchor, ConstraintAdjustment, Gravity},
    xdg_toplevel::ResizeEdge,
};

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
    ClipboardText {
        text: String,
    },
    Disconnected,
}

/// Sender handle used by background tasks to wake and message the Wayland event loop.
pub type UiSender = CalloopSender<ClientUiEvent>;

struct RemoteWindow {
    window: Window,
    configured: bool,
    width: u32,
    height: u32,
    maximized: bool,
    fullscreen: bool,
    activated: bool,
    tiled: bool,
    pending_frame: Option<(u32, u32, Vec<u8>)>,
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
    pool: SlotPool,
    conn: Connection,
    qh: QueueHandle<Self>,

    seat: Option<wl_seat::WlSeat>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    themed_pointer: Option<ThemedPointer>,
    data_device: Option<DataDevice>,
    copy_paste_source: Option<CopyPasteSource>,
    focused_window_id: WindowId,
    pointer_window_id: WindowId,
    last_button_serial: u32,
    last_input_serial: u32,
    pressed_keys: HashSet<u32>,
    last_cursor: (bool, CursorIcon),
    border_resize_active: bool,

    windows: HashMap<WindowId, RemoteWindow>,
    popups: HashMap<WindowId, RemotePopup>,
    surface_to_remote: HashMap<u32, WindowId>,

    ctrl_tx: mpsc::UnboundedSender<ControlMessage>,
    last_clipboard: String,
    exit: bool,
}

impl ClientApp {
    fn handle_ui_event(&mut self, event: ClientUiEvent) {
        match event {
            ClientUiEvent::WindowCreated {
                window_id,
                title,
                app_id,
                width,
                height,
            } => {
                if self.windows.contains_key(&window_id) {
                    return;
                }
                let wl_surf = self.compositor_state.create_surface(&self.qh);
                let surf_id = wl_surf.id().protocol_id();
                let window = self.xdg_shell.create_window(
                    wl_surf,
                    WindowDecorations::ClientOnly,
                    &self.qh,
                );
                window.set_title(if title.is_empty() {
                    format!("RemoteWayland #{window_id}")
                } else {
                    title
                });
                if !app_id.is_empty() {
                    window.set_app_id(app_id);
                }
                window.commit();

                self.surface_to_remote.insert(surf_id, window_id);
                self.windows.insert(
                    window_id,
                    RemoteWindow {
                        window,
                        configured: false,
                        width: width.max(64),
                        height: height.max(64),
                        maximized: false,
                        fullscreen: false,
                        activated: true,
                        tiled: false,
                        pending_frame: None,
                    },
                );
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
                if let (Some(seat), Some(win)) = (&self.seat, self.windows.get(&window_id)) {
                    win.window.move_(seat, self.last_button_serial);
                }
            }
            ClientUiEvent::StartResize { window_id, edge } => {
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
                    self.last_cursor = (visible, icon);
                    if !self.border_resize_active {
                        if let Some(tp) = &self.themed_pointer {
                            if !visible {
                                let _ = tp.hide_cursor();
                            } else {
                                let _ = tp.set_cursor(&self.conn, icon);
                            }
                        }
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

        let state_changed = new_w != win.width
            || new_h != win.height
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

        if let Some((fw, fh, bgra)) = win.pending_frame.take() {
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
        _: &[u32],
        _: &[Keysym],
    ) {
        self.last_input_serial = serial;
        let surf_id = surface.id().protocol_id();
        if let Some(&window_id) = self.surface_to_remote.get(&surf_id) {
            self.focused_window_id = window_id;
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
        for keycode in self.pressed_keys.drain() {
            let _ = self.ctrl_tx.send(ControlMessage::KeyEvent {
                window_id,
                keycode,
                pressed: false,
            });
        }
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
        self.pressed_keys.insert(event.raw_code);
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
        self.pressed_keys.remove(&event.raw_code);
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
                    self.pointer_window_id = window_id;
                    self.last_cursor = (true, CursorIcon::Default);
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
                        if let Some(tp) = &self.themed_pointer {
                            let _ = tp.set_cursor(&self.conn, self.last_cursor.1);
                        }
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
                        if let Some(tp) = &self.themed_pointer {
                            if !self.last_cursor.0 {
                                let _ = tp.hide_cursor();
                            } else {
                                let _ = tp.set_cursor(&self.conn, self.last_cursor.1);
                            }
                        }
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
                    self.last_cursor = (true, CursorIcon::Default);
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

const SUPPORTED_TEXT_MIMES: &[&str] = &[
    "text/plain;charset=utf-8",
    "text/plain;charset=UTF-8",
    "UTF8_STRING",
    "text/plain",
    "STRING",
    "TEXT",
];

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
            let mut file = File::from(OwnedFd::from(read_pipe));
            let mut text = String::new();
            if file.read_to_string(&mut text).is_ok() && !text.is_empty() {
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
                let mut file = File::from(OwnedFd::from(write_pipe));
                let _ = file.write_all(text.as_bytes());
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
    let pool = SlotPool::new(1920 * 1080 * 4, &shm).context("failed to create SlotPool")?;

    let mut app = ClientApp {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor_state,
        xdg_shell,
        shm,
        data_device_manager_state,
        pool,
        conn,
        qh,
        seat: None,
        keyboard: None,
        themed_pointer: None,
        data_device: None,
        copy_paste_source: None,
        focused_window_id: 1,
        pointer_window_id: 1,
        last_button_serial: 0,
        last_input_serial: 0,
        pressed_keys: HashSet::new(),
        last_cursor: (true, CursorIcon::Default),
        border_resize_active: false,
        windows: HashMap::new(),
        popups: HashMap::new(),
        surface_to_remote: HashMap::new(),
        ctrl_tx,
        last_clipboard: String::new(),
        exit: false,
    };

    while !app.exit {
        event_loop.dispatch(Duration::from_millis(16), &mut app)?;
    }

    Ok(())
}
