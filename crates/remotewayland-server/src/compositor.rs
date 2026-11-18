//! Headless Smithay Wayland compositor for RemoteWayland.
//!
//! Runs on a dedicated `calloop` thread, accepts Wayland client connections on an
//! isolated `WAYLAND_DISPLAY` socket, tracks `xdg_toplevel` and `xdg_popup`
//! windows, extracts committed SHM frames, and forwards input/configure events.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::os::unix::io::OwnedFd;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::channel::{self, Channel, Sender as CalloopSender};
use calloop::generic::Generic;
use calloop::{EventLoop, Interest, Mode, PostAction};
use remotewayland_proto::WindowId;
use smithay::backend::input::KeyState;
use smithay::desktop::utils::send_frames_surface_tree;
use smithay::desktop::{PopupKind, PopupManager, Space, Window};
use smithay::input::keyboard::{FilterResult, XkbConfig};
use smithay::input::pointer::{AxisFrame, ButtonEvent, CursorImageStatus, MotionEvent};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as DecorationMode;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_output;
use smithay::reexports::wayland_server::protocol::wl_seat;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{backend::ObjectId, Client, Display, DisplayHandle, Resource};
use smithay::utils::{Logical, Point, Rectangle, Serial, SERIAL_COUNTER};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    get_parent, with_states, with_surface_tree_downward, CompositorClientState, CompositorHandler,
    CompositorState, SubsurfaceCachedState, TraversalAction,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::selection::data_device::{
    request_data_device_client_selection, set_data_device_focus, set_data_device_selection,
    ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
};
use smithay::wayland::selection::{SelectionHandler, SelectionSource, SelectionTarget};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, SurfaceCachedState, ToplevelSurface, XdgPopupSurfaceData,
    XdgShellHandler, XdgShellState, XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{with_buffer_contents, ShmHandler, ShmState};
use smithay::wayland::socket::ListeningSocketSource;
use smithay::{
    delegate_compositor, delegate_data_device, delegate_output, delegate_seat, delegate_shm,
    delegate_xdg_decoration, delegate_xdg_shell,
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
    ClipboardText {
        text: String,
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

struct TrackedWindow {
    window_id: WindowId,
    window: Window,
    last_title: String,
    last_app_id: String,
    last_width: u32,
    last_height: u32,
    geo_offset: (i32, i32),
    announced: bool,
    latest_bgra: Option<(u32, u32, Vec<u8>)>,
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

#[allow(dead_code)]
struct AppState {
    display_handle: DisplayHandle,
    compositor_state: CompositorState,
    shm_state: ShmState,
    xdg_shell_state: XdgShellState,
    xdg_decoration_state: XdgDecorationState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,
    output_manager_state: OutputManagerState,
    output: Output,
    seat: Seat<Self>,
    space: Space<Window>,
    popups: PopupManager,
    next_window_id: WindowId,
    focused_window_id: WindowId,
    clipboard_text: Option<String>,
    pending_clipboard_mime: Option<String>,
    start_time: Instant,
    /// Map from `WlSurface::id()` (`ObjectId`, unique across clients) -> WindowId (for toplevels & popups).
    surface_to_window: HashMap<ObjectId, WindowId>,
    /// Cached RGBA buffer per `WlSurface::id()` for surface-tree compositing.
    surface_buffers: HashMap<ObjectId, (u32, u32, Vec<u8>)>,
    windows: HashMap<WindowId, TrackedWindow>,
    popup_windows: HashMap<WindowId, TrackedPopup>,
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
}

impl AppState {
    fn alloc_window_id(&mut self) -> WindowId {
        let id = self.next_window_id;
        self.next_window_id += 1;
        id
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
                    let loc = sub_state.pending().location;
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
                }
            }
            CompositorCommand::CloseWindow { window_id } => {
                if let Some(tracked) = self.windows.get(&window_id) {
                    if let Some(toplevel) = tracked.window.toplevel() {
                        toplevel.send_close();
                    }
                } else if let Some(popup) = self.popup_windows.get(&window_id) {
                    popup.surface.send_popup_done();
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
                    if let (Some(keyboard), Some(tracked)) =
                        (self.seat.get_keyboard(), self.windows.get(&window_id))
                    {
                        if let Some(toplevel) = tracked.window.toplevel() {
                            let root_surface = toplevel.wl_surface().clone();
                            if keyboard.current_focus().as_ref() != Some(&root_surface) {
                                keyboard.set_focus(self, Some(root_surface.clone()), serial);
                            }
                            for (popup, _) in PopupManager::popups_for_surface(&root_surface) {
                                let _ = PopupManager::dismiss_popup(&root_surface, &popup);
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
                            if keyboard.current_focus().as_ref() != Some(toplevel.wl_surface()) {
                                keyboard.set_focus(
                                    self,
                                    Some(toplevel.wl_surface().clone()),
                                    serial,
                                );
                            }
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
                if self.clipboard_text.as_deref() == Some(text.as_str()) {
                    return;
                }
                self.clipboard_text = Some(text);
                set_data_device_selection(
                    &self.display_handle,
                    &self.seat,
                    vec![
                        "text/plain;charset=utf-8".to_string(),
                        "text/plain;charset=UTF-8".to_string(),
                        "UTF8_STRING".to_string(),
                        "text/plain".to_string(),
                        "STRING".to_string(),
                        "TEXT".to_string(),
                    ],
                    (),
                );
            }
            CompositorCommand::SyncWindows => {
                for tracked in self.windows.values() {
                    if tracked.announced {
                        let _ = self.event_tx.send(CompositorEvent::WindowCreated {
                            window_id: tracked.window_id,
                            title: tracked.last_title.clone(),
                            app_id: tracked.last_app_id.clone(),
                            width: tracked.last_width,
                            height: tracked.last_height,
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

        with_surface_tree_downward(
            root_surface,
            (-crop_x, -crop_y),
            |surf, states, (ox, oy)| {
                let (nx, ny) = if surf == root_surface {
                    (*ox, *oy)
                } else {
                    let mut sub_state = states.cached_state.get::<SubsurfaceCachedState>();
                    let loc = sub_state.pending().location;
                    (ox + loc.x, oy + loc.y)
                };
                if let Some((sw, sh, srgba)) = self.surface_buffers.get(&surf.id()) {
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
                title,
                app_id,
                width: win_w,
                height: win_h,
            });
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
                    let loc = sub_state.pending().location;
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
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        self.popups.commit(surface);

        let mut cur = surface.clone();
        let window_id = loop {
            let key = cur.id();
            if let Some(&wid) = self.surface_to_window.get(&key) {
                break wid;
            }
            match get_parent(&cur) {
                Some(parent) => cur = parent,
                None => return,
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

        if let Some(tracked) = self.windows.get_mut(&window_id) {
            tracked.window.on_commit();
            if let Some(toplevel) = tracked.window.toplevel() {
                let root_surface = toplevel.wl_surface().clone();
                self.recomposite_window(window_id, &root_surface);
            }
        } else if self.popup_windows.contains_key(&window_id) {
            self.recomposite_popup(window_id);
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
                announced: false,
                latest_bgra: None,
            },
        );
        self.focused_window_id = window_id;
        if let Some(keyboard) = self.seat.get_keyboard() {
            let serial = SERIAL_COUNTER.next_serial();
            keyboard.set_focus(self, Some(wl_surface), serial);
        }
        debug!(window_id, ?surf_id, "new xdg_toplevel registered");
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        let surf_id = surface.wl_surface().id();
        self.surface_buffers.remove(&surf_id);
        if let Some(window_id) = self.surface_to_window.remove(&surf_id) {
            if let Some(tracked) = self.windows.remove(&window_id) {
                self.space.unmap_elem(&tracked.window);
                if tracked.announced {
                    let _ = self
                        .event_tx
                        .send(CompositorEvent::WindowClosed { window_id });
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
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Maximized);
        });
        surface.send_pending_configure();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
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
        });
        let _ = self.popups.track_popup(PopupKind::Xdg(surface.clone()));
        let _ = surface.send_configure();

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
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) {
        let dh = &self.display_handle;
        let client = focused.and_then(|s| dh.get_client(s.id()).ok());
        set_data_device_focus(dh, seat, client);
    }
    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        let (visible, shape_name) = match image {
            CursorImageStatus::Hidden => (false, String::new()),
            CursorImageStatus::Named(icon) => (true, icon.name().to_string()),
            CursorImageStatus::Surface(_) => (true, "default".to_string()),
        };
        let _ = self.event_tx.send(CompositorEvent::CursorUpdate {
            window_id: self.focused_window_id,
            visible,
            shape_name,
        });
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
        if ty != SelectionTarget::Clipboard {
            return;
        }
        // Remote client now owns or cleared the selection, so invalidate cached host text.
        self.clipboard_text = None;
        if let Some(src) = source {
            let mime_types = src.mime_types();
            let preferred = [
                "text/plain;charset=utf-8",
                "text/plain;charset=UTF-8",
                "UTF8_STRING",
                "text/plain",
                "STRING",
                "TEXT",
            ]
            .into_iter()
            .find(|m| mime_types.iter().any(|t| t == m));
            if let Some(mime) = preferred {
                // Note: Smithay invokes `SelectionHandler::new_selection` BEFORE storing the new
                // `OfferReplySource::Client(source)` into `SeatData::set_clipboard_selection`.
                // Defer `request_data_device_client_selection` until right after dispatch finishes.
                self.pending_clipboard_mime = Some(mime.to_string());
            }
        }
    }

    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        _mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        _user_data: &Self::SelectionUserData,
    ) {
        if ty == SelectionTarget::Clipboard {
            if let Some(text) = self.clipboard_text.clone() {
                std::thread::spawn(move || {
                    let mut file = std::fs::File::from(fd);
                    let _ = file.write_all(text.as_bytes());
                });
            }
        }
    }
}

impl AppState {
    fn poll_pending_clipboard(&mut self) {
        let Some(mime) = self.pending_clipboard_mime.take() else {
            return;
        };
        let Ok((mut read_pipe, write_pipe)) = std::io::pipe() else {
            return;
        };
        let write_fd: OwnedFd = write_pipe.into();
        if request_data_device_client_selection(&self.seat, mime, write_fd).is_ok() {
            let _ = self.display_handle.flush_clients();
            let event_tx = self.event_tx.clone();
            std::thread::spawn(move || {
                let mut text = String::new();
                if read_pipe.read_to_string(&mut text).is_ok() && !text.is_empty() {
                    let _ = event_tx.send(CompositorEvent::ClipboardText { text });
                }
            });
        }
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

impl XdgDecorationHandler for AppState {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ClientSide);
        });
        toplevel.send_pending_configure();
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: DecorationMode) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(mode);
        });
        toplevel.send_pending_configure();

        let surf_id = toplevel.wl_surface().id();
        if let Some(&window_id) = self.surface_to_window.get(&surf_id) {
            let _ = self.event_tx.send(CompositorEvent::SetDecorationMode {
                window_id,
                server_side: mode == DecorationMode::ServerSide,
            });
        }
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ClientSide);
        });
        toplevel.send_pending_configure();
    }
}

delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_xdg_shell!(AppState);
delegate_xdg_decoration!(AppState);
delegate_seat!(AppState);
delegate_data_device!(AppState);
delegate_output!(AppState);

/// Start the headless Smithay Wayland compositor on a background OS thread
/// and optionally launch `command` inside the isolated `WAYLAND_DISPLAY`.
pub fn start_compositor(
    command: Vec<String>,
) -> Result<(CompositorHandle, mpsc::UnboundedReceiver<CompositorEvent>)> {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx): (CalloopSender<CompositorCommand>, Channel<CompositorCommand>) =
        channel::channel();

    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<OsString>>(1);

    std::thread::Builder::new()
        .name("rw-compositor".into())
        .spawn(move || {
            if let Err(err) = run_compositor_loop(event_tx, cmd_rx, ready_tx.clone(), command) {
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

fn run_compositor_loop(
    event_tx: mpsc::UnboundedSender<CompositorEvent>,
    cmd_rx: Channel<CompositorCommand>,
    ready_tx: std::sync::mpsc::SyncSender<Result<OsString>>,
    command: Vec<String>,
) -> Result<()> {
    let mut event_loop: EventLoop<AppState> =
        EventLoop::try_new().context("failed to create calloop EventLoop")?;
    let display: Display<AppState> =
        Display::new().context("failed to create Wayland Display")?;
    let dh = display.handle();

    let compositor_state = CompositorState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    let xdg_shell_state = XdgShellState::new::<AppState>(&dh);
    let xdg_decoration_state = XdgDecorationState::new::<AppState>(&dh);
    let mut seat_state = SeatState::new();
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let output_manager_state = OutputManagerState::new_with_xdg_output::<AppState>(&dh);

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

    let mut state = AppState {
        display_handle: dh.clone(),
        compositor_state,
        shm_state,
        xdg_shell_state,
        xdg_decoration_state,
        seat_state,
        data_device_state,
        output_manager_state,
        output,
        seat,
        space,
        popups: PopupManager::default(),
        next_window_id: 1,
        focused_window_id: 1,
        clipboard_text: None,
        pending_clipboard_mime: None,
        start_time: Instant::now(),
        surface_to_window: HashMap::new(),
        surface_buffers: HashMap::new(),
        windows: HashMap::new(),
        popup_windows: HashMap::new(),
        event_tx,
    };

    let socket_source =
        ListeningSocketSource::new_auto().context("failed to bind Wayland listening socket")?;
    let socket_name = socket_source.socket_name().to_os_string();
    info!(?socket_name, "headless Wayland compositor listening");

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
                state.poll_pending_clipboard();
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

    if !command.is_empty() {
        crate::audio::ensure_virtual_audio_devices();
        let mut proc = Command::new(&command[0]);
        proc.args(&command[1..])
            .env("WAYLAND_DISPLAY", &socket_name)
            .env("PULSE_SINK", crate::audio::VIRTUAL_SINK_NAME)
            .env(
                "PULSE_SOURCE",
                format!("{}.monitor", crate::audio::VIRTUAL_MIC_NAME),
            )
            .env("GDK_BACKEND", "wayland")
            .env("QT_QPA_PLATFORM", "wayland")
            .env("MOZ_ENABLE_WAYLAND", "1")
            .env("SDL_VIDEODRIVER", "wayland");
        match proc.spawn() {
            Ok(child) => info!(pid = child.id(), cmd = ?command, "launched target application"),
            Err(err) => error!(error = %err, cmd = ?command, "failed to launch target application"),
        }
    }

    loop {
        event_loop.dispatch(Duration::from_millis(16), &mut state)?;
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
        state.display_handle.flush_clients()?;
        state.popups.cleanup();
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
