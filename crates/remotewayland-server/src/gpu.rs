//! GPU buffer import and zero-copy VRAM frame handoff for hardware-accelerated Wayland clients.
//!
//! Opens an offscreen GLES context on a DRM render node so the compositor can advertise
//! `zwp_linux_dmabuf_v1` (and `wp_linux_drm_syncobj_v1` where the kernel supports it). Clients
//! (Mesa EGL/Vulkan, Xwayland glamor, NVIDIA's EGL) then render on the GPU and hand over dmabufs.
//!
//! When `SmartTileGpuEngine` supports Vulkan `DMA-BUF` sharing on the same render node, client
//! dmabufs (resolving vendor modifiers, DCC/CCS compression, external YUV, and `wp_viewporter`
//! crops) are rendered directly into pooled Vulkan-exported linear `Dmabuf` textures in VRAM
//! and handed straight to `SmartTileEncoder::encode_gpu_texture` with **zero CPU pixel copies**.
//! Otherwise, falls back to asynchronous GLES PBO readback into CPU `BGRA`.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use anyhow::{anyhow, bail, Context, Result};
use remotewayland_proto::smart_tile::gpu::{engine_for_drm_dev, SmartTileGpuEngine};
use remotewayland_proto::smart_tile::vram::{ExportedDmabufTexture, GpuTextureFrame};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufFlags};
use smithay::backend::allocator::{Buffer as _, Format, Fourcc, Modifier};
use smithay::backend::drm::{DrmDeviceFd, DrmNode, NodeType};
use smithay::backend::egl::fence::EGLFence;
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::gles::{
    GlesFrame, GlesMapping, GlesRenderer, GlesTexProgram, GlesTexture, Uniform, UniformName,
    UniformType,
};
use smithay::backend::renderer::{
    Bind, Color32F, ExportMem, Frame, ImportDma, ImportMem, Offscreen, Renderer, Texture,
};
use smithay::utils::{Buffer as BufferCoord, DeviceFd, Physical, Rectangle, Size, Transform};
use smithay::wayland::drm_syncobj::supports_syncobj_eventfd;
use tracing::{debug, info, warn};

/// Readback textures kept for reuse; one per buffer size, so windows and subsurfaces of
/// different sizes don't reallocate on every frame.
const MAX_CACHED_TARGETS: usize = 8;

/// CPU surface frames kept uploaded for compositing.
const MAX_CACHED_UPLOADS: usize = 16;

/// Area-average (box) downscaling: each output pixel averages the source texels its footprint
/// (`ratio` texels across) covers, weighted by coverage. Texels are sampled at their centres, so
/// the sampler's filter doesn't matter. Footprints up to [`MAX_BOX_RATIO`] texels are supported.
const BOX_DOWNSCALE_SHADER: &str = r#"#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
uniform vec2 tex_size;
uniform vec2 ratio;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

void main() {
    vec2 center = v_coords * tex_size;
    vec2 lo = center - 0.5 * ratio;
    vec2 hi = center + 0.5 * ratio;
    vec2 base = floor(lo);
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int j = 0; j < 8; j++) {
        float ty = base.y + float(j);
        if (ty >= hi.y) break;
        float wy = min(ty + 1.0, hi.y) - max(ty, lo.y);
        for (int i = 0; i < 8; i++) {
            float tx = base.x + float(i);
            if (tx >= hi.x) break;
            float w = wy * (min(tx + 1.0, hi.x) - max(tx, lo.x));
            sum += texture2D(tex, (vec2(tx, ty) + 0.5) / tex_size) * w;
            total += w;
        }
    }
    vec4 color = sum / max(total, 1e-6);

#if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0) * alpha;
#else
    color = color * alpha;
#endif

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
"#;

/// The largest downscale ratio [`BOX_DOWNSCALE_SHADER`] handles: its loops take at most 8 taps.
pub const MAX_BOX_RATIO: f64 = 7.0;

/// Maximum idle pooled zero-copy Vulkan/GLES DMABUF slots retained across window sizes.
const MAX_IDLE_ZERO_COPY_SLOTS: usize = 16;

/// How the compositor picks the GPU that clients render on.
#[derive(Debug, Clone, Default)]
pub enum GpuMode {
    /// Use the first hardware render node EGL reports that initializes.
    #[default]
    Auto,
    /// Use the given DRM node (`/dev/dri/renderD128`, or a primary `/dev/dri/cardN` node).
    Node(PathBuf),
    /// Offer clients only `wl_shm` (software rendering).
    Disabled,
}

/// A client frame as tightly packed, premultiplied BGRA rows: `(width, height, bytes)`.
pub type BgraFrame = (u32, u32, Vec<u8>);

/// A single Vulkan-allocated linear `Dmabuf` texture shared with GLES (`Bind<Dmabuf>`).
struct ZeroCopySlot {
    exported: ExportedDmabufTexture,
    dmabuf: Dmabuf,
    rb_swizzle: bool,
}

#[derive(Default)]
struct ZeroCopyPoolInner {
    free_slots: Vec<ZeroCopySlot>,
}

struct CheckedOutSlot {
    slot: Option<ZeroCopySlot>,
    pool: Arc<Mutex<ZeroCopyPoolInner>>,
}

impl Drop for CheckedOutSlot {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            if let Ok(mut guard) = self.pool.lock() {
                if guard.free_slots.len() < MAX_IDLE_ZERO_COPY_SLOTS {
                    guard.free_slots.push(slot);
                }
            }
        }
    }
}

/// Reference-counted handle to a VRAM-resident window texture (recycled automatically on drop).
#[derive(Clone)]
pub struct GpuFrameHandle {
    inner: Arc<CheckedOutSlot>,
    pub crop_x: u32,
    pub crop_y: u32,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Debug for GpuFrameHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFrameHandle")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("crop_x", &self.crop_x)
            .field("crop_y", &self.crop_y)
            .finish()
    }
}

impl PartialEq for GpuFrameHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            && self.crop_x == other.crop_x
            && self.crop_y == other.crop_y
            && self.width == other.width
            && self.height == other.height
    }
}
impl Eq for GpuFrameHandle {}

impl GpuFrameHandle {
    pub fn as_texture_frame(&self) -> GpuTextureFrame {
        let slot = self.inner.slot.as_ref().expect("CheckedOutSlot live");
        GpuTextureFrame {
            engine: Arc::clone(&slot.exported.engine),
            texture: slot.exported.texture.clone(),
            view: slot.exported.view.clone(),
            crop_x: self.crop_x,
            crop_y: self.crop_y,
            width: self.width,
            height: self.height,
            rb_swizzle: slot.rb_swizzle,
            external_memory: true,
        }
    }

    pub fn read_to_cpu_bgra(&self) -> Option<Vec<u8>> {
        self.as_texture_frame().read_to_cpu_bgra()
    }
}

/// Window or surface pixels: either CPU-side `BGRA` (`wl_shm` / fallback) or VRAM-resident (`dmabuf`).
#[derive(Clone, Debug)]
pub enum FramePixels {
    Cpu(Arc<Vec<u8>>),
    Gpu(GpuFrameHandle),
}

impl PartialEq for FramePixels {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(a), Self::Cpu(b)) => Arc::ptr_eq(a, b) || a == b,
            (Self::Gpu(a), Self::Gpu(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for FramePixels {}

impl FramePixels {
    /// Materialize CPU `BGRA` bytes on demand (zero-cost clone for `Cpu`; GPU staging readback for
    /// cold paths like `Snapshot`, `PopupFrame`, cursors, `Lz4Delta`, or multi-subsurface trees).
    pub fn to_cpu_bgra(&self) -> Arc<Vec<u8>> {
        match self {
            Self::Cpu(b) => Arc::clone(b),
            Self::Gpu(g) => Arc::new(
                g.read_to_cpu_bgra()
                    .unwrap_or_else(|| vec![0u8; (g.width as usize) * (g.height as usize) * 4]),
            ),
        }
    }
}

/// A committed surface frame `(width, height, pixels)`.
pub type ImportedFrame = (u32, u32, FramePixels);

/// A surface frame `(pixels, width, height, destination)` for [`GpuImporter::composite`].
pub type CompositeLayer = (FramePixels, u32, u32, Rectangle<i32, Physical>);

/// A GPU copy or zero-copy VRAM blit that has been queued.
pub enum PendingReadback {
    ZeroCopy(GpuFrameHandle),
    Cpu {
        mapping: GlesMapping,
        /// The copy is RGBA and must be swizzled to BGRA.
        swizzle: bool,
    },
}

/// Result of starting a dmabuf import/blit.
pub enum Readback {
    /// The frame is ready synchronously.
    Ready(ImportedFrame),
    /// The GPU work is queued; call [`GpuImporter::finish_readback`] once `fence` is readable.
    Pending {
        readback: PendingReadback,
        fence: OwnedFd,
    },
    /// The GPU work is queued, but the driver can't export its fence as an fd.
    Polled {
        readback: PendingReadback,
        fence: EGLFence,
    },
}

/// How completion of queued GPU work can be observed.
enum QueuedFence {
    Native(OwnedFd),
    Polled(EGLFence),
}

struct ZeroCopyConfig {
    engine: Arc<SmartTileGpuEngine>,
    fourcc: Fourcc,
    bgra_format: bool,
    rb_swizzle: bool,
    render_transform: Transform,
    pool: Arc<Mutex<ZeroCopyPoolInner>>,
}

/// Offscreen GLES renderer that turns client dmabufs into VRAM `GpuFrameHandle`s (or CPU BGRA fallback).
pub struct GpuImporter {
    renderer: GlesRenderer,
    node: DrmNode,
    /// Device for importing explicit-sync timelines, when the kernel supports it.
    syncobj_device: Option<DrmDeviceFd>,
    /// Zero-copy Vulkan <-> GLES DMABUF pool when supported on this render node.
    zero_copy: Option<ZeroCopyConfig>,
    /// Reusable CPU-readback render targets keyed by size (fallback path).
    targets: HashMap<(i32, i32), GlesTexture>,
    /// Whether the driver reads back `GL_BGRA` directly; otherwise read RGBA and swizzle.
    bgra_readback: bool,
    /// Textures of CPU surface frames uploaded for [`GpuImporter::composite`], keyed by
    /// the (immutable) frame bytes they hold.
    uploads: Vec<(Weak<Vec<u8>>, GlesTexture)>,
    /// [`BOX_DOWNSCALE_SHADER`], compiled on first use; `Err` if it failed to compile.
    box_program: Option<std::result::Result<GlesTexProgram, ()>>,
}

impl GpuImporter {
    /// Open the render node selected by `mode`. Returns `Ok(None)` for [`GpuMode::Disabled`].
    pub fn new(mode: &GpuMode) -> Result<Option<Self>> {
        let wanted = match mode {
            GpuMode::Disabled => return Ok(None),
            GpuMode::Auto => None,
            GpuMode::Node(path) => Some(render_node_for(path)?),
        };

        // smithay panics rather than erroring if libEGL can't be loaded.
        if !libegl_available() {
            bail!("libEGL.so.1 is not installed");
        }

        let candidates: Vec<(EGLDevice, DrmNode)> = EGLDevice::enumerate()
            .context("EGL cannot enumerate GPU devices")?
            .filter(|dev| !dev.is_software())
            .filter_map(|dev| {
                let node = dev.try_get_render_node().ok().flatten()?;
                Some((dev, node))
            })
            .filter(|(_, node)| wanted.is_none_or(|w| w.dev_id() == node.dev_id()))
            .collect();

        if candidates.is_empty() {
            return Err(match &wanted {
                Some(w) => anyhow!("EGL reports no GPU for {}", node_path(w)),
                None => anyhow!("no hardware EGL device with a DRM render node found"),
            });
        }

        let mut errors = Vec::new();
        for (device, node) in candidates {
            match Self::open(device, node) {
                Ok(gpu) => return Ok(Some(gpu)),
                Err(err) => {
                    warn!(node = %node_path(&node), error = %format!("{err:#}"), "GPU unusable");
                    errors.push(format!("{}: {err:#}", node_path(&node)));
                }
            }
        }
        bail!("no usable GPU ({})", errors.join("; "))
    }

    fn open(device: EGLDevice, node: DrmNode) -> Result<Self> {
        // SAFETY: the display, context, and renderer are only used from the compositor thread.
        let display = unsafe { EGLDisplay::new(device) }.context("failed to open EGL display")?;
        let context = EGLContext::new(&display).context("failed to create EGL context")?;
        let renderer =
            unsafe { GlesRenderer::new(context) }.context("failed to create GLES renderer")?;

        let mut renderer = renderer;
        if renderer.dmabuf_formats().iter().next().is_none() {
            bail!("driver supports no dmabuf formats");
        }
        // Clients will hand this GPU every frame, so make sure frames can actually be read back
        // before advertising it; otherwise windows would stay blank instead of using wl_shm.
        let bgra_readback = match probe_readback(&mut renderer, true) {
            Ok(()) => true,
            Err(bgra_err) => {
                debug!(error = %format!("{bgra_err:#}"), "GL_BGRA readback unavailable");
                probe_readback(&mut renderer, false).context("cannot read rendered frames back")?;
                false
            }
        };

        let zero_copy = probe_zero_copy(&mut renderer, node);

        let syncobj_device = open_syncobj_device(&node);
        info!(
            node = %node_path(&node),
            explicit_sync = syncobj_device.is_some(),
            zero_copy_vram = zero_copy.is_some(),
            "GPU acceleration available to clients"
        );
        Ok(Self {
            renderer,
            node,
            syncobj_device,
            zero_copy,
            targets: HashMap::new(),
            bgra_readback,
            uploads: Vec::new(),
            box_program: None,
        })
    }

    /// The render node clients should allocate buffers on.
    pub fn node(&self) -> DrmNode {
        self.node
    }

    /// The dmabuf formats and modifiers clients may submit.
    pub fn formats(&self) -> Vec<Format> {
        self.renderer.dmabuf_formats().into_iter().collect()
    }

    /// Device for `wp_linux_drm_syncobj_v1`, if the kernel supports syncobj eventfds.
    pub fn syncobj_device(&self) -> Option<DrmDeviceFd> {
        self.syncobj_device.clone()
    }

    /// Check that a client dmabuf can be imported; called on `zwp_linux_buffer_params_v1.create`.
    pub fn can_import(&mut self, dmabuf: &Dmabuf) -> bool {
        match self.renderer.import_dmabuf(dmabuf, None) {
            Ok(_) => true,
            Err(err) => {
                debug!(error = ?err, format = ?dmabuf.format(), "rejecting client dmabuf");
                false
            }
        }
    }

    /// Queue a VRAM-to-VRAM blit (zero CPU copy) or fallback CPU readback of `dmabuf` without blocking.
    ///
    /// The client buffer must stay unreleased until the fence signals (`finish_readback`).
    pub fn start_readback(
        &mut self,
        dmabuf: &Dmabuf,
        crop: Option<Rectangle<f64, BufferCoord>>,
    ) -> Result<Readback> {
        let buffer_size = dmabuf.size();
        if buffer_size.w <= 0 || buffer_size.h <= 0 {
            bail!("empty dmabuf");
        }
        let src = crop.unwrap_or_else(|| Rectangle::from_size(buffer_size.to_f64()));
        let size: Size<i32, BufferCoord> = (
            (src.size.w.round() as i32).max(1),
            (src.size.h.round() as i32).max(1),
        )
            .into();
        let texture = self
            .renderer
            .import_dmabuf(dmabuf, None)
            .map_err(|e| anyhow!("dmabuf import failed: {e:?}"))?;

        if let Some(zc_res) = self.try_start_zero_copy(&texture, src, size) {
            return Ok(zc_res);
        }

        let key = (size.w, size.h);
        if !self.targets.contains_key(&key) && self.targets.len() >= MAX_CACHED_TARGETS {
            self.targets.clear();
        }
        let target = match self.targets.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let texture: GlesTexture = self
                    .renderer
                    .create_buffer(Fourcc::Abgr8888, size)
                    .map_err(|e| anyhow!("failed to allocate readback texture: {e:?}"))?;
                e.insert(texture)
            }
        };

        let out_size: Size<i32, Physical> = (size.w, size.h).into();
        let full = Rectangle::from_size(out_size);
        let mut fb = self
            .renderer
            .bind(target)
            .map_err(|e| anyhow!("failed to bind readback texture: {e:?}"))?;
        {
            // Drawing (rather than reading the client texture directly) resolves tiled and
            // compressed modifiers, external-only (YUV) formats, and y-inverted buffers.
            let mut frame = self
                .renderer
                .render(&mut fb, out_size, Transform::Normal)
                .map_err(|e| anyhow!("failed to start readback frame: {e:?}"))?;
            frame
                .clear(Color32F::TRANSPARENT, &[full])
                .map_err(|e| anyhow!("{e:?}"))?;
            Frame::render_texture_from_to(
                &mut frame,
                &texture,
                src,
                full,
                &[full],
                &[],
                Transform::Normal,
                1.0,
            )
            .map_err(|e| anyhow!("failed to draw client dmabuf: {e:?}"))?;
            let _ = frame
                .finish()
                .map_err(|e| anyhow!("failed to finish readback frame: {e:?}"))?;
        }

        let region = Rectangle::from_size(size);
        let mut readback = None;
        if self.bgra_readback {
            match self
                .renderer
                .copy_framebuffer(&fb, region, Fourcc::Argb8888)
            {
                Ok(mapping) => {
                    readback = Some(PendingReadback::Cpu {
                        mapping,
                        swizzle: false,
                    })
                }
                Err(err) => debug!(error = ?err, "GL_BGRA readback failed; reading RGBA instead"),
            }
        }
        let readback = match readback {
            Some(r) => r,
            None => PendingReadback::Cpu {
                mapping: self
                    .renderer
                    .copy_framebuffer(&fb, region, Fourcc::Abgr8888)
                    .map_err(|e| anyhow!("failed to read back frame: {e:?}"))?,
                swizzle: true,
            },
        };
        drop(fb);

        match self.queue_fence() {
            Some(QueuedFence::Native(fence)) => Ok(Readback::Pending { readback, fence }),
            Some(QueuedFence::Polled(fence)) => Ok(Readback::Polled { readback, fence }),
            None => Ok(Readback::Ready(self.finish_readback(readback)?)),
        }
    }

    fn try_start_zero_copy(
        &mut self,
        client_tex: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        size: Size<i32, BufferCoord>,
    ) -> Option<Readback> {
        let zc = self.zero_copy.as_ref()?;
        let w = size.w as u32;
        let h = size.h as u32;
        let mut slot = self.checkout_zero_copy_slot(w, h)?;

        let out_size: Size<i32, Physical> = (size.w, size.h).into();
        let full = Rectangle::from_size(out_size);
        let render_transform = zc.render_transform;
        let pool = Arc::clone(&zc.pool);

        let blit_ok = (|| -> Result<()> {
            let mut fb = self
                .renderer
                .bind(&mut slot.dmabuf)
                .map_err(|e| anyhow!("failed to bind zero-copy dmabuf: {e:?}"))?;
            let mut frame = self
                .renderer
                .render(&mut fb, out_size, render_transform)
                .map_err(|e| anyhow!("failed to start zero-copy frame: {e:?}"))?;
            frame
                .clear(Color32F::TRANSPARENT, &[full])
                .map_err(|e| anyhow!("{e:?}"))?;
            Frame::render_texture_from_to(
                &mut frame,
                client_tex,
                src,
                full,
                &[full],
                &[],
                Transform::Normal,
                1.0,
            )
            .map_err(|e| anyhow!("failed to draw client dmabuf into zero-copy slot: {e:?}"))?;
            let _ = frame
                .finish()
                .map_err(|e| anyhow!("failed to finish zero-copy frame: {e:?}"))?;
            Ok(())
        })();

        if let Err(err) = blit_ok {
            debug!(error = %format!("{err:#}"), "zero-copy dmabuf blit failed; falling back to CPU readback");
            return None;
        }

        let handle = GpuFrameHandle {
            inner: Arc::new(CheckedOutSlot {
                slot: Some(slot),
                pool,
            }),
            crop_x: 0,
            crop_y: 0,
            width: w,
            height: h,
        };
        let readback = PendingReadback::ZeroCopy(handle);

        match self.queue_fence() {
            Some(QueuedFence::Native(fence)) => Some(Readback::Pending { readback, fence }),
            Some(QueuedFence::Polled(fence)) => Some(Readback::Polled { readback, fence }),
            None => {
                // Ensure GL commands retired if the driver gave no fence at all.
                let _ = self.renderer.with_context(|gl| unsafe { gl.Finish() });
                self.finish_readback(readback).ok().map(Readback::Ready)
            }
        }
    }

    /// A free pooled zero-copy slot of `width`x`height`, or a newly allocated one.
    fn checkout_zero_copy_slot(&self, width: u32, height: u32) -> Option<ZeroCopySlot> {
        let zc = self.zero_copy.as_ref()?;
        let pooled = zc.pool.lock().ok().and_then(|mut guard| {
            let pos = guard
                .free_slots
                .iter()
                .position(|s| s.exported.width == width && s.exported.height == height)?;
            Some(guard.free_slots.swap_remove(pos))
        });
        match pooled {
            Some(slot) => Some(slot),
            None => allocate_zero_copy_slot(
                &zc.engine,
                self.node,
                width,
                height,
                zc.fourcc,
                zc.bgra_format,
                zc.rb_swizzle,
            ),
        }
    }

    /// Draw `layers` bottom to top, each scaled into its destination rectangle, onto a
    /// transparent `width`x`height` VRAM frame, so a window's surface tree is composited
    /// without reading any surface back to the CPU. CPU layers (`wl_shm`) are uploaded, once
    /// per buffer. Blocks until the GPU has finished.
    ///
    /// Returns `None` without zero-copy support or if any step fails; the caller then
    /// composites on the CPU instead.
    pub fn composite(
        &mut self,
        width: u32,
        height: u32,
        layers: &[CompositeLayer],
    ) -> Option<GpuFrameHandle> {
        self.draw_layers(width, height, layers, None)
    }

    /// Downscale `frame` to a `width`x`height` VRAM frame with an area-average (box) filter.
    /// Returns `None` if the GPU can't (no zero-copy support, a footprint wider than
    /// [`MAX_BOX_RATIO`], or a failed step).
    pub fn downscale(
        &mut self,
        frame: &GpuFrameHandle,
        width: u32,
        height: u32,
    ) -> Option<GpuFrameHandle> {
        let ratio_x = f64::from(frame.width) / f64::from(width);
        let ratio_y = f64::from(frame.height) / f64::from(height);
        if ratio_x > MAX_BOX_RATIO || ratio_y > MAX_BOX_RATIO {
            return None;
        }
        let program = self.box_program()?;
        let dst = Rectangle::from_size((width as i32, height as i32).into());
        let layer = (
            FramePixels::Gpu(frame.clone()),
            frame.width,
            frame.height,
            dst,
        );
        self.draw_layers(width, height, &[layer], Some(&program))
    }

    /// [`BOX_DOWNSCALE_SHADER`], compiled once.
    fn box_program(&mut self) -> Option<GlesTexProgram> {
        let program = self.box_program.get_or_insert_with(|| {
            let uniforms = [
                UniformName::new("tex_size", UniformType::_2f),
                UniformName::new("ratio", UniformType::_2f),
            ];
            self.renderer
                .compile_custom_texture_shader(BOX_DOWNSCALE_SHADER, &uniforms)
                .map_err(|err| warn!(error = ?err, "failed to compile the box downscale shader"))
        });
        program.as_ref().ok().cloned()
    }

    /// Draw `layers` onto a new `width`x`height` VRAM frame, with `program` (and its
    /// `tex_size`/`ratio` uniforms) instead of the default texture shader if given.
    fn draw_layers(
        &mut self,
        width: u32,
        height: u32,
        layers: &[CompositeLayer],
        program: Option<&GlesTexProgram>,
    ) -> Option<GpuFrameHandle> {
        let render_transform = self.zero_copy.as_ref()?.render_transform;
        let pool = Arc::clone(&self.zero_copy.as_ref()?.pool);
        let mut slot = self.checkout_zero_copy_slot(width, height)?;
        self.uploads.retain(|(bytes, _)| bytes.strong_count() > 0);

        let drawn = (|| -> Result<()> {
            let mut textures = Vec::with_capacity(layers.len());
            for (pixels, w, h, dst) in layers {
                let (texture, src) = match pixels {
                    FramePixels::Gpu(handle) => {
                        let source = handle.inner.slot.as_ref().expect("CheckedOutSlot live");
                        let texture = self
                            .renderer
                            .import_dmabuf(&source.dmabuf, None)
                            .map_err(|e| anyhow!("failed to import surface frame: {e:?}"))?;
                        let src = Rectangle::<i32, BufferCoord>::new(
                            (handle.crop_x as i32, handle.crop_y as i32).into(),
                            (handle.width as i32, handle.height as i32).into(),
                        );
                        (texture, src)
                    }
                    FramePixels::Cpu(bytes) => {
                        let size: Size<i32, BufferCoord> = (*w as i32, *h as i32).into();
                        (self.upload(bytes, size)?, Rectangle::from_size(size))
                    }
                };
                textures.push((texture, src.to_f64(), *dst));
            }

            let out_size: Size<i32, Physical> = (width as i32, height as i32).into();
            let full = Rectangle::from_size(out_size);
            let mut fb = self
                .renderer
                .bind(&mut slot.dmabuf)
                .map_err(|e| anyhow!("failed to bind composite target: {e:?}"))?;
            let mut frame = self
                .renderer
                .render(&mut fb, out_size, render_transform)
                .map_err(|e| anyhow!("failed to start composite frame: {e:?}"))?;
            frame
                .clear(Color32F::TRANSPARENT, &[full])
                .map_err(|e| anyhow!("{e:?}"))?;
            for (texture, src, dst) in &textures {
                // Damage is relative to `dst`.
                let damage = [Rectangle::from_size(dst.size)];
                let tex_size = texture.size();
                let uniforms = [
                    Uniform::new("tex_size", (tex_size.w as f32, tex_size.h as f32)),
                    Uniform::new(
                        "ratio",
                        (
                            (src.size.w / f64::from(dst.size.w)) as f32,
                            (src.size.h / f64::from(dst.size.h)) as f32,
                        ),
                    ),
                ];
                GlesFrame::render_texture_from_to(
                    &mut frame,
                    texture,
                    *src,
                    *dst,
                    &damage,
                    &[],
                    Transform::Normal,
                    1.0,
                    program,
                    if program.is_some() { &uniforms } else { &[] },
                )
                .map_err(|e| anyhow!("failed to draw surface frame: {e:?}"))?;
            }
            let _ = frame
                .finish()
                .map_err(|e| anyhow!("failed to finish composite frame: {e:?}"))?;
            Ok(())
        })();
        if let Err(err) = drawn {
            debug!(error = %format!("{err:#}"), "GPU composite failed; compositing on the CPU");
            return None;
        }
        // The encoder reads the slot through Vulkan, which doesn't wait for GL work.
        let _ = self.renderer.with_context(|gl| unsafe { gl.Finish() });

        Some(GpuFrameHandle {
            inner: Arc::new(CheckedOutSlot {
                slot: Some(slot),
                pool,
            }),
            crop_x: 0,
            crop_y: 0,
            width,
            height,
        })
    }

    /// A texture holding CPU `BGRA` bytes, reused while the same buffer stays current.
    fn upload(
        &mut self,
        bytes: &Arc<Vec<u8>>,
        size: Size<i32, BufferCoord>,
    ) -> Result<GlesTexture> {
        let weak = Arc::downgrade(bytes);
        if let Some((_, texture)) = self.uploads.iter().find(|(b, _)| b.ptr_eq(&weak)) {
            return Ok(texture.clone());
        }
        if bytes.len() != (size.w * size.h * 4) as usize {
            bail!("CPU layer has {} bytes for {size:?}", bytes.len());
        }
        let texture = self
            .renderer
            .import_memory(bytes, Fourcc::Argb8888, size, false)
            .map_err(|e| anyhow!("failed to upload surface frame: {e:?}"))?;
        if self.uploads.len() >= MAX_CACHED_UPLOADS {
            self.uploads.remove(0);
        }
        self.uploads.push((weak, texture.clone()));
        Ok(texture)
    }

    /// Complete a queued zero-copy VRAM blit or map a fallback CPU copy once its fence signals.
    pub fn finish_readback(&mut self, readback: PendingReadback) -> Result<ImportedFrame> {
        match readback {
            PendingReadback::ZeroCopy(handle) => {
                Ok((handle.width, handle.height, FramePixels::Gpu(handle)))
            }
            PendingReadback::Cpu { mapping, swizzle } => {
                let (w, h) = (mapping.width(), mapping.height());
                let mut bgra = self
                    .renderer
                    .map_texture(&mapping)
                    .map_err(|e| anyhow!("failed to map readback: {e:?}"))?
                    .to_vec();
                if swizzle {
                    for px in bgra.as_chunks_mut::<4>().0 {
                        px.swap(0, 2);
                    }
                }
                let expected = w as usize * h as usize * 4;
                if bgra.len() != expected {
                    bail!(
                        "readback returned {} bytes, expected {expected}",
                        bgra.len()
                    );
                }
                Ok((w, h, FramePixels::Cpu(Arc::new(bgra))))
            }
        }
    }

    /// Insert and flush a fence after the queued commands, exported as a sync file if the driver
    /// supports native fences.
    fn queue_fence(&mut self) -> Option<QueuedFence> {
        let display = self.renderer.egl_context().display().clone();
        let fence = self
            .renderer
            .with_context(|gl| {
                let fence = EGLFence::create(&display);
                // SAFETY: the context is current; flushing submits the fence for export.
                unsafe { gl.Flush() };
                fence
            })
            .ok()?
            .ok()?;
        if fence.is_native() {
            if let Ok(fd) = fence.export() {
                return Some(QueuedFence::Native(fd));
            }
        }
        Some(QueuedFence::Polled(fence))
    }
}

fn allocate_zero_copy_slot(
    engine: &Arc<SmartTileGpuEngine>,
    node: DrmNode,
    width: u32,
    height: u32,
    fourcc: Fourcc,
    bgra_format: bool,
    rb_swizzle: bool,
) -> Option<ZeroCopySlot> {
    let (exported, fd) =
        ExportedDmabufTexture::try_allocate(Arc::clone(engine), width, height, bgra_format)?;
    let mut builder = Dmabuf::builder(
        (width as i32, height as i32),
        fourcc,
        Modifier::Linear,
        DmabufFlags::empty(),
    );
    if !builder.add_plane(fd, 0, exported.offset, exported.stride) {
        return None;
    }
    builder.set_node(node);
    let dmabuf = builder.build()?;
    Some(ZeroCopySlot {
        exported,
        dmabuf,
        rb_swizzle,
    })
}

/// Verify end-to-end Vulkan DMABUF export -> GLES `Bind<Dmabuf>` render -> Vulkan `wgpu` compute
/// readback (including Y-orientation, channel order, and pooled second-frame cache invalidation).
#[allow(clippy::unnecessary_cast)]
fn probe_zero_copy(renderer: &mut GlesRenderer, node: DrmNode) -> Option<ZeroCopyConfig> {
    let dev_id = node.dev_id() as u64;
    let engine = engine_for_drm_dev(dev_id)?;
    if !engine.supports_dmabuf() {
        debug!("Vulkan adapter does not advertise DMA-BUF external memory; using CPU readback");
        return None;
    }

    let render_formats = renderer
        .egl_context()
        .display()
        .dmabuf_render_formats()
        .clone();
    let candidates = [(Fourcc::Argb8888, true), (Fourcc::Abgr8888, false)];

    for (fourcc, bgra_format) in candidates {
        let fmt = Format {
            code: fourcc,
            modifier: Modifier::Linear,
        };
        if !render_formats.contains(&fmt) {
            continue;
        }
        let Some(mut slot) =
            allocate_zero_copy_slot(&engine, node, 16, 16, fourcc, bgra_format, false)
        else {
            continue;
        };

        for render_transform in [Transform::Normal, Transform::Flipped180] {
            let draw_res = (|| -> Result<()> {
                let mut fb = renderer
                    .bind(&mut slot.dmabuf)
                    .map_err(|e| anyhow!("{e:?}"))?;
                let out_size: Size<i32, Physical> = (16, 16).into();
                let top_half = Rectangle::new((0, 0).into(), (16, 8).into());
                let bot_half = Rectangle::new((0, 8).into(), (16, 8).into());
                let mut frame = renderer
                    .render(&mut fb, out_size, render_transform)
                    .map_err(|e| anyhow!("{e:?}"))?;
                // Top half pure red (BGRA = [0, 0, 255, 255]), bottom half pure blue (BGRA = [255, 0, 0, 255])
                frame
                    .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[top_half])
                    .map_err(|e| anyhow!("{e:?}"))?;
                frame
                    .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[bot_half])
                    .map_err(|e| anyhow!("{e:?}"))?;
                let _ = frame.finish().map_err(|e| anyhow!("{e:?}"))?;
                Ok(())
            })();
            if draw_res.is_err() {
                continue;
            }
            let _ = renderer.with_context(|gl| unsafe { gl.Finish() });

            let mut matched_swizzle = None;
            for rb_swizzle in [false, true] {
                let tf = GpuTextureFrame {
                    engine: Arc::clone(&engine),
                    texture: slot.exported.texture.clone(),
                    view: slot.exported.view.clone(),
                    crop_x: 0,
                    crop_y: 0,
                    width: 16,
                    height: 16,
                    rb_swizzle,
                    external_memory: true,
                };
                let Some(bytes) = tf.read_to_cpu_bgra() else {
                    continue;
                };
                if bytes.len() == 16 * 16 * 4
                    && bytes[0..4] == [0, 0, 255, 255]
                    && bytes[(12 * 16) * 4..(12 * 16) * 4 + 4] == [255, 0, 0, 255]
                {
                    matched_swizzle = Some(rb_swizzle);
                    break;
                }
            }
            let Some(rb_swizzle) = matched_swizzle else {
                continue;
            };

            // Verify second-frame overwrite on the same pooled slot invalidates Vulkan caches cleanly.
            let second_ok = (|| -> Result<bool> {
                let mut fb = renderer
                    .bind(&mut slot.dmabuf)
                    .map_err(|e| anyhow!("{e:?}"))?;
                let out_size: Size<i32, Physical> = (16, 16).into();
                let full = Rectangle::from_size(out_size);
                let mut frame = renderer
                    .render(&mut fb, out_size, render_transform)
                    .map_err(|e| anyhow!("{e:?}"))?;
                frame
                    .clear(Color32F::new(0.0, 1.0, 0.0, 1.0), &[full])
                    .map_err(|e| anyhow!("{e:?}"))?;
                let _ = frame.finish().map_err(|e| anyhow!("{e:?}"))?;
                Ok(true)
            })()
            .unwrap_or(false);
            if !second_ok {
                continue;
            }
            let _ = renderer.with_context(|gl| unsafe { gl.Finish() });

            let tf2 = GpuTextureFrame {
                engine: Arc::clone(&engine),
                texture: slot.exported.texture.clone(),
                view: slot.exported.view.clone(),
                crop_x: 0,
                crop_y: 0,
                width: 16,
                height: 16,
                rb_swizzle,
                external_memory: true,
            };
            if let Some(bytes2) = tf2.read_to_cpu_bgra() {
                if bytes2.len() == 16 * 16 * 4 && bytes2[0..4] == [0, 255, 0, 255] {
                    info!(
                        adapter = %engine.adapter_name,
                        ?fourcc,
                        rb_swizzle,
                        ?render_transform,
                        "verified zero-copy DMABUF -> Vulkan SmartTile encoder pipeline"
                    );
                    return Some(ZeroCopyConfig {
                        engine,
                        fourcc,
                        bgra_format,
                        rb_swizzle,
                        render_transform,
                        pool: Arc::new(Mutex::new(ZeroCopyPoolInner::default())),
                    });
                }
            }
        }
    }

    debug!("zero-copy DMABUF -> Vulkan probe did not find a compatible linear render format; using CPU readback");
    None
}

/// Render a known color into a 1x1 texture and read it back, checking the bytes, as `GL_BGRA`
/// (`bgra`) or `GL_RGBA`.
fn probe_readback(renderer: &mut GlesRenderer, bgra: bool) -> Result<()> {
    let size: Size<i32, BufferCoord> = (1, 1).into();
    let mut texture: GlesTexture = renderer
        .create_buffer(Fourcc::Abgr8888, size)
        .map_err(|e| anyhow!("failed to allocate probe texture: {e:?}"))?;
    let mut fb = renderer
        .bind(&mut texture)
        .map_err(|e| anyhow!("failed to bind probe texture: {e:?}"))?;
    let out_size: Size<i32, Physical> = (1, 1).into();
    {
        let mut frame = renderer
            .render(&mut fb, out_size, Transform::Normal)
            .map_err(|e| anyhow!("failed to render probe frame: {e:?}"))?;
        frame
            .clear(
                Color32F::new(1.0, 0.0, 0.0, 1.0),
                &[Rectangle::from_size(out_size)],
            )
            .map_err(|e| anyhow!("failed to clear probe frame: {e:?}"))?;
        let _ = frame
            .finish()
            .map_err(|e| anyhow!("failed to finish probe frame: {e:?}"))?;
    }
    let format = if bgra {
        Fourcc::Argb8888
    } else {
        Fourcc::Abgr8888
    };
    let mapping = renderer
        .copy_framebuffer(&fb, Rectangle::from_size(size), format)
        .map_err(|e| anyhow!("failed to copy probe frame: {e:?}"))?;
    let bytes = renderer
        .map_texture(&mapping)
        .map_err(|e| anyhow!("failed to map probe frame: {e:?}"))?;
    let expected: [u8; 4] = if bgra {
        [0, 0, 255, 255]
    } else {
        [255, 0, 0, 255]
    };
    if bytes != expected {
        bail!("probe frame read back as {bytes:?}, expected {expected:?}");
    }
    Ok(())
}

/// Resolve a user-supplied DRM node path to its render node.
fn render_node_for(path: &std::path::Path) -> Result<DrmNode> {
    let node = DrmNode::from_path(path)
        .with_context(|| format!("{} is not a DRM device node", path.display()))?;
    if node.ty() == NodeType::Render {
        return Ok(node);
    }
    node.node_with_type(NodeType::Render)
        .ok_or_else(|| anyhow!("{} has no render node", path.display()))?
        .with_context(|| format!("failed to find the render node for {}", path.display()))
}

fn node_path(node: &DrmNode) -> String {
    node.dev_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("DRM device {}:{}", node.major(), node.minor()))
}

/// Whether the system EGL library can be loaded.
fn libegl_available() -> bool {
    // SAFETY: dlopen with a valid C string; the handle is intentionally kept open since smithay
    // loads the same library right after.
    let handle =
        unsafe { libc::dlopen(c"libEGL.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    !handle.is_null()
}

/// Open the render node for importing explicit-sync timelines, if syncobj eventfds work.
fn open_syncobj_device(node: &DrmNode) -> Option<DrmDeviceFd> {
    let path = node.dev_path()?;
    let file = OpenOptions::new().read(true).write(true).open(&path).ok()?;
    let device = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
    supports_syncobj_eventfd(&device).then_some(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::damage::FrameDamage;
    use crate::media::WindowVideoEncoder;
    use remotewayland_proto::smart_tile::SmartTileDecoder;
    use remotewayland_proto::VideoCodec;
    use smithay::backend::renderer::sync::Fence as _;
    use std::os::fd::AsRawFd;
    use std::time::Instant;

    fn wait_readback_ready(gpu: &mut GpuImporter, rb: Readback) -> Result<ImportedFrame> {
        match rb {
            Readback::Ready(f) => Ok(f),
            Readback::Pending { readback, fence } => {
                let mut pfd = libc::pollfd {
                    fd: fence.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let _ = unsafe { libc::poll(&mut pfd, 1, 2000) };
                gpu.finish_readback(readback)
            }
            Readback::Polled { readback, fence } => {
                let start = Instant::now();
                while !fence.is_signaled() && start.elapsed().as_millis() < 2000 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                gpu.finish_readback(readback)
            }
        }
    }

    #[test]
    fn composite_draws_scaled_layers_in_vram() {
        let Ok(Some(mut gpu)) = GpuImporter::new(&GpuMode::Auto) else {
            eprintln!("Skipping GPU composite test: no hardware EGL GPU available");
            return;
        };
        let Some(zc) = gpu.zero_copy.as_ref() else {
            eprintln!("Skipping GPU composite test: adapter does not support DMABUF sharing");
            return;
        };
        let (engine, fourcc, bgra) = (Arc::clone(&zc.engine), zc.fourcc, zc.bgra_format);
        let node = gpu.node();

        // A client surface filled with `fill`, with `corner` in its top-left quadrant.
        let mut client_frame = |w: u32, h: u32, fill: Color32F, corner: Color32F| {
            let mut slot = allocate_zero_copy_slot(&engine, node, w, h, fourcc, bgra, false)
                .expect("allocate client dmabuf");
            {
                let mut fb = gpu.renderer.bind(&mut slot.dmabuf).unwrap();
                let size: Size<i32, Physical> = (w as i32, h as i32).into();
                let mut frame = gpu
                    .renderer
                    .render(&mut fb, size, Transform::Normal)
                    .unwrap();
                frame.clear(fill, &[Rectangle::from_size(size)]).unwrap();
                let quadrant = Rectangle::from_size((w as i32 / 2, h as i32 / 2).into());
                frame.clear(corner, &[quadrant]).unwrap();
                let _ = frame.finish().unwrap();
                let _ = gpu.renderer.with_context(|gl| unsafe { gl.Finish() });
            }
            let rb = gpu.start_readback(&slot.dmabuf, None).unwrap();
            match wait_readback_ready(&mut gpu, rb).unwrap().2 {
                FramePixels::Gpu(handle) => (slot, handle),
                FramePixels::Cpu(_) => panic!("expected a zero-copy frame"),
            }
        };
        let red = Color32F::new(1.0, 0.0, 0.0, 1.0);
        let blue = Color32F::new(0.0, 0.0, 1.0, 1.0);
        let green = Color32F::new(0.0, 1.0, 0.0, 1.0);
        let white = Color32F::new(1.0, 1.0, 1.0, 1.0);
        let (_root_slot, root) = client_frame(64, 48, blue, red);
        let (_sub_slot, sub) = client_frame(8, 8, green, white);

        let rect = |x, y, w, h| Rectangle::new((x, y).into(), (w, h).into());
        // The subsurface also as a CPU (`wl_shm`) frame.
        let sub_cpu = FramePixels::Cpu(Arc::new(sub.read_to_cpu_bgra().unwrap()));
        let layers = [
            (FramePixels::Gpu(root), 64, 48, rect(0, 0, 64, 48)),
            // Upscaled 2x.
            (FramePixels::Gpu(sub), 8, 8, rect(24, 16, 16, 16)),
            // Hanging off the bottom-right corner.
            (sub_cpu, 8, 8, rect(52, 36, 16, 16)),
        ];
        let out = gpu.composite(64, 48, &layers).expect("GPU composite");
        let px = out.read_to_cpu_bgra().expect("read composite");
        let at = |x: usize, y: usize| <[u8; 4]>::try_from(&px[(y * 64 + x) * 4..][..4]).unwrap();
        const RED: [u8; 4] = [0, 0, 255, 255];
        const BLUE: [u8; 4] = [255, 0, 0, 255];
        const GREEN: [u8; 4] = [0, 255, 0, 255];
        const WHITE: [u8; 4] = [255, 255, 255, 255];
        assert_eq!(at(4, 4), RED, "root top-left quadrant");
        assert_eq!(at(60, 4), BLUE, "root outside its quadrant");
        assert_eq!(at(4, 44), BLUE, "root outside its quadrant");
        assert_eq!(at(26, 18), WHITE, "subsurface top-left quadrant, scaled");
        assert_eq!(at(37, 29), GREEN, "subsurface, scaled");
        assert_eq!(at(22, 30), BLUE, "left of the subsurface");
        assert_eq!(at(53, 37), WHITE, "clipped subsurface top-left quadrant");
        assert_eq!(at(63, 47), GREEN, "clipped subsurface");
    }

    #[test]
    fn downscale_box_filters_in_vram_like_the_cpu() {
        let Ok(Some(mut gpu)) = GpuImporter::new(&GpuMode::Auto) else {
            eprintln!("Skipping GPU downscale test: no hardware EGL GPU available");
            return;
        };
        if gpu.zero_copy.is_none() {
            eprintln!("Skipping GPU downscale test: adapter does not support DMABUF sharing");
            return;
        }
        // A 1px checkerboard (which nearest or bilinear sampling would alias) over a
        // premultiplied alpha gradient.
        let (sw, sh, dw, dh) = (40u32, 30u32, 24u32, 18u32);
        let src: Vec<u8> = (0..sh)
            .flat_map(|y| {
                (0..sw).flat_map(move |x| {
                    let a = (255 * x / (sw - 1)) as u8;
                    let c = if (x + y) % 2 == 0 { a } else { a / 4 };
                    [c, c / 2, c, a]
                })
            })
            .collect();
        let full = Rectangle::from_size((sw as i32, sh as i32).into());
        let layer = (FramePixels::Cpu(Arc::new(src.clone())), sw, sh, full);
        let uploaded = gpu.composite(sw, sh, &[layer]).expect("upload to VRAM");
        let scaled = gpu.downscale(&uploaded, dw, dh).expect("GPU downscale");
        assert_eq!((scaled.width, scaled.height), (dw, dh));
        let got = scaled.read_to_cpu_bgra().expect("read downscaled frame");

        let mut want = vec![0u8; (dw * dh * 4) as usize];
        crate::downscale::BoxDownscaler::default().downscale(&src, (sw, sh), &mut want, (dw, dh));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| g.abs_diff(*w))
            .max()
            .unwrap();
        assert!(
            worst <= 2,
            "GPU and CPU box filters differ by up to {worst}"
        );
    }

    #[test]
    fn zero_copy_dmabuf_to_vram_encoder_roundtrip_and_1080p_bench() {
        let Ok(Some(mut gpu)) = GpuImporter::new(&GpuMode::Auto) else {
            eprintln!("Skipping hardware DMABUF zero-copy test: no hardware EGL GPU available");
            return;
        };
        let Some(zc_engine) = gpu.zero_copy.as_ref().map(|z| Arc::clone(&z.engine)) else {
            eprintln!(
                "Skipping zero-copy DMABUF test: adapter does not support linear DMABUF sharing"
            );
            return;
        };
        let zc_fourcc = gpu.zero_copy.as_ref().unwrap().fourcc;
        let zc_bgra = gpu.zero_copy.as_ref().unwrap().bgra_format;

        let (w, h) = (1920u32, 1080u32);
        // Simulate a GPU-accelerated Wayland client rendering 1080p frames into its own Dmabuf!
        let mut client_slot =
            allocate_zero_copy_slot(&zc_engine, gpu.node(), w, h, zc_fourcc, zc_bgra, false)
                .expect("allocate client dmabuf");

        let render_client_scene = |renderer: &mut GlesRenderer, dmabuf: &mut Dmabuf, step: u32| {
            let mut fb = renderer.bind(dmabuf).expect("bind client dmabuf");
            let out_size: Size<i32, Physical> = (w as i32, h as i32).into();
            let full = Rectangle::from_size(out_size);
            let mut frame = renderer
                .render(&mut fb, out_size, Transform::Normal)
                .expect("render client dmabuf");
            frame
                .clear(Color32F::new(0.12, 0.14, 0.18, 1.0), &[full])
                .unwrap();
            // Draw moving colorful rectangles simulating animated GPU window content
            for i in 0..16u32 {
                let rx = (((i * 137 + step * 7) % (w - 240)) as i32).max(0);
                let ry = (((i * 97 + step * 5) % (h - 160)) as i32).max(0);
                let rect = Rectangle::new((rx, ry).into(), (220, 140).into());
                let r = ((i * 53 + step * 3) % 255) as f32 / 255.0;
                let g = ((i * 89 + step * 7) % 255) as f32 / 255.0;
                let b = ((i * 173 + step * 11) % 255) as f32 / 255.0;
                frame.clear(Color32F::new(r, g, b, 1.0), &[rect]).unwrap();
            }
            let _ = frame.finish().unwrap();
            let _ = renderer.with_context(|gl| unsafe { gl.Finish() });
        };

        let mut win_enc = WindowVideoEncoder::new(VideoCodec::SmartTile);
        let mut dec = SmartTileDecoder::new();

        // Frame 0: Keyframe
        render_client_scene(&mut gpu.renderer, &mut client_slot.dmabuf, 0);
        let rb0 = gpu.start_readback(&client_slot.dmabuf, None).unwrap();
        let (fw, fh, px0) = wait_readback_ready(&mut gpu, rb0).unwrap();
        assert_eq!((fw, fh), (w, h));
        assert!(
            matches!(px0, FramePixels::Gpu(_)),
            "Expected zero-copy FramePixels::Gpu from GpuImporter"
        );
        let expected_f0 = px0.to_cpu_bgra();
        let enc0 = win_enc
            .encode_frame(w, h, 1_000_000, &px0, &FrameDamage::Full, 75)
            .unwrap();
        assert!(enc0.is_keyframe);
        let dec0 = dec.decode(w, h, &enc0.data).unwrap();
        assert_eq!(
            dec0,
            expected_f0.as_slice(),
            "Zero-copy DMABUF keyframe mismatch"
        );
        drop(px0);

        // Benchmark 8 frames of 1080p full-window GPU DMABUF motion (zero CPU pixel copies!)
        let mut total_import_us = 0u128;
        let mut total_encode_us = 0u128;
        let mut last_expected = expected_f0;
        let mut latest_dec = dec0;
        let iters = 8u32;
        for step in 1..=iters {
            render_client_scene(&mut gpu.renderer, &mut client_slot.dmabuf, step);

            let t_imp = Instant::now();
            let rb = gpu.start_readback(&client_slot.dmabuf, None).unwrap();
            let (_, _, px) = wait_readback_ready(&mut gpu, rb).unwrap();
            total_import_us += t_imp.elapsed().as_micros();

            assert!(matches!(px, FramePixels::Gpu(_)));
            let t_enc = Instant::now();
            let encoded = win_enc
                .encode_frame(
                    w,
                    h,
                    (step as u64 + 1) * 16_666_667,
                    &px,
                    &FrameDamage::Full,
                    75,
                )
                .unwrap();
            total_encode_us += t_enc.elapsed().as_micros();

            latest_dec = dec.decode(w, h, &encoded.data).unwrap();
            if step == iters {
                last_expected = px.to_cpu_bgra();
            }
        }

        // Idle lossless refresh after motion -> converges 100% bit-identically in VRAM!
        while let Some(ref_frame) = win_enc.encode_lossless_refresh(w, h, 400_000_000) {
            latest_dec = dec.decode(w, h, &ref_frame.data).unwrap();
        }
        assert!(
            latest_dec.as_slice() == last_expected.as_slice(),
            "End-to-end DMABUF -> VRAM SmartTile -> Decoder lossless convergence after motion mismatch"
        );

        // Benchmark small UI updates (8x 1-tile cursor blinks on settled 1080p GPU terminal/browser window)
        let mut ui_imp_us = 0u128;
        let mut ui_enc_us = 0u128;
        let mut ui_bytes = 0usize;
        let mut px_ui_last = None;
        for b in 0..iters {
            {
                let mut fb = gpu.renderer.bind(&mut client_slot.dmabuf).unwrap();
                let out_size: Size<i32, Physical> = (w as i32, h as i32).into();
                let cursor_rect = Rectangle::new((24, 24).into(), (12, 24).into());
                let mut frame = gpu
                    .renderer
                    .render(&mut fb, out_size, Transform::Normal)
                    .unwrap();
                let c = if b % 2 == 0 {
                    Color32F::new(1.0, 1.0, 1.0, 1.0)
                } else {
                    Color32F::new(0.12, 0.14, 0.18, 1.0)
                };
                frame.clear(c, &[cursor_rect]).unwrap();
                let _ = frame.finish().unwrap();
                let _ = gpu.renderer.with_context(|gl| unsafe { gl.Finish() });
            }
            let t_ui0 = Instant::now();
            let rb_ui = gpu.start_readback(&client_slot.dmabuf, None).unwrap();
            let (_, _, px_ui) = wait_readback_ready(&mut gpu, rb_ui).unwrap();
            ui_imp_us += t_ui0.elapsed().as_micros();
            let t_ui_enc0 = Instant::now();
            let enc_ui = win_enc
                .encode_frame(
                    w,
                    h,
                    500_000_000 + (b as u64) * 300_000_000,
                    &px_ui,
                    &FrameDamage::Full,
                    75,
                )
                .unwrap();
            ui_enc_us += t_ui_enc0.elapsed().as_micros();
            assert_eq!(enc_ui.dirty_tiles, 1);
            ui_bytes = enc_ui.data.len();
            latest_dec = dec.decode(w, h, &enc_ui.data).unwrap();
            px_ui_last = Some(px_ui);
        }
        last_expected = px_ui_last.unwrap().to_cpu_bgra();
        // Rapid <220ms blinks enter lossy mode; idle refresh refines that 1 tile back to lossless!
        while let Some(ref_frame) = win_enc.encode_lossless_refresh(w, h, 900_000_000) {
            latest_dec = dec.decode(w, h, &ref_frame.data).unwrap();
        }
        assert!(
            latest_dec.as_slice() == last_expected.as_slice(),
            "End-to-end DMABUF -> VRAM SmartTile 1-tile UI update lossless convergence mismatch"
        );

        eprintln!(
            "\n[Zero-Copy 1080p DMABUF -> VRAM SmartTile Benchmark ({})]\n  Motion (1080p): import/blit = {:.2} ms, VRAM encode = {:.2} ms, total = {:.2} ms\n  Cursor/UI (1 tile): import/blit = {:.2} ms, VRAM encode = {:.2} ms, total = {:.2} ms ({} B)",
            zc_engine.adapter_name,
            (total_import_us as f64 / iters as f64) / 1000.0,
            (total_encode_us as f64 / iters as f64) / 1000.0,
            ((total_import_us + total_encode_us) as f64 / iters as f64) / 1000.0,
            (ui_imp_us as f64 / iters as f64) / 1000.0,
            (ui_enc_us as f64 / iters as f64) / 1000.0,
            ((ui_imp_us + ui_enc_us) as f64 / iters as f64) / 1000.0,
            ui_bytes,
        );
    }
}
