//! GPU buffer import for hardware-accelerated Wayland clients.
//!
//! Opens an offscreen GLES context on a DRM render node so the compositor can advertise
//! `zwp_linux_dmabuf_v1` (and `wp_linux_drm_syncobj_v1` where the kernel supports it). Clients
//! (Mesa EGL/Vulkan, Xwayland glamor, NVIDIA's EGL) then render on the GPU and hand over dmabufs,
//! which are drawn into a linear offscreen texture and read back as BGRA for the existing CPU
//! video encoder.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::fd::OwnedFd;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Format, Fourcc};
use smithay::backend::drm::{DrmDeviceFd, DrmNode, NodeType};
use smithay::backend::egl::fence::EGLFence;
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::gles::{GlesMapping, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{
    Bind, Color32F, ExportMem, Frame, ImportDma, Offscreen, Renderer, Texture,
};
use smithay::utils::{Buffer as BufferCoord, DeviceFd, Physical, Rectangle, Size, Transform};
use smithay::wayland::drm_syncobj::supports_syncobj_eventfd;
use tracing::{debug, info, warn};

/// Readback textures kept for reuse; one per buffer size, so windows and subsurfaces of
/// different sizes don't reallocate on every frame.
const MAX_CACHED_TARGETS: usize = 8;

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

/// A GPU copy that has been queued but not yet mapped.
pub struct PendingReadback {
    mapping: GlesMapping,
    /// The copy is RGBA and must be swizzled to BGRA.
    swizzle: bool,
}

/// Result of starting a readback.
pub enum Readback {
    /// The frame was read synchronously (the driver has no fences at all).
    Ready(BgraFrame),
    /// The copy is queued; call [`GpuImporter::finish_readback`] once `fence` is readable.
    Pending {
        readback: PendingReadback,
        fence: OwnedFd,
    },
    /// The copy is queued, but the driver can't export its fence as a file descriptor; call
    /// [`GpuImporter::finish_readback`] once [`Fence::is_signaled`] returns true.
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

/// Offscreen GLES renderer that turns client dmabufs into CPU-side BGRA frames.
pub struct GpuImporter {
    renderer: GlesRenderer,
    node: DrmNode,
    /// Device for importing explicit-sync timelines, when the kernel supports it.
    syncobj_device: Option<DrmDeviceFd>,
    /// Reusable render targets keyed by size.
    targets: HashMap<(i32, i32), GlesTexture>,
    /// Whether the driver reads back `GL_BGRA` directly; otherwise read RGBA and swizzle.
    bgra_readback: bool,
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

        let syncobj_device = open_syncobj_device(&node);
        info!(
            node = %node_path(&node),
            explicit_sync = syncobj_device.is_some(),
            "GPU acceleration available to clients"
        );
        Ok(Self {
            renderer,
            node,
            syncobj_device,
            targets: HashMap::new(),
            bgra_readback,
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

    /// Queue a copy of a dmabuf (or the `crop` region of it, in buffer pixels) into
    /// CPU-readable memory without waiting for the GPU.
    ///
    /// The client buffer must stay unreleased until the readback is finished.
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
            // The trait method, not `GlesFrame`'s inherent one that takes a custom shader.
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
            // GL orders the copy below after this draw, so the frame's own fence isn't needed.
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
                    readback = Some(PendingReadback {
                        mapping,
                        swizzle: false,
                    })
                }
                // The startup probe showed BGRA works, so this is likely transient; read this
                // frame as RGBA without giving up on BGRA for later ones.
                Err(err) => debug!(error = ?err, "GL_BGRA readback failed; reading RGBA instead"),
            }
        }
        let readback = match readback {
            Some(r) => r,
            None => PendingReadback {
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

    /// Map a queued copy. Blocks only if its fence hasn't signalled yet.
    pub fn finish_readback(&mut self, readback: PendingReadback) -> Result<BgraFrame> {
        let (w, h) = (readback.mapping.width(), readback.mapping.height());
        let mut bgra = self
            .renderer
            .map_texture(&readback.mapping)
            .map_err(|e| anyhow!("failed to map readback: {e:?}"))?
            .to_vec();
        if readback.swizzle {
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
        Ok((w, h, bgra))
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
