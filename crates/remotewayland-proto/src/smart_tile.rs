//! Pure-Rust adaptive `SmartTile` desktop & video codec (`VideoCodec::SmartTile`).
//!
//! Combines:
//! - Sub-region vertical & horizontal scroll detection (`CopyRectOp`) for near-zero-bandwidth
//!   lossless scrolling even inside windows with fixed toolbars and sidebars.
//! - 64x64 dirty-tile skipping so unchanged regions cost 0 bytes.
//! - Per-tile adaptive lossless (raw/XOR BGRA) vs. lossy 8x8 integer DCT on 4:2:0 YCbCr
//!   with both Intra (I-tile) and zero-MV Inter-frame residual (P-tile) prediction.
//! - Bit-exact encoder/decoder reference reconstruction (zero drift across x86_64/aarch64)
//!   and automatic lossless refinement on idle.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

#[path = "smart_tile_simd.rs"]
pub mod simd;

#[path = "smart_tile_gpu.rs"]
pub mod gpu;

#[path = "smart_tile_vram.rs"]
pub mod vram;

#[path = "smart_tile_rans.rs"]
pub mod rans;

#[cfg(test)]
thread_local! {
    static FORCE_SCALAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_DISABLE_GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[inline]
fn snapshot_test_overrides() -> (bool, bool) {
    (FORCE_SCALAR.get(), FORCE_DISABLE_GPU.get())
}

#[cfg(not(test))]
#[inline]
const fn snapshot_test_overrides() -> (bool, bool) {
    (false, false)
}

#[cfg(test)]
#[inline]
fn apply_test_overrides(overrides: (bool, bool)) {
    FORCE_SCALAR.set(overrides.0);
    FORCE_DISABLE_GPU.set(overrides.1);
}

#[cfg(not(test))]
#[inline]
const fn apply_test_overrides(_overrides: (bool, bool)) {}

#[inline]
fn use_simd() -> bool {
    #[cfg(test)]
    {
        !FORCE_SCALAR.get()
    }
    #[cfg(not(test))]
    {
        true
    }
}

#[inline]
fn use_gpu() -> bool {
    #[cfg(test)]
    {
        !FORCE_DISABLE_GPU.get() && !FORCE_SCALAR.get()
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Tile edge length in pixels (must be a multiple of 16 for 4:2:0 8x8 DCT blocks).
pub const TILE_SIZE: u32 = 64;

const PACKET_VERSION: u8 = 1;
const FLAG_KEYFRAME: u8 = 1 << 0;
/// Frame carries a dominant scroll motion vector `(mv_dx, mv_dy)` (`i16`, `i16`) after copy rects,
/// allowing tiles to use `MODE_MV_SKIP` or `MODE_FLAG_MV` against the previous reference frame.
const FLAG_FRAME_MV: u8 = 1 << 1;
/// At least one tile in this frame uses `MODE_FLAG_LOCAL_MV` against the previous reference frame,
/// so the decoder must snapshot `bgra` before decoding tiles.
const FLAG_PREV_REF: u8 = 1 << 2;

const MODE_LOSSLESS_RAW: u8 = 0;
const MODE_LOSSLESS_XOR: u8 = 1;
const MODE_DCT_INTRA: u8 = 2;
const MODE_DCT_INTER: u8 = 3;
const MODE_DCT_HALF_INTRA: u8 = 4;
const MODE_DCT_HALF_INTER: u8 = 5;
const MODE_DCT_QUARTER_INTRA: u8 = 6;
const MODE_DCT_QUARTER_INTER: u8 = 7;
/// Lossless signed modular residual (`cur - ref`) encoded with Golomb-Rice coding
/// so idle refinement reuses the lossy preview already on the client.
const MODE_LOSSLESS_RESIDUAL: u8 = 8;
/// Exact bit-for-bit copy of `(x0 - mv_dx, y0 - mv_dy, tw, th)` from the previous reference frame.
const MODE_MV_SKIP: u8 = 9;

/// Bit flag OR'd onto any `MODE_DCT_*INTER` tile mode to carry a per-tile 2D motion vector
/// `(dx: i8, dy: i8)` for panning video and moving objects.
const MODE_FLAG_LOCAL_MV: u8 = 1 << 5;
/// Bit flag OR'd onto any `MODE_DCT_*INTER` tile mode to predict from `(x0 - mv_dx, y0 - mv_dy)`
/// in the previous reference frame instead of `(x0, y0)`.
const MODE_FLAG_MV: u8 = 1 << 6;
/// Bit flag OR'd onto any `MODE_DCT_*` or `MODE_LOSSLESS_RESIDUAL` tile mode to indicate an alpha plane is included.
const MODE_FLAG_ALPHA: u8 = 1 << 7;

/// Window after a tile's last change during which a subsequent change increments its motion streak.
const MOTION_WINDOW: Duration = Duration::from_millis(220);

/// Number of consecutive motion updates before an opaque tile switches to lossy DCT.
const MOTION_STREAK_THRESHOLD: u8 = 2;

/// How long a lossy tile must stay unchanged before it is refined inside a regular frame
/// while other regions (e.g. a playing video) keep the window from going idle.
const SETTLED_REFINE_DELAY: Duration = Duration::from_millis(250);

/// Maximum settled lossy tiles refined per regular frame (bounded further by raw bytes).
const SETTLED_REFINE_MAX_TILES: usize = 6;

/// Raw (pre-LZ4) byte cap for settled refinement tiles appended to a single regular frame.
const SETTLED_REFINE_MAX_RAW_BYTES: usize = 48 * 1024;

/// Maximum number of lossy tiles refined losslessly per progressive refinement packet,
/// keeping Wi-Fi latency low and allowing immediate preemption by new motion frames.
pub const DEFAULT_REFRESH_CHUNK_TILES: usize = 48;

/// Target compressed byte budget per progressive lossless refinement packet.
/// Prevents high-entropy paused-video frames from producing 300–550 KB Wi-Fi bursts.
pub const DEFAULT_REFRESH_MAX_BYTES: usize = 48 * 1024;

/// Maximum supported framebuffer width/height in pixels for untrusted decoded frames.
pub const MAX_FRAME_DIM: u32 = 16_384;

/// Maximum allowed decoded BGRA framebuffer size in bytes (128 MiB).
pub const MAX_DECODED_FRAME_BYTES: usize = 128 * 1024 * 1024;

/// Validate decoded frame dimensions and return `width * height * 4` without overflow.
pub fn validate_frame_dims(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 || width > MAX_FRAME_DIM || height > MAX_FRAME_DIM {
        bail!("invalid frame dimensions: {width}x{height} (max {MAX_FRAME_DIM})");
    }
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .context("frame dimension byte-size overflow")?;
    if bytes > MAX_DECODED_FRAME_BYTES {
        bail!("decoded frame too large: {bytes} bytes (max {MAX_DECODED_FRAME_BYTES})");
    }
    Ok(bytes)
}

/// Compress a `SmartTile` packet using Zstd-style two-stage `LZ4 + 4-way interleaved rANS`
/// and emit whichever representation is strictly smallest.
/// Small lossless UI/cursor packets (`< 8 KB` LZ4) skip the rANS pass for sub-millisecond latency,
/// while lossy frames, cold keyframes, and idle lossless refresh chunks all benefit from rANS.
fn compress_packet_adaptive(raw: &[u8], used_lossy: bool) -> Vec<u8> {
    let lz4 = lz4_flex::compress_prepend_size(raw);
    let try_rans = if used_lossy {
        raw.len() >= 128
    } else {
        lz4.len() >= 8192
    };
    if try_rans && raw.len() < (rans::RANS_MAGIC_BIT as usize) {
        if let Some(lz4_rans) = rans::compress_lz4_body_with_rans(raw.len(), &lz4) {
            return lz4_rans;
        }
    }
    lz4
}

/// Decompress a size-prepended LZ4 or 4-way interleaved rANS payload while enforcing `max_uncompressed` BEFORE allocating.
pub fn decompress_lz4_bounded(payload: &[u8], max_uncompressed: usize) -> Result<Vec<u8>> {
    if payload.len() < 4 {
        bail!(
            "LZ4 payload too short for size prefix: {} bytes",
            payload.len()
        );
    }
    let header_word = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    if (header_word & rans::RANS_MAGIC_BIT) != 0 {
        return rans::decompress_rans_bounded(payload, max_uncompressed);
    }
    let declared = header_word as usize;
    if declared > max_uncompressed {
        bail!("LZ4 declared uncompressed size {declared} exceeds limit {max_uncompressed}");
    }
    lz4_flex::decompress(&payload[4..], declared)
        .context("failed to decompress bounded LZ4 payload")
}

/// In-place rectangular pixel copy (used for lossless scroll acceleration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyRectOp {
    pub dst_x: u16,
    pub dst_y: u16,
    pub w: u16,
    pub h: u16,
    pub src_x: u16,
    pub src_y: u16,
}

/// Apply an in-place rectangular BGRA copy on a `width x height` framebuffer,
/// safely handling overlapping upward, downward, leftward, and rightward shifts.
pub fn apply_copy_rect(bgra: &mut [u8], width: u32, height: u32, op: CopyRectOp) {
    let w = op.w as usize;
    let h = op.h as usize;
    let sx = op.src_x as usize;
    let sy = op.src_y as usize;
    let dx = op.dst_x as usize;
    let dy = op.dst_y as usize;
    let fb_w = width as usize;
    let fb_h = height as usize;

    let Some(needed) = fb_w.checked_mul(fb_h).and_then(|px| px.checked_mul(4)) else {
        return;
    };
    if bgra.len() < needed
        || w == 0
        || h == 0
        || sx + w > fb_w
        || dx + w > fb_w
        || sy + h > fb_h
        || dy + h > fb_h
        || (sx == dx && sy == dy)
    {
        return;
    }

    let row_bytes = w * 4;
    let stride = fb_w * 4;
    if dy > sy {
        // Copying downward: iterate bottom-to-top to avoid overwriting source rows.
        for row in (0..h).rev() {
            let src_off = (sy + row) * stride + sx * 4;
            let dst_off = (dy + row) * stride + dx * 4;
            bgra.copy_within(src_off..src_off + row_bytes, dst_off);
        }
    } else {
        // Copying upward or horizontally: iterate top-to-bottom.
        for row in 0..h {
            let src_off = (sy + row) * stride + sx * 4;
            let dst_off = (dy + row) * stride + dx * 4;
            bgra.copy_within(src_off..src_off + row_bytes, dst_off);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic fixed-point 8x8 orthonormal DCT-II / IDCT-III (pure i32)
// ---------------------------------------------------------------------------

// Orthonormal 8-point DCT basis constants scaled by 16384 (1 << 14):
// C4 = round(1/sqrt(8) * 16384) = round(0.5 * cos(4*pi/16) * 16384) = 5793
// Ck = round(0.5 * cos(k*pi/16) * 16384)
const C1: i32 = 8034;
const C2: i32 = 7568;
const C3: i32 = 6811;
const C4: i32 = 5793;
const C5: i32 = 4551;
const C6: i32 = 3135;
const C7: i32 = 1598;

#[inline]
fn dct_1d(x: [i32; 8], shift: u32) -> [i32; 8] {
    let round = 1i32 << (shift - 1);
    let s0 = x[0] + x[7];
    let s1 = x[1] + x[6];
    let s2 = x[2] + x[5];
    let s3 = x[3] + x[4];
    let d0 = x[0] - x[7];
    let d1 = x[1] - x[6];
    let d2 = x[2] - x[5];
    let d3 = x[3] - x[4];

    [
        (C4 * (s0 + s1 + s2 + s3) + round) >> shift,
        (C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3 + round) >> shift,
        (C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3 + round) >> shift,
        (C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3 + round) >> shift,
        (C4 * (s0 - s1 - s2 + s3) + round) >> shift,
        (C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3 + round) >> shift,
        (C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3 + round) >> shift,
        (C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3 + round) >> shift,
    ]
}

#[inline]
fn idct_1d(x: [i32; 8], shift: u32) -> [i32; 8] {
    let round = 1i32 << (shift - 1);
    let a0 = C4.wrapping_mul(x[0].wrapping_add(x[4]));
    let a1 = C4.wrapping_mul(x[0].wrapping_sub(x[4]));
    let b0 = C2.wrapping_mul(x[2]).wrapping_add(C6.wrapping_mul(x[6]));
    let b1 = C6.wrapping_mul(x[2]).wrapping_sub(C2.wrapping_mul(x[6]));

    let e0 = a0.wrapping_add(b0);
    let e1 = a1.wrapping_add(b1);
    let e2 = a1.wrapping_sub(b1);
    let e3 = a0.wrapping_sub(b0);

    let o0 = C1
        .wrapping_mul(x[1])
        .wrapping_add(C3.wrapping_mul(x[3]))
        .wrapping_add(C5.wrapping_mul(x[5]))
        .wrapping_add(C7.wrapping_mul(x[7]));
    let o1 = C3
        .wrapping_mul(x[1])
        .wrapping_sub(C7.wrapping_mul(x[3]))
        .wrapping_sub(C1.wrapping_mul(x[5]))
        .wrapping_sub(C5.wrapping_mul(x[7]));
    let o2 = C5
        .wrapping_mul(x[1])
        .wrapping_sub(C1.wrapping_mul(x[3]))
        .wrapping_add(C7.wrapping_mul(x[5]))
        .wrapping_add(C3.wrapping_mul(x[7]));
    let o3 = C7
        .wrapping_mul(x[1])
        .wrapping_sub(C5.wrapping_mul(x[3]))
        .wrapping_add(C3.wrapping_mul(x[5]))
        .wrapping_sub(C1.wrapping_mul(x[7]));

    [
        e0.wrapping_add(o0).wrapping_add(round) >> shift,
        e1.wrapping_add(o1).wrapping_add(round) >> shift,
        e2.wrapping_add(o2).wrapping_add(round) >> shift,
        e3.wrapping_add(o3).wrapping_add(round) >> shift,
        e3.wrapping_sub(o3).wrapping_add(round) >> shift,
        e2.wrapping_sub(o2).wrapping_add(round) >> shift,
        e1.wrapping_sub(o1).wrapping_add(round) >> shift,
        e0.wrapping_sub(o0).wrapping_add(round) >> shift,
    ]
}

/// Forward 2D 8x8 integer DCT. Input values are in `[-255, 255]`.
pub fn dct_8x8(input: &[i32; 64]) -> [i32; 64] {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { simd::avx2::dct_8x8(input) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            return unsafe { simd::neon::dct_8x8(input) };
        }
    }
    #[allow(unreachable_code)]
    simd::dct_8x8_scalar(input)
}

/// Inverse 2D 8x8 integer DCT.
pub fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { simd::avx2::idct_8x8(coeffs) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            return unsafe { simd::neon::idct_8x8(coeffs) };
        }
    }
    #[allow(unreachable_code)]
    simd::idct_8x8_scalar(coeffs)
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Standard JPEG Annex K luminance quantization matrix (Q=50 baseline).
const BASE_LUMA_QUANT: [u32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG Annex K chrominance quantization matrix (Q=50 baseline).
const BASE_CHROMA_QUANT: [u32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

fn build_quant_table(base: &[u32; 64], quality: u8) -> [i32; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut table = [1i32; 64];
    for i in 0..64 {
        let v = ((base[i] * scale + 50) / 100).clamp(1, 255);
        table[i] = v as i32;
    }
    table
}

#[inline]
fn quantize_div(val: i32, q: i32) -> i32 {
    if val >= 0 {
        (val + (q >> 1)) / q
    } else {
        -((-val + (q >> 1)) / q)
    }
}

// ---------------------------------------------------------------------------
// Deterministic integer full-range BT.601 BGRA <-> 4:2:0 YCbCr conversion
// ---------------------------------------------------------------------------

#[inline]
fn rgb_to_ycbcr(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let y = ((77 * r + 150 * g + 29 * b + 128) >> 8).clamp(0, 255);
    let cb = (((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128).clamp(0, 255);
    let cr = (((128 * r - 107 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255);
    (y, cb, cr)
}

#[inline]
fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let c = cb - 128;
    let d = cr - 128;
    let r = (y + ((359 * d + 128) >> 8)).clamp(0, 255) as u8;
    let g = (y - ((88 * c + 183 * d + 128) >> 8)).clamp(0, 255) as u8;
    let b = (y + ((454 * c + 128) >> 8)).clamp(0, 255) as u8;
    (r, g, b)
}

/// Extract a tile `(x0, y0, tw, th)` from a BGRA framebuffer into padded 64x64 Y and 32x32 Cb/Cr planes.
/// Edge pixels are replicated to fill partial tiles so 8x8 DCT blocks have no artificial sharp borders.
#[allow(clippy::too_many_arguments)]
fn bgra_tile_to_ycbcr420(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    y_plane: &mut [i32; 64 * 64],
    cb_plane: &mut [i32; 32 * 32],
    cr_plane: &mut [i32; 32 * 32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::bgra_tile_to_ycbcr420(
                    bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            unsafe {
                simd::neon::bgra_tile_to_ycbcr420(
                    bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    simd::bgra_tile_to_ycbcr420_scalar(bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane);
}

/// Extract a motion-predicted tile `(x0_hpel, y0_hpel)` (half-pixel units: `2 * sx + hx`, `2 * sy + hy`)
/// into padded 64x64 Y and 32x32 Cb/Cr planes using bit-exact integer bilinear interpolation.
#[allow(clippy::too_many_arguments)]
fn bgra_tile_to_ycbcr420_hpel(
    bgra: &[u8],
    fb_w: u32,
    x0_hpel: u32,
    y0_hpel: u32,
    tw: u32,
    th: u32,
    y_plane: &mut [i32; 64 * 64],
    cb_plane: &mut [i32; 32 * 32],
    cr_plane: &mut [i32; 32 * 32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::bgra_tile_to_ycbcr420_hpel(
                    bgra, fb_w, x0_hpel, y0_hpel, tw, th, y_plane, cb_plane, cr_plane,
                );
            }
            return;
        }
    }
    simd::bgra_tile_to_ycbcr420_hpel_scalar(
        bgra, fb_w, x0_hpel, y0_hpel, tw, th, y_plane, cb_plane, cr_plane,
    );
}

/// Write reconstructed 64x64 Y + 32x32 Cb/Cr (+ optional 64x64 Alpha) planes back into a BGRA tile.
/// When `a_plane` is `Some`, RGB channels are clamped to `<= A` to preserve valid premultiplied ARGB8888.
#[allow(clippy::too_many_arguments)]
fn ycbcr420_to_bgra_tile(
    y_plane: &[i32; 64 * 64],
    cb_plane: &[i32; 32 * 32],
    cr_plane: &[i32; 32 * 32],
    a_plane: Option<&[i32; 64 * 64]>,
    bgra: &mut [u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::ycbcr420_to_bgra_tile(
                    y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            unsafe {
                simd::neon::ycbcr420_to_bgra_tile(
                    y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    simd::ycbcr420_to_bgra_tile_scalar(
        y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
    );
}

/// Extract the alpha channel of a tile `(x0, y0, tw, th)` into a padded 64x64 plane.
fn bgra_tile_to_alpha(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    a_plane: &mut [i32; 64 * 64],
) {
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    for py in 0..64usize {
        let sy = (y0 as usize) + py.min(max_y);
        let row_base = sy * stride;
        for px in 0..64usize {
            let sx = (x0 as usize) + px.min(max_x);
            a_plane[py * 64 + px] = bgra[row_base + sx * 4 + 3] as i32;
        }
    }
}

/// Extract the alpha channel of a motion-predicted tile `(x0_hpel, y0_hpel)` into a padded 64x64 plane.
fn bgra_tile_to_alpha_hpel(
    bgra: &[u8],
    fb_w: u32,
    x0_hpel: u32,
    y0_hpel: u32,
    tw: u32,
    th: u32,
    a_plane: &mut [i32; 64 * 64],
) {
    let sx = x0_hpel >> 1;
    let sy = y0_hpel >> 1;
    let hx = (x0_hpel & 1) as usize;
    let hy = (y0_hpel & 1) as usize;
    if hx == 0 && hy == 0 {
        bgra_tile_to_alpha(bgra, fb_w, sx, sy, tw, th, a_plane);
        return;
    }
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    for py in 0..64usize {
        let cy0 = py.min(max_y);
        let cy1 = (cy0 + hy).min(max_y);
        let row0 = (sy as usize + cy0) * stride;
        let row1 = (sy as usize + cy1) * stride;
        for px in 0..64usize {
            let cx0 = px.min(max_x);
            let cx1 = (cx0 + hx).min(max_x);
            let a00 = bgra[row0 + (sx as usize + cx0) * 4 + 3] as i32;
            let a01 = bgra[row0 + (sx as usize + cx1) * 4 + 3] as i32;
            let a10 = bgra[row1 + (sx as usize + cx0) * 4 + 3] as i32;
            let a11 = bgra[row1 + (sx as usize + cx1) * 4 + 3] as i32;
            a_plane[py * 64 + px] = match (hx, hy) {
                (1, 0) => (a00 + a01 + 1) >> 1,
                (0, 1) => (a00 + a10 + 1) >> 1,
                _ => (a00 + a01 + a10 + a11 + 2) >> 2,
            };
        }
    }
}

/// Extract a 2x2-downsampled tile `(x0, y0, tw, th)` into padded 32x32 Y and 16x16 Cb/Cr planes.
/// Used for progressive fast motion/scroll encoding at HiDPI (4x fewer pixels & DCT blocks).
#[allow(clippy::too_many_arguments)]
fn bgra_tile_to_ycbcr420_half(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    y_plane: &mut [i32; 32 * 32],
    cb_plane: &mut [i32; 16 * 16],
    cr_plane: &mut [i32; 16 * 16],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::bgra_tile_to_ycbcr420_half(
                    bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
                );
            }
            return;
        }
    }
    simd::bgra_tile_to_ycbcr420_half_scalar(
        bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
    );
}

/// Write reconstructed 32x32 Y + 16x16 Cb/Cr (+ optional 32x32 Alpha) planes back into a BGRA tile
/// using deterministic integer bilinear upscaling (bit-identical on encoder and decoder).
#[allow(clippy::too_many_arguments)]
fn ycbcr420_half_to_bgra_tile(
    y_plane: &[i32; 32 * 32],
    cb_plane: &[i32; 16 * 16],
    cr_plane: &[i32; 16 * 16],
    a_plane: Option<&[i32; 32 * 32]>,
    bgra: &mut [u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::ycbcr420_half_to_bgra_tile(
                    y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
                );
            }
            return;
        }
    }
    simd::ycbcr420_half_to_bgra_tile_scalar(
        y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
    );
}

/// Extract a 2x2-downsampled alpha plane `(32x32)` for a tile `(x0, y0, tw, th)`.
fn bgra_tile_to_alpha_half(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    a_plane: &mut [i32; 32 * 32],
) {
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    for hy in 0..32usize {
        let sy0 = (y0 as usize) + (hy * 2).min(max_y);
        let sy1 = (y0 as usize) + (hy * 2 + 1).min(max_y);
        let row0 = sy0 * stride;
        let row1 = sy1 * stride;
        for hx in 0..32usize {
            let sx0 = (x0 as usize) + (hx * 2).min(max_x);
            let sx1 = (x0 as usize) + (hx * 2 + 1).min(max_x);
            let a00 = bgra[row0 + sx0 * 4 + 3] as i32;
            let a01 = bgra[row0 + sx1 * 4 + 3] as i32;
            let a10 = bgra[row1 + sx0 * 4 + 3] as i32;
            let a11 = bgra[row1 + sx1 * 4 + 3] as i32;
            a_plane[hy * 32 + hx] = (a00 + a01 + a10 + a11 + 2) >> 2;
        }
    }
}

/// Extract a 4x4-downsampled tile `(x0, y0, tw, th)` into padded 16x16 Y and 8x8 Cb/Cr planes.
/// Used as a heavy-congestion fallback for large HiDPI video/motion updates (16x fewer pixels).
#[allow(clippy::too_many_arguments)]
fn bgra_tile_to_ycbcr420_quarter(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    y_plane: &mut [i32; 16 * 16],
    cb_plane: &mut [i32; 8 * 8],
    cr_plane: &mut [i32; 8 * 8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::bgra_tile_to_ycbcr420_quarter(
                    bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
                );
            }
            return;
        }
    }
    simd::bgra_tile_to_ycbcr420_quarter_scalar(
        bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
    );
}

/// Write reconstructed 16x16 Y + 8x8 Cb/Cr (+ optional 16x16 Alpha) planes back into a BGRA tile
/// using deterministic integer 4x4 bilinear upscaling (bit-identical on encoder and decoder).
#[allow(clippy::too_many_arguments)]
fn ycbcr420_quarter_to_bgra_tile(
    y_plane: &[i32; 16 * 16],
    cb_plane: &[i32; 8 * 8],
    cr_plane: &[i32; 8 * 8],
    a_plane: Option<&[i32; 16 * 16]>,
    bgra: &mut [u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::ycbcr420_quarter_to_bgra_tile(
                    y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
                );
            }
            return;
        }
    }
    simd::ycbcr420_quarter_to_bgra_tile_scalar(
        y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
    );
}

/// Extract a 4x4-downsampled alpha plane `(16x16)` for a tile `(x0, y0, tw, th)`.
fn bgra_tile_to_alpha_quarter(
    bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    a_plane: &mut [i32; 16 * 16],
) {
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    for qy in 0..16usize {
        let base_y = qy * 4;
        for qx in 0..16usize {
            let base_x = qx * 4;
            let mut sum_a = 0i32;
            for dy in 0..4usize {
                let sy = (y0 as usize) + (base_y + dy).min(max_y);
                let row = sy * stride;
                for dx in 0..4usize {
                    let sx = (x0 as usize) + (base_x + dx).min(max_x);
                    sum_a += bgra[row + sx * 4 + 3] as i32;
                }
            }
            a_plane[qy * 16 + qx] = (sum_a + 8) >> 4;
        }
    }
}

// ---------------------------------------------------------------------------
// Compact RLE + zigzag DCT block serialization
// ---------------------------------------------------------------------------

/// Write a quantized signed coefficient using 1 byte for `[-127, 127]` or 3 bytes (`0x80` + i16 LE).
#[inline]
fn write_coeff(out: &mut Vec<u8>, v: i32) {
    if (-127..=127).contains(&v) {
        out.push(v as i8 as u8);
    } else {
        out.push(0x80);
        let clamped = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }
}

#[inline]
fn read_coeff(data: &[u8], pos: &mut usize) -> Result<i32> {
    if *pos >= data.len() {
        bail!("unexpected EOF reading DCT coefficient");
    }
    let tag = data[*pos];
    *pos += 1;
    if tag == 0x80 {
        if *pos + 2 > data.len() {
            bail!("unexpected EOF reading i16 DCT coefficient");
        }
        let v = i16::from_le_bytes([data[*pos], data[*pos + 1]]) as i32;
        *pos += 2;
        Ok(v)
    } else {
        Ok(tag as i8 as i32)
    }
}

/// Serialize one quantized 8x8 block (DC delta-coded against `prev_dc`, AC zero-run + EOB).
fn encode_rle_block(q_block: &[i32; 64], prev_dc: &mut i32, out: &mut Vec<u8>) {
    let dc = q_block[0];
    write_coeff(out, dc - *prev_dc);
    *prev_dc = dc;

    let mut last_nz = 0usize;
    for zi in (1..64).rev() {
        if q_block[ZIGZAG[zi]] != 0 {
            last_nz = zi;
            break;
        }
    }
    if last_nz == 0 {
        out.push(0); // EOB immediately
        return;
    }

    let mut zeros = 0u8;
    for zi in 1..=last_nz {
        let v = q_block[ZIGZAG[zi]];
        if v == 0 {
            zeros += 1;
        } else {
            // Token byte: bit 7 = 0 means (run + 1) followed by coeff; 0 = EOB.
            // Since last_nz <= 63, zeros <= 62, so (zeros + 1) is always in 1..=63.
            out.push(zeros + 1);
            write_coeff(out, v);
            zeros = 0;
        }
    }
    if last_nz < 63 {
        out.push(0); // EOB
    }
}

/// Reversible integer 2x2 lifting Walsh-Hadamard transform over 4 quantized 8x8 DC coefficients
/// `(d00, d01, d10, d11)` representing a 16x16 superblock.
/// Maps identical `[c, c, c, c]` blocks to `(c, 0, 0, 0)` with zero dynamic-range expansion and 100% lossless inverse!
#[inline]
fn fwd_wht_2x2_reversible(d00: i32, d01: i32, d10: i32, d11: i32) -> (i32, i32, i32, i32) {
    let h0 = d01 - d00;
    let l0 = d00 + (h0 >> 1);
    let h1 = d11 - d10;
    let l1 = d10 + (h1 >> 1);
    let w10 = l1 - l0;
    let w00 = l0 + (w10 >> 1);
    let w11 = h1 - h0;
    let w01 = h0 + (w11 >> 1);
    (w00, w01, w10, w11)
}

#[inline]
fn inv_wht_2x2_reversible(w00: i32, w01: i32, w10: i32, w11: i32) -> (i32, i32, i32, i32) {
    let h0 = w01.wrapping_sub(w11 >> 1);
    let h1 = h0.wrapping_add(w11);
    let l0 = w00.wrapping_sub(w10 >> 1);
    let l1 = l0.wrapping_add(w10);
    let d00 = l0.wrapping_sub(h0 >> 1);
    let d01 = d00.wrapping_add(h0);
    let d10 = l1.wrapping_sub(h1 >> 1);
    let d11 = d10.wrapping_add(h1);
    (d00, d01, d10, d11)
}

#[inline]
fn apply_fwd_superblock_wht_dcs(dcs: &mut [i32], blocks_x: usize, blocks_y: usize) {
    let sb_x = blocks_x / 2;
    let sb_y = blocks_y / 2;
    for sy in 0..sb_y {
        let r0 = (sy * 2) * blocks_x;
        let r1 = r0 + blocks_x;
        for sx in 0..sb_x {
            let c0 = sx * 2;
            let (w00, w01, w10, w11) = fwd_wht_2x2_reversible(
                dcs[r0 + c0],
                dcs[r0 + c0 + 1],
                dcs[r1 + c0],
                dcs[r1 + c0 + 1],
            );
            dcs[r0 + c0] = w00;
            dcs[r0 + c0 + 1] = w01;
            dcs[r1 + c0] = w10;
            dcs[r1 + c0 + 1] = w11;
        }
    }
}

#[inline]
fn apply_inv_superblock_wht_dcs(dcs: &mut [i32], blocks_x: usize, blocks_y: usize) {
    let sb_x = blocks_x / 2;
    let sb_y = blocks_y / 2;
    for sy in 0..sb_y {
        let r0 = (sy * 2) * blocks_x;
        let r1 = r0 + blocks_x;
        for sx in 0..sb_x {
            let c0 = sx * 2;
            let (d00, d01, d10, d11) = inv_wht_2x2_reversible(
                dcs[r0 + c0],
                dcs[r0 + c0 + 1],
                dcs[r1 + c0],
                dcs[r1 + c0 + 1],
            );
            dcs[r0 + c0] = d00;
            dcs[r0 + c0 + 1] = d01;
            dcs[r1 + c0] = d10;
            dcs[r1 + c0 + 1] = d11;
        }
    }
}

#[inline]
fn encode_plane_qblocks(
    q_blocks: &mut [[i32; 64]],
    blocks_x: usize,
    blocks_y: usize,
    out: &mut Vec<u8>,
) {
    let count = blocks_x * blocks_y;
    let mut dcs = [0i32; 64];
    for i in 0..count {
        dcs[i] = q_blocks[i][0];
    }
    apply_fwd_superblock_wht_dcs(&mut dcs[..count], blocks_x, blocks_y);

    let mut prev_dc = 0i32;
    let sb_x = blocks_x / 2;
    let sb_y = blocks_y / 2;
    for i in 0..count {
        let by = i / blocks_x;
        let bx = i % blocks_x;
        let in_sb = by < sb_y * 2 && bx < sb_x * 2;
        if in_sb && (by & 1) == 0 && (bx & 1) == 0 {
            let dc = dcs[i];
            q_blocks[i][0] = dc;
            encode_rle_block(&q_blocks[i], &mut prev_dc, out);
        } else if in_sb {
            // Superblock WHT detail coefficient (w01, w10, w11) is already zero-centered!
            let mut zero_pred = 0i32;
            q_blocks[i][0] = dcs[i];
            encode_rle_block(&q_blocks[i], &mut zero_pred, out);
        } else {
            let dc = dcs[i];
            q_blocks[i][0] = dc;
            encode_rle_block(&q_blocks[i], &mut prev_dc, out);
        }
    }
}

/// Like [`encode_plane_qblocks`], but reads blocks straight from the GPU's packed layout (32 words
/// per block, two `i16` coefficients per word in natural order) and emits identical bytes.
/// Blocks without AC coefficients, the common case on UI content, skip unpacking and the
/// zigzag scan.
fn encode_plane_packed(words: &[i32], blocks_x: usize, blocks_y: usize, out: &mut Vec<u8>) {
    let count = blocks_x * blocks_y;
    let mut dcs = [0i32; 64];
    for (i, dc) in dcs.iter_mut().enumerate().take(count) {
        *dc = (words[i * 32] as u32 & 0xFFFF) as u16 as i16 as i32;
    }
    apply_fwd_superblock_wht_dcs(&mut dcs[..count], blocks_x, blocks_y);

    let mut prev_dc = 0i32;
    let sb_x = blocks_x / 2;
    let sb_y = blocks_y / 2;
    for (i, &dc) in dcs.iter().enumerate().take(count) {
        let by = i / blocks_x;
        let bx = i % blocks_x;
        let in_sb = by < sb_y * 2 && bx < sb_x * 2;
        // Superblock WHT detail coefficients (w01, w10, w11) are already zero-centered.
        let mut zero_pred = 0i32;
        let pred = if in_sb && ((by & 1) != 0 || (bx & 1) != 0) {
            &mut zero_pred
        } else {
            &mut prev_dc
        };
        let blk = &words[i * 32..i * 32 + 32];
        let ac_bits = blk[1..]
            .iter()
            .fold((blk[0] as u32) >> 16, |acc, &w| acc | w as u32);
        if ac_bits == 0 {
            write_coeff(out, dc - *pred);
            *pred = dc;
            out.push(0); // EOB immediately
        } else {
            let mut q = [0i32; 64];
            for (j, &w) in blk.iter().enumerate() {
                let w = w as u32;
                q[j * 2] = (w & 0xFFFF) as u16 as i16 as i32;
                q[j * 2 + 1] = (w >> 16) as u16 as i16 as i32;
            }
            q[0] = dc;
            encode_rle_block(&q, pred, out);
        }
    }
}

/// Encode a single plane (`plane_w x plane_w`, 64 or 32) of 8x8 DCT blocks and reconstruct
/// the exact decoded plane in-place (into `recon_plane`) so encoder and decoder stay bit-identical.
#[allow(clippy::too_many_arguments)]
fn encode_and_reconstruct_plane(
    cur_plane: &[i32],
    ref_plane: Option<&[i32]>,
    recon_plane: &mut [i32],
    plane_dim: usize,
    blocks_x: usize,
    blocks_y: usize,
    quant: &[i32; 64],
    out: &mut Vec<u8>,
) {
    let count = blocks_x * blocks_y;
    let mut q_blocks_uninit: [std::mem::MaybeUninit<[i32; 64]>; 64] =
        [const { std::mem::MaybeUninit::uninit() }; 64];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0i32; 64];
            if let Some(ref_p) = ref_plane {
                for r in 0..8 {
                    let row_off = (by * 8 + r) * plane_dim + bx * 8;
                    for c in 0..8 {
                        let idx = row_off + c;
                        block[r * 8 + c] = cur_plane[idx] - ref_p[idx];
                    }
                }
            } else {
                for r in 0..8 {
                    let row_off = (by * 8 + r) * plane_dim + bx * 8;
                    for c in 0..8 {
                        block[r * 8 + c] = cur_plane[row_off + c] - 128;
                    }
                }
            }
            let dct = dct_8x8(&block);
            let q_slot = q_blocks_uninit[by * blocks_x + bx].write([0i32; 64]);
            let mut dequant = [0i32; 64];
            let q0 = quantize_div(dct[0], quant[0]);
            q_slot[0] = q0;
            dequant[0] = q0 * quant[0];
            let mut has_ac = false;
            for i in 1..64 {
                let qv = quantize_div(dct[i], quant[i]);
                q_slot[i] = qv;
                if qv != 0 {
                    has_ac = true;
                    dequant[i] = qv * quant[i];
                }
            }

            if !has_ac {
                let col0 = dequant[0].wrapping_mul(C4).wrapping_add(1 << 10) >> 11;
                let dc_spatial = col0.wrapping_mul(C4).wrapping_add(1 << 16) >> 17;
                if let Some(ref_p) = ref_plane {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            let idx = row_off + c;
                            recon_plane[idx] = (ref_p[idx] + dc_spatial).clamp(0, 255);
                        }
                    }
                } else {
                    let v = (dc_spatial + 128).clamp(0, 255);
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        recon_plane[row_off..row_off + 8].fill(v);
                    }
                }
            } else {
                let idct = idct_8x8(&dequant);
                if let Some(ref_p) = ref_plane {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            let idx = row_off + c;
                            recon_plane[idx] = (ref_p[idx] + idct[r * 8 + c]).clamp(0, 255);
                        }
                    }
                } else {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            recon_plane[row_off + c] = (idct[r * 8 + c] + 128).clamp(0, 255);
                        }
                    }
                }
            }
        }
    }
    let q_blocks_init = unsafe {
        std::slice::from_raw_parts_mut(q_blocks_uninit.as_mut_ptr() as *mut [i32; 64], count)
    };
    encode_plane_qblocks(q_blocks_init, blocks_x, blocks_y, out);
    deblock_and_sao_plane(recon_plane, plane_dim, blocks_x, blocks_y, quant[0]);
}

#[allow(clippy::too_many_arguments)]
fn decode_plane(
    data: &[u8],
    pos: &mut usize,
    ref_plane: Option<&[i32]>,
    recon_plane: &mut [i32],
    plane_dim: usize,
    blocks_x: usize,
    blocks_y: usize,
    quant: &[i32; 64],
) -> Result<()> {
    let mut prev_dc = 0i32;
    let count = blocks_x * blocks_y;
    let sb_x = blocks_x / 2;
    let sb_y = blocks_y / 2;
    let mut ac_blocks: [std::mem::MaybeUninit<[i32; 64]>; 64] =
        [const { std::mem::MaybeUninit::uninit() }; 64];
    let mut has_ac = [false; 64];
    let mut dcs = [0i32; 64];

    for i in 0..count {
        let by = i / blocks_x;
        let bx = i % blocks_x;
        let in_sb = by < sb_y * 2 && bx < sb_x * 2;
        let dc_delta = read_coeff(data, pos)?;
        let dc = if in_sb && !((by & 1) == 0 && (bx & 1) == 0) {
            dc_delta
        } else {
            let v = prev_dc.saturating_add(dc_delta);
            prev_dc = v;
            v
        };
        dcs[i] = dc;

        if *pos >= data.len() {
            bail!("unexpected EOF reading DCT RLE token");
        }
        let first_tok = data[*pos];
        *pos += 1;
        if first_tok != 0 {
            let blk = ac_blocks[i].write([0i32; 64]);
            has_ac[i] = true;
            let mut tok = first_tok;
            let mut zi = 1usize;
            loop {
                let run = (tok - 1) as usize;
                zi += run;
                if zi >= 64 {
                    bail!("DCT RLE run overflow");
                }
                let zz = ZIGZAG[zi];
                let c = read_coeff(data, pos)?;
                blk[zz] = c.saturating_mul(quant[zz]).clamp(-32_768, 32_768);
                zi += 1;
                if zi >= 64 {
                    break;
                }
                if *pos >= data.len() {
                    bail!("unexpected EOF reading DCT RLE token");
                }
                tok = data[*pos];
                *pos += 1;
                if tok == 0 {
                    break;
                }
            }
        }
    }
    apply_inv_superblock_wht_dcs(&mut dcs[..count], blocks_x, blocks_y);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let b_idx = by * blocks_x + bx;
            let deq0 = dcs[b_idx].saturating_mul(quant[0]).clamp(-32_768, 32_768);
            if !has_ac[b_idx] {
                let col0 = deq0.wrapping_mul(C4).wrapping_add(1 << 10) >> 11;
                let dc_spatial = col0.wrapping_mul(C4).wrapping_add(1 << 16) >> 17;
                if let Some(ref_p) = ref_plane {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            let idx = row_off + c;
                            recon_plane[idx] = (ref_p[idx] + dc_spatial).clamp(0, 255);
                        }
                    }
                } else {
                    let v = (dc_spatial + 128).clamp(0, 255);
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        recon_plane[row_off..row_off + 8].fill(v);
                    }
                }
            } else {
                let dequant = unsafe { ac_blocks[b_idx].assume_init_mut() };
                dequant[0] = deq0;
                let idct = idct_8x8(dequant);
                if let Some(ref_p) = ref_plane {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            let idx = row_off + c;
                            recon_plane[idx] = (ref_p[idx] + idct[r * 8 + c]).clamp(0, 255);
                        }
                    }
                } else {
                    for r in 0..8 {
                        let row_off = (by * 8 + r) * plane_dim + bx * 8;
                        for c in 0..8 {
                            recon_plane[row_off + c] = (idct[r * 8 + c] + 128).clamp(0, 255);
                        }
                    }
                }
            }
        }
    }
    deblock_and_sao_plane(recon_plane, plane_dim, blocks_x, blocks_y, quant[0]);
    Ok(())
}

/// HEVC-style in-loop 4-tap 8x8 boundary deblocking filter (`[p1, p0 | q0, q1]`) combined with
/// edge-preserving Sample Adaptive Offset (SAO) anti-ringing damping.
/// Eliminates 8x8 block grid edges inside the temporal reference loop while preserving sharp 1-px UI text.
fn deblock_and_sao_plane(
    plane: &mut [i32],
    plane_dim: usize,
    blocks_x: usize,
    blocks_y: usize,
    q_dc: i32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::deblock_and_sao_plane(plane, plane_dim, blocks_x, blocks_y, q_dc);
            }
            return;
        }
    }
    simd::deblock_and_sao_plane_scalar(plane, plane_dim, blocks_x, blocks_y, q_dc);
}

// ---------------------------------------------------------------------------
// Sub-region vertical & horizontal scroll detection
// ---------------------------------------------------------------------------

fn hash_row_strip(
    bgra: &[u8],
    width: usize,
    y: usize,
    x_start: usize,
    x_end: usize,
) -> (u64, bool) {
    let stride = width * 4;
    let base = y * stride;
    let step = ((x_end - x_start) / 96).max(1);
    let mut h: u64 = 0xcbf29ce484222325;
    let first_off = base + x_start * 4;
    let first_px = u32::from_ne_bytes([
        bgra[first_off],
        bgra[first_off + 1],
        bgra[first_off + 2],
        bgra[first_off + 3],
    ]);
    let mut uniform = true;
    let mut x = x_start;
    while x < x_end {
        let off = base + x * 4;
        let px = u32::from_ne_bytes([bgra[off], bgra[off + 1], bgra[off + 2], bgra[off + 3]]);
        if px != first_px {
            uniform = false;
        }
        h ^= px as u64;
        h = h.wrapping_mul(0x100000001b3);
        x += step;
    }
    (h, uniform)
}

fn hash_col_strip(
    bgra: &[u8],
    width: usize,
    x: usize,
    y_start: usize,
    y_end: usize,
) -> (u64, bool) {
    let stride = width * 4;
    let step = ((y_end - y_start) / 96).max(1);
    let mut h: u64 = 0xcbf29ce484222325;
    let first_off = y_start * stride + x * 4;
    let first_px = u32::from_ne_bytes([
        bgra[first_off],
        bgra[first_off + 1],
        bgra[first_off + 2],
        bgra[first_off + 3],
    ]);
    let mut uniform = true;
    let mut y = y_start;
    while y < y_end {
        let off = y * stride + x * 4;
        let px = u32::from_ne_bytes([bgra[off], bgra[off + 1], bgra[off + 2], bgra[off + 3]]);
        if px != first_px {
            uniform = false;
        }
        h ^= px as u64;
        h = h.wrapping_mul(0x100000001b3);
        y += step;
    }
    (h, uniform)
}

/// Detect a dominant vertical or horizontal scroll sub-rectangle between `prev` and `cur`.
pub fn detect_scroll_copy_rect(
    prev: &[u8],
    cur: &[u8],
    width: u32,
    height: u32,
) -> Option<CopyRectOp> {
    let w = width as usize;
    let h = height as usize;
    if w < 192 || h < 192 || prev.len() != cur.len() {
        return None;
    }
    if let Some(op) = detect_vertical_scroll(prev, cur, w, h) {
        return Some(op);
    }
    detect_horizontal_scroll(prev, cur, w, h)
}

fn detect_vertical_scroll(prev: &[u8], cur: &[u8], w: usize, h: usize) -> Option<CopyRectOp> {
    // 1. Fast exact-hash 3-band probe (handles both small and huge page-down scrolls).
    let bands = [(w / 4, (w * 3) / 4), (w / 6, w / 2), (w / 2, (w * 5) / 6)];
    for (x_start, x_end) in bands {
        if x_end <= x_start + 32 {
            continue;
        }
        if let Some(op) = detect_vertical_scroll_band(prev, cur, w, h, x_start, x_end) {
            return Some(op);
        }
    }
    // 2. Subpixel/smooth-scroll resilient 1D row-profile SAD fallback (for Chromium/VS Code
    //    where fractional smooth scrolling perturbs antialiased font pixels by a few levels).
    for (x_start, x_end) in bands {
        if x_end <= x_start + 32 {
            continue;
        }
        if let Some(op) = detect_vertical_scroll_fuzzy(prev, cur, w, h, x_start, x_end) {
            return Some(op);
        }
    }
    None
}

/// Compute a 4-segment mean-luma profile for row `y` across `[x_start, x_end)`.
/// Invariant to small subpixel antialiasing differences during fractional smooth scrolling.
fn row_luma_profile(bgra: &[u8], width: usize, y: usize, x_start: usize, x_end: usize) -> [i32; 4] {
    let stride = width * 4;
    let base = y * stride;
    let span = x_end - x_start;
    let seg_w = (span / 4).max(1);
    let step = (seg_w / 16).max(1);
    let mut prof = [0i32; 4];
    #[allow(clippy::needless_range_loop)]
    for s in 0..4 {
        let sx0 = x_start + s * seg_w;
        let sx1 = (sx0 + seg_w).min(x_end);
        let mut sum = 0i32;
        let mut cnt = 0i32;
        let mut x = sx0;
        while x < sx1 {
            let off = base + x * 4;
            let b = bgra[off] as i32;
            let g = bgra[off + 1] as i32;
            let r = bgra[off + 2] as i32;
            sum += (r + 2 * g + b) >> 2;
            cnt += 1;
            x += step;
        }
        prof[s] = if cnt > 0 { sum / cnt } else { 0 };
    }
    prof
}

#[inline]
fn profile_sad(a: &[i32; 4], b: &[i32; 4]) -> i32 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() + (a[3] - b[3]).abs()
}

/// Fuzzy vertical scroll detector using 1D row-luma profile correlation.
/// Catches Chromium/VS Code fractional smooth-scrolling even when every antialiased pixel shifts slightly.
fn detect_vertical_scroll_fuzzy(
    prev: &[u8],
    cur: &[u8],
    w: usize,
    h: usize,
    x_start: usize,
    x_end: usize,
) -> Option<CopyRectOp> {
    let mut prev_prof = Vec::with_capacity(h);
    let mut cur_prof = Vec::with_capacity(h);
    for y in 0..h {
        prev_prof.push(row_luma_profile(prev, w, y, x_start, x_end));
        cur_prof.push(row_luma_profile(cur, w, y, x_start, x_end));
    }

    // Focus search on rows that actually changed (skip static top/bottom bars).
    let max_dy = (h / 3).min(240) as isize;
    let y_lo = (h / 8).max(16);
    let y_hi = h.saturating_sub(y_lo);
    if y_hi <= y_lo + 64 {
        return None;
    }

    // Baseline SAD at dy = 0 (how much the band changed without shifting).
    let mut base_sad: i64 = 0;
    let mut changed_samples = 0usize;
    for y in (y_lo..y_hi).step_by(2) {
        let d = profile_sad(&cur_prof[y], &prev_prof[y]);
        base_sad += d as i64;
        if d > 6 {
            changed_samples += 1;
        }
    }
    if changed_samples < 16 || base_sad < 200 {
        return None;
    }

    // Coarse-to-fine search over dy in [-max_dy, max_dy] (excluding 0).
    let mut best_dy: isize = 0;
    let mut best_sad: i64 = i64::MAX;

    // Coarse pass: step 2 rows, step 2 dy.
    for dy in (-max_dy..=max_dy).step_by(2) {
        if dy == 0 {
            continue;
        }
        let mut sad: i64 = 0;
        let mut count: i64 = 0;
        for y in (y_lo..y_hi).step_by(4) {
            let py = y as isize - dy;
            if py >= 0 && (py as usize) < h {
                sad += profile_sad(&cur_prof[y], &prev_prof[py as usize]) as i64;
                count += 1;
            }
        }
        if count > 0 {
            let norm_sad = sad * ((y_hi - y_lo) as i64 / 2) / count;
            if norm_sad < best_sad {
                best_sad = norm_sad;
                best_dy = dy;
            }
        }
    }

    // Fine pass: refine +/- 2 pixels around best_dy.
    let center = best_dy;
    for dy in (center - 2).max(-max_dy)..=(center + 2).min(max_dy) {
        if dy == 0 {
            continue;
        }
        let mut sad: i64 = 0;
        let mut count: i64 = 0;
        for y in (y_lo..y_hi).step_by(2) {
            let py = y as isize - dy;
            if py >= 0 && (py as usize) < h {
                sad += profile_sad(&cur_prof[y], &prev_prof[py as usize]) as i64;
                count += 1;
            }
        }
        if count > 0 {
            let norm_sad = sad * ((y_hi - y_lo) as i64 / 2) / count;
            if norm_sad < best_sad {
                best_sad = norm_sad;
                best_dy = dy;
            }
        }
    }

    // Require shifting by best_dy to explain at least 65% of the row-profile difference.
    if best_dy == 0 || best_sad * 100 > base_sad * 35 {
        return None;
    }

    // Find the longest contiguous vertical span where shifted row profiles match closely.
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    let mut run_len = 0usize;

    for y in 0..h {
        let py = y as isize - best_dy;
        let matches = if py >= 0 && (py as usize) < h {
            let d_shift = profile_sad(&cur_prof[y], &prev_prof[py as usize]);
            let d_zero = profile_sad(&cur_prof[y], &prev_prof[y]);
            // Match if shifted error is small AND either it actually moved or is part of uniform spacing.
            d_shift <= 12 && (d_shift <= d_zero || d_zero <= 2)
        } else {
            false
        };
        if matches {
            if run_len == 0 {
                run_start = y;
            }
            run_len += 1;
            if run_len > best_len {
                best_len = run_len;
                best_start = run_start;
            }
        } else {
            run_len = 0;
        }
    }

    if best_len < 64 {
        return None;
    }

    // Refine horizontal bounds in 16-pixel columns using approximate pixel match
    // (mean per-channel error <= 14 across sample rows, allowing up to 25% overlay rows).
    let stride = w * 4;
    let cols = w / 16;
    let mut sample_rows = Vec::with_capacity(12);
    let step_y = (best_len / 12).max(1);
    let mut sy = best_start;
    while sy < best_start + best_len && sample_rows.len() < 12 {
        sample_rows.push(sy);
        sy += step_y;
    }
    if sample_rows.is_empty() {
        return None;
    }

    let max_mismatches = sample_rows.len() / 4;
    let mut col_matches = vec![false; cols];
    #[allow(clippy::needless_range_loop)]
    for c in 0..cols {
        let x0 = c * 16;
        let mut mismatches = 0usize;
        for &y in &sample_rows {
            let py = (y as isize - best_dy) as usize;
            let c_off = y * stride + x0 * 4;
            let p_off = py * stride + x0 * 4;
            // Sample 4 pixels across the 16-pixel column.
            let mut err_sum = 0u32;
            for px in (0..16).step_by(4) {
                let co = c_off + px * 4;
                let po = p_off + px * 4;
                err_sum += cur[co].abs_diff(prev[po]) as u32
                    + cur[co + 1].abs_diff(prev[po + 1]) as u32
                    + cur[co + 2].abs_diff(prev[po + 2]) as u32;
            }
            // 4 pixels * 3 channels = 12 channels; mean <= 14 -> sum <= 168.
            if err_sum > 168 {
                mismatches += 1;
                if mismatches > max_mismatches {
                    break;
                }
            }
        }
        col_matches[c] = mismatches <= max_mismatches;
    }

    let mut bx_start = 0usize;
    let mut bx_len = 0usize;
    let mut cx_start = 0usize;
    let mut cx_len = 0usize;
    for (c, &m) in col_matches.iter().enumerate() {
        if m {
            if cx_len == 0 {
                cx_start = c;
            }
            cx_len += 1;
            if cx_len > bx_len {
                bx_len = cx_len;
                bx_start = cx_start;
            }
        } else {
            cx_len = 0;
        }
    }

    let rect_x = bx_start * 16;
    let rect_w = bx_len * 16;
    if rect_w < 128 || rect_w * best_len < 64 * 64 * 6 {
        return None;
    }

    let src_y = (best_start as isize - best_dy) as usize;
    Some(CopyRectOp {
        dst_x: rect_x as u16,
        dst_y: best_start as u16,
        w: rect_w as u16,
        h: best_len as u16,
        src_x: rect_x as u16,
        src_y: src_y as u16,
    })
}

fn detect_vertical_scroll_band(
    prev: &[u8],
    cur: &[u8],
    w: usize,
    h: usize,
    x_start: usize,
    x_end: usize,
) -> Option<CopyRectOp> {
    let mut prev_hashes = Vec::with_capacity(h);
    let mut cur_hashes = Vec::with_capacity(h);
    let mut prev_map: HashMap<u64, usize> = HashMap::with_capacity(h);

    for y in 0..h {
        let (ph, p_unif) = hash_row_strip(prev, w, y, x_start, x_end);
        let (ch, c_unif) = hash_row_strip(cur, w, y, x_start, x_end);
        prev_hashes.push((ph, p_unif));
        cur_hashes.push((ch, c_unif));
        if !p_unif {
            prev_map.entry(ph).or_insert(y);
        }
    }

    let max_dy = (h / 2) as isize;
    let mut votes: HashMap<isize, usize> = HashMap::new();
    for y in 0..h {
        let (ch, c_unif) = cur_hashes[y];
        if c_unif || ch == prev_hashes[y].0 {
            continue;
        }
        if let Some(&py) = prev_map.get(&ch) {
            let dy = y as isize - py as isize;
            if dy != 0 && dy.abs() <= max_dy {
                *votes.entry(dy).or_insert(0) += 1;
            }
        }
    }

    let (best_dy, best_votes) = votes.into_iter().max_by_key(|&(_, v)| v)?;
    if best_votes < 24 {
        return None;
    }

    // Find the longest contiguous vertical run of rows matching `cur[y] == prev[y - best_dy]`
    // in the probed strip (allowing uniform background rows inside the scrolled viewport).
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    let mut run_len = 0usize;
    let mut run_non_uniform = 0usize;
    let mut best_non_uniform = 0usize;

    for y in 0..h {
        let py = y as isize - best_dy;
        let matches = if py >= 0 && (py as usize) < h {
            let (ch, c_unif) = cur_hashes[y];
            let (ph, _) = prev_hashes[py as usize];
            ch == ph && (c_unif || ch != prev_hashes[y].0)
        } else {
            false
        };
        if matches {
            if run_len == 0 {
                run_start = y;
                run_non_uniform = 0;
            }
            run_len += 1;
            if !cur_hashes[y].1 {
                run_non_uniform += 1;
            }
            if run_len > best_len {
                best_len = run_len;
                best_start = run_start;
                best_non_uniform = run_non_uniform;
            }
        } else {
            run_len = 0;
        }
    }

    if best_len < 64 || best_non_uniform < 16 {
        return None;
    }

    // Refine horizontal bounds `[x_left, x_right)` in 16-pixel columns to exclude fixed sidebars/borders.
    // Allow up to 25% mismatched sample rows per column so small floating badges/overlays don't
    // reject the scroll rect (the dirty-tile pass automatically fixes any overlay tiles losslessly).
    let stride = w * 4;
    let cols = w / 16;
    let mut sample_rows = Vec::with_capacity(12);
    let step_y = (best_len / 12).max(1);
    let mut sy = best_start;
    while sy < best_start + best_len && sample_rows.len() < 12 {
        if !cur_hashes[sy].1 {
            sample_rows.push(sy);
        }
        sy += step_y;
    }
    if sample_rows.is_empty() {
        return None;
    }

    let max_mismatches = sample_rows.len() / 4;
    let mut col_matches = vec![false; cols];
    #[allow(clippy::needless_range_loop)]
    for c in 0..cols {
        let x0 = c * 16;
        let mut mismatches = 0usize;
        for &y in &sample_rows {
            let py = (y as isize - best_dy) as usize;
            let c_off = y * stride + x0 * 4;
            let p_off = py * stride + x0 * 4;
            if cur[c_off..c_off + 64] != prev[p_off..p_off + 64] {
                mismatches += 1;
                if mismatches > max_mismatches {
                    break;
                }
            }
        }
        col_matches[c] = mismatches <= max_mismatches;
    }

    // Find longest contiguous matching column span.
    let mut bx_start = 0usize;
    let mut bx_len = 0usize;
    let mut cx_start = 0usize;
    let mut cx_len = 0usize;
    for (c, &m) in col_matches.iter().enumerate() {
        if m {
            if cx_len == 0 {
                cx_start = c;
            }
            cx_len += 1;
            if cx_len > bx_len {
                bx_len = cx_len;
                bx_start = cx_start;
            }
        } else {
            cx_len = 0;
        }
    }

    let rect_x = bx_start * 16;
    let rect_w = bx_len * 16;
    if rect_w < 128 || rect_w * best_len < 64 * 64 * 6 {
        return None;
    }

    let src_y = (best_start as isize - best_dy) as usize;
    Some(CopyRectOp {
        dst_x: rect_x as u16,
        dst_y: best_start as u16,
        w: rect_w as u16,
        h: best_len as u16,
        src_x: rect_x as u16,
        src_y: src_y as u16,
    })
}

fn detect_horizontal_scroll(prev: &[u8], cur: &[u8], w: usize, h: usize) -> Option<CopyRectOp> {
    let y_start = h / 4;
    let y_end = (h * 3) / 4;

    let mut prev_hashes = Vec::with_capacity(w);
    let mut cur_hashes = Vec::with_capacity(w);
    let mut prev_map: HashMap<u64, usize> = HashMap::with_capacity(w);

    for x in 0..w {
        let (ph, p_unif) = hash_col_strip(prev, w, x, y_start, y_end);
        let (ch, c_unif) = hash_col_strip(cur, w, x, y_start, y_end);
        prev_hashes.push((ph, p_unif));
        cur_hashes.push((ch, c_unif));
        if !p_unif {
            prev_map.entry(ph).or_insert(x);
        }
    }

    let max_dx = (w / 2) as isize;
    let mut votes: HashMap<isize, usize> = HashMap::new();
    for x in 0..w {
        let (ch, c_unif) = cur_hashes[x];
        if c_unif || ch == prev_hashes[x].0 {
            continue;
        }
        if let Some(&px) = prev_map.get(&ch) {
            let dx = x as isize - px as isize;
            if dx != 0 && dx.abs() <= max_dx {
                *votes.entry(dx).or_insert(0) += 1;
            }
        }
    }

    let (best_dx, best_votes) = votes.into_iter().max_by_key(|&(_, v)| v)?;
    if best_votes < 24 {
        return None;
    }

    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    let mut run_len = 0usize;
    let mut run_non_uniform = 0usize;
    let mut best_non_uniform = 0usize;

    for x in 0..w {
        let px = x as isize - best_dx;
        let matches = if px >= 0 && (px as usize) < w {
            let (ch, c_unif) = cur_hashes[x];
            let (ph, _) = prev_hashes[px as usize];
            ch == ph && (c_unif || ch != prev_hashes[x].0)
        } else {
            false
        };
        if matches {
            if run_len == 0 {
                run_start = x;
                run_non_uniform = 0;
            }
            run_len += 1;
            if !cur_hashes[x].1 {
                run_non_uniform += 1;
            }
            if run_len > best_len {
                best_len = run_len;
                best_start = run_start;
                best_non_uniform = run_non_uniform;
            }
        } else {
            run_len = 0;
        }
    }

    if best_len < 64 || best_non_uniform < 16 {
        return None;
    }

    // Refine vertical bounds in 16-pixel rows.
    let stride = w * 4;
    let rows = h / 16;
    let mut sample_cols = Vec::with_capacity(12);
    let step_x = (best_len / 12).max(1);
    let mut sx = best_start;
    while sx < best_start + best_len && sample_cols.len() < 12 {
        if !cur_hashes[sx].1 {
            sample_cols.push(sx);
        }
        sx += step_x;
    }
    if sample_cols.is_empty() {
        return None;
    }

    let mut row_matches = vec![false; rows];
    #[allow(clippy::needless_range_loop)]
    for r in 0..rows {
        let y0 = r * 16;
        let mut ok = true;
        'outer: for dy in (0..16).step_by(4) {
            let y = y0 + dy;
            for &x in &sample_cols {
                let px = (x as isize - best_dx) as usize;
                let c_off = y * stride + x * 4;
                let p_off = y * stride + px * 4;
                if cur[c_off..c_off + 4] != prev[p_off..p_off + 4] {
                    ok = false;
                    break 'outer;
                }
            }
        }
        row_matches[r] = ok;
    }

    let mut by_start = 0usize;
    let mut by_len = 0usize;
    let mut cy_start = 0usize;
    let mut cy_len = 0usize;
    for (r, &m) in row_matches.iter().enumerate() {
        if m {
            if cy_len == 0 {
                cy_start = r;
            }
            cy_len += 1;
            if cy_len > by_len {
                by_len = cy_len;
                by_start = cy_start;
            }
        } else {
            cy_len = 0;
        }
    }

    let rect_y = by_start * 16;
    let rect_h = by_len * 16;
    if rect_h < 128 || best_len * rect_h < 64 * 64 * 6 {
        return None;
    }

    let src_x = (best_start as isize - best_dx) as usize;
    Some(CopyRectOp {
        dst_x: best_start as u16,
        dst_y: rect_y as u16,
        w: best_len as u16,
        h: rect_h as u16,
        src_x: src_x as u16,
        src_y: rect_y as u16,
    })
}

// ---------------------------------------------------------------------------
// SmartTileEncoder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct TileTrack {
    motion_streak: u8,
    last_change: Option<Instant>,
    is_lossy: bool,
}

/// Per tile (row-major, `cols` x `rows`), whether any of the `rects` `(x, y, w, h)` touches it.
fn damaged_tile_mask(
    rects: &[(u32, u32, u32, u32)],
    width: u32,
    height: u32,
    cols: usize,
    rows: usize,
) -> Vec<bool> {
    let mut mask = vec![false; cols * rows];
    for &(x, y, w, h) in rects {
        if w == 0 || h == 0 || x >= width || y >= height {
            continue;
        }
        let x1 = x.saturating_add(w).min(width);
        let y1 = y.saturating_add(h).min(height);
        let (tx0, tx1) = ((x / TILE_SIZE) as usize, ((x1 - 1) / TILE_SIZE) as usize);
        let (ty0, ty1) = ((y / TILE_SIZE) as usize, ((y1 - 1) / TILE_SIZE) as usize);
        for ty in ty0..=ty1 {
            mask[ty * cols + tx0..=ty * cols + tx1].fill(true);
        }
    }
    mask
}

/// Output of `SmartTileEncoder::encode` or `encode_lossless_refresh`.
pub struct SmartTilePacket {
    pub is_keyframe: bool,
    pub used_lossy: bool,
    pub had_copy_rect: bool,
    pub dirty_tiles: usize,
    pub payload: Vec<u8>,
}

/// Per-window adaptive `SmartTile` encoder.
#[derive(Default)]
pub struct SmartTileEncoder {
    width: u32,
    height: u32,
    /// Bit-exact reconstructed reference framebuffer matching the client decoder (empty when `vram_resident`).
    ref_bgra: Vec<u8>,
    /// Reusable snapshot of `ref_bgra` before the current frame, used only when a frame MV is active.
    prev_ref_bgra: Vec<u8>,
    /// Most recent lossless source framebuffer (used for automatic idle lossless refinement when CPU-resident).
    source_bgra: Vec<u8>,
    /// Persistent VRAM buffers when encoding zero-copy GPU `Dmabuf` textures.
    vram_state: Option<Box<vram::VramWindowState>>,
    /// True when `ref_bgra` and `source_bgra` reside in `vram_state` on the GPU rather than CPU RAM.
    vram_resident: bool,
    tiles: Vec<TileTrack>,
    scratch_packet: Vec<u8>,
    /// Hysteresis latch for 2x2 half-res mode on HiDPI (enter under congestion/full-screen video,
    /// exit when uncongested so normal scrolling stays sharp at full resolution).
    in_half_res: bool,
    /// Hysteresis latch for 4x4 quarter-res congestion mode (enter <= 28, exit >= 40)
    /// to prevent oscillating across the quarter-res <-> half-res bandwidth cliff.
    in_quarter_res: bool,
    /// Consecutive full-screen (>= 80% dirty) frames with no scroll detected;
    /// throttles scroll probing to 1-in-8 frames during full-screen video playback.
    no_scroll_streak: u8,
}

impl SmartTileEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if any tile is currently displaying lossy DCT pixels and awaits idle refresh.
    pub fn has_pending_lossless_refresh(&self) -> bool {
        self.tiles.iter().any(|t| t.is_lossy)
    }

    /// Lossy tiles left unchanged for at least `SETTLED_REFINE_DELAY`, eligible for lossless
    /// refinement inside a regular frame. Must be called after this frame's dirty scan so that
    /// tiles changed in this frame have a fresh `last_change` and are excluded.
    fn settled_refine_candidates(
        tiles: &[TileTrack],
        cols: usize,
        rows: usize,
        quality: u8,
        now: Instant,
    ) -> Vec<(usize, usize, usize)> {
        let limit = if quality <= 40 {
            1
        } else {
            SETTLED_REFINE_MAX_TILES
        };
        let mut out = Vec::new();
        for ty in 0..rows {
            for tx in 0..cols {
                let tidx = ty * cols + tx;
                let t = &tiles[tidx];
                if !t.is_lossy {
                    continue;
                }
                let settled = t
                    .last_change
                    .is_none_or(|lc| now.duration_since(lc) >= SETTLED_REFINE_DELAY);
                if settled {
                    out.push((tx, ty, tidx));
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    }

    /// Rewrite the `u32` tile count in `scratch_packet` (located after any frame MV).
    fn patch_tile_count(&mut self, count: usize) {
        let mut off = 4;
        if (self.scratch_packet[1] & FLAG_FRAME_MV) != 0 {
            off += 4;
        }
        self.scratch_packet[off..off + 4].copy_from_slice(&(count as u32).to_le_bytes());
    }

    /// Encode a new `BGRA` frame at the given DCT quality (`1..=100`).
    pub fn encode(&mut self, width: u32, height: u32, bgra: &[u8], quality: u8) -> SmartTilePacket {
        self.encode_with_damage(width, height, bgra, quality, None)
    }

    /// Like [`Self::encode`], given which pixels changed since the previous frame passed to this
    /// encoder: `damage` rectangles `(x, y, w, h)`, or `None` for anywhere. Tiles outside the
    /// damage are taken as unchanged without comparing them, so damage that misses a change
    /// leaves the client showing stale pixels until the tile is damaged again.
    pub fn encode_with_damage(
        &mut self,
        width: u32,
        height: u32,
        bgra: &[u8],
        quality: u8,
        damage: Option<&[(u32, u32, u32, u32)]>,
    ) -> SmartTilePacket {
        let expected_len = (width as usize) * (height as usize) * 4;
        assert_eq!(bgra.len(), expected_len, "BGRA buffer size mismatch");

        let cols = width.div_ceil(TILE_SIZE) as usize;
        let rows = height.div_ceil(TILE_SIZE) as usize;
        let total_tiles = cols * rows;
        let now = Instant::now();

        if self.vram_resident && width == self.width && height == self.height {
            if let Some(vs) = self.vram_state.as_ref() {
                if let Some(dl_ref) = vs.download_ref_bgra() {
                    self.ref_bgra = dl_ref;
                }
                if self.has_pending_lossless_refresh() {
                    if let Some(dl_cur) = vs.download_cur_bgra() {
                        self.source_bgra = dl_cur;
                    }
                }
            }
            self.vram_resident = false;
        }

        let is_keyframe = width != self.width
            || height != self.height
            || self.ref_bgra.len() != expected_len
            || self.tiles.len() != total_tiles;

        if is_keyframe {
            self.width = width;
            self.height = height;
            self.vram_resident = false;
            self.ref_bgra.resize(expected_len, 0);
            self.prev_ref_bgra.clear();
            self.source_bgra.clear();
            self.tiles.clear();
            self.tiles.resize(total_tiles, TileTrack::default());
            self.in_half_res = false;
            self.in_quarter_res = false;
            self.no_scroll_streak = 0;
        }

        // 1. Identify dirty 64x64 tiles and update per-tile motion streaks.
        let source_valid = !is_keyframe && self.source_bgra.len() == expected_len;
        let damaged = damage
            .filter(|_| !is_keyframe)
            .map(|rects| damaged_tile_mask(rects, width, height, cols, rows));
        let mut dirty_tiles = Vec::new();
        for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (height - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let x0 = (tx as u32) * TILE_SIZE;
                let tw = (width - x0).min(TILE_SIZE);
                let tidx = ty * cols + tx;

                // A lossy tile's reference holds a DCT approximation that never equals the source,
                // so compare it against the last lossless source frame instead; otherwise it would be
                // re-encoded (lossy) on every frame for as long as anything else keeps changing.
                let cmp_base = if source_valid && self.tiles[tidx].is_lossy {
                    &self.source_bgra
                } else {
                    &self.ref_bgra
                };
                let maybe_changed = damaged.as_ref().is_none_or(|mask| mask[tidx]);
                let dirty = is_keyframe
                    || (maybe_changed && !tile_equals(cmp_base, bgra, width, x0, y0, tw, th));
                if dirty {
                    let tr = &mut self.tiles[tidx];
                    if is_keyframe {
                        tr.motion_streak = 0;
                    } else if let Some(prev_t) = tr.last_change {
                        if now.duration_since(prev_t) <= MOTION_WINDOW {
                            tr.motion_streak = tr.motion_streak.saturating_add(1);
                        } else {
                            tr.motion_streak = 1;
                        }
                    } else {
                        tr.motion_streak = 1;
                    }
                    tr.last_change = Some(now);
                    dirty_tiles.push((tx, ty, x0, y0, tw, th, tidx));
                } else if let Some(prev_t) = self.tiles[tidx].last_change {
                    if now.duration_since(prev_t) > MOTION_WINDOW {
                        self.tiles[tidx].motion_streak = 0;
                    }
                }
            }
        }

        // 2. Detect dominant scroll motion vector `(mv_dx, mv_dy)` for per-tile motion compensation.
        //    Unlike global CopyRect, each tile independently tests `(mv_dx, mv_dy)` against an
        //    unmodified snapshot of the previous reference frame, so stationary UI can never be
        //    pulled along while scrolled tiles get 0-byte lossless MV-Skip or small MV-Inter residuals.
        let can_scroll = !is_keyframe && dirty_tiles.len() >= 12;
        if !can_scroll {
            self.no_scroll_streak = 0;
        }
        let should_probe_scroll =
            can_scroll && (self.no_scroll_streak < 3 || self.no_scroll_streak >= 10);

        let frame_mv: Option<(i16, i16)> = if should_probe_scroll {
            detect_scroll_copy_rect(&self.ref_bgra, bgra, width, height).and_then(|op| {
                let dx = op.dst_x as i32 - op.src_x as i32;
                let dy = op.dst_y as i32 - op.src_y as i32;
                if (dx != 0 || dy != 0)
                    && dx >= i16::MIN as i32
                    && dx <= i16::MAX as i32
                    && dy >= i16::MIN as i32
                    && dy <= i16::MAX as i32
                {
                    Some((dx as i16, dy as i16))
                } else {
                    None
                }
            })
        } else {
            None
        };
        if frame_mv.is_some() {
            self.no_scroll_streak = 0;
        } else if can_scroll {
            self.no_scroll_streak = if self.no_scroll_streak >= 10 {
                3
            } else {
                self.no_scroll_streak + 1
            };
        }
        // Large motion/scroll/video updates immediately enter fast progressive lossy mode.
        let massive_motion = !is_keyframe && dirty_tiles.len() >= (total_tiles / 6).max(12);
        let enable_local_mv = !is_keyframe && frame_mv.is_none() && massive_motion;
        let is_hidpi = width >= 1600 || height >= 1000;
        let full_screen_video =
            !is_keyframe && frame_mv.is_none() && dirty_tiles.len() * 2 >= total_tiles;

        let quality = quality.clamp(20, 95);
        // Congestion-driven half-res (with hysteresis): keep uncongested scrolling at crisp full-res,
        // and drop to half-res on full-screen HiDPI video or moderate congestion (`quality <= 56`).
        if !is_hidpi || !massive_motion {
            self.in_half_res = false;
        } else if quality <= 56 || (full_screen_video && quality <= 68) {
            self.in_half_res = true;
        } else if quality >= 68 && !full_screen_video {
            self.in_half_res = false;
        }
        let prefer_half_res = self.in_half_res;

        // Under sustained Wi-Fi congestion on large HiDPI video/motion frames, drop to 4x4 quarter-res
        // DCT (6 blocks per 64x64 tile) with hysteresis (enter <= 28, exit >= 40) so we don't
        // oscillate across the quarter-res <-> half-res bandwidth cliff every few frames.
        if !is_hidpi || !massive_motion {
            self.in_quarter_res = false;
        } else if quality <= 28 {
            self.in_quarter_res = true;
        } else if quality >= 40 {
            self.in_quarter_res = false;
        }
        let prefer_quarter_res = self.in_quarter_res;
        let luma_q = build_quant_table(&BASE_LUMA_QUANT, quality);
        let chroma_q = build_quant_table(&BASE_CHROMA_QUANT, quality);

        let dirty_tile_count = dirty_tiles.len();
        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        let mut flags = 0u8;
        if is_keyframe {
            flags |= FLAG_KEYFRAME;
        }
        if frame_mv.is_some() {
            flags |= FLAG_FRAME_MV;
        }
        self.scratch_packet.push(flags);
        self.scratch_packet.push(quality);
        self.scratch_packet.push(0); // 0 copy rects
        if let Some((mv_dx, mv_dy)) = frame_mv {
            self.scratch_packet.extend_from_slice(&mv_dx.to_le_bytes());
            self.scratch_packet.extend_from_slice(&mv_dy.to_le_bytes());
        }
        self.scratch_packet
            .extend_from_slice(&(dirty_tile_count as u32).to_le_bytes());

        let num_workers = if dirty_tile_count >= 48 && rows >= 4 {
            std::thread::available_parallelism()
                .map(|n| n.get().min(4))
                .unwrap_or(1)
        } else {
            1
        };

        // Attempt unified hardware GPU compute shader batching when encoding a large batch of full-res tiles.
        let mut source_copied_during_gpu = false;
        let gpu_encoded = if use_gpu()
            && !is_keyframe
            && !prefer_half_res
            && !prefer_quarter_res
            && dirty_tile_count >= gpu::GPU_MIN_BATCH_TILES
        {
            let source_bgra_ref = &mut self.source_bgra;
            let prev_ref_buf = &mut self.prev_ref_bgra;
            gpu::global_gpu_engine().and_then(|engine| {
                encode_tiles_with_gpu(
                    engine,
                    &mut self.ref_bgra,
                    prev_ref_buf,
                    frame_mv.is_some() || enable_local_mv,
                    bgra,
                    &mut self.tiles,
                    width,
                    height,
                    &dirty_tiles,
                    massive_motion,
                    frame_mv,
                    &luma_q,
                    &chroma_q,
                    &mut self.scratch_packet,
                    || {
                        source_bgra_ref.resize(expected_len, 0);
                        source_bgra_ref.copy_from_slice(bgra);
                        source_copied_during_gpu = true;
                    },
                )
            })
        } else {
            None
        };

        let (used_lossy, used_local_mv) = if let Some(res) = gpu_encoded {
            res
        } else {
            let prev_ref_slice: Option<&[u8]> = if frame_mv.is_some() || enable_local_mv {
                self.prev_ref_bgra.resize(expected_len, 0);
                self.prev_ref_bgra.copy_from_slice(&self.ref_bgra);
                Some(self.prev_ref_bgra.as_slice())
            } else {
                None
            };
            if num_workers > 1 {
                // Partition dirty tiles by horizontal tile-row bands so each worker owns
                // a disjoint mutable slice of `self.ref_bgra` and `self.tiles` with zero locks.
                let rows_per_band = rows.div_ceil(num_workers);
                let stride = width as usize * 4;
                let mut band_tiles: Vec<Vec<(usize, usize, u32, u32, u32, u32, usize)>> =
                    vec![Vec::new(); num_workers];
                for item in dirty_tiles {
                    let band = (item.1 / rows_per_band).min(num_workers - 1);
                    band_tiles[band].push(item);
                }

                let mut ref_rem = self.ref_bgra.as_mut_slice();
                let mut tile_rem = self.tiles.as_mut_slice();
                let mut band_slices = Vec::with_capacity(num_workers);
                let mut start_row = 0usize;
                #[allow(clippy::needless_range_loop)]
                for band_idx in 0..num_workers {
                    if start_row >= rows {
                        break;
                    }
                    let end_row = ((band_idx + 1) * rows_per_band).min(rows);
                    let y_offset = ((start_row as u32) * TILE_SIZE).min(height);
                    let y_end = ((end_row as u32) * TILE_SIZE).min(height);
                    let px_rows = (y_end - y_offset) as usize;
                    let byte_len = px_rows * stride;
                    let (ref_chunk, next_ref) = ref_rem.split_at_mut(byte_len);
                    ref_rem = next_ref;

                    let tile_len = (end_row - start_row) * cols;
                    let (tile_chunk, next_tiles) = tile_rem.split_at_mut(tile_len);
                    tile_rem = next_tiles;

                    let cur_start_byte = (y_offset as usize) * stride;
                    let cur_chunk = &bgra[cur_start_byte..cur_start_byte + byte_len];
                    band_slices.push((
                        ref_chunk,
                        cur_chunk,
                        tile_chunk,
                        y_offset,
                        start_row * cols,
                        std::mem::take(&mut band_tiles[band_idx]),
                    ));
                    start_row = end_row;
                }

                let test_overrides = snapshot_test_overrides();
                let mut band_outputs: Vec<(Vec<u8>, bool, bool)> = Vec::with_capacity(num_workers);
                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(num_workers);
                    for (ref_chunk, cur_chunk, tile_chunk, y_offset, tidx_offset, items) in
                        band_slices
                    {
                        handles.push(s.spawn(move || {
                            apply_test_overrides(test_overrides);
                            let mut out = Vec::with_capacity(items.len() * 512);
                            let mut local_lossy = false;
                            let mut local_mv_used = false;
                            encode_tile_batch(
                                ref_chunk,
                                prev_ref_slice,
                                cur_chunk,
                                tile_chunk,
                                width,
                                height,
                                y_offset,
                                tidx_offset,
                                &items,
                                is_keyframe,
                                massive_motion,
                                prefer_half_res,
                                prefer_quarter_res,
                                frame_mv,
                                &luma_q,
                                &chroma_q,
                                &mut out,
                                &mut local_lossy,
                                &mut local_mv_used,
                            );
                            (out, local_lossy, local_mv_used)
                        }));
                    }
                    for h in handles {
                        band_outputs.push(h.join().expect("SmartTile worker panicked"));
                    }
                });

                let mut any_lossy = false;
                let mut any_local_mv = false;
                for (chunk_out, lossy, lmv) in band_outputs {
                    self.scratch_packet.extend_from_slice(&chunk_out);
                    any_lossy |= lossy;
                    any_local_mv |= lmv;
                }
                (any_lossy, any_local_mv)
            } else {
                let mut any_lossy = false;
                let mut any_local_mv = false;
                encode_tile_batch(
                    &mut self.ref_bgra,
                    prev_ref_slice,
                    bgra,
                    &mut self.tiles,
                    width,
                    height,
                    0,
                    0,
                    &dirty_tiles,
                    is_keyframe,
                    massive_motion,
                    prefer_half_res,
                    prefer_quarter_res,
                    frame_mv,
                    &luma_q,
                    &chroma_q,
                    &mut self.scratch_packet,
                    &mut any_lossy,
                    &mut any_local_mv,
                );
                (any_lossy, any_local_mv)
            }
        };

        if used_local_mv {
            self.scratch_packet[1] |= FLAG_PREV_REF;
        }

        // Piggyback lossless refinement of settled lossy tiles (e.g. UI next to a playing video)
        // so they sharpen while other regions keep streaming instead of waiting for full idle.
        if !is_keyframe {
            let refine = Self::settled_refine_candidates(&self.tiles, cols, rows, quality, now);
            if !refine.is_empty() {
                let mut residual_samples =
                    Vec::with_capacity((TILE_SIZE as usize) * (TILE_SIZE as usize) * 4);
                let raw_start = self.scratch_packet.len();
                let mut refined = 0usize;
                for &(tx, ty, tidx) in &refine {
                    if self.scratch_packet.len() - raw_start >= SETTLED_REFINE_MAX_RAW_BYTES {
                        break;
                    }
                    let x0 = (tx as u32) * TILE_SIZE;
                    let y0 = (ty as u32) * TILE_SIZE;
                    let tw = (width - x0).min(TILE_SIZE);
                    let th = (height - y0).min(TILE_SIZE);
                    self.scratch_packet
                        .extend_from_slice(&(tx as u16).to_le_bytes());
                    self.scratch_packet
                        .extend_from_slice(&(ty as u16).to_le_bytes());
                    write_lossless_refinement_tile(
                        &mut self.ref_bgra,
                        bgra,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                        &mut residual_samples,
                        &mut self.scratch_packet,
                    );
                    self.tiles[tidx].is_lossy = false;
                    self.tiles[tidx].motion_streak = 0;
                    refined += 1;
                }
                self.patch_tile_count(dirty_tile_count + refined);
            }
        }

        // Only keep a lossless source copy when at least one tile is lossy and awaits idle refresh.
        // Saves a full 33 MB memcpy on every lossless UI frame at 4K!
        if !source_copied_during_gpu && (used_lossy || self.has_pending_lossless_refresh()) {
            self.source_bgra.resize(expected_len, 0);
            self.source_bgra.copy_from_slice(bgra);
        }

        let payload = compress_packet_adaptive(&self.scratch_packet, used_lossy);
        SmartTilePacket {
            is_keyframe,
            used_lossy,
            had_copy_rect: frame_mv.is_some(),
            dirty_tiles: dirty_tile_count,
            payload,
        }
    }

    /// Encode a GPU-resident `GpuTextureFrame` (`Dmabuf` window) directly in VRAM without copying
    /// full window framebuffers to CPU memory. Falls back to `read_to_cpu_bgra` only if GPU
    /// compute is disabled (`FORCE_DISABLE_GPU`) or VRAM allocation fails.
    pub fn encode_gpu_texture(
        &mut self,
        frame: &vram::GpuTextureFrame,
        quality: u8,
    ) -> Result<SmartTilePacket> {
        let width = frame.width;
        let height = frame.height;
        if width == 0 || height == 0 {
            bail!("empty GpuTextureFrame");
        }
        if !use_gpu() {
            let cpu_bgra = frame
                .read_to_cpu_bgra()
                .context("failed to read GpuTextureFrame for CPU fallback")?;
            return Ok(self.encode(width, height, &cpu_bgra, quality));
        }

        let cols = width.div_ceil(TILE_SIZE) as usize;
        let rows = height.div_ceil(TILE_SIZE) as usize;
        let total_tiles = cols * rows;
        let now = Instant::now();

        let mut is_keyframe =
            width != self.width || height != self.height || self.tiles.len() != total_tiles;

        // `cur_buf` only holds the previous frame's source if the last frame was VRAM-encoded too.
        let cur_holds_prev_source = self.vram_resident;

        if self.vram_state.is_none() {
            self.vram_state = Some(Box::new(vram::VramWindowState::new(Arc::clone(
                &frame.engine,
            ))));
        }

        if !is_keyframe && !self.vram_resident {
            let expected_len = (width as usize) * (height as usize) * 4;
            if self.ref_bgra.len() == expected_len {
                if !self
                    .vram_state
                    .as_mut()
                    .unwrap()
                    .upload_ref_bgra(width, height, &self.ref_bgra)
                {
                    is_keyframe = true;
                }
            } else {
                is_keyframe = true;
            }
        }

        if is_keyframe {
            self.width = width;
            self.height = height;
            self.tiles.clear();
            self.tiles.resize(total_tiles, TileTrack::default());
            self.in_half_res = false;
            self.in_quarter_res = false;
            self.no_scroll_streak = 0;
        }
        // Release any CPU-side full-frame buffers once VRAM-resident.
        if !self.ref_bgra.is_empty() {
            self.ref_bgra = Vec::new();
        }
        if !self.prev_ref_bgra.is_empty() {
            self.prev_ref_bgra = Vec::new();
        }
        if !self.source_bgra.is_empty() {
            self.source_bgra = Vec::new();
        }
        self.vram_resident = true;

        let probe_scroll =
            !is_keyframe && (self.no_scroll_streak < 3 || self.no_scroll_streak >= 10);
        let enable_fast_ui = !is_keyframe && !self.in_half_res && !self.in_quarter_res;

        let Some((scan_words_slice, gpu_scroll_mv, fast_extract_slice)) = self
            .vram_state
            .as_mut()
            .unwrap()
            .ingest_and_scan(frame, is_keyframe, probe_scroll, enable_fast_ui)
        else {
            self.vram_resident = false;
            let cpu_bgra = frame
                .read_to_cpu_bgra()
                .context("failed to read GpuTextureFrame after VRAM scan failure")?;
            return Ok(self.encode(width, height, &cpu_bgra, quality));
        };

        let mut dirty_tiles = Vec::new();
        for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (height - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let x0 = (tx as u32) * TILE_SIZE;
                let tw = (width - x0).min(TILE_SIZE);
                let tidx = ty * cols + tx;
                let flags = scan_words_slice[tidx * 2];
                let solid_px = scan_words_slice[tidx * 2 + 1];
                // Lossy tiles never match their DCT reference; judge them by source changes only.
                let dirty_flag = if cur_holds_prev_source && self.tiles[tidx].is_lossy {
                    vram::SCAN_FLAG_SRC_CHANGED
                } else {
                    vram::SCAN_FLAG_DIRTY
                };
                let dirty = is_keyframe || (flags & dirty_flag) != 0;
                if dirty {
                    let tr = &mut self.tiles[tidx];
                    if is_keyframe {
                        tr.motion_streak = 0;
                    } else if let Some(prev_t) = tr.last_change {
                        if now.duration_since(prev_t) <= MOTION_WINDOW {
                            tr.motion_streak = tr.motion_streak.saturating_add(1);
                        } else {
                            tr.motion_streak = 1;
                        }
                    } else {
                        tr.motion_streak = 1;
                    }
                    tr.last_change = Some(now);
                    dirty_tiles.push((tx, ty, x0, y0, tw, th, tidx, flags, solid_px));
                } else if let Some(prev_t) = self.tiles[tidx].last_change {
                    if now.duration_since(prev_t) > MOTION_WINDOW {
                        self.tiles[tidx].motion_streak = 0;
                    }
                }
            }
        }

        let can_scroll = !is_keyframe && dirty_tiles.len() >= 12;
        if !can_scroll {
            self.no_scroll_streak = 0;
        }
        let frame_mv = if can_scroll && probe_scroll {
            gpu_scroll_mv
        } else {
            None
        };
        if frame_mv.is_some() {
            self.no_scroll_streak = 0;
        } else if can_scroll {
            self.no_scroll_streak = if self.no_scroll_streak >= 10 {
                3
            } else {
                self.no_scroll_streak + 1
            };
        }

        let massive_motion = !is_keyframe && dirty_tiles.len() >= (total_tiles / 6).max(12);
        let enable_local_mv = !is_keyframe && frame_mv.is_none() && massive_motion;
        let is_hidpi = width >= 1600 || height >= 1000;
        let full_screen_video =
            !is_keyframe && frame_mv.is_none() && dirty_tiles.len() * 2 >= total_tiles;

        let quality = quality.clamp(20, 95);
        if !is_hidpi || !massive_motion {
            self.in_half_res = false;
        } else if quality <= 56 || (full_screen_video && quality <= 68) {
            self.in_half_res = true;
        } else if quality >= 68 && !full_screen_video {
            self.in_half_res = false;
        }
        let prefer_half_res = self.in_half_res;

        if !is_hidpi || !massive_motion {
            self.in_quarter_res = false;
        } else if quality <= 28 {
            self.in_quarter_res = true;
        } else if quality >= 40 {
            self.in_quarter_res = false;
        }
        let prefer_quarter_res = self.in_quarter_res;

        let scale: u32 = if prefer_quarter_res {
            4
        } else if prefer_half_res {
            2
        } else {
            1
        };
        let luma_q = build_quant_table(&BASE_LUMA_QUANT, quality);
        let chroma_q = build_quant_table(&BASE_CHROMA_QUANT, quality);

        // Append settled lossy tiles as lossless refinements (forced non-lossy below), so they
        // sharpen while other regions keep the window busy.
        let num_changed = dirty_tiles.len();
        if !is_keyframe {
            for (tx, ty, tidx) in
                Self::settled_refine_candidates(&self.tiles, cols, rows, quality, now)
            {
                let x0 = (tx as u32) * TILE_SIZE;
                let y0 = (ty as u32) * TILE_SIZE;
                let tw = (width - x0).min(TILE_SIZE);
                let th = (height - y0).min(TILE_SIZE);
                let flags = scan_words_slice[tidx * 2];
                let solid_px = scan_words_slice[tidx * 2 + 1];
                dirty_tiles.push((tx, ty, x0, y0, tw, th, tidx, flags, solid_px));
            }
        }

        let dirty_tile_count = dirty_tiles.len();
        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        let mut pkt_flags = 0u8;
        if is_keyframe {
            pkt_flags |= FLAG_KEYFRAME;
        }
        if frame_mv.is_some() {
            pkt_flags |= FLAG_FRAME_MV;
        }
        self.scratch_packet.push(pkt_flags);
        self.scratch_packet.push(quality);
        self.scratch_packet.push(0); // 0 copy rects
        if let Some((mv_dx, mv_dy)) = frame_mv {
            self.scratch_packet.extend_from_slice(&mv_dx.to_le_bytes());
            self.scratch_packet.extend_from_slice(&mv_dy.to_le_bytes());
        }
        self.scratch_packet
            .extend_from_slice(&(dirty_tile_count as u32).to_le_bytes());

        // Single-pass Fast-UI path: when `1..=FAST_UI_MAX_TILES` lossless UI/cursor tiles changed,
        // Pass 1 already extracted their `(cur, ref)` pixels into `fast_extract_slice`, so we skip
        // the second synchronous `execute_encode_pass` + `dev.poll` roundtrip completely and commit
        // `cur -> ref` asynchronously on the GPU queue!
        let can_fast_ui = enable_fast_ui
            && !dirty_tiles.is_empty()
            && dirty_tiles.len() <= vram::FAST_UI_MAX_TILES
            && frame_mv.is_none()
            && !massive_motion
            && dirty_tiles
                .iter()
                .all(|&(_, _, _, _, _, _, tidx, sflags, _)| {
                    let wants_lossy = self.tiles[tidx].motion_streak >= MOTION_STREAK_THRESHOLD;
                    let is_solid = (sflags & vram::SCAN_FLAG_SOLID) != 0;
                    let slot_plus1 = (sflags >> 8) as usize;
                    !wants_lossy
                        && (is_solid
                            || (slot_plus1 > 0 && slot_plus1 * 32768 <= fast_extract_slice.len()))
                });

        if can_fast_ui {
            let mut commit_rects = Vec::with_capacity(dirty_tiles.len());
            for &(tx, ty, x0, y0, tw, th, tidx, sflags, solid_px) in &dirty_tiles {
                self.scratch_packet
                    .extend_from_slice(&(tx as u16).to_le_bytes());
                self.scratch_packet
                    .extend_from_slice(&(ty as u16).to_le_bytes());
                if (sflags & vram::SCAN_FLAG_SOLID) != 0 {
                    self.scratch_packet.push(MODE_LOSSLESS_RAW);
                    let px_bytes = solid_px.to_le_bytes();
                    let count = (tw as usize) * (th as usize);
                    self.scratch_packet.reserve(count * 4);
                    for _ in 0..count {
                        self.scratch_packet.extend_from_slice(&px_bytes);
                    }
                } else {
                    let slot = ((sflags >> 8) as usize) - 1;
                    let ex_base = slot * 32768;
                    let cur_tile = &fast_extract_slice[ex_base..ex_base + 16384];
                    let ref_tile = &fast_extract_slice[ex_base + 16384..ex_base + 32768];
                    write_lossless_tile_extracted(
                        cur_tile,
                        ref_tile,
                        tw,
                        th,
                        false,
                        &mut self.scratch_packet,
                    );
                }
                self.tiles[tidx].is_lossy = false;
                commit_rects.push(vram::TileRect { x0, y0, tw, th });
            }
            self.vram_state
                .as_mut()
                .unwrap()
                .commit_cur_to_ref_tiles(&commit_rects);
            let payload = compress_packet_adaptive(&self.scratch_packet, false);
            return Ok(SmartTilePacket {
                is_keyframe: false,
                used_lossy: false,
                had_copy_rect: false,
                dirty_tiles: dirty_tile_count,
                payload,
            });
        }

        let mut cmds = Vec::with_capacity(dirty_tile_count);
        let mut num_lossy_opaque = 0usize;
        let mut num_extract = 0usize;
        for (i, &(_tx, _ty, x0, y0, tw, th, tidx, sflags, _solid_px)) in
            dirty_tiles.iter().enumerate()
        {
            let wants_lossy = !is_keyframe
                && i < num_changed
                && (self.tiles[tidx].motion_streak >= MOTION_STREAK_THRESHOLD || massive_motion);
            let is_opaque = (sflags & vram::SCAN_FLAG_OPAQUE) != 0;
            let is_solid = (sflags & vram::SCAN_FLAG_SOLID) != 0;

            let mut cflags = 0u32;
            if wants_lossy {
                cflags |= vram::CMD_FLAG_WANTS_LOSSY;
            }
            if is_opaque {
                cflags |= vram::CMD_FLAG_OPAQUE;
            }
            if is_solid {
                cflags |= vram::CMD_FLAG_SOLID;
            }

            let (batch_idx, extract_idx) = if wants_lossy && is_opaque {
                let b = num_lossy_opaque as u32;
                num_lossy_opaque += 1;
                (b, u32::MAX)
            } else if !wants_lossy && is_solid {
                (u32::MAX, u32::MAX)
            } else {
                let e = num_extract as u32;
                num_extract += 1;
                (u32::MAX, e)
            };

            cmds.push(vram::VramTileCmd {
                x0,
                y0,
                tw,
                th,
                flags: cflags,
                batch_idx,
                extract_idx,
                _pad: 0,
            });
        }

        let mut alpha_scatter_rects: Vec<vram::TileRect> = Vec::new();
        let mut alpha_scatter_pixels: Vec<u8> = Vec::new();
        let tiles_slice = &mut self.tiles;
        let out_buf = &mut self.scratch_packet;

        let pass_res = self.vram_state.as_mut().unwrap().execute_encode_pass(
            &cmds,
            num_lossy_opaque,
            num_extract,
            scale,
            frame_mv,
            enable_local_mv,
            &luma_q,
            &chroma_q,
            |mv_res, qblocks_flat, extract_bytes| {
                let mut any_lossy = false;
                let mut any_local_mv = false;
                let mut refine_ref_local: Vec<u8> = Vec::new();
                let mut refine_samples = Vec::new();

                for (cmd_idx, &(tx, ty, x0, y0, tw, th, tidx, _sflags, solid_px)) in
                    dirty_tiles.iter().enumerate()
                {
                    let cmd = cmds[cmd_idx];
                    let mv_word = mv_res[cmd_idx];
                    if (mv_word & vram::MV_RESULT_SKIP) != 0 {
                        out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                        out_buf.extend_from_slice(&(ty as u16).to_le_bytes());
                        out_buf.push(MODE_MV_SKIP);
                        tiles_slice[tidx].is_lossy = false;
                        continue;
                    }

                    let wants_lossy = (cmd.flags & vram::CMD_FLAG_WANTS_LOSSY) != 0;
                    let is_opaque = (cmd.flags & vram::CMD_FLAG_OPAQUE) != 0;
                    let is_solid = (cmd.flags & vram::CMD_FLAG_SOLID) != 0;

                    if wants_lossy && is_opaque {
                        any_lossy = true;
                        tiles_slice[tidx].is_lossy = true;
                        out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                        out_buf.extend_from_slice(&(ty as u16).to_le_bytes());

                        let batch_idx = cmd.batch_idx as usize;
                        let q_tile_base = batch_idx * vram::qblocks_words_per_tile(scale);
                        let use_inter = qblocks_flat[q_tile_base] != 0;
                        let q_blocks_base = q_tile_base + 1;

                        let mv_flag = (mv_word & 0xFF) as u8;
                        let (mode_inter, mode_intra, scaled_w, scaled_h) = match scale {
                            4 => (
                                MODE_DCT_QUARTER_INTER,
                                MODE_DCT_QUARTER_INTRA,
                                (tw as usize).div_ceil(4),
                                (th as usize).div_ceil(4),
                            ),
                            2 => (
                                MODE_DCT_HALF_INTER,
                                MODE_DCT_HALF_INTRA,
                                (tw as usize).div_ceil(2),
                                (th as usize).div_ceil(2),
                            ),
                            _ => (MODE_DCT_INTER, MODE_DCT_INTRA, tw as usize, th as usize),
                        };
                        let luma_bx = scaled_w.div_ceil(8);
                        let luma_by = scaled_h.div_ceil(8);
                        let chroma_bx = scaled_w.div_ceil(2).div_ceil(8);
                        let chroma_by = scaled_h.div_ceil(2).div_ceil(8);
                        let num_luma = luma_bx * luma_by;
                        let num_chroma = chroma_bx * chroma_by;

                        if use_inter {
                            out_buf.push(mode_inter | mv_flag);
                            if mv_flag == MODE_FLAG_LOCAL_MV {
                                let ldx = ((mv_word >> 8) & 0xFF) as u8;
                                let ldy = ((mv_word >> 16) & 0xFF) as u8;
                                out_buf.push(ldx);
                                out_buf.push(ldy);
                                any_local_mv = true;
                            }
                        } else {
                            out_buf.push(mode_intra);
                        }

                        let luma_end = q_blocks_base + num_luma * 32;
                        let cb_end = luma_end + num_chroma * 32;
                        let cr_end = cb_end + num_chroma * 32;
                        encode_plane_packed(
                            &qblocks_flat[q_blocks_base..luma_end],
                            luma_bx,
                            luma_by,
                            out_buf,
                        );
                        encode_plane_packed(
                            &qblocks_flat[luma_end..cb_end],
                            chroma_bx,
                            chroma_by,
                            out_buf,
                        );
                        encode_plane_packed(
                            &qblocks_flat[cb_end..cr_end],
                            chroma_bx,
                            chroma_by,
                            out_buf,
                        );
                    } else if !wants_lossy && is_solid {
                        out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                        out_buf.extend_from_slice(&(ty as u16).to_le_bytes());
                        out_buf.push(MODE_LOSSLESS_RAW);
                        let px_bytes = solid_px.to_le_bytes();
                        let count = (tw as usize) * (th as usize);
                        out_buf.reserve(count * 4);
                        for _ in 0..count {
                            out_buf.extend_from_slice(&px_bytes);
                        }
                        tiles_slice[tidx].is_lossy = false;
                    } else if !wants_lossy {
                        out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                        out_buf.extend_from_slice(&(ty as u16).to_le_bytes());
                        let ex_base = (cmd.extract_idx as usize) * 32768;
                        let cur_tile = &extract_bytes[ex_base..ex_base + 16384];
                        let ref_tile = &extract_bytes[ex_base + 16384..ex_base + 32768];
                        if cmd_idx >= num_changed {
                            // Settled lossy tile: residual against the lossy preview on the client.
                            refine_ref_local.clear();
                            refine_ref_local.extend_from_slice(ref_tile);
                            write_lossless_refinement_tile(
                                &mut refine_ref_local,
                                cur_tile,
                                64,
                                0,
                                0,
                                tw,
                                th,
                                &mut refine_samples,
                                out_buf,
                            );
                            tiles_slice[tidx].motion_streak = 0;
                        } else {
                            write_lossless_tile_extracted(
                                cur_tile,
                                ref_tile,
                                tw,
                                th,
                                is_keyframe,
                                out_buf,
                            );
                        }
                        tiles_slice[tidx].is_lossy = false;
                    } else {
                        // Semi-transparent lossy alpha tile: encode on 64x64 local buffer with zero-MV
                        let ex_base = (cmd.extract_idx as usize) * 32768;
                        let cur_tile = &extract_bytes[ex_base..ex_base + 16384];
                        let mut ref_local = [0u8; 16384];
                        ref_local.copy_from_slice(&extract_bytes[ex_base + 16384..ex_base + 32768]);
                        let mut dummy_track = [tiles_slice[tidx]];
                        let mut local_lossy = false;
                        let mut local_mv_unused = false;
                        encode_tile_batch(
                            &mut ref_local,
                            None,
                            cur_tile,
                            &mut dummy_track,
                            64,
                            64,
                            0,
                            0,
                            &[(tx, ty, 0, 0, tw, th, 0)],
                            false,
                            massive_motion,
                            prefer_half_res,
                            prefer_quarter_res,
                            None,
                            &luma_q,
                            &chroma_q,
                            out_buf,
                            &mut local_lossy,
                            &mut local_mv_unused,
                        );
                        any_lossy |= local_lossy;
                        tiles_slice[tidx].is_lossy = dummy_track[0].is_lossy;
                        alpha_scatter_rects.push(vram::TileRect { x0, y0, tw, th });
                        alpha_scatter_pixels.extend_from_slice(&ref_local);
                    }
                }
                (any_lossy, any_local_mv)
            },
        );

        let (used_lossy, used_local_mv) = pass_res.context("VRAM SmartTile encode pass failed")?;

        if !alpha_scatter_rects.is_empty() {
            self.vram_state
                .as_mut()
                .unwrap()
                .scatter_recon_tiles(&alpha_scatter_rects, &alpha_scatter_pixels);
        }

        if used_local_mv {
            self.scratch_packet[1] |= FLAG_PREV_REF;
        }

        let payload = compress_packet_adaptive(&self.scratch_packet, used_lossy);
        Ok(SmartTilePacket {
            is_keyframe,
            used_lossy,
            had_copy_rect: frame_mv.is_some(),
            dirty_tiles: dirty_tile_count,
            payload,
        })
    }

    /// Encode a progressive lossless refinement chunk for up to `max_tiles` lossy tiles,
    /// stopping early once the compressed packet approaches `max_compressed_bytes`.
    /// Returns `None` if all tiles are already lossless.
    pub fn encode_lossless_refresh(
        &mut self,
        max_tiles: usize,
        max_compressed_bytes: usize,
    ) -> Option<SmartTilePacket> {
        if self.width == 0 || self.height == 0 || !self.has_pending_lossless_refresh() {
            return None;
        }
        let tile_limit = max_tiles.max(1);
        let byte_budget = max_compressed_bytes.max(16 * 1024);
        let cols = self.width.div_ceil(TILE_SIZE) as usize;
        let rows = self.height.div_ceil(TILE_SIZE) as usize;

        if self.vram_resident {
            let vram_state = self.vram_state.as_mut()?;
            let mut candidates = Vec::with_capacity(tile_limit);
            let mut rects = Vec::with_capacity(tile_limit);
            'collect: for ty in 0..rows {
                let y0 = (ty as u32) * TILE_SIZE;
                let th = (self.height - y0).min(TILE_SIZE);
                for tx in 0..cols {
                    let tidx = ty * cols + tx;
                    if !self.tiles[tidx].is_lossy {
                        continue;
                    }
                    let x0 = (tx as u32) * TILE_SIZE;
                    let tw = (self.width - x0).min(TILE_SIZE);
                    candidates.push((tx, ty, tw, th, tidx));
                    rects.push(vram::TileRect { x0, y0, tw, th });
                    if candidates.len() >= tile_limit {
                        break 'collect;
                    }
                }
            }
            if candidates.is_empty() {
                return None;
            }

            self.scratch_packet.clear();
            self.scratch_packet.push(PACKET_VERSION);
            self.scratch_packet.push(0);
            self.scratch_packet.push(80);
            self.scratch_packet.push(0);
            self.scratch_packet.extend_from_slice(&0u32.to_le_bytes());

            let scratch_packet = &mut self.scratch_packet;
            let tiles_slice = &mut self.tiles;
            let mut refresh_count = 0usize;
            let mut ready_payload: Option<Vec<u8>> = None;

            vram_state.gather_refresh_tiles(&rects, |extract_bytes| {
                let mut next_check_raw_len = byte_budget;
                let mut residual_samples =
                    Vec::with_capacity((TILE_SIZE as usize) * (TILE_SIZE as usize) * 4);
                let mut ref_local = [0u8; 16384];

                for (idx, &(tx, ty, tw, th, tidx)) in candidates.iter().enumerate() {
                    let ex_base = idx * 32768;
                    let cur_tile = &extract_bytes[ex_base..ex_base + 16384];
                    ref_local.copy_from_slice(&extract_bytes[ex_base + 16384..ex_base + 32768]);

                    scratch_packet.extend_from_slice(&(tx as u16).to_le_bytes());
                    scratch_packet.extend_from_slice(&(ty as u16).to_le_bytes());
                    write_lossless_refinement_tile(
                        &mut ref_local,
                        cur_tile,
                        64,
                        0,
                        0,
                        tw,
                        th,
                        &mut residual_samples,
                        scratch_packet,
                    );
                    tiles_slice[tidx].is_lossy = false;
                    tiles_slice[tidx].motion_streak = 0;
                    refresh_count += 1;

                    if scratch_packet.len() >= next_check_raw_len {
                        scratch_packet[4..8].copy_from_slice(&(refresh_count as u32).to_le_bytes());
                        let compressed = lz4_flex::compress_prepend_size(scratch_packet);
                        if compressed.len() >= byte_budget {
                            let final_pkt = rans::compress_lz4_body_with_rans(
                                scratch_packet.len(),
                                &compressed,
                            )
                            .unwrap_or(compressed);
                            ready_payload = Some(final_pkt);
                            break;
                        }
                        let remaining_comp = byte_budget - compressed.len();
                        let raw_per_comp =
                            (scratch_packet.len() / compressed.len().max(1)).clamp(1, 32);
                        let extra_raw = (remaining_comp * raw_per_comp).max(16 * 1024);
                        next_check_raw_len = scratch_packet.len() + extra_raw;
                    }
                }
            })?;

            if refresh_count == 0 {
                return None;
            }
            vram_state.commit_cur_to_ref_tiles(&rects[..refresh_count]);
            if !self.has_pending_lossless_refresh() {
                self.in_half_res = false;
                self.in_quarter_res = false;
                self.no_scroll_streak = 0;
            }
            let payload = match ready_payload {
                Some(p) => p,
                None => {
                    self.scratch_packet[4..8]
                        .copy_from_slice(&(refresh_count as u32).to_le_bytes());
                    compress_packet_adaptive(&self.scratch_packet, false)
                }
            };
            return Some(SmartTilePacket {
                is_keyframe: false,
                used_lossy: false,
                had_copy_rect: false,
                dirty_tiles: refresh_count,
                payload,
            });
        }

        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        self.scratch_packet.push(0); // not keyframe
        self.scratch_packet.push(80);
        self.scratch_packet.push(0); // 0 copy rects
        self.scratch_packet.extend_from_slice(&0u32.to_le_bytes()); // patched once tile count is known

        let mut refresh_count = 0usize;
        let mut next_check_raw_len = byte_budget;
        let mut ready_payload: Option<Vec<u8>> = None;
        let mut residual_samples =
            Vec::with_capacity((TILE_SIZE as usize) * (TILE_SIZE as usize) * 4);

        'outer: for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (self.height - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let tidx = ty * cols + tx;
                if !self.tiles[tidx].is_lossy {
                    continue;
                }
                let x0 = (tx as u32) * TILE_SIZE;
                let tw = (self.width - x0).min(TILE_SIZE);
                self.scratch_packet
                    .extend_from_slice(&(tx as u16).to_le_bytes());
                self.scratch_packet
                    .extend_from_slice(&(ty as u16).to_le_bytes());
                write_lossless_refinement_tile(
                    &mut self.ref_bgra,
                    &self.source_bgra,
                    self.width,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut residual_samples,
                    &mut self.scratch_packet,
                );
                self.tiles[tidx].is_lossy = false;
                self.tiles[tidx].motion_streak = 0;
                refresh_count += 1;

                if refresh_count >= tile_limit {
                    break 'outer;
                }
                if self.scratch_packet.len() >= next_check_raw_len {
                    self.scratch_packet[4..8]
                        .copy_from_slice(&(refresh_count as u32).to_le_bytes());
                    let compressed = lz4_flex::compress_prepend_size(&self.scratch_packet);
                    if compressed.len() >= byte_budget {
                        let final_pkt = rans::compress_lz4_body_with_rans(
                            self.scratch_packet.len(),
                            &compressed,
                        )
                        .unwrap_or(compressed);
                        ready_payload = Some(final_pkt);
                        break 'outer;
                    }
                    // Estimate how many additional raw bytes fit within the remaining compressed budget.
                    let remaining_comp = byte_budget - compressed.len();
                    let raw_per_comp =
                        (self.scratch_packet.len() / compressed.len().max(1)).clamp(1, 32);
                    let extra_raw = (remaining_comp * raw_per_comp).max(16 * 1024);
                    next_check_raw_len = self.scratch_packet.len() + extra_raw;
                }
            }
        }
        if refresh_count == 0 {
            return None;
        }

        if !self.has_pending_lossless_refresh() {
            self.in_half_res = false;
            self.in_quarter_res = false;
            self.no_scroll_streak = 0;
        }

        let payload = match ready_payload {
            Some(p) => p,
            None => {
                self.scratch_packet[4..8].copy_from_slice(&(refresh_count as u32).to_le_bytes());
                compress_packet_adaptive(&self.scratch_packet, false)
            }
        };
        Some(SmartTilePacket {
            is_keyframe: false,
            used_lossy: false,
            had_copy_rect: false,
            dirty_tiles: refresh_count,
            payload,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_tiles_with_gpu(
    engine: &gpu::SmartTileGpuEngine,
    ref_bgra: &mut [u8],
    prev_ref_buf: &mut Vec<u8>,
    has_prev_ref: bool,
    cur_bgra: &[u8],
    tiles: &mut [TileTrack],
    fb_w: u32,
    fb_h: u32,
    items: &[(usize, usize, u32, u32, u32, u32, usize)],
    massive_motion: bool,
    frame_mv: Option<(i16, i16)>,
    luma_q: &[i32; 64],
    chroma_q: &[i32; 64],
    out: &mut Vec<u8>,
    on_submitted: impl FnOnce(),
) -> Option<(bool, bool)> {
    #[allow(clippy::large_enum_variant)]
    enum TilePlan {
        MvSkip {
            tx: usize,
            ty: usize,
            x0: u32,
            y0: u32,
            tw: u32,
            th: u32,
            tidx: usize,
            sx: u32,
            sy: u32,
        },
        GpuOpaqueLossy {
            tx: usize,
            ty: usize,
            x0: u32,
            y0: u32,
            tw: u32,
            th: u32,
            tidx: usize,
            batch_idx: usize,
        },
        CpuFallback {
            item: (usize, usize, u32, u32, u32, u32, usize),
        },
    }

    let mut plans = Vec::with_capacity(items.len());
    let mut gpu_descs = Vec::with_capacity(items.len());
    let mut needs_prev_copy_on_cpu = false;
    let rows_per_band = if items.len() >= 48 && fb_h.div_ceil(TILE_SIZE) >= 4 {
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(1);
        (fb_h.div_ceil(TILE_SIZE) as usize).div_ceil(num_workers)
    } else {
        usize::MAX
    };

    for &item in items {
        let (tx, ty, x0, y0, tw, th, tidx) = item;
        let mv_src: Option<(u32, u32)> = match (has_prev_ref, frame_mv) {
            (true, Some((dx, dy))) => {
                let sx = x0 as i32 - dx as i32;
                let sy = y0 as i32 - dy as i32;
                if sx >= 0 && sy >= 0 && (sx as u32) + tw <= fb_w && (sy as u32) + th <= fb_h {
                    Some((sx as u32, sy as u32))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some((sx, sy)) = mv_src {
            if tile_equals_shifted(ref_bgra, cur_bgra, fb_w, sx, sy, x0, y0, tw, th) {
                needs_prev_copy_on_cpu = true;
                plans.push(TilePlan::MvSkip {
                    tx,
                    ty,
                    x0,
                    y0,
                    tw,
                    th,
                    tidx,
                    sx,
                    sy,
                });
                continue;
            }
        }

        let wants_lossy = tiles[tidx].motion_streak >= MOTION_STREAK_THRESHOLD || massive_motion;
        if !wants_lossy || !tile_is_opaque(cur_bgra, fb_w, x0, y0, tw, th) {
            if has_prev_ref {
                needs_prev_copy_on_cpu = true;
            }
            plans.push(TilePlan::CpuFallback { item });
            continue;
        }

        // Perform 2D motion estimation across up to 4 parallel bands when `gpu_descs.len() >= 48`.
        // `pred_x` and `pred_y` are in half-pixel (1/2-pel) units (`2 * sx + hx`, `2 * sy + hy`).
        let batch_idx = gpu_descs.len();
        gpu_descs.push(gpu::GpuTileDesc {
            x0,
            y0,
            tw,
            th,
            pred_x: x0 * 2,
            pred_y: y0 * 2,
            has_alpha: 0,
            use_inter: 1,
        });
        plans.push(TilePlan::GpuOpaqueLossy {
            tx,
            ty,
            x0,
            y0,
            tw,
            th,
            tidx,
            batch_idx,
        });
    }

    if gpu_descs.len() < gpu::GPU_MIN_BATCH_TILES {
        return None;
    }

    if needs_prev_copy_on_cpu {
        prev_ref_buf.resize(ref_bgra.len(), 0);
        prev_ref_buf.copy_from_slice(ref_bgra);
    }
    let prev_ref_full: Option<&[u8]> = if needs_prev_copy_on_cpu {
        Some(prev_ref_buf.as_slice())
    } else {
        None
    };

    let prepare_descs =
        |ref_slice: &[u8], descs: &mut [gpu::GpuTileDesc]| -> Vec<(u8, Option<(i8, i8)>)> {
            let eval_mv_range = |slice: &mut [gpu::GpuTileDesc]| {
                let mut last_mv: (i8, i8) = (0, 0);
                let mut cur_b = usize::MAX;
                for desc in slice.iter_mut() {
                    let ty = (desc.y0 / TILE_SIZE) as usize;
                    let b = ty / rows_per_band;
                    if b != cur_b {
                        cur_b = b;
                        last_mv = (0, 0);
                    }
                    if let (true, Some((dx, dy))) = (has_prev_ref, frame_mv) {
                        let sx = desc.x0 as i32 - dx as i32;
                        let sy = desc.y0 as i32 - dy as i32;
                        if sx >= 0
                            && sy >= 0
                            && (sx as u32) + desc.tw <= fb_w
                            && (sy as u32) + desc.th <= fb_h
                        {
                            let mv_sad = tile_bgra_sad(
                                ref_slice, cur_bgra, fb_w, sx as u32, sy as u32, desc.x0, desc.y0,
                                desc.tw, desc.th,
                            );
                            let zero_sad = tile_bgra_sad(
                                ref_slice, cur_bgra, fb_w, desc.x0, desc.y0, desc.x0, desc.y0,
                                desc.tw, desc.th,
                            );
                            if mv_sad < zero_sad {
                                desc.pred_x = (sx as u32) * 2;
                                desc.pred_y = (sy as u32) * 2;
                                desc.has_alpha = MODE_FLAG_MV as u32;
                            }
                        }
                    } else if has_prev_ref {
                        if let Some((ldx, ldy, sx, sy)) = search_local_mv_2d(
                            ref_slice, cur_bgra, fb_w, fb_h, desc.x0, desc.y0, desc.y0, desc.tw,
                            desc.th, last_mv,
                        ) {
                            last_mv = (ldx, ldy);
                            desc.pred_x = sx;
                            desc.pred_y = sy;
                            desc.has_alpha = (MODE_FLAG_LOCAL_MV as u32)
                                | ((ldx as u8 as u32) << 8)
                                | ((ldy as u8 as u32) << 16);
                        }
                    }
                }
            };

            if rows_per_band < usize::MAX && descs.len() >= 48 {
                // Split `descs` at exact band boundaries so each worker computes identical `last_local_mv` chains.
                let mut band_splits = Vec::with_capacity(4);
                let mut rem = &mut descs[..];
                let mut next_band = 1usize;
                while !rem.is_empty() && next_band < 4 {
                    let split_pos = rem
                        .iter()
                        .position(|d| ((d.y0 / TILE_SIZE) as usize) >= next_band * rows_per_band);
                    if let Some(pos) = split_pos {
                        let (head, tail) = rem.split_at_mut(pos);
                        if !head.is_empty() {
                            band_splits.push(head);
                        }
                        rem = tail;
                    }
                    next_band += 1;
                }
                if !rem.is_empty() {
                    band_splits.push(rem);
                }
                let test_overrides = snapshot_test_overrides();
                std::thread::scope(|s| {
                    for chunk in band_splits {
                        s.spawn(move || {
                            apply_test_overrides(test_overrides);
                            eval_mv_range(chunk);
                        });
                    }
                });
            } else {
                eval_mv_range(descs);
            }

            let mut mv_info = Vec::with_capacity(descs.len());
            for d in descs.iter_mut() {
                let packed = d.has_alpha;
                d.has_alpha = 0;
                let flag = (packed & 0xFF) as u8;
                let lmv = if flag == MODE_FLAG_LOCAL_MV {
                    let ldx = ((packed >> 8) & 0xFF) as u8 as i8;
                    let ldy = ((packed >> 16) & 0xFF) as u8 as i8;
                    Some((ldx, ldy))
                } else {
                    None
                };
                mv_info.push((flag, lmv));
            }
            mv_info
        };

    let stride = fb_w as usize * 4;
    let total_rows = fb_h.div_ceil(TILE_SIZE) as usize;
    let cols = fb_w.div_ceil(TILE_SIZE) as usize;
    let mut used_lossy = false;
    let mut used_local_mv = false;

    let apply_plan_range = |plan_slice: &[TilePlan],
                            batch_mv_info: &[(u8, Option<(i8, i8)>)],
                            ref_chunk: &mut [u8],
                            cur_chunk: &[u8],
                            tile_chunk: &mut [TileTrack],
                            y_offset: u32,
                            tidx_offset: usize,
                            qblocks_flat: &[i32],
                            recon_flat: &[u8],
                            out_buf: &mut Vec<u8>,
                            local_lossy: &mut bool,
                            local_mv_used: &mut bool| {
        let unpack_blk = |words: &[i32]| -> [i32; 64] {
            let mut blk = [0i32; 64];
            for i in 0..32 {
                let w = words[i] as u32;
                blk[i * 2] = (w & 0xFFFF) as u16 as i16 as i32;
                blk[i * 2 + 1] = (w >> 16) as u16 as i16 as i32;
            }
            blk
        };

        for plan in plan_slice {
            match *plan {
                TilePlan::MvSkip {
                    tx,
                    ty,
                    x0,
                    y0,
                    tw,
                    th,
                    tidx,
                    sx,
                    sy,
                } => {
                    out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                    out_buf.extend_from_slice(&(ty as u16).to_le_bytes());
                    out_buf.push(MODE_MV_SKIP);
                    copy_tile_from_prev(
                        prev_ref_full.unwrap(),
                        ref_chunk,
                        fb_w,
                        sx,
                        sy,
                        x0,
                        y0 - y_offset,
                        tw,
                        th,
                    );
                    tile_chunk[tidx - tidx_offset].is_lossy = false;
                }
                TilePlan::GpuOpaqueLossy {
                    tx,
                    ty,
                    x0,
                    y0,
                    tw,
                    th,
                    tidx,
                    batch_idx,
                    ..
                } => {
                    *local_lossy = true;
                    out_buf.extend_from_slice(&(tx as u16).to_le_bytes());
                    out_buf.extend_from_slice(&(ty as u16).to_le_bytes());

                    let (mv_flag, local_mv) = batch_mv_info[batch_idx];
                    let q_tile_base = batch_idx * 3073;
                    let use_inter = qblocks_flat[q_tile_base] != 0;
                    let q_blocks_base = q_tile_base + 1;

                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    let num_luma = luma_bx * luma_by;
                    let num_chroma = chroma_bx * chroma_by;

                    if use_inter {
                        out_buf.push(MODE_DCT_INTER | mv_flag);
                        if let Some((ldx, ldy)) = local_mv {
                            out_buf.push(ldx as u8);
                            out_buf.push(ldy as u8);
                            *local_mv_used = true;
                        }
                    } else {
                        out_buf.push(MODE_DCT_INTRA);
                    }

                    let mut seq_blk = 0usize;
                    let mut plane_blks = [[0i32; 64]; 64];
                    for blk_dst in plane_blks.iter_mut().take(num_luma) {
                        let s = q_blocks_base + seq_blk * 32;
                        *blk_dst = unpack_blk(&qblocks_flat[s..s + 32]);
                        seq_blk += 1;
                    }
                    encode_plane_qblocks(&mut plane_blks[..num_luma], luma_bx, luma_by, out_buf);

                    for blk_dst in plane_blks.iter_mut().take(num_chroma) {
                        let s = q_blocks_base + seq_blk * 32;
                        *blk_dst = unpack_blk(&qblocks_flat[s..s + 32]);
                        seq_blk += 1;
                    }
                    encode_plane_qblocks(
                        &mut plane_blks[..num_chroma],
                        chroma_bx,
                        chroma_by,
                        out_buf,
                    );

                    for blk_dst in plane_blks.iter_mut().take(num_chroma) {
                        let s = q_blocks_base + seq_blk * 32;
                        *blk_dst = unpack_blk(&qblocks_flat[s..s + 32]);
                        seq_blk += 1;
                    }
                    encode_plane_qblocks(
                        &mut plane_blks[..num_chroma],
                        chroma_bx,
                        chroma_by,
                        out_buf,
                    );

                    // Scatter reconstructed 64x64 BGRA tile back into `ref_chunk`.
                    let tile_byte_base = batch_idx * 4096 * 4;
                    let row_bytes = tw as usize * 4;
                    let rel_y0 = (y0 - y_offset) as usize;
                    for py in 0..(th as usize) {
                        let src_off = tile_byte_base + py * 64 * 4;
                        let dst_off = (rel_y0 + py) * stride + (x0 as usize) * 4;
                        ref_chunk[dst_off..dst_off + row_bytes]
                            .copy_from_slice(&recon_flat[src_off..src_off + row_bytes]);
                    }
                    tile_chunk[tidx - tidx_offset].is_lossy = true;
                }
                TilePlan::CpuFallback { item } => {
                    encode_tile_batch(
                        ref_chunk,
                        prev_ref_full,
                        cur_chunk,
                        tile_chunk,
                        fb_w,
                        fb_h,
                        y_offset,
                        tidx_offset,
                        std::slice::from_ref(&item),
                        false,
                        massive_motion,
                        false,
                        false,
                        frame_mv,
                        luma_q,
                        chroma_q,
                        out_buf,
                        local_lossy,
                        local_mv_used,
                    );
                }
            }
        }
    };

    let plan_ty = |p: &TilePlan| -> usize {
        match *p {
            TilePlan::MvSkip { ty, .. } => ty,
            TilePlan::GpuOpaqueLossy { ty, .. } => ty,
            TilePlan::CpuFallback { item } => item.1,
        }
    };

    engine.encode_fullres_tiles_batch_with(
        cur_bgra,
        ref_bgra,
        prev_ref_full,
        fb_w,
        fb_h,
        &mut gpu_descs,
        luma_q,
        chroma_q,
        prepare_descs,
        on_submitted,
        |ref_bgra, batch_mv_info, qblocks_flat, recon_flat| {
            let mv_info_ref = batch_mv_info.as_slice();
            if rows_per_band < usize::MAX && plans.len() >= 48 {
                let num_workers = total_rows.div_ceil(rows_per_band).max(1);
                let mut band_slices = Vec::with_capacity(num_workers);
                let mut ref_rem = &mut ref_bgra[..];
                let mut tile_rem = &mut tiles[..];
                let mut plan_rem = &plans[..];
                let mut start_row = 0usize;

                while start_row < total_rows {
                    let end_row = (start_row + rows_per_band).min(total_rows);
                    let y_offset = (start_row as u32) * TILE_SIZE;
                    let y_end = ((end_row as u32) * TILE_SIZE).min(fb_h);
                    let byte_len = ((y_end - y_offset) as usize) * stride;
                    let (ref_chunk, next_ref) = ref_rem.split_at_mut(byte_len);
                    ref_rem = next_ref;

                    let tile_len = (end_row - start_row) * cols;
                    let (tile_chunk, next_tiles) = tile_rem.split_at_mut(tile_len);
                    tile_rem = next_tiles;

                    let split_pos = plan_rem
                        .iter()
                        .position(|p| plan_ty(p) >= end_row)
                        .unwrap_or(plan_rem.len());
                    let (plan_chunk, next_plans) = plan_rem.split_at(split_pos);
                    plan_rem = next_plans;

                    let cur_start_byte = (y_offset as usize) * stride;
                    let cur_chunk = &cur_bgra[cur_start_byte..cur_start_byte + byte_len];
                    band_slices.push((
                        plan_chunk,
                        ref_chunk,
                        cur_chunk,
                        tile_chunk,
                        y_offset,
                        start_row * cols,
                    ));
                    start_row = end_row;
                }

                let test_overrides = snapshot_test_overrides();
                let mut band_outputs = Vec::with_capacity(band_slices.len());
                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(band_slices.len());
                    for (plan_chunk, ref_chunk, cur_chunk, tile_chunk, y_offset, tidx_offset) in
                        band_slices
                    {
                        handles.push(s.spawn(move || {
                            apply_test_overrides(test_overrides);
                            let mut local_out = Vec::with_capacity(plan_chunk.len() * 512);
                            let mut l_lossy = false;
                            let mut l_mv = false;
                            apply_plan_range(
                                plan_chunk,
                                mv_info_ref,
                                ref_chunk,
                                cur_chunk,
                                tile_chunk,
                                y_offset,
                                tidx_offset,
                                qblocks_flat,
                                recon_flat,
                                &mut local_out,
                                &mut l_lossy,
                                &mut l_mv,
                            );
                            (local_out, l_lossy, l_mv)
                        }));
                    }
                    for h in handles {
                        band_outputs.push(h.join().expect("SmartTile RLE worker panicked"));
                    }
                });

                for (chunk_out, l_lossy, l_mv) in band_outputs {
                    out.extend_from_slice(&chunk_out);
                    used_lossy |= l_lossy;
                    used_local_mv |= l_mv;
                }
            } else {
                apply_plan_range(
                    &plans,
                    mv_info_ref,
                    ref_bgra,
                    cur_bgra,
                    tiles,
                    0,
                    0,
                    qblocks_flat,
                    recon_flat,
                    out,
                    &mut used_lossy,
                    &mut used_local_mv,
                );
            }
        },
    )?;

    Some((used_lossy, used_local_mv))
}

#[allow(clippy::too_many_arguments)]
fn encode_tile_batch(
    ref_bgra: &mut [u8],
    prev_ref_full: Option<&[u8]>,
    cur_bgra: &[u8],
    tiles: &mut [TileTrack],
    fb_w: u32,
    fb_h: u32,
    y_offset: u32,
    tidx_offset: usize,
    items: &[(usize, usize, u32, u32, u32, u32, usize)],
    is_keyframe: bool,
    massive_motion: bool,
    prefer_half_res: bool,
    prefer_quarter_res: bool,
    frame_mv: Option<(i16, i16)>,
    luma_q: &[i32; 64],
    chroma_q: &[i32; 64],
    out: &mut Vec<u8>,
    used_lossy: &mut bool,
    used_local_mv: &mut bool,
) {
    let mut cur_y = [0i32; 64 * 64];
    let mut cur_cb = [0i32; 32 * 32];
    let mut cur_cr = [0i32; 32 * 32];
    let mut cur_a = [0i32; 64 * 64];
    let mut ref_y = [0i32; 64 * 64];
    let mut ref_cb = [0i32; 32 * 32];
    let mut ref_cr = [0i32; 32 * 32];
    let mut ref_a = [0i32; 64 * 64];
    let mut recon_y = [0i32; 64 * 64];
    let mut recon_cb = [0i32; 32 * 32];
    let mut recon_cr = [0i32; 32 * 32];
    let mut recon_a = [0i32; 64 * 64];

    let mut cur_hy = [0i32; 32 * 32];
    let mut cur_hcb = [0i32; 16 * 16];
    let mut cur_hcr = [0i32; 16 * 16];
    let mut cur_ha = [0i32; 32 * 32];
    let mut ref_hy = [0i32; 32 * 32];
    let mut ref_hcb = [0i32; 16 * 16];
    let mut ref_hcr = [0i32; 16 * 16];
    let mut ref_ha = [0i32; 32 * 32];
    let mut recon_hy = [0i32; 32 * 32];
    let mut recon_hcb = [0i32; 16 * 16];
    let mut recon_hcr = [0i32; 16 * 16];
    let mut recon_ha = [0i32; 32 * 32];

    let mut cur_qy = [0i32; 16 * 16];
    let mut cur_qcb = [0i32; 8 * 8];
    let mut cur_qcr = [0i32; 8 * 8];
    let mut cur_qa = [0i32; 16 * 16];
    let mut ref_qy = [0i32; 16 * 16];
    let mut ref_qcb = [0i32; 8 * 8];
    let mut ref_qcr = [0i32; 8 * 8];
    let mut ref_qa = [0i32; 16 * 16];
    let mut recon_qy = [0i32; 16 * 16];
    let mut recon_qcb = [0i32; 8 * 8];
    let mut recon_qcr = [0i32; 8 * 8];
    let mut recon_qa = [0i32; 16 * 16];

    let mut last_local_mv: (i8, i8) = (0, 0);

    for &(tx, ty, x0, y0, tw, th, tidx) in items {
        let local_y0 = y0 - y_offset;
        let local_tidx = tidx - tidx_offset;
        out.extend_from_slice(&(tx as u16).to_le_bytes());
        out.extend_from_slice(&(ty as u16).to_le_bytes());

        let mv_src: Option<(u32, u32)> = match (prev_ref_full, frame_mv) {
            (Some(_), Some((dx, dy))) => {
                let sx = x0 as i32 - dx as i32;
                let sy = y0 as i32 - dy as i32;
                if sx >= 0 && sy >= 0 && (sx as u32) + tw <= fb_w && (sy as u32) + th <= fb_h {
                    Some((sx as u32, sy as u32))
                } else {
                    None
                }
            }
            _ => None,
        };

        // Fast exact-match per-tile Motion Skip: 0 pixel bytes, 100% lossless!
        if let (Some(prev_full), Some((sx, sy))) = (prev_ref_full, mv_src) {
            if tile_equals_shifted(prev_full, cur_bgra, fb_w, sx, sy, x0, local_y0, tw, th) {
                out.push(MODE_MV_SKIP);
                copy_tile_from_prev(prev_full, ref_bgra, fb_w, sx, sy, x0, local_y0, tw, th);
                tiles[local_tidx].is_lossy = false;
                continue;
            }
        }

        let wants_lossy = !is_keyframe
            && (tiles[local_tidx].motion_streak >= MOTION_STREAK_THRESHOLD || massive_motion);
        let has_alpha = wants_lossy && !tile_is_opaque(cur_bgra, fb_w, x0, local_y0, tw, th);
        let alpha_flag = if has_alpha { MODE_FLAG_ALPHA } else { 0 };

        // Pick the best temporal predictor:
        // 1. Frame scroll MV `(sx, sy)` if a scroll was detected, else
        // 2. Lightweight per-tile 2D diamond search + 1/2-pel refinement `(local_dx_hpel, local_dy_hpel)`.
        // `pred_x_hpel` and `pred_y_hpel` are in half-pixel units (`2 * sx + hx`, `2 * sy + hy`).
        let (pred_buf, pred_x_hpel, pred_y_hpel, mv_flag, local_mv): (
            &[u8],
            u32,
            u32,
            u8,
            Option<(i8, i8)>,
        ) = if wants_lossy {
            if let (Some(prev_full), Some((sx, sy))) = (prev_ref_full, mv_src) {
                let mv_sad = tile_bgra_sad(prev_full, cur_bgra, fb_w, sx, sy, x0, local_y0, tw, th);
                let zero_sad =
                    tile_bgra_sad(ref_bgra, cur_bgra, fb_w, x0, local_y0, x0, local_y0, tw, th);
                if mv_sad < zero_sad {
                    (prev_full, sx * 2, sy * 2, MODE_FLAG_MV, None)
                } else {
                    (ref_bgra, x0 * 2, local_y0 * 2, 0, None)
                }
            } else if let Some(prev_full) = prev_ref_full {
                if let Some((ldx_h, ldy_h, sx_h, sy_h)) = search_local_mv_2d(
                    prev_full,
                    cur_bgra,
                    fb_w,
                    fb_h,
                    x0,
                    y0,
                    local_y0,
                    tw,
                    th,
                    last_local_mv,
                ) {
                    last_local_mv = (ldx_h, ldy_h);
                    (
                        prev_full,
                        sx_h,
                        sy_h,
                        MODE_FLAG_LOCAL_MV,
                        Some((ldx_h, ldy_h)),
                    )
                } else {
                    (ref_bgra, x0 * 2, local_y0 * 2, 0, None)
                }
            } else {
                (ref_bgra, x0 * 2, local_y0 * 2, 0, None)
            }
        } else {
            (ref_bgra, x0 * 2, local_y0 * 2, 0, None)
        };
        let pred_x = pred_x_hpel >> 1;
        let pred_y = pred_y_hpel >> 1;

        if wants_lossy && prefer_quarter_res {
            *used_lossy = true;
            bgra_tile_to_ycbcr420_quarter(
                cur_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
                &mut cur_qy,
                &mut cur_qcb,
                &mut cur_qcr,
            );
            bgra_tile_to_ycbcr420_quarter(
                pred_buf,
                fb_w,
                pred_x,
                pred_y,
                tw,
                th,
                &mut ref_qy,
                &mut ref_qcb,
                &mut ref_qcr,
            );
            if has_alpha {
                bgra_tile_to_alpha_quarter(cur_bgra, fb_w, x0, local_y0, tw, th, &mut cur_qa);
                bgra_tile_to_alpha_quarter(pred_buf, fb_w, pred_x, pred_y, tw, th, &mut ref_qa);
            }

            let use_inter = should_use_inter_tile_quarter(&cur_qy, &ref_qy);
            let qw = (tw as usize).div_ceil(4);
            let qh = (th as usize).div_ceil(4);
            let luma_bx = qw.div_ceil(8);
            let luma_by = qh.div_ceil(8);
            let chroma_bx = qw.div_ceil(2).div_ceil(8);
            let chroma_by = qh.div_ceil(2).div_ceil(8);

            if use_inter {
                out.push(MODE_DCT_QUARTER_INTER | alpha_flag | mv_flag);
                if let Some((ldx, ldy)) = local_mv {
                    out.push(ldx as u8);
                    out.push(ldy as u8);
                    *used_local_mv = true;
                }
                encode_and_reconstruct_plane(
                    &cur_qy,
                    Some(&ref_qy),
                    &mut recon_qy,
                    16,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_qcb,
                    Some(&ref_qcb),
                    &mut recon_qcb,
                    8,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_qcr,
                    Some(&ref_qcr),
                    &mut recon_qcr,
                    8,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_qa,
                        Some(&ref_qa),
                        &mut recon_qa,
                        16,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            } else {
                out.push(MODE_DCT_QUARTER_INTRA | alpha_flag);
                encode_and_reconstruct_plane(
                    &cur_qy,
                    None,
                    &mut recon_qy,
                    16,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_qcb,
                    None,
                    &mut recon_qcb,
                    8,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_qcr,
                    None,
                    &mut recon_qcr,
                    8,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_qa,
                        None,
                        &mut recon_qa,
                        16,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            }

            ycbcr420_quarter_to_bgra_tile(
                &recon_qy,
                &recon_qcb,
                &recon_qcr,
                if has_alpha { Some(&recon_qa) } else { None },
                ref_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
            );
            tiles[local_tidx].is_lossy = true;
        } else if wants_lossy && prefer_half_res {
            *used_lossy = true;
            bgra_tile_to_ycbcr420_half(
                cur_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
                &mut cur_hy,
                &mut cur_hcb,
                &mut cur_hcr,
            );
            bgra_tile_to_ycbcr420_half(
                pred_buf,
                fb_w,
                pred_x,
                pred_y,
                tw,
                th,
                &mut ref_hy,
                &mut ref_hcb,
                &mut ref_hcr,
            );
            if has_alpha {
                bgra_tile_to_alpha_half(cur_bgra, fb_w, x0, local_y0, tw, th, &mut cur_ha);
                bgra_tile_to_alpha_half(pred_buf, fb_w, pred_x, pred_y, tw, th, &mut ref_ha);
            }

            let use_inter = should_use_inter_tile_half(&cur_hy, &ref_hy);
            let hw = (tw as usize).div_ceil(2);
            let hh = (th as usize).div_ceil(2);
            let luma_bx = hw.div_ceil(8);
            let luma_by = hh.div_ceil(8);
            let chroma_bx = hw.div_ceil(2).div_ceil(8);
            let chroma_by = hh.div_ceil(2).div_ceil(8);

            if use_inter {
                out.push(MODE_DCT_HALF_INTER | alpha_flag | mv_flag);
                if let Some((ldx, ldy)) = local_mv {
                    out.push(ldx as u8);
                    out.push(ldy as u8);
                    *used_local_mv = true;
                }
                encode_and_reconstruct_plane(
                    &cur_hy,
                    Some(&ref_hy),
                    &mut recon_hy,
                    32,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_hcb,
                    Some(&ref_hcb),
                    &mut recon_hcb,
                    16,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_hcr,
                    Some(&ref_hcr),
                    &mut recon_hcr,
                    16,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_ha,
                        Some(&ref_ha),
                        &mut recon_ha,
                        32,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            } else {
                out.push(MODE_DCT_HALF_INTRA | alpha_flag);
                encode_and_reconstruct_plane(
                    &cur_hy,
                    None,
                    &mut recon_hy,
                    32,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_hcb,
                    None,
                    &mut recon_hcb,
                    16,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_hcr,
                    None,
                    &mut recon_hcr,
                    16,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_ha,
                        None,
                        &mut recon_ha,
                        32,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            }

            ycbcr420_half_to_bgra_tile(
                &recon_hy,
                &recon_hcb,
                &recon_hcr,
                if has_alpha { Some(&recon_ha) } else { None },
                ref_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
            );
            tiles[local_tidx].is_lossy = true;
        } else if wants_lossy {
            *used_lossy = true;
            bgra_tile_to_ycbcr420(
                cur_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
                &mut cur_y,
                &mut cur_cb,
                &mut cur_cr,
            );
            bgra_tile_to_ycbcr420_hpel(
                pred_buf,
                fb_w,
                pred_x_hpel,
                pred_y_hpel,
                tw,
                th,
                &mut ref_y,
                &mut ref_cb,
                &mut ref_cr,
            );
            if has_alpha {
                bgra_tile_to_alpha(cur_bgra, fb_w, x0, local_y0, tw, th, &mut cur_a);
                bgra_tile_to_alpha_hpel(
                    pred_buf,
                    fb_w,
                    pred_x_hpel,
                    pred_y_hpel,
                    tw,
                    th,
                    &mut ref_a,
                );
            }

            let use_inter = should_use_inter_tile(&cur_y, &ref_y);
            let luma_bx = (tw as usize).div_ceil(8);
            let luma_by = (th as usize).div_ceil(8);
            let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
            let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);

            if use_inter {
                out.push(MODE_DCT_INTER | alpha_flag | mv_flag);
                if let Some((ldx, ldy)) = local_mv {
                    out.push(ldx as u8);
                    out.push(ldy as u8);
                    *used_local_mv = true;
                }
                encode_and_reconstruct_plane(
                    &cur_y,
                    Some(&ref_y),
                    &mut recon_y,
                    64,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_cb,
                    Some(&ref_cb),
                    &mut recon_cb,
                    32,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_cr,
                    Some(&ref_cr),
                    &mut recon_cr,
                    32,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_a,
                        Some(&ref_a),
                        &mut recon_a,
                        64,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            } else {
                out.push(MODE_DCT_INTRA | alpha_flag);
                encode_and_reconstruct_plane(
                    &cur_y,
                    None,
                    &mut recon_y,
                    64,
                    luma_bx,
                    luma_by,
                    luma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_cb,
                    None,
                    &mut recon_cb,
                    32,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                encode_and_reconstruct_plane(
                    &cur_cr,
                    None,
                    &mut recon_cr,
                    32,
                    chroma_bx,
                    chroma_by,
                    chroma_q,
                    out,
                );
                if has_alpha {
                    encode_and_reconstruct_plane(
                        &cur_a,
                        None,
                        &mut recon_a,
                        64,
                        luma_bx,
                        luma_by,
                        luma_q,
                        out,
                    );
                }
            }

            ycbcr420_to_bgra_tile(
                &recon_y,
                &recon_cb,
                &recon_cr,
                if has_alpha { Some(&recon_a) } else { None },
                ref_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
            );
            tiles[local_tidx].is_lossy = true;
        } else {
            write_lossless_tile(
                ref_bgra,
                cur_bgra,
                fb_w,
                x0,
                local_y0,
                tw,
                th,
                is_keyframe,
                out,
            );
            tiles[local_tidx].is_lossy = false;
        }
    }
}

fn should_use_inter_tile_half(cur_y: &[i32; 32 * 32], ref_y: &[i32; 32 * 32]) -> bool {
    let mut sum_y: i64 = 0;
    let mut inter_sad: i64 = 0;
    for i in (0..1024).step_by(2) {
        sum_y += cur_y[i] as i64;
        inter_sad += (cur_y[i] - ref_y[i]).abs() as i64;
    }
    let mean = (sum_y / 512) as i32;
    let mut intra_sad: i64 = 0;
    for i in (0..1024).step_by(2) {
        intra_sad += (cur_y[i] - mean).abs() as i64;
    }
    inter_sad * 10 <= intra_sad * 9
}

fn should_use_inter_tile_quarter(cur_y: &[i32; 16 * 16], ref_y: &[i32; 16 * 16]) -> bool {
    let mut sum_y: i64 = 0;
    let mut inter_sad: i64 = 0;
    for i in 0..256 {
        sum_y += cur_y[i] as i64;
        inter_sad += (cur_y[i] - ref_y[i]).abs() as i64;
    }
    let mean = (sum_y / 256) as i32;
    let mut intra_sad: i64 = 0;
    for &y in cur_y {
        intra_sad += (y - mean).abs() as i64;
    }
    inter_sad * 10 <= intra_sad * 9
}

fn tile_equals(a: &[u8], b: &[u8], fb_w: u32, x0: u32, y0: u32, tw: u32, th: u32) -> bool {
    tile_equals_shifted(a, b, fb_w, x0, y0, x0, y0, tw, th)
}

#[allow(clippy::too_many_arguments)]
fn tile_equals_shifted(
    src_fb: &[u8],
    dst_fb: &[u8],
    fb_w: u32,
    sx: u32,
    sy: u32,
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                simd::avx2::tile_equals_shifted(src_fb, dst_fb, fb_w, sx, sy, dx, dy, tw, th)
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            return unsafe {
                simd::neon::tile_equals_shifted(src_fb, dst_fb, fb_w, sx, sy, dx, dy, tw, th)
            };
        }
    }
    #[allow(unreachable_code)]
    simd::tile_equals_shifted_scalar(src_fb, dst_fb, fb_w, sx, sy, dx, dy, tw, th)
}

#[allow(clippy::too_many_arguments)]
fn copy_tile_from_prev(
    src_fb: &[u8],
    dst_fb: &mut [u8],
    fb_w: u32,
    sx: u32,
    sy: u32,
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
) {
    let stride = fb_w as usize * 4;
    let row_bytes = tw as usize * 4;
    for py in 0..(th as usize) {
        let s_off = (sy as usize + py) * stride + (sx as usize) * 4;
        let d_off = (dy as usize + py) * stride + (dx as usize) * 4;
        dst_fb[d_off..d_off + row_bytes].copy_from_slice(&src_fb[s_off..s_off + row_bytes]);
    }
}

/// Fast sampled RGB SAD between `src_fb[sx, sy]` and `dst_fb[dx, dy]` (every 4th pixel)
/// to choose between zero-MV and frame-scroll-MV temporal prediction.
#[allow(clippy::too_many_arguments)]
fn tile_bgra_sad(
    src_fb: &[u8],
    dst_fb: &[u8],
    fb_w: u32,
    sx: u32,
    sy: u32,
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
) -> u64 {
    let stride = fb_w as usize * 4;
    let mut sad = 0u64;
    let mut py = 0usize;
    while py < th as usize {
        let s_row = (sy as usize + py) * stride + (sx as usize) * 4;
        let d_row = (dy as usize + py) * stride + (dx as usize) * 4;
        let mut px = 0usize;
        while px < tw as usize {
            let sp = s_row + px * 4;
            let dp = d_row + px * 4;
            sad += (src_fb[sp] as i32 - dst_fb[dp] as i32).unsigned_abs() as u64
                + (src_fb[sp + 1] as i32 - dst_fb[dp + 1] as i32).unsigned_abs() as u64
                + (src_fb[sp + 2] as i32 - dst_fb[dp + 2] as i32).unsigned_abs() as u64;
            px += 4;
        }
        py += 2;
    }
    sad
}

/// Ultra-fast 8x8-grid sampled RGB SAD (64 pixels per 64x64 tile) for 2D diamond motion search.
#[allow(clippy::too_many_arguments)]
fn tile_bgra_sad_fast(
    src_fb: &[u8],
    dst_fb: &[u8],
    fb_w: u32,
    sx: u32,
    sy: u32,
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
) -> u64 {
    let stride = fb_w as usize * 4;
    let mut sad = 0u64;
    let mut py = 4usize.min(th as usize - 1);
    while py < th as usize {
        let s_row = (sy as usize + py) * stride + (sx as usize) * 4;
        let d_row = (dy as usize + py) * stride + (dx as usize) * 4;
        let mut px = 4usize.min(tw as usize - 1);
        while px < tw as usize {
            let sp = s_row + px * 4;
            let dp = d_row + px * 4;
            sad += (src_fb[sp] as i32 - dst_fb[dp] as i32).unsigned_abs() as u64
                + (src_fb[sp + 1] as i32 - dst_fb[dp + 1] as i32).unsigned_abs() as u64
                + (src_fb[sp + 2] as i32 - dst_fb[dp + 2] as i32).unsigned_abs() as u64;
            px += 8;
        }
        py += 8;
    }
    sad
}

/// Sampled 8x8-grid RGB SAD at half-pixel reference coordinates `(x0_hpel, y0_hpel)`
/// (`2 * sx + hx`, `2 * sy + hy`, `hx, hy in {0, 1}`).
#[allow(clippy::too_many_arguments)]
fn tile_bgra_sad_hpel_fast(
    src_fb: &[u8],
    dst_fb: &[u8],
    fb_w: u32,
    x0_hpel: u32,
    y0_hpel: u32,
    dx: u32,
    dy: u32,
    tw: u32,
    th: u32,
) -> u64 {
    let sx = x0_hpel >> 1;
    let sy = y0_hpel >> 1;
    let hx = (x0_hpel & 1) as usize;
    let hy = (y0_hpel & 1) as usize;
    if hx == 0 && hy == 0 {
        return tile_bgra_sad_fast(src_fb, dst_fb, fb_w, sx, sy, dx, dy, tw, th);
    }
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    let mut sad = 0u64;
    let mut py = 4usize.min(max_y);
    while py < th as usize {
        let cy1 = (py + hy).min(max_y);
        let s_row0 = (sy as usize + py) * stride;
        let s_row1 = (sy as usize + cy1) * stride;
        let d_row = (dy as usize + py) * stride + (dx as usize) * 4;
        let mut px = 4usize.min(max_x);
        while px < tw as usize {
            let cx1 = (px + hx).min(max_x);
            let i00 = s_row0 + (sx as usize + px) * 4;
            let i01 = s_row0 + (sx as usize + cx1) * 4;
            let i10 = s_row1 + (sx as usize + px) * 4;
            let i11 = s_row1 + (sx as usize + cx1) * 4;
            let (sb, sg, sr) = match (hx, hy) {
                (1, 0) => (
                    (src_fb[i00] as i32 + src_fb[i01] as i32 + 1) >> 1,
                    (src_fb[i00 + 1] as i32 + src_fb[i01 + 1] as i32 + 1) >> 1,
                    (src_fb[i00 + 2] as i32 + src_fb[i01 + 2] as i32 + 1) >> 1,
                ),
                (0, 1) => (
                    (src_fb[i00] as i32 + src_fb[i10] as i32 + 1) >> 1,
                    (src_fb[i00 + 1] as i32 + src_fb[i10 + 1] as i32 + 1) >> 1,
                    (src_fb[i00 + 2] as i32 + src_fb[i10 + 2] as i32 + 1) >> 1,
                ),
                _ => (
                    (src_fb[i00] as i32
                        + src_fb[i01] as i32
                        + src_fb[i10] as i32
                        + src_fb[i11] as i32
                        + 2)
                        >> 2,
                    (src_fb[i00 + 1] as i32
                        + src_fb[i01 + 1] as i32
                        + src_fb[i10 + 1] as i32
                        + src_fb[i11 + 1] as i32
                        + 2)
                        >> 2,
                    (src_fb[i00 + 2] as i32
                        + src_fb[i01 + 2] as i32
                        + src_fb[i10 + 2] as i32
                        + src_fb[i11 + 2] as i32
                        + 2)
                        >> 2,
                ),
            };
            let dp = d_row + px * 4;
            sad += (sb - dst_fb[dp] as i32).unsigned_abs() as u64
                + (sg - dst_fb[dp + 1] as i32).unsigned_abs() as u64
                + (sr - dst_fb[dp + 2] as i32).unsigned_abs() as u64;
            px += 8;
        }
        py += 8;
    }
    sad
}

/// Lightweight 4-step 8-neighbor Three-Step Search (`±24 px` integer range, refined to `1/2-pel`
/// sub-pixel precision in half-pixel units `[-48, +48]`) seeded from `(0, 0)` and the left-neighbor
/// tile's half-pel MV `seed_mv_hpel`. Returns `Some((dx_hpel, dy_hpel, sx_hpel, sy_hpel))` only
/// when the motion-compensated match beats zero-MV by at least 20%.
#[allow(clippy::too_many_arguments)]
fn search_local_mv_2d(
    prev_full: &[u8],
    cur_band: &[u8],
    fb_w: u32,
    fb_h: u32,
    x0: u32,
    y0: u32,
    local_y0: u32,
    tw: u32,
    th: u32,
    seed_mv_hpel: (i8, i8),
) -> Option<(i8, i8, u32, u32)> {
    const MAX_RANGE_INT: i32 = 24;
    const MAX_RANGE_HPEL: i32 = 48;
    let zero_sad = tile_bgra_sad_fast(prev_full, cur_band, fb_w, x0, y0, x0, local_y0, tw, th);
    // Skip search on near-stationary tiles (mean RGB diff < ~2/channel across 64 samples).
    if zero_sad < 384 {
        return None;
    }

    let mut best_dx_hpel = 0i32;
    let mut best_dy_hpel = 0i32;
    let mut best_sad = zero_sad;

    let eval_hpel = |dx_h: i32, dy_h: i32| -> Option<(u64, u32, u32)> {
        if dx_h.abs() > MAX_RANGE_HPEL || dy_h.abs() > MAX_RANGE_HPEL {
            return None;
        }
        let sx_h = (x0 as i32) * 2 - dx_h;
        let sy_h = (y0 as i32) * 2 - dy_h;
        if sx_h < 0 || sy_h < 0 {
            return None;
        }
        let sx = (sx_h as u32) >> 1;
        let sy = (sy_h as u32) >> 1;
        if sx + tw > fb_w || sy + th > fb_h {
            return None;
        }
        let sad = tile_bgra_sad_hpel_fast(
            prev_full,
            cur_band,
            fb_w,
            sx_h as u32,
            sy_h as u32,
            x0,
            local_y0,
            tw,
            th,
        );
        Some((sad, sx_h as u32, sy_h as u32))
    };

    // Test spatial neighbor MV predictor first (including its half-pel fractional phase!).
    if seed_mv_hpel != (0, 0) {
        if let Some((sad, _, _)) = eval_hpel(seed_mv_hpel.0 as i32, seed_mv_hpel.1 as i32) {
            if sad < best_sad {
                best_sad = sad;
                best_dx_hpel = seed_mv_hpel.0 as i32;
                best_dy_hpel = seed_mv_hpel.1 as i32;
            }
        }
    }

    // 4-step logarithmic 8-neighbor integer search (8 -> 4 -> 2 -> 1 px, i.e., 16 -> 8 -> 4 -> 2 hpel).
    let mut best_int_dx = best_dx_hpel / 2;
    let mut best_int_dy = best_dy_hpel / 2;
    for step in [8i32, 4, 2, 1] {
        if best_sad * 5 <= zero_sad && step >= 4 {
            continue;
        }
        let cx = best_int_dx;
        let cy = best_int_dy;
        for (ndx, ndy) in [
            (cx - step, cy),
            (cx + step, cy),
            (cx, cy - step),
            (cx, cy + step),
            (cx - step, cy - step),
            (cx + step, cy - step),
            (cx - step, cy + step),
            (cx + step, cy + step),
        ] {
            if ndx.abs() <= MAX_RANGE_INT && ndy.abs() <= MAX_RANGE_INT {
                if let Some((sad, _, _)) = eval_hpel(ndx * 2, ndy * 2) {
                    if sad < best_sad {
                        best_sad = sad;
                        best_int_dx = ndx;
                        best_int_dy = ndy;
                        best_dx_hpel = ndx * 2;
                        best_dy_hpel = ndy * 2;
                    }
                }
            }
        }
    }

    // Sub-pixel 1/2-pel refinement step (`±1` half-pel unit around the winning integer/seed vector).
    let cx_h = best_dx_hpel;
    let cy_h = best_dy_hpel;
    for (nhx, nhy) in [
        (cx_h - 1, cy_h),
        (cx_h + 1, cy_h),
        (cx_h, cy_h - 1),
        (cx_h, cy_h + 1),
        (cx_h - 1, cy_h - 1),
        (cx_h + 1, cy_h - 1),
        (cx_h - 1, cy_h + 1),
        (cx_h + 1, cy_h + 1),
    ] {
        if let Some((sad, _, _)) = eval_hpel(nhx, nhy) {
            if sad < best_sad {
                best_sad = sad;
                best_dx_hpel = nhx;
                best_dy_hpel = nhy;
            }
        }
    }

    if (best_dx_hpel != 0 || best_dy_hpel != 0) && best_sad * 10 <= zero_sad * 8 {
        let sx_hpel = ((x0 as i32) * 2 - best_dx_hpel) as u32;
        let sy_hpel = ((y0 as i32) * 2 - best_dy_hpel) as u32;
        Some((best_dx_hpel as i8, best_dy_hpel as i8, sx_hpel, sy_hpel))
    } else {
        None
    }
}

fn tile_is_opaque(bgra: &[u8], fb_w: u32, x0: u32, y0: u32, tw: u32, th: u32) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { simd::avx2::tile_is_opaque(bgra, fb_w, x0, y0, tw, th) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            return unsafe { simd::neon::tile_is_opaque(bgra, fb_w, x0, y0, tw, th) };
        }
    }
    #[allow(unreachable_code)]
    simd::tile_is_opaque_scalar(bgra, fb_w, x0, y0, tw, th)
}

/// Compare temporal residual energy vs. spatial intra energy on the luma plane.
fn should_use_inter_tile(cur_y: &[i32; 64 * 64], ref_y: &[i32; 64 * 64]) -> bool {
    // Sample every 4th pixel for speed (1024 samples per tile).
    let mut sum_y: i64 = 0;
    let mut inter_sad: i64 = 0;
    for i in (0..4096).step_by(4) {
        sum_y += cur_y[i] as i64;
        inter_sad += (cur_y[i] - ref_y[i]).abs() as i64;
    }
    let mean = (sum_y / 1024) as i32;
    let mut intra_sad: i64 = 0;
    for i in (0..4096).step_by(4) {
        intra_sad += (cur_y[i] - mean).abs() as i64;
    }
    inter_sad * 10 <= intra_sad * 9
}

fn count_identical_pixels(
    ref_bgra: &[u8],
    cur_bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                simd::avx2::count_identical_pixels(ref_bgra, cur_bgra, fb_w, x0, y0, tw, th)
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            return unsafe {
                simd::neon::count_identical_pixels(ref_bgra, cur_bgra, fb_w, x0, y0, tw, th)
            };
        }
    }
    #[allow(unreachable_code)]
    simd::count_identical_pixels_scalar(ref_bgra, cur_bgra, fb_w, x0, y0, tw, th)
}

pub fn xor_encode_row(ref_row: &mut [u8], cur_row: &[u8], out: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::xor_encode_row(ref_row, cur_row, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            unsafe {
                simd::neon::xor_encode_row(ref_row, cur_row, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    simd::xor_encode_row_scalar(ref_row, cur_row, out);
}

pub fn xor_decode_row(dst_row: &mut [u8], src_row: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd() && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                simd::avx2::xor_decode_row(dst_row, src_row);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd() {
            unsafe {
                simd::neon::xor_decode_row(dst_row, src_row);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    simd::xor_decode_row_scalar(dst_row, src_row);
}

#[allow(clippy::too_many_arguments)]
fn write_lossless_tile(
    ref_bgra: &mut [u8],
    cur_bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    force_raw: bool,
    out: &mut Vec<u8>,
) {
    let stride = fb_w as usize * 4;
    let row_bytes = tw as usize * 4;
    let use_xor = if force_raw {
        false
    } else {
        // Count unchanged pixels: if >= 40% are identical, XOR produces long zero runs for LZ4.
        let same = count_identical_pixels(ref_bgra, cur_bgra, fb_w, x0, y0, tw, th);
        let total = (tw as usize) * (th as usize);
        same * 5 >= total * 2
    };

    if use_xor {
        out.push(MODE_LOSSLESS_XOR);
        out.reserve(row_bytes * (th as usize));
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            xor_encode_row(
                &mut ref_bgra[off..off + row_bytes],
                &cur_bgra[off..off + row_bytes],
                out,
            );
        }
    } else {
        out.push(MODE_LOSSLESS_RAW);
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            out.extend_from_slice(&cur_bgra[off..off + row_bytes]);
            ref_bgra[off..off + row_bytes].copy_from_slice(&cur_bgra[off..off + row_bytes]);
        }
    }
}

fn write_lossless_tile_extracted(
    cur_tile_64: &[u8],
    ref_tile_64: &[u8],
    tw: u32,
    th: u32,
    force_raw: bool,
    out: &mut Vec<u8>,
) {
    let stride = 64usize * 4;
    let row_bytes = tw as usize * 4;
    let use_xor = if force_raw {
        false
    } else {
        let same = count_identical_pixels(ref_tile_64, cur_tile_64, 64, 0, 0, tw, th);
        let total = (tw as usize) * (th as usize);
        same * 5 >= total * 2
    };

    if use_xor {
        out.push(MODE_LOSSLESS_XOR);
        out.reserve(row_bytes * (th as usize));
        for py in 0..(th as usize) {
            let off = py * stride;
            for b in 0..row_bytes {
                out.push(cur_tile_64[off + b] ^ ref_tile_64[off + b]);
            }
        }
    } else {
        out.push(MODE_LOSSLESS_RAW);
        out.reserve(row_bytes * (th as usize));
        for py in 0..(th as usize) {
            let off = py * stride;
            out.extend_from_slice(&cur_tile_64[off..off + row_bytes]);
        }
    }
}

#[inline]
fn zigzag_i8(d: i8) -> u8 {
    ((d << 1) ^ (d >> 7)) as u8
}

#[inline]
fn unzigzag_u8(u: u8) -> i8 {
    ((u >> 1) ^ 0u8.wrapping_sub(u & 1)) as i8
}

const RICE_MAX_UNARY: u32 = 15;

struct RiceBitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    bits: u32,
}

impl<'a> RiceBitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    fn write_bits(&mut self, val: u32, count: u32) {
        if count == 0 {
            return;
        }
        self.acc = (self.acc << count) | (val as u64);
        self.bits += count;
        while self.bits >= 8 {
            self.bits -= 8;
            self.out.push((self.acc >> self.bits) as u8);
        }
    }

    #[inline]
    fn write_rice(&mut self, u: u8, k: u8) {
        let q = (u >> k) as u32;
        if q < RICE_MAX_UNARY {
            // `q` zero bits followed by a `1` bit = value `1` in `q + 1` bits.
            self.write_bits(1, q + 1);
            if k > 0 {
                let rem = (u & ((1u8 << k) - 1)) as u32;
                self.write_bits(rem, k as u32);
            }
        } else {
            // Escape: 15 zero bits + `1` bit + raw 8-bit `u`.
            self.write_bits(1, RICE_MAX_UNARY + 1);
            self.write_bits(u as u32, 8);
        }
    }

    fn finish(mut self) {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.out.push((self.acc << pad) as u8);
            self.bits = 0;
        }
    }
}

struct RiceBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u64,
    bits: u32,
}

impl<'a> RiceBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    fn ensure_bits(&mut self, need: u32) -> Result<()> {
        while self.bits < need {
            if self.byte_pos >= self.data.len() {
                bail!("unexpected EOF reading lossless residual bitstream");
            }
            self.acc = (self.acc << 8) | (self.data[self.byte_pos] as u64);
            self.byte_pos += 1;
            self.bits += 8;
        }
        Ok(())
    }

    #[inline]
    fn read_bits(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.ensure_bits(count)?;
        self.bits -= count;
        let mask = (1u64 << count) - 1;
        Ok(((self.acc >> self.bits) & mask) as u32)
    }

    #[inline]
    fn read_rice(&mut self, k: u8) -> Result<u8> {
        let mut q = 0u32;
        loop {
            let bit = self.read_bits(1)?;
            if bit == 1 {
                break;
            }
            q += 1;
            if q > RICE_MAX_UNARY {
                bail!("invalid Rice unary prefix in lossless residual");
            }
        }
        if q < RICE_MAX_UNARY {
            let rem = self.read_bits(k as u32)?;
            Ok(((q << k) | rem) as u8)
        } else {
            Ok(self.read_bits(8)? as u8)
        }
    }
}

/// Encode a lossless refinement tile reusing the lossy `ref_bgra` preview already on the client
/// via signed modular delta (`cur - ref`) + Golomb-Rice coding, falling back to raw/XOR if larger.
#[allow(clippy::too_many_arguments)]
fn write_lossless_refinement_tile(
    ref_bgra: &mut [u8],
    cur_bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    samples: &mut Vec<u8>,
    out: &mut Vec<u8>,
) {
    let stride = fb_w as usize * 4;
    let has_alpha = !tile_is_opaque(cur_bgra, fb_w, x0, y0, tw, th);
    let channels = if has_alpha { 4usize } else { 3usize };
    let num_pixels = (tw as usize) * (th as usize);
    let raw_bytes = num_pixels * 4;

    samples.clear();
    let mut costs = [0usize; 6];
    for py in 0..(th as usize) {
        let row_off = (y0 as usize + py) * stride + (x0 as usize) * 4;
        for px in 0..(tw as usize) {
            let p = row_off + px * 4;
            for c in 0..channels {
                let d = cur_bgra[p + c].wrapping_sub(ref_bgra[p + c]) as i8;
                let u = zigzag_i8(d);
                samples.push(u);
                for k in 0..6u8 {
                    let q = (u >> k) as u32;
                    costs[k as usize] += if q < RICE_MAX_UNARY {
                        (q + 1 + k as u32) as usize
                    } else {
                        (RICE_MAX_UNARY + 1 + 8) as usize
                    };
                }
            }
        }
    }

    let best_k = (0..6u8).min_by_key(|&k| costs[k as usize]).unwrap();
    let encoded_bytes = costs[best_k as usize].div_ceil(8);

    // Only emit Golomb-Rice residual if it meaningfully beats raw BGRA (and fits u16 length).
    if encoded_bytes + 3 < (raw_bytes * 7) / 8 && encoded_bytes <= u16::MAX as usize {
        let alpha_flag = if has_alpha { MODE_FLAG_ALPHA } else { 0 };
        out.push(MODE_LOSSLESS_RESIDUAL | alpha_flag);
        out.push(best_k);
        out.extend_from_slice(&(encoded_bytes as u16).to_le_bytes());

        let start_len = out.len();
        out.reserve(encoded_bytes);
        let mut bw = RiceBitWriter::new(out);
        for &u in samples.iter() {
            bw.write_rice(u, best_k);
        }
        bw.finish();
        debug_assert_eq!(out.len() - start_len, encoded_bytes);

        let row_bytes = tw as usize * 4;
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            ref_bgra[off..off + row_bytes].copy_from_slice(&cur_bgra[off..off + row_bytes]);
        }
    } else {
        write_lossless_tile(ref_bgra, cur_bgra, fb_w, x0, y0, tw, th, false, out);
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_lossless_residual_tile(
    raw: &[u8],
    pos: &mut usize,
    has_alpha: bool,
    bgra_mut: &mut [u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) -> Result<()> {
    if *pos + 3 > raw.len() {
        bail!("unexpected EOF reading lossless residual header");
    }
    let k = raw[*pos];
    if k > 5 {
        bail!("invalid Rice parameter k={k} in lossless residual");
    }
    let bitstream_len = u16::from_le_bytes([raw[*pos + 1], raw[*pos + 2]]) as usize;
    *pos += 3;
    if *pos + bitstream_len > raw.len() {
        bail!("unexpected EOF reading lossless residual payload");
    }
    let bitstream = &raw[*pos..*pos + bitstream_len];
    *pos += bitstream_len;

    let stride = fb_w as usize * 4;
    let channels = if has_alpha { 4usize } else { 3usize };
    let mut br = RiceBitReader::new(bitstream);
    for py in 0..(th as usize) {
        let row_off = (y0 as usize + py) * stride + (x0 as usize) * 4;
        for px in 0..(tw as usize) {
            let p = row_off + px * 4;
            for c in 0..channels {
                let u = br.read_rice(k)?;
                let d = unzigzag_u8(u) as u8;
                bgra_mut[p + c] = bgra_mut[p + c].wrapping_add(d);
            }
            if !has_alpha {
                bgra_mut[p + 3] = 255;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SmartTileDecoder
// ---------------------------------------------------------------------------

struct DctDecodeScratch {
    ref_y: [i32; 64 * 64],
    ref_cb: [i32; 32 * 32],
    ref_cr: [i32; 32 * 32],
    ref_a: [i32; 64 * 64],
    recon_y: [i32; 64 * 64],
    recon_cb: [i32; 32 * 32],
    recon_cr: [i32; 32 * 32],
    recon_a: [i32; 64 * 64],
    ref_hy: [i32; 32 * 32],
    ref_hcb: [i32; 16 * 16],
    ref_hcr: [i32; 16 * 16],
    ref_ha: [i32; 32 * 32],
    recon_hy: [i32; 32 * 32],
    recon_hcb: [i32; 16 * 16],
    recon_hcr: [i32; 16 * 16],
    recon_ha: [i32; 32 * 32],
    ref_qy: [i32; 16 * 16],
    ref_qcb: [i32; 8 * 8],
    ref_qcr: [i32; 8 * 8],
    ref_qa: [i32; 16 * 16],
    recon_qy: [i32; 16 * 16],
    recon_qcb: [i32; 8 * 8],
    recon_qcr: [i32; 8 * 8],
    recon_qa: [i32; 16 * 16],
}

impl Default for DctDecodeScratch {
    fn default() -> Self {
        Self {
            ref_y: [0; 64 * 64],
            ref_cb: [0; 32 * 32],
            ref_cr: [0; 32 * 32],
            ref_a: [0; 64 * 64],
            recon_y: [0; 64 * 64],
            recon_cb: [0; 32 * 32],
            recon_cr: [0; 32 * 32],
            recon_a: [0; 64 * 64],
            ref_hy: [0; 32 * 32],
            ref_hcb: [0; 16 * 16],
            ref_hcr: [0; 16 * 16],
            ref_ha: [0; 32 * 32],
            recon_hy: [0; 32 * 32],
            recon_hcb: [0; 16 * 16],
            recon_hcr: [0; 16 * 16],
            recon_ha: [0; 32 * 32],
            ref_qy: [0; 16 * 16],
            ref_qcb: [0; 8 * 8],
            ref_qcr: [0; 8 * 8],
            ref_qa: [0; 16 * 16],
            recon_qy: [0; 16 * 16],
            recon_qcb: [0; 8 * 8],
            recon_qcr: [0; 8 * 8],
            recon_qa: [0; 16 * 16],
        }
    }
}

/// Per-window `SmartTile` decoder.
#[derive(Default)]
pub struct SmartTileDecoder {
    width: u32,
    height: u32,
    bgra: Arc<Vec<u8>>,
    /// Reusable snapshot of `bgra` before the current frame, populated when `FLAG_FRAME_MV | FLAG_PREV_REF` is set.
    prev_bgra: Vec<u8>,
    /// Lazily allocated DCT plane scratch buffers so lossless frames avoid zeroing ~104 KiB on the stack.
    dct_scratch: Option<Box<DctDecodeScratch>>,
}

impl SmartTileDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a `SmartTile` packet into the persistent BGRA framebuffer and return a clone.
    pub fn decode(&mut self, width: u32, height: u32, payload: &[u8]) -> Result<Vec<u8>> {
        let (buf, _) = self.decode_with_damage(width, height, payload)?;
        Ok((*buf).clone())
    }

    /// Decode a `SmartTile` packet into the persistent BGRA framebuffer, returning a shared
    /// `Arc<Vec<u8>>` (zero-copy when the UI thread has finished presenting the previous frame)
    /// and the list of damaged physical pixel rectangles `(x, y, w, h)` for partial compositor damage.
    pub fn decode_with_damage(
        &mut self,
        width: u32,
        height: u32,
        payload: &[u8],
    ) -> Result<(Arc<Vec<u8>>, Vec<(i32, i32, i32, i32)>)> {
        let expected_len = validate_frame_dims(width, height)?;
        let cols = width.div_ceil(TILE_SIZE) as usize;
        let rows = height.div_ceil(TILE_SIZE) as usize;
        let max_tiles = cols * rows;
        let max_uncompressed = expected_len
            .saturating_add(max_tiles.saturating_mul(32))
            .saturating_add(4096)
            .min(MAX_DECODED_FRAME_BYTES + 2 * 1024 * 1024);

        let raw = decompress_lz4_bounded(payload, max_uncompressed)
            .context("failed to decompress SmartTile packet")?;
        if raw.len() < 8 {
            bail!("SmartTile packet too short: {} bytes", raw.len());
        }
        let version = raw[0];
        if version != PACKET_VERSION {
            bail!("unsupported SmartTile packet version: {version}");
        }
        let flags = raw[1];
        let quality = raw[2];
        let num_copy_rects = raw[3] as usize;
        if num_copy_rects > 16 {
            bail!("SmartTile copy_rect count {num_copy_rects} exceeds limit 16");
        }
        let mut pos = 4usize;

        let bgra_mut = Arc::make_mut(&mut self.bgra);
        let full_damage = (flags & FLAG_KEYFRAME) != 0
            || width != self.width
            || height != self.height
            || bgra_mut.len() != expected_len;
        if full_damage {
            self.width = width;
            self.height = height;
            bgra_mut.resize(expected_len, 0);
        }

        let mut damage = Vec::new();
        if full_damage {
            damage.push((0, 0, width as i32, height as i32));
        }

        // 1. Apply CopyRect ops in-place.
        for _ in 0..num_copy_rects {
            if pos + 12 > raw.len() {
                bail!("unexpected EOF reading CopyRectOp");
            }
            let op = CopyRectOp {
                dst_x: u16::from_le_bytes([raw[pos], raw[pos + 1]]),
                dst_y: u16::from_le_bytes([raw[pos + 2], raw[pos + 3]]),
                w: u16::from_le_bytes([raw[pos + 4], raw[pos + 5]]),
                h: u16::from_le_bytes([raw[pos + 6], raw[pos + 7]]),
                src_x: u16::from_le_bytes([raw[pos + 8], raw[pos + 9]]),
                src_y: u16::from_le_bytes([raw[pos + 10], raw[pos + 11]]),
            };
            pos += 12;
            apply_copy_rect(bgra_mut, width, height, op);
            if !full_damage {
                damage.push((op.dst_x as i32, op.dst_y as i32, op.w as i32, op.h as i32));
            }
        }

        // 2. Optional frame-level scroll motion vector `(mv_dx, mv_dy)` and/or previous-frame snapshot.
        let frame_mv: Option<(i16, i16)> = if (flags & FLAG_FRAME_MV) != 0 {
            if pos + 4 > raw.len() {
                bail!("unexpected EOF reading SmartTile frame MV");
            }
            let dx = i16::from_le_bytes([raw[pos], raw[pos + 1]]);
            let dy = i16::from_le_bytes([raw[pos + 2], raw[pos + 3]]);
            pos += 4;
            Some((dx, dy))
        } else {
            None
        };
        let has_prev_snapshot = (flags & (FLAG_FRAME_MV | FLAG_PREV_REF)) != 0;
        if has_prev_snapshot {
            self.prev_bgra.resize(expected_len, 0);
            self.prev_bgra.copy_from_slice(bgra_mut);
        }

        if pos + 4 > raw.len() {
            bail!("unexpected EOF reading SmartTile tile count");
        }
        let num_tiles =
            u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
        pos += 4;
        if num_tiles > max_tiles {
            bail!("SmartTile tile count {num_tiles} exceeds grid maximum {max_tiles}");
        }

        let luma_q = build_quant_table(&BASE_LUMA_QUANT, quality);
        let chroma_q = build_quant_table(&BASE_CHROMA_QUANT, quality);
        let stride = width as usize * 4;
        let dct_scratch = &mut self.dct_scratch;

        for _ in 0..num_tiles {
            if pos + 5 > raw.len() {
                bail!("unexpected EOF reading SmartTile tile header");
            }
            let tx = u16::from_le_bytes([raw[pos], raw[pos + 1]]) as u32;
            let ty = u16::from_le_bytes([raw[pos + 2], raw[pos + 3]]) as u32;
            let mode = raw[pos + 4];
            pos += 5;

            let x0 = tx * TILE_SIZE;
            let y0 = ty * TILE_SIZE;
            if x0 >= width || y0 >= height {
                bail!("SmartTile out of bounds: ({tx}, {ty}) for {width}x{height}");
            }
            let tw = (width - x0).min(TILE_SIZE);
            let th = (height - y0).min(TILE_SIZE);
            let row_bytes = tw as usize * 4;
            if !full_damage {
                damage.push((x0 as i32, y0 as i32, tw as i32, th as i32));
            }

            let has_alpha = (mode & MODE_FLAG_ALPHA) != 0;
            let use_mv = (mode & MODE_FLAG_MV) != 0;
            let use_local_mv = (mode & MODE_FLAG_LOCAL_MV) != 0;
            if use_mv && use_local_mv {
                bail!("invalid SmartTile mode with both MV and LOCAL_MV flags: {mode}");
            }
            let base_mode = mode & !(MODE_FLAG_ALPHA | MODE_FLAG_MV | MODE_FLAG_LOCAL_MV);
            let mv_src = |x0: u32, y0: u32, tw: u32, th: u32| -> Result<(u32, u32)> {
                let (dx, dy) = frame_mv.context("tile uses MV without FLAG_FRAME_MV")?;
                let sx = x0 as i32 - dx as i32;
                let sy = y0 as i32 - dy as i32;
                if sx < 0 || sy < 0 || (sx as u32) + tw > width || (sy as u32) + th > height {
                    bail!("SmartTile MV ({dx}, {dy}) out of bounds at ({x0}, {y0})");
                }
                Ok((sx as u32, sy as u32))
            };
            let read_local_mv_hpel_src = |pos: &mut usize| -> Result<(u32, u32)> {
                if !has_prev_snapshot {
                    bail!("tile uses LOCAL_MV without FLAG_PREV_REF");
                }
                if *pos + 2 > raw.len() {
                    bail!("unexpected EOF reading SmartTile local MV");
                }
                let dx_h = raw[*pos] as i8 as i32;
                let dy_h = raw[*pos + 1] as i8 as i32;
                *pos += 2;
                let sx_h = (x0 as i32) * 2 - dx_h;
                let sy_h = (y0 as i32) * 2 - dy_h;
                if sx_h < 0 || sy_h < 0 {
                    bail!("SmartTile local MV ({dx_h}, {dy_h}) out of bounds at ({x0}, {y0})");
                }
                let sx = (sx_h as u32) >> 1;
                let sy = (sy_h as u32) >> 1;
                if sx + tw > width || sy + th > height {
                    bail!("SmartTile local MV ({dx_h}, {dy_h}) out of bounds at ({x0}, {y0})");
                }
                Ok((sx_h as u32, sy_h as u32))
            };
            match base_mode {
                MODE_LOSSLESS_RAW if !has_alpha && !use_mv && !use_local_mv => {
                    let needed = row_bytes * th as usize;
                    if pos + needed > raw.len() {
                        bail!("unexpected EOF reading lossless raw tile");
                    }
                    for py in 0..(th as usize) {
                        let dst = (y0 as usize + py) * stride + (x0 as usize) * 4;
                        bgra_mut[dst..dst + row_bytes].copy_from_slice(&raw[pos..pos + row_bytes]);
                        pos += row_bytes;
                    }
                }
                MODE_LOSSLESS_XOR if !has_alpha && !use_mv && !use_local_mv => {
                    let needed = row_bytes * th as usize;
                    if pos + needed > raw.len() {
                        bail!("unexpected EOF reading lossless XOR tile");
                    }
                    for py in 0..(th as usize) {
                        let dst = (y0 as usize + py) * stride + (x0 as usize) * 4;
                        xor_decode_row(
                            &mut bgra_mut[dst..dst + row_bytes],
                            &raw[pos..pos + row_bytes],
                        );
                        pos += row_bytes;
                    }
                }
                MODE_MV_SKIP if !has_alpha && !use_mv && !use_local_mv => {
                    let (sx, sy) = mv_src(x0, y0, tw, th)?;
                    copy_tile_from_prev(&self.prev_bgra, bgra_mut, width, sx, sy, x0, y0, tw, th);
                }
                MODE_LOSSLESS_RESIDUAL if !use_mv && !use_local_mv => {
                    decode_lossless_residual_tile(
                        &raw, &mut pos, has_alpha, bgra_mut, width, x0, y0, tw, th,
                    )?;
                }
                MODE_DCT_INTRA if !use_mv && !use_local_mv => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_y,
                        64,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_cb,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_cr,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            None,
                            &mut sc.recon_a,
                            64,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_to_bgra_tile(
                        &sc.recon_y,
                        &sc.recon_cb,
                        &sc.recon_cr,
                        if has_alpha { Some(&sc.recon_a) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                MODE_DCT_INTER => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    let (pred_buf, px0_hpel, py0_hpel): (&[u8], u32, u32) = if use_mv {
                        let (sx, sy) = mv_src(x0, y0, tw, th)?;
                        (&self.prev_bgra, sx * 2, sy * 2)
                    } else if use_local_mv {
                        let (sx_h, sy_h) = read_local_mv_hpel_src(&mut pos)?;
                        (&self.prev_bgra, sx_h, sy_h)
                    } else {
                        (&*bgra_mut, x0 * 2, y0 * 2)
                    };
                    bgra_tile_to_ycbcr420_hpel(
                        pred_buf,
                        width,
                        px0_hpel,
                        py0_hpel,
                        tw,
                        th,
                        &mut sc.ref_y,
                        &mut sc.ref_cb,
                        &mut sc.ref_cr,
                    );
                    if has_alpha {
                        bgra_tile_to_alpha_hpel(
                            pred_buf,
                            width,
                            px0_hpel,
                            py0_hpel,
                            tw,
                            th,
                            &mut sc.ref_a,
                        );
                    }
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_y),
                        &mut sc.recon_y,
                        64,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_cb),
                        &mut sc.recon_cb,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_cr),
                        &mut sc.recon_cr,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            Some(&sc.ref_a),
                            &mut sc.recon_a,
                            64,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_to_bgra_tile(
                        &sc.recon_y,
                        &sc.recon_cb,
                        &sc.recon_cr,
                        if has_alpha { Some(&sc.recon_a) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                MODE_DCT_HALF_INTRA if !use_mv && !use_local_mv => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let hw = (tw as usize).div_ceil(2);
                    let hh = (th as usize).div_ceil(2);
                    let luma_bx = hw.div_ceil(8);
                    let luma_by = hh.div_ceil(8);
                    let chroma_bx = hw.div_ceil(2).div_ceil(8);
                    let chroma_by = hh.div_ceil(2).div_ceil(8);
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_hy,
                        32,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_hcb,
                        16,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_hcr,
                        16,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            None,
                            &mut sc.recon_ha,
                            32,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_half_to_bgra_tile(
                        &sc.recon_hy,
                        &sc.recon_hcb,
                        &sc.recon_hcr,
                        if has_alpha { Some(&sc.recon_ha) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                MODE_DCT_HALF_INTER => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let hw = (tw as usize).div_ceil(2);
                    let hh = (th as usize).div_ceil(2);
                    let luma_bx = hw.div_ceil(8);
                    let luma_by = hh.div_ceil(8);
                    let chroma_bx = hw.div_ceil(2).div_ceil(8);
                    let chroma_by = hh.div_ceil(2).div_ceil(8);
                    let (pred_buf, px0, py0): (&[u8], u32, u32) = if use_mv {
                        let (sx, sy) = mv_src(x0, y0, tw, th)?;
                        (&self.prev_bgra, sx, sy)
                    } else if use_local_mv {
                        let (sx_h, sy_h) = read_local_mv_hpel_src(&mut pos)?;
                        (&self.prev_bgra, sx_h >> 1, sy_h >> 1)
                    } else {
                        (&*bgra_mut, x0, y0)
                    };
                    bgra_tile_to_ycbcr420_half(
                        pred_buf,
                        width,
                        px0,
                        py0,
                        tw,
                        th,
                        &mut sc.ref_hy,
                        &mut sc.ref_hcb,
                        &mut sc.ref_hcr,
                    );
                    if has_alpha {
                        bgra_tile_to_alpha_half(pred_buf, width, px0, py0, tw, th, &mut sc.ref_ha);
                    }
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_hy),
                        &mut sc.recon_hy,
                        32,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_hcb),
                        &mut sc.recon_hcb,
                        16,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_hcr),
                        &mut sc.recon_hcr,
                        16,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            Some(&sc.ref_ha),
                            &mut sc.recon_ha,
                            32,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_half_to_bgra_tile(
                        &sc.recon_hy,
                        &sc.recon_hcb,
                        &sc.recon_hcr,
                        if has_alpha { Some(&sc.recon_ha) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                MODE_DCT_QUARTER_INTRA if !use_mv && !use_local_mv => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let qw = (tw as usize).div_ceil(4);
                    let qh = (th as usize).div_ceil(4);
                    let luma_bx = qw.div_ceil(8);
                    let luma_by = qh.div_ceil(8);
                    let chroma_bx = qw.div_ceil(2).div_ceil(8);
                    let chroma_by = qh.div_ceil(2).div_ceil(8);
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_qy,
                        16,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_qcb,
                        8,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        None,
                        &mut sc.recon_qcr,
                        8,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            None,
                            &mut sc.recon_qa,
                            16,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_quarter_to_bgra_tile(
                        &sc.recon_qy,
                        &sc.recon_qcb,
                        &sc.recon_qcr,
                        if has_alpha { Some(&sc.recon_qa) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                MODE_DCT_QUARTER_INTER => {
                    let sc =
                        dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let qw = (tw as usize).div_ceil(4);
                    let qh = (th as usize).div_ceil(4);
                    let luma_bx = qw.div_ceil(8);
                    let luma_by = qh.div_ceil(8);
                    let chroma_bx = qw.div_ceil(2).div_ceil(8);
                    let chroma_by = qh.div_ceil(2).div_ceil(8);
                    let (pred_buf, px0, py0): (&[u8], u32, u32) = if use_mv {
                        let (sx, sy) = mv_src(x0, y0, tw, th)?;
                        (&self.prev_bgra, sx, sy)
                    } else if use_local_mv {
                        let (sx_h, sy_h) = read_local_mv_hpel_src(&mut pos)?;
                        (&self.prev_bgra, sx_h >> 1, sy_h >> 1)
                    } else {
                        (&*bgra_mut, x0, y0)
                    };
                    bgra_tile_to_ycbcr420_quarter(
                        pred_buf,
                        width,
                        px0,
                        py0,
                        tw,
                        th,
                        &mut sc.ref_qy,
                        &mut sc.ref_qcb,
                        &mut sc.ref_qcr,
                    );
                    if has_alpha {
                        bgra_tile_to_alpha_quarter(
                            pred_buf,
                            width,
                            px0,
                            py0,
                            tw,
                            th,
                            &mut sc.ref_qa,
                        );
                    }
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_qy),
                        &mut sc.recon_qy,
                        16,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_qcb),
                        &mut sc.recon_qcb,
                        8,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&sc.ref_qcr),
                        &mut sc.recon_qcr,
                        8,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    if has_alpha {
                        decode_plane(
                            &raw,
                            &mut pos,
                            Some(&sc.ref_qa),
                            &mut sc.recon_qa,
                            16,
                            luma_bx,
                            luma_by,
                            &luma_q,
                        )?;
                    }
                    ycbcr420_quarter_to_bgra_tile(
                        &sc.recon_qy,
                        &sc.recon_qcb,
                        &sc.recon_qcr,
                        if has_alpha { Some(&sc.recon_qa) } else { None },
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    );
                }
                _ => bail!("unknown SmartTile mode: {mode}"),
            }
        }

        Ok((Arc::clone(&self.bgra), damage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damaged_tile_mask_marks_touched_tiles() {
        // 3x2 tiles over a 150x100 frame.
        let mask = damaged_tile_mask(&[(60, 10, 10, 10), (140, 90, 50, 50)], 150, 100, 3, 2);
        assert_eq!(mask, vec![true, true, false, false, false, true]);
        assert_eq!(damaged_tile_mask(&[], 150, 100, 3, 2), vec![false; 6]);
        assert_eq!(
            damaged_tile_mask(&[(500, 500, 10, 10), (0, 0, 0, 10)], 150, 100, 3, 2),
            vec![false; 6]
        );
    }

    #[test]
    fn encode_with_damage_compares_only_damaged_tiles() {
        let (w, h) = (256u32, 128u32);
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();
        let mut frame = vec![0x40u8; (w * h * 4) as usize];
        let key = enc.encode_with_damage(w, h, &frame, 90, Some(&[]));
        assert!(key.is_keyframe, "damage never skips a keyframe's tiles");
        assert_eq!(key.dirty_tiles, 8);
        dec.decode(w, h, &key.payload).unwrap();

        let paint = |frame: &mut [u8], x: u32, y: u32, v: u8| {
            let i = ((y * w + x) * 4) as usize;
            frame[i..i + 4].copy_from_slice(&[v, v, v, 0xff]);
        };
        // Two tiles change; damage names one of them and an unchanged one.
        paint(&mut frame, 10, 10, 0x90);
        paint(&mut frame, 200, 100, 0x90);
        let pkt = enc.encode_with_damage(w, h, &frame, 90, Some(&[(0, 0, 20, 20), (70, 70, 4, 4)]));
        assert!(!pkt.is_keyframe);
        assert_eq!(pkt.dirty_tiles, 1);
        let out = dec.decode(w, h, &pkt.payload).unwrap();
        assert_eq!(
            &out[((10 * w + 10) * 4) as usize..][..4],
            &[0x90, 0x90, 0x90, 0xff]
        );
        // The undamaged change is trusted away, as a compositor would.
        assert_eq!(&out[((100 * w + 200) * 4) as usize..][..4], &[0x40; 4]);

        // Full damage finds it.
        let pkt = enc.encode_with_damage(w, h, &frame, 90, None);
        assert_eq!(pkt.dirty_tiles, 1);
        let out = dec.decode(w, h, &pkt.payload).unwrap();
        assert_eq!(
            &out[((100 * w + 200) * 4) as usize..][..4],
            &[0x90, 0x90, 0x90, 0xff]
        );

        // No damage: nothing is encoded.
        let pkt = enc.encode_with_damage(w, h, &frame, 90, Some(&[]));
        assert_eq!(pkt.dirty_tiles, 0);
    }

    #[test]
    fn dct_idct_8x8_roundtrip_is_accurate() {
        let mut block = [0i32; 64];
        for r in 0..8 {
            for c in 0..8 {
                block[r * 8 + c] = ((r * 19 + c * 13) % 200) as i32 - 100;
            }
        }
        let coeffs = dct_8x8(&block);
        let recon = idct_8x8(&coeffs);
        for i in 0..64 {
            assert!(
                (block[i] - recon[i]).abs() <= 2,
                "mismatch at {i}: orig={}, recon={}",
                block[i],
                recon[i]
            );
        }
    }

    #[test]
    fn vertical_subregion_scroll_uses_copy_rect_and_stays_lossless() {
        let w = 512u32;
        let h = 512u32;
        let mut frame1 = vec![0u8; (w * h * 4) as usize];
        // Fixed left sidebar (0..96) + fixed top bar (0..48) + scrolling text-like pattern.
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                if x < 96 || y < 48 {
                    // Textured fixed sidebar & top bar (distinct per row/col so shifting breaks match).
                    frame1[idx] = (40 + ((y * 7 + x) % 50)) as u8;
                    frame1[idx + 1] = 40;
                    frame1[idx + 2] = 45;
                    frame1[idx + 3] = 255;
                } else {
                    // Distinct per-row pattern so scroll hashes are unique.
                    let v = ((y.wrapping_mul(37) ^ x.wrapping_mul(17)) & 0xFF) as u8;
                    frame1[idx] = v;
                    frame1[idx + 1] = v.wrapping_add((y & 0xFF) as u8);
                    frame1[idx + 2] = v.wrapping_mul(3);
                    frame1[idx + 3] = 255;
                }
            }
        }

        // Scroll content region (x >= 96, y >= 48) UP by 24 pixels.
        let mut frame2 = frame1.clone();
        let dy = 24u32;
        for y in 48..(h - dy) {
            for x in 96..w {
                let src = (((y + dy) * w + x) * 4) as usize;
                let dst = ((y * w + x) * 4) as usize;
                frame2[dst..dst + 4].copy_from_slice(&frame1[src..src + 4]);
            }
        }
        // Newly revealed strip at the bottom.
        for y in (h - dy)..h {
            for x in 96..w {
                let dst = ((y * w + x) * 4) as usize;
                frame2[dst] = 200;
                frame2[dst + 1] = 180;
                frame2[dst + 2] = 160;
                frame2[dst + 3] = 255;
            }
        }

        let op =
            detect_scroll_copy_rect(&frame1, &frame2, w, h).expect("should detect vertical scroll");
        assert_eq!(op.src_y as i32 - op.dst_y as i32, dy as i32);
        assert!(op.dst_x >= 96);
        assert!(op.dst_y >= 48);

        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        let p1 = enc.encode(w, h, &frame1, 80);
        assert!(p1.is_keyframe);
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(d1, frame1);

        let p2 = enc.encode(w, h, &frame2, 80);
        assert!(!p2.is_keyframe);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, enc.ref_bgra);
        // Purely stationary sidebar tile column (x = 0..64) must remain 100% untouched even before refresh.
        for y in 0..h {
            for x in 0..64u32 {
                let idx = ((y * w + x) * 4) as usize;
                assert_eq!(&d2[idx..idx + 4], &frame2[idx..idx + 4]);
            }
        }
        while let Some(refresh) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &refresh.payload).unwrap();
        }
        assert_eq!(*dec.bgra, frame2);
    }

    #[test]
    fn continuous_motion_uses_lossy_dct_and_idle_refresh_restores_exact_pixels() {
        let w = 256u32;
        let h = 256u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        let make_frame = |phase: u32| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    buf[idx] = ((x + phase * 5) & 0xFF) as u8;
                    buf[idx + 1] = ((y + phase * 3) & 0xFF) as u8;
                    buf[idx + 2] = (((x ^ y) + phase * 7) & 0xFF) as u8;
                    buf[idx + 3] = 255;
                }
            }
            buf
        };

        let f0 = make_frame(0);
        let p0 = enc.encode(w, h, &f0, 75);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        // Consecutive motion frames trigger lossy DCT (I/P tiles) with zero encoder/decoder drift.
        let f1 = make_frame(1);
        let p1 = enc.encode(w, h, &f1, 75);
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(
            d1, enc.ref_bgra,
            "encoder and decoder reference must match bit-for-bit"
        );

        let f2 = make_frame(2);
        let p2 = enc.encode(w, h, &f2, 75);
        assert!(p2.used_lossy);
        assert!(enc.has_pending_lossless_refresh());
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, enc.ref_bgra, "P-tile reference must stay bit-exact");
        // Verify visual quality is close to source.
        let mut total_err: u64 = 0;
        for (a, b) in d2.iter().zip(f2.iter()) {
            total_err += a.abs_diff(*b) as u64;
        }
        let mean_err = total_err as f64 / f2.len() as f64;
        assert!(mean_err < 10.0, "mean channel error too high: {mean_err}");

        // Chunked idle lossless refinement restores 100% exact pixels.
        let mut refresh_chunks = 0usize;
        while let Some(refresh) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &refresh.payload).unwrap();
            refresh_chunks += 1;
        }
        assert!(
            refresh_chunks >= 1,
            "should emit at least one refresh chunk"
        );
        assert!(!enc.has_pending_lossless_refresh());
        assert_eq!(*dec.bgra, f2);
    }

    #[test]
    fn transparent_tiles_use_lossy_alpha_dct_and_refine_losslessly() {
        // Test full-res (128x128), half-res (1600x1000 @ q=60), and quarter-res (1600x1000 @ q=26).
        for &(w, h, q) in &[(128u32, 128u32, 60u8), (1600, 1000, 60), (1600, 1000, 26)] {
            let mut enc = SmartTileEncoder::new();
            let mut dec = SmartTileDecoder::new();

            let make_frame = |step: u32| -> Vec<u8> {
                let mut buf = vec![0u8; (w * h * 4) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let idx = ((y * w + x) * 4) as usize;
                        // Mix of transparent border (A=0), semi-transparent body (A=160), and gradient.
                        let a = if x < 16 || y < 16 {
                            0u8
                        } else {
                            ((120 + ((x + y + step * 7) % 100)) as u8).min(240)
                        };
                        // Valid premultiplied ARGB8888 (B, G, R <= A).
                        let b = (((x + step * 11) & 0xFF) as u8).min(a);
                        let g = (((y + step * 13) & 0xFF) as u8).min(a);
                        let r = ((((x ^ y) + step * 17) & 0xFF) as u8).min(a);
                        buf[idx] = b;
                        buf[idx + 1] = g;
                        buf[idx + 2] = r;
                        buf[idx + 3] = a;
                    }
                }
                buf
            };

            let f0 = make_frame(0);
            let p0 = enc.encode(w, h, &f0, q);
            assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

            // Consecutive motion frames use lossy alpha DCT (I-tile & P-tile) with bit-exact ref.
            let mut saw_lossy = false;
            let mut last_frame = f0;
            for step in 1..=3u32 {
                last_frame = make_frame(step);
                let pkt = enc.encode(w, h, &last_frame, q);
                saw_lossy |= pkt.used_lossy;
                let out = dec.decode(w, h, &pkt.payload).unwrap();
                assert_eq!(
                    out, enc.ref_bgra,
                    "alpha DCT reference drift for {w}x{h} q={q} step={step}"
                );
                // Every reconstructed pixel must remain valid premultiplied ARGB8888 (B, G, R <= A).
                for px in out.as_chunks::<4>().0 {
                    assert!(
                        px[0] <= px[3] && px[1] <= px[3] && px[2] <= px[3],
                        "invalid premultiplied pixel {:?} for {w}x{h} q={q}",
                        px
                    );
                }
            }
            assert!(saw_lossy, "expected lossy alpha DCT for {w}x{h} q={q}");

            // Idle lossless refinement restores 100% bit-exact RGBA pixels.
            while let Some(chunk) =
                enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
            {
                let _ = dec.decode(w, h, &chunk.payload).unwrap();
            }
            assert!(!enc.has_pending_lossless_refresh());
            assert_eq!(*dec.bgra, last_frame);
        }
    }

    #[test]
    fn static_ui_tile_beside_playing_video_settles_without_idle() {
        apply_test_overrides((false, true));
        let w = 1920u32;
        let h = 1080u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();
        let cols = w.div_ceil(TILE_SIZE) as usize;
        let ui_tidx = cols + 1; // tile (1, 1): outside the video region

        let make_frame = |video_seed: u32, ui_on: bool| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            let mut rng = video_seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    let px = if x >= 384 && y >= 192 {
                        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        let n = (rng >> 24) as u8;
                        [
                            n,
                            (n / 2).wrapping_add(x as u8),
                            (n / 3).wrapping_add(y as u8),
                        ]
                    } else if ui_on && (64..128).contains(&x) && (64..128).contains(&y) {
                        // Sharp text-like pattern that DCT cannot reproduce exactly.
                        let on = (x * 7 + y * 3) % 5 < 2;
                        if on {
                            [250, 250, 250]
                        } else {
                            [20, 30, 40]
                        }
                    } else {
                        [35, 35, 40]
                    };
                    buf[idx..idx + 4].copy_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            buf
        };

        let mut decode_and_check = |enc: &SmartTileEncoder, pkt: &SmartTilePacket| -> Vec<u8> {
            let out = dec.decode(w, h, &pkt.payload).unwrap();
            assert_eq!(out, enc.ref_bgra, "decoder must track encoder reference");
            out
        };

        let f = make_frame(0, false);
        let p0 = enc.encode(w, h, &f, 50);
        decode_and_check(&enc, &p0);
        for i in 1..3 {
            let pkt = enc.encode(w, h, &make_frame(i, false), 50);
            decode_and_check(&enc, &pkt);
        }

        // UI tile changes once mid-playback: massive motion makes it lossy (half-res).
        let p = enc.encode(w, h, &make_frame(3, true), 50);
        decode_and_check(&enc, &p);
        assert!(enc.tiles[ui_tidx].is_lossy);
        let changed_at = enc.tiles[ui_tidx].last_change;

        // Video keeps playing; the unchanged UI tile must not be re-encoded every frame.
        for i in 4..7 {
            let pkt = enc.encode(w, h, &make_frame(i, true), 50);
            decode_and_check(&enc, &pkt);
            assert_eq!(
                enc.tiles[ui_tidx].last_change, changed_at,
                "unchanged lossy UI tile was treated as dirty"
            );
        }

        // Once settled, it is refined losslessly inside a regular video frame.
        std::thread::sleep(SETTLED_REFINE_DELAY + Duration::from_millis(20));
        let src = make_frame(7, true);
        let pkt = enc.encode(w, h, &src, 50);
        let out = decode_and_check(&enc, &pkt);
        assert!(!enc.tiles[ui_tidx].is_lossy, "UI tile should be refined");
        for y in 64..128usize {
            let off = (y * w as usize + 64) * 4;
            assert_eq!(
                out[off..off + 256],
                src[off..off + 256],
                "UI row {y} not lossless"
            );
        }
        apply_test_overrides((false, false));
    }

    #[test]
    fn hidpi_progressive_half_res_motion_and_chunked_lossless_refinement() {
        let w = 1920u32;
        let h = 1080u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        let make_frame = |shift: u32| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    if x < 128 || y < 64 {
                        // Stationary toolbar & sidebar
                        buf[idx] = 35;
                        buf[idx + 1] = 35;
                        buf[idx + 2] = 40;
                        buf[idx + 3] = 255;
                    } else {
                        let sy = y + shift;
                        let lum = if (sy % 32) < 16 { 220u8 } else { 30u8 };
                        buf[idx] = lum;
                        buf[idx + 1] = lum;
                        buf[idx + 2] = lum;
                        buf[idx + 3] = 255;
                    }
                }
            }
            buf
        };

        let f0 = make_frame(0);
        let p0 = enc.encode(w, h, &f0, 75);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        let f1 = make_frame(12);
        let p1 = enc.encode(w, h, &f1, 75);
        assert!(p1.used_lossy);
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(
            d1, enc.ref_bgra,
            "half-res DCT reference must match bit-for-bit"
        );

        // Second motion frame exercises half-res P-tiles.
        let f2 = make_frame(24);
        let p2 = enc.encode(w, h, &f2, 75);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(
            d2, enc.ref_bgra,
            "half-res P-tile reference must stay bit-exact"
        );

        // Progressive chunked refinement should take multiple small chunks and end 100% lossless.
        let mut chunks = 0usize;
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            assert!(chunk.dirty_tiles <= DEFAULT_REFRESH_CHUNK_TILES);
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
            chunks += 1;
        }
        assert!(
            chunks >= 2,
            "expected multiple progressive chunks, got {chunks}"
        );
        assert_eq!(*dec.bgra, f2);
    }

    #[test]
    fn quarter_res_congestion_fallback_stays_bit_exact_and_refines_losslessly() {
        let w = 1600u32;
        let h = 1000u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        let make_frame = |phase: u32| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    buf[idx] = ((x.wrapping_mul(3) + phase * 11) & 0xFF) as u8;
                    buf[idx + 1] = ((y.wrapping_mul(5) + phase * 7) & 0xFF) as u8;
                    buf[idx + 2] = (((x ^ y) + phase * 13) & 0xFF) as u8;
                    buf[idx + 3] = 255;
                }
            }
            buf
        };

        let f0 = make_frame(0);
        let p0 = enc.encode(w, h, &f0, 75);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        // Heavy congestion quality (<= 28) on massive HiDPI motion triggers quarter-res DCT.
        let f1 = make_frame(1);
        let p1_half = {
            let mut enc_half = SmartTileEncoder::new();
            let _ = enc_half.encode(w, h, &f0, 75);
            enc_half.encode(w, h, &f1, 48)
        };
        let p1 = enc.encode(w, h, &f1, 28);
        assert!(p1.used_lossy);
        assert!(
            p1.payload.len() < p1_half.payload.len(),
            "quarter-res at q=28 ({}) should be smaller than half-res at q=48 ({})",
            p1.payload.len(),
            p1_half.payload.len()
        );
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(
            d1, enc.ref_bgra,
            "quarter-res I-tile reference must match bit-for-bit"
        );

        // Second congested frame exercises quarter-res P-tiles.
        let f2 = make_frame(2);
        let p2 = enc.encode(w, h, &f2, 26);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(
            d2, enc.ref_bgra,
            "quarter-res P-tile reference must stay bit-exact"
        );

        // Idle lossless refinement restores 100% exact pixels.
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
        }
        assert_eq!(*dec.bgra, f2);
    }

    #[test]
    fn lossless_refresh_respects_byte_budget_on_high_entropy_frames() {
        let w = 640u32;
        let h = 480u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        // Deterministic pseudo-random photographic-like high-entropy noise.
        let make_noisy = |seed_init: u64| -> Vec<u8> {
            let mut s = seed_init;
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for px in buf.as_chunks_mut::<4>().0 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                px[0] = (s >> 16) as u8;
                px[1] = (s >> 24) as u8;
                px[2] = (s >> 32) as u8;
                px[3] = 255;
            }
            buf
        };

        let f0 = make_noisy(1);
        let p0 = enc.encode(w, h, &f0, 70);
        let _ = dec.decode(w, h, &p0.payload).unwrap();

        let f1 = make_noisy(2);
        let p1 = enc.encode(w, h, &f1, 40);
        assert!(p1.used_lossy);
        let _ = dec.decode(w, h, &p1.payload).unwrap();

        let mut max_chunk_bytes = 0usize;
        let mut chunks = 0usize;
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            max_chunk_bytes = max_chunk_bytes.max(chunk.payload.len());
            assert!(
                chunk.dirty_tiles < DEFAULT_REFRESH_CHUNK_TILES,
                "high-entropy chunk should stop early well below 48 tiles, got {}",
                chunk.dirty_tiles
            );
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
            chunks += 1;
        }
        assert!(chunks >= 4, "expected byte-budgeted chunks, got {chunks}");
        assert!(
            max_chunk_bytes <= DEFAULT_REFRESH_MAX_BYTES + 32 * 1024,
            "chunk size {max_chunk_bytes} exceeded budget ceiling"
        );
        assert_eq!(*dec.bgra, f1);
    }

    #[test]
    fn parallel_band_slicing_handles_non_divisible_tile_row_counts() {
        // 1008x310 has 16x5 = 80 tiles (>= 48 dirty tiles -> triggers 4-worker parallel path),
        // where 5 rows / 4 workers leaves the 4th band past the bottom of the framebuffer.
        let w = 1008u32;
        let h = 310u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        let frame = vec![123u8; (w * h * 4) as usize];
        let pkt = enc.encode(w, h, &frame, 80);
        assert!(pkt.is_keyframe);
        let decoded = dec.decode(w, h, &pkt.payload).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn lossless_residual_refinement_reuses_lossy_preview_and_saves_bandwidth() {
        let w = 256u32;
        let h = 256u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        // Smooth photographic-like gradient with subtle noise (realistic video/photo content).
        let make_photo = |phase: u32| -> Vec<u8> {
            let mut s = 0x1234_5678_9abc_def0u64 ^ (phase as u64);
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let jitter = ((s >> 58) as i32) - 3; // [-3, +3] subtle film grain
                    let idx = ((y * w + x) * 4) as usize;
                    buf[idx] = ((x as i32 + phase as i32 * 5 + jitter).clamp(0, 255)) as u8;
                    buf[idx + 1] = ((y as i32 + phase as i32 * 3 + jitter).clamp(0, 255)) as u8;
                    buf[idx + 2] = (((x as i32 + y as i32) / 2 + phase as i32 * 2 + jitter)
                        .clamp(0, 255)) as u8;
                    buf[idx + 3] = 255;
                }
            }
            buf
        };

        let f0 = make_photo(0);
        let p0 = enc.encode(w, h, &f0, 75);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        let f1 = make_photo(1);
        let p1 = enc.encode(w, h, &f1, 75);
        let _ = dec.decode(w, h, &p1.payload).unwrap();

        let f2 = make_photo(2);
        let p2 = enc.encode(w, h, &f2, 75);
        assert!(p2.used_lossy);
        let _ = dec.decode(w, h, &p2.payload).unwrap();

        // Measure total lossless refinement bytes vs. raw LZ4 size of the same frame.
        let raw_lz4_len = lz4_flex::compress_prepend_size(&f2).len();
        let mut total_refresh_bytes = 0usize;
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            total_refresh_bytes += chunk.payload.len();
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
        }
        assert_eq!(
            *dec.bgra, f2,
            "lossless residual refinement must be 100% bit-exact"
        );
        assert!(
            total_refresh_bytes < (raw_lz4_len * 3) / 4,
            "expected Golomb-Rice lossless residual ({total_refresh_bytes} B) to significantly beat raw LZ4 ({raw_lz4_len} B)"
        );
    }

    #[test]
    fn per_tile_mv_skip_and_mv_inter_handle_scroll_without_pulling_stationary_ui() {
        let w = 640u32;
        let h = 512u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        // Stationary left sidebar (x < 100, straddling tile col 1) + stationary top bar (y < 40)
        // + richly textured scrolling content area.
        let make_frame = |scroll_y: u32, jitter: i32| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    if x < 100 || y < 40 {
                        buf[idx] = (40 + ((y * 7 + x * 3) % 50)) as u8;
                        buf[idx + 1] = 42;
                        buf[idx + 2] = 48;
                        buf[idx + 3] = 255;
                    } else {
                        let sy = y + scroll_y;
                        let hi = (sy >> 8).wrapping_mul(19);
                        let v = ((sy.wrapping_mul(37) ^ x.wrapping_mul(17) ^ hi) & 0xFF) as i32;
                        buf[idx] = (v + jitter).clamp(0, 255) as u8;
                        buf[idx + 1] = (((v + ((sy.wrapping_mul(3) + hi) as i32 & 0xFF)) & 0xFF)
                            + jitter)
                            .clamp(0, 255) as u8;
                        buf[idx + 2] = (((v * 3 + hi as i32) & 0xFF) + jitter).clamp(0, 255) as u8;
                        buf[idx + 3] = 255;
                    }
                }
            }
            buf
        };

        let f0 = make_frame(0, 0);
        let p0 = enc.encode(w, h, &f0, 80);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        // 1. Exact scroll by 24px: interior scrolled tiles use 0-byte lossless MODE_MV_SKIP.
        let f1 = make_frame(24, 0);
        let p1 = enc.encode(w, h, &f1, 80);
        assert!(p1.had_copy_rect, "should detect frame scroll MV");
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(d1, enc.ref_bgra);
        // Interior scrolled region (x >= 128, y in 64..h-64) must already be 100% bit-exact lossless!
        for y in 64..(h - 64) {
            for x in 128..w {
                let idx = ((y * w + x) * 4) as usize;
                assert_eq!(&d1[idx..idx + 4], &f1[idx..idx + 4]);
            }
        }
        // Stationary sidebar (x < 100, including straddling tile x=64..100) must not be pulled along.
        for y in 0..h {
            for x in 0..64u32 {
                let idx = ((y * w + x) * 4) as usize;
                assert_eq!(&d1[idx..idx + 4], &f1[idx..idx + 4]);
            }
        }

        // 2. Smooth scroll with subpixel antialiasing jitter (+2): exercises MV-Inter DCT residual.
        let f2 = make_frame(48, 2);
        let p2 = enc.encode(w, h, &f2, 76);
        assert!(p2.had_copy_rect, "should detect smooth-scroll MV");
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(
            d2, enc.ref_bgra,
            "MV-Inter DCT reference must stay bit-exact"
        );

        // Idle lossless refinement restores 100% exact pixels.
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
        }
        assert_eq!(*dec.bgra, f2);
    }

    #[test]
    fn local_2d_mv_handles_diagonal_panning_video_with_bit_exact_reference() {
        let w = 640u32;
        let h = 512u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        // Richly textured 2D landscape panning diagonally by (+9, +7) per frame with subtle film grain.
        let make_pan_frame = |step: i32| -> Vec<u8> {
            let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ (step as u64);
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let grain = ((s >> 61) as i32) - 2; // [-2, +2]
                    let wx = x as i32 - step * 9;
                    let wy = y as i32 - step * 7;
                    // Multi-frequency 2D texture so diagonal pan has a sharp unique SAD minimum.
                    let pat1 = ((wx.wrapping_mul(5) + wy.wrapping_mul(3)) & 63) * 3;
                    let pat2 = (((wx >> 3) ^ (wy >> 3)).wrapping_mul(29) & 127) + 40;
                    let idx = ((y * w + x) * 4) as usize;
                    buf[idx] = (pat1 + grain).clamp(0, 255) as u8;
                    buf[idx + 1] = (pat2 + grain).clamp(0, 255) as u8;
                    buf[idx + 2] = (((pat1 + pat2) / 2) + grain).clamp(0, 255) as u8;
                    buf[idx + 3] = 255;
                }
            }
            buf
        };

        let f0 = make_pan_frame(0);
        let p0 = enc.encode(w, h, &f0, 75);
        assert_eq!(dec.decode(w, h, &p0.payload).unwrap(), f0);

        let f1 = make_pan_frame(1);
        let p1 = enc.encode(w, h, &f1, 60);
        assert!(
            !p1.had_copy_rect,
            "diagonal pan should not trigger 1D scroll MV"
        );
        assert!(p1.used_lossy);
        let raw1 = decompress_lz4_bounded(&p1.payload, MAX_DECODED_FRAME_BYTES).unwrap();
        assert_ne!(
            raw1[1] & FLAG_PREV_REF,
            0,
            "expected diagonal pan to use per-tile 2D local MVs (FLAG_PREV_REF)"
        );
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(
            d1, enc.ref_bgra,
            "2D local MV reference must match bit-for-bit"
        );

        let f2 = make_pan_frame(2);
        let p2 = enc.encode(w, h, &f2, 50);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(
            d2, enc.ref_bgra,
            "consecutive 2D local MV frame must stay bit-exact"
        );

        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
        }
        assert_eq!(*dec.bgra, f2);
    }

    #[test]
    fn decoder_rejects_adversarial_packets_and_handles_tiny_dimensions() {
        let mut dec = SmartTileDecoder::new();

        // 1. Zero or huge dimensions rejected before allocation.
        assert!(dec.decode(0, 64, &[0, 0, 0, 0]).is_err());
        assert!(dec.decode(MAX_FRAME_DIM + 1, 64, &[0, 0, 0, 0]).is_err());
        assert!(dec.decode(16_384, 16_384, &[0, 0, 0, 0]).is_err());

        // 2. LZ4 decompression bomb (declared 4 GiB uncompressed size) rejected before allocation.
        let bomb = [0xFFu8, 0xFF, 0xFF, 0xFF, 0x00, 0x00];
        assert!(dec.decode(64, 64, &bomb).is_err());

        // 3. Uncapped num_tiles DoS rejected against grid maximum.
        let mut bad_tiles = vec![PACKET_VERSION, FLAG_KEYFRAME, 80, 0];
        bad_tiles.extend_from_slice(&10_000u32.to_le_bytes()); // 64x64 has max 1 tile
        let bad_tiles_lz4 = lz4_flex::compress_prepend_size(&bad_tiles);
        assert!(dec.decode(64, 64, &bad_tiles_lz4).is_err());

        // 4. Conflicting MV + LOCAL_MV flags rejected.
        let mut bad_mv = vec![PACKET_VERSION, FLAG_FRAME_MV | FLAG_PREV_REF, 80, 0];
        bad_mv.extend_from_slice(&0i16.to_le_bytes());
        bad_mv.extend_from_slice(&0i16.to_le_bytes());
        bad_mv.extend_from_slice(&1u32.to_le_bytes());
        bad_mv.extend_from_slice(&0u16.to_le_bytes());
        bad_mv.extend_from_slice(&0u16.to_le_bytes());
        bad_mv.push(MODE_DCT_INTER | MODE_FLAG_MV | MODE_FLAG_LOCAL_MV);
        let bad_mv_lz4 = lz4_flex::compress_prepend_size(&bad_mv);
        assert!(dec.decode(64, 64, &bad_mv_lz4).is_err());

        // 5. Crafted huge dc_delta in DCT block cannot panic with i32 overflow.
        let mut overflow_dct = vec![PACKET_VERSION, FLAG_KEYFRAME, 20, 0];
        overflow_dct.extend_from_slice(&1u32.to_le_bytes());
        overflow_dct.extend_from_slice(&0u16.to_le_bytes());
        overflow_dct.extend_from_slice(&0u16.to_le_bytes());
        overflow_dct.push(MODE_DCT_QUARTER_INTRA);
        for _ in 0..6 {
            // 6 blocks for quarter-res (4 Y + 1 Cb + 1 Cr): escaped i16::MAX DC + EOB
            overflow_dct.push(0x80);
            overflow_dct.extend_from_slice(&i16::MAX.to_le_bytes());
            overflow_dct.push(0);
        }
        let overflow_lz4 = lz4_flex::compress_prepend_size(&overflow_dct);
        assert!(dec.decode(64, 64, &overflow_lz4).is_ok());

        // 6. Tiny edge dimensions (1x1, 3x5) roundtrip cleanly without panics.
        for &(w, h) in &[(1u32, 1u32), (3, 5)] {
            let mut enc = SmartTileEncoder::new();
            let mut tiny_dec = SmartTileDecoder::new();
            let f0 = vec![77u8; (w * h * 4) as usize];
            let p0 = enc.encode(w, h, &f0, 75);
            assert_eq!(tiny_dec.decode(w, h, &p0.payload).unwrap(), f0);
        }

        // 7. Public apply_copy_rect safely ignores undersized slices.
        let mut small = vec![0u8; 16];
        apply_copy_rect(
            &mut small,
            64,
            64,
            CopyRectOp {
                dst_x: 0,
                dst_y: 8,
                w: 32,
                h: 32,
                src_x: 0,
                src_y: 0,
            },
        );
    }

    #[test]
    fn simd_kernels_match_scalar_bit_for_bit() {
        let mut rng = 0x853c_49e6_748f_ea9bu64;
        let mut next_u32 = || -> u32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 32) as u32
        };

        // 1. 2D 8x8 DCT and IDCT bit-exactness across 256 random blocks (including full [-255, 255] & [-32768, 32768] bounds).
        for iter in 0..256 {
            let mut input = [0i32; 64];
            for v in &mut input {
                *v = match iter {
                    0 => -255,
                    1 => 255,
                    _ => (next_u32() % 511) as i32 - 255,
                };
            }
            let dct_simd = dct_8x8(&input);
            let dct_ref = simd::dct_8x8_scalar(&input);
            assert_eq!(dct_simd, dct_ref, "DCT mismatch on iteration {iter}");

            let idct_simd = idct_8x8(&dct_simd);
            let idct_ref = simd::idct_8x8_scalar(&dct_ref);
            assert_eq!(idct_simd, idct_ref, "IDCT mismatch on iteration {iter}");
        }

        // Also test full valid DCT output range (e.g. DC up to 32768, AC coefficients in [-4096, 4096]) on IDCT.
        let mut extreme_coeffs = [0i32; 64];
        extreme_coeffs[0] = 32_768;
        for (i, v) in extreme_coeffs.iter_mut().enumerate().skip(1) {
            *v = if i % 2 == 0 { 4096 } else { -4096 };
        }
        assert_eq!(
            idct_8x8(&extreme_coeffs),
            simd::idct_8x8_scalar(&extreme_coeffs)
        );

        // 2. Full-res and half-res BGRA <-> YCbCr 4:2:0 conversions (with and without alpha, full 64x64 and partial tiles).
        let fb_w = 140u32;
        let fb_h = 140u32;
        let mut src_bgra = vec![0u8; (fb_w * fb_h * 4) as usize];
        for b in &mut src_bgra {
            *b = (next_u32() & 0xFF) as u8;
        }

        for &(x0, y0, tw, th) in &[
            (0u32, 0u32, 64u32, 64u32),
            (12, 8, 64, 53),
            (64, 64, 45, 37),
        ] {
            // Full-res forward
            let mut y_simd = [0i32; 64 * 64];
            let mut cb_simd = [0i32; 32 * 32];
            let mut cr_simd = [0i32; 32 * 32];
            let mut y_ref = [0i32; 64 * 64];
            let mut cb_ref = [0i32; 32 * 32];
            let mut cr_ref = [0i32; 32 * 32];

            bgra_tile_to_ycbcr420(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut y_simd,
                &mut cb_simd,
                &mut cr_simd,
            );
            simd::bgra_tile_to_ycbcr420_scalar(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut y_ref,
                &mut cb_ref,
                &mut cr_ref,
            );
            assert_eq!(y_simd, y_ref);
            assert_eq!(cb_simd, cb_ref);
            assert_eq!(cr_simd, cr_ref);

            // Full-res inverse (opaque & transparent)
            let mut a_plane = [0i32; 64 * 64];
            bgra_tile_to_alpha(&src_bgra, fb_w, x0, y0, tw, th, &mut a_plane);

            for alpha_opt in [None, Some(&a_plane)] {
                let mut out_simd = vec![0u8; src_bgra.len()];
                let mut out_ref = vec![0u8; src_bgra.len()];
                ycbcr420_to_bgra_tile(
                    &y_simd,
                    &cb_simd,
                    &cr_simd,
                    alpha_opt,
                    &mut out_simd,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                simd::ycbcr420_to_bgra_tile_scalar(
                    &y_ref,
                    &cb_ref,
                    &cr_ref,
                    alpha_opt,
                    &mut out_ref,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                assert_eq!(out_simd, out_ref);
            }

            // Half-res forward
            let mut hy_simd = [0i32; 32 * 32];
            let mut hcb_simd = [0i32; 16 * 16];
            let mut hcr_simd = [0i32; 16 * 16];
            let mut hy_ref = [0i32; 32 * 32];
            let mut hcb_ref = [0i32; 16 * 16];
            let mut hcr_ref = [0i32; 16 * 16];

            bgra_tile_to_ycbcr420_half(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut hy_simd,
                &mut hcb_simd,
                &mut hcr_simd,
            );
            simd::bgra_tile_to_ycbcr420_half_scalar(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut hy_ref,
                &mut hcb_ref,
                &mut hcr_ref,
            );
            assert_eq!(hy_simd, hy_ref);
            assert_eq!(hcb_simd, hcb_ref);
            assert_eq!(hcr_simd, hcr_ref);

            // Half-res inverse (opaque & transparent)
            let mut ha_plane = [0i32; 32 * 32];
            bgra_tile_to_alpha_half(&src_bgra, fb_w, x0, y0, tw, th, &mut ha_plane);

            for alpha_opt in [None, Some(&ha_plane)] {
                let mut out_simd = vec![0u8; src_bgra.len()];
                let mut out_ref = vec![0u8; src_bgra.len()];
                ycbcr420_half_to_bgra_tile(
                    &hy_simd,
                    &hcb_simd,
                    &hcr_simd,
                    alpha_opt,
                    &mut out_simd,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                simd::ycbcr420_half_to_bgra_tile_scalar(
                    &hy_ref,
                    &hcb_ref,
                    &hcr_ref,
                    alpha_opt,
                    &mut out_ref,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                assert_eq!(out_simd, out_ref);
            }

            // Quarter-res forward & inverse (opaque & transparent)
            let mut qy_simd = [0i32; 16 * 16];
            let mut qcb_simd = [0i32; 8 * 8];
            let mut qcr_simd = [0i32; 8 * 8];
            let mut qy_ref = [0i32; 16 * 16];
            let mut qcb_ref = [0i32; 8 * 8];
            let mut qcr_ref = [0i32; 8 * 8];

            bgra_tile_to_ycbcr420_quarter(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut qy_simd,
                &mut qcb_simd,
                &mut qcr_simd,
            );
            simd::bgra_tile_to_ycbcr420_quarter_scalar(
                &src_bgra,
                fb_w,
                x0,
                y0,
                tw,
                th,
                &mut qy_ref,
                &mut qcb_ref,
                &mut qcr_ref,
            );
            assert_eq!(qy_simd, qy_ref);
            assert_eq!(qcb_simd, qcb_ref);
            assert_eq!(qcr_simd, qcr_ref);

            let mut qa_plane = [0i32; 16 * 16];
            bgra_tile_to_alpha_quarter(&src_bgra, fb_w, x0, y0, tw, th, &mut qa_plane);
            for alpha_opt in [None, Some(&qa_plane)] {
                let mut out_simd = vec![0u8; src_bgra.len()];
                let mut out_ref = vec![0u8; src_bgra.len()];
                ycbcr420_quarter_to_bgra_tile(
                    &qy_simd,
                    &qcb_simd,
                    &qcr_simd,
                    alpha_opt,
                    &mut out_simd,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                simd::ycbcr420_quarter_to_bgra_tile_scalar(
                    &qy_ref,
                    &qcb_ref,
                    &qcr_ref,
                    alpha_opt,
                    &mut out_ref,
                    fb_w,
                    x0,
                    y0,
                    tw,
                    th,
                );
                assert_eq!(out_simd, out_ref);
            }
        }

        // 3. tile_equals_shifted, tile_is_opaque, count_identical_pixels, and xor_encode/decode_row.
        let mut other_bgra = src_bgra.clone();
        for &(tw, th) in &[(64u32, 64u32), (61, 47), (15, 9)] {
            assert!(tile_equals(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th));
            assert_eq!(
                tile_equals(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th),
                simd::tile_equals_scalar(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th)
            );
            assert_eq!(
                count_identical_pixels(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th),
                (tw * th) as usize
            );
        }

        // Flip every 3rd pixel and test count_identical_pixels + tile_equals
        for (i, px) in other_bgra.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if i % 3 == 0 {
                px[1] ^= 0x55;
            }
        }
        for &(tw, th) in &[(64u32, 64u32), (61, 47), (15, 9)] {
            assert_eq!(
                tile_equals(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th),
                simd::tile_equals_scalar(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th)
            );
            assert_eq!(
                count_identical_pixels(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th),
                simd::count_identical_pixels_scalar(&src_bgra, &other_bgra, fb_w, 4, 4, tw, th)
            );
            assert_eq!(
                tile_is_opaque(&src_bgra, fb_w, 4, 4, tw, th),
                simd::tile_is_opaque_scalar(&src_bgra, fb_w, 4, 4, tw, th)
            );
        }

        // Test opaque detection when all alphas are 255 vs. one non-255 pixel at the tail
        let mut opaque_buf = vec![255u8; (fb_w * fb_h * 4) as usize];
        assert!(tile_is_opaque(&opaque_buf, fb_w, 0, 0, 64, 64));
        assert!(tile_is_opaque(&opaque_buf, fb_w, 0, 0, 61, 47));
        // Perturb the very last pixel's alpha in a 61x47 partial tile
        let last_alpha_idx = ((46 * fb_w + 60) * 4 + 3) as usize;
        opaque_buf[last_alpha_idx] = 254;
        assert!(!tile_is_opaque(&opaque_buf, fb_w, 0, 0, 61, 47));
        assert_eq!(
            tile_is_opaque(&opaque_buf, fb_w, 0, 0, 61, 47),
            simd::tile_is_opaque_scalar(&opaque_buf, fb_w, 0, 0, 61, 47)
        );

        // Test XOR encode/decode row across non-multiple-of-32 lengths
        for len in [256usize, 244, 63, 7, 1] {
            let mut ref_simd = src_bgra[..len].to_vec();
            let mut ref_scalar = ref_simd.clone();
            let cur = &other_bgra[..len];
            let mut out_simd = Vec::new();
            let mut out_scalar = Vec::new();

            xor_encode_row(&mut ref_simd, cur, &mut out_simd);
            simd::xor_encode_row_scalar(&mut ref_scalar, cur, &mut out_scalar);
            assert_eq!(out_simd, out_scalar);
            assert_eq!(ref_simd, ref_scalar);

            let mut dec_simd = src_bgra[..len].to_vec();
            xor_decode_row(&mut dec_simd, &out_simd);
            assert_eq!(&dec_simd[..], cur);
        }

        // Test deblock_and_sao_plane SIMD vs Scalar across full-res, half-res, quarter-res, and partial block grids
        for &(pdim, bx, by, q_dc) in &[
            (64usize, 8usize, 8usize, 8i32),
            (64, 7, 5, 24),
            (32, 4, 4, 16),
            (32, 3, 2, 32),
            (16, 2, 2, 40),
            (8, 1, 1, 12),
        ] {
            let mut p_simd = vec![0i32; pdim * pdim];
            for (k, v) in p_simd.iter_mut().enumerate() {
                *v = (((k * 37) ^ ((k >> 3) * 19)) & 0xFF) as i32;
            }
            let mut p_scal = p_simd.clone();
            deblock_and_sao_plane(&mut p_simd, pdim, bx, by, q_dc);
            simd::deblock_and_sao_plane_scalar(&mut p_scal, pdim, bx, by, q_dc);
            assert_eq!(
                p_simd, p_scal,
                "deblock_and_sao_plane SIMD vs Scalar mismatch for pdim={pdim} {bx}x{by} q_dc={q_dc}"
            );
        }

        // Quick throughput sanity check comparing SIMD vs Scalar kernels.
        let iters = 2000usize;
        let sample_block = [42i32; 64];
        let t0 = Instant::now();
        let mut acc_simd = 0i32;
        for _ in 0..iters {
            let d = dct_8x8(&sample_block);
            let id = idct_8x8(&d);
            acc_simd = acc_simd.wrapping_add(id[0]);
        }
        let d_simd = t0.elapsed();

        let t1 = Instant::now();
        let mut acc_scalar = 0i32;
        for _ in 0..iters {
            let d = simd::dct_8x8_scalar(&sample_block);
            let id = simd::idct_8x8_scalar(&d);
            acc_scalar = acc_scalar.wrapping_add(id[0]);
        }
        let d_scalar = t1.elapsed();
        assert_eq!(acc_simd, acc_scalar);

        let mut y_p = [0i32; 64 * 64];
        let mut cb_p = [0i32; 32 * 32];
        let mut cr_p = [0i32; 32 * 32];
        let mut out_fb = src_bgra.clone();
        let c_iters = 400usize;
        let tc0 = Instant::now();
        for _ in 0..c_iters {
            bgra_tile_to_ycbcr420(
                &src_bgra, fb_w, 0, 0, 64, 64, &mut y_p, &mut cb_p, &mut cr_p,
            );
            ycbcr420_to_bgra_tile(&y_p, &cb_p, &cr_p, None, &mut out_fb, fb_w, 0, 0, 64, 64);
        }
        let dc_simd = tc0.elapsed();

        let tc1 = Instant::now();
        for _ in 0..c_iters {
            simd::bgra_tile_to_ycbcr420_scalar(
                &src_bgra, fb_w, 0, 0, 64, 64, &mut y_p, &mut cb_p, &mut cr_p,
            );
            simd::ycbcr420_to_bgra_tile_scalar(
                &y_p,
                &cb_p,
                &cr_p,
                None,
                &mut out_fb,
                fb_w,
                0,
                0,
                64,
                64,
            );
        }
        let dc_scalar = tc1.elapsed();
        println!(
            "SIMD vs Scalar: DCT+IDCT {:?} vs {:?} ({:.2}x), YCbCr420 {:?} vs {:?} ({:.2}x)",
            d_simd,
            d_scalar,
            d_scalar.as_secs_f64() / d_simd.as_secs_f64().max(1e-9),
            dc_simd,
            dc_scalar,
            dc_scalar.as_secs_f64() / dc_simd.as_secs_f64().max(1e-9),
        );

        // Additional kernel benchmarks: Half-res YCbCr, tile_equals, tile_is_opaque, count_identical_pixels, xor_encode/decode
        let mut hy_p = [0i32; 32 * 32];
        let mut hcb_p = [0i32; 16 * 16];
        let mut hcr_p = [0i32; 16 * 16];
        let th0 = Instant::now();
        for _ in 0..c_iters {
            bgra_tile_to_ycbcr420_half(
                &src_bgra, fb_w, 0, 0, 64, 64, &mut hy_p, &mut hcb_p, &mut hcr_p,
            );
            ycbcr420_half_to_bgra_tile(
                &hy_p,
                &hcb_p,
                &hcr_p,
                None,
                &mut out_fb,
                fb_w,
                0,
                0,
                64,
                64,
            );
        }
        let dh_simd = th0.elapsed();

        let th1 = Instant::now();
        for _ in 0..c_iters {
            simd::bgra_tile_to_ycbcr420_half_scalar(
                &src_bgra, fb_w, 0, 0, 64, 64, &mut hy_p, &mut hcb_p, &mut hcr_p,
            );
            simd::ycbcr420_half_to_bgra_tile_scalar(
                &hy_p,
                &hcb_p,
                &hcr_p,
                None,
                &mut out_fb,
                fb_w,
                0,
                0,
                64,
                64,
            );
        }
        let dh_scalar = th1.elapsed();

        // Tile opaque check (10000 tiles)
        opaque_buf[last_alpha_idx] = 255;
        let op_iters = 10_000usize;
        let top0 = Instant::now();
        let mut op_acc = 0usize;
        for _ in 0..op_iters {
            op_acc +=
                std::hint::black_box(tile_is_opaque(&opaque_buf, fb_w, 0, 0, 64, 64)) as usize;
        }
        let dop_simd = top0.elapsed();

        let top1 = Instant::now();
        for _ in 0..op_iters {
            op_acc +=
                std::hint::black_box(simd::tile_is_opaque_scalar(&opaque_buf, fb_w, 0, 0, 64, 64))
                    as usize;
        }
        let dop_scalar = top1.elapsed();
        assert_eq!(op_acc, op_iters * 2);

        // Count identical pixels (5000 tiles)
        let id_iters = 5_000usize;
        let tid0 = Instant::now();
        let mut id_acc_s = 0usize;
        for _ in 0..id_iters {
            id_acc_s += std::hint::black_box(count_identical_pixels(
                &src_bgra,
                &other_bgra,
                fb_w,
                0,
                0,
                64,
                64,
            ));
        }
        let did_simd = tid0.elapsed();

        let tid1 = Instant::now();
        let mut id_acc_r = 0usize;
        for _ in 0..id_iters {
            id_acc_r += std::hint::black_box(simd::count_identical_pixels_scalar(
                &src_bgra,
                &other_bgra,
                fb_w,
                0,
                0,
                64,
                64,
            ));
        }
        let did_scalar = tid1.elapsed();
        assert_eq!(id_acc_s, id_acc_r);

        // 1080p full-frame XOR delta encode + decode (1920x1080 * 4 = 8.29 MB, 100 frames)
        let xor_len = 1920 * 1080 * 4;
        let mut xor_ref = vec![0xAAu8; xor_len];
        let xor_cur = vec![0x55u8; xor_len];
        let mut xor_out = Vec::with_capacity(xor_len);
        let xor_iters = 100usize;
        let tx0 = Instant::now();
        for _ in 0..xor_iters {
            xor_out.clear();
            xor_encode_row(
                std::hint::black_box(&mut xor_ref),
                std::hint::black_box(&xor_cur),
                &mut xor_out,
            );
            xor_decode_row(
                std::hint::black_box(&mut xor_ref),
                std::hint::black_box(&xor_out),
            );
        }
        let dx_simd = tx0.elapsed();

        let tx1 = Instant::now();
        for _ in 0..xor_iters {
            xor_out.clear();
            simd::xor_encode_row_scalar(
                std::hint::black_box(&mut xor_ref),
                std::hint::black_box(&xor_cur),
                &mut xor_out,
            );
            simd::xor_decode_row_scalar(
                std::hint::black_box(&mut xor_ref),
                std::hint::black_box(&xor_out),
            );
        }
        let dx_scalar = tx1.elapsed();

        println!(
            "Half-res YCbCr420: {:?} vs {:?} ({:.2}x)\n\
             tile_is_opaque (10k tiles): {:?} vs {:?} ({:.2}x)\n\
             count_identical_pixels (5k tiles): {:?} vs {:?} ({:.2}x)\n\
             1080p XOR encode+decode (100 frames): {:?} vs {:?} ({:.2}x)",
            dh_simd,
            dh_scalar,
            dh_scalar.as_secs_f64() / dh_simd.as_secs_f64().max(1e-9),
            dop_simd,
            dop_scalar,
            dop_scalar.as_secs_f64() / dop_simd.as_secs_f64().max(1e-9),
            did_simd,
            did_scalar,
            did_scalar.as_secs_f64() / did_simd.as_secs_f64().max(1e-9),
            dx_simd,
            dx_scalar,
            dx_scalar.as_secs_f64() / dx_simd.as_secs_f64().max(1e-9),
        );

        // End-to-end 1080p (1920x1080) SmartTileEncoder & SmartTileDecoder benchmark:
        // 1. Full-res 1080p lossy DCT motion sequence (q = 75)
        // 2. Half-res 1080p lossy DCT motion sequence (q = 50)
        // 3. Stationary/partial UI frame diffing (95% unchanged tiles)
        let w = 1920u32;
        let h = 1080u32;
        let frames: Vec<Vec<u8>> = (0..6u32)
            .map(|step| {
                let mut buf = vec![0u8; (w * h * 4) as usize];
                for y in 0..h {
                    let row = (y * w * 4) as usize;
                    for x in 0..w {
                        let idx = row + (x * 4) as usize;
                        buf[idx] = ((x.wrapping_mul(3) + step * 11) & 0xFF) as u8;
                        buf[idx + 1] = ((y.wrapping_mul(5) + step * 7) & 0xFF) as u8;
                        buf[idx + 2] = (((x ^ y) + step * 13) & 0xFF) as u8;
                        buf[idx + 3] = 255;
                    }
                }
                buf
            })
            .collect();

        let run_e2e = |force_scalar: bool,
                       disable_gpu: bool,
                       quality: u8|
         -> (Duration, Duration, usize, usize, Vec<u8>) {
            struct OverrideReset;
            impl Drop for OverrideReset {
                fn drop(&mut self) {
                    apply_test_overrides((false, false));
                }
            }
            let _reset = OverrideReset;
            apply_test_overrides((force_scalar, disable_gpu));
            let mut enc = SmartTileEncoder::new();
            let mut dec = SmartTileDecoder::new();
            // Prime with keyframe + first motion frame
            let p0 = enc.encode(w, h, &frames[0], quality);
            let _ = dec.decode(w, h, &p0.payload).unwrap();
            let p1 = enc.encode(w, h, &frames[1], quality);
            let _ = dec.decode(w, h, &p1.payload).unwrap();

            let mut enc_time = Duration::ZERO;
            let mut dec_time = Duration::ZERO;
            let mut total_pkt_bytes = 0usize;
            let mut total_lz4_only_bytes = 0usize;
            let mut last_dec = Vec::new();
            for f in &frames[2..6] {
                let t_e = Instant::now();
                let pkt = enc.encode(w, h, f, quality);
                enc_time += t_e.elapsed();
                total_pkt_bytes += pkt.payload.len();
                let raw = decompress_lz4_bounded(&pkt.payload, MAX_DECODED_FRAME_BYTES).unwrap();
                total_lz4_only_bytes += lz4_flex::compress_prepend_size(&raw).len();

                let t_d = Instant::now();
                let (buf, _) = dec.decode_with_damage(w, h, &pkt.payload).unwrap();
                dec_time += t_d.elapsed();
                last_dec = (*buf).clone();
            }
            drop(_reset);
            (
                enc_time / 4,
                dec_time / 4,
                total_pkt_bytes / 4,
                total_lz4_only_bytes / 4,
                last_dec,
            )
        };

        let (e_full_gpu, _d_full_gpu, pkt_full_bytes, lz4_full_bytes, out_full_gpu) =
            run_e2e(false, false, 75);
        let (e_full_simd, d_full_simd, _, _, out_full_simd) = run_e2e(false, true, 75);
        let (e_full_scal, d_full_scal, _, _, out_full_scal) = run_e2e(true, true, 75);
        assert_eq!(
            out_full_simd, out_full_scal,
            "end-to-end 1080p full-res SIMD vs Scalar must be bit-identical"
        );
        assert_eq!(
            out_full_gpu, out_full_simd,
            "end-to-end 1080p full-res GPU vs SIMD must be bit-identical"
        );

        let (e_half_simd, d_half_simd, pkt_half_bytes, lz4_half_bytes, out_half_simd) =
            run_e2e(false, true, 50);
        let (e_half_scal, d_half_scal, _, _, out_half_scal) = run_e2e(true, true, 50);
        assert_eq!(
            out_half_simd, out_half_scal,
            "end-to-end 1080p half-res must be bit-identical"
        );

        if let Some(eng) = gpu::global_gpu_engine() {
            if let Ok(lt) = eng.last_timing.lock() {
                println!(
                    "GPU 1080p Stage Breakdown: PCIe Upload {:?} | GPU Compute+Readback {:?} | CPU RLE+Scatter {:?}",
                    lt.0, lt.1, lt.2
                );
            }
        }
        println!(
            "1080p Full-Res Motion (per frame): Encode GPU {:?} | SIMD {:?} | Scalar {:?} (GPU {:.2}x vs Scalar, {:.2}x vs SIMD) | Decode SIMD {:?} vs Scalar {:?} ({:.2}x) | Bandwidth: {} B (rANS) vs {} B (LZ4)\n\
             1080p Half-Res Motion (per frame): Encode SIMD {:?} vs Scalar {:?} ({:.2}x) | Decode SIMD {:?} vs Scalar {:?} ({:.2}x) | Bandwidth: {} B (rANS) vs {} B (LZ4)",
            e_full_gpu, e_full_simd, e_full_scal,
            e_full_scal.as_secs_f64() / e_full_gpu.as_secs_f64().max(1e-9),
            e_full_simd.as_secs_f64() / e_full_gpu.as_secs_f64().max(1e-9),
            d_full_simd, d_full_scal, d_full_scal.as_secs_f64() / d_full_simd.as_secs_f64().max(1e-9),
            pkt_full_bytes, lz4_full_bytes,
            e_half_simd, e_half_scal, e_half_scal.as_secs_f64() / e_half_simd.as_secs_f64().max(1e-9),
            d_half_simd, d_half_scal, d_half_scal.as_secs_f64() / d_half_simd.as_secs_f64().max(1e-9),
            pkt_half_bytes, lz4_half_bytes,
        );
    }

    #[test]
    fn gpu_compute_shaders_match_cpu_bit_for_bit() {
        let Some(engine) = gpu::SmartTileGpuEngine::try_new_any() else {
            println!("No Vulkan/GLES wgpu adapter available in test environment; skipping GPU shader test");
            return;
        };
        println!(
            "Testing SmartTile GPU compute shaders on adapter '{}' ({:?})",
            engine.adapter_name, engine.device_type
        );

        let mut rng = 0x7654_3210_fedc_ba98u64;
        let mut next_u32 = || -> u32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 32) as u32
        };

        // 1. Verify `batch_dct_quant_idct` and `batch_idct` against scalar `dct_8x8` + `quantize_div` + `idct_8x8`.
        let num_blocks = 256usize;
        let quant = build_quant_table(&BASE_LUMA_QUANT, 75);
        let mut in_blocks = vec![[0i32; 64]; num_blocks];
        for blk in &mut in_blocks {
            for v in blk.iter_mut() {
                *v = (next_u32() % 511) as i32 - 255;
            }
        }
        let mut gpu_q = vec![[0i32; 64]; num_blocks];
        let mut gpu_idct = vec![[0i32; 64]; num_blocks];
        assert!(engine.batch_dct_quant_idct(&in_blocks, &quant, &mut gpu_q, &mut gpu_idct));

        let mut dequant_blocks = vec![[0i32; 64]; num_blocks];
        for i in 0..num_blocks {
            let cpu_dct = simd::dct_8x8_scalar(&in_blocks[i]);
            let mut cpu_q = [0i32; 64];
            let mut cpu_deq = [0i32; 64];
            for k in 0..64 {
                let qv = quantize_div(cpu_dct[k], quant[k]);
                cpu_q[k] = qv;
                cpu_deq[k] = (qv * quant[k]).clamp(-32_768, 32_768);
            }
            let cpu_idct = simd::idct_8x8_scalar(&cpu_deq);
            assert_eq!(
                gpu_q[i], cpu_q,
                "GPU vs CPU quantized DCT block mismatch at {i}"
            );
            assert_eq!(
                gpu_idct[i], cpu_idct,
                "GPU vs CPU IDCT block mismatch at {i}"
            );
            dequant_blocks[i] = cpu_deq;
        }

        let mut gpu_idct_only = vec![[0i32; 64]; num_blocks];
        assert!(engine.batch_idct(&dequant_blocks, &mut gpu_idct_only));
        assert_eq!(
            gpu_idct_only, gpu_idct,
            "GPU batch_idct_only must match batch_dct_quant_idct"
        );

        // 2. Verify `scan_dirty_and_opaque_tiles` on a 300x200 frame (5x4 = 20 tiles, including partial edge tiles).
        let w = 300u32;
        let h = 200u32;
        let ref_fb = vec![255u8; (w * h * 4) as usize];
        let mut cur_fb = ref_fb.clone();
        // Mark tile (1, 1) dirty (opaque)
        cur_fb[((70 * w + 70) * 4) as usize] = 12;
        // Mark partial edge tile (4, 3) transparent (and dirty)
        cur_fb[((195 * w + 295) * 4 + 3) as usize] = 128;

        let flags = engine
            .scan_dirty_and_opaque_tiles(&ref_fb, &cur_fb, w, h)
            .expect("scan_dirty_and_opaque_tiles should succeed");
        let cols = w.div_ceil(TILE_SIZE) as usize;
        let rows = h.div_ceil(TILE_SIZE) as usize;
        assert_eq!(flags.len(), cols * rows);
        for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (h - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let x0 = (tx as u32) * TILE_SIZE;
                let tw = (w - x0).min(TILE_SIZE);
                let tidx = ty * cols + tx;
                let cpu_dirty = !simd::tile_equals_scalar(&ref_fb, &cur_fb, w, x0, y0, tw, th);
                let cpu_opaque = simd::tile_is_opaque_scalar(&cur_fb, w, x0, y0, tw, th);
                let gpu_dirty = (flags[tidx] & 1) != 0;
                let gpu_opaque = (flags[tidx] & 2) != 0;
                assert_eq!(gpu_dirty, cpu_dirty, "dirty mismatch at tile ({tx}, {ty})");
                assert_eq!(
                    gpu_opaque, cpu_opaque,
                    "opaque mismatch at tile ({tx}, {ty})"
                );
            }
        }

        // 3. Verify full end-to-end `encode_tiles_with_gpu` vs CPU `encode_tile_batch` bit-exactness!
        let ew = 640u32;
        let eh = 512u32;
        let make_frame = |step: u32| -> Vec<u8> {
            let mut buf = vec![0u8; (ew * eh * 4) as usize];
            for y in 0..eh {
                for x in 0..ew {
                    let idx = ((y * ew + x) * 4) as usize;
                    buf[idx] = ((x.wrapping_mul(5) + step * 13) & 0xFF) as u8;
                    buf[idx + 1] = ((y.wrapping_mul(7) + step * 9) & 0xFF) as u8;
                    buf[idx + 2] = (((x ^ y) + step * 17) & 0xFF) as u8;
                    buf[idx + 3] = 255;
                }
            }
            buf
        };
        let f0 = make_frame(0);
        let f1 = make_frame(1);
        let f2 = make_frame(2);

        // Encode sequence with GPU enabled (using `engine` directly if global_gpu_engine skipped llvmpipe)
        let (p1_cpu, p2_cpu, enc_cpu) = {
            struct OverrideReset;
            impl Drop for OverrideReset {
                fn drop(&mut self) {
                    apply_test_overrides((false, false));
                }
            }
            let _reset = OverrideReset;
            apply_test_overrides((false, true));
            let mut enc_cpu = SmartTileEncoder::new();
            let _ = enc_cpu.encode(ew, eh, &f0, 75);
            let p1_cpu = enc_cpu.encode(ew, eh, &f1, 75);
            let p2_cpu = enc_cpu.encode(ew, eh, &f2, 75);
            (p1_cpu, p2_cpu, enc_cpu)
        };

        // Now run the exact same sequence through `encode_tiles_with_gpu` using `engine`.
        let mut enc_gpu = SmartTileEncoder::new();
        let _ = enc_gpu.encode(ew, eh, &f0, 75);
        for (frame, cpu_pkt) in [(&f1, &p1_cpu), (&f2, &p2_cpu)] {
            let ecols = ew.div_ceil(TILE_SIZE) as usize;
            let erows = eh.div_ceil(TILE_SIZE) as usize;
            let mut dirty = Vec::new();
            let now = Instant::now();
            for ty in 0..erows {
                let y0 = (ty as u32) * TILE_SIZE;
                let th = (eh - y0).min(TILE_SIZE);
                for tx in 0..ecols {
                    let x0 = (tx as u32) * TILE_SIZE;
                    let tw = (ew - x0).min(TILE_SIZE);
                    let tidx = ty * ecols + tx;
                    if !tile_equals(&enc_gpu.ref_bgra, frame, ew, x0, y0, tw, th) {
                        enc_gpu.tiles[tidx].motion_streak =
                            enc_gpu.tiles[tidx].motion_streak.saturating_add(1);
                        enc_gpu.tiles[tidx].last_change = Some(now);
                        dirty.push((tx, ty, x0, y0, tw, th, tidx));
                    }
                }
            }
            enc_gpu.prev_ref_bgra.resize(frame.len(), 0);
            enc_gpu.prev_ref_bgra.copy_from_slice(&enc_gpu.ref_bgra);
            let lq = build_quant_table(&BASE_LUMA_QUANT, 75);
            let cq = build_quant_table(&BASE_CHROMA_QUANT, 75);
            let mut raw_pkt = vec![PACKET_VERSION, 0, 75, 0];
            raw_pkt.extend_from_slice(&(dirty.len() as u32).to_le_bytes());
            let (used_lossy_gpu, used_lmv) = encode_tiles_with_gpu(
                &engine,
                &mut enc_gpu.ref_bgra,
                &mut enc_gpu.prev_ref_bgra,
                true,
                frame,
                &mut enc_gpu.tiles,
                ew,
                eh,
                &dirty,
                true,
                None,
                &lq,
                &cq,
                &mut raw_pkt,
                || {},
            )
            .expect("GPU batch encode should succeed");
            if used_lmv {
                raw_pkt[1] |= FLAG_PREV_REF;
            }
            let gpu_payload = compress_packet_adaptive(&raw_pkt, used_lossy_gpu);
            assert_eq!(
                gpu_payload, cpu_pkt.payload,
                "GPU compute shader encoded packet must be 100% bit-identical to CPU!"
            );
        }
        assert_eq!(
            enc_gpu.ref_bgra, enc_cpu.ref_bgra,
            "GPU and CPU reference framebuffers must match bit-for-bit!"
        );
    }

    #[test]
    fn subpixel_hpel_motion_and_rans_entropy_coding_are_bit_exact() {
        // 1. Verify 4-way interleaved rANS roundtrip across empty, constant, sparse, and dense random payloads.
        assert_eq!(
            rans::decompress_rans_bounded(&rans::compress_rans_prepend_size(&[]), 1024).unwrap(),
            Vec::<u8>::new()
        );
        let const_buf = vec![42u8; 777];
        assert_eq!(
            rans::decompress_rans_bounded(&rans::compress_rans_prepend_size(&const_buf), 1024)
                .unwrap(),
            const_buf
        );

        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut skew_buf = Vec::with_capacity(8195);
        for _ in 0..8195 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 56) as u8;
            // Realistic skewed DCT RLE symbol distribution (zeros & small tokens dominate)
            skew_buf.push(if r < 150 {
                0
            } else if r < 210 {
                1
            } else if r < 240 {
                255
            } else {
                r
            });
        }
        let rans_comp = rans::compress_rans_prepend_size(&skew_buf);
        let lz4_comp = lz4_flex::compress_prepend_size(&skew_buf);
        assert!(
            rans_comp.len() < lz4_comp.len(),
            "expected rANS ({} B) to compress skewed DCT distribution smaller than LZ4 ({} B)",
            rans_comp.len(),
            lz4_comp.len()
        );
        assert_eq!(
            rans::decompress_rans_bounded(&rans_comp, 16384).unwrap(),
            skew_buf
        );

        // 2. Verify fractional half-pixel (`1.5 px`, `-2.5 px`) diagonal panning video across GPU, AVX2 SIMD, and Scalar!
        let w = 640u32;
        let h = 512u32;
        let world_w = (w * 2 + 128) as usize;
        let world_h = (h * 2 + 128) as usize;
        let mut world = vec![0u8; world_w * world_h * 3];
        for wy in 0..world_h {
            for wx in 0..world_w {
                let idx = (wy * world_w + wx) * 3;
                let v0 = (((wx * 3 + wy * 5) >> 1) & 0xFF) as u8;
                let v1 = (((wx ^ wy) >> 1) & 0xFF) as u8;
                let v2 = (((wx * 7).wrapping_sub(wy * 3) >> 2) & 0xFF) as u8;
                world[idx] = v0;
                world[idx + 1] = v1;
                world[idx + 2] = v2;
            }
        }
        // Sample `world` at half-pixel shifts `(step * 3, step * 5)` (i.e. +1.5 px, +2.5 px per frame!)
        let sample_hpel_frame = |step: usize| -> Vec<u8> {
            let ox = 32 + step * 3;
            let oy = 32 + step * 5;
            let mut fb = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let wx = ox + x * 2;
                    let wy = oy + y * 2;
                    let widx = (wy * world_w + wx) * 3;
                    let didx = (y * w as usize + x) * 4;
                    fb[didx] = world[widx];
                    fb[didx + 1] = world[widx + 1];
                    fb[didx + 2] = world[widx + 2];
                    fb[didx + 3] = 255;
                }
            }
            fb
        };

        let f0 = sample_hpel_frame(0);
        let f1 = sample_hpel_frame(1);
        let f2 = sample_hpel_frame(2);

        let (p1_scal, p2_scal, enc_scal, p1_simd, p2_simd, enc_simd) = {
            struct OverrideReset;
            impl Drop for OverrideReset {
                fn drop(&mut self) {
                    apply_test_overrides((false, false));
                }
            }
            let _reset = OverrideReset;
            apply_test_overrides((true, true));
            let mut enc_scal = SmartTileEncoder::new();
            let _ = enc_scal.encode(w, h, &f0, 75);
            let p1_scal = enc_scal.encode(w, h, &f1, 75);
            let p2_scal = enc_scal.encode(w, h, &f2, 75);

            apply_test_overrides((false, true));
            let mut enc_simd = SmartTileEncoder::new();
            let _ = enc_simd.encode(w, h, &f0, 75);
            let p1_simd = enc_simd.encode(w, h, &f1, 75);
            let p2_simd = enc_simd.encode(w, h, &f2, 75);
            (p1_scal, p2_scal, enc_scal, p1_simd, p2_simd, enc_simd)
        };

        assert_eq!(
            p1_simd.payload, p1_scal.payload,
            "Half-pel SIMD vs Scalar packet 1 mismatch"
        );
        assert_eq!(
            p2_simd.payload, p2_scal.payload,
            "Half-pel SIMD vs Scalar packet 2 mismatch"
        );
        assert_eq!(enc_simd.ref_bgra, enc_scal.ref_bgra);

        let mut dec_clean = SmartTileDecoder::new();
        let mut enc_gpu = SmartTileEncoder::new();
        let p0_gpu = enc_gpu.encode(w, h, &f0, 75);
        let d0 = dec_clean.decode(w, h, &p0_gpu.payload).unwrap();
        assert_eq!(
            d0, enc_gpu.ref_bgra,
            "Decoded frame 0 must match GPU encoder reference"
        );

        let p1_gpu = enc_gpu.encode(w, h, &f1, 75);
        let d1 = dec_clean.decode(w, h, &p1_gpu.payload).unwrap();
        assert_eq!(
            d1, enc_gpu.ref_bgra,
            "Decoded frame 1 must match GPU encoder reference"
        );

        let p2_gpu = enc_gpu.encode(w, h, &f2, 75);
        let d2 = dec_clean.decode(w, h, &p2_gpu.payload).unwrap();

        assert_eq!(
            p1_gpu.payload, p1_simd.payload,
            "Half-pel GPU vs SIMD packet 1 mismatch"
        );
        assert_eq!(
            p2_gpu.payload, p2_simd.payload,
            "Half-pel GPU vs SIMD packet 2 mismatch"
        );
        assert_eq!(
            d2, enc_gpu.ref_bgra,
            "Decoded frame 2 must match GPU encoder reference"
        );
        assert_eq!(
            d2, enc_scal.ref_bgra,
            "Decoded frame 2 must match Scalar encoder reference"
        );

        // Verify that the lossy half-pel packets actually engaged two-stage LZ4 + rANS entropy coding!
        let hdr1 = u32::from_le_bytes([
            p1_gpu.payload[0],
            p1_gpu.payload[1],
            p1_gpu.payload[2],
            p1_gpu.payload[3],
        ]);
        assert_ne!(
            hdr1 & rans::RANS_MAGIC_BIT,
            0,
            "Expected lossy SmartTilePacket to select LZ4 + 4-way rANS compression"
        );
    }

    #[test]
    fn vram_static_ui_tile_beside_playing_video_settles_without_idle() {
        let Some(engine) = gpu::SmartTileGpuEngine::try_new_any().map(Arc::new) else {
            eprintln!("Skipping VRAM settle test: no wgpu adapter available");
            return;
        };
        let (w, h) = (1920u32, 1080u32);
        let tex = engine.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test_vram_settle_tex"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let upload = |bgra: &[u8]| {
            engine.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bgra,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        };
        let frame = vram::GpuTextureFrame {
            engine: Arc::clone(&engine),
            texture: tex.clone(),
            view,
            crop_x: 0,
            crop_y: 0,
            width: w,
            height: h,
            rb_swizzle: false,
            external_memory: false,
        };
        let make_frame = |video_seed: u32, ui_on: bool| -> Vec<u8> {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            let mut rng = video_seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 4) as usize;
                    let px = if x >= 384 && y >= 192 {
                        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        let n = (rng >> 24) as u8;
                        [
                            n,
                            (n / 2).wrapping_add(x as u8),
                            (n / 3).wrapping_add(y as u8),
                        ]
                    } else if ui_on && (64..128).contains(&x) && (64..128).contains(&y) {
                        let on = (x * 7 + y * 3) % 5 < 2;
                        if on {
                            [250, 250, 250]
                        } else {
                            [20, 30, 40]
                        }
                    } else {
                        [35, 35, 40]
                    };
                    buf[idx..idx + 4].copy_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            buf
        };

        let cols = w.div_ceil(TILE_SIZE) as usize;
        let ui_tidx = cols + 1;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();
        let mut step = |enc: &mut SmartTileEncoder, src: &[u8]| -> Vec<u8> {
            upload(src);
            let pkt = enc.encode_gpu_texture(&frame, 50).unwrap();
            dec.decode(w, h, &pkt.payload).unwrap().to_vec()
        };

        for i in 0..3 {
            step(&mut enc, &make_frame(i, false));
        }
        step(&mut enc, &make_frame(3, true));
        assert!(enc.vram_resident);
        assert!(enc.tiles[ui_tidx].is_lossy);
        let changed_at = enc.tiles[ui_tidx].last_change;
        for i in 4..7 {
            step(&mut enc, &make_frame(i, true));
            assert_eq!(
                enc.tiles[ui_tidx].last_change, changed_at,
                "unchanged lossy UI tile was treated as dirty (VRAM)"
            );
        }
        std::thread::sleep(SETTLED_REFINE_DELAY + Duration::from_millis(20));
        let src = make_frame(7, true);
        let out = step(&mut enc, &src);
        assert!(
            !enc.tiles[ui_tidx].is_lossy,
            "UI tile should be refined (VRAM)"
        );
        for y in 64..128usize {
            let off = (y * w as usize + 64) * 4;
            assert_eq!(
                out[off..off + 256],
                src[off..off + 256],
                "UI row {y} not lossless"
            );
        }
        // Remaining lossy video converges bit-exactly via idle refresh.
        while let Some(pkt) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            let _ = dec.decode(w, h, &pkt.payload).unwrap();
        }
        assert!(*dec.bgra == src, "VRAM stream must converge losslessly");
    }

    #[test]
    fn vram_resident_gpu_texture_encoding_is_bit_exact_and_zero_cpu_buffer() {
        let Some(engine) = gpu::SmartTileGpuEngine::try_new_any().map(Arc::new) else {
            eprintln!("Skipping VRAM texture test: no wgpu adapter available");
            return;
        };

        let (crop_x, crop_y) = (16u32, 12u32);
        let (w, h) = (1600u32, 1024u32);
        let (tex_w, tex_h) = (w + crop_x + 8, h + crop_y + 8);

        let tex = engine.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test_vram_win_tex"),
            size: wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let upload_window_to_tex = |win_bgra: &[u8]| {
            let mut padded = vec![77u8; (tex_w * tex_h * 4) as usize];
            for y in 0..h as usize {
                let s = y * (w as usize) * 4;
                let d = ((crop_y as usize + y) * (tex_w as usize) + (crop_x as usize)) * 4;
                padded[d..d + (w as usize) * 4].copy_from_slice(&win_bgra[s..s + (w as usize) * 4]);
            }
            engine.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_w * 4),
                    rows_per_image: Some(tex_h),
                },
                wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: 1,
                },
            );
        };

        let make_scene = |shift_x: i32, shift_y: i32| -> Vec<u8> {
            let mut fb = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let idx = ((y as u32 * w + x as u32) * 4) as usize;
                    // Top-left 128x128: solid UI header; bottom-right 64x64: translucent alpha badge
                    if x < 128 && y < 128 {
                        fb[idx] = 40;
                        fb[idx + 1] = 44;
                        fb[idx + 2] = 52;
                        fb[idx + 3] = 255;
                    } else if x >= (w as i32 - 64) && y >= (h as i32 - 64) {
                        let a = 180u8;
                        fb[idx] = (((x + shift_x) * 3) as u8).min(a);
                        fb[idx + 1] = (((y + shift_y) * 5) as u8).min(a);
                        fb[idx + 2] = 120u8.min(a);
                        fb[idx + 3] = a;
                    } else {
                        let sx = (x + shift_x).rem_euclid(512);
                        let sy = (y + shift_y).rem_euclid(512);
                        fb[idx] = ((sx * 5 + sy * 3) & 0xFF) as u8;
                        fb[idx + 1] = (((sx * 7) ^ (sy * 11)) & 0xFF) as u8;
                        fb[idx + 2] = ((sx * 13 + sy * 17) & 0xFF) as u8;
                        fb[idx + 3] = 255;
                    }
                }
            }
            fb
        };

        let gpu_frame = vram::GpuTextureFrame {
            engine: Arc::clone(&engine),
            texture: tex.clone(),
            view,
            crop_x,
            crop_y,
            width: w,
            height: h,
            rb_swizzle: false,
            external_memory: false,
        };

        let assert_frames_eq = |got: &[u8], exp: &[u8], label: &str| {
            assert_eq!(got.len(), exp.len(), "{label}: len mismatch");
            if got != exp {
                for i in (0..got.len()).step_by(4) {
                    if got[i..i + 4] != exp[i..i + 4] {
                        let px_idx = (i / 4) as u32;
                        let x = px_idx % w;
                        let y = px_idx / w;
                        panic!(
                            "{label}: first pixel mismatch at ({x}, {y}) tile=({}, {}): got={:?} exp={:?}",
                            x / 64,
                            y / 64,
                            &got[i..i + 4],
                            &exp[i..i + 4]
                        );
                    }
                }
            }
        };

        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        // 1. Keyframe (solid + lossless + alpha)
        let f0 = make_scene(0, 0);
        upload_window_to_tex(&f0);
        let pkt0 = enc.encode_gpu_texture(&gpu_frame, 75).unwrap();
        assert!(pkt0.is_keyframe);
        assert_eq!(
            enc.ref_bgra.capacity(),
            0,
            "CPU ref_bgra must stay unallocated"
        );
        assert_eq!(
            enc.source_bgra.capacity(),
            0,
            "CPU source_bgra must stay unallocated"
        );
        let dec0 = dec.decode(w, h, &pkt0.payload).unwrap().to_vec();
        assert_frames_eq(&dec0, &f0, "Keyframe");

        // 2. Small 1-tile cursor blink update
        let mut f_cursor = f0.clone();
        for py in 16..32 {
            for px in 16..24 {
                let idx = ((py * w + px) * 4) as usize;
                f_cursor[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        upload_window_to_tex(&f_cursor);
        let pkt_cur = enc.encode_gpu_texture(&gpu_frame, 75).unwrap();
        assert_eq!(pkt_cur.dirty_tiles, 1);
        let dec_cur = dec.decode(w, h, &pkt_cur.payload).unwrap().to_vec();
        assert_frames_eq(&dec_cur, &f_cursor, "Single-tile UI update");

        // 3. Vertical scroll (+32 px) -> exercises GPU scroll detector + MV_SKIP
        let mut f_scroll = f_cursor.clone();
        let stride = (w * 4) as usize;
        for y in (160..(h as usize - 96)).rev() {
            let src = (y - 32) * stride + 128 * 4;
            let dst = y * stride + 128 * 4;
            let len = (w as usize - 256) * 4;
            f_scroll.copy_within(src..src + len, dst);
        }
        upload_window_to_tex(&f_scroll);
        let pkt_scr = enc.encode_gpu_texture(&gpu_frame, 75).unwrap();
        assert!(
            pkt_scr.had_copy_rect,
            "Expected GPU scroll summary to detect vertical scroll"
        );
        let dec_scr = dec.decode(w, h, &pkt_scr.payload).unwrap().to_vec();
        let vram_ref_scr = enc
            .vram_state
            .as_ref()
            .unwrap()
            .download_ref_bgra()
            .unwrap();
        assert_frames_eq(
            &dec_scr,
            &vram_ref_scr,
            "Zero drift after GPU scroll encode",
        );

        // 4. Full-res (Q=75), Half-res (Q=50), and Quarter-res (Q=26) motion frames + 1/2-pel local MV!
        let mut last_src = f_scroll;
        for &(step, q) in &[(1, 75u8), (2, 75u8), (3, 50u8), (4, 26u8), (5, 78u8)] {
            last_src = make_scene(step * 5, step * 3);
            upload_window_to_tex(&last_src);
            let pkt = enc.encode_gpu_texture(&gpu_frame, q).unwrap();
            assert!(pkt.used_lossy);
            assert_eq!(enc.ref_bgra.capacity(), 0);
            assert_eq!(enc.source_bgra.capacity(), 0);
            let dec_frame = dec.decode(w, h, &pkt.payload).unwrap().to_vec();
            let vram_ref = enc
                .vram_state
                .as_ref()
                .unwrap()
                .download_ref_bgra()
                .unwrap();
            assert_frames_eq(
                &dec_frame,
                &vram_ref,
                &format!("VRAM vs Decoder step={step} Q={q}"),
            );
        }

        // 5. Progressive idle lossless refresh directly from VRAM -> 100% bit-identical to `last_src`!
        assert!(enc.has_pending_lossless_refresh());
        let mut chunks = 0usize;
        let mut latest_dec = Vec::new();
        while let Some(ref_pkt) = enc.encode_lossless_refresh(48, 256 * 1024) {
            latest_dec = dec.decode(w, h, &ref_pkt.payload).unwrap();
            chunks += 1;
            assert!(chunks <= 64);
        }
        assert!(!enc.has_pending_lossless_refresh());
        assert_eq!(enc.ref_bgra.capacity(), 0);
        assert_eq!(enc.source_bgra.capacity(), 0);
        let vram_final = enc
            .vram_state
            .as_ref()
            .unwrap()
            .download_ref_bgra()
            .unwrap();
        assert_frames_eq(
            &latest_dec,
            &last_src,
            "Final idle lossless refresh (decoder)",
        );
        assert_frames_eq(
            &vram_final,
            &last_src,
            "Final idle lossless refresh (VRAM ref)",
        );
    }

    #[test]
    fn packed_plane_encoder_matches_unpacked() {
        let mut rng = 0x1234_5678u32;
        let mut next = || {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            rng >> 8
        };
        for (bx, by) in [(8usize, 8usize), (4, 4), (3, 5), (8, 3), (1, 1), (7, 8)] {
            for density in [0u32, 2, 10, 60] {
                let count = bx * by;
                let mut words = vec![0i32; count * 32];
                let mut blocks = vec![[0i32; 64]; count];
                for b in 0..count {
                    for c in 0..64 {
                        let v = if c == 0 {
                            (next() % 2001) as i32 - 1000
                        } else if next() % 100 < density {
                            let mag = if next() % 10 == 0 { 3000 } else { 40 };
                            (next() % (2 * mag + 1)) as i32 - mag as i32
                        } else {
                            0
                        };
                        blocks[b][c] = v;
                        let w = &mut words[b * 32 + c / 2];
                        let bits = (v as i16 as u16 as u32) << (16 * (c % 2));
                        *w = (*w as u32 | bits) as i32;
                    }
                }
                let mut expected = Vec::new();
                encode_plane_qblocks(&mut blocks, bx, by, &mut expected);
                let mut got = Vec::new();
                encode_plane_packed(&words, bx, by, &mut got);
                assert_eq!(got, expected, "{bx}x{by} density {density}");
            }
        }
    }
}
