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

/// Decompress a size-prepended LZ4 payload while enforcing `max_uncompressed` BEFORE allocating.
pub fn decompress_lz4_bounded(payload: &[u8], max_uncompressed: usize) -> Result<Vec<u8>> {
    if payload.len() < 4 {
        bail!("LZ4 payload too short for size prefix: {} bytes", payload.len());
    }
    let declared = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if declared > max_uncompressed {
        bail!(
            "LZ4 declared uncompressed size {declared} exceeds limit {max_uncompressed}"
        );
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
    let a0 = C4 * (x[0] + x[4]);
    let a1 = C4 * (x[0] - x[4]);
    let b0 = C2 * x[2] + C6 * x[6];
    let b1 = C6 * x[2] - C2 * x[6];

    let e0 = a0 + b0;
    let e1 = a1 + b1;
    let e2 = a1 - b1;
    let e3 = a0 - b0;

    let o0 = C1 * x[1] + C3 * x[3] + C5 * x[5] + C7 * x[7];
    let o1 = C3 * x[1] - C7 * x[3] - C1 * x[5] - C5 * x[7];
    let o2 = C5 * x[1] - C1 * x[3] + C7 * x[5] + C3 * x[7];
    let o3 = C7 * x[1] - C5 * x[3] + C3 * x[5] - C1 * x[7];

    [
        (e0 + o0 + round) >> shift,
        (e1 + o1 + round) >> shift,
        (e2 + o2 + round) >> shift,
        (e3 + o3 + round) >> shift,
        (e3 - o3 + round) >> shift,
        (e2 - o2 + round) >> shift,
        (e1 - o1 + round) >> shift,
        (e0 - o0 + round) >> shift,
    ]
}

/// Forward 2D 8x8 integer DCT. Input values are in `[-255, 255]`.
pub fn dct_8x8(input: &[i32; 64]) -> [i32; 64] {
    let mut tmp = [0i32; 64];
    for r in 0..8 {
        let off = r * 8;
        let row = dct_1d(
            [
                input[off],
                input[off + 1],
                input[off + 2],
                input[off + 3],
                input[off + 4],
                input[off + 5],
                input[off + 6],
                input[off + 7],
            ],
            11,
        );
        tmp[off..off + 8].copy_from_slice(&row);
    }
    let mut out = [0i32; 64];
    for c in 0..8 {
        let col = dct_1d(
            [
                tmp[c],
                tmp[8 + c],
                tmp[16 + c],
                tmp[24 + c],
                tmp[32 + c],
                tmp[40 + c],
                tmp[48 + c],
                tmp[56 + c],
            ],
            17,
        );
        for r in 0..8 {
            out[r * 8 + c] = col[r];
        }
    }
    out
}

/// Inverse 2D 8x8 integer DCT.
pub fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
    let mut tmp = [0i32; 64];
    for c in 0..8 {
        let col = idct_1d(
            [
                coeffs[c],
                coeffs[8 + c],
                coeffs[16 + c],
                coeffs[24 + c],
                coeffs[32 + c],
                coeffs[40 + c],
                coeffs[48 + c],
                coeffs[56 + c],
            ],
            11,
        );
        for r in 0..8 {
            tmp[r * 8 + c] = col[r];
        }
    }
    let mut out = [0i32; 64];
    for r in 0..8 {
        let off = r * 8;
        let row = idct_1d(
            [
                tmp[off],
                tmp[off + 1],
                tmp[off + 2],
                tmp[off + 3],
                tmp[off + 4],
                tmp[off + 5],
                tmp[off + 6],
                tmp[off + 7],
            ],
            17,
        );
        out[off..off + 8].copy_from_slice(&row);
    }
    out
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
    let stride = fb_w as usize * 4;
    let mut cb_full = [0i32; 64 * 64];
    let mut cr_full = [0i32; 64 * 64];

    for py in 0..64usize {
        let sy = (y0 as usize) + py.min(th as usize - 1);
        let row_base = sy * stride;
        for px in 0..64usize {
            let sx = (x0 as usize) + px.min(tw as usize - 1);
            let idx = row_base + sx * 4;
            let b = bgra[idx] as i32;
            let g = bgra[idx + 1] as i32;
            let r = bgra[idx + 2] as i32;
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let out_idx = py * 64 + px;
            y_plane[out_idx] = y;
            cb_full[out_idx] = cb;
            cr_full[out_idx] = cr;
        }
    }

    for cy in 0..32usize {
        let y2 = cy * 2;
        for cx in 0..32usize {
            let x2 = cx * 2;
            let i00 = y2 * 64 + x2;
            let i01 = i00 + 1;
            let i10 = i00 + 64;
            let i11 = i10 + 1;
            cb_plane[cy * 32 + cx] = (cb_full[i00] + cb_full[i01] + cb_full[i10] + cb_full[i11] + 2) >> 2;
            cr_plane[cy * 32 + cx] = (cr_full[i00] + cr_full[i01] + cr_full[i10] + cr_full[i11] + 2) >> 2;
        }
    }
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
    let stride = fb_w as usize * 4;
    for py in 0..(th as usize) {
        let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
        let cy = py >> 1;
        for px in 0..(tw as usize) {
            let y_idx = py * 64 + px;
            let y = y_plane[y_idx];
            let c_idx = cy * 32 + (px >> 1);
            let (mut r, mut g, mut b) = ycbcr_to_rgb(y, cb_plane[c_idx], cr_plane[c_idx]);
            let a = match a_plane {
                Some(ap) => {
                    let av = ap[y_idx].clamp(0, 255) as u8;
                    r = r.min(av);
                    g = g.min(av);
                    b = b.min(av);
                    av
                }
                None => 255,
            };
            let dst = row_base + px * 4;
            bgra[dst] = b;
            bgra[dst + 1] = g;
            bgra[dst + 2] = r;
            bgra[dst + 3] = a;
        }
    }
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

/// Extract a 2x2-downsampled tile `(x0, y0, tw, th)` into padded 32x32 Y and 16x16 Cb/Cr planes.
/// Used for progressive fast motion/scroll encoding at HiDPI (4x fewer pixels & DCT blocks).
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
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    let mut cb_half = [0i32; 32 * 32];
    let mut cr_half = [0i32; 32 * 32];

    for hy in 0..32usize {
        let sy0 = (y0 as usize) + (hy * 2).min(max_y);
        let sy1 = (y0 as usize) + (hy * 2 + 1).min(max_y);
        let row0 = sy0 * stride;
        let row1 = sy1 * stride;
        for hx in 0..32usize {
            let sx0 = (x0 as usize) + (hx * 2).min(max_x);
            let sx1 = (x0 as usize) + (hx * 2 + 1).min(max_x);
            let i00 = row0 + sx0 * 4;
            let i01 = row0 + sx1 * 4;
            let i10 = row1 + sx0 * 4;
            let i11 = row1 + sx1 * 4;
            let b = (bgra[i00] as i32 + bgra[i01] as i32 + bgra[i10] as i32 + bgra[i11] as i32 + 2) >> 2;
            let g = (bgra[i00 + 1] as i32 + bgra[i01 + 1] as i32 + bgra[i10 + 1] as i32 + bgra[i11 + 1] as i32 + 2) >> 2;
            let r = (bgra[i00 + 2] as i32 + bgra[i01 + 2] as i32 + bgra[i10 + 2] as i32 + bgra[i11 + 2] as i32 + 2) >> 2;
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let out_idx = hy * 32 + hx;
            y_plane[out_idx] = y;
            cb_half[out_idx] = cb;
            cr_half[out_idx] = cr;
        }
    }

    for cy in 0..16usize {
        let y2 = cy * 2;
        for cx in 0..16usize {
            let x2 = cx * 2;
            let i00 = y2 * 32 + x2;
            let i01 = i00 + 1;
            let i10 = i00 + 32;
            let i11 = i10 + 1;
            cb_plane[cy * 16 + cx] = (cb_half[i00] + cb_half[i01] + cb_half[i10] + cb_half[i11] + 2) >> 2;
            cr_plane[cy * 16 + cx] = (cr_half[i00] + cr_half[i01] + cr_half[i10] + cr_half[i11] + 2) >> 2;
        }
    }
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
    let stride = fb_w as usize * 4;
    for py in 0..(th as usize) {
        let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
        let hy0 = py >> 1;
        let hy1 = if (py & 1) == 0 { hy0 } else { (hy0 + 1).min(31) };
        let r0 = hy0 * 32;
        let r1 = hy1 * 32;
        let cy = hy0 >> 1;
        for px in 0..(tw as usize) {
            let hx0 = px >> 1;
            let hx1 = if (px & 1) == 0 { hx0 } else { (hx0 + 1).min(31) };
            let y = (y_plane[r0 + hx0] + y_plane[r0 + hx1] + y_plane[r1 + hx0] + y_plane[r1 + hx1] + 2) >> 2;
            let c_idx = cy * 16 + (hx0 >> 1);
            let (mut r, mut g, mut b) = ycbcr_to_rgb(y, cb_plane[c_idx], cr_plane[c_idx]);
            let a = match a_plane {
                Some(ap) => {
                    let av_i = (ap[r0 + hx0] + ap[r0 + hx1] + ap[r1 + hx0] + ap[r1 + hx1] + 2) >> 2;
                    let av = av_i.clamp(0, 255) as u8;
                    r = r.min(av);
                    g = g.min(av);
                    b = b.min(av);
                    av
                }
                None => 255,
            };
            let dst = row_base + px * 4;
            bgra[dst] = b;
            bgra[dst + 1] = g;
            bgra[dst + 2] = r;
            bgra[dst + 3] = a;
        }
    }
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
    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    let mut cb_quarter = [0i32; 16 * 16];
    let mut cr_quarter = [0i32; 16 * 16];

    for qy in 0..16usize {
        let base_y = qy * 4;
        for qx in 0..16usize {
            let base_x = qx * 4;
            let mut sum_b = 0i32;
            let mut sum_g = 0i32;
            let mut sum_r = 0i32;
            for dy in 0..4usize {
                let sy = (y0 as usize) + (base_y + dy).min(max_y);
                let row = sy * stride;
                for dx in 0..4usize {
                    let sx = (x0 as usize) + (base_x + dx).min(max_x);
                    let idx = row + sx * 4;
                    sum_b += bgra[idx] as i32;
                    sum_g += bgra[idx + 1] as i32;
                    sum_r += bgra[idx + 2] as i32;
                }
            }
            let b = (sum_b + 8) >> 4;
            let g = (sum_g + 8) >> 4;
            let r = (sum_r + 8) >> 4;
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let out_idx = qy * 16 + qx;
            y_plane[out_idx] = y;
            cb_quarter[out_idx] = cb;
            cr_quarter[out_idx] = cr;
        }
    }

    for cy in 0..8usize {
        let y2 = cy * 2;
        for cx in 0..8usize {
            let x2 = cx * 2;
            let i00 = y2 * 16 + x2;
            let i01 = i00 + 1;
            let i10 = i00 + 16;
            let i11 = i10 + 1;
            cb_plane[cy * 8 + cx] =
                (cb_quarter[i00] + cb_quarter[i01] + cb_quarter[i10] + cb_quarter[i11] + 2) >> 2;
            cr_plane[cy * 8 + cx] =
                (cr_quarter[i00] + cr_quarter[i01] + cr_quarter[i10] + cr_quarter[i11] + 2) >> 2;
        }
    }
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
    let stride = fb_w as usize * 4;
    for py in 0..(th as usize) {
        let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
        let qy0 = py >> 2;
        let qy1 = (qy0 + 1).min(15);
        let wy = (py & 3) as i32;
        let wy0 = 4 - wy;
        let r0 = qy0 * 16;
        let r1 = qy1 * 16;
        let cy = qy0 >> 1;
        for px in 0..(tw as usize) {
            let qx0 = px >> 2;
            let qx1 = (qx0 + 1).min(15);
            let wx = (px & 3) as i32;
            let wx0 = 4 - wx;
            let top = y_plane[r0 + qx0] * wx0 + y_plane[r0 + qx1] * wx;
            let bot = y_plane[r1 + qx0] * wx0 + y_plane[r1 + qx1] * wx;
            let y = (top * wy0 + bot * wy + 8) >> 4;
            let c_idx = cy * 8 + (qx0 >> 1);
            let (mut r, mut g, mut b) = ycbcr_to_rgb(y, cb_plane[c_idx], cr_plane[c_idx]);
            let a = match a_plane {
                Some(ap) => {
                    let a_top = ap[r0 + qx0] * wx0 + ap[r0 + qx1] * wx;
                    let a_bot = ap[r1 + qx0] * wx0 + ap[r1 + qx1] * wx;
                    let av = ((a_top * wy0 + a_bot * wy + 8) >> 4).clamp(0, 255) as u8;
                    r = r.min(av);
                    g = g.min(av);
                    b = b.min(av);
                    av
                }
                None => 255,
            };
            let dst = row_base + px * 4;
            bgra[dst] = b;
            bgra[dst + 1] = g;
            bgra[dst + 2] = r;
            bgra[dst + 3] = a;
        }
    }
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

/// Deserialize one quantized 8x8 block.
fn decode_rle_block(data: &[u8], pos: &mut usize, prev_dc: &mut i32) -> Result<[i32; 64]> {
    let mut q_block = [0i32; 64];
    let dc_delta = read_coeff(data, pos)?;
    let dc = prev_dc.saturating_add(dc_delta);
    *prev_dc = dc;
    q_block[0] = dc;

    let mut zi = 1usize;
    while zi < 64 {
        if *pos >= data.len() {
            bail!("unexpected EOF reading DCT RLE token");
        }
        let tok = data[*pos];
        *pos += 1;
        if tok == 0 {
            break; // EOB
        }
        let run = (tok - 1) as usize;
        zi += run;
        if zi >= 64 {
            bail!("DCT RLE run overflow");
        }
        q_block[ZIGZAG[zi]] = read_coeff(data, pos)?;
        zi += 1;
    }
    Ok(q_block)
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
    let mut prev_dc = 0i32;
    let is_inter = ref_plane.is_some();
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0i32; 64];
            for r in 0..8 {
                let row_off = (by * 8 + r) * plane_dim + bx * 8;
                for c in 0..8 {
                    let idx = row_off + c;
                    block[r * 8 + c] = if let Some(ref_p) = ref_plane {
                        cur_plane[idx] - ref_p[idx]
                    } else {
                        cur_plane[idx] - 128
                    };
                }
            }
            let dct = dct_8x8(&block);
            let mut q_block = [0i32; 64];
            let mut dequant = [0i32; 64];
            for i in 0..64 {
                let qv = quantize_div(dct[i], quant[i]);
                q_block[i] = qv;
                dequant[i] = qv * quant[i];
            }
            encode_rle_block(&q_block, &mut prev_dc, out);

            let idct = idct_8x8(&dequant);
            for r in 0..8 {
                let row_off = (by * 8 + r) * plane_dim + bx * 8;
                for c in 0..8 {
                    let idx = row_off + c;
                    let val = if is_inter {
                        ref_plane.unwrap()[idx] + idct[r * 8 + c]
                    } else {
                        idct[r * 8 + c] + 128
                    };
                    recon_plane[idx] = val.clamp(0, 255);
                }
            }
        }
    }
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
    let is_inter = ref_plane.is_some();
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let q_block = decode_rle_block(data, pos, &mut prev_dc)?;
            let mut dequant = [0i32; 64];
            for i in 0..64 {
                // Clamp dequantized coefficients so crafted streams cannot overflow `idct_1d` i32 math.
                dequant[i] = q_block[i].saturating_mul(quant[i]).clamp(-32_768, 32_768);
            }
            let idct = idct_8x8(&dequant);
            for r in 0..8 {
                let row_off = (by * 8 + r) * plane_dim + bx * 8;
                for c in 0..8 {
                    let idx = row_off + c;
                    let val = if is_inter {
                        ref_plane.unwrap()[idx] + idct[r * 8 + c]
                    } else {
                        idct[r * 8 + c] + 128
                    };
                    recon_plane[idx] = val.clamp(0, 255);
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sub-region vertical & horizontal scroll detection
// ---------------------------------------------------------------------------

fn hash_row_strip(bgra: &[u8], width: usize, y: usize, x_start: usize, x_end: usize) -> (u64, bool) {
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

fn hash_col_strip(bgra: &[u8], width: usize, x: usize, y_start: usize, y_end: usize) -> (u64, bool) {
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
pub fn detect_scroll_copy_rect(prev: &[u8], cur: &[u8], width: u32, height: u32) -> Option<CopyRectOp> {
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
    let bands = [
        (w / 4, (w * 3) / 4),
        (w / 6, w / 2),
        (w / 2, (w * 5) / 6),
    ];
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

#[derive(Clone, Copy)]
struct TileTrack {
    motion_streak: u8,
    last_change: Option<Instant>,
    is_lossy: bool,
}

impl Default for TileTrack {
    fn default() -> Self {
        Self {
            motion_streak: 0,
            last_change: None,
            is_lossy: false,
        }
    }
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
    /// Bit-exact reconstructed reference framebuffer matching the client decoder.
    ref_bgra: Vec<u8>,
    /// Reusable snapshot of `ref_bgra` before the current frame, used only when a frame MV is active.
    prev_ref_bgra: Vec<u8>,
    /// Most recent lossless source framebuffer (used for automatic idle lossless refinement).
    source_bgra: Vec<u8>,
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

    /// Encode a new `BGRA` frame at the given DCT quality (`1..=100`).
    pub fn encode(&mut self, width: u32, height: u32, bgra: &[u8], quality: u8) -> SmartTilePacket {
        let expected_len = (width as usize) * (height as usize) * 4;
        assert_eq!(bgra.len(), expected_len, "BGRA buffer size mismatch");

        let cols = width.div_ceil(TILE_SIZE) as usize;
        let rows = height.div_ceil(TILE_SIZE) as usize;
        let total_tiles = cols * rows;
        let now = Instant::now();

        let is_keyframe = width != self.width
            || height != self.height
            || self.ref_bgra.len() != expected_len
            || self.tiles.len() != total_tiles;

        if is_keyframe {
            self.width = width;
            self.height = height;
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
        let mut dirty_tiles = Vec::new();
        for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (height - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let x0 = (tx as u32) * TILE_SIZE;
                let tw = (width - x0).min(TILE_SIZE);
                let tidx = ty * cols + tx;

                let dirty = is_keyframe
                    || !tile_equals(&self.ref_bgra, bgra, width, x0, y0, tw, th);
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
        let prev_ref_slice: Option<&[u8]> = if frame_mv.is_some() || enable_local_mv {
            self.prev_ref_bgra.resize(expected_len, 0);
            self.prev_ref_bgra.copy_from_slice(&self.ref_bgra);
            Some(self.prev_ref_bgra.as_slice())
        } else {
            None
        };
        let is_hidpi = width >= 1600 || height >= 1000;
        let full_screen_video = !is_keyframe && frame_mv.is_none() && dirty_tiles.len() * 2 >= total_tiles;

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

        let (used_lossy, used_local_mv) = if num_workers > 1 {
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

            let mut band_outputs: Vec<(Vec<u8>, bool, bool)> = Vec::with_capacity(num_workers);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(num_workers);
                for (ref_chunk, cur_chunk, tile_chunk, y_offset, tidx_offset, items) in band_slices
                {
                    handles.push(s.spawn(move || {
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
        };

        if used_local_mv {
            self.scratch_packet[1] |= FLAG_PREV_REF;
        }

        // Only keep a lossless source copy when at least one tile is lossy and awaits idle refresh.
        // Saves a full 33 MB memcpy on every lossless UI frame at 4K!
        if used_lossy || self.has_pending_lossless_refresh() {
            self.source_bgra.resize(expected_len, 0);
            self.source_bgra.copy_from_slice(bgra);
        }

        let payload = lz4_flex::compress_prepend_size(&self.scratch_packet);
        SmartTilePacket {
            is_keyframe,
            used_lossy,
            had_copy_rect: frame_mv.is_some(),
            dirty_tiles: dirty_tile_count,
            payload,
        }
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

        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        self.scratch_packet.push(0); // not keyframe
        self.scratch_packet.push(80);
        self.scratch_packet.push(0); // 0 copy rects
        self.scratch_packet.extend_from_slice(&0u32.to_le_bytes()); // patched once tile count is known

        let mut refresh_count = 0usize;
        let mut next_check_raw_len = byte_budget;
        let mut ready_payload: Option<Vec<u8>> = None;
        let mut residual_samples = Vec::with_capacity((TILE_SIZE as usize) * (TILE_SIZE as usize) * 4);

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
                self.scratch_packet.extend_from_slice(&(tx as u16).to_le_bytes());
                self.scratch_packet.extend_from_slice(&(ty as u16).to_le_bytes());
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
                        ready_payload = Some(compressed);
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
                lz4_flex::compress_prepend_size(&self.scratch_packet)
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
        // 2. Lightweight per-tile 2D diamond search `(local_dx, local_dy)` for panning video / moving objects.
        let (pred_buf, pred_x, pred_y, mv_flag, local_mv): (
            &[u8],
            u32,
            u32,
            u8,
            Option<(i8, i8)>,
        ) = if wants_lossy {
            if let (Some(prev_full), Some((sx, sy))) = (prev_ref_full, mv_src) {
                let mv_sad =
                    tile_bgra_sad(prev_full, cur_bgra, fb_w, sx, sy, x0, local_y0, tw, th);
                let zero_sad =
                    tile_bgra_sad(ref_bgra, cur_bgra, fb_w, x0, local_y0, x0, local_y0, tw, th);
                if mv_sad < zero_sad {
                    (prev_full, sx, sy, MODE_FLAG_MV, None)
                } else {
                    (ref_bgra, x0, local_y0, 0, None)
                }
            } else if let Some(prev_full) = prev_ref_full {
                if let Some((ldx, ldy, sx, sy)) = search_local_mv_2d(
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
                    last_local_mv = (ldx, ldy);
                    (prev_full, sx, sy, MODE_FLAG_LOCAL_MV, Some((ldx, ldy)))
                } else {
                    (ref_bgra, x0, local_y0, 0, None)
                }
            } else {
                (ref_bgra, x0, local_y0, 0, None)
            }
        } else {
            (ref_bgra, x0, local_y0, 0, None)
        };

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
            bgra_tile_to_ycbcr420(
                pred_buf,
                fb_w,
                pred_x,
                pred_y,
                tw,
                th,
                &mut ref_y,
                &mut ref_cb,
                &mut ref_cr,
            );
            if has_alpha {
                bgra_tile_to_alpha(cur_bgra, fb_w, x0, local_y0, tw, th, &mut cur_a);
                bgra_tile_to_alpha(pred_buf, fb_w, pred_x, pred_y, tw, th, &mut ref_a);
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
    for i in 0..256 {
        intra_sad += (cur_y[i] - mean).abs() as i64;
    }
    inter_sad * 10 <= intra_sad * 9
}

fn tile_equals(a: &[u8], b: &[u8], fb_w: u32, x0: u32, y0: u32, tw: u32, th: u32) -> bool {
    let stride = fb_w as usize * 4;
    let row_bytes = tw as usize * 4;
    for py in 0..(th as usize) {
        let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
        if a[off..off + row_bytes] != b[off..off + row_bytes] {
            return false;
        }
    }
    true
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
    let stride = fb_w as usize * 4;
    let row_bytes = tw as usize * 4;
    for py in 0..(th as usize) {
        let s_off = (sy as usize + py) * stride + (sx as usize) * 4;
        let d_off = (dy as usize + py) * stride + (dx as usize) * 4;
        if src_fb[s_off..s_off + row_bytes] != dst_fb[d_off..d_off + row_bytes] {
            return false;
        }
    }
    true
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

/// Lightweight 4-step 8-neighbor Three-Step Search (`±24 px` range) seeded from
/// `(0, 0)` and the left-neighbor tile's MV `seed_mv`. Returns `Some((dx, dy, sx, sy))` only
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
    seed_mv: (i8, i8),
) -> Option<(i8, i8, u32, u32)> {
    const MAX_RANGE: i32 = 24;
    let zero_sad = tile_bgra_sad_fast(prev_full, cur_band, fb_w, x0, y0, x0, local_y0, tw, th);
    // Skip search on near-stationary tiles (mean RGB diff < ~2/channel across 64 samples).
    if zero_sad < 384 {
        return None;
    }

    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = zero_sad;

    let eval = |dx: i32, dy: i32| -> Option<(u64, u32, u32)> {
        if dx.abs() > MAX_RANGE || dy.abs() > MAX_RANGE {
            return None;
        }
        let sx = x0 as i32 - dx;
        let sy = y0 as i32 - dy;
        if sx < 0 || sy < 0 || (sx as u32) + tw > fb_w || (sy as u32) + th > fb_h {
            return None;
        }
        let sad = tile_bgra_sad_fast(
            prev_full,
            cur_band,
            fb_w,
            sx as u32,
            sy as u32,
            x0,
            local_y0,
            tw,
            th,
        );
        Some((sad, sx as u32, sy as u32))
    };

    // Test spatial neighbor MV predictor first (immediately nails uniform camera pans).
    if seed_mv != (0, 0) {
        if let Some((sad, _, _)) = eval(seed_mv.0 as i32, seed_mv.1 as i32) {
            if sad < best_sad {
                best_sad = sad;
                best_dx = seed_mv.0 as i32;
                best_dy = seed_mv.1 as i32;
            }
        }
    }

    // 4-step logarithmic 8-neighbor search (Three-Step Search: cardinal + diagonal): 8 -> 4 -> 2 -> 1 px.
    for step in [8i32, 4, 2, 1] {
        if best_sad * 5 <= zero_sad && step >= 4 {
            // Spatial predictor already found a great match; skip coarse steps and refine finely.
            continue;
        }
        let cx = best_dx;
        let cy = best_dy;
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
            if let Some((sad, _, _)) = eval(ndx, ndy) {
                if sad < best_sad {
                    best_sad = sad;
                    best_dx = ndx;
                    best_dy = ndy;
                }
            }
        }
    }

    if (best_dx != 0 || best_dy != 0) && best_sad * 10 <= zero_sad * 8 {
        let sx = (x0 as i32 - best_dx) as u32;
        let sy = (y0 as i32 - best_dy) as u32;
        Some((best_dx as i8, best_dy as i8, sx, sy))
    } else {
        None
    }
}

fn tile_is_opaque(bgra: &[u8], fb_w: u32, x0: u32, y0: u32, tw: u32, th: u32) -> bool {
    let stride = fb_w as usize * 4;
    for py in 0..(th as usize) {
        let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
        for px in 0..(tw as usize) {
            if bgra[row_base + px * 4 + 3] != 255 {
                return false;
            }
        }
    }
    true
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
        let mut same = 0usize;
        let total = (tw as usize) * (th as usize);
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            for px in 0..(tw as usize) {
                let p = off + px * 4;
                if ref_bgra[p..p + 4] == cur_bgra[p..p + 4] {
                    same += 1;
                }
            }
        }
        same * 5 >= total * 2
    };

    if use_xor {
        out.push(MODE_LOSSLESS_XOR);
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            for b in 0..row_bytes {
                let c = cur_bgra[off + b];
                out.push(c ^ ref_bgra[off + b]);
                ref_bgra[off + b] = c;
            }
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
            let read_local_mv_src = |pos: &mut usize| -> Result<(u32, u32)> {
                if !has_prev_snapshot {
                    bail!("tile uses LOCAL_MV without FLAG_PREV_REF");
                }
                if *pos + 2 > raw.len() {
                    bail!("unexpected EOF reading SmartTile local MV");
                }
                let dx = raw[*pos] as i8 as i32;
                let dy = raw[*pos + 1] as i8 as i32;
                *pos += 2;
                let sx = x0 as i32 - dx;
                let sy = y0 as i32 - dy;
                if sx < 0 || sy < 0 || (sx as u32) + tw > width || (sy as u32) + th > height {
                    bail!("SmartTile local MV ({dx}, {dy}) out of bounds at ({x0}, {y0})");
                }
                Ok((sx as u32, sy as u32))
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
                        for b in 0..row_bytes {
                            bgra_mut[dst + b] ^= raw[pos + b];
                        }
                        pos += row_bytes;
                    }
                }
                MODE_MV_SKIP if !has_alpha && !use_mv && !use_local_mv => {
                    let (sx, sy) = mv_src(x0, y0, tw, th)?;
                    copy_tile_from_prev(&self.prev_bgra, bgra_mut, width, sx, sy, x0, y0, tw, th);
                }
                MODE_LOSSLESS_RESIDUAL if !use_mv && !use_local_mv => {
                    decode_lossless_residual_tile(
                        &raw,
                        &mut pos,
                        has_alpha,
                        bgra_mut,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                    )?;
                }
                MODE_DCT_INTRA if !use_mv && !use_local_mv => {
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_y, 64, luma_bx, luma_by, &luma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_cb, 32, chroma_bx, chroma_by, &chroma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_cr, 32, chroma_bx, chroma_by, &chroma_q)?;
                    if has_alpha {
                        decode_plane(&raw, &mut pos, None, &mut sc.recon_a, 64, luma_bx, luma_by, &luma_q)?;
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
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    let (pred_buf, px0, py0): (&[u8], u32, u32) = if use_mv {
                        let (sx, sy) = mv_src(x0, y0, tw, th)?;
                        (&self.prev_bgra, sx, sy)
                    } else if use_local_mv {
                        let (sx, sy) = read_local_mv_src(&mut pos)?;
                        (&self.prev_bgra, sx, sy)
                    } else {
                        (&*bgra_mut, x0, y0)
                    };
                    bgra_tile_to_ycbcr420(
                        pred_buf,
                        width,
                        px0,
                        py0,
                        tw,
                        th,
                        &mut sc.ref_y,
                        &mut sc.ref_cb,
                        &mut sc.ref_cr,
                    );
                    if has_alpha {
                        bgra_tile_to_alpha(pred_buf, width, px0, py0, tw, th, &mut sc.ref_a);
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
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let hw = (tw as usize).div_ceil(2);
                    let hh = (th as usize).div_ceil(2);
                    let luma_bx = hw.div_ceil(8);
                    let luma_by = hh.div_ceil(8);
                    let chroma_bx = hw.div_ceil(2).div_ceil(8);
                    let chroma_by = hh.div_ceil(2).div_ceil(8);
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_hy, 32, luma_bx, luma_by, &luma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_hcb, 16, chroma_bx, chroma_by, &chroma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_hcr, 16, chroma_bx, chroma_by, &chroma_q)?;
                    if has_alpha {
                        decode_plane(&raw, &mut pos, None, &mut sc.recon_ha, 32, luma_bx, luma_by, &luma_q)?;
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
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
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
                        let (sx, sy) = read_local_mv_src(&mut pos)?;
                        (&self.prev_bgra, sx, sy)
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
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
                    let qw = (tw as usize).div_ceil(4);
                    let qh = (th as usize).div_ceil(4);
                    let luma_bx = qw.div_ceil(8);
                    let luma_by = qh.div_ceil(8);
                    let chroma_bx = qw.div_ceil(2).div_ceil(8);
                    let chroma_by = qh.div_ceil(2).div_ceil(8);
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_qy, 16, luma_bx, luma_by, &luma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_qcb, 8, chroma_bx, chroma_by, &chroma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut sc.recon_qcr, 8, chroma_bx, chroma_by, &chroma_q)?;
                    if has_alpha {
                        decode_plane(&raw, &mut pos, None, &mut sc.recon_qa, 16, luma_bx, luma_by, &luma_q)?;
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
                    let sc = dct_scratch.get_or_insert_with(|| Box::new(DctDecodeScratch::default()));
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
                        let (sx, sy) = read_local_mv_src(&mut pos)?;
                        (&self.prev_bgra, sx, sy)
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
                        bgra_tile_to_alpha_quarter(pred_buf, width, px0, py0, tw, th, &mut sc.ref_qa);
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

        let op = detect_scroll_copy_rect(&frame1, &frame2, w, h).expect("should detect vertical scroll");
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
        assert_eq!(d1, enc.ref_bgra, "encoder and decoder reference must match bit-for-bit");

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
        assert!(refresh_chunks >= 1, "should emit at least one refresh chunk");
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
                for px in out.chunks_exact(4) {
                    assert!(
                        px[0] <= px[3] && px[1] <= px[3] && px[2] <= px[3],
                        "invalid premultiplied pixel {:?} for {w}x{h} q={q}",
                        px
                    );
                }
            }
            assert!(saw_lossy, "expected lossy alpha DCT for {w}x{h} q={q}");

            // Idle lossless refinement restores 100% bit-exact RGBA pixels.
            while let Some(chunk) = enc
                .encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
            {
                let _ = dec.decode(w, h, &chunk.payload).unwrap();
            }
            assert!(!enc.has_pending_lossless_refresh());
            assert_eq!(*dec.bgra, last_frame);
        }
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
        assert_eq!(d1, enc.ref_bgra, "half-res DCT reference must match bit-for-bit");

        // Second motion frame exercises half-res P-tiles.
        let f2 = make_frame(24);
        let p2 = enc.encode(w, h, &f2, 75);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, enc.ref_bgra, "half-res P-tile reference must stay bit-exact");

        // Progressive chunked refinement should take multiple small chunks and end 100% lossless.
        let mut chunks = 0usize;
        while let Some(chunk) =
            enc.encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
        {
            assert!(chunk.dirty_tiles <= DEFAULT_REFRESH_CHUNK_TILES);
            let _ = dec.decode(w, h, &chunk.payload).unwrap();
            chunks += 1;
        }
        assert!(chunks >= 2, "expected multiple progressive chunks, got {chunks}");
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
        assert_eq!(d1, enc.ref_bgra, "quarter-res I-tile reference must match bit-for-bit");

        // Second congested frame exercises quarter-res P-tiles.
        let f2 = make_frame(2);
        let p2 = enc.encode(w, h, &f2, 26);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, enc.ref_bgra, "quarter-res P-tile reference must stay bit-exact");

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
            for px in buf.chunks_exact_mut(4) {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
                    buf[idx + 2] =
                        (((x as i32 + y as i32) / 2 + phase as i32 * 2 + jitter).clamp(0, 255))
                            as u8;
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
        assert_eq!(*dec.bgra, f2, "lossless residual refinement must be 100% bit-exact");
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
                        buf[idx + 1] =
                            (((v + ((sy.wrapping_mul(3) + hi) as i32 & 0xFF)) & 0xFF) + jitter)
                                .clamp(0, 255) as u8;
                        buf[idx + 2] =
                            (((v * 3 + hi as i32) & 0xFF) + jitter).clamp(0, 255) as u8;
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
        assert_eq!(d2, enc.ref_bgra, "MV-Inter DCT reference must stay bit-exact");

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
        assert!(!p1.had_copy_rect, "diagonal pan should not trigger 1D scroll MV");
        assert!(p1.used_lossy);
        let raw1 = lz4_flex::decompress_size_prepended(&p1.payload).unwrap();
        assert_ne!(
            raw1[1] & FLAG_PREV_REF,
            0,
            "expected diagonal pan to use per-tile 2D local MVs (FLAG_PREV_REF)"
        );
        let d1 = dec.decode(w, h, &p1.payload).unwrap();
        assert_eq!(d1, enc.ref_bgra, "2D local MV reference must match bit-for-bit");

        let f2 = make_pan_frame(2);
        let p2 = enc.encode(w, h, &f2, 50);
        assert!(p2.used_lossy);
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, enc.ref_bgra, "consecutive 2D local MV frame must stay bit-exact");

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
}
