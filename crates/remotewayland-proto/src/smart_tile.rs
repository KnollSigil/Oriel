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
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// Tile edge length in pixels (must be a multiple of 16 for 4:2:0 8x8 DCT blocks).
pub const TILE_SIZE: u32 = 64;

const PACKET_VERSION: u8 = 1;
const FLAG_KEYFRAME: u8 = 1 << 0;

const MODE_LOSSLESS_RAW: u8 = 0;
const MODE_LOSSLESS_XOR: u8 = 1;
const MODE_DCT_INTRA: u8 = 2;
const MODE_DCT_INTER: u8 = 3;

/// Window after a tile's last change during which a subsequent change increments its motion streak.
const MOTION_WINDOW: Duration = Duration::from_millis(220);

/// Number of consecutive motion updates before an opaque tile switches to lossy DCT.
const MOTION_STREAK_THRESHOLD: u8 = 2;

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

    if w == 0
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

/// Write reconstructed 64x64 Y + 32x32 Cb/Cr planes back into a BGRA framebuffer tile (`A = 255`).
fn ycbcr420_to_bgra_tile(
    y_plane: &[i32; 64 * 64],
    cb_plane: &[i32; 32 * 32],
    cr_plane: &[i32; 32 * 32],
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
            let y = y_plane[py * 64 + px];
            let c_idx = cy * 32 + (px >> 1);
            let (r, g, b) = ycbcr_to_rgb(y, cb_plane[c_idx], cr_plane[c_idx]);
            let dst = row_base + px * 4;
            bgra[dst] = b;
            bgra[dst + 1] = g;
            bgra[dst + 2] = r;
            bgra[dst + 3] = 255;
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
    let dc = *prev_dc + dc_delta;
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
                dequant[i] = q_block[i] * quant[i];
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
    let x_start = w / 4;
    let x_end = (w * 3) / 4;

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
    // in the central strip (allowing uniform background rows inside the scrolled viewport).
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

    let mut col_matches = vec![false; cols];
    for c in 0..cols {
        let x0 = c * 16;
        let mut ok = true;
        for &y in &sample_rows {
            let py = (y as isize - best_dy) as usize;
            let c_off = y * stride + x0 * 4;
            let p_off = py * stride + x0 * 4;
            if cur[c_off..c_off + 64] != prev[p_off..p_off + 64] {
                ok = false;
                break;
            }
        }
        col_matches[c] = ok;
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
    pub payload: Vec<u8>,
}

/// Per-window adaptive `SmartTile` encoder.
#[derive(Default)]
pub struct SmartTileEncoder {
    width: u32,
    height: u32,
    /// Bit-exact reconstructed reference framebuffer matching the client decoder.
    ref_bgra: Vec<u8>,
    /// Most recent lossless source framebuffer (used for automatic idle lossless refinement).
    source_bgra: Vec<u8>,
    tiles: Vec<TileTrack>,
    scratch_packet: Vec<u8>,
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
            self.source_bgra.resize(expected_len, 0);
            self.tiles.clear();
            self.tiles.resize(total_tiles, TileTrack::default());
        }
        self.source_bgra.copy_from_slice(bgra);

        // 1. Sub-region scroll CopyRect detection (skipped on keyframes).
        let copy_rect = if !is_keyframe {
            detect_scroll_copy_rect(&self.ref_bgra, bgra, width, height)
        } else {
            None
        };
        if let Some(op) = copy_rect {
            apply_copy_rect(&mut self.ref_bgra, width, height, op);
        }

        // 2. Identify dirty 64x64 tiles and update per-tile motion streaks.
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

        // Massive non-scroll update (e.g., video starting or fast camera cut) can enter lossy immediately.
        let massive_motion = !is_keyframe
            && copy_rect.is_none()
            && dirty_tiles.len() >= (total_tiles / 3).max(8);

        let quality = quality.clamp(35, 95);
        let luma_q = build_quant_table(&BASE_LUMA_QUANT, quality);
        let chroma_q = build_quant_table(&BASE_CHROMA_QUANT, quality);

        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        self.scratch_packet
            .push(if is_keyframe { FLAG_KEYFRAME } else { 0 });
        self.scratch_packet.push(quality);
        if let Some(op) = copy_rect {
            self.scratch_packet.push(1);
            self.scratch_packet.extend_from_slice(&op.dst_x.to_le_bytes());
            self.scratch_packet.extend_from_slice(&op.dst_y.to_le_bytes());
            self.scratch_packet.extend_from_slice(&op.w.to_le_bytes());
            self.scratch_packet.extend_from_slice(&op.h.to_le_bytes());
            self.scratch_packet.extend_from_slice(&op.src_x.to_le_bytes());
            self.scratch_packet.extend_from_slice(&op.src_y.to_le_bytes());
        } else {
            self.scratch_packet.push(0);
        }
        self.scratch_packet
            .extend_from_slice(&(dirty_tiles.len() as u32).to_le_bytes());

        let mut cur_y = [0i32; 64 * 64];
        let mut cur_cb = [0i32; 32 * 32];
        let mut cur_cr = [0i32; 32 * 32];
        let mut ref_y = [0i32; 64 * 64];
        let mut ref_cb = [0i32; 32 * 32];
        let mut ref_cr = [0i32; 32 * 32];
        let mut recon_y = [0i32; 64 * 64];
        let mut recon_cb = [0i32; 32 * 32];
        let mut recon_cr = [0i32; 32 * 32];
        let mut used_lossy = false;

        for (tx, ty, x0, y0, tw, th, tidx) in dirty_tiles {
            self.scratch_packet.extend_from_slice(&(tx as u16).to_le_bytes());
            self.scratch_packet.extend_from_slice(&(ty as u16).to_le_bytes());

            let wants_lossy = !is_keyframe
                && copy_rect.is_none()
                && (self.tiles[tidx].motion_streak >= MOTION_STREAK_THRESHOLD || massive_motion)
                && tile_is_opaque(bgra, width, x0, y0, tw, th);

            if wants_lossy {
                used_lossy = true;
                bgra_tile_to_ycbcr420(bgra, width, x0, y0, tw, th, &mut cur_y, &mut cur_cb, &mut cur_cr);
                bgra_tile_to_ycbcr420(
                    &self.ref_bgra,
                    width,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut ref_y,
                    &mut ref_cb,
                    &mut ref_cr,
                );

                // Choose Inter (P-tile) vs Intra (I-tile) by comparing temporal residual SAD
                // against spatial AC energy (deviation from tile mean).
                let use_inter = should_use_inter_tile(&cur_y, &ref_y);
                let luma_bx = (tw as usize).div_ceil(8);
                let luma_by = (th as usize).div_ceil(8);
                let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);

                if use_inter {
                    self.scratch_packet.push(MODE_DCT_INTER);
                    encode_and_reconstruct_plane(
                        &cur_y,
                        Some(&ref_y),
                        &mut recon_y,
                        64,
                        luma_bx,
                        luma_by,
                        &luma_q,
                        &mut self.scratch_packet,
                    );
                    encode_and_reconstruct_plane(
                        &cur_cb,
                        Some(&ref_cb),
                        &mut recon_cb,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                        &mut self.scratch_packet,
                    );
                    encode_and_reconstruct_plane(
                        &cur_cr,
                        Some(&ref_cr),
                        &mut recon_cr,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                        &mut self.scratch_packet,
                    );
                } else {
                    self.scratch_packet.push(MODE_DCT_INTRA);
                    encode_and_reconstruct_plane(
                        &cur_y,
                        None,
                        &mut recon_y,
                        64,
                        luma_bx,
                        luma_by,
                        &luma_q,
                        &mut self.scratch_packet,
                    );
                    encode_and_reconstruct_plane(
                        &cur_cb,
                        None,
                        &mut recon_cb,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                        &mut self.scratch_packet,
                    );
                    encode_and_reconstruct_plane(
                        &cur_cr,
                        None,
                        &mut recon_cr,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                        &mut self.scratch_packet,
                    );
                }

                // Update reference buffer with the EXACT reconstructed pixels (zero drift!).
                ycbcr420_to_bgra_tile(
                    &recon_y,
                    &recon_cb,
                    &recon_cr,
                    &mut self.ref_bgra,
                    width,
                    x0,
                    y0,
                    tw,
                    th,
                );
                self.tiles[tidx].is_lossy = true;
            } else {
                write_lossless_tile(
                    &mut self.ref_bgra,
                    bgra,
                    width,
                    x0,
                    y0,
                    tw,
                    th,
                    is_keyframe,
                    &mut self.scratch_packet,
                );
                self.tiles[tidx].is_lossy = false;
            }
        }

        let payload = lz4_flex::compress_prepend_size(&self.scratch_packet);
        SmartTilePacket {
            is_keyframe,
            used_lossy,
            payload,
        }
    }

    /// Encode a lossless refinement packet for any tiles that were previously sent lossy.
    /// Returns `None` if all tiles are already lossless.
    pub fn encode_lossless_refresh(&mut self) -> Option<SmartTilePacket> {
        if self.width == 0 || self.height == 0 || !self.has_pending_lossless_refresh() {
            return None;
        }
        let cols = self.width.div_ceil(TILE_SIZE) as usize;
        let rows = self.height.div_ceil(TILE_SIZE) as usize;
        let mut refresh_tiles = Vec::new();
        for ty in 0..rows {
            let y0 = (ty as u32) * TILE_SIZE;
            let th = (self.height - y0).min(TILE_SIZE);
            for tx in 0..cols {
                let tidx = ty * cols + tx;
                if self.tiles[tidx].is_lossy {
                    let x0 = (tx as u32) * TILE_SIZE;
                    let tw = (self.width - x0).min(TILE_SIZE);
                    refresh_tiles.push((tx, ty, x0, y0, tw, th, tidx));
                }
            }
        }
        if refresh_tiles.is_empty() {
            return None;
        }

        self.scratch_packet.clear();
        self.scratch_packet.push(PACKET_VERSION);
        self.scratch_packet.push(0); // not keyframe
        self.scratch_packet.push(80);
        self.scratch_packet.push(0); // 0 copy rects
        self.scratch_packet
            .extend_from_slice(&(refresh_tiles.len() as u32).to_le_bytes());

        for (tx, ty, x0, y0, tw, th, tidx) in refresh_tiles {
            self.scratch_packet.extend_from_slice(&(tx as u16).to_le_bytes());
            self.scratch_packet.extend_from_slice(&(ty as u16).to_le_bytes());
            write_lossless_tile(
                &mut self.ref_bgra,
                &self.source_bgra,
                self.width,
                x0,
                y0,
                tw,
                th,
                false,
                &mut self.scratch_packet,
            );
            self.tiles[tidx].is_lossy = false;
            self.tiles[tidx].motion_streak = 0;
        }

        let payload = lz4_flex::compress_prepend_size(&self.scratch_packet);
        Some(SmartTilePacket {
            is_keyframe: false,
            used_lossy: false,
            payload,
        })
    }
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

// ---------------------------------------------------------------------------
// SmartTileDecoder
// ---------------------------------------------------------------------------

/// Per-window `SmartTile` decoder.
#[derive(Default)]
pub struct SmartTileDecoder {
    width: u32,
    height: u32,
    bgra: Vec<u8>,
}

impl SmartTileDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a `SmartTile` packet into the persistent BGRA framebuffer and return a clone.
    pub fn decode(&mut self, width: u32, height: u32, payload: &[u8]) -> Result<Vec<u8>> {
        let raw = lz4_flex::decompress_size_prepended(payload)
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
        let mut pos = 4usize;

        let expected_len = (width as usize) * (height as usize) * 4;
        if (flags & FLAG_KEYFRAME) != 0
            || width != self.width
            || height != self.height
            || self.bgra.len() != expected_len
        {
            self.width = width;
            self.height = height;
            self.bgra.resize(expected_len, 0);
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
            apply_copy_rect(&mut self.bgra, width, height, op);
        }

        if pos + 4 > raw.len() {
            bail!("unexpected EOF reading SmartTile tile count");
        }
        let num_tiles = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
        pos += 4;

        let luma_q = build_quant_table(&BASE_LUMA_QUANT, quality);
        let chroma_q = build_quant_table(&BASE_CHROMA_QUANT, quality);
        let stride = width as usize * 4;

        let mut ref_y = [0i32; 64 * 64];
        let mut ref_cb = [0i32; 32 * 32];
        let mut ref_cr = [0i32; 32 * 32];
        let mut recon_y = [0i32; 64 * 64];
        let mut recon_cb = [0i32; 32 * 32];
        let mut recon_cr = [0i32; 32 * 32];

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

            match mode {
                MODE_LOSSLESS_RAW => {
                    let needed = row_bytes * th as usize;
                    if pos + needed > raw.len() {
                        bail!("unexpected EOF reading lossless raw tile");
                    }
                    for py in 0..(th as usize) {
                        let dst = (y0 as usize + py) * stride + (x0 as usize) * 4;
                        self.bgra[dst..dst + row_bytes].copy_from_slice(&raw[pos..pos + row_bytes]);
                        pos += row_bytes;
                    }
                }
                MODE_LOSSLESS_XOR => {
                    let needed = row_bytes * th as usize;
                    if pos + needed > raw.len() {
                        bail!("unexpected EOF reading lossless XOR tile");
                    }
                    for py in 0..(th as usize) {
                        let dst = (y0 as usize + py) * stride + (x0 as usize) * 4;
                        for b in 0..row_bytes {
                            self.bgra[dst + b] ^= raw[pos + b];
                        }
                        pos += row_bytes;
                    }
                }
                MODE_DCT_INTRA => {
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    decode_plane(&raw, &mut pos, None, &mut recon_y, 64, luma_bx, luma_by, &luma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut recon_cb, 32, chroma_bx, chroma_by, &chroma_q)?;
                    decode_plane(&raw, &mut pos, None, &mut recon_cr, 32, chroma_bx, chroma_by, &chroma_q)?;
                    ycbcr420_to_bgra_tile(&recon_y, &recon_cb, &recon_cr, &mut self.bgra, width, x0, y0, tw, th);
                }
                MODE_DCT_INTER => {
                    let luma_bx = (tw as usize).div_ceil(8);
                    let luma_by = (th as usize).div_ceil(8);
                    let chroma_bx = ((tw as usize).div_ceil(2)).div_ceil(8);
                    let chroma_by = ((th as usize).div_ceil(2)).div_ceil(8);
                    bgra_tile_to_ycbcr420(
                        &self.bgra,
                        width,
                        x0,
                        y0,
                        tw,
                        th,
                        &mut ref_y,
                        &mut ref_cb,
                        &mut ref_cr,
                    );
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&ref_y),
                        &mut recon_y,
                        64,
                        luma_bx,
                        luma_by,
                        &luma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&ref_cb),
                        &mut recon_cb,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    decode_plane(
                        &raw,
                        &mut pos,
                        Some(&ref_cr),
                        &mut recon_cr,
                        32,
                        chroma_bx,
                        chroma_by,
                        &chroma_q,
                    )?;
                    ycbcr420_to_bgra_tile(&recon_y, &recon_cb, &recon_cr, &mut self.bgra, width, x0, y0, tw, th);
                }
                other => bail!("unknown SmartTile mode: {other}"),
            }
        }

        Ok(self.bgra.clone())
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
        assert!(!p2.used_lossy);
        // Even on synthetic high-entropy rows, CopyRect + dirty-tile should cut >93% vs 1 MiB raw.
        assert!(
            p2.payload.len() < 64 * 1024,
            "scrolled frame should compress to < 64 KiB, got {} bytes",
            p2.payload.len()
        );
        let d2 = dec.decode(w, h, &p2.payload).unwrap();
        assert_eq!(d2, frame2);
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

        // Idle lossless refinement restores 100% exact pixels.
        let refresh = enc.encode_lossless_refresh().expect("should emit refresh");
        assert!(!enc.has_pending_lossless_refresh());
        let d_ref = dec.decode(w, h, &refresh.payload).unwrap();
        assert_eq!(d_ref, f2);
    }

    #[test]
    fn transparent_tiles_always_stay_lossless() {
        let w = 128u32;
        let h = 128u32;
        let mut enc = SmartTileEncoder::new();
        let mut dec = SmartTileDecoder::new();

        for step in 0..4u8 {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for px in buf.chunks_exact_mut(4) {
                px[0] = step * 40;
                px[1] = 100;
                px[2] = 200;
                px[3] = 128; // semi-transparent
            }
            let pkt = enc.encode(w, h, &buf, 60);
            assert!(!pkt.used_lossy);
            let out = dec.decode(w, h, &pkt.payload).unwrap();
            assert_eq!(out, buf);
        }
    }
}
