//! Architecture-specific SIMD kernels for `SmartTile` (`x86_64` AVX2 and `aarch64` NEON)
//! plus scalar reference implementations.
//!
//! **Bit-Exactness Invariant:**
//! Every SIMD kernel in this module is 100% bit-identical to its scalar reference counterpart.
//! Because `SmartTile` relies on zero encoder/decoder reference drift across `x86_64` and `aarch64`,
//! these kernels use exact 32-bit fixed-point arithmetic matching the scalar definitions.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use super::{dct_1d, idct_1d, rgb_to_ycbcr, ycbcr_to_rgb, C1, C2, C3, C4, C5, C6, C7};

// ---------------------------------------------------------------------------
// Scalar reference implementations (exposed for fallback & bit-exactness tests)
// ---------------------------------------------------------------------------

pub fn dct_8x8_scalar(input: &[i32; 64]) -> [i32; 64] {
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

pub fn idct_8x8_scalar(coeffs: &[i32; 64]) -> [i32; 64] {
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

#[allow(clippy::too_many_arguments)]
pub fn bgra_tile_to_ycbcr420_scalar(
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
            cb_plane[cy * 32 + cx] =
                (cb_full[i00] + cb_full[i01] + cb_full[i10] + cb_full[i11] + 2) >> 2;
            cr_plane[cy * 32 + cx] =
                (cr_full[i00] + cr_full[i01] + cr_full[i10] + cr_full[i11] + 2) >> 2;
        }
    }
}

/// Extract a motion-predicted tile from `bgra` at half-pixel coordinates `(x0_hpel, y0_hpel)`
/// (where `x0_hpel = 2 * sx + hx`, `y0_hpel = 2 * sy + hy`, `hx, hy in {0, 1}`) into padded
/// 64x64 Y and 32x32 Cb/Cr planes using bit-exact integer bilinear interpolation.
#[allow(clippy::too_many_arguments)]
pub fn bgra_tile_to_ycbcr420_hpel_scalar(
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
    let sx = x0_hpel >> 1;
    let sy = y0_hpel >> 1;
    let hx = (x0_hpel & 1) as usize;
    let hy = (y0_hpel & 1) as usize;
    if hx == 0 && hy == 0 {
        bgra_tile_to_ycbcr420_scalar(bgra, fb_w, sx, sy, tw, th, y_plane, cb_plane, cr_plane);
        return;
    }

    let stride = fb_w as usize * 4;
    let max_x = tw as usize - 1;
    let max_y = th as usize - 1;
    let mut cb_full = [0i32; 64 * 64];
    let mut cr_full = [0i32; 64 * 64];

    for py in 0..64usize {
        let cy0 = py.min(max_y);
        let cy1 = (cy0 + hy).min(max_y);
        let row0 = (sy as usize + cy0) * stride;
        let row1 = (sy as usize + cy1) * stride;
        for px in 0..64usize {
            let cx0 = px.min(max_x);
            let cx1 = (cx0 + hx).min(max_x);
            let col0 = (sx as usize + cx0) * 4;
            let col1 = (sx as usize + cx1) * 4;
            let i00 = row0 + col0;
            let i01 = row0 + col1;
            let i10 = row1 + col0;
            let i11 = row1 + col1;
            let (b, g, r) = match (hx, hy) {
                (1, 0) => (
                    (bgra[i00] as i32 + bgra[i01] as i32 + 1) >> 1,
                    (bgra[i00 + 1] as i32 + bgra[i01 + 1] as i32 + 1) >> 1,
                    (bgra[i00 + 2] as i32 + bgra[i01 + 2] as i32 + 1) >> 1,
                ),
                (0, 1) => (
                    (bgra[i00] as i32 + bgra[i10] as i32 + 1) >> 1,
                    (bgra[i00 + 1] as i32 + bgra[i10 + 1] as i32 + 1) >> 1,
                    (bgra[i00 + 2] as i32 + bgra[i10 + 2] as i32 + 1) >> 1,
                ),
                _ => (
                    (bgra[i00] as i32 + bgra[i01] as i32 + bgra[i10] as i32 + bgra[i11] as i32 + 2)
                        >> 2,
                    (bgra[i00 + 1] as i32
                        + bgra[i01 + 1] as i32
                        + bgra[i10 + 1] as i32
                        + bgra[i11 + 1] as i32
                        + 2)
                        >> 2,
                    (bgra[i00 + 2] as i32
                        + bgra[i01 + 2] as i32
                        + bgra[i10 + 2] as i32
                        + bgra[i11 + 2] as i32
                        + 2)
                        >> 2,
                ),
            };
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
            cb_plane[cy * 32 + cx] =
                (cb_full[i00] + cb_full[i01] + cb_full[i10] + cb_full[i11] + 2) >> 2;
            cr_plane[cy * 32 + cx] =
                (cr_full[i00] + cr_full[i01] + cr_full[i10] + cr_full[i11] + 2) >> 2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ycbcr420_to_bgra_tile_scalar(
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

#[allow(clippy::too_many_arguments)]
pub fn bgra_tile_to_ycbcr420_half_scalar(
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
            let b = (bgra[i00] as i32 + bgra[i01] as i32 + bgra[i10] as i32 + bgra[i11] as i32 + 2)
                >> 2;
            let g = (bgra[i00 + 1] as i32
                + bgra[i01 + 1] as i32
                + bgra[i10 + 1] as i32
                + bgra[i11 + 1] as i32
                + 2)
                >> 2;
            let r = (bgra[i00 + 2] as i32
                + bgra[i01 + 2] as i32
                + bgra[i10 + 2] as i32
                + bgra[i11 + 2] as i32
                + 2)
                >> 2;
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
            cb_plane[cy * 16 + cx] =
                (cb_half[i00] + cb_half[i01] + cb_half[i10] + cb_half[i11] + 2) >> 2;
            cr_plane[cy * 16 + cx] =
                (cr_half[i00] + cr_half[i01] + cr_half[i10] + cr_half[i11] + 2) >> 2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ycbcr420_half_to_bgra_tile_scalar(
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
        let hy1 = if (py & 1) == 0 {
            hy0
        } else {
            (hy0 + 1).min(31)
        };
        let r0 = hy0 * 32;
        let r1 = hy1 * 32;
        let cy = hy0 >> 1;
        for px in 0..(tw as usize) {
            let hx0 = px >> 1;
            let hx1 = if (px & 1) == 0 {
                hx0
            } else {
                (hx0 + 1).min(31)
            };
            let y =
                (y_plane[r0 + hx0] + y_plane[r0 + hx1] + y_plane[r1 + hx0] + y_plane[r1 + hx1] + 2)
                    >> 2;
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

#[allow(clippy::too_many_arguments)]
pub fn bgra_tile_to_ycbcr420_quarter_scalar(
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

#[allow(clippy::too_many_arguments)]
pub fn ycbcr420_quarter_to_bgra_tile_scalar(
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

pub fn tile_equals_scalar(
    a: &[u8],
    b: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) -> bool {
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
pub fn tile_equals_shifted_scalar(
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

pub fn tile_is_opaque_scalar(bgra: &[u8], fb_w: u32, x0: u32, y0: u32, tw: u32, th: u32) -> bool {
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

pub fn count_identical_pixels_scalar(
    ref_bgra: &[u8],
    cur_bgra: &[u8],
    fb_w: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
) -> usize {
    let stride = fb_w as usize * 4;
    let mut same = 0usize;
    for py in 0..(th as usize) {
        let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
        for px in 0..(tw as usize) {
            let p = off + px * 4;
            if ref_bgra[p..p + 4] == cur_bgra[p..p + 4] {
                same += 1;
            }
        }
    }
    same
}

pub fn xor_encode_row_scalar(ref_row: &mut [u8], cur_row: &[u8], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + cur_row.len(), 0);
    let dst = &mut out[start..start + cur_row.len()];
    for ((d, r), &c) in dst.iter_mut().zip(ref_row.iter_mut()).zip(cur_row.iter()) {
        *d = c ^ *r;
        *r = c;
    }
}

pub fn xor_decode_row_scalar(dst_row: &mut [u8], src_row: &[u8]) {
    for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
        *d ^= s;
    }
}

pub fn deblock_and_sao_plane_scalar(
    plane: &mut [i32],
    plane_dim: usize,
    blocks_x: usize,
    blocks_y: usize,
    q_dc: i32,
) {
    let beta = (q_dc + 4).clamp(6, 36);
    let tc = ((q_dc + 3) >> 2).clamp(1, 10);
    let active_w = (blocks_x * 8).min(plane_dim);
    let active_h = (blocks_y * 8).min(plane_dim);

    for by in 0..blocks_y {
        for r in 0..8 {
            let y = by * 8 + r;
            if y >= active_h {
                continue;
            }
            let row_off = y * plane_dim;
            for bx in 1..blocks_x {
                let x = bx * 8;
                let p1 = plane[row_off + x - 2];
                let p0 = plane[row_off + x - 1];
                let q0 = plane[row_off + x];
                let q1 = plane[row_off + x + 1];
                if (p0 - q0).abs() < beta
                    && (p1 - p0).abs() < (beta >> 1)
                    && (q1 - q0).abs() < (beta >> 1)
                {
                    let delta = (((q0 - p0) * 3 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    plane[row_off + x - 1] = (p0 + delta).clamp(0, 255);
                    plane[row_off + x] = (q0 - delta).clamp(0, 255);
                    let delta1 = (delta >> 1).clamp(-(tc >> 1), tc >> 1);
                    plane[row_off + x - 2] = (p1 + delta1).clamp(0, 255);
                    plane[row_off + x + 1] = (q1 - delta1).clamp(0, 255);
                }
            }
        }
    }

    for bx in 0..blocks_x {
        for c in 0..8 {
            let x = bx * 8 + c;
            if x >= active_w {
                continue;
            }
            for by in 1..blocks_y {
                let y = by * 8;
                let p1 = plane[(y - 2) * plane_dim + x];
                let p0 = plane[(y - 1) * plane_dim + x];
                let q0 = plane[y * plane_dim + x];
                let q1 = plane[(y + 1) * plane_dim + x];
                if (p0 - q0).abs() < beta
                    && (p1 - p0).abs() < (beta >> 1)
                    && (q1 - q0).abs() < (beta >> 1)
                {
                    let delta = (((q0 - p0) * 3 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    plane[(y - 1) * plane_dim + x] = (p0 + delta).clamp(0, 255);
                    plane[y * plane_dim + x] = (q0 - delta).clamp(0, 255);
                    let delta1 = (delta >> 1).clamp(-(tc >> 1), tc >> 1);
                    plane[(y - 2) * plane_dim + x] = (p1 + delta1).clamp(0, 255);
                    plane[(y + 1) * plane_dim + x] = (q1 - delta1).clamp(0, 255);
                }
            }
        }
    }

    let sao_thresh = (beta >> 1).max(3);
    for by in 0..blocks_y {
        for r in 0..8 {
            let y = by * 8 + r;
            if y >= active_h {
                continue;
            }
            let row_off = y * plane_dim;
            for bx in 0..blocks_x {
                let base = row_off + bx * 8;
                let mut line = [0i32; 8];
                line.copy_from_slice(&plane[base..base + 8]);
                for c in 1..7usize {
                    let prev = line[c - 1];
                    let cur = line[c];
                    let next = line[c + 1];
                    let d_left = cur - prev;
                    let d_right = cur - next;
                    if d_left.abs() < sao_thresh && d_right.abs() < sao_thresh {
                        if d_left > 0 && d_right > 0 {
                            let step = ((d_left.min(d_right) + 1) >> 2).clamp(1, 2);
                            plane[base + c] = (cur - step).clamp(0, 255);
                        } else if d_left < 0 && d_right < 0 {
                            let step = (((-d_left).min(-d_right) + 1) >> 2).clamp(1, 2);
                            plane[base + c] = (cur + step).clamp(0, 255);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 Hardware SIMD Kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose_8x8_epi32(
        r0: &mut __m256i,
        r1: &mut __m256i,
        r2: &mut __m256i,
        r3: &mut __m256i,
        r4: &mut __m256i,
        r5: &mut __m256i,
        r6: &mut __m256i,
        r7: &mut __m256i,
    ) {
        let t0 = _mm256_unpacklo_epi32(*r0, *r1);
        let t1 = _mm256_unpackhi_epi32(*r0, *r1);
        let t2 = _mm256_unpacklo_epi32(*r2, *r3);
        let t3 = _mm256_unpackhi_epi32(*r2, *r3);
        let t4 = _mm256_unpacklo_epi32(*r4, *r5);
        let t5 = _mm256_unpackhi_epi32(*r4, *r5);
        let t6 = _mm256_unpacklo_epi32(*r6, *r7);
        let t7 = _mm256_unpackhi_epi32(*r6, *r7);

        let tt0 = _mm256_unpacklo_epi64(t0, t2);
        let tt1 = _mm256_unpackhi_epi64(t0, t2);
        let tt2 = _mm256_unpacklo_epi64(t1, t3);
        let tt3 = _mm256_unpackhi_epi64(t1, t3);
        let tt4 = _mm256_unpacklo_epi64(t4, t6);
        let tt5 = _mm256_unpackhi_epi64(t4, t6);
        let tt6 = _mm256_unpacklo_epi64(t5, t7);
        let tt7 = _mm256_unpackhi_epi64(t5, t7);

        *r0 = _mm256_permute2x128_si256(tt0, tt4, 0x20);
        *r1 = _mm256_permute2x128_si256(tt1, tt5, 0x20);
        *r2 = _mm256_permute2x128_si256(tt2, tt6, 0x20);
        *r3 = _mm256_permute2x128_si256(tt3, tt7, 0x20);
        *r4 = _mm256_permute2x128_si256(tt0, tt4, 0x31);
        *r5 = _mm256_permute2x128_si256(tt1, tt5, 0x31);
        *r6 = _mm256_permute2x128_si256(tt2, tt6, 0x31);
        *r7 = _mm256_permute2x128_si256(tt3, tt7, 0x31);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dct_1d_8cols(
        r0: &mut __m256i,
        r1: &mut __m256i,
        r2: &mut __m256i,
        r3: &mut __m256i,
        r4: &mut __m256i,
        r5: &mut __m256i,
        r6: &mut __m256i,
        r7: &mut __m256i,
        shift: i32,
    ) {
        let round = _mm256_set1_epi32(1i32 << (shift - 1));
        let c1 = _mm256_set1_epi32(C1);
        let c2 = _mm256_set1_epi32(C2);
        let c3 = _mm256_set1_epi32(C3);
        let c4 = _mm256_set1_epi32(C4);
        let c5 = _mm256_set1_epi32(C5);
        let c6 = _mm256_set1_epi32(C6);
        let c7 = _mm256_set1_epi32(C7);

        let s0 = _mm256_add_epi32(*r0, *r7);
        let s1 = _mm256_add_epi32(*r1, *r6);
        let s2 = _mm256_add_epi32(*r2, *r5);
        let s3 = _mm256_add_epi32(*r3, *r4);
        let d0 = _mm256_sub_epi32(*r0, *r7);
        let d1 = _mm256_sub_epi32(*r1, *r6);
        let d2 = _mm256_sub_epi32(*r2, *r5);
        let d3 = _mm256_sub_epi32(*r3, *r4);

        let s01 = _mm256_add_epi32(s0, s1);
        let s23 = _mm256_add_epi32(s2, s3);
        let y0 = _mm256_add_epi32(_mm256_mullo_epi32(c4, _mm256_add_epi32(s01, s23)), round);

        let y1 = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(c1, d0), _mm256_mullo_epi32(c3, d1)),
                _mm256_add_epi32(_mm256_mullo_epi32(c5, d2), _mm256_mullo_epi32(c7, d3)),
            ),
            round,
        );

        let y2 = _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(c2, s0), _mm256_mullo_epi32(c6, s1)),
                _mm256_add_epi32(_mm256_mullo_epi32(c6, s2), _mm256_mullo_epi32(c2, s3)),
            ),
            round,
        );

        let y3 = _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_sub_epi32(_mm256_mullo_epi32(c3, d0), _mm256_mullo_epi32(c7, d1)),
                _mm256_add_epi32(_mm256_mullo_epi32(c1, d2), _mm256_mullo_epi32(c5, d3)),
            ),
            round,
        );

        let s03_sub_12 = _mm256_sub_epi32(_mm256_add_epi32(s0, s3), _mm256_add_epi32(s1, s2));
        let y4 = _mm256_add_epi32(_mm256_mullo_epi32(c4, s03_sub_12), round);

        let y5 = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_sub_epi32(_mm256_mullo_epi32(c5, d0), _mm256_mullo_epi32(c1, d1)),
                _mm256_add_epi32(_mm256_mullo_epi32(c7, d2), _mm256_mullo_epi32(c3, d3)),
            ),
            round,
        );

        let y6 = _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(c6, s0), _mm256_mullo_epi32(c2, s2)),
                _mm256_add_epi32(_mm256_mullo_epi32(c2, s1), _mm256_mullo_epi32(c6, s3)),
            ),
            round,
        );

        let y7 = _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(c7, d0), _mm256_mullo_epi32(c3, d2)),
                _mm256_add_epi32(_mm256_mullo_epi32(c5, d1), _mm256_mullo_epi32(c1, d3)),
            ),
            round,
        );

        if shift == 11 {
            *r0 = _mm256_srai_epi32(y0, 11);
            *r1 = _mm256_srai_epi32(y1, 11);
            *r2 = _mm256_srai_epi32(y2, 11);
            *r3 = _mm256_srai_epi32(y3, 11);
            *r4 = _mm256_srai_epi32(y4, 11);
            *r5 = _mm256_srai_epi32(y5, 11);
            *r6 = _mm256_srai_epi32(y6, 11);
            *r7 = _mm256_srai_epi32(y7, 11);
        } else {
            *r0 = _mm256_srai_epi32(y0, 17);
            *r1 = _mm256_srai_epi32(y1, 17);
            *r2 = _mm256_srai_epi32(y2, 17);
            *r3 = _mm256_srai_epi32(y3, 17);
            *r4 = _mm256_srai_epi32(y4, 17);
            *r5 = _mm256_srai_epi32(y5, 17);
            *r6 = _mm256_srai_epi32(y6, 17);
            *r7 = _mm256_srai_epi32(y7, 17);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn idct_1d_8cols(
        r0: &mut __m256i,
        r1: &mut __m256i,
        r2: &mut __m256i,
        r3: &mut __m256i,
        r4: &mut __m256i,
        r5: &mut __m256i,
        r6: &mut __m256i,
        r7: &mut __m256i,
        shift: i32,
    ) {
        let round = _mm256_set1_epi32(1i32 << (shift - 1));
        let c1 = _mm256_set1_epi32(C1);
        let c2 = _mm256_set1_epi32(C2);
        let c3 = _mm256_set1_epi32(C3);
        let c4 = _mm256_set1_epi32(C4);
        let c5 = _mm256_set1_epi32(C5);
        let c6 = _mm256_set1_epi32(C6);
        let c7 = _mm256_set1_epi32(C7);

        let a0 = _mm256_mullo_epi32(c4, _mm256_add_epi32(*r0, *r4));
        let a1 = _mm256_mullo_epi32(c4, _mm256_sub_epi32(*r0, *r4));
        let b0 = _mm256_add_epi32(_mm256_mullo_epi32(c2, *r2), _mm256_mullo_epi32(c6, *r6));
        let b1 = _mm256_sub_epi32(_mm256_mullo_epi32(c6, *r2), _mm256_mullo_epi32(c2, *r6));

        let e0 = _mm256_add_epi32(a0, b0);
        let e1 = _mm256_add_epi32(a1, b1);
        let e2 = _mm256_sub_epi32(a1, b1);
        let e3 = _mm256_sub_epi32(a0, b0);

        let o0 = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(c1, *r1), _mm256_mullo_epi32(c3, *r3)),
            _mm256_add_epi32(_mm256_mullo_epi32(c5, *r5), _mm256_mullo_epi32(c7, *r7)),
        );
        let o1 = _mm256_sub_epi32(
            _mm256_sub_epi32(_mm256_mullo_epi32(c3, *r1), _mm256_mullo_epi32(c7, *r3)),
            _mm256_add_epi32(_mm256_mullo_epi32(c1, *r5), _mm256_mullo_epi32(c5, *r7)),
        );
        let o2 = _mm256_add_epi32(
            _mm256_sub_epi32(_mm256_mullo_epi32(c5, *r1), _mm256_mullo_epi32(c1, *r3)),
            _mm256_add_epi32(_mm256_mullo_epi32(c7, *r5), _mm256_mullo_epi32(c3, *r7)),
        );
        let o3 = _mm256_sub_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(c7, *r1), _mm256_mullo_epi32(c3, *r5)),
            _mm256_add_epi32(_mm256_mullo_epi32(c5, *r3), _mm256_mullo_epi32(c1, *r7)),
        );

        let y0 = _mm256_add_epi32(_mm256_add_epi32(e0, o0), round);
        let y1 = _mm256_add_epi32(_mm256_add_epi32(e1, o1), round);
        let y2 = _mm256_add_epi32(_mm256_add_epi32(e2, o2), round);
        let y3 = _mm256_add_epi32(_mm256_add_epi32(e3, o3), round);
        let y4 = _mm256_add_epi32(_mm256_sub_epi32(e3, o3), round);
        let y5 = _mm256_add_epi32(_mm256_sub_epi32(e2, o2), round);
        let y6 = _mm256_add_epi32(_mm256_sub_epi32(e1, o1), round);
        let y7 = _mm256_add_epi32(_mm256_sub_epi32(e0, o0), round);

        if shift == 11 {
            *r0 = _mm256_srai_epi32(y0, 11);
            *r1 = _mm256_srai_epi32(y1, 11);
            *r2 = _mm256_srai_epi32(y2, 11);
            *r3 = _mm256_srai_epi32(y3, 11);
            *r4 = _mm256_srai_epi32(y4, 11);
            *r5 = _mm256_srai_epi32(y5, 11);
            *r6 = _mm256_srai_epi32(y6, 11);
            *r7 = _mm256_srai_epi32(y7, 11);
        } else {
            *r0 = _mm256_srai_epi32(y0, 17);
            *r1 = _mm256_srai_epi32(y1, 17);
            *r2 = _mm256_srai_epi32(y2, 17);
            *r3 = _mm256_srai_epi32(y3, 17);
            *r4 = _mm256_srai_epi32(y4, 17);
            *r5 = _mm256_srai_epi32(y5, 17);
            *r6 = _mm256_srai_epi32(y6, 17);
            *r7 = _mm256_srai_epi32(y7, 17);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dct_8x8(input: &[i32; 64]) -> [i32; 64] {
        let p = input.as_ptr();
        let mut r0 = _mm256_loadu_si256(p as *const __m256i);
        let mut r1 = _mm256_loadu_si256(p.add(8) as *const __m256i);
        let mut r2 = _mm256_loadu_si256(p.add(16) as *const __m256i);
        let mut r3 = _mm256_loadu_si256(p.add(24) as *const __m256i);
        let mut r4 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let mut r5 = _mm256_loadu_si256(p.add(40) as *const __m256i);
        let mut r6 = _mm256_loadu_si256(p.add(48) as *const __m256i);
        let mut r7 = _mm256_loadu_si256(p.add(56) as *const __m256i);

        // Transpose rows -> columns so 8-lane vertical butterfly computes all 8 1D row DCTs at once.
        transpose_8x8_epi32(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7,
        );
        dct_1d_8cols(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 11,
        );

        // Transpose back so columns of the intermediate matrix become vertical lanes, then run column DCTs.
        transpose_8x8_epi32(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7,
        );
        dct_1d_8cols(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 17,
        );

        let mut out = [0i32; 64];
        let op = out.as_mut_ptr();
        _mm256_storeu_si256(op as *mut __m256i, r0);
        _mm256_storeu_si256(op.add(8) as *mut __m256i, r1);
        _mm256_storeu_si256(op.add(16) as *mut __m256i, r2);
        _mm256_storeu_si256(op.add(24) as *mut __m256i, r3);
        _mm256_storeu_si256(op.add(32) as *mut __m256i, r4);
        _mm256_storeu_si256(op.add(40) as *mut __m256i, r5);
        _mm256_storeu_si256(op.add(48) as *mut __m256i, r6);
        _mm256_storeu_si256(op.add(56) as *mut __m256i, r7);
        out
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
        let p = coeffs.as_ptr();
        let mut r0 = _mm256_loadu_si256(p as *const __m256i);
        let mut r1 = _mm256_loadu_si256(p.add(8) as *const __m256i);
        let mut r2 = _mm256_loadu_si256(p.add(16) as *const __m256i);
        let mut r3 = _mm256_loadu_si256(p.add(24) as *const __m256i);
        let mut r4 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let mut r5 = _mm256_loadu_si256(p.add(40) as *const __m256i);
        let mut r6 = _mm256_loadu_si256(p.add(48) as *const __m256i);
        let mut r7 = _mm256_loadu_si256(p.add(56) as *const __m256i);

        // 1. 1D IDCT along columns (shift = 11)
        idct_1d_8cols(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 11,
        );

        // 2. Transpose -> 1D IDCT along rows (shift = 17) -> Transpose back to row-major
        transpose_8x8_epi32(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7,
        );
        idct_1d_8cols(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 17,
        );
        transpose_8x8_epi32(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7,
        );

        let mut out = [0i32; 64];
        let op = out.as_mut_ptr();
        _mm256_storeu_si256(op as *mut __m256i, r0);
        _mm256_storeu_si256(op.add(8) as *mut __m256i, r1);
        _mm256_storeu_si256(op.add(16) as *mut __m256i, r2);
        _mm256_storeu_si256(op.add(24) as *mut __m256i, r3);
        _mm256_storeu_si256(op.add(32) as *mut __m256i, r4);
        _mm256_storeu_si256(op.add(40) as *mut __m256i, r5);
        _mm256_storeu_si256(op.add(48) as *mut __m256i, r6);
        _mm256_storeu_si256(op.add(56) as *mut __m256i, r7);
        out
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rgb_to_ycbcr_8px(b: __m256i, g: __m256i, r: __m256i) -> (__m256i, __m256i, __m256i) {
        let zero = _mm256_setzero_si256();
        let max255 = _mm256_set1_epi32(255);
        let c128 = _mm256_set1_epi32(128);

        // y = ((77 * r + 150 * g + 29 * b + 128) >> 8).clamp(0, 255)
        let y_num = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(77)),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(150)),
            ),
            _mm256_add_epi32(_mm256_mullo_epi32(b, _mm256_set1_epi32(29)), c128),
        );
        let y = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(y_num, 8), zero), max255);

        // cb = (((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128).clamp(0, 255)
        let cb_num = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(-43)),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(-85)),
            ),
            _mm256_add_epi32(_mm256_slli_epi32(b, 7), c128),
        );
        let cb = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(_mm256_srai_epi32(cb_num, 8), c128), zero),
            max255,
        );

        // cr = (((128 * r - 107 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255)
        let cr_num = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_slli_epi32(r, 7),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(-107)),
            ),
            _mm256_add_epi32(_mm256_mullo_epi32(b, _mm256_set1_epi32(-21)), c128),
        );
        let cr = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(_mm256_srai_epi32(cr_num, 8), c128), zero),
            max255,
        );

        (y, cb, cr)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bgra_tile_to_ycbcr420(
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
        if tw < 64 {
            bgra_tile_to_ycbcr420_scalar(bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane);
            return;
        }
        let stride = fb_w as usize * 4;
        let mask_ff = _mm256_set1_epi32(0xFF);
        let two = _mm256_set1_epi32(2);
        let mut cb_full = [0i32; 64 * 64];
        let mut cr_full = [0i32; 64 * 64];

        for py in 0..64usize {
            let sy = (y0 as usize) + py.min(th as usize - 1);
            let row_ptr = bgra.as_ptr().add(sy * stride + (x0 as usize) * 4);
            let out_row = py * 64;
            for chunk in 0..8usize {
                let px = chunk * 8;
                let pix = _mm256_loadu_si256(row_ptr.add(px * 4) as *const __m256i);
                let b = _mm256_and_si256(pix, mask_ff);
                let g = _mm256_and_si256(_mm256_srli_epi32(pix, 8), mask_ff);
                let r = _mm256_and_si256(_mm256_srli_epi32(pix, 16), mask_ff);
                let (y, cb, cr) = rgb_to_ycbcr_8px(b, g, r);
                let dst_off = out_row + px;
                _mm256_storeu_si256(y_plane.as_mut_ptr().add(dst_off) as *mut __m256i, y);
                _mm256_storeu_si256(cb_full.as_mut_ptr().add(dst_off) as *mut __m256i, cb);
                _mm256_storeu_si256(cr_full.as_mut_ptr().add(dst_off) as *mut __m256i, cr);
            }
        }

        // Vectorized 2x2 chroma box-filter downsampling (64x64 -> 32x32).
        for cy in 0..32usize {
            let r0 = (cy * 2) * 64;
            let r1 = r0 + 64;
            let out_base = cy * 32;
            for chunk in 0..4usize {
                let cx = chunk * 8;
                let src_off0 = r0 + cx * 2;
                let src_off1 = r1 + cx * 2;

                // Load 16 horizontal samples as two 8-lane vectors and sum adjacent pairs via hadd + permute.
                let cb_0a = _mm256_loadu_si256(cb_full.as_ptr().add(src_off0) as *const __m256i);
                let cb_0b =
                    _mm256_loadu_si256(cb_full.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cb_1a = _mm256_loadu_si256(cb_full.as_ptr().add(src_off1) as *const __m256i);
                let cb_1b =
                    _mm256_loadu_si256(cb_full.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cb_v0 = _mm256_add_epi32(cb_0a, cb_1a);
                let cb_v1 = _mm256_add_epi32(cb_0b, cb_1b);
                let cb_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cb_v0, cb_v1), 0xD8);
                let cb_avg = _mm256_srai_epi32(_mm256_add_epi32(cb_h, two), 2);
                _mm256_storeu_si256(
                    cb_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cb_avg,
                );

                let cr_0a = _mm256_loadu_si256(cr_full.as_ptr().add(src_off0) as *const __m256i);
                let cr_0b =
                    _mm256_loadu_si256(cr_full.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cr_1a = _mm256_loadu_si256(cr_full.as_ptr().add(src_off1) as *const __m256i);
                let cr_1b =
                    _mm256_loadu_si256(cr_full.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cr_v0 = _mm256_add_epi32(cr_0a, cr_1a);
                let cr_v1 = _mm256_add_epi32(cr_0b, cr_1b);
                let cr_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cr_v0, cr_v1), 0xD8);
                let cr_avg = _mm256_srai_epi32(_mm256_add_epi32(cr_h, two), 2);
                _mm256_storeu_si256(
                    cr_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cr_avg,
                );
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bgra_tile_to_ycbcr420_hpel(
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
        let sx = x0_hpel >> 1;
        let sy = y0_hpel >> 1;
        let hx = (x0_hpel & 1) as usize;
        let hy = (y0_hpel & 1) as usize;
        if hx == 0 && hy == 0 {
            bgra_tile_to_ycbcr420(bgra, fb_w, sx, sy, tw, th, y_plane, cb_plane, cr_plane);
            return;
        }
        if tw < 64 || th < 64 {
            bgra_tile_to_ycbcr420_hpel_scalar(
                bgra, fb_w, x0_hpel, y0_hpel, tw, th, y_plane, cb_plane, cr_plane,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let mask_ff = _mm256_set1_epi32(0xFF);
        let one = _mm256_set1_epi32(1);
        let two = _mm256_set1_epi32(2);
        let mut cb_full = [0i32; 64 * 64];
        let mut cr_full = [0i32; 64 * 64];

        for py in 0..64usize {
            let cy1 = (py + hy).min(63);
            let row0_ptr = bgra
                .as_ptr()
                .add((sy as usize + py) * stride + (sx as usize) * 4);
            let row1_ptr = bgra
                .as_ptr()
                .add((sy as usize + cy1) * stride + (sx as usize) * 4);
            let out_row = py * 64;
            for chunk in 0..8usize {
                let px = chunk * 8;
                let pix00 = _mm256_loadu_si256(row0_ptr.add(px * 4) as *const __m256i);
                let b00 = _mm256_and_si256(pix00, mask_ff);
                let g00 = _mm256_and_si256(_mm256_srli_epi32(pix00, 8), mask_ff);
                let r00 = _mm256_and_si256(_mm256_srli_epi32(pix00, 16), mask_ff);

                let (b, g, r) = if hx == 1 && hy == 0 {
                    let pix01 = if chunk < 7 {
                        _mm256_loadu_si256(row0_ptr.add((px + 1) * 4) as *const __m256i)
                    } else {
                        let perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);
                        _mm256_permutevar8x32_epi32(pix00, perm)
                    };
                    let b01 = _mm256_and_si256(pix01, mask_ff);
                    let g01 = _mm256_and_si256(_mm256_srli_epi32(pix01, 8), mask_ff);
                    let r01 = _mm256_and_si256(_mm256_srli_epi32(pix01, 16), mask_ff);
                    (
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(b00, b01), one), 1),
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(g00, g01), one), 1),
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(r00, r01), one), 1),
                    )
                } else if hx == 0 && hy == 1 {
                    let pix10 = _mm256_loadu_si256(row1_ptr.add(px * 4) as *const __m256i);
                    let b10 = _mm256_and_si256(pix10, mask_ff);
                    let g10 = _mm256_and_si256(_mm256_srli_epi32(pix10, 8), mask_ff);
                    let r10 = _mm256_and_si256(_mm256_srli_epi32(pix10, 16), mask_ff);
                    (
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(b00, b10), one), 1),
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(g00, g10), one), 1),
                        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(r00, r10), one), 1),
                    )
                } else {
                    let pix10 = _mm256_loadu_si256(row1_ptr.add(px * 4) as *const __m256i);
                    let (pix01, pix11) = if chunk < 7 {
                        (
                            _mm256_loadu_si256(row0_ptr.add((px + 1) * 4) as *const __m256i),
                            _mm256_loadu_si256(row1_ptr.add((px + 1) * 4) as *const __m256i),
                        )
                    } else {
                        let perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);
                        (
                            _mm256_permutevar8x32_epi32(pix00, perm),
                            _mm256_permutevar8x32_epi32(pix10, perm),
                        )
                    };
                    let b01 = _mm256_and_si256(pix01, mask_ff);
                    let g01 = _mm256_and_si256(_mm256_srli_epi32(pix01, 8), mask_ff);
                    let r01 = _mm256_and_si256(_mm256_srli_epi32(pix01, 16), mask_ff);
                    let b10 = _mm256_and_si256(pix10, mask_ff);
                    let g10 = _mm256_and_si256(_mm256_srli_epi32(pix10, 8), mask_ff);
                    let r10 = _mm256_and_si256(_mm256_srli_epi32(pix10, 16), mask_ff);
                    let b11 = _mm256_and_si256(pix11, mask_ff);
                    let g11 = _mm256_and_si256(_mm256_srli_epi32(pix11, 8), mask_ff);
                    let r11 = _mm256_and_si256(_mm256_srli_epi32(pix11, 16), mask_ff);
                    (
                        _mm256_srai_epi32(
                            _mm256_add_epi32(
                                _mm256_add_epi32(b00, b01),
                                _mm256_add_epi32(_mm256_add_epi32(b10, b11), two),
                            ),
                            2,
                        ),
                        _mm256_srai_epi32(
                            _mm256_add_epi32(
                                _mm256_add_epi32(g00, g01),
                                _mm256_add_epi32(_mm256_add_epi32(g10, g11), two),
                            ),
                            2,
                        ),
                        _mm256_srai_epi32(
                            _mm256_add_epi32(
                                _mm256_add_epi32(r00, r01),
                                _mm256_add_epi32(_mm256_add_epi32(r10, r11), two),
                            ),
                            2,
                        ),
                    )
                };
                let (y, cb, cr) = rgb_to_ycbcr_8px(b, g, r);
                let dst_off = out_row + px;
                _mm256_storeu_si256(y_plane.as_mut_ptr().add(dst_off) as *mut __m256i, y);
                _mm256_storeu_si256(cb_full.as_mut_ptr().add(dst_off) as *mut __m256i, cb);
                _mm256_storeu_si256(cr_full.as_mut_ptr().add(dst_off) as *mut __m256i, cr);
            }
        }

        for cy in 0..32usize {
            let r0 = (cy * 2) * 64;
            let r1 = r0 + 64;
            let out_base = cy * 32;
            for chunk in 0..4usize {
                let cx = chunk * 8;
                let src_off0 = r0 + cx * 2;
                let src_off1 = r1 + cx * 2;

                let cb_0a = _mm256_loadu_si256(cb_full.as_ptr().add(src_off0) as *const __m256i);
                let cb_0b =
                    _mm256_loadu_si256(cb_full.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cb_1a = _mm256_loadu_si256(cb_full.as_ptr().add(src_off1) as *const __m256i);
                let cb_1b =
                    _mm256_loadu_si256(cb_full.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cb_v0 = _mm256_add_epi32(cb_0a, cb_1a);
                let cb_v1 = _mm256_add_epi32(cb_0b, cb_1b);
                let cb_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cb_v0, cb_v1), 0xD8);
                let cb_avg = _mm256_srai_epi32(_mm256_add_epi32(cb_h, two), 2);
                _mm256_storeu_si256(
                    cb_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cb_avg,
                );

                let cr_0a = _mm256_loadu_si256(cr_full.as_ptr().add(src_off0) as *const __m256i);
                let cr_0b =
                    _mm256_loadu_si256(cr_full.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cr_1a = _mm256_loadu_si256(cr_full.as_ptr().add(src_off1) as *const __m256i);
                let cr_1b =
                    _mm256_loadu_si256(cr_full.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cr_v0 = _mm256_add_epi32(cr_0a, cr_1a);
                let cr_v1 = _mm256_add_epi32(cr_0b, cr_1b);
                let cr_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cr_v0, cr_v1), 0xD8);
                let cr_avg = _mm256_srai_epi32(_mm256_add_epi32(cr_h, two), 2);
                _mm256_storeu_si256(
                    cr_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cr_avg,
                );
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ycbcr420_to_bgra_tile(
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
        if tw < 64 {
            ycbcr420_to_bgra_tile_scalar(
                y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let zero = _mm256_setzero_si256();
        let max255 = _mm256_set1_epi32(255);
        let c128 = _mm256_set1_epi32(128);
        let c359 = _mm256_set1_epi32(359);
        let c88 = _mm256_set1_epi32(88);
        let c183 = _mm256_set1_epi32(183);
        let c454 = _mm256_set1_epi32(454);
        let alpha_opaque = _mm256_set1_epi32(-16777216i32); // 0xFF00_0000
        let dup_idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

        for py in 0..(th as usize) {
            let dst_row = bgra
                .as_mut_ptr()
                .add((y0 as usize + py) * stride + (x0 as usize) * 4);
            let y_row = y_plane.as_ptr().add(py * 64);
            let c_row = (py >> 1) * 32;
            for chunk in 0..8usize {
                let px = chunk * 8;
                let y = _mm256_loadu_si256(y_row.add(px) as *const __m256i);

                // Load 4 chroma samples and duplicate each horizontally -> [c0, c0, c1, c1, c2, c2, c3, c3]
                let cb4 =
                    _mm_loadu_si128(cb_plane.as_ptr().add(c_row + (px >> 1)) as *const __m128i);
                let cr4 =
                    _mm_loadu_si128(cr_plane.as_ptr().add(c_row + (px >> 1)) as *const __m128i);
                let cb = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cb4), dup_idx);
                let cr = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cr4), dup_idx);

                let c = _mm256_sub_epi32(cb, c128);
                let d = _mm256_sub_epi32(cr, c128);

                let r_raw = _mm256_add_epi32(
                    y,
                    _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(c359, d), c128), 8),
                );
                let g_raw = _mm256_sub_epi32(
                    y,
                    _mm256_srai_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(c88, c),
                                _mm256_mullo_epi32(c183, d),
                            ),
                            c128,
                        ),
                        8,
                    ),
                );
                let b_raw = _mm256_add_epi32(
                    y,
                    _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(c454, c), c128), 8),
                );

                let mut r = _mm256_min_epi32(_mm256_max_epi32(r_raw, zero), max255);
                let mut g = _mm256_min_epi32(_mm256_max_epi32(g_raw, zero), max255);
                let mut b = _mm256_min_epi32(_mm256_max_epi32(b_raw, zero), max255);

                let a_shifted = match a_plane {
                    Some(ap) => {
                        let a_load =
                            _mm256_loadu_si256(ap.as_ptr().add(py * 64 + px) as *const __m256i);
                        let av = _mm256_min_epi32(_mm256_max_epi32(a_load, zero), max255);
                        r = _mm256_min_epi32(r, av);
                        g = _mm256_min_epi32(g, av);
                        b = _mm256_min_epi32(b, av);
                        _mm256_slli_epi32(av, 24)
                    }
                    None => alpha_opaque,
                };

                let bgra_px = _mm256_or_si256(
                    _mm256_or_si256(b, _mm256_slli_epi32(g, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(r, 16), a_shifted),
                );
                _mm256_storeu_si256(dst_row.add(px * 4) as *mut __m256i, bgra_px);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bgra_tile_to_ycbcr420_half(
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
        if tw < 64 {
            bgra_tile_to_ycbcr420_half_scalar(
                bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let max_y = th as usize - 1;
        let mask_ff = _mm256_set1_epi32(0xFF);
        let two = _mm256_set1_epi32(2);
        let mut cb_half = [0i32; 32 * 32];
        let mut cr_half = [0i32; 32 * 32];

        for hy in 0..32usize {
            let sy0 = (y0 as usize) + (hy * 2).min(max_y);
            let sy1 = (y0 as usize) + (hy * 2 + 1).min(max_y);
            let row0 = bgra.as_ptr().add(sy0 * stride + (x0 as usize) * 4);
            let row1 = bgra.as_ptr().add(sy1 * stride + (x0 as usize) * 4);
            let out_row = hy * 32;

            for chunk in 0..4usize {
                let hx = chunk * 8;
                // Load 16 source pixels from row0 and row1 (64 bytes each)
                let p0_a = _mm256_loadu_si256(row0.add(hx * 8) as *const __m256i);
                let p0_b = _mm256_loadu_si256(row0.add(hx * 8 + 32) as *const __m256i);
                let p1_a = _mm256_loadu_si256(row1.add(hx * 8) as *const __m256i);
                let p1_b = _mm256_loadu_si256(row1.add(hx * 8 + 32) as *const __m256i);

                let b0_a = _mm256_and_si256(p0_a, mask_ff);
                let b0_b = _mm256_and_si256(p0_b, mask_ff);
                let b1_a = _mm256_and_si256(p1_a, mask_ff);
                let b1_b = _mm256_and_si256(p1_b, mask_ff);
                let b_v0 = _mm256_add_epi32(b0_a, b1_a);
                let b_v1 = _mm256_add_epi32(b0_b, b1_b);
                let b_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(b_v0, b_v1), 0xD8);
                let b = _mm256_srai_epi32(_mm256_add_epi32(b_h, two), 2);

                let g0_a = _mm256_and_si256(_mm256_srli_epi32(p0_a, 8), mask_ff);
                let g0_b = _mm256_and_si256(_mm256_srli_epi32(p0_b, 8), mask_ff);
                let g1_a = _mm256_and_si256(_mm256_srli_epi32(p1_a, 8), mask_ff);
                let g1_b = _mm256_and_si256(_mm256_srli_epi32(p1_b, 8), mask_ff);
                let g_v0 = _mm256_add_epi32(g0_a, g1_a);
                let g_v1 = _mm256_add_epi32(g0_b, g1_b);
                let g_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(g_v0, g_v1), 0xD8);
                let g = _mm256_srai_epi32(_mm256_add_epi32(g_h, two), 2);

                let r0_a = _mm256_and_si256(_mm256_srli_epi32(p0_a, 16), mask_ff);
                let r0_b = _mm256_and_si256(_mm256_srli_epi32(p0_b, 16), mask_ff);
                let r1_a = _mm256_and_si256(_mm256_srli_epi32(p1_a, 16), mask_ff);
                let r1_b = _mm256_and_si256(_mm256_srli_epi32(p1_b, 16), mask_ff);
                let r_v0 = _mm256_add_epi32(r0_a, r1_a);
                let r_v1 = _mm256_add_epi32(r0_b, r1_b);
                let r_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(r_v0, r_v1), 0xD8);
                let r = _mm256_srai_epi32(_mm256_add_epi32(r_h, two), 2);

                let (y, cb, cr) = rgb_to_ycbcr_8px(b, g, r);
                let dst_off = out_row + hx;
                _mm256_storeu_si256(y_plane.as_mut_ptr().add(dst_off) as *mut __m256i, y);
                _mm256_storeu_si256(cb_half.as_mut_ptr().add(dst_off) as *mut __m256i, cb);
                _mm256_storeu_si256(cr_half.as_mut_ptr().add(dst_off) as *mut __m256i, cr);
            }
        }

        // Vectorized 2x2 chroma box-filter downsampling (32x32 -> 16x16).
        for cy in 0..16usize {
            let r0 = (cy * 2) * 32;
            let r1 = r0 + 32;
            let out_base = cy * 16;
            for chunk in 0..2usize {
                let cx = chunk * 8;
                let src_off0 = r0 + cx * 2;
                let src_off1 = r1 + cx * 2;

                let cb_0a = _mm256_loadu_si256(cb_half.as_ptr().add(src_off0) as *const __m256i);
                let cb_0b =
                    _mm256_loadu_si256(cb_half.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cb_1a = _mm256_loadu_si256(cb_half.as_ptr().add(src_off1) as *const __m256i);
                let cb_1b =
                    _mm256_loadu_si256(cb_half.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cb_v0 = _mm256_add_epi32(cb_0a, cb_1a);
                let cb_v1 = _mm256_add_epi32(cb_0b, cb_1b);
                let cb_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cb_v0, cb_v1), 0xD8);
                let cb_avg = _mm256_srai_epi32(_mm256_add_epi32(cb_h, two), 2);
                _mm256_storeu_si256(
                    cb_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cb_avg,
                );

                let cr_0a = _mm256_loadu_si256(cr_half.as_ptr().add(src_off0) as *const __m256i);
                let cr_0b =
                    _mm256_loadu_si256(cr_half.as_ptr().add(src_off0 + 8) as *const __m256i);
                let cr_1a = _mm256_loadu_si256(cr_half.as_ptr().add(src_off1) as *const __m256i);
                let cr_1b =
                    _mm256_loadu_si256(cr_half.as_ptr().add(src_off1 + 8) as *const __m256i);
                let cr_v0 = _mm256_add_epi32(cr_0a, cr_1a);
                let cr_v1 = _mm256_add_epi32(cr_0b, cr_1b);
                let cr_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cr_v0, cr_v1), 0xD8);
                let cr_avg = _mm256_srai_epi32(_mm256_add_epi32(cr_h, two), 2);
                _mm256_storeu_si256(
                    cr_plane.as_mut_ptr().add(out_base + cx) as *mut __m256i,
                    cr_avg,
                );
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ycbcr420_half_to_bgra_tile(
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
        if tw < 64 {
            ycbcr420_half_to_bgra_tile_scalar(
                y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let zero = _mm256_setzero_si256();
        let max255 = _mm256_set1_epi32(255);
        let two = _mm256_set1_epi32(2);
        let c128 = _mm256_set1_epi32(128);
        let c359 = _mm256_set1_epi32(359);
        let c88 = _mm256_set1_epi32(88);
        let c183 = _mm256_set1_epi32(183);
        let c454 = _mm256_set1_epi32(454);
        let alpha_opaque = _mm256_set1_epi32(-16777216i32); // 0xFF00_0000

        // For 8 output pixels `px..px+8`, `hx0 = px >> 1` takes 4 values `[h, h, h+1, h+1, h+2, h+2, h+3, h+3]`,
        // and `hx1` takes `[h, h+1, h+1, h+2, h+2, h+3, h+3, (h+4).min(31)]`.
        // Since `h` is at most 28, we can load 8 values starting at `h` (clamped at 24 for the last chunk)
        // and use `permutevar8x32` to gather `hx0` and `hx1` in one instruction!
        let perm_hx0_norm = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
        let perm_hx1_norm = _mm256_setr_epi32(0, 1, 1, 2, 2, 3, 3, 4);
        // For the last chunk (`chunk == 7`, `h = 28`), we load 8 i32s from offset 24 (`[24..32]`),
        // so indices `28, 29, 30, 31` correspond to lane offsets `4, 5, 6, 7`.
        let perm_hx0_last = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
        let perm_hx1_last = _mm256_setr_epi32(4, 5, 5, 6, 6, 7, 7, 7);
        let perm_chroma = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

        for py in 0..(th as usize) {
            let dst_row = bgra
                .as_mut_ptr()
                .add((y0 as usize + py) * stride + (x0 as usize) * 4);
            let hy0 = py >> 1;
            let hy1 = if (py & 1) == 0 {
                hy0
            } else {
                (hy0 + 1).min(31)
            };
            let r0 = hy0 * 32;
            let r1 = hy1 * 32;
            let c_row = (hy0 >> 1) * 16;

            for chunk in 0..8usize {
                let px = chunk * 8;
                let (load_off, p_hx0, p_hx1) = if chunk < 7 {
                    (chunk * 4, perm_hx0_norm, perm_hx1_norm)
                } else {
                    (24usize, perm_hx0_last, perm_hx1_last)
                };

                let y_r0_src =
                    _mm256_loadu_si256(y_plane.as_ptr().add(r0 + load_off) as *const __m256i);
                let y_r1_src =
                    _mm256_loadu_si256(y_plane.as_ptr().add(r1 + load_off) as *const __m256i);

                let y00 = _mm256_permutevar8x32_epi32(y_r0_src, p_hx0);
                let y01 = _mm256_permutevar8x32_epi32(y_r0_src, p_hx1);
                let y10 = _mm256_permutevar8x32_epi32(y_r1_src, p_hx0);
                let y11 = _mm256_permutevar8x32_epi32(y_r1_src, p_hx1);
                let y_sum = _mm256_add_epi32(
                    _mm256_add_epi32(y00, y01),
                    _mm256_add_epi32(_mm256_add_epi32(y10, y11), two),
                );
                let y = _mm256_srai_epi32(y_sum, 2);

                // Each 8-pixel chunk spans 2 chroma samples (`c_idx = c_row + chunk * 2` and `+ 1`), each repeated 4 times.
                let c_off = c_row + chunk * 2;
                let cb2 = _mm_setr_epi32(cb_plane[c_off], cb_plane[c_off + 1], 0, 0);
                let cr2 = _mm_setr_epi32(cr_plane[c_off], cr_plane[c_off + 1], 0, 0);
                let cb = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cb2), perm_chroma);
                let cr = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cr2), perm_chroma);

                let c = _mm256_sub_epi32(cb, c128);
                let d = _mm256_sub_epi32(cr, c128);

                let r_raw = _mm256_add_epi32(
                    y,
                    _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(c359, d), c128), 8),
                );
                let g_raw = _mm256_sub_epi32(
                    y,
                    _mm256_srai_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(c88, c),
                                _mm256_mullo_epi32(c183, d),
                            ),
                            c128,
                        ),
                        8,
                    ),
                );
                let b_raw = _mm256_add_epi32(
                    y,
                    _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(c454, c), c128), 8),
                );

                let mut r = _mm256_min_epi32(_mm256_max_epi32(r_raw, zero), max255);
                let mut g = _mm256_min_epi32(_mm256_max_epi32(g_raw, zero), max255);
                let mut b = _mm256_min_epi32(_mm256_max_epi32(b_raw, zero), max255);

                let a_shifted = match a_plane {
                    Some(ap) => {
                        let a_r0_src =
                            _mm256_loadu_si256(ap.as_ptr().add(r0 + load_off) as *const __m256i);
                        let a_r1_src =
                            _mm256_loadu_si256(ap.as_ptr().add(r1 + load_off) as *const __m256i);
                        let a00 = _mm256_permutevar8x32_epi32(a_r0_src, p_hx0);
                        let a01 = _mm256_permutevar8x32_epi32(a_r0_src, p_hx1);
                        let a10 = _mm256_permutevar8x32_epi32(a_r1_src, p_hx0);
                        let a11 = _mm256_permutevar8x32_epi32(a_r1_src, p_hx1);
                        let a_avg = _mm256_srai_epi32(
                            _mm256_add_epi32(
                                _mm256_add_epi32(a00, a01),
                                _mm256_add_epi32(_mm256_add_epi32(a10, a11), two),
                            ),
                            2,
                        );
                        let av = _mm256_min_epi32(_mm256_max_epi32(a_avg, zero), max255);
                        r = _mm256_min_epi32(r, av);
                        g = _mm256_min_epi32(g, av);
                        b = _mm256_min_epi32(b, av);
                        _mm256_slli_epi32(av, 24)
                    }
                    None => alpha_opaque,
                };

                let bgra_px = _mm256_or_si256(
                    _mm256_or_si256(b, _mm256_slli_epi32(g, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(r, 16), a_shifted),
                );
                _mm256_storeu_si256(dst_row.add(px * 4) as *mut __m256i, bgra_px);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bgra_tile_to_ycbcr420_quarter(
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
        if tw < 64 {
            bgra_tile_to_ycbcr420_quarter_scalar(
                bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let max_y = th as usize - 1;
        let mask_ff = _mm256_set1_epi32(0xFF);
        let two = _mm256_set1_epi32(2);
        let eight = _mm256_set1_epi32(8);
        let perm_quad = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let mut cb_quarter = [0i32; 16 * 16];
        let mut cr_quarter = [0i32; 16 * 16];

        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn hsum_4x4(
            s0: __m256i,
            s1: __m256i,
            s2: __m256i,
            s3: __m256i,
            perm_quad: __m256i,
            eight: __m256i,
        ) -> __m256i {
            let t01 = _mm256_hadd_epi32(s0, s1);
            let t23 = _mm256_hadd_epi32(s2, s3);
            let u = _mm256_hadd_epi32(t01, t23);
            let ordered = _mm256_permutevar8x32_epi32(u, perm_quad);
            _mm256_srai_epi32(_mm256_add_epi32(ordered, eight), 4)
        }

        for qy in 0..16usize {
            let base_y = qy * 4;
            let sy0 = (y0 as usize) + base_y.min(max_y);
            let sy1 = (y0 as usize) + (base_y + 1).min(max_y);
            let sy2 = (y0 as usize) + (base_y + 2).min(max_y);
            let sy3 = (y0 as usize) + (base_y + 3).min(max_y);
            let r0 = bgra.as_ptr().add(sy0 * stride + (x0 as usize) * 4);
            let r1 = bgra.as_ptr().add(sy1 * stride + (x0 as usize) * 4);
            let r2 = bgra.as_ptr().add(sy2 * stride + (x0 as usize) * 4);
            let r3 = bgra.as_ptr().add(sy3 * stride + (x0 as usize) * 4);
            let out_row = qy * 16;

            for chunk in 0..2usize {
                let byte_base = chunk * 128; // 32 pixels = 128 bytes -> 8 quarter-res outputs
                let mut sb = [_mm256_setzero_si256(); 4];
                let mut sg = [_mm256_setzero_si256(); 4];
                let mut sr = [_mm256_setzero_si256(); 4];
                for k in 0..4usize {
                    let off = byte_base + k * 32;
                    let p0 = _mm256_loadu_si256(r0.add(off) as *const __m256i);
                    let p1 = _mm256_loadu_si256(r1.add(off) as *const __m256i);
                    let p2 = _mm256_loadu_si256(r2.add(off) as *const __m256i);
                    let p3 = _mm256_loadu_si256(r3.add(off) as *const __m256i);

                    let b01 = _mm256_add_epi32(
                        _mm256_and_si256(p0, mask_ff),
                        _mm256_and_si256(p1, mask_ff),
                    );
                    let b23 = _mm256_add_epi32(
                        _mm256_and_si256(p2, mask_ff),
                        _mm256_and_si256(p3, mask_ff),
                    );
                    sb[k] = _mm256_add_epi32(b01, b23);

                    let g01 = _mm256_add_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(p0, 8), mask_ff),
                        _mm256_and_si256(_mm256_srli_epi32(p1, 8), mask_ff),
                    );
                    let g23 = _mm256_add_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(p2, 8), mask_ff),
                        _mm256_and_si256(_mm256_srli_epi32(p3, 8), mask_ff),
                    );
                    sg[k] = _mm256_add_epi32(g01, g23);

                    let r01 = _mm256_add_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(p0, 16), mask_ff),
                        _mm256_and_si256(_mm256_srli_epi32(p1, 16), mask_ff),
                    );
                    let r23 = _mm256_add_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(p2, 16), mask_ff),
                        _mm256_and_si256(_mm256_srli_epi32(p3, 16), mask_ff),
                    );
                    sr[k] = _mm256_add_epi32(r01, r23);
                }

                let b = hsum_4x4(sb[0], sb[1], sb[2], sb[3], perm_quad, eight);
                let g = hsum_4x4(sg[0], sg[1], sg[2], sg[3], perm_quad, eight);
                let r = hsum_4x4(sr[0], sr[1], sr[2], sr[3], perm_quad, eight);
                let (y, cb, cr) = rgb_to_ycbcr_8px(b, g, r);
                let dst_off = out_row + chunk * 8;
                _mm256_storeu_si256(y_plane.as_mut_ptr().add(dst_off) as *mut __m256i, y);
                _mm256_storeu_si256(cb_quarter.as_mut_ptr().add(dst_off) as *mut __m256i, cb);
                _mm256_storeu_si256(cr_quarter.as_mut_ptr().add(dst_off) as *mut __m256i, cr);
            }
        }

        // Vectorized 2x2 chroma box-filter downsampling (16x16 -> 8x8).
        for cy in 0..8usize {
            let r0 = (cy * 2) * 16;
            let r1 = r0 + 16;
            let out_base = cy * 8;

            let cb_0a = _mm256_loadu_si256(cb_quarter.as_ptr().add(r0) as *const __m256i);
            let cb_0b = _mm256_loadu_si256(cb_quarter.as_ptr().add(r0 + 8) as *const __m256i);
            let cb_1a = _mm256_loadu_si256(cb_quarter.as_ptr().add(r1) as *const __m256i);
            let cb_1b = _mm256_loadu_si256(cb_quarter.as_ptr().add(r1 + 8) as *const __m256i);
            let cb_v0 = _mm256_add_epi32(cb_0a, cb_1a);
            let cb_v1 = _mm256_add_epi32(cb_0b, cb_1b);
            let cb_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cb_v0, cb_v1), 0xD8);
            let cb_avg = _mm256_srai_epi32(_mm256_add_epi32(cb_h, two), 2);
            _mm256_storeu_si256(cb_plane.as_mut_ptr().add(out_base) as *mut __m256i, cb_avg);

            let cr_0a = _mm256_loadu_si256(cr_quarter.as_ptr().add(r0) as *const __m256i);
            let cr_0b = _mm256_loadu_si256(cr_quarter.as_ptr().add(r0 + 8) as *const __m256i);
            let cr_1a = _mm256_loadu_si256(cr_quarter.as_ptr().add(r1) as *const __m256i);
            let cr_1b = _mm256_loadu_si256(cr_quarter.as_ptr().add(r1 + 8) as *const __m256i);
            let cr_v0 = _mm256_add_epi32(cr_0a, cr_1a);
            let cr_v1 = _mm256_add_epi32(cr_0b, cr_1b);
            let cr_h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(cr_v0, cr_v1), 0xD8);
            let cr_avg = _mm256_srai_epi32(_mm256_add_epi32(cr_h, two), 2);
            _mm256_storeu_si256(cr_plane.as_mut_ptr().add(out_base) as *mut __m256i, cr_avg);
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ycbcr420_quarter_to_bgra_tile(
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
        if tw < 64 {
            ycbcr420_quarter_to_bgra_tile_scalar(
                y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let zero = _mm256_setzero_si256();
        let max255 = _mm256_set1_epi32(255);
        let eight = _mm256_set1_epi32(8);
        let alpha_opaque = _mm256_set1_epi32(-16777216i32); // 0xFF00_0000

        let v_wx = _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3);
        let v_wx0 = _mm256_setr_epi32(4, 3, 2, 1, 4, 3, 2, 1);
        let perm_qx0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
        let perm_qx1 = _mm256_setr_epi32(1, 1, 1, 1, 2, 2, 2, 2);

        let mut yb = [0i32; 24];
        let mut ab = [0i32; 24];

        for py in 0..(th as usize) {
            let dst_row = bgra
                .as_mut_ptr()
                .add((y0 as usize + py) * stride + (x0 as usize) * 4);
            let qy0 = py >> 2;
            let qy1 = (qy0 + 1).min(15);
            let wy = (py & 3) as i32;
            let wy0 = 4 - wy;
            let v_wy0 = _mm256_set1_epi32(wy0);
            let v_wy = _mm256_set1_epi32(wy);
            let r0 = qy0 * 16;
            let r1 = qy1 * 16;
            let c_row = (qy0 >> 1) * 8;

            // Pre-blend the 16 horizontal luma samples vertically (`y_r0 * wy0 + y_r1 * wy`).
            let y0_lo = _mm256_loadu_si256(y_plane.as_ptr().add(r0) as *const __m256i);
            let y0_hi = _mm256_loadu_si256(y_plane.as_ptr().add(r0 + 8) as *const __m256i);
            let y1_lo = _mm256_loadu_si256(y_plane.as_ptr().add(r1) as *const __m256i);
            let y1_hi = _mm256_loadu_si256(y_plane.as_ptr().add(r1 + 8) as *const __m256i);
            let yb_lo = _mm256_add_epi32(
                _mm256_mullo_epi32(y0_lo, v_wy0),
                _mm256_mullo_epi32(y1_lo, v_wy),
            );
            let yb_hi = _mm256_add_epi32(
                _mm256_mullo_epi32(y0_hi, v_wy0),
                _mm256_mullo_epi32(y1_hi, v_wy),
            );
            _mm256_storeu_si256(yb.as_mut_ptr() as *mut __m256i, yb_lo);
            _mm256_storeu_si256(yb.as_mut_ptr().add(8) as *mut __m256i, yb_hi);
            yb[16] = yb[15];

            if let Some(ap) = a_plane {
                let a0_lo = _mm256_loadu_si256(ap.as_ptr().add(r0) as *const __m256i);
                let a0_hi = _mm256_loadu_si256(ap.as_ptr().add(r0 + 8) as *const __m256i);
                let a1_lo = _mm256_loadu_si256(ap.as_ptr().add(r1) as *const __m256i);
                let a1_hi = _mm256_loadu_si256(ap.as_ptr().add(r1 + 8) as *const __m256i);
                let ab_lo = _mm256_add_epi32(
                    _mm256_mullo_epi32(a0_lo, v_wy0),
                    _mm256_mullo_epi32(a1_lo, v_wy),
                );
                let ab_hi = _mm256_add_epi32(
                    _mm256_mullo_epi32(a0_hi, v_wy0),
                    _mm256_mullo_epi32(a1_hi, v_wy),
                );
                _mm256_storeu_si256(ab.as_mut_ptr() as *mut __m256i, ab_lo);
                _mm256_storeu_si256(ab.as_mut_ptr().add(8) as *mut __m256i, ab_hi);
                ab[16] = ab[15];
            }

            for chunk in 0..8usize {
                let px = chunk * 8;
                let load_off = chunk * 2;
                let y_src = _mm256_loadu_si256(yb.as_ptr().add(load_off) as *const __m256i);
                let y_q0 = _mm256_permutevar8x32_epi32(y_src, perm_qx0);
                let y_q1 = _mm256_permutevar8x32_epi32(y_src, perm_qx1);
                let y = _mm256_srai_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(
                            _mm256_mullo_epi32(y_q0, v_wx0),
                            _mm256_mullo_epi32(y_q1, v_wx),
                        ),
                        eight,
                    ),
                    4,
                );

                // All 8 pixels in `chunk` share the exact same chroma sample `c_row + chunk`!
                let c_idx = c_row + chunk;
                let c = cb_plane[c_idx] - 128;
                let d = cr_plane[c_idx] - 128;
                let r_off = _mm256_set1_epi32((359 * d + 128) >> 8);
                let g_off = _mm256_set1_epi32((88 * c + 183 * d + 128) >> 8);
                let b_off = _mm256_set1_epi32((454 * c + 128) >> 8);

                let mut r =
                    _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(y, r_off), zero), max255);
                let mut g =
                    _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(y, g_off), zero), max255);
                let mut b =
                    _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(y, b_off), zero), max255);

                let a_shifted = if a_plane.is_some() {
                    let a_src = _mm256_loadu_si256(ab.as_ptr().add(load_off) as *const __m256i);
                    let a_q0 = _mm256_permutevar8x32_epi32(a_src, perm_qx0);
                    let a_q1 = _mm256_permutevar8x32_epi32(a_src, perm_qx1);
                    let a_val = _mm256_srai_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(a_q0, v_wx0),
                                _mm256_mullo_epi32(a_q1, v_wx),
                            ),
                            eight,
                        ),
                        4,
                    );
                    let av = _mm256_min_epi32(_mm256_max_epi32(a_val, zero), max255);
                    r = _mm256_min_epi32(r, av);
                    g = _mm256_min_epi32(g, av);
                    b = _mm256_min_epi32(b, av);
                    _mm256_slli_epi32(av, 24)
                } else {
                    alpha_opaque
                };

                let bgra_px = _mm256_or_si256(
                    _mm256_or_si256(b, _mm256_slli_epi32(g, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(r, 16), a_shifted),
                );
                _mm256_storeu_si256(dst_row.add(px * 4) as *mut __m256i, bgra_px);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile_equals_shifted(
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
        let chunks32 = row_bytes / 32;
        for py in 0..(th as usize) {
            let s_off = (sy as usize + py) * stride + (sx as usize) * 4;
            let d_off = (dy as usize + py) * stride + (dx as usize) * 4;
            let sp = src_fb.as_ptr().add(s_off);
            let dp = dst_fb.as_ptr().add(d_off);
            for c in 0..chunks32 {
                let va = _mm256_loadu_si256(sp.add(c * 32) as *const __m256i);
                let vb = _mm256_loadu_si256(dp.add(c * 32) as *const __m256i);
                let eq = _mm256_cmpeq_epi8(va, vb);
                if _mm256_movemask_epi8(eq) != -1 {
                    return false;
                }
            }
            let rem_start = chunks32 * 32;
            if rem_start < row_bytes
                && src_fb[s_off + rem_start..s_off + row_bytes]
                    != dst_fb[d_off + rem_start..d_off + row_bytes]
            {
                return false;
            }
        }
        true
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn tile_is_opaque(
        bgra: &[u8],
        fb_w: u32,
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> bool {
        let stride = fb_w as usize * 4;
        let pixels = tw as usize;
        let chunks8 = pixels / 8;
        let alpha_mask = _mm256_set1_epi32(-16777216i32); // 0xFF00_0000
        for py in 0..(th as usize) {
            let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
            let rp = bgra.as_ptr().add(row_base);
            for c in 0..chunks8 {
                let v = _mm256_loadu_si256(rp.add(c * 32) as *const __m256i);
                let masked = _mm256_and_si256(v, alpha_mask);
                let eq = _mm256_cmpeq_epi32(masked, alpha_mask);
                if _mm256_movemask_epi8(eq) != -1 {
                    return false;
                }
            }
            for px in (chunks8 * 8)..pixels {
                if bgra[row_base + px * 4 + 3] != 255 {
                    return false;
                }
            }
        }
        true
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn count_identical_pixels(
        ref_bgra: &[u8],
        cur_bgra: &[u8],
        fb_w: u32,
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> usize {
        let stride = fb_w as usize * 4;
        let pixels = tw as usize;
        let chunks8 = pixels / 8;
        let mut same = 0usize;
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            let rp = ref_bgra.as_ptr().add(off);
            let cp = cur_bgra.as_ptr().add(off);
            for c in 0..chunks8 {
                let vr = _mm256_loadu_si256(rp.add(c * 32) as *const __m256i);
                let vc = _mm256_loadu_si256(cp.add(c * 32) as *const __m256i);
                let eq = _mm256_cmpeq_epi32(vr, vc);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32;
                same += mask.count_ones() as usize;
            }
            for px in (chunks8 * 8)..pixels {
                let p = off + px * 4;
                if ref_bgra[p..p + 4] == cur_bgra[p..p + 4] {
                    same += 1;
                }
            }
        }
        same
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn xor_encode_row(ref_row: &mut [u8], cur_row: &[u8], out: &mut Vec<u8>) {
        let len = cur_row.len();
        let start = out.len();
        out.resize(start + len, 0);
        let dst = &mut out[start..start + len];
        let chunks32 = len / 32;
        let rp = ref_row.as_mut_ptr();
        let cp = cur_row.as_ptr();
        let dp = dst.as_mut_ptr();

        for c in 0..chunks32 {
            let off = c * 32;
            let vr = _mm256_loadu_si256(rp.add(off) as *const __m256i);
            let vc = _mm256_loadu_si256(cp.add(off) as *const __m256i);
            let vx = _mm256_xor_si256(vc, vr);
            _mm256_storeu_si256(dp.add(off) as *mut __m256i, vx);
            _mm256_storeu_si256(rp.add(off) as *mut __m256i, vc);
        }
        for i in (chunks32 * 32)..len {
            let c = cur_row[i];
            dst[i] = c ^ ref_row[i];
            ref_row[i] = c;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn xor_decode_row(dst_row: &mut [u8], src_row: &[u8]) {
        let len = dst_row.len();
        let chunks32 = len / 32;
        let dp = dst_row.as_mut_ptr();
        let sp = src_row.as_ptr();
        for c in 0..chunks32 {
            let off = c * 32;
            let vd = _mm256_loadu_si256(dp.add(off) as *const __m256i);
            let vs = _mm256_loadu_si256(sp.add(off) as *const __m256i);
            _mm256_storeu_si256(dp.add(off) as *mut __m256i, _mm256_xor_si256(vd, vs));
        }
        for i in (chunks32 * 32)..len {
            dst_row[i] ^= src_row[i];
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose_2x4x4_epi32(
        r0: &mut __m256i,
        r1: &mut __m256i,
        r2: &mut __m256i,
        r3: &mut __m256i,
    ) {
        let t0 = _mm256_unpacklo_epi32(*r0, *r1);
        let t1 = _mm256_unpackhi_epi32(*r0, *r1);
        let t2 = _mm256_unpacklo_epi32(*r2, *r3);
        let t3 = _mm256_unpackhi_epi32(*r2, *r3);
        *r0 = _mm256_unpacklo_epi64(t0, t2);
        *r1 = _mm256_unpackhi_epi64(t0, t2);
        *r2 = _mm256_unpacklo_epi64(t1, t3);
        *r3 = _mm256_unpackhi_epi64(t1, t3);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn deblock_8_lanes_avx2(
        vp1: &mut __m256i,
        vp0: &mut __m256i,
        vq0: &mut __m256i,
        vq1: &mut __m256i,
        v_beta: __m256i,
        v_beta_half: __m256i,
        v_tc: __m256i,
        v_neg_tc: __m256i,
        v_tc_half: __m256i,
        v_neg_tc_half: __m256i,
    ) {
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(255);
        let v_four = _mm256_set1_epi32(4);

        let diff_pq0 = _mm256_abs_epi32(_mm256_sub_epi32(*vp0, *vq0));
        let diff_p10 = _mm256_abs_epi32(_mm256_sub_epi32(*vp1, *vp0));
        let diff_q10 = _mm256_abs_epi32(_mm256_sub_epi32(*vq1, *vq0));

        let m0 = _mm256_cmpgt_epi32(v_beta, diff_pq0);
        let m1 = _mm256_cmpgt_epi32(v_beta_half, diff_p10);
        let m2 = _mm256_cmpgt_epi32(v_beta_half, diff_q10);
        let mask = _mm256_and_si256(m0, _mm256_and_si256(m1, m2));

        let qp0 = _mm256_sub_epi32(*vq0, *vp0);
        let qp0_3 = _mm256_add_epi32(qp0, _mm256_slli_epi32(qp0, 1));
        let pq1 = _mm256_sub_epi32(*vp1, *vq1);
        let num = _mm256_add_epi32(_mm256_add_epi32(qp0_3, pq1), v_four);
        let delta = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(num, 3), v_neg_tc), v_tc);
        let delta1 = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32(delta, 1), v_neg_tc_half),
            v_tc_half,
        );

        let np0 = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(*vp0, delta), zero),
            max_val,
        );
        let nq0 = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_sub_epi32(*vq0, delta), zero),
            max_val,
        );
        let np1 = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(*vp1, delta1), zero),
            max_val,
        );
        let nq1 = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_sub_epi32(*vq1, delta1), zero),
            max_val,
        );

        *vp1 = _mm256_blendv_epi8(*vp1, np1, mask);
        *vp0 = _mm256_blendv_epi8(*vp0, np0, mask);
        *vq0 = _mm256_blendv_epi8(*vq0, nq0, mask);
        *vq1 = _mm256_blendv_epi8(*vq1, nq1, mask);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn deblock_and_sao_plane(
        plane: &mut [i32],
        plane_dim: usize,
        blocks_x: usize,
        blocks_y: usize,
        q_dc: i32,
    ) {
        if blocks_x == 0
            || blocks_y == 0
            || blocks_x * 8 > plane_dim
            || blocks_y * 8 > plane_dim
            || plane.len() < (blocks_y * 8 - 1) * plane_dim + blocks_x * 8
        {
            deblock_and_sao_plane_scalar(plane, plane_dim, blocks_x, blocks_y, q_dc);
            return;
        }

        let beta = (q_dc + 4).clamp(6, 36);
        let tc = ((q_dc + 3) >> 2).clamp(1, 10);
        let v_beta = _mm256_set1_epi32(beta);
        let v_beta_half = _mm256_set1_epi32(beta >> 1);
        let v_tc = _mm256_set1_epi32(tc);
        let v_neg_tc = _mm256_set1_epi32(-tc);
        let v_tc_half = _mm256_set1_epi32(tc >> 1);
        let v_neg_tc_half = _mm256_set1_epi32(-(tc >> 1));

        let base_ptr = plane.as_mut_ptr();

        // 1. Vertical 8x8 block boundaries (8 rows processed simultaneously per boundary via 2x4x4 transpose)
        if blocks_x > 1 {
            for by in 0..blocks_y {
                let y0 = by * 8;
                for bx in 1..blocks_x {
                    let x = bx * 8;
                    let p = base_ptr.add(y0 * plane_dim + x - 2);
                    let r0 = _mm_loadu_si128(p as *const __m128i);
                    let r1 = _mm_loadu_si128(p.add(plane_dim) as *const __m128i);
                    let r2 = _mm_loadu_si128(p.add(plane_dim * 2) as *const __m128i);
                    let r3 = _mm_loadu_si128(p.add(plane_dim * 3) as *const __m128i);
                    let r4 = _mm_loadu_si128(p.add(plane_dim * 4) as *const __m128i);
                    let r5 = _mm_loadu_si128(p.add(plane_dim * 5) as *const __m128i);
                    let r6 = _mm_loadu_si128(p.add(plane_dim * 6) as *const __m128i);
                    let r7 = _mm_loadu_si128(p.add(plane_dim * 7) as *const __m128i);

                    let mut vp1 = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r4, 1);
                    let mut vp0 = _mm256_inserti128_si256(_mm256_castsi128_si256(r1), r5, 1);
                    let mut vq0 = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r6, 1);
                    let mut vq1 = _mm256_inserti128_si256(_mm256_castsi128_si256(r3), r7, 1);

                    transpose_2x4x4_epi32(&mut vp1, &mut vp0, &mut vq0, &mut vq1);
                    deblock_8_lanes_avx2(
                        &mut vp1,
                        &mut vp0,
                        &mut vq0,
                        &mut vq1,
                        v_beta,
                        v_beta_half,
                        v_tc,
                        v_neg_tc,
                        v_tc_half,
                        v_neg_tc_half,
                    );
                    transpose_2x4x4_epi32(&mut vp1, &mut vp0, &mut vq0, &mut vq1);

                    _mm_storeu_si128(p as *mut __m128i, _mm256_castsi256_si128(vp1));
                    _mm_storeu_si128(
                        p.add(plane_dim) as *mut __m128i,
                        _mm256_castsi256_si128(vp0),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 2) as *mut __m128i,
                        _mm256_castsi256_si128(vq0),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 3) as *mut __m128i,
                        _mm256_castsi256_si128(vq1),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 4) as *mut __m128i,
                        _mm256_extracti128_si256(vp1, 1),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 5) as *mut __m128i,
                        _mm256_extracti128_si256(vp0, 1),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 6) as *mut __m128i,
                        _mm256_extracti128_si256(vq0, 1),
                    );
                    _mm_storeu_si128(
                        p.add(plane_dim * 7) as *mut __m128i,
                        _mm256_extracti128_si256(vq1, 1),
                    );
                }
            }
        }

        // 2. Horizontal 8x8 block boundaries (8 columns processed simultaneously in contiguous __m256i)
        if blocks_y > 1 {
            for by in 1..blocks_y {
                let y = by * 8;
                for bx in 0..blocks_x {
                    let x = bx * 8;
                    let p_p1 = base_ptr.add((y - 2) * plane_dim + x);
                    let p_p0 = base_ptr.add((y - 1) * plane_dim + x);
                    let p_q0 = base_ptr.add(y * plane_dim + x);
                    let p_q1 = base_ptr.add((y + 1) * plane_dim + x);

                    let mut vp1 = _mm256_loadu_si256(p_p1 as *const __m256i);
                    let mut vp0 = _mm256_loadu_si256(p_p0 as *const __m256i);
                    let mut vq0 = _mm256_loadu_si256(p_q0 as *const __m256i);
                    let mut vq1 = _mm256_loadu_si256(p_q1 as *const __m256i);

                    deblock_8_lanes_avx2(
                        &mut vp1,
                        &mut vp0,
                        &mut vq0,
                        &mut vq1,
                        v_beta,
                        v_beta_half,
                        v_tc,
                        v_neg_tc,
                        v_tc_half,
                        v_neg_tc_half,
                    );

                    _mm256_storeu_si256(p_p1 as *mut __m256i, vp1);
                    _mm256_storeu_si256(p_p0 as *mut __m256i, vp0);
                    _mm256_storeu_si256(p_q0 as *mut __m256i, vq0);
                    _mm256_storeu_si256(p_q1 as *mut __m256i, vq1);
                }
            }
        }

        // 3. Edge-preserving Sample Adaptive Offset (SAO) 1D Edge Offset anti-ringing damping
        let sao_thresh = (beta >> 1).max(3);
        let v_sao_thresh = _mm256_set1_epi32(sao_thresh);
        let zero = _mm256_setzero_si256();
        let v_one = _mm256_set1_epi32(1);
        let v_two = _mm256_set1_epi32(2);
        let max_val = _mm256_set1_epi32(255);
        let idx_prev = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        let idx_next = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

        let active_rows = blocks_y * 8;
        for y in 0..active_rows {
            let row_ptr = base_ptr.add(y * plane_dim);
            for bx in 0..blocks_x {
                let ptr = row_ptr.add(bx * 8);
                let v_cur = _mm256_loadu_si256(ptr as *const __m256i);
                let v_prev = _mm256_permutevar8x32_epi32(v_cur, idx_prev);
                let v_next = _mm256_permutevar8x32_epi32(v_cur, idx_next);

                let d_left = _mm256_sub_epi32(v_cur, v_prev);
                let d_right = _mm256_sub_epi32(v_cur, v_next);
                let abs_l = _mm256_abs_epi32(d_left);
                let abs_r = _mm256_abs_epi32(d_right);

                let m_thresh = _mm256_and_si256(
                    _mm256_cmpgt_epi32(v_sao_thresh, abs_l),
                    _mm256_cmpgt_epi32(v_sao_thresh, abs_r),
                );
                let min_abs = _mm256_min_epi32(abs_l, abs_r);
                let step = _mm256_min_epi32(
                    _mm256_max_epi32(
                        _mm256_srai_epi32(_mm256_add_epi32(min_abs, v_one), 2),
                        v_one,
                    ),
                    v_two,
                );
                let m_pos = _mm256_and_si256(
                    m_thresh,
                    _mm256_and_si256(
                        _mm256_cmpgt_epi32(d_left, zero),
                        _mm256_cmpgt_epi32(d_right, zero),
                    ),
                );
                let m_neg = _mm256_and_si256(
                    m_thresh,
                    _mm256_and_si256(
                        _mm256_cmpgt_epi32(zero, d_left),
                        _mm256_cmpgt_epi32(zero, d_right),
                    ),
                );

                let sub_val = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sub_epi32(v_cur, step), zero),
                    max_val,
                );
                let add_val = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_add_epi32(v_cur, step), zero),
                    max_val,
                );
                let res =
                    _mm256_blendv_epi8(_mm256_blendv_epi8(v_cur, sub_val, m_pos), add_val, m_neg);
                _mm256_storeu_si256(ptr as *mut __m256i, res);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON Hardware SIMD Kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn transpose_4x4_s32(
        r0: &mut int32x4_t,
        r1: &mut int32x4_t,
        r2: &mut int32x4_t,
        r3: &mut int32x4_t,
    ) {
        let t0 = vtrnq_s32(*r0, *r1);
        let t1 = vtrnq_s32(*r2, *r3);
        *r0 = vcombine_s32(vget_low_s32(t0.0), vget_low_s32(t1.0));
        *r1 = vcombine_s32(vget_low_s32(t0.1), vget_low_s32(t1.1));
        *r2 = vcombine_s32(vget_high_s32(t0.0), vget_high_s32(t1.0));
        *r3 = vcombine_s32(vget_high_s32(t0.1), vget_high_s32(t1.1));
    }

    #[inline]
    unsafe fn dct_1d_4cols(
        r0: &mut int32x4_t,
        r1: &mut int32x4_t,
        r2: &mut int32x4_t,
        r3: &mut int32x4_t,
        r4: &mut int32x4_t,
        r5: &mut int32x4_t,
        r6: &mut int32x4_t,
        r7: &mut int32x4_t,
        shift: i32,
    ) {
        let round = vdupq_n_s32(1i32 << (shift - 1));
        let neg_shift = vdupq_n_s32(-shift);

        let s0 = vaddq_s32(*r0, *r7);
        let s1 = vaddq_s32(*r1, *r6);
        let s2 = vaddq_s32(*r2, *r5);
        let s3 = vaddq_s32(*r3, *r4);
        let d0 = vsubq_s32(*r0, *r7);
        let d1 = vsubq_s32(*r1, *r6);
        let d2 = vsubq_s32(*r2, *r5);
        let d3 = vsubq_s32(*r3, *r4);

        let y0 = vaddq_s32(
            vmulq_n_s32(vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3)), C4),
            round,
        );
        let y1 = vaddq_s32(
            vaddq_s32(
                vaddq_s32(vmulq_n_s32(d0, C1), vmulq_n_s32(d1, C3)),
                vaddq_s32(vmulq_n_s32(d2, C5), vmulq_n_s32(d3, C7)),
            ),
            round,
        );
        let y2 = vaddq_s32(
            vsubq_s32(
                vaddq_s32(vmulq_n_s32(s0, C2), vmulq_n_s32(s1, C6)),
                vaddq_s32(vmulq_n_s32(s2, C6), vmulq_n_s32(s3, C2)),
            ),
            round,
        );
        let y3 = vaddq_s32(
            vsubq_s32(
                vsubq_s32(vmulq_n_s32(d0, C3), vmulq_n_s32(d1, C7)),
                vaddq_s32(vmulq_n_s32(d2, C1), vmulq_n_s32(d3, C5)),
            ),
            round,
        );
        let y4 = vaddq_s32(
            vmulq_n_s32(vsubq_s32(vaddq_s32(s0, s3), vaddq_s32(s1, s2)), C4),
            round,
        );
        let y5 = vaddq_s32(
            vaddq_s32(
                vsubq_s32(vmulq_n_s32(d0, C5), vmulq_n_s32(d1, C1)),
                vaddq_s32(vmulq_n_s32(d2, C7), vmulq_n_s32(d3, C3)),
            ),
            round,
        );
        let y6 = vaddq_s32(
            vsubq_s32(
                vaddq_s32(vmulq_n_s32(s0, C6), vmulq_n_s32(s2, C2)),
                vaddq_s32(vmulq_n_s32(s1, C2), vmulq_n_s32(s3, C6)),
            ),
            round,
        );
        let y7 = vaddq_s32(
            vsubq_s32(
                vaddq_s32(vmulq_n_s32(d0, C7), vmulq_n_s32(d2, C3)),
                vaddq_s32(vmulq_n_s32(d1, C5), vmulq_n_s32(d3, C1)),
            ),
            round,
        );

        *r0 = vshlq_s32(y0, neg_shift);
        *r1 = vshlq_s32(y1, neg_shift);
        *r2 = vshlq_s32(y2, neg_shift);
        *r3 = vshlq_s32(y3, neg_shift);
        *r4 = vshlq_s32(y4, neg_shift);
        *r5 = vshlq_s32(y5, neg_shift);
        *r6 = vshlq_s32(y6, neg_shift);
        *r7 = vshlq_s32(y7, neg_shift);
    }

    #[inline]
    unsafe fn idct_1d_4cols(
        r0: &mut int32x4_t,
        r1: &mut int32x4_t,
        r2: &mut int32x4_t,
        r3: &mut int32x4_t,
        r4: &mut int32x4_t,
        r5: &mut int32x4_t,
        r6: &mut int32x4_t,
        r7: &mut int32x4_t,
        shift: i32,
    ) {
        let round = vdupq_n_s32(1i32 << (shift - 1));
        let neg_shift = vdupq_n_s32(-shift);

        let a0 = vmulq_n_s32(vaddq_s32(*r0, *r4), C4);
        let a1 = vmulq_n_s32(vsubq_s32(*r0, *r4), C4);
        let b0 = vaddq_s32(vmulq_n_s32(*r2, C2), vmulq_n_s32(*r6, C6));
        let b1 = vsubq_s32(vmulq_n_s32(*r2, C6), vmulq_n_s32(*r6, C2));

        let e0 = vaddq_s32(a0, b0);
        let e1 = vaddq_s32(a1, b1);
        let e2 = vsubq_s32(a1, b1);
        let e3 = vsubq_s32(a0, b0);

        let o0 = vaddq_s32(
            vaddq_s32(vmulq_n_s32(*r1, C1), vmulq_n_s32(*r3, C3)),
            vaddq_s32(vmulq_n_s32(*r5, C5), vmulq_n_s32(*r7, C7)),
        );
        let o1 = vsubq_s32(
            vsubq_s32(vmulq_n_s32(*r1, C3), vmulq_n_s32(*r3, C7)),
            vaddq_s32(vmulq_n_s32(*r5, C1), vmulq_n_s32(*r7, C5)),
        );
        let o2 = vaddq_s32(
            vsubq_s32(vmulq_n_s32(*r1, C5), vmulq_n_s32(*r3, C1)),
            vaddq_s32(vmulq_n_s32(*r5, C7), vmulq_n_s32(*r7, C3)),
        );
        let o3 = vsubq_s32(
            vaddq_s32(vmulq_n_s32(*r1, C7), vmulq_n_s32(*r5, C3)),
            vaddq_s32(vmulq_n_s32(*r3, C5), vmulq_n_s32(*r7, C1)),
        );

        *r0 = vshlq_s32(vaddq_s32(vaddq_s32(e0, o0), round), neg_shift);
        *r1 = vshlq_s32(vaddq_s32(vaddq_s32(e1, o1), round), neg_shift);
        *r2 = vshlq_s32(vaddq_s32(vaddq_s32(e2, o2), round), neg_shift);
        *r3 = vshlq_s32(vaddq_s32(vaddq_s32(e3, o3), round), neg_shift);
        *r4 = vshlq_s32(vaddq_s32(vsubq_s32(e3, o3), round), neg_shift);
        *r5 = vshlq_s32(vaddq_s32(vsubq_s32(e2, o2), round), neg_shift);
        *r6 = vshlq_s32(vaddq_s32(vsubq_s32(e1, o1), round), neg_shift);
        *r7 = vshlq_s32(vaddq_s32(vsubq_s32(e0, o0), round), neg_shift);
    }

    pub unsafe fn dct_8x8(input: &[i32; 64]) -> [i32; 64] {
        let mut tmp = [0i32; 64];
        // Transpose 8x8 via two 4x4 blocks per half and run 1D row DCTs
        for half in 0..2 {
            let r_base = half * 4;
            let mut l0 = vld1q_s32(input.as_ptr().add(r_base * 8));
            let mut l1 = vld1q_s32(input.as_ptr().add((r_base + 1) * 8));
            let mut l2 = vld1q_s32(input.as_ptr().add((r_base + 2) * 8));
            let mut l3 = vld1q_s32(input.as_ptr().add((r_base + 3) * 8));
            let mut h0 = vld1q_s32(input.as_ptr().add(r_base * 8 + 4));
            let mut h1 = vld1q_s32(input.as_ptr().add((r_base + 1) * 8 + 4));
            let mut h2 = vld1q_s32(input.as_ptr().add((r_base + 2) * 8 + 4));
            let mut h3 = vld1q_s32(input.as_ptr().add((r_base + 3) * 8 + 4));
            transpose_4x4_s32(&mut l0, &mut l1, &mut l2, &mut l3);
            transpose_4x4_s32(&mut h0, &mut h1, &mut h2, &mut h3);
            dct_1d_4cols(
                &mut l0, &mut l1, &mut l2, &mut l3, &mut h0, &mut h1, &mut h2, &mut h3, 11,
            );
            transpose_4x4_s32(&mut l0, &mut l1, &mut l2, &mut l3);
            transpose_4x4_s32(&mut h0, &mut h1, &mut h2, &mut h3);
            vst1q_s32(tmp.as_mut_ptr().add(r_base * 8), l0);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 1) * 8), l1);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 2) * 8), l2);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 3) * 8), l3);
            vst1q_s32(tmp.as_mut_ptr().add(r_base * 8 + 4), h0);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 1) * 8 + 4), h1);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 2) * 8 + 4), h2);
            vst1q_s32(tmp.as_mut_ptr().add((r_base + 3) * 8 + 4), h3);
        }

        let mut out = [0i32; 64];
        for col_half in 0..2 {
            let c_off = col_half * 4;
            let mut r0 = vld1q_s32(tmp.as_ptr().add(c_off));
            let mut r1 = vld1q_s32(tmp.as_ptr().add(8 + c_off));
            let mut r2 = vld1q_s32(tmp.as_ptr().add(16 + c_off));
            let mut r3 = vld1q_s32(tmp.as_ptr().add(24 + c_off));
            let mut r4 = vld1q_s32(tmp.as_ptr().add(32 + c_off));
            let mut r5 = vld1q_s32(tmp.as_ptr().add(40 + c_off));
            let mut r6 = vld1q_s32(tmp.as_ptr().add(48 + c_off));
            let mut r7 = vld1q_s32(tmp.as_ptr().add(56 + c_off));
            dct_1d_4cols(
                &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 17,
            );
            vst1q_s32(out.as_mut_ptr().add(c_off), r0);
            vst1q_s32(out.as_mut_ptr().add(8 + c_off), r1);
            vst1q_s32(out.as_mut_ptr().add(16 + c_off), r2);
            vst1q_s32(out.as_mut_ptr().add(24 + c_off), r3);
            vst1q_s32(out.as_mut_ptr().add(32 + c_off), r4);
            vst1q_s32(out.as_mut_ptr().add(40 + c_off), r5);
            vst1q_s32(out.as_mut_ptr().add(48 + c_off), r6);
            vst1q_s32(out.as_mut_ptr().add(56 + c_off), r7);
        }
        out
    }

    pub unsafe fn idct_8x8(coeffs: &[i32; 64]) -> [i32; 64] {
        let mut tmp = [0i32; 64];
        for col_half in 0..2 {
            let c_off = col_half * 4;
            let mut r0 = vld1q_s32(coeffs.as_ptr().add(c_off));
            let mut r1 = vld1q_s32(coeffs.as_ptr().add(8 + c_off));
            let mut r2 = vld1q_s32(coeffs.as_ptr().add(16 + c_off));
            let mut r3 = vld1q_s32(coeffs.as_ptr().add(24 + c_off));
            let mut r4 = vld1q_s32(coeffs.as_ptr().add(32 + c_off));
            let mut r5 = vld1q_s32(coeffs.as_ptr().add(40 + c_off));
            let mut r6 = vld1q_s32(coeffs.as_ptr().add(48 + c_off));
            let mut r7 = vld1q_s32(coeffs.as_ptr().add(56 + c_off));
            idct_1d_4cols(
                &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 11,
            );
            vst1q_s32(tmp.as_mut_ptr().add(c_off), r0);
            vst1q_s32(tmp.as_mut_ptr().add(8 + c_off), r1);
            vst1q_s32(tmp.as_mut_ptr().add(16 + c_off), r2);
            vst1q_s32(tmp.as_mut_ptr().add(24 + c_off), r3);
            vst1q_s32(tmp.as_mut_ptr().add(32 + c_off), r4);
            vst1q_s32(tmp.as_mut_ptr().add(40 + c_off), r5);
            vst1q_s32(tmp.as_mut_ptr().add(48 + c_off), r6);
            vst1q_s32(tmp.as_mut_ptr().add(56 + c_off), r7);
        }

        let mut out = [0i32; 64];
        for half in 0..2 {
            let r_base = half * 4;
            let mut r0 = vld1q_s32(tmp.as_ptr().add(r_base * 8));
            let mut r1 = vld1q_s32(tmp.as_ptr().add((r_base + 1) * 8));
            let mut r2 = vld1q_s32(tmp.as_ptr().add((r_base + 2) * 8));
            let mut r3 = vld1q_s32(tmp.as_ptr().add((r_base + 3) * 8));
            let mut r4 = vld1q_s32(tmp.as_ptr().add(r_base * 8 + 4));
            let mut r5 = vld1q_s32(tmp.as_ptr().add((r_base + 1) * 8 + 4));
            let mut r6 = vld1q_s32(tmp.as_ptr().add((r_base + 2) * 8 + 4));
            let mut r7 = vld1q_s32(tmp.as_ptr().add((r_base + 3) * 8 + 4));
            transpose_4x4_s32(&mut r0, &mut r1, &mut r2, &mut r3);
            transpose_4x4_s32(&mut r4, &mut r5, &mut r6, &mut r7);
            idct_1d_4cols(
                &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, 17,
            );
            transpose_4x4_s32(&mut r0, &mut r1, &mut r2, &mut r3);
            transpose_4x4_s32(&mut r4, &mut r5, &mut r6, &mut r7);
            vst1q_s32(out.as_mut_ptr().add(r_base * 8), r0);
            vst1q_s32(out.as_mut_ptr().add((r_base + 1) * 8), r1);
            vst1q_s32(out.as_mut_ptr().add((r_base + 2) * 8), r2);
            vst1q_s32(out.as_mut_ptr().add((r_base + 3) * 8), r3);
            vst1q_s32(out.as_mut_ptr().add(r_base * 8 + 4), r4);
            vst1q_s32(out.as_mut_ptr().add((r_base + 1) * 8 + 4), r5);
            vst1q_s32(out.as_mut_ptr().add((r_base + 2) * 8 + 4), r6);
            vst1q_s32(out.as_mut_ptr().add((r_base + 3) * 8 + 4), r7);
        }
        out
    }

    #[inline]
    unsafe fn rgb_to_ycbcr_4px(
        b: int32x4_t,
        g: int32x4_t,
        r: int32x4_t,
    ) -> (int32x4_t, int32x4_t, int32x4_t) {
        let zero = vdupq_n_s32(0);
        let max255 = vdupq_n_s32(255);
        let c128 = vdupq_n_s32(128);

        let y_num = vaddq_s32(
            vaddq_s32(vmulq_n_s32(r, 77), vmulq_n_s32(g, 150)),
            vaddq_s32(vmulq_n_s32(b, 29), c128),
        );
        let y = vminq_s32(vmaxq_s32(vshrq_n_s32(y_num, 8), zero), max255);

        let cb_num = vaddq_s32(
            vaddq_s32(vmulq_n_s32(r, -43), vmulq_n_s32(g, -85)),
            vaddq_s32(vshlq_n_s32(b, 7), c128),
        );
        let cb = vminq_s32(
            vmaxq_s32(vaddq_s32(vshrq_n_s32(cb_num, 8), c128), zero),
            max255,
        );

        let cr_num = vaddq_s32(
            vaddq_s32(vshlq_n_s32(r, 7), vmulq_n_s32(g, -107)),
            vaddq_s32(vmulq_n_s32(b, -21), c128),
        );
        let cr = vminq_s32(
            vmaxq_s32(vaddq_s32(vshrq_n_s32(cr_num, 8), c128), zero),
            max255,
        );

        (y, cb, cr)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bgra_tile_to_ycbcr420(
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
        if tw < 64 {
            bgra_tile_to_ycbcr420_scalar(bgra, fb_w, x0, y0, tw, th, y_plane, cb_plane, cr_plane);
            return;
        }
        let stride = fb_w as usize * 4;
        let mask_ff = vdupq_n_u32(0xFF);
        let two = vdupq_n_s32(2);
        let mut cb_full = [0i32; 64 * 64];
        let mut cr_full = [0i32; 64 * 64];

        for py in 0..64usize {
            let sy = (y0 as usize) + py.min(th as usize - 1);
            let row_ptr = bgra.as_ptr().add(sy * stride + (x0 as usize) * 4);
            let out_row = py * 64;
            for chunk in 0..16usize {
                let px = chunk * 4;
                let pix = vld1q_u32(row_ptr.add(px * 4) as *const u32);
                let b = vreinterpretq_s32_u32(vandq_u32(pix, mask_ff));
                let g = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(pix, 8), mask_ff));
                let r = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(pix, 16), mask_ff));
                let (y, cb, cr) = rgb_to_ycbcr_4px(b, g, r);
                let dst_off = out_row + px;
                vst1q_s32(y_plane.as_mut_ptr().add(dst_off), y);
                vst1q_s32(cb_full.as_mut_ptr().add(dst_off), cb);
                vst1q_s32(cr_full.as_mut_ptr().add(dst_off), cr);
            }
        }

        for cy in 0..32usize {
            let r0 = (cy * 2) * 64;
            let r1 = r0 + 64;
            let out_base = cy * 32;
            for chunk in 0..8usize {
                let cx = chunk * 4;
                let src_off0 = r0 + cx * 2;
                let src_off1 = r1 + cx * 2;

                let cb_0a = vld1q_s32(cb_full.as_ptr().add(src_off0));
                let cb_0b = vld1q_s32(cb_full.as_ptr().add(src_off0 + 4));
                let cb_1a = vld1q_s32(cb_full.as_ptr().add(src_off1));
                let cb_1b = vld1q_s32(cb_full.as_ptr().add(src_off1 + 4));
                let cb_sum = vpaddq_s32(vaddq_s32(cb_0a, cb_1a), vaddq_s32(cb_0b, cb_1b));
                vst1q_s32(
                    cb_plane.as_mut_ptr().add(out_base + cx),
                    vshrq_n_s32(vaddq_s32(cb_sum, two), 2),
                );

                let cr_0a = vld1q_s32(cr_full.as_ptr().add(src_off0));
                let cr_0b = vld1q_s32(cr_full.as_ptr().add(src_off0 + 4));
                let cr_1a = vld1q_s32(cr_full.as_ptr().add(src_off1));
                let cr_1b = vld1q_s32(cr_full.as_ptr().add(src_off1 + 4));
                let cr_sum = vpaddq_s32(vaddq_s32(cr_0a, cr_1a), vaddq_s32(cr_0b, cr_1b));
                vst1q_s32(
                    cr_plane.as_mut_ptr().add(out_base + cx),
                    vshrq_n_s32(vaddq_s32(cr_sum, two), 2),
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ycbcr420_to_bgra_tile(
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
        if tw < 64 {
            ycbcr420_to_bgra_tile_scalar(
                y_plane, cb_plane, cr_plane, a_plane, bgra, fb_w, x0, y0, tw, th,
            );
            return;
        }
        let stride = fb_w as usize * 4;
        let zero = vdupq_n_s32(0);
        let max255 = vdupq_n_s32(255);
        let c128 = vdupq_n_s32(128);
        let alpha_opaque = vdupq_n_u32(0xFF00_0000);

        for py in 0..(th as usize) {
            let dst_row = bgra
                .as_mut_ptr()
                .add((y0 as usize + py) * stride + (x0 as usize) * 4);
            let y_row = y_plane.as_ptr().add(py * 64);
            let c_row = (py >> 1) * 32;
            for chunk in 0..16usize {
                let px = chunk * 4;
                let y = vld1q_s32(y_row.add(px));
                let c_idx = c_row + (px >> 1);
                let cb_2 = vld1_s32(cb_plane.as_ptr().add(c_idx));
                let cr_2 = vld1_s32(cr_plane.as_ptr().add(c_idx));
                let cb_z = vzip_s32(cb_2, cb_2);
                let cr_z = vzip_s32(cr_2, cr_2);
                let cb = vcombine_s32(cb_z.0, cb_z.1);
                let cr = vcombine_s32(cr_z.0, cr_z.1);

                let c = vsubq_s32(cb, c128);
                let d = vsubq_s32(cr, c128);

                let r_raw = vaddq_s32(y, vshrq_n_s32(vaddq_s32(vmulq_n_s32(d, 359), c128), 8));
                let g_raw = vsubq_s32(
                    y,
                    vshrq_n_s32(
                        vaddq_s32(vaddq_s32(vmulq_n_s32(c, 88), vmulq_n_s32(d, 183)), c128),
                        8,
                    ),
                );
                let b_raw = vaddq_s32(y, vshrq_n_s32(vaddq_s32(vmulq_n_s32(c, 454), c128), 8));

                let mut r = vminq_s32(vmaxq_s32(r_raw, zero), max255);
                let mut g = vminq_s32(vmaxq_s32(g_raw, zero), max255);
                let mut b = vminq_s32(vmaxq_s32(b_raw, zero), max255);

                let a_shifted = match a_plane {
                    Some(ap) => {
                        let a_load = vld1q_s32(ap.as_ptr().add(py * 64 + px));
                        let av = vminq_s32(vmaxq_s32(a_load, zero), max255);
                        r = vminq_s32(r, av);
                        g = vminq_s32(g, av);
                        b = vminq_s32(b, av);
                        vshlq_n_u32(vreinterpretq_u32_s32(av), 24)
                    }
                    None => alpha_opaque,
                };

                let bgra_px = vorrq_u32(
                    vorrq_u32(
                        vreinterpretq_u32_s32(b),
                        vshlq_n_u32(vreinterpretq_u32_s32(g), 8),
                    ),
                    vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(r), 16), a_shifted),
                );
                vst1q_u32(dst_row.add(px * 4) as *mut u32, bgra_px);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile_equals_shifted(
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
        let chunks16 = row_bytes / 16;
        for py in 0..(th as usize) {
            let s_off = (sy as usize + py) * stride + (sx as usize) * 4;
            let d_off = (dy as usize + py) * stride + (dx as usize) * 4;
            let sp = src_fb.as_ptr().add(s_off);
            let dp = dst_fb.as_ptr().add(d_off);
            for c in 0..chunks16 {
                let va = vld1q_u8(sp.add(c * 16));
                let vb = vld1q_u8(dp.add(c * 16));
                let diff = veorq_u8(va, vb);
                if vmaxvq_u8(diff) != 0 {
                    return false;
                }
            }
            let rem_start = chunks16 * 16;
            if rem_start < row_bytes
                && src_fb[s_off + rem_start..s_off + row_bytes]
                    != dst_fb[d_off + rem_start..d_off + row_bytes]
            {
                return false;
            }
        }
        true
    }

    pub unsafe fn tile_is_opaque(
        bgra: &[u8],
        fb_w: u32,
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> bool {
        let stride = fb_w as usize * 4;
        let pixels = tw as usize;
        let chunks4 = pixels / 4;
        let alpha_mask = vdupq_n_u32(0xFF00_0000);
        for py in 0..(th as usize) {
            let row_base = (y0 as usize + py) * stride + (x0 as usize) * 4;
            let rp = bgra.as_ptr().add(row_base);
            for c in 0..chunks4 {
                let v = vld1q_u32(rp.add(c * 16) as *const u32);
                let masked = vandq_u32(v, alpha_mask);
                if vminvq_u32(masked) != 0xFF00_0000 {
                    return false;
                }
            }
            for px in (chunks4 * 4)..pixels {
                if bgra[row_base + px * 4 + 3] != 255 {
                    return false;
                }
            }
        }
        true
    }

    pub unsafe fn count_identical_pixels(
        ref_bgra: &[u8],
        cur_bgra: &[u8],
        fb_w: u32,
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> usize {
        let stride = fb_w as usize * 4;
        let pixels = tw as usize;
        let chunks4 = pixels / 4;
        let mut same = 0usize;
        for py in 0..(th as usize) {
            let off = (y0 as usize + py) * stride + (x0 as usize) * 4;
            let rp = ref_bgra.as_ptr().add(off);
            let cp = cur_bgra.as_ptr().add(off);
            for c in 0..chunks4 {
                let vr = vld1q_u32(rp.add(c * 16) as *const u32);
                let vc = vld1q_u32(cp.add(c * 16) as *const u32);
                let eq = vceqq_u32(vr, vc);
                let ones = vshrq_n_u32(eq, 31);
                same += vaddvq_u32(ones) as usize;
            }
            for px in (chunks4 * 4)..pixels {
                let p = off + px * 4;
                if ref_bgra[p..p + 4] == cur_bgra[p..p + 4] {
                    same += 1;
                }
            }
        }
        same
    }

    pub unsafe fn xor_encode_row(ref_row: &mut [u8], cur_row: &[u8], out: &mut Vec<u8>) {
        let len = cur_row.len();
        let start = out.len();
        out.resize(start + len, 0);
        let dst = &mut out[start..start + len];
        let chunks16 = len / 16;
        let rp = ref_row.as_mut_ptr();
        let cp = cur_row.as_ptr();
        let dp = dst.as_mut_ptr();
        for c in 0..chunks16 {
            let off = c * 16;
            let vr = vld1q_u8(rp.add(off));
            let vc = vld1q_u8(cp.add(off));
            vst1q_u8(dp.add(off), veorq_u8(vc, vr));
            vst1q_u8(rp.add(off), vc);
        }
        for i in (chunks16 * 16)..len {
            let c = cur_row[i];
            dst[i] = c ^ ref_row[i];
            ref_row[i] = c;
        }
    }

    pub unsafe fn xor_decode_row(dst_row: &mut [u8], src_row: &[u8]) {
        let len = dst_row.len();
        let chunks16 = len / 16;
        let dp = dst_row.as_mut_ptr();
        let sp = src_row.as_ptr();
        for c in 0..chunks16 {
            let off = c * 16;
            let vd = vld1q_u8(dp.add(off));
            let vs = vld1q_u8(sp.add(off));
            vst1q_u8(dp.add(off), veorq_u8(vd, vs));
        }
        for i in (chunks16 * 16)..len {
            dst_row[i] ^= src_row[i];
        }
    }
}
