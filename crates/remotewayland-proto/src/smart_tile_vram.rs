//! Zero-Copy VRAM-Resident `SmartTile` Encoder & Vulkan `DMA-BUF` Interop.
//!
//! Keeps per-window `cur_bgra`, `ref_bgra`, and `prev_ref_bgra` permanently in GPU VRAM so
//! hardware-accelerated client `Dmabuf` windows are ingested, scanned, motion-estimated
//! (`1/2-pel` diamond search + scroll detection), DCT/Quant/IDCT/Deblock/SAO-encoded across
//! Full-Res, Half-Res, and Quarter-Res modes, and progressively refined on idle without
//! ever copying full window framebuffers to CPU memory.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
#[cfg(unix)]
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::gpu::SmartTileGpuEngine;
use super::{CopyRectOp, MODE_FLAG_LOCAL_MV, MODE_FLAG_MV, TILE_SIZE};

pub const SCAN_FLAG_DIRTY: u32 = 1 << 0;
pub const SCAN_FLAG_OPAQUE: u32 = 1 << 1;
pub const SCAN_FLAG_SOLID: u32 = 1 << 2;
/// Tile differs from the previous frame's ingested source (`cur_bgra` before this ingest).
/// Unlike `SCAN_FLAG_DIRTY` (vs. the possibly-lossy reference), this ignores DCT error.
pub const SCAN_FLAG_SRC_CHANGED: u32 = 1 << 3;

pub const FAST_UI_MAX_TILES: usize = 8;

#[inline]
pub const fn qblocks_words_per_tile(scale: u32) -> usize {
    match scale {
        4 => 193,
        2 => 769,
        _ => 3073,
    }
}

pub const CMD_FLAG_WANTS_LOSSY: u32 = 1 << 0;
pub const CMD_FLAG_OPAQUE: u32 = 1 << 1;
pub const CMD_FLAG_SOLID: u32 = 1 << 2;

pub const MV_RESULT_SKIP: u32 = 1 << 31;

const SMART_TILE_VRAM_WGSL: &str = r#"
const C1: i32 = 8034;
const C2: i32 = 7568;
const C3: i32 = 6811;
const C4: i32 = 5793;
const C5: i32 = 4551;
const C6: i32 = 3135;
const C7: i32 = 1598;

fn rgb_to_y(r: i32, g: i32, b: i32) -> i32 {
    return clamp((77 * r + 150 * g + 29 * b + 128) >> 8u, 0, 255);
}

fn rgb_to_cb(r: i32, g: i32, b: i32) -> i32 {
    return clamp((((-43 * r - 85 * g + 128 * b + 128) >> 8u) + 128), 0, 255);
}

fn rgb_to_cr(r: i32, g: i32, b: i32) -> i32 {
    return clamp((((128 * r - 107 * g - 21 * b + 128) >> 8u) + 128), 0, 255);
}

fn ycbcr_to_bgra(y: i32, cb: i32, cr: i32, a_in: i32) -> u32 {
    let c = cb - 128;
    let d = cr - 128;
    let av = clamp(a_in, 0, 255);
    let r = min(clamp(y + ((359 * d + 128) >> 8u), 0, 255), av);
    let g = min(clamp(y - ((88 * c + 183 * d + 128) >> 8u), 0, 255), av);
    let b = min(clamp(y + ((454 * c + 128) >> 8u), 0, 255), av);
    return u32(b) | (u32(g) << 8u) | (u32(r) << 16u) | (u32(av) << 24u);
}

fn quantize_div(val: i32, q: i32) -> i32 {
    let half = q >> 1u;
    if (val >= 0) {
        return (val + half) / q;
    } else {
        return -((-val + half) / q);
    }
}

fn pack_i16x2(v0: i32, v1: i32) -> i32 {
    return bitcast<i32>((u32(v0) & 0xFFFFu) | ((u32(v1) & 0xFFFFu) << 16u));
}

fn unpack_i16_lo(packed: i32) -> i32 {
    return (packed << 16u) >> 16u;
}

fn unpack_i16_hi(packed: i32) -> i32 {
    return packed >> 16u;
}

// ---------------------------------------------------------------------------
// Kernel 1: Ingest `texture_2d<f32>` (cropped) -> `cur_bgra` + Dirty/Opaque/Solid Scan
// ---------------------------------------------------------------------------

struct IngestParams {
    fb_w: u32,
    fb_h: u32,
    cols: u32,
    rows: u32,
    crop_x: u32,
    crop_y: u32,
    rb_swizzle: u32,
    is_keyframe: u32,
}

struct ScanOutBuffer {
    dirty_count: atomic<u32>,
    fast_count: atomic<u32>,
    tiles: array<u32>,
}

@group(0) @binding(0) var<uniform> ingest_params: IngestParams;
@group(0) @binding(1) var ingest_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> ingest_ref_bgra: array<u32>;
@group(0) @binding(3) var<storage, read_write> ingest_cur_bgra: array<u32>;
@group(0) @binding(4) var<storage, read_write> ingest_scan_out: ScanOutBuffer;
@group(0) @binding(5) var<storage, read_write> ingest_fast_extract: array<u32>;

var<workgroup> wg_scan_dirty: atomic<u32>;
var<workgroup> wg_scan_src_changed: atomic<u32>;
var<workgroup> wg_scan_non_opaque: atomic<u32>;
var<workgroup> wg_scan_non_solid: atomic<u32>;
var<workgroup> wg_top_left_px: u32;
var<workgroup> wg_fast_slot: u32;

fn load_tex_bgra(x: u32, y: u32, crop_x: u32, crop_y: u32, rb_swizzle: u32) -> u32 {
    let c = textureLoad(ingest_tex, vec2<i32>(i32(crop_x + x), i32(crop_y + y)), 0);
    let r = u32(clamp(c.x * 255.0 + 0.5, 0.0, 255.0));
    let g = u32(clamp(c.y * 255.0 + 0.5, 0.0, 255.0));
    let b = u32(clamp(c.z * 255.0 + 0.5, 0.0, 255.0));
    let a = u32(clamp(c.w * 255.0 + 0.5, 0.0, 255.0));
    if (rb_swizzle != 0u) {
        return r | (g << 8u) | (b << 16u) | (a << 24u);
    }
    return b | (g << 8u) | (r << 16u) | (a << 24u);
}

@compute @workgroup_size(64, 1, 1)
fn ingest_and_scan_vram(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tidx = wid.x;
    let total_tiles = ingest_params.cols * ingest_params.rows;
    if (tidx >= total_tiles) {
        return;
    }
    let tx = tidx % ingest_params.cols;
    let ty = tidx / ingest_params.cols;
    let x0 = tx * 64u;
    let y0 = ty * 64u;
    let tw = min(ingest_params.fb_w - x0, 64u);
    let th = min(ingest_params.fb_h - y0, 64u);

    if (lid.x == 0u) {
        atomicStore(&wg_scan_dirty, 0u);
        atomicStore(&wg_scan_src_changed, 0u);
        atomicStore(&wg_scan_non_opaque, 0u);
        atomicStore(&wg_scan_non_solid, 0u);
        wg_top_left_px = load_tex_bgra(
            x0,
            y0,
            ingest_params.crop_x,
            ingest_params.crop_y,
            ingest_params.rb_swizzle,
        );
    }
    workgroupBarrier();

    let solid_ref = wg_top_left_px;
    let py = lid.x;
    let force_dirty = (ingest_params.is_keyframe & 1u) != 0u;
    if (py < th) {
        let row_base = (y0 + py) * ingest_params.fb_w + x0;
        var row_dirty = select(0u, 1u, force_dirty);
        var row_src_changed = select(0u, 1u, force_dirty);
        var row_non_opaque = 0u;
        var row_non_solid = 0u;
        for (var px = 0u; px < tw; px = px + 1u) {
            let cur_px = load_tex_bgra(
                x0 + px,
                y0 + py,
                ingest_params.crop_x,
                ingest_params.crop_y,
                ingest_params.rb_swizzle,
            );
            let idx = row_base + px;
            if (row_src_changed == 0u && cur_px != ingest_cur_bgra[idx]) {
                row_src_changed = 1u;
            }
            ingest_cur_bgra[idx] = cur_px;
            if (row_dirty == 0u && cur_px != ingest_ref_bgra[idx]) {
                row_dirty = 1u;
            }
            if ((cur_px & 0xFF000000u) != 0xFF000000u) {
                row_non_opaque = 1u;
            }
            if (cur_px != solid_ref) {
                row_non_solid = 1u;
            }
        }
        if (row_dirty != 0u) {
            atomicOr(&wg_scan_dirty, 1u);
        }
        if (row_src_changed != 0u) {
            atomicOr(&wg_scan_src_changed, 1u);
        }
        if (row_non_opaque != 0u) {
            atomicOr(&wg_scan_non_opaque, 1u);
        }
        if (row_non_solid != 0u) {
            atomicOr(&wg_scan_non_solid, 1u);
        }
    }
    workgroupBarrier();

    if (lid.x == 0u) {
        let dirty = atomicLoad(&wg_scan_dirty);
        let opaque = select(2u, 0u, atomicLoad(&wg_scan_non_opaque) != 0u);
        let solid = select(4u, 0u, atomicLoad(&wg_scan_non_solid) != 0u);
        let src_changed = select(0u, 8u, atomicLoad(&wg_scan_src_changed) != 0u);
        var slot = 0xFFFFu;
        if (dirty != 0u) {
            atomicAdd(&ingest_scan_out.dirty_count, 1u);
            if ((ingest_params.is_keyframe & 2u) != 0u && solid == 0u) {
                slot = atomicAdd(&ingest_scan_out.fast_count, 1u);
            }
        }
        wg_fast_slot = slot;
        let slot_enc = select(0u, (slot + 1u) << 8u, slot < 8u);
        ingest_scan_out.tiles[tidx * 2u] = dirty | opaque | solid | src_changed | slot_enc;
        ingest_scan_out.tiles[tidx * 2u + 1u] = solid_ref;
    }
    workgroupBarrier();

    if (wg_fast_slot < 8u) {
        let out_base = wg_fast_slot * 8192u;
        for (var i = py; i < 4096u; i = i + 64u) {
            let r = i >> 6u;
            let c = i & 63u;
            if (r < th && c < tw) {
                let fb_idx = (y0 + r) * ingest_params.fb_w + (x0 + c);
                ingest_fast_extract[out_base + i] = ingest_cur_bgra[fb_idx];
                ingest_fast_extract[out_base + 4096u + i] = ingest_ref_bgra[fb_idx];
            } else {
                ingest_fast_extract[out_base + i] = 0u;
                ingest_fast_extract[out_base + 4096u + i] = 0u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Row & Column Scroll Hash + 4-Segment Luma Profile Summaries
// Outputs `(fb_h + fb_w) * 4` u32s: `[prev_hash, cur_hash, prev_prof, cur_prof]`.
// ---------------------------------------------------------------------------

struct ScrollParams {
    fb_w: u32,
    fb_h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> scroll_params: ScrollParams;
@group(0) @binding(1) var<storage, read> scroll_ref_bgra: array<u32>;
@group(0) @binding(2) var<storage, read> scroll_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read_write> scroll_out: array<u32>;
@group(0) @binding(4) var<storage, read> scroll_scan_meta: array<u32>;

fn px_luma_fast(px: u32) -> u32 {
    let b = px & 0xFFu;
    let g = (px >> 8u) & 0xFFu;
    let r = (px >> 16u) & 0xFFu;
    return (r + 2u * g + b) >> 2u;
}

@compute @workgroup_size(64, 1, 1)
fn compute_scroll_summaries(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if (scroll_scan_meta[0] < 12u) {
        return;
    }
    let idx = gid.x;
    let w = scroll_params.fb_w;
    let h = scroll_params.fb_h;

    if (idx < h) {
        let y = idx;
        let x_start = w >> 2u;
        let x_end = (w * 3u) >> 2u;
        let span = max(x_end - x_start, 1u);
        let h_step = max(span >> 6u, 1u);
        let row_base = y * w;

        var ph: u32 = 2166136261u;
        var ch: u32 = 2166136261u;
        let p_first = scroll_ref_bgra[row_base + x_start] & 0x00FFFFFFu;
        let c_first = scroll_cur_bgra[row_base + x_start] & 0x00FFFFFFu;
        var p_non_unif = 0u;
        var c_non_unif = 0u;
        for (var x = x_start; x < x_end; x = x + h_step) {
            let pv = scroll_ref_bgra[row_base + x] & 0x00FFFFFFu;
            let cv = scroll_cur_bgra[row_base + x] & 0x00FFFFFFu;
            if (pv != p_first) { p_non_unif = 1u; }
            if (cv != c_first) { c_non_unif = 1u; }
            ph = (ph ^ pv) * 16777619u;
            ch = (ch ^ cv) * 16777619u;
        }

        let seg_w = max(span >> 2u, 1u);
        let p_step = max(seg_w >> 4u, 1u);
        var p_prof: u32 = 0u;
        var c_prof: u32 = 0u;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let sx0 = x_start + s * seg_w;
            let sx1 = min(sx0 + seg_w, x_end);
            var p_sum = 0u;
            var c_sum = 0u;
            var cnt = 0u;
            for (var x = sx0; x < sx1; x = x + p_step) {
                p_sum = p_sum + px_luma_fast(scroll_ref_bgra[row_base + x]);
                c_sum = c_sum + px_luma_fast(scroll_cur_bgra[row_base + x]);
                cnt = cnt + 1u;
            }
            let pm = select(0u, p_sum / cnt, cnt > 0u) & 0xFFu;
            let cm = select(0u, c_sum / cnt, cnt > 0u) & 0xFFu;
            p_prof = p_prof | (pm << (s * 8u));
            c_prof = c_prof | (cm << (s * 8u));
        }

        let out_base = y * 4u;
        scroll_out[out_base] = (ph & 0xFFFFFFFEu) | p_non_unif;
        scroll_out[out_base + 1u] = (ch & 0xFFFFFFFEu) | c_non_unif;
        scroll_out[out_base + 2u] = p_prof;
        scroll_out[out_base + 3u] = c_prof;
    }

    if (idx < w) {
        let x = idx;
        let y_start = h >> 2u;
        let y_end = (h * 3u) >> 2u;
        let span = max(y_end - y_start, 1u);
        let h_step = max(span >> 6u, 1u);

        var ph: u32 = 2166136261u;
        var ch: u32 = 2166136261u;
        let p_first = scroll_ref_bgra[y_start * w + x] & 0x00FFFFFFu;
        let c_first = scroll_cur_bgra[y_start * w + x] & 0x00FFFFFFu;
        var p_non_unif = 0u;
        var c_non_unif = 0u;
        for (var y = y_start; y < y_end; y = y + h_step) {
            let pv = scroll_ref_bgra[y * w + x] & 0x00FFFFFFu;
            let cv = scroll_cur_bgra[y * w + x] & 0x00FFFFFFu;
            if (pv != p_first) { p_non_unif = 1u; }
            if (cv != c_first) { c_non_unif = 1u; }
            ph = (ph ^ pv) * 16777619u;
            ch = (ch ^ cv) * 16777619u;
        }

        let seg_h = max(span >> 2u, 1u);
        let p_step = max(seg_h >> 4u, 1u);
        var p_prof: u32 = 0u;
        var c_prof: u32 = 0u;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let sy0 = y_start + s * seg_h;
            let sy1 = min(sy0 + seg_h, y_end);
            var p_sum = 0u;
            var c_sum = 0u;
            var cnt = 0u;
            for (var y = sy0; y < sy1; y = y + p_step) {
                p_sum = p_sum + px_luma_fast(scroll_ref_bgra[y * w + x]);
                c_sum = c_sum + px_luma_fast(scroll_cur_bgra[y * w + x]);
                cnt = cnt + 1u;
            }
            let pm = select(0u, p_sum / cnt, cnt > 0u) & 0xFFu;
            let cm = select(0u, c_sum / cnt, cnt > 0u) & 0xFFu;
            p_prof = p_prof | (pm << (s * 8u));
            c_prof = c_prof | (cm << (s * 8u));
        }

        let out_base = (h + x) * 4u;
        scroll_out[out_base] = (ph & 0xFFFFFFFEu) | p_non_unif;
        scroll_out[out_base + 1u] = (ch & 0xFFFFFFFEu) | c_non_unif;
        scroll_out[out_base + 2u] = p_prof;
        scroll_out[out_base + 3u] = c_prof;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Per-Tile MV Skip, 1/2-Pel 2D Motion Search, & Sparse Tile Extract
// ---------------------------------------------------------------------------

struct VramPrepParams {
    fb_w: u32,
    fb_h: u32,
    num_cmds: u32,
    scale: u32,
    frame_dx: i32,
    frame_dy: i32,
    has_frame_mv: u32,
    enable_local_mv: u32,
}

struct VramTileCmd {
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    flags: u32,
    batch_idx: u32,
    extract_idx: u32,
    _pad: u32,
}

struct VramTileDesc {
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    pred_x: u32,
    pred_y: u32,
    pred_from_prev: u32,
    use_inter: u32,
}

@group(0) @binding(0) var<uniform> prep_params: VramPrepParams;
@group(0) @binding(1) var<storage, read> prep_cmds: array<VramTileCmd>;
@group(0) @binding(2) var<storage, read> prep_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read> prep_prev_ref_bgra: array<u32>;
@group(0) @binding(4) var<storage, read_write> prep_ref_bgra: array<u32>;
@group(0) @binding(5) var<storage, read_write> prep_descs: array<VramTileDesc>;
@group(0) @binding(6) var<storage, read_write> prep_mv_results: array<u32>;
@group(0) @binding(7) var<storage, read_write> prep_extract_out: array<u32>;

var<workgroup> wg_mv_diff: atomic<u32>;
var<workgroup> wg_sad_acc: array<u32, 64>;
var<workgroup> wg_sad_acc2: array<u32, 64>;
var<workgroup> wg_is_skip: u32;

fn sample_hpel_sad_lane(
    sx_h: u32,
    sy_h: u32,
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    fb_w: u32,
    lane: u32,
) -> u32 {
    let max_x = tw - 1u;
    let max_y = th - 1u;
    let py = min(4u, max_y) + (lane >> 3u) * 8u;
    let px = min(4u, max_x) + (lane & 7u) * 8u;
    if (py >= th || px >= tw) {
        return 0u;
    }
    let sx = sx_h >> 1u;
    let sy = sy_h >> 1u;
    let hx = sx_h & 1u;
    let hy = sy_h & 1u;

    let dp = prep_cur_bgra[(y0 + py) * fb_w + (x0 + px)];
    let db = i32(dp & 0xFFu);
    let dg = i32((dp >> 8u) & 0xFFu);
    let dr = i32((dp >> 16u) & 0xFFu);

    let p00 = prep_prev_ref_bgra[(sy + py) * fb_w + (sx + px)];
    var sb = i32(p00 & 0xFFu);
    var sg = i32((p00 >> 8u) & 0xFFu);
    var sr = i32((p00 >> 16u) & 0xFFu);

    if (hx != 0u || hy != 0u) {
        let cx1 = min(px + hx, max_x);
        let cy1 = min(py + hy, max_y);
        if (hx == 1u && hy == 0u) {
            let p01 = prep_prev_ref_bgra[(sy + py) * fb_w + (sx + cx1)];
            sb = (sb + i32(p01 & 0xFFu) + 1) >> 1u;
            sg = (sg + i32((p01 >> 8u) & 0xFFu) + 1) >> 1u;
            sr = (sr + i32((p01 >> 16u) & 0xFFu) + 1) >> 1u;
        } else if (hx == 0u && hy == 1u) {
            let p10 = prep_prev_ref_bgra[(sy + cy1) * fb_w + (sx + px)];
            sb = (sb + i32(p10 & 0xFFu) + 1) >> 1u;
            sg = (sg + i32((p10 >> 8u) & 0xFFu) + 1) >> 1u;
            sr = (sr + i32((p10 >> 16u) & 0xFFu) + 1) >> 1u;
        } else {
            let p01 = prep_prev_ref_bgra[(sy + py) * fb_w + (sx + cx1)];
            let p10 = prep_prev_ref_bgra[(sy + cy1) * fb_w + (sx + px)];
            let p11 = prep_prev_ref_bgra[(sy + cy1) * fb_w + (sx + cx1)];
            sb = (sb + i32(p01 & 0xFFu) + i32(p10 & 0xFFu) + i32(p11 & 0xFFu) + 2) >> 2u;
            sg = (sg + i32((p01 >> 8u) & 0xFFu) + i32((p10 >> 8u) & 0xFFu) + i32((p11 >> 8u) & 0xFFu) + 2) >> 2u;
            sr = (sr + i32((p01 >> 16u) & 0xFFu) + i32((p10 >> 16u) & 0xFFu) + i32((p11 >> 16u) & 0xFFu) + 2) >> 2u;
        }
    }
    return u32(abs(sb - db) + abs(sg - dg) + abs(sr - dr));
}

@compute @workgroup_size(64, 1, 1)
fn prepare_vram_tile_mvs(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let cmd_idx = wid.x;
    if (cmd_idx >= prep_params.num_cmds) {
        return;
    }
    let cmd = prep_cmds[cmd_idx];
    let lane = lid.x;
    let fb_w = prep_params.fb_w;
    let fb_h = prep_params.fb_h;
    let wants_lossy = (cmd.flags & 1u) != 0u;
    let is_opaque = (cmd.flags & 2u) != 0u;
    let is_solid = (cmd.flags & 4u) != 0u;
    let is_gpu_lossy = wants_lossy && is_opaque;

    if (lane == 0u) {
        atomicStore(&wg_mv_diff, 0u);
        wg_is_skip = 0u;
    }
    workgroupBarrier();

    var mv_in_bounds = false;
    var mv_sx = 0u;
    var mv_sy = 0u;
    if (prep_params.has_frame_mv != 0u) {
        let sx_i = i32(cmd.x0) - prep_params.frame_dx;
        let sy_i = i32(cmd.y0) - prep_params.frame_dy;
        if (sx_i >= 0 && sy_i >= 0 && u32(sx_i) + cmd.tw <= fb_w && u32(sy_i) + cmd.th <= fb_h) {
            mv_in_bounds = true;
            mv_sx = u32(sx_i);
            mv_sy = u32(sy_i);
        }
    }

    if (mv_in_bounds) {
        if (lane < cmd.th) {
            let s_base = (mv_sy + lane) * fb_w + mv_sx;
            let d_base = (cmd.y0 + lane) * fb_w + cmd.x0;
            var diff = 0u;
            for (var px = 0u; px < cmd.tw; px = px + 1u) {
                if (prep_prev_ref_bgra[s_base + px] != prep_cur_bgra[d_base + px]) {
                    diff = 1u;
                    break;
                }
            }
            if (diff != 0u) {
                atomicOr(&wg_mv_diff, 1u);
            }
        }
    }
    workgroupBarrier();

    if (mv_in_bounds && atomicLoad(&wg_mv_diff) == 0u) {
        if (lane < cmd.th) {
            let s_base = (mv_sy + lane) * fb_w + mv_sx;
            let d_base = (cmd.y0 + lane) * fb_w + cmd.x0;
            for (var px = 0u; px < cmd.tw; px = px + 1u) {
                prep_ref_bgra[d_base + px] = prep_prev_ref_bgra[s_base + px];
            }
        }
        if (lane == 0u) {
            prep_mv_results[cmd_idx] = 0x80000000u;
            if (is_gpu_lossy) {
                prep_descs[cmd.batch_idx] = VramTileDesc(
                    cmd.x0, cmd.y0, cmd.tw, cmd.th,
                    cmd.x0 * 2u, cmd.y0 * 2u, 0u, 0xFFFFFFFFu,
                );
            }
            wg_is_skip = 1u;
        }
    }
    workgroupBarrier();
    if (wg_is_skip != 0u) {
        return;
    }

    var pred_x_h = cmd.x0 * 2u;
    var pred_y_h = cmd.y0 * 2u;
    var pred_from_prev = 0u;
    var mv_res = 0u;

    // Frame-scroll MV vs Zero-MV check for lossy opaque tiles
    if (is_gpu_lossy && mv_in_bounds) {
        var my_mv_sad = 0u;
        var my_zero_sad = 0u;
        if (lane < cmd.th && (lane & 1u) == 0u) {
            let s_row = (mv_sy + lane) * fb_w + mv_sx;
            let z_row = (cmd.y0 + lane) * fb_w + cmd.x0;
            for (var px = 0u; px < cmd.tw; px = px + 4u) {
                let cp = prep_cur_bgra[z_row + px];
                let sp = prep_prev_ref_bgra[s_row + px];
                let zp = prep_prev_ref_bgra[z_row + px];
                let cb = i32(cp & 0xFFu); let cg = i32((cp >> 8u) & 0xFFu); let cr = i32((cp >> 16u) & 0xFFu);
                my_mv_sad = my_mv_sad + u32(abs(i32(sp & 0xFFu) - cb) + abs(i32((sp >> 8u) & 0xFFu) - cg) + abs(i32((sp >> 16u) & 0xFFu) - cr));
                my_zero_sad = my_zero_sad + u32(abs(i32(zp & 0xFFu) - cb) + abs(i32((zp >> 8u) & 0xFFu) - cg) + abs(i32((zp >> 16u) & 0xFFu) - cr));
            }
        }
        wg_sad_acc[lane] = my_mv_sad;
        wg_sad_acc2[lane] = my_zero_sad;
    } else {
        wg_sad_acc[lane] = 0u;
        wg_sad_acc2[lane] = 0u;
    }
    workgroupBarrier();

    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            wg_sad_acc[lane] = wg_sad_acc[lane] + wg_sad_acc[lane + s];
            wg_sad_acc2[lane] = wg_sad_acc2[lane] + wg_sad_acc2[lane + s];
        }
        workgroupBarrier();
    }

    if (is_gpu_lossy && mv_in_bounds && wg_sad_acc[0] < wg_sad_acc2[0]) {
        pred_x_h = mv_sx * 2u;
        pred_y_h = mv_sy * 2u;
        pred_from_prev = 1u;
        mv_res = 64u; // MODE_FLAG_MV = 1 << 6
    }

    // Parallel 2D logarithmic diamond + 1/2-pel motion estimation
    let do_local_mv = is_gpu_lossy && (prep_params.has_frame_mv == 0u) && (prep_params.enable_local_mv != 0u);
    if (do_local_mv) {
        wg_sad_acc[lane] = sample_hpel_sad_lane(cmd.x0 * 2u, cmd.y0 * 2u, cmd.x0, cmd.y0, cmd.tw, cmd.th, fb_w, lane);
    }
    workgroupBarrier();
    if (do_local_mv) {
        for (var s = 32u; s > 0u; s = s >> 1u) {
            if (lane < s) {
                wg_sad_acc[lane] = wg_sad_acc[lane] + wg_sad_acc[lane + s];
            }
            workgroupBarrier();
        }
    }
    let zero_sad = wg_sad_acc[0];
    var best_sad = zero_sad;
    var best_int_dx: i32 = 0;
    var best_int_dy: i32 = 0;
    var best_dx_h: i32 = 0;
    var best_dy_h: i32 = 0;

    if (do_local_mv && zero_sad >= 384u) {
        let steps = array<i32, 4>(8, 4, 2, 1);
        let dirs_x = array<i32, 8>(-1, 1, 0, 0, -1, 1, -1, 1);
        let dirs_y = array<i32, 8>(0, 0, -1, 1, -1, -1, 1, 1);

        for (var st_idx = 0u; st_idx < 4u; st_idx = st_idx + 1u) {
            let step = steps[st_idx];
            let skip_step = (best_sad * 5u <= zero_sad) && (step >= 4);
            let cx = best_int_dx;
            let cy = best_int_dy;
            for (var d = 0u; d < 8u; d = d + 1u) {
                let ndx = cx + dirs_x[d] * step;
                let ndy = cy + dirs_y[d] * step;
                let dx_h = ndx * 2;
                let dy_h = ndy * 2;
                let sx_h = i32(cmd.x0) * 2 - dx_h;
                let sy_h = i32(cmd.y0) * 2 - dy_h;
                let valid = !skip_step
                    && abs(ndx) <= 24
                    && abs(ndy) <= 24
                    && sx_h >= 0
                    && sy_h >= 0
                    && (u32(sx_h) >> 1u) + cmd.tw <= fb_w
                    && (u32(sy_h) >> 1u) + cmd.th <= fb_h;
                if (valid) {
                    wg_sad_acc[lane] = sample_hpel_sad_lane(u32(sx_h), u32(sy_h), cmd.x0, cmd.y0, cmd.tw, cmd.th, fb_w, lane);
                } else {
                    wg_sad_acc[lane] = 0x0FFFFFFFu;
                }
                workgroupBarrier();
                for (var s = 32u; s > 0u; s = s >> 1u) {
                    if (lane < s) {
                        wg_sad_acc[lane] = wg_sad_acc[lane] + wg_sad_acc[lane + s];
                    }
                    workgroupBarrier();
                }
                if (valid && wg_sad_acc[0] < best_sad) {
                    best_sad = wg_sad_acc[0];
                    best_int_dx = ndx;
                    best_int_dy = ndy;
                    best_dx_h = dx_h;
                    best_dy_h = dy_h;
                }
            }
        }

        // Half-pel refinement when encoding at full resolution (`scale == 1`)
        if (prep_params.scale == 1u) {
            let cx_h = best_dx_h;
            let cy_h = best_dy_h;
            for (var d = 0u; d < 8u; d = d + 1u) {
                let nhx = cx_h + dirs_x[d];
                let nhy = cy_h + dirs_y[d];
                let sx_h = i32(cmd.x0) * 2 - nhx;
                let sy_h = i32(cmd.y0) * 2 - nhy;
                let valid = abs(nhx) <= 48
                    && abs(nhy) <= 48
                    && sx_h >= 0
                    && sy_h >= 0
                    && (u32(sx_h) >> 1u) + cmd.tw <= fb_w
                    && (u32(sy_h) >> 1u) + cmd.th <= fb_h;
                if (valid) {
                    wg_sad_acc[lane] = sample_hpel_sad_lane(u32(sx_h), u32(sy_h), cmd.x0, cmd.y0, cmd.tw, cmd.th, fb_w, lane);
                } else {
                    wg_sad_acc[lane] = 0x0FFFFFFFu;
                }
                workgroupBarrier();
                for (var s = 32u; s > 0u; s = s >> 1u) {
                    if (lane < s) {
                        wg_sad_acc[lane] = wg_sad_acc[lane] + wg_sad_acc[lane + s];
                    }
                    workgroupBarrier();
                }
                if (valid && wg_sad_acc[0] < best_sad) {
                    best_sad = wg_sad_acc[0];
                    best_dx_h = nhx;
                    best_dy_h = nhy;
                }
            }
        }

        if ((best_dx_h != 0 || best_dy_h != 0) && best_sad * 10u <= zero_sad * 8u) {
            pred_x_h = u32(i32(cmd.x0) * 2 - best_dx_h);
            pred_y_h = u32(i32(cmd.y0) * 2 - best_dy_h);
            pred_from_prev = 1u;
            let ldx_u8 = u32(best_dx_h) & 0xFFu;
            let ldy_u8 = u32(best_dy_h) & 0xFFu;
            mv_res = 32u | (ldx_u8 << 8u) | (ldy_u8 << 16u); // MODE_FLAG_LOCAL_MV = 1 << 5
        }
    }

    if (lane == 0u) {
        prep_mv_results[cmd_idx] = mv_res;
        if (is_gpu_lossy) {
            prep_descs[cmd.batch_idx] = VramTileDesc(
                cmd.x0, cmd.y0, cmd.tw, cmd.th,
                pred_x_h, pred_y_h, pred_from_prev, 1u,
            );
        }
    }

    // Solid lossless UI tiles update `ref_bgra` directly in VRAM with 0 PCIe readback!
    if (!wants_lossy && is_solid) {
        if (lane < cmd.th) {
            let base = (cmd.y0 + lane) * fb_w + cmd.x0;
            for (var px = 0u; px < cmd.tw; px = px + 1u) {
                prep_ref_bgra[base + px] = prep_cur_bgra[base + px];
            }
        }
    } else if (cmd.extract_idx != 0xFFFFFFFFu) {
        // Pack `(cur_tile, ref_tile)` into `prep_extract_out` (8192 u32s per extracted tile)
        let out_base = cmd.extract_idx * 8192u;
        for (var i = lane; i < 4096u; i = i + 64u) {
            let py = i >> 6u;
            let px = i & 63u;
            if (py < cmd.th && px < cmd.tw) {
                let fb_idx = (cmd.y0 + py) * fb_w + (cmd.x0 + px);
                let cp = prep_cur_bgra[fb_idx];
                let rp = prep_ref_bgra[fb_idx];
                prep_extract_out[out_base + i] = cp;
                prep_extract_out[out_base + 4096u + i] = rp;
                if (!wants_lossy) {
                    prep_ref_bgra[fb_idx] = cp;
                }
            } else {
                prep_extract_out[out_base + i] = 0u;
                prep_extract_out[out_base + 4096u + i] = 0u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Unified VRAM Tile Encoder (Full-Res 1x, Half-Res 2x, Quarter-Res 4x)
// Reconstructs BGRA directly into `vram_ref_bgra` in VRAM!
// ---------------------------------------------------------------------------

struct VramBatchParams {
    fb_w: u32,
    fb_h: u32,
    num_tiles: u32,
    scale: u32,
    luma_q: array<vec4<i32>, 16>,
    chroma_q: array<vec4<i32>, 16>,
}

@group(0) @binding(0) var<uniform> vtb_params: VramBatchParams;
@group(0) @binding(1) var<storage, read> vtb_descs: array<VramTileDesc>;
@group(0) @binding(2) var<storage, read> vtb_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read> vtb_prev_ref_bgra: array<u32>;
@group(0) @binding(4) var<storage, read_write> vtb_qblocks: array<i32>;
@group(0) @binding(5) var<storage, read_write> vtb_ref_bgra: array<u32>;

var<workgroup> vwg_y_packed: array<i32, 4096>;
var<workgroup> vwg_c_packed: array<u32, 1024>;
var<workgroup> vwg_inter_sum: array<i32, 64>;
var<workgroup> vwg_inter_sad: array<i32, 64>;
var<workgroup> vwg_intra_sad: array<i32, 64>;
var<workgroup> vwg_use_inter: u32;
var<workgroup> vwg_multi_blk: array<i32, 512>;
var<workgroup> vwg_multi_tmp: array<i32, 512>;
var<workgroup> vwg_multi_ref: array<i32, 512>;

fn vget_luma_q(idx: u32) -> i32 {
    let v = vtb_params.luma_q[idx >> 2u];
    let lane = idx & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn vget_chroma_q(idx: u32) -> i32 {
    let v = vtb_params.chroma_q[idx >> 2u];
    let lane = idx & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn read_pred_px(idx: u32, from_prev: u32) -> u32 {
    if (from_prev != 0u) {
        return vtb_prev_ref_bgra[idx];
    }
    return vtb_ref_bgra[idx];
}

fn vsample_pred_rgb_hpel(
    base_sx: u32,
    base_sy: u32,
    px: u32,
    py: u32,
    max_x: u32,
    max_y: u32,
    hx: u32,
    hy: u32,
    fb_w: u32,
    from_prev: u32,
) -> vec3<i32> {
    let cx0 = min(px, max_x);
    let cy0 = min(py, max_y);
    let p00 = read_pred_px((base_sy + cy0) * fb_w + (base_sx + cx0), from_prev);
    let b00 = i32(p00 & 0xFFu);
    let g00 = i32((p00 >> 8u) & 0xFFu);
    let r00 = i32((p00 >> 16u) & 0xFFu);
    if (hx == 0u && hy == 0u) {
        return vec3<i32>(r00, g00, b00);
    }
    let cx1 = min(cx0 + hx, max_x);
    let cy1 = min(cy0 + hy, max_y);
    if (hx == 1u && hy == 0u) {
        let p01 = read_pred_px((base_sy + cy0) * fb_w + (base_sx + cx1), from_prev);
        return vec3<i32>(
            (r00 + i32((p01 >> 16u) & 0xFFu) + 1) >> 1u,
            (g00 + i32((p01 >> 8u) & 0xFFu) + 1) >> 1u,
            (b00 + i32(p01 & 0xFFu) + 1) >> 1u,
        );
    }
    if (hx == 0u && hy == 1u) {
        let p10 = read_pred_px((base_sy + cy1) * fb_w + (base_sx + cx0), from_prev);
        return vec3<i32>(
            (r00 + i32((p10 >> 16u) & 0xFFu) + 1) >> 1u,
            (g00 + i32((p10 >> 8u) & 0xFFu) + 1) >> 1u,
            (b00 + i32(p10 & 0xFFu) + 1) >> 1u,
        );
    }
    let p01 = read_pred_px((base_sy + cy0) * fb_w + (base_sx + cx1), from_prev);
    let p10 = read_pred_px((base_sy + cy1) * fb_w + (base_sx + cx0), from_prev);
    let p11 = read_pred_px((base_sy + cy1) * fb_w + (base_sx + cx1), from_prev);
    return vec3<i32>(
        (r00 + i32((p01 >> 16u) & 0xFFu) + i32((p10 >> 16u) & 0xFFu) + i32((p11 >> 16u) & 0xFFu) + 2) >> 2u,
        (g00 + i32((p01 >> 8u) & 0xFFu) + i32((p10 >> 8u) & 0xFFu) + i32((p11 >> 8u) & 0xFFu) + 2) >> 2u,
        (b00 + i32(p01 & 0xFFu) + i32(p10 & 0xFFu) + i32(p11 & 0xFFu) + 2) >> 2u,
    );
}

@compute @workgroup_size(64, 1, 1)
fn encode_vram_tiles_batch(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tidx = wid.x;
    if (tidx >= vtb_params.num_tiles) {
        return;
    }
    let desc = vtb_descs[tidx];
    if (desc.use_inter == 0xFFFFFFFFu) {
        return;
    }
    let lane = lid.x;
    let fb_w = vtb_params.fb_w;
    let scale = vtb_params.scale;
    let max_x = desc.tw - 1u;
    let max_y = desc.th - 1u;
    let base_sx = desc.pred_x >> 1u;
    let base_sy = desc.pred_y >> 1u;
    let hx = desc.pred_x & 1u;
    let hy = desc.pred_y & 1u;

    var my_sum_y = 0;
    var my_inter_sad = 0;
    var sample_count = 1024;

    if (scale == 1u) {
        for (var qidx = lane; qidx < 1024u; qidx = qidx + 64u) {
            let cy = qidx >> 5u;
            let cx = qidx & 31u;
            let py0 = cy * 2u;
            let py1 = py0 + 1u;
            let px0 = cx * 2u;
            let px1 = px0 + 1u;

            let sy0 = desc.y0 + min(py0, max_y);
            let sy1 = desc.y0 + min(py1, max_y);
            let sx0 = desc.x0 + min(px0, max_x);
            let sx1 = desc.x0 + min(px1, max_x);

            let p00 = vtb_cur_bgra[sy0 * fb_w + sx0];
            let p01 = vtb_cur_bgra[sy0 * fb_w + sx1];
            let p10 = vtb_cur_bgra[sy1 * fb_w + sx0];
            let p11 = vtb_cur_bgra[sy1 * fb_w + sx1];

            let b00 = i32(p00 & 0xFFu); let g00 = i32((p00 >> 8u) & 0xFFu); let r00 = i32((p00 >> 16u) & 0xFFu);
            let b01 = i32(p01 & 0xFFu); let g01 = i32((p01 >> 8u) & 0xFFu); let r01 = i32((p01 >> 16u) & 0xFFu);
            let b10 = i32(p10 & 0xFFu); let g10 = i32((p10 >> 8u) & 0xFFu); let r10 = i32((p10 >> 16u) & 0xFFu);
            let b11 = i32(p11 & 0xFFu); let g11 = i32((p11 >> 8u) & 0xFFu); let r11 = i32((p11 >> 16u) & 0xFFu);

            let y00 = rgb_to_y(r00, g00, b00);
            let y01 = rgb_to_y(r01, g01, b01);
            let y10 = rgb_to_y(r10, g10, b10);
            let y11 = rgb_to_y(r11, g11, b11);

            let cb_cur = (rgb_to_cb(r00, g00, b00) + rgb_to_cb(r01, g01, b01) + rgb_to_cb(r10, g10, b10) + rgb_to_cb(r11, g11, b11) + 2) >> 2u;
            let cr_cur = (rgb_to_cr(r00, g00, b00) + rgb_to_cr(r01, g01, b01) + rgb_to_cr(r10, g10, b10) + rgb_to_cr(r11, g11, b11) + 2) >> 2u;

            let rgb00 = vsample_pred_rgb_hpel(base_sx, base_sy, px0, py0, max_x, max_y, hx, hy, fb_w, desc.pred_from_prev);
            let rgb01 = vsample_pred_rgb_hpel(base_sx, base_sy, px1, py0, max_x, max_y, hx, hy, fb_w, desc.pred_from_prev);
            let rgb10 = vsample_pred_rgb_hpel(base_sx, base_sy, px0, py1, max_x, max_y, hx, hy, fb_w, desc.pred_from_prev);
            let rgb11 = vsample_pred_rgb_hpel(base_sx, base_sy, px1, py1, max_x, max_y, hx, hy, fb_w, desc.pred_from_prev);

            let ry00 = rgb_to_y(rgb00.x, rgb00.y, rgb00.z);
            let ry01 = rgb_to_y(rgb01.x, rgb01.y, rgb01.z);
            let ry10 = rgb_to_y(rgb10.x, rgb10.y, rgb10.z);
            let ry11 = rgb_to_y(rgb11.x, rgb11.y, rgb11.z);

            vwg_y_packed[py0 * 64u + px0] = pack_i16x2(y00, ry00);
            vwg_y_packed[py0 * 64u + px1] = pack_i16x2(y01, ry01);
            vwg_y_packed[py1 * 64u + px0] = pack_i16x2(y10, ry10);
            vwg_y_packed[py1 * 64u + px1] = pack_i16x2(y11, ry11);

            let cb_ref = (rgb_to_cb(rgb00.x, rgb00.y, rgb00.z) + rgb_to_cb(rgb01.x, rgb01.y, rgb01.z) + rgb_to_cb(rgb10.x, rgb10.y, rgb10.z) + rgb_to_cb(rgb11.x, rgb11.y, rgb11.z) + 2) >> 2u;
            let cr_ref = (rgb_to_cr(rgb00.x, rgb00.y, rgb00.z) + rgb_to_cr(rgb01.x, rgb01.y, rgb01.z) + rgb_to_cr(rgb10.x, rgb10.y, rgb10.z) + rgb_to_cr(rgb11.x, rgb11.y, rgb11.z) + 2) >> 2u;
            vwg_c_packed[qidx] = u32(cb_cur) | (u32(cr_cur) << 8u) | (u32(cb_ref) << 16u) | (u32(cr_ref) << 24u);

            if ((cx & 1u) == 0u) {
                my_sum_y = my_sum_y + y00 + y10;
                my_inter_sad = my_inter_sad + abs(y00 - ry00) + abs(y10 - ry10);
            }
        }
    } else if (scale == 2u) {
        sample_count = 1024;
        for (var hidx = lane; hidx < 1024u; hidx = hidx + 64u) {
            let hy_p = hidx >> 5u;
            let hx_p = hidx & 31u;
            let oy0 = min(hy_p * 2u, max_y);
            let oy1 = min(hy_p * 2u + 1u, max_y);
            let ox0 = min(hx_p * 2u, max_x);
            let ox1 = min(hx_p * 2u + 1u, max_x);

            let c00 = vtb_cur_bgra[(desc.y0 + oy0) * fb_w + (desc.x0 + ox0)];
            let c01 = vtb_cur_bgra[(desc.y0 + oy0) * fb_w + (desc.x0 + ox1)];
            let c10 = vtb_cur_bgra[(desc.y0 + oy1) * fb_w + (desc.x0 + ox0)];
            let c11 = vtb_cur_bgra[(desc.y0 + oy1) * fb_w + (desc.x0 + ox1)];
            let cb = (i32(c00 & 0xFFu) + i32(c01 & 0xFFu) + i32(c10 & 0xFFu) + i32(c11 & 0xFFu) + 2) >> 2u;
            let cg = (i32((c00 >> 8u) & 0xFFu) + i32((c01 >> 8u) & 0xFFu) + i32((c10 >> 8u) & 0xFFu) + i32((c11 >> 8u) & 0xFFu) + 2) >> 2u;
            let cr = (i32((c00 >> 16u) & 0xFFu) + i32((c01 >> 16u) & 0xFFu) + i32((c10 >> 16u) & 0xFFu) + i32((c11 >> 16u) & 0xFFu) + 2) >> 2u;
            let cy_v = rgb_to_y(cr, cg, cb);
            let ccb_v = rgb_to_cb(cr, cg, cb);
            let ccr_v = rgb_to_cr(cr, cg, cb);

            let r00 = read_pred_px((base_sy + oy0) * fb_w + (base_sx + ox0), desc.pred_from_prev);
            let r01 = read_pred_px((base_sy + oy0) * fb_w + (base_sx + ox1), desc.pred_from_prev);
            let r10 = read_pred_px((base_sy + oy1) * fb_w + (base_sx + ox0), desc.pred_from_prev);
            let r11 = read_pred_px((base_sy + oy1) * fb_w + (base_sx + ox1), desc.pred_from_prev);
            let rb = (i32(r00 & 0xFFu) + i32(r01 & 0xFFu) + i32(r10 & 0xFFu) + i32(r11 & 0xFFu) + 2) >> 2u;
            let rg = (i32((r00 >> 8u) & 0xFFu) + i32((r01 >> 8u) & 0xFFu) + i32((r10 >> 8u) & 0xFFu) + i32((r11 >> 8u) & 0xFFu) + 2) >> 2u;
            let rr = (i32((r00 >> 16u) & 0xFFu) + i32((r01 >> 16u) & 0xFFu) + i32((r10 >> 16u) & 0xFFu) + i32((r11 >> 16u) & 0xFFu) + 2) >> 2u;
            let ry_v = rgb_to_y(rr, rg, rb);
            let rcb_v = rgb_to_cb(rr, rg, rb);
            let rcr_v = rgb_to_cr(rr, rg, rb);

            vwg_y_packed[hidx] = pack_i16x2(cy_v, ry_v);
            vwg_c_packed[hidx] = u32(ccb_v) | (u32(ccr_v) << 8u) | (u32(rcb_v) << 16u) | (u32(rcr_v) << 24u);
            my_sum_y = my_sum_y + cy_v;
            my_inter_sad = my_inter_sad + abs(cy_v - ry_v);
        }
    } else {
        sample_count = 256;
        for (var qidx = lane; qidx < 256u; qidx = qidx + 64u) {
            let qy_p = qidx >> 4u;
            let qx_p = qidx & 15u;
            var csb = 0; var csg = 0; var csr = 0;
            var rsb = 0; var rsg = 0; var rsr = 0;
            for (var dy = 0u; dy < 4u; dy = dy + 1u) {
                let oy = min(qy_p * 4u + dy, max_y);
                for (var dx = 0u; dx < 4u; dx = dx + 1u) {
                    let ox = min(qx_p * 4u + dx, max_x);
                    let cp = vtb_cur_bgra[(desc.y0 + oy) * fb_w + (desc.x0 + ox)];
                    let rp = read_pred_px((base_sy + oy) * fb_w + (base_sx + ox), desc.pred_from_prev);
                    csb = csb + i32(cp & 0xFFu); csg = csg + i32((cp >> 8u) & 0xFFu); csr = csr + i32((cp >> 16u) & 0xFFu);
                    rsb = rsb + i32(rp & 0xFFu); rsg = rsg + i32((rp >> 8u) & 0xFFu); rsr = rsr + i32((rp >> 16u) & 0xFFu);
                }
            }
            let cb = (csb + 8) >> 4u; let cg = (csg + 8) >> 4u; let cr = (csr + 8) >> 4u;
            let rb = (rsb + 8) >> 4u; let rg = (rsg + 8) >> 4u; let rr = (rsr + 8) >> 4u;
            let cy_v = rgb_to_y(cr, cg, cb);
            let ry_v = rgb_to_y(rr, rg, rb);
            vwg_y_packed[qidx] = pack_i16x2(cy_v, ry_v);
            vwg_c_packed[qidx] = u32(rgb_to_cb(cr, cg, cb))
                | (u32(rgb_to_cr(cr, cg, cb)) << 8u)
                | (u32(rgb_to_cb(rr, rg, rb)) << 16u)
                | (u32(rgb_to_cr(rr, rg, rb)) << 24u);
            my_sum_y = my_sum_y + cy_v;
            my_inter_sad = my_inter_sad + abs(cy_v - ry_v);
        }
    }
    vwg_inter_sum[lane] = my_sum_y;
    vwg_inter_sad[lane] = my_inter_sad;
    workgroupBarrier();

    // Downsample chroma 2x2 in LDS for scale == 2 (32x32 -> 16x16) and scale == 4 (16x16 -> 8x8)
    var c_down_tmp: array<u32, 4>;
    if (scale == 2u) {
        for (var sub = 0u; sub < 4u; sub = sub + 1u) {
            let ci = sub * 64u + lane;
            let cy = ci >> 4u;
            let cx = ci & 15u;
            let i00 = (cy * 2u) * 32u + (cx * 2u);
            let p00 = vwg_c_packed[i00];
            let p01 = vwg_c_packed[i00 + 1u];
            let p10 = vwg_c_packed[i00 + 32u];
            let p11 = vwg_c_packed[i00 + 33u];
            let ccb = ((p00 & 0xFFu) + (p01 & 0xFFu) + (p10 & 0xFFu) + (p11 & 0xFFu) + 2u) >> 2u;
            let ccr = (((p00 >> 8u) & 0xFFu) + ((p01 >> 8u) & 0xFFu) + ((p10 >> 8u) & 0xFFu) + ((p11 >> 8u) & 0xFFu) + 2u) >> 2u;
            let rcb = (((p00 >> 16u) & 0xFFu) + ((p01 >> 16u) & 0xFFu) + ((p10 >> 16u) & 0xFFu) + ((p11 >> 16u) & 0xFFu) + 2u) >> 2u;
            let rcr = (((p00 >> 24u) & 0xFFu) + ((p01 >> 24u) & 0xFFu) + ((p10 >> 24u) & 0xFFu) + ((p11 >> 24u) & 0xFFu) + 2u) >> 2u;
            c_down_tmp[sub] = ccb | (ccr << 8u) | (rcb << 16u) | (rcr << 24u);
        }
    } else if (scale == 4u) {
        let cy = lane >> 3u;
        let cx = lane & 7u;
        let i00 = (cy * 2u) * 16u + (cx * 2u);
        let p00 = vwg_c_packed[i00];
        let p01 = vwg_c_packed[i00 + 1u];
        let p10 = vwg_c_packed[i00 + 16u];
        let p11 = vwg_c_packed[i00 + 17u];
        let ccb = ((p00 & 0xFFu) + (p01 & 0xFFu) + (p10 & 0xFFu) + (p11 & 0xFFu) + 2u) >> 2u;
        let ccr = (((p00 >> 8u) & 0xFFu) + ((p01 >> 8u) & 0xFFu) + ((p10 >> 8u) & 0xFFu) + ((p11 >> 8u) & 0xFFu) + 2u) >> 2u;
        let rcb = (((p00 >> 16u) & 0xFFu) + ((p01 >> 16u) & 0xFFu) + ((p10 >> 16u) & 0xFFu) + ((p11 >> 16u) & 0xFFu) + 2u) >> 2u;
        let rcr = (((p00 >> 24u) & 0xFFu) + ((p01 >> 24u) & 0xFFu) + ((p10 >> 24u) & 0xFFu) + ((p11 >> 24u) & 0xFFu) + 2u) >> 2u;
        c_down_tmp[0] = ccb | (ccr << 8u) | (rcb << 16u) | (rcr << 24u);
    }
    workgroupBarrier();
    if (scale == 2u) {
        for (var sub = 0u; sub < 4u; sub = sub + 1u) {
            vwg_c_packed[sub * 64u + lane] = c_down_tmp[sub];
        }
    } else if (scale == 4u) {
        vwg_c_packed[lane] = c_down_tmp[0];
    }

    for (var stride_r = 32u; stride_r > 0u; stride_r = stride_r >> 1u) {
        if (lane < stride_r) {
            vwg_inter_sum[lane] = vwg_inter_sum[lane] + vwg_inter_sum[lane + stride_r];
            vwg_inter_sad[lane] = vwg_inter_sad[lane] + vwg_inter_sad[lane + stride_r];
        }
        workgroupBarrier();
    }
    let mean_y = vwg_inter_sum[0] / sample_count;
    let total_inter_sad = vwg_inter_sad[0];

    var my_intra_sad = 0;
    if (scale == 1u) {
        for (var sidx = lane; sidx < 1024u; sidx = sidx + 64u) {
            let cur_y_val = unpack_i16_lo(vwg_y_packed[sidx * 4u]);
            my_intra_sad = my_intra_sad + abs(cur_y_val - mean_y);
        }
    } else if (scale == 2u) {
        for (var sidx = lane; sidx < 1024u; sidx = sidx + 64u) {
            let cur_y_val = unpack_i16_lo(vwg_y_packed[sidx]);
            my_intra_sad = my_intra_sad + abs(cur_y_val - mean_y);
        }
    } else {
        for (var sidx = lane; sidx < 256u; sidx = sidx + 64u) {
            let cur_y_val = unpack_i16_lo(vwg_y_packed[sidx]);
            my_intra_sad = my_intra_sad + abs(cur_y_val - mean_y);
        }
    }
    vwg_intra_sad[lane] = my_intra_sad;
    workgroupBarrier();

    for (var stride_r = 32u; stride_r > 0u; stride_r = stride_r >> 1u) {
        if (lane < stride_r) {
            vwg_intra_sad[lane] = vwg_intra_sad[lane] + vwg_intra_sad[lane + stride_r];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let total_intra_sad = vwg_intra_sad[0];
        vwg_use_inter = select(0u, 1u, (desc.use_inter != 0u) && (total_inter_sad * 10 <= total_intra_sad * 9));
    }
    workgroupBarrier();

    let is_inter = vwg_use_inter != 0u;
    let y_stride = 64u / scale;
    let c_stride = 32u / scale;
    let scaled_w = (desc.tw + scale - 1u) / scale;
    let scaled_h = (desc.th + scale - 1u) / scale;
    let luma_bx = (scaled_w + 7u) >> 3u;
    let luma_by = (scaled_h + 7u) >> 3u;
    let chroma_w = (scaled_w + 1u) >> 1u;
    let chroma_h = (scaled_h + 1u) >> 1u;
    let chroma_bx = (chroma_w + 7u) >> 3u;
    let chroma_by = (chroma_h + 7u) >> 3u;
    let num_luma = luma_bx * luma_by;
    let num_chroma = chroma_bx * chroma_by;
    let total_active_blocks = num_luma + 2u * num_chroma;

    let tile_qstride = select(select(3073u, 769u, scale == 2u), 193u, scale == 4u);
    let qout_tile_base = tidx * tile_qstride;
    if (lane == 0u) {
        vtb_qblocks[qout_tile_base] = select(0, 1, is_inter);
    }
    let qout_blocks_base = qout_tile_base + 1u;

    let num_batches = (total_active_blocks + 7u) >> 3u;
    for (var b_batch = 0u; b_batch < num_batches; b_batch = b_batch + 1u) {
        for (var sub = 0u; sub < 8u; sub = sub + 1u) {
            let elem = sub * 64u + lane;
            let local_blk = elem >> 6u;
            let k = elem & 63u;
            let br = k >> 3u;
            let bc = k & 7u;
            let seq_blk = b_batch * 8u + local_blk;

            if (seq_blk < total_active_blocks) {
                let is_y = seq_blk < num_luma;
                let is_cb = !is_y && (seq_blk < num_luma + num_chroma);
                if (is_y) {
                    let by = seq_blk / luma_bx;
                    let bx = seq_blk % luma_bx;
                    let p_idx = (by * 8u + br) * y_stride + (bx * 8u + bc);
                    let yp = vwg_y_packed[p_idx];
                    let cur_y = unpack_i16_lo(yp);
                    let rv = select(128, unpack_i16_hi(yp), is_inter);
                    vwg_multi_ref[elem] = rv;
                    vwg_multi_blk[elem] = cur_y - rv;
                } else {
                    let c_seq = select(seq_blk - num_luma - num_chroma, seq_blk - num_luma, is_cb);
                    let by = c_seq / chroma_bx;
                    let bx = c_seq % chroma_bx;
                    let p_idx = (by * 8u + br) * c_stride + (bx * 8u + bc);
                    let cp = vwg_c_packed[p_idx];
                    let cur_c = i32(select((cp >> 8u) & 0xFFu, cp & 0xFFu, is_cb));
                    let ref_c = i32(select((cp >> 24u) & 0xFFu, (cp >> 16u) & 0xFFu, is_cb));
                    let rv = select(128, ref_c, is_inter);
                    vwg_multi_ref[elem] = rv;
                    vwg_multi_blk[elem] = cur_c - rv;
                }
            }
        }
        workgroupBarrier();

        let local_blk = lane >> 3u;
        let line_idx = lane & 7u;
        let seq_blk_lane = b_batch * 8u + local_blk;
        let is_y_lane = seq_blk_lane < num_luma;
        let is_cb_lane = !is_y_lane && (seq_blk_lane < num_luma + num_chroma);

        if (seq_blk_lane < total_active_blocks) {
            let ro = local_blk * 64u + line_idx * 8u;
            let x0 = vwg_multi_blk[ro];
            let x1 = vwg_multi_blk[ro + 1u];
            let x2 = vwg_multi_blk[ro + 2u];
            let x3 = vwg_multi_blk[ro + 3u];
            let x4 = vwg_multi_blk[ro + 4u];
            let x5 = vwg_multi_blk[ro + 5u];
            let x6 = vwg_multi_blk[ro + 6u];
            let x7 = vwg_multi_blk[ro + 7u];

            let s0 = x0 + x7; let s1 = x1 + x6; let s2 = x2 + x5; let s3 = x3 + x4;
            let d0 = x0 - x7; let d1 = x1 - x6; let d2 = x2 - x5; let d3 = x3 - x4;

            vwg_multi_tmp[ro]      = (C4 * (s0 + s1 + s2 + s3) + 1024) >> 11u;
            vwg_multi_tmp[ro + 1u] = (C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3 + 1024) >> 11u;
            vwg_multi_tmp[ro + 2u] = (C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3 + 1024) >> 11u;
            vwg_multi_tmp[ro + 3u] = (C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3 + 1024) >> 11u;
            vwg_multi_tmp[ro + 4u] = (C4 * (s0 - s1 - s2 + s3) + 1024) >> 11u;
            vwg_multi_tmp[ro + 5u] = (C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3 + 1024) >> 11u;
            vwg_multi_tmp[ro + 6u] = (C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3 + 1024) >> 11u;
            vwg_multi_tmp[ro + 7u] = (C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3 + 1024) >> 11u;
        }
        workgroupBarrier();

        if (seq_blk_lane < total_active_blocks) {
            let co = local_blk * 64u + line_idx;
            let x0 = vwg_multi_tmp[co];
            let x1 = vwg_multi_tmp[co + 8u];
            let x2 = vwg_multi_tmp[co + 16u];
            let x3 = vwg_multi_tmp[co + 24u];
            let x4 = vwg_multi_tmp[co + 32u];
            let x5 = vwg_multi_tmp[co + 40u];
            let x6 = vwg_multi_tmp[co + 48u];
            let x7 = vwg_multi_tmp[co + 56u];

            let s0 = x0 + x7; let s1 = x1 + x6; let s2 = x2 + x5; let s3 = x3 + x4;
            let d0 = x0 - x7; let d1 = x1 - x6; let d2 = x2 - x5; let d3 = x3 - x4;

            vwg_multi_blk[co]       = (C4 * (s0 + s1 + s2 + s3) + 65536) >> 17u;
            vwg_multi_blk[co + 8u]  = (C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3 + 65536) >> 17u;
            vwg_multi_blk[co + 16u] = (C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3 + 65536) >> 17u;
            vwg_multi_blk[co + 24u] = (C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3 + 65536) >> 17u;
            vwg_multi_blk[co + 32u] = (C4 * (s0 - s1 - s2 + s3) + 65536) >> 17u;
            vwg_multi_blk[co + 40u] = (C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3 + 65536) >> 17u;
            vwg_multi_blk[co + 48u] = (C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3 + 65536) >> 17u;
            vwg_multi_blk[co + 56u] = (C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3 + 65536) >> 17u;
        }
        workgroupBarrier();

        for (var sub = 0u; sub < 4u; sub = sub + 1u) {
            let pair_idx = sub * 64u + lane;
            let p_blk = pair_idx >> 5u;
            let p_k = pair_idx & 31u;
            let seq_blk = b_batch * 8u + p_blk;
            if (seq_blk < total_active_blocks) {
                let is_y = seq_blk < num_luma;
                let k0 = p_k * 2u;
                let k1 = k0 + 1u;
                let e0 = p_blk * 64u + k0;
                let e1 = e0 + 1u;
                let q0_step = select(vget_chroma_q(k0), vget_luma_q(k0), is_y);
                let q1_step = select(vget_chroma_q(k1), vget_luma_q(k1), is_y);
                let q0 = quantize_div(vwg_multi_blk[e0], q0_step);
                let q1 = quantize_div(vwg_multi_blk[e1], q1_step);
                let packed = (u32(q0) & 0xFFFFu) | ((u32(q1) & 0xFFFFu) << 16u);
                vtb_qblocks[qout_blocks_base + seq_blk * 32u + p_k] = bitcast<i32>(packed);
                vwg_multi_blk[e0] = clamp(q0 * q0_step, -32768, 32768);
                vwg_multi_blk[e1] = clamp(q1 * q1_step, -32768, 32768);
            }
        }
        workgroupBarrier();

        if (seq_blk_lane < total_active_blocks) {
            let co = local_blk * 64u + line_idx;
            let x0 = vwg_multi_blk[co];
            let x1 = vwg_multi_blk[co + 8u];
            let x2 = vwg_multi_blk[co + 16u];
            let x3 = vwg_multi_blk[co + 24u];
            let x4 = vwg_multi_blk[co + 32u];
            let x5 = vwg_multi_blk[co + 40u];
            let x6 = vwg_multi_blk[co + 48u];
            let x7 = vwg_multi_blk[co + 56u];

            let a0 = C4 * (x0 + x4); let a1 = C4 * (x0 - x4);
            let b0 = C2 * x2 + C6 * x6; let b1 = C6 * x2 - C2 * x6;
            let e0 = a0 + b0; let e1 = a1 + b1; let e2 = a1 - b1; let e3 = a0 - b0;
            let o0 = C1 * x1 + C3 * x3 + C5 * x5 + C7 * x7;
            let o1 = C3 * x1 - C7 * x3 - C1 * x5 - C5 * x7;
            let o2 = C5 * x1 - C1 * x3 + C7 * x5 + C3 * x7;
            let o3 = C7 * x1 - C5 * x3 + C3 * x5 - C1 * x7;

            vwg_multi_tmp[co]       = (e0 + o0 + 1024) >> 11u;
            vwg_multi_tmp[co + 8u]  = (e1 + o1 + 1024) >> 11u;
            vwg_multi_tmp[co + 16u] = (e2 + o2 + 1024) >> 11u;
            vwg_multi_tmp[co + 24u] = (e3 + o3 + 1024) >> 11u;
            vwg_multi_tmp[co + 32u] = (e3 - o3 + 1024) >> 11u;
            vwg_multi_tmp[co + 40u] = (e2 - o2 + 1024) >> 11u;
            vwg_multi_tmp[co + 48u] = (e1 - o1 + 1024) >> 11u;
            vwg_multi_tmp[co + 56u] = (e0 - o0 + 1024) >> 11u;
        }
        workgroupBarrier();

        if (seq_blk_lane < total_active_blocks) {
            let ro = local_blk * 64u + line_idx * 8u;
            let x0 = vwg_multi_tmp[ro];
            let x1 = vwg_multi_tmp[ro + 1u];
            let x2 = vwg_multi_tmp[ro + 2u];
            let x3 = vwg_multi_tmp[ro + 3u];
            let x4 = vwg_multi_tmp[ro + 4u];
            let x5 = vwg_multi_tmp[ro + 5u];
            let x6 = vwg_multi_tmp[ro + 6u];
            let x7 = vwg_multi_tmp[ro + 7u];

            let a0 = C4 * (x0 + x4); let a1 = C4 * (x0 - x4);
            let b0 = C2 * x2 + C6 * x6; let b1 = C6 * x2 - C2 * x6;
            let e0 = a0 + b0; let e1 = a1 + b1; let e2 = a1 - b1; let e3 = a0 - b0;
            let o0 = C1 * x1 + C3 * x3 + C5 * x5 + C7 * x7;
            let o1 = C3 * x1 - C7 * x3 - C1 * x5 - C5 * x7;
            let o2 = C5 * x1 - C1 * x3 + C7 * x5 + C3 * x7;
            let o3 = C7 * x1 - C5 * x3 + C3 * x5 - C1 * x7;

            var y_out: array<i32, 8>;
            y_out[0] = clamp(vwg_multi_ref[ro]      + ((e0 + o0 + 65536) >> 17u), 0, 255);
            y_out[1] = clamp(vwg_multi_ref[ro + 1u] + ((e1 + o1 + 65536) >> 17u), 0, 255);
            y_out[2] = clamp(vwg_multi_ref[ro + 2u] + ((e2 + o2 + 65536) >> 17u), 0, 255);
            y_out[3] = clamp(vwg_multi_ref[ro + 3u] + ((e3 + o3 + 65536) >> 17u), 0, 255);
            y_out[4] = clamp(vwg_multi_ref[ro + 4u] + ((e3 - o3 + 65536) >> 17u), 0, 255);
            y_out[5] = clamp(vwg_multi_ref[ro + 5u] + ((e2 - o2 + 65536) >> 17u), 0, 255);
            y_out[6] = clamp(vwg_multi_ref[ro + 6u] + ((e1 - o1 + 65536) >> 17u), 0, 255);
            y_out[7] = clamp(vwg_multi_ref[ro + 7u] + ((e0 - o0 + 65536) >> 17u), 0, 255);

            if (is_y_lane) {
                let by = seq_blk_lane / luma_bx;
                let bx = seq_blk_lane % luma_bx;
                let dst_base = (by * 8u + line_idx) * y_stride + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    vwg_y_packed[dst_base + c] = y_out[c];
                }
            } else if (is_cb_lane) {
                let c_seq = seq_blk_lane - num_luma;
                let by = c_seq / chroma_bx;
                let bx = c_seq % chroma_bx;
                let dst_base = (by * 8u + line_idx) * c_stride + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    let c_idx = dst_base + c;
                    vwg_c_packed[c_idx] = (vwg_c_packed[c_idx] & 0xFFFFFF00u) | u32(y_out[c]);
                }
            } else {
                let c_seq = seq_blk_lane - num_luma - num_chroma;
                let by = c_seq / chroma_bx;
                let bx = c_seq % chroma_bx;
                let dst_base = (by * 8u + line_idx) * c_stride + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    let c_idx = dst_base + c;
                    vwg_c_packed[c_idx] = (vwg_c_packed[c_idx] & 0xFFFF00FFu) | (u32(y_out[c]) << 8u);
                }
            }
        }
        workgroupBarrier();
    }

    // In-loop 4-tap 8x8 deblocking + 1D SAO anti-ringing across active plane stride
    let luma_q0 = vget_luma_q(0u);
    let beta_y = clamp(luma_q0 + 4, 6, 36);
    let tc_y = clamp((luma_q0 + 3) >> 2u, 1, 10);
    let chroma_q0 = vget_chroma_q(0u);
    let beta_c = clamp(chroma_q0 + 4, 6, 36);
    let tc_c = clamp((chroma_q0 + 3) >> 2u, 1, 10);

    let active_yh = min(luma_by * 8u, y_stride);
    if (lane < active_yh) {
        let row_off = lane * y_stride;
        for (var bx = 1u; bx < luma_bx; bx = bx + 1u) {
            let x = bx * 8u;
            let p1 = vwg_y_packed[row_off + x - 2u];
            let p0 = vwg_y_packed[row_off + x - 1u];
            let q0 = vwg_y_packed[row_off + x];
            let q1 = vwg_y_packed[row_off + x + 1u];
            if (abs(p0 - q0) < beta_y && abs(p1 - p0) < (beta_y >> 1u) && abs(q1 - q0) < (beta_y >> 1u)) {
                let delta = clamp(((q0 - p0) * 3 + (p1 - q1) + 4) >> 3u, -tc_y, tc_y);
                let delta1 = clamp(delta >> 1u, -(tc_y >> 1u), tc_y >> 1u);
                vwg_y_packed[row_off + x - 2u] = clamp(p1 + delta1, 0, 255);
                vwg_y_packed[row_off + x - 1u] = clamp(p0 + delta, 0, 255);
                vwg_y_packed[row_off + x]      = clamp(q0 - delta, 0, 255);
                vwg_y_packed[row_off + x + 1u] = clamp(q1 - delta1, 0, 255);
            }
        }
    }
    let active_ch = min(chroma_by * 8u, c_stride);
    if (lane < active_ch) {
        let row_off = lane * c_stride;
        for (var bx = 1u; bx < chroma_bx; bx = bx + 1u) {
            let x = bx * 8u;
            let cp_p1 = vwg_c_packed[row_off + x - 2u];
            let cp_p0 = vwg_c_packed[row_off + x - 1u];
            let cp_q0 = vwg_c_packed[row_off + x];
            let cp_q1 = vwg_c_packed[row_off + x + 1u];

            var cb_p1 = i32(cp_p1 & 0xFFu); var cr_p1 = i32((cp_p1 >> 8u) & 0xFFu);
            var cb_p0 = i32(cp_p0 & 0xFFu); var cr_p0 = i32((cp_p0 >> 8u) & 0xFFu);
            var cb_q0 = i32(cp_q0 & 0xFFu); var cr_q0 = i32((cp_q0 >> 8u) & 0xFFu);
            var cb_q1 = i32(cp_q1 & 0xFFu); var cr_q1 = i32((cp_q1 >> 8u) & 0xFFu);

            if (abs(cb_p0 - cb_q0) < beta_c && abs(cb_p1 - cb_p0) < (beta_c >> 1u) && abs(cb_q1 - cb_q0) < (beta_c >> 1u)) {
                let d = clamp(((cb_q0 - cb_p0) * 3 + (cb_p1 - cb_q1) + 4) >> 3u, -tc_c, tc_c);
                let d1 = clamp(d >> 1u, -(tc_c >> 1u), tc_c >> 1u);
                cb_p1 = clamp(cb_p1 + d1, 0, 255); cb_p0 = clamp(cb_p0 + d, 0, 255);
                cb_q0 = clamp(cb_q0 - d, 0, 255);  cb_q1 = clamp(cb_q1 - d1, 0, 255);
            }
            if (abs(cr_p0 - cr_q0) < beta_c && abs(cr_p1 - cr_p0) < (beta_c >> 1u) && abs(cr_q1 - cr_q0) < (beta_c >> 1u)) {
                let d = clamp(((cr_q0 - cr_p0) * 3 + (cr_p1 - cr_q1) + 4) >> 3u, -tc_c, tc_c);
                let d1 = clamp(d >> 1u, -(tc_c >> 1u), tc_c >> 1u);
                cr_p1 = clamp(cr_p1 + d1, 0, 255); cr_p0 = clamp(cr_p0 + d, 0, 255);
                cr_q0 = clamp(cr_q0 - d, 0, 255);  cr_q1 = clamp(cr_q1 - d1, 0, 255);
            }
            vwg_c_packed[row_off + x - 2u] = u32(cb_p1) | (u32(cr_p1) << 8u);
            vwg_c_packed[row_off + x - 1u] = u32(cb_p0) | (u32(cr_p0) << 8u);
            vwg_c_packed[row_off + x]      = u32(cb_q0) | (u32(cr_q0) << 8u);
            vwg_c_packed[row_off + x + 1u] = u32(cb_q1) | (u32(cr_q1) << 8u);
        }
    }
    workgroupBarrier();

    let active_yw = min(luma_bx * 8u, y_stride);
    if (lane < active_yw) {
        let x = lane;
        for (var by = 1u; by < luma_by; by = by + 1u) {
            let y = by * 8u;
            let p1 = vwg_y_packed[(y - 2u) * y_stride + x];
            let p0 = vwg_y_packed[(y - 1u) * y_stride + x];
            let q0 = vwg_y_packed[y * y_stride + x];
            let q1 = vwg_y_packed[(y + 1u) * y_stride + x];
            if (abs(p0 - q0) < beta_y && abs(p1 - p0) < (beta_y >> 1u) && abs(q1 - q0) < (beta_y >> 1u)) {
                let delta = clamp(((q0 - p0) * 3 + (p1 - q1) + 4) >> 3u, -tc_y, tc_y);
                let delta1 = clamp(delta >> 1u, -(tc_y >> 1u), tc_y >> 1u);
                vwg_y_packed[(y - 2u) * y_stride + x] = clamp(p1 + delta1, 0, 255);
                vwg_y_packed[(y - 1u) * y_stride + x] = clamp(p0 + delta, 0, 255);
                vwg_y_packed[y * y_stride + x]        = clamp(q0 - delta, 0, 255);
                vwg_y_packed[(y + 1u) * y_stride + x] = clamp(q1 - delta1, 0, 255);
            }
        }
    }
    let active_cw = min(chroma_bx * 8u, c_stride);
    if (lane < active_cw) {
        let x = lane;
        for (var by = 1u; by < chroma_by; by = by + 1u) {
            let y = by * 8u;
            let cp_p1 = vwg_c_packed[(y - 2u) * c_stride + x];
            let cp_p0 = vwg_c_packed[(y - 1u) * c_stride + x];
            let cp_q0 = vwg_c_packed[y * c_stride + x];
            let cp_q1 = vwg_c_packed[(y + 1u) * c_stride + x];

            var cb_p1 = i32(cp_p1 & 0xFFu); var cr_p1 = i32((cp_p1 >> 8u) & 0xFFu);
            var cb_p0 = i32(cp_p0 & 0xFFu); var cr_p0 = i32((cp_p0 >> 8u) & 0xFFu);
            var cb_q0 = i32(cp_q0 & 0xFFu); var cr_q0 = i32((cp_q0 >> 8u) & 0xFFu);
            var cb_q1 = i32(cp_q1 & 0xFFu); var cr_q1 = i32((cp_q1 >> 8u) & 0xFFu);

            if (abs(cb_p0 - cb_q0) < beta_c && abs(cb_p1 - cb_p0) < (beta_c >> 1u) && abs(cb_q1 - cb_q0) < (beta_c >> 1u)) {
                let d = clamp(((cb_q0 - cb_p0) * 3 + (cb_p1 - cb_q1) + 4) >> 3u, -tc_c, tc_c);
                let d1 = clamp(d >> 1u, -(tc_c >> 1u), tc_c >> 1u);
                cb_p1 = clamp(cb_p1 + d1, 0, 255); cb_p0 = clamp(cb_p0 + d, 0, 255);
                cb_q0 = clamp(cb_q0 - d, 0, 255);  cb_q1 = clamp(cb_q1 - d1, 0, 255);
            }
            if (abs(cr_p0 - cr_q0) < beta_c && abs(cr_p1 - cr_p0) < (beta_c >> 1u) && abs(cr_q1 - cr_q0) < (beta_c >> 1u)) {
                let d = clamp(((cr_q0 - cr_p0) * 3 + (cr_p1 - cr_q1) + 4) >> 3u, -tc_c, tc_c);
                let d1 = clamp(d >> 1u, -(tc_c >> 1u), tc_c >> 1u);
                cr_p1 = clamp(cr_p1 + d1, 0, 255); cr_p0 = clamp(cr_p0 + d, 0, 255);
                cr_q0 = clamp(cr_q0 - d, 0, 255);  cr_q1 = clamp(cr_q1 - d1, 0, 255);
            }
            vwg_c_packed[(y - 2u) * c_stride + x] = u32(cb_p1) | (u32(cr_p1) << 8u);
            vwg_c_packed[(y - 1u) * c_stride + x] = u32(cb_p0) | (u32(cr_p0) << 8u);
            vwg_c_packed[y * c_stride + x]        = u32(cb_q0) | (u32(cr_q0) << 8u);
            vwg_c_packed[(y + 1u) * c_stride + x] = u32(cb_q1) | (u32(cr_q1) << 8u);
        }
    }
    workgroupBarrier();

    let sao_thresh_y = max(beta_y >> 1u, 3);
    if (lane < active_yh) {
        let row_off = lane * y_stride;
        for (var bx = 0u; bx < luma_bx; bx = bx + 1u) {
            let base = row_off + bx * 8u;
            var line: array<i32, 8>;
            for (var c = 0u; c < 8u; c = c + 1u) {
                line[c] = vwg_y_packed[base + c];
            }
            for (var c = 1u; c < 7u; c = c + 1u) {
                let cur = line[c];
                let d_left = cur - line[c - 1u];
                let d_right = cur - line[c + 1u];
                if (abs(d_left) < sao_thresh_y && abs(d_right) < sao_thresh_y) {
                    if (d_left > 0 && d_right > 0) {
                        let step = clamp((min(d_left, d_right) + 1) >> 2u, 1, 2);
                        vwg_y_packed[base + c] = clamp(cur - step, 0, 255);
                    } else if (d_left < 0 && d_right < 0) {
                        let step = clamp((min(-d_left, -d_right) + 1) >> 2u, 1, 2);
                        vwg_y_packed[base + c] = clamp(cur + step, 0, 255);
                    }
                }
            }
        }
    }
    let sao_thresh_c = max(beta_c >> 1u, 3);
    if (lane < active_ch) {
        let row_off = lane * c_stride;
        for (var bx = 0u; bx < chroma_bx; bx = bx + 1u) {
            let base = row_off + bx * 8u;
            var cb_line: array<i32, 8>;
            var cr_line: array<i32, 8>;
            for (var c = 0u; c < 8u; c = c + 1u) {
                let cp = vwg_c_packed[base + c];
                cb_line[c] = i32(cp & 0xFFu);
                cr_line[c] = i32((cp >> 8u) & 0xFFu);
            }
            for (var c = 1u; c < 7u; c = c + 1u) {
                var cb_out = cb_line[c];
                let cb_dl = cb_out - cb_line[c - 1u];
                let cb_dr = cb_out - cb_line[c + 1u];
                if (abs(cb_dl) < sao_thresh_c && abs(cb_dr) < sao_thresh_c) {
                    if (cb_dl > 0 && cb_dr > 0) {
                        let step = clamp((min(cb_dl, cb_dr) + 1) >> 2u, 1, 2);
                        cb_out = clamp(cb_out - step, 0, 255);
                    } else if (cb_dl < 0 && cb_dr < 0) {
                        let step = clamp((min(-cb_dl, -cb_dr) + 1) >> 2u, 1, 2);
                        cb_out = clamp(cb_out + step, 0, 255);
                    }
                }
                var cr_out = cr_line[c];
                let cr_dl = cr_out - cr_line[c - 1u];
                let cr_dr = cr_out - cr_line[c + 1u];
                if (abs(cr_dl) < sao_thresh_c && abs(cr_dr) < sao_thresh_c) {
                    if (cr_dl > 0 && cr_dr > 0) {
                        let step = clamp((min(cr_dl, cr_dr) + 1) >> 2u, 1, 2);
                        cr_out = clamp(cr_out - step, 0, 255);
                    } else if (cr_dl < 0 && cr_dr < 0) {
                        let step = clamp((min(-cr_dl, -cr_dr) + 1) >> 2u, 1, 2);
                        cr_out = clamp(cr_out + step, 0, 255);
                    }
                }
                vwg_c_packed[base + c] = u32(cb_out) | (u32(cr_out) << 8u);
            }
        }
    }
    workgroupBarrier();

    // Write reconstructed BGRA directly into `vtb_ref_bgra` in VRAM!
    for (var idx = lane; idx < 4096u; idx = idx + 64u) {
        let py = idx >> 6u;
        let px = idx & 63u;
        if (py < desc.th && px < desc.tw) {
            var y_val = 0;
            var cp = 0u;
            if (scale == 1u) {
                y_val = vwg_y_packed[idx];
                cp = vwg_c_packed[(py >> 1u) * 32u + (px >> 1u)];
            } else if (scale == 2u) {
                let hy0 = py >> 1u;
                let hy1 = select(min(hy0 + 1u, 31u), hy0, (py & 1u) == 0u);
                let hx0 = px >> 1u;
                let hx1 = select(min(hx0 + 1u, 31u), hx0, (px & 1u) == 0u);
                y_val = (vwg_y_packed[hy0 * 32u + hx0]
                    + vwg_y_packed[hy0 * 32u + hx1]
                    + vwg_y_packed[hy1 * 32u + hx0]
                    + vwg_y_packed[hy1 * 32u + hx1] + 2) >> 2u;
                cp = vwg_c_packed[(hy0 >> 1u) * 16u + (hx0 >> 1u)];
            } else {
                let qy0 = py >> 2u;
                let qy1 = min(qy0 + 1u, 15u);
                let wy = i32(py & 3u);
                let wy0 = 4 - wy;
                let qx0 = px >> 2u;
                let qx1 = min(qx0 + 1u, 15u);
                let wx = i32(px & 3u);
                let wx0 = 4 - wx;
                let r0 = qy0 * 16u;
                let r1 = qy1 * 16u;
                let top = vwg_y_packed[r0 + qx0] * wx0 + vwg_y_packed[r0 + qx1] * wx;
                let bot = vwg_y_packed[r1 + qx0] * wx0 + vwg_y_packed[r1 + qx1] * wx;
                y_val = (top * wy0 + bot * wy + 8) >> 4u;
                cp = vwg_c_packed[(qy0 >> 1u) * 8u + (qx0 >> 1u)];
            }
            let cb_val = i32(cp & 0xFFu);
            let cr_val = i32((cp >> 8u) & 0xFFu);
            vtb_ref_bgra[(desc.y0 + py) * fb_w + (desc.x0 + px)] = ycbcr_to_bgra(y_val, cb_val, cr_val, 255);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Refresh / Alpha Tile Gather & Scatter Helpers
// mode == 0: gather `(cur_tile, ref_tile)` into `tile_io` for idle lossless refresh
// mode == 1: commit `cur_bgra -> ref_bgra` in VRAM for refined tiles
// mode == 2: scatter `tile_io[i * 4096 ..]` into `ref_bgra` for rare alpha tiles
// ---------------------------------------------------------------------------

struct TileSyncParams {
    fb_w: u32,
    fb_h: u32,
    num_rects: u32,
    mode: u32,
}

struct TileRect {
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
}

@group(0) @binding(0) var<uniform> sync_params: TileSyncParams;
@group(0) @binding(1) var<storage, read> sync_rects: array<TileRect>;
@group(0) @binding(2) var<storage, read> sync_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read_write> sync_ref_bgra: array<u32>;
@group(0) @binding(4) var<storage, read_write> sync_tile_io: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn sync_vram_tiles(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ridx = wid.x;
    if (ridx >= sync_params.num_rects) {
        return;
    }
    let rect = sync_rects[ridx];
    let lane = lid.x;
    let fb_w = sync_params.fb_w;

    if (sync_params.mode == 0u) {
        let base_out = ridx * 8192u;
        for (var i = lane; i < 4096u; i = i + 64u) {
            let py = i >> 6u;
            let px = i & 63u;
            if (py < rect.th && px < rect.tw) {
                let fb_idx = (rect.y0 + py) * fb_w + (rect.x0 + px);
                sync_tile_io[base_out + i] = sync_cur_bgra[fb_idx];
                sync_tile_io[base_out + 4096u + i] = sync_ref_bgra[fb_idx];
            } else {
                sync_tile_io[base_out + i] = 0u;
                sync_tile_io[base_out + 4096u + i] = 0u;
            }
        }
    } else if (sync_params.mode == 1u) {
        if (lane < rect.th) {
            let row = (rect.y0 + lane) * fb_w + rect.x0;
            for (var px = 0u; px < rect.tw; px = px + 1u) {
                sync_ref_bgra[row + px] = sync_cur_bgra[row + px];
            }
        }
    } else {
        let base_in = ridx * 4096u;
        for (var i = lane; i < 4096u; i = i + 64u) {
            let py = i >> 6u;
            let px = i & 63u;
            if (py < rect.th && px < rect.tw) {
                let fb_idx = (rect.y0 + py) * fb_w + (rect.x0 + px);
                sync_ref_bgra[fb_idx] = sync_tile_io[base_in + i];
            }
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IngestParamsUniform {
    fb_w: u32,
    fb_h: u32,
    cols: u32,
    rows: u32,
    crop_x: u32,
    crop_y: u32,
    rb_swizzle: u32,
    is_keyframe: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScrollParamsUniform {
    fb_w: u32,
    fb_h: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VramPrepParamsUniform {
    pub fb_w: u32,
    pub fb_h: u32,
    pub num_cmds: u32,
    pub scale: u32,
    pub frame_dx: i32,
    pub frame_dy: i32,
    pub has_frame_mv: u32,
    pub enable_local_mv: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VramTileCmd {
    pub x0: u32,
    pub y0: u32,
    pub tw: u32,
    pub th: u32,
    pub flags: u32,
    pub batch_idx: u32,
    pub extract_idx: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VramTileDesc {
    pub x0: u32,
    pub y0: u32,
    pub tw: u32,
    pub th: u32,
    pub pred_x: u32,
    pub pred_y: u32,
    pub pred_from_prev: u32,
    pub use_inter: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VramBatchParamsUniform {
    fb_w: u32,
    fb_h: u32,
    num_tiles: u32,
    scale: u32,
    luma_q: [i32; 64],
    chroma_q: [i32; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileSyncParamsUniform {
    fb_w: u32,
    fb_h: u32,
    num_rects: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TileRect {
    pub x0: u32,
    pub y0: u32,
    pub tw: u32,
    pub th: u32,
}

pub(crate) struct VramPipelines {
    ingest_bgl: wgpu::BindGroupLayout,
    ingest_pipeline: wgpu::ComputePipeline,
    scroll_bgl: wgpu::BindGroupLayout,
    scroll_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    prep_pipeline: wgpu::ComputePipeline,
    batch_bgl: wgpu::BindGroupLayout,
    batch_pipeline: wgpu::ComputePipeline,
    sync_bgl: wgpu::BindGroupLayout,
    sync_pipeline: wgpu::ComputePipeline,
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl VramPipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("smart_tile_vram.wgsl"),
                    source: wgpu::ShaderSource::Wgsl(SMART_TILE_VRAM_WGSL.into()),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };

        let ingest_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram_ingest_bgl"),
            entries: &[
                bgl_uniform(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                bgl_storage(2, true),
                bgl_storage(3, false),
                bgl_storage(4, false),
                bgl_storage(5, false),
            ],
        });
        let ingest_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram_ingest_pl"),
            bind_group_layouts: &[Some(&ingest_bgl)],
            immediate_size: 0,
        });
        let ingest_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ingest_and_scan_vram_pipeline"),
            layout: Some(&ingest_pl),
            module: &shader,
            entry_point: Some("ingest_and_scan_vram"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scroll_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram_scroll_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, true),
                bgl_storage(2, true),
                bgl_storage(3, false),
                bgl_storage(4, true),
            ],
        });
        let scroll_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram_scroll_pl"),
            bind_group_layouts: &[Some(&scroll_bgl)],
            immediate_size: 0,
        });
        let scroll_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_scroll_summaries_pipeline"),
            layout: Some(&scroll_pl),
            module: &shader,
            entry_point: Some("compute_scroll_summaries"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram_prep_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, true),
                bgl_storage(2, true),
                bgl_storage(3, true),
                bgl_storage(4, false),
                bgl_storage(5, false),
                bgl_storage(6, false),
                bgl_storage(7, false),
            ],
        });
        let prep_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram_prep_pl"),
            bind_group_layouts: &[Some(&prep_bgl)],
            immediate_size: 0,
        });
        let prep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prepare_vram_tile_mvs_pipeline"),
            layout: Some(&prep_pl),
            module: &shader,
            entry_point: Some("prepare_vram_tile_mvs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let batch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram_batch_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, true),
                bgl_storage(2, true),
                bgl_storage(3, true),
                bgl_storage(4, false),
                bgl_storage(5, false),
            ],
        });
        let batch_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram_batch_pl"),
            bind_group_layouts: &[Some(&batch_bgl)],
            immediate_size: 0,
        });
        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("encode_vram_tiles_batch_pipeline"),
            layout: Some(&batch_pl),
            module: &shader,
            entry_point: Some("encode_vram_tiles_batch"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sync_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram_sync_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, true),
                bgl_storage(2, true),
                bgl_storage(3, false),
                bgl_storage(4, false),
            ],
        });
        let sync_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram_sync_pl"),
            bind_group_layouts: &[Some(&sync_bgl)],
            immediate_size: 0,
        });
        let sync_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sync_vram_tiles_pipeline"),
            layout: Some(&sync_pl),
            module: &shader,
            entry_point: Some("sync_vram_tiles"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            ingest_bgl,
            ingest_pipeline,
            scroll_bgl,
            scroll_pipeline,
            prep_bgl,
            prep_pipeline,
            batch_bgl,
            batch_pipeline,
            sync_bgl,
            sync_pipeline,
        }
    }
}

/// GPU texture frame descriptor passed to `SmartTileEncoder::encode_gpu_texture`.
#[derive(Clone)]
pub struct GpuTextureFrame {
    pub engine: Arc<SmartTileGpuEngine>,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub crop_x: u32,
    pub crop_y: u32,
    pub width: u32,
    pub height: u32,
    pub rb_swizzle: bool,
    pub external_memory: bool,
}

impl std::fmt::Debug for GpuTextureFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTextureFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("crop_x", &self.crop_x)
            .field("crop_y", &self.crop_y)
            .field("rb_swizzle", &self.rb_swizzle)
            .finish()
    }
}

/// Exported Vulkan linear `DMA-BUF` texture shared between compositor GLES rendering and `wgpu`.
pub struct ExportedDmabufTexture {
    pub engine: Arc<SmartTileGpuEngine>,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset: u32,
    pub bgra_format: bool,
}

impl ExportedDmabufTexture {
    /// Allocate a `DRM_FORMAT_MOD_LINEAR` Vulkan `VkImage` backed by exportable `DMA_BUF_EXT`
    /// memory, returning `(ExportedDmabufTexture, OwnedFd)` for import into EGL/GLES (`Dmabuf`).
    #[cfg(unix)]
    pub fn try_allocate(
        engine: Arc<SmartTileGpuEngine>,
        width: u32,
        height: u32,
        bgra_format: bool,
    ) -> Option<(Self, OwnedFd)> {
        use ash::vk;

        if width == 0 || height == 0 || !engine.supports_dmabuf() {
            return None;
        }

        let wgpu_format = if bgra_format {
            wgpu::TextureFormat::Bgra8Unorm
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let vk_format = if bgra_format {
            vk::Format::B8G8R8A8_UNORM
        } else {
            vk::Format::R8G8B8A8_UNORM
        };

        let (hal_tex, fd, stride, offset) = unsafe {
            let hal_dev = engine.device().as_hal::<wgpu::hal::api::Vulkan>()?;
            let raw_dev = hal_dev.raw_device();
            let raw_inst = hal_dev.shared_instance().raw_instance();
            let phys = hal_dev.raw_physical_device();

            let modifiers = [0u64]; // DRM_FORMAT_MOD_LINEAR
            let mut mod_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let mut ext_mem_img = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            let img_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_mem_img)
                .push_next(&mut mod_list);

            let vk_img = raw_dev.create_image(&img_info, None).ok()?;
            let mem_reqs = raw_dev.get_image_memory_requirements(vk_img);
            let mem_props = raw_inst.get_physical_device_memory_properties(phys);

            let find_mem = |req_flags: vk::MemoryPropertyFlags| -> Option<u32> {
                for (i, mt) in mem_props.memory_types_as_slice().iter().enumerate() {
                    if (mem_reqs.memory_type_bits & (1 << i)) != 0
                        && mt.property_flags.contains(req_flags)
                    {
                        return Some(i as u32);
                    }
                }
                None
            };
            let Some(mem_type_index) = find_mem(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| find_mem(vk::MemoryPropertyFlags::empty()))
            else {
                raw_dev.destroy_image(vk_img, None);
                return None;
            };

            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(vk_img);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_info)
                .push_next(&mut dedicated_info);

            let vk_mem = match raw_dev.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(_) => {
                    raw_dev.destroy_image(vk_img, None);
                    return None;
                }
            };
            if raw_dev.bind_image_memory(vk_img, vk_mem, 0).is_err() {
                raw_dev.free_memory(vk_mem, None);
                raw_dev.destroy_image(vk_img, None);
                return None;
            }

            let subres = vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                mip_level: 0,
                array_layer: 0,
            };
            let layout = raw_dev.get_image_subresource_layout(vk_img, subres);
            let stride_u32 = u32::try_from(layout.row_pitch).ok()?;
            let offset_u32 = u32::try_from(layout.offset).ok()?;

            let ext_fd_loader = ash::khr::external_memory_fd::Device::new(raw_inst, raw_dev);
            let get_fd_info = vk::MemoryGetFdInfoKHR::default()
                .memory(vk_mem)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let raw_fd = match ext_fd_loader.get_memory_fd(&get_fd_info) {
                Ok(fd) => fd,
                Err(_) => {
                    raw_dev.free_memory(vk_mem, None);
                    raw_dev.destroy_image(vk_img, None);
                    return None;
                }
            };
            let owned_fd = OwnedFd::from_raw_fd(raw_fd);

            let hal_desc = wgpu::hal::TextureDescriptor {
                label: Some("exported_dmabuf_tex"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu::TextureUses::RESOURCE
                    | wgpu::TextureUses::COPY_SRC
                    | wgpu::TextureUses::COPY_DST,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let hal_texture = hal_dev.texture_from_raw(
                vk_img,
                &hal_desc,
                None,
                wgpu::hal::vulkan::TextureMemory::Dedicated(vk_mem),
            );
            (hal_texture, owned_fd, stride_u32, offset_u32)
        };

        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("exported_dmabuf_wgpu_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let texture = unsafe {
            engine
                .device()
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_tex,
                    &wgpu_desc,
                    wgpu::TextureUses::UNINITIALIZED,
                )
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Some((
            Self {
                engine,
                texture,
                view,
                width,
                height,
                stride,
                offset,
                bgra_format,
            },
            fd,
        ))
    }
}

/// Ingest bind groups kept per window. Each keeps its frame's texture (and dmabuf) alive,
/// and every live dmabuf makes each GPU submission slightly more expensive.
const INGEST_BG_CACHE_LEN: usize = 4;

/// Write `bytes` to `buffer` unless they are what was last written (`last`).
fn write_uniform_if_changed(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    last: &mut Vec<u8>,
    bytes: &[u8],
) {
    if last.as_slice() == bytes {
        return;
    }
    queue.write_buffer(buffer, 0, bytes);
    last.clear();
    last.extend_from_slice(bytes);
}

/// Build a Vulkan memory-visibility barrier command buffer (`MEMORY_WRITE -> SHADER_READ`)
/// so external GLES/EGL writes to a shared `DMA-BUF` invalidate Vulkan L0/L1/GL2 caches.
fn build_external_memory_barrier_cmd(device: &wgpu::Device) -> Option<wgpu::CommandBuffer> {
    #[cfg(unix)]
    unsafe {
        use ash::vk;
        let hal_dev = device.as_hal::<wgpu::hal::api::Vulkan>()?;
        let raw_dev = hal_dev.raw_device().clone();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ext_dmabuf_barrier_enc"),
        });
        let recorded = enc.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal_enc| {
            let hal_enc = hal_enc?;
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            raw_dev.cmd_pipeline_barrier(
                hal_enc.raw_handle(),
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
                &[],
                &[],
            );
            Some(())
        });
        recorded?;
        Some(enc.finish())
    }
    #[cfg(not(unix))]
    {
        let _ = device;
        None
    }
}

impl GpuTextureFrame {
    /// Read the cropped `GpuTextureFrame` back into tightly-packed CPU `BGRA` bytes.
    /// Used only for startup zero-copy probing and rare cold fallbacks (`Snapshot`, `Lz4Delta`, etc.).
    pub fn read_to_cpu_bgra(&self) -> Option<Vec<u8>> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let mut win = VramWindowState::new(Arc::clone(&self.engine));
        let _ = win.ingest_and_scan(self, true, false, false)?;
        win.download_cur_bgra()
    }
}

/// Per-window VRAM-resident state owned by `SmartTileEncoder`.
pub struct VramWindowState {
    engine: Arc<SmartTileGpuEngine>,
    width: u32,
    height: u32,
    frame_capacity: u64,
    cur_buf: Option<wgpu::Buffer>,
    ref_buf: Option<wgpu::Buffer>,
    prev_ref_buf: Option<wgpu::Buffer>,

    ingest_uniform: wgpu::Buffer,
    /// Bytes last written to `ingest_uniform`/`scroll_uniform`, so unchanged parameters
    /// aren't re-uploaded every frame (each `write_buffer` costs a staging buffer and an
    /// extra command buffer at submit).
    last_ingest_uniform: Vec<u8>,
    last_scroll_uniform: Vec<u8>,
    /// Ingest bind groups by frame view and bound buffers (see `INGEST_BG_CACHE_LEN`).
    ingest_bg_cache: Vec<(wgpu::TextureView, [wgpu::Buffer; 4], wgpu::BindGroup)>,
    scroll_bg_cache: Option<([wgpu::Buffer; 4], wgpu::BindGroup)>,
    scroll_uniform: wgpu::Buffer,
    prep_uniform: wgpu::Buffer,
    batch_uniform: wgpu::Buffer,
    sync_uniform: wgpu::Buffer,
    dummy_buf: wgpu::Buffer,

    tiles_capacity: usize,
    scan_buf: Option<wgpu::Buffer>,
    scan_staging: Option<wgpu::Buffer>,
    scan_host: Vec<u32>,

    scroll_capacity_u32: usize,
    scroll_buf: Option<wgpu::Buffer>,
    scroll_staging: Option<wgpu::Buffer>,
    scroll_host: Vec<u32>,

    cmd_capacity: usize,
    cmds_buf: Option<wgpu::Buffer>,
    mv_res_buf: Option<wgpu::Buffer>,
    mv_res_staging: Option<wgpu::Buffer>,

    lossy_capacity: usize,
    qblocks_byte_capacity: u64,
    descs_buf: Option<wgpu::Buffer>,
    qblocks_buf: Option<wgpu::Buffer>,
    qblocks_staging: Option<wgpu::Buffer>,

    extract_capacity: usize,
    extract_buf: Option<wgpu::Buffer>,
    extract_staging: Option<wgpu::Buffer>,
    fast_extract_host: Vec<u8>,

    rects_capacity: usize,
    rects_buf: Option<wgpu::Buffer>,
    scatter_buf: Option<wgpu::Buffer>,
}

impl VramWindowState {
    pub fn new(engine: Arc<SmartTileGpuEngine>) -> Self {
        let dev = engine.device();
        let mk_uni = |label: &str, size: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let dummy_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vram_dummy_buf"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            last_ingest_uniform: Vec::new(),
            last_scroll_uniform: Vec::new(),
            ingest_bg_cache: Vec::new(),
            scroll_bg_cache: None,
            ingest_uniform: mk_uni(
                "vram_ingest_uni",
                std::mem::size_of::<IngestParamsUniform>(),
            ),
            scroll_uniform: mk_uni(
                "vram_scroll_uni",
                std::mem::size_of::<ScrollParamsUniform>(),
            ),
            prep_uniform: mk_uni(
                "vram_prep_uni",
                std::mem::size_of::<VramPrepParamsUniform>(),
            ),
            batch_uniform: mk_uni(
                "vram_batch_uni",
                std::mem::size_of::<VramBatchParamsUniform>(),
            ),
            sync_uniform: mk_uni(
                "vram_sync_uni",
                std::mem::size_of::<TileSyncParamsUniform>(),
            ),
            dummy_buf,
            engine,
            width: 0,
            height: 0,
            frame_capacity: 0,
            cur_buf: None,
            ref_buf: None,
            prev_ref_buf: None,
            tiles_capacity: 0,
            scan_buf: None,
            scan_staging: None,
            scan_host: Vec::new(),
            scroll_capacity_u32: 0,
            scroll_buf: None,
            scroll_staging: None,
            scroll_host: Vec::new(),
            cmd_capacity: 0,
            cmds_buf: None,
            mv_res_buf: None,
            mv_res_staging: None,
            lossy_capacity: 0,
            qblocks_byte_capacity: 0,
            descs_buf: None,
            qblocks_buf: None,
            qblocks_staging: None,
            extract_capacity: 0,
            extract_buf: None,
            extract_staging: None,
            fast_extract_host: Vec::new(),
            rects_capacity: 0,
            rects_buf: None,
            scatter_buf: None,
        }
    }

    #[inline]
    pub fn engine(&self) -> &Arc<SmartTileGpuEngine> {
        &self.engine
    }

    pub fn ensure_dimensions(&mut self, width: u32, height: u32) -> bool {
        let Some(frame_bytes) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
        else {
            return false;
        };
        if frame_bytes == 0 {
            return false;
        }
        let cols = width.div_ceil(TILE_SIZE) as usize;
        let rows = height.div_ceil(TILE_SIZE) as usize;
        let total_tiles = cols * rows;
        if total_tiles == 0 || total_tiles > 65535 {
            return false;
        }

        if (width, height) != (self.width, self.height) {
            // Frames of the old size won't come back, and buffers may be reallocated below;
            // don't keep their textures and buffers alive in cached bind groups.
            self.ingest_bg_cache.clear();
            self.scroll_bg_cache = None;
        }

        let dev = self.engine.device();
        let frame_u64 = frame_bytes as u64;
        if self.frame_capacity < frame_u64 {
            let cap = frame_u64.next_power_of_two().max(4 * 1024 * 1024);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            self.cur_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_win_cur"),
                size: cap,
                usage,
                mapped_at_creation: false,
            }));
            self.ref_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_win_ref"),
                size: cap,
                usage,
                mapped_at_creation: false,
            }));
            self.prev_ref_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_win_prev_ref"),
                size: cap,
                usage,
                mapped_at_creation: false,
            }));
            self.frame_capacity = cap;
        }

        if self.tiles_capacity < total_tiles {
            let cap = total_tiles.next_power_of_two().max(256);
            let bytes = ((2 + cap * 2) * 4) as u64;
            self.scan_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_scan_buf"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.scan_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_scan_staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.tiles_capacity = cap;
        }

        let scroll_u32 = (width as usize + height as usize) * 4;
        if self.scroll_capacity_u32 < scroll_u32 {
            let cap = scroll_u32.next_power_of_two().max(4096);
            let bytes = (cap * 4) as u64;
            self.scroll_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_scroll_buf"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.scroll_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_scroll_staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.scroll_capacity_u32 = cap;
        }

        if self.cmd_capacity == 0 {
            let cap = 128usize;
            self.cmds_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_cmds_buf"),
                size: (cap * std::mem::size_of::<VramTileCmd>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.mv_res_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_mv_res_buf"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.mv_res_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_mv_res_staging"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cmd_capacity = cap;
        }

        if self.extract_capacity == 0 {
            let cap = 16usize;
            let ex_bytes = (cap * 8192 * 4) as u64;
            self.extract_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_buf"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.extract_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_staging"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.extract_capacity = cap;
        }

        self.width = width;
        self.height = height;
        true
    }

    /// Upload CPU `ref_bgra` into VRAM `ref_buf` (used only if switching from `wl_shm` to `dmabuf`).
    pub fn upload_ref_bgra(&mut self, width: u32, height: u32, ref_bgra: &[u8]) -> bool {
        if !self.ensure_dimensions(width, height) {
            return false;
        }
        let Some(ref_buf) = self.ref_buf.as_ref() else {
            return false;
        };
        self.engine.queue().write_buffer(ref_buf, 0, ref_bgra);
        true
    }

    fn download_vram_buffer(&self, src: &wgpu::Buffer) -> Option<Vec<u8>> {
        let bytes = (self.width as u64)
            .checked_mul(self.height as u64)?
            .checked_mul(4)?;
        let dev = self.engine.device();
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vram_dl_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_dl_enc"),
        });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.engine.queue().submit(Some(enc.finish()));
        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let out = slice.get_mapped_range().ok()?.to_vec();
        staging.unmap();
        Some(out)
    }

    pub fn download_cur_bgra(&self) -> Option<Vec<u8>> {
        self.download_vram_buffer(self.cur_buf.as_ref()?)
    }

    pub fn download_ref_bgra(&self) -> Option<Vec<u8>> {
        self.download_vram_buffer(self.ref_buf.as_ref()?)
    }

    /// Pass 1: Ingest `GpuTextureFrame` into `cur_buf`, scan dirty/opaque/solid tiles,
    /// optimistically extract up to `FAST_UI_MAX_TILES` non-solid UI tiles, and optionally
    /// compute scroll hashes/profiles in the same GPU submit.
    pub fn ingest_and_scan(
        &mut self,
        frame: &GpuTextureFrame,
        is_keyframe: bool,
        probe_scroll: bool,
        enable_fast_ui: bool,
    ) -> Option<(&[u32], Option<(i16, i16)>, &[u8])> {
        if !Arc::ptr_eq(&self.engine, &frame.engine) {
            *self = Self::new(Arc::clone(&frame.engine));
        }
        if !self.ensure_dimensions(frame.width, frame.height) {
            return None;
        }
        let cols = frame.width.div_ceil(TILE_SIZE);
        let rows = frame.height.div_ceil(TILE_SIZE);
        let total_tiles = (cols * rows) as usize;
        let do_scroll = probe_scroll && !is_keyframe && frame.width >= 192 && frame.height >= 192;
        let do_fast_ui = enable_fast_ui && !is_keyframe;

        let dev = self.engine.device();
        let queue = self.engine.queue();
        let pipes = self.engine.vram_pipelines();

        let ingest_flags = u32::from(is_keyframe) | (u32::from(do_fast_ui) << 1);
        let ingest_uni = IngestParamsUniform {
            fb_w: frame.width,
            fb_h: frame.height,
            cols,
            rows,
            crop_x: frame.crop_x,
            crop_y: frame.crop_y,
            rb_swizzle: u32::from(frame.rb_swizzle),
            is_keyframe: ingest_flags,
        };
        write_uniform_if_changed(
            queue,
            &self.ingest_uniform,
            &mut self.last_ingest_uniform,
            bytemuck::bytes_of(&ingest_uni),
        );

        let ref_buf = self.ref_buf.as_ref()?;
        let cur_buf = self.cur_buf.as_ref()?;
        let scan_buf = self.scan_buf.as_ref()?;
        let scan_staging = self.scan_staging.as_ref()?;
        let fast_ex_buf = if do_fast_ui {
            self.extract_buf.as_ref()?
        } else {
            &self.dummy_buf
        };

        let ingest_key = [
            ref_buf.clone(),
            cur_buf.clone(),
            scan_buf.clone(),
            fast_ex_buf.clone(),
        ];
        let cached_ingest = self
            .ingest_bg_cache
            .iter()
            .find(|(view, bufs, _)| *view == frame.view && *bufs == ingest_key)
            .map(|(_, _, bg)| bg.clone());
        let ingest_bg = if let Some(bg) = cached_ingest {
            bg
        } else {
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vram_ingest_bg"),
                layout: &pipes.ingest_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ingest_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&frame.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ref_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cur_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scan_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: fast_ex_buf.as_entire_binding(),
                    },
                ],
            });
            // Frames come from a small pool of slots, so their views repeat. Keep only a few:
            // a cached bind group keeps its slot's texture (and dmabuf) alive.
            if self.ingest_bg_cache.len() >= INGEST_BG_CACHE_LEN {
                self.ingest_bg_cache.remove(0);
            }
            self.ingest_bg_cache
                .push((frame.view.clone(), ingest_key, bg.clone()));
            bg
        };

        let scroll_bytes = ((frame.width as usize + frame.height as usize) * 16) as u64;
        let scroll_bg = if do_scroll {
            let s_uni = ScrollParamsUniform {
                fb_w: frame.width,
                fb_h: frame.height,
                _pad0: 0,
                _pad1: 0,
            };
            write_uniform_if_changed(
                queue,
                &self.scroll_uniform,
                &mut self.last_scroll_uniform,
                bytemuck::bytes_of(&s_uni),
            );
            let scroll_buf = self.scroll_buf.as_ref()?;
            let scroll_key = [
                ref_buf.clone(),
                cur_buf.clone(),
                scroll_buf.clone(),
                scan_buf.clone(),
            ];
            match &self.scroll_bg_cache {
                Some((key, bg)) if *key == scroll_key => Some(bg.clone()),
                _ => {
                    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vram_scroll_bg"),
                        layout: &pipes.scroll_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.scroll_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: ref_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cur_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: scroll_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: scan_buf.as_entire_binding(),
                            },
                        ],
                    });
                    self.scroll_bg_cache = Some((scroll_key, bg.clone()));
                    Some(bg)
                }
            }
        } else {
            None
        };

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_ingest_enc"),
        });
        enc.clear_buffer(scan_buf, 0, Some(8));
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vram_ingest_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipes.ingest_pipeline);
            cpass.set_bind_group(0, &ingest_bg, &[]);
            cpass.dispatch_workgroups(total_tiles as u32, 1, 1);

            if let Some(ref sbg) = scroll_bg {
                cpass.set_pipeline(&pipes.scroll_pipeline);
                cpass.set_bind_group(0, sbg, &[]);
                let wg = frame.width.max(frame.height).div_ceil(64);
                cpass.dispatch_workgroups(wg, 1, 1);
            }
        }

        let scan_bytes = ((2 + total_tiles * 2) * 4) as u64;
        let fast_ex_bytes = (FAST_UI_MAX_TILES * 8192 * 4) as u64;
        enc.copy_buffer_to_buffer(scan_buf, 0, scan_staging, 0, scan_bytes);
        if do_scroll {
            enc.copy_buffer_to_buffer(
                self.scroll_buf.as_ref()?,
                0,
                self.scroll_staging.as_ref()?,
                0,
                scroll_bytes,
            );
        }
        if do_fast_ui {
            enc.copy_buffer_to_buffer(
                self.extract_buf.as_ref()?,
                0,
                self.extract_staging.as_ref()?,
                0,
                fast_ex_bytes,
            );
        }

        if frame.external_memory {
            if let Some(barrier_cmd) = build_external_memory_barrier_cmd(dev) {
                queue.submit([barrier_cmd, enc.finish()]);
            } else {
                queue.submit(Some(enc.finish()));
            }
        } else {
            queue.submit(Some(enc.finish()));
        }

        let scan_slice = scan_staging.slice(0..scan_bytes);
        let (tx1, rx1) = std::sync::mpsc::channel();
        scan_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });

        let rx2 = if do_scroll {
            let scroll_slice = self.scroll_staging.as_ref()?.slice(0..scroll_bytes);
            let (tx2, rx2) = std::sync::mpsc::channel();
            scroll_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx2.send(r);
            });
            Some(rx2)
        } else {
            None
        };

        let rx3 = if do_fast_ui {
            let ex_slice = self.extract_staging.as_ref()?.slice(0..fast_ex_bytes);
            let (tx3, rx3) = std::sync::mpsc::channel();
            ex_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx3.send(r);
            });
            Some(rx3)
        } else {
            None
        };

        let _ = dev.poll(wgpu::PollType::wait_indefinitely());

        let unmap_pass1 =
            |s_scan: &wgpu::Buffer, s_scr: Option<&wgpu::Buffer>, s_ex: Option<&wgpu::Buffer>| {
                s_scan.unmap();
                if let Some(b) = s_scr {
                    b.unmap();
                }
                if let Some(b) = s_ex {
                    b.unmap();
                }
            };
        let scr_opt = if do_scroll {
            self.scroll_staging.as_ref()
        } else {
            None
        };
        let ex_opt = if do_fast_ui {
            self.extract_staging.as_ref()
        } else {
            None
        };

        if rx1.recv().ok().and_then(|r| r.ok()).is_none() {
            unmap_pass1(scan_staging, scr_opt, ex_opt);
            return None;
        }
        if let Some(rx2) = rx2 {
            if rx2.recv().ok().and_then(|r| r.ok()).is_none() {
                unmap_pass1(scan_staging, scr_opt, ex_opt);
                return None;
            }
        }
        if let Some(rx3) = rx3 {
            if rx3.recv().ok().and_then(|r| r.ok()).is_none() {
                unmap_pass1(scan_staging, scr_opt, ex_opt);
                return None;
            }
        }

        let (dirty_count, fast_count) = {
            let mapped = scan_slice.get_mapped_range().ok()?;
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            let dc = words[0] as usize;
            let fc = words[1] as usize;
            self.scan_host.clear();
            self.scan_host.extend_from_slice(&words[2..]);
            (dc, fc)
        };
        scan_staging.unmap();

        let detected_mv = if do_scroll {
            let scroll_staging = self.scroll_staging.as_ref()?;
            let mv = if dirty_count >= 12 {
                let mapped = scroll_staging
                    .slice(0..scroll_bytes)
                    .get_mapped_range()
                    .ok()?;
                let words: &[u32] = bytemuck::cast_slice(&mapped);
                self.scroll_host.clear();
                self.scroll_host.extend_from_slice(words);
                detect_scroll_mv_from_summaries(
                    &self.scroll_host,
                    frame.width as usize,
                    frame.height as usize,
                )
            } else {
                None
            };
            scroll_staging.unmap();
            mv
        } else {
            None
        };

        self.fast_extract_host.clear();
        if do_fast_ui {
            let ex_staging = self.extract_staging.as_ref()?;
            if dirty_count > 0 && dirty_count <= FAST_UI_MAX_TILES && fast_count > 0 {
                let copy_slots = fast_count.min(FAST_UI_MAX_TILES);
                let copy_bytes = (copy_slots * 8192 * 4) as u64;
                let mapped = ex_staging.slice(0..copy_bytes).get_mapped_range().ok()?;
                self.fast_extract_host.extend_from_slice(&mapped);
            }
            ex_staging.unmap();
        }

        Some((&self.scan_host, detected_mv, &self.fast_extract_host))
    }

    /// Pass 2: Snapshot `ref_buf -> prev_ref_buf` (if needed), run GPU MV skip + 1/2-pel search,
    /// encode lossy opaque tiles (Full/Half/Quarter-Res) + update `ref_buf` in VRAM, and read back
    /// only compact quantized coefficients and any sparse non-solid lossless/alpha tiles.
    pub fn execute_encode_pass<R, F>(
        &mut self,
        cmds: &[VramTileCmd],
        num_lossy_opaque: usize,
        num_extract: usize,
        scale: u32,
        frame_mv: Option<(i16, i16)>,
        enable_local_mv: bool,
        luma_q: &[i32; 64],
        chroma_q: &[i32; 64],
        consume: F,
    ) -> Option<R>
    where
        F: FnOnce(&[u32], &[i32], &[u8]) -> R,
    {
        if cmds.is_empty() {
            return Some(consume(&[], &[], &[]));
        }
        let dev = self.engine.device();
        let queue = self.engine.queue();
        let pipes = self.engine.vram_pipelines();

        if self.cmd_capacity < cmds.len() {
            let cap = cmds.len().next_power_of_two().max(128);
            self.cmds_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_cmds_buf"),
                size: (cap * std::mem::size_of::<VramTileCmd>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.mv_res_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_mv_res_buf"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.mv_res_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_mv_res_staging"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cmd_capacity = cap;
        }

        let words_per_tile = qblocks_words_per_tile(scale);
        let q_bytes = (num_lossy_opaque * words_per_tile * 4) as u64;

        if num_lossy_opaque > 0 && self.lossy_capacity < num_lossy_opaque {
            let cap = num_lossy_opaque.next_power_of_two().max(128);
            self.descs_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_descs_buf"),
                size: (cap * std::mem::size_of::<VramTileDesc>()) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.lossy_capacity = cap;
        }
        if q_bytes > 0 && self.qblocks_byte_capacity < q_bytes {
            let qb_cap = q_bytes.next_power_of_two().max(128 * 769 * 4);
            self.qblocks_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_qblocks_buf"),
                size: qb_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.qblocks_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_qblocks_staging"),
                size: qb_cap,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.qblocks_byte_capacity = qb_cap;
        }

        if num_extract > 0 && self.extract_capacity < num_extract {
            let cap = num_extract.next_power_of_two().max(16);
            let ex_bytes = (cap * 8192 * 4) as u64;
            self.extract_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_buf"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.extract_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_staging"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.extract_capacity = cap;
        }

        let (frame_dx, frame_dy, has_frame_mv) = match frame_mv {
            Some((dx, dy)) => (dx as i32, dy as i32, 1u32),
            None => (0, 0, 0),
        };
        let prep_uni = VramPrepParamsUniform {
            fb_w: self.width,
            fb_h: self.height,
            num_cmds: cmds.len() as u32,
            scale,
            frame_dx,
            frame_dy,
            has_frame_mv,
            enable_local_mv: u32::from(enable_local_mv),
        };
        queue.write_buffer(&self.prep_uniform, 0, bytemuck::bytes_of(&prep_uni));
        queue.write_buffer(self.cmds_buf.as_ref()?, 0, bytemuck::cast_slice(cmds));

        if num_lossy_opaque > 0 {
            let batch_uni = VramBatchParamsUniform {
                fb_w: self.width,
                fb_h: self.height,
                num_tiles: num_lossy_opaque as u32,
                scale,
                luma_q: *luma_q,
                chroma_q: *chroma_q,
            };
            queue.write_buffer(&self.batch_uniform, 0, bytemuck::bytes_of(&batch_uni));
        }

        let cur_buf = self.cur_buf.as_ref()?;
        let prev_ref_buf = self.prev_ref_buf.as_ref()?;
        let ref_buf = self.ref_buf.as_ref()?;
        let cmds_buf = self.cmds_buf.as_ref()?;
        let mv_res_buf = self.mv_res_buf.as_ref()?;
        let mv_res_staging = self.mv_res_staging.as_ref()?;
        let descs_binding_buf = if num_lossy_opaque > 0 {
            self.descs_buf.as_ref()?
        } else {
            &self.dummy_buf
        };
        let extract_binding_buf = if num_extract > 0 {
            self.extract_buf.as_ref()?
        } else {
            &self.dummy_buf
        };

        let prep_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vram_prep_bg"),
            layout: &pipes.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.prep_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cmds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cur_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prev_ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: descs_binding_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mv_res_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: extract_binding_buf.as_entire_binding(),
                },
            ],
        });

        let batch_bg = if num_lossy_opaque > 0 {
            Some(dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vram_batch_bg"),
                layout: &pipes.batch_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.batch_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.descs_buf.as_ref()?.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cur_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: prev_ref_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.qblocks_buf.as_ref()?.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ref_buf.as_entire_binding(),
                    },
                ],
            }))
        } else {
            None
        };

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_encode_enc"),
        });

        if has_frame_mv != 0 || enable_local_mv {
            let frame_bytes = (self.width as u64) * (self.height as u64) * 4;
            enc.copy_buffer_to_buffer(ref_buf, 0, prev_ref_buf, 0, frame_bytes);
        }

        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vram_encode_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipes.prep_pipeline);
            cpass.set_bind_group(0, &prep_bg, &[]);
            cpass.dispatch_workgroups(cmds.len() as u32, 1, 1);

            if let Some(ref bbg) = batch_bg {
                cpass.set_pipeline(&pipes.batch_pipeline);
                cpass.set_bind_group(0, bbg, &[]);
                cpass.dispatch_workgroups(num_lossy_opaque as u32, 1, 1);
            }
        }

        let mv_bytes = (cmds.len() * 4) as u64;
        let ex_bytes = (num_extract * 8192 * 4) as u64;

        enc.copy_buffer_to_buffer(mv_res_buf, 0, mv_res_staging, 0, mv_bytes);
        if q_bytes > 0 {
            enc.copy_buffer_to_buffer(
                self.qblocks_buf.as_ref()?,
                0,
                self.qblocks_staging.as_ref()?,
                0,
                q_bytes,
            );
        }
        if ex_bytes > 0 {
            enc.copy_buffer_to_buffer(
                self.extract_buf.as_ref()?,
                0,
                self.extract_staging.as_ref()?,
                0,
                ex_bytes,
            );
        }

        queue.submit(Some(enc.finish()));

        let mv_slice = mv_res_staging.slice(0..mv_bytes);
        let (tx_mv, rx_mv) = std::sync::mpsc::channel();
        mv_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_mv.send(r);
        });

        let rx_q = if q_bytes > 0 {
            let qs = self.qblocks_staging.as_ref()?.slice(0..q_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            qs.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            Some(rx)
        } else {
            None
        };

        let rx_ex = if ex_bytes > 0 {
            let es = self.extract_staging.as_ref()?.slice(0..ex_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            es.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            Some(rx)
        } else {
            None
        };

        let _ = dev.poll(wgpu::PollType::wait_indefinitely());

        let unmap_all =
            |s_mv: &wgpu::Buffer, s_q: Option<&wgpu::Buffer>, s_ex: Option<&wgpu::Buffer>| {
                s_mv.unmap();
                if let Some(b) = s_q {
                    b.unmap();
                }
                if let Some(b) = s_ex {
                    b.unmap();
                }
            };

        let q_stg_opt = if q_bytes > 0 {
            self.qblocks_staging.as_ref()
        } else {
            None
        };
        let ex_stg_opt = if ex_bytes > 0 {
            self.extract_staging.as_ref()
        } else {
            None
        };

        if rx_mv.recv().ok().and_then(|r| r.ok()).is_none() {
            unmap_all(mv_res_staging, q_stg_opt, ex_stg_opt);
            return None;
        }
        if let Some(rx) = rx_q {
            if rx.recv().ok().and_then(|r| r.ok()).is_none() {
                unmap_all(mv_res_staging, q_stg_opt, ex_stg_opt);
                return None;
            }
        }
        if let Some(rx) = rx_ex {
            if rx.recv().ok().and_then(|r| r.ok()).is_none() {
                unmap_all(mv_res_staging, q_stg_opt, ex_stg_opt);
                return None;
            }
        }

        let out = {
            let mv_map = mv_slice.get_mapped_range().ok()?;
            let mv_u32: &[u32] = bytemuck::cast_slice(&mv_map);
            let q_map_opt = if q_bytes > 0 {
                Some(
                    self.qblocks_staging
                        .as_ref()?
                        .slice(0..q_bytes)
                        .get_mapped_range()
                        .ok()?,
                )
            } else {
                None
            };
            let ex_map_opt = if ex_bytes > 0 {
                Some(
                    self.extract_staging
                        .as_ref()?
                        .slice(0..ex_bytes)
                        .get_mapped_range()
                        .ok()?,
                )
            } else {
                None
            };
            let q_i32: &[i32] = match q_map_opt.as_ref() {
                Some(m) => bytemuck::cast_slice(m),
                None => &[],
            };
            let ex_u8: &[u8] = match ex_map_opt.as_ref() {
                Some(m) => m,
                None => &[],
            };
            consume(mv_u32, q_i32, ex_u8)
        };
        unmap_all(mv_res_staging, q_stg_opt, ex_stg_opt);
        Some(out)
    }

    fn ensure_rects_buf(&mut self, count: usize) {
        if self.rects_capacity < count {
            let cap = count.next_power_of_two().max(64);
            self.rects_buf = Some(self.engine.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_rects_buf"),
                size: (cap * std::mem::size_of::<TileRect>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.rects_capacity = cap;
        }
    }

    /// Gather `(cur_tile, ref_tile)` pairs from VRAM for up to `rects.len()` lossy tiles during idle refresh.
    pub fn gather_refresh_tiles<R, F: FnOnce(&[u8]) -> R>(
        &mut self,
        rects: &[TileRect],
        consume: F,
    ) -> Option<R> {
        if rects.is_empty() {
            return None;
        }
        self.ensure_rects_buf(rects.len());
        let dev = self.engine.device();
        let queue = self.engine.queue();
        let pipes = self.engine.vram_pipelines();

        if self.extract_capacity < rects.len() {
            let cap = rects.len().next_power_of_two().max(64);
            let ex_bytes = (cap * 8192 * 4) as u64;
            self.extract_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_buf"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.extract_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_extract_staging"),
                size: ex_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.extract_capacity = cap;
        }

        let uni = TileSyncParamsUniform {
            fb_w: self.width,
            fb_h: self.height,
            num_rects: rects.len() as u32,
            mode: 0,
        };
        queue.write_buffer(&self.sync_uniform, 0, bytemuck::bytes_of(&uni));
        queue.write_buffer(self.rects_buf.as_ref()?, 0, bytemuck::cast_slice(rects));

        let extract_buf = self.extract_buf.as_ref()?;
        let extract_staging = self.extract_staging.as_ref()?;
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vram_gather_refresh_bg"),
            layout: &pipes.sync_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sync_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rects_buf.as_ref()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cur_buf.as_ref()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ref_buf.as_ref()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: extract_buf.as_entire_binding(),
                },
            ],
        });

        let bytes = (rects.len() * 8192 * 4) as u64;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_gather_refresh_enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vram_gather_refresh_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipes.sync_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(rects.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(extract_buf, 0, extract_staging, 0, bytes);
        queue.submit(Some(enc.finish()));

        let slice = extract_staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            extract_staging.unmap();
            return None;
        }
        let res = {
            let mapped = slice.get_mapped_range().ok()?;
            consume(&mapped)
        };
        extract_staging.unmap();
        Some(res)
    }

    /// Commit `cur_buf -> ref_buf` in VRAM for the `rects` tiles that were refined to lossless.
    pub fn commit_cur_to_ref_tiles(&mut self, rects: &[TileRect]) {
        if rects.is_empty() {
            return;
        }
        self.ensure_rects_buf(rects.len());
        let dev = self.engine.device();
        let queue = self.engine.queue();
        let pipes = self.engine.vram_pipelines();

        let uni = TileSyncParamsUniform {
            fb_w: self.width,
            fb_h: self.height,
            num_rects: rects.len() as u32,
            mode: 1,
        };
        queue.write_buffer(&self.sync_uniform, 0, bytemuck::bytes_of(&uni));
        let Some(rects_buf) = self.rects_buf.as_ref() else {
            return;
        };
        let (Some(cur_buf), Some(ref_buf)) = (self.cur_buf.as_ref(), self.ref_buf.as_ref()) else {
            return;
        };
        queue.write_buffer(rects_buf, 0, bytemuck::cast_slice(rects));

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vram_commit_bg"),
            layout: &pipes.sync_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sync_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rects_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cur_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dummy_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_commit_enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vram_commit_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipes.sync_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(rects.len() as u32, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }

    /// Upload reconstructed pixels for rare semi-transparent alpha tiles back into `ref_buf` in VRAM.
    pub fn scatter_recon_tiles(&mut self, rects: &[TileRect], recon_tiles_bgra: &[u8]) {
        if rects.is_empty() {
            return;
        }
        self.ensure_rects_buf(rects.len());
        let dev = self.engine.device();
        let queue = self.engine.queue();
        let pipes = self.engine.vram_pipelines();

        let needed = (rects.len() * 4096 * 4) as u64;
        if self.scatter_buf.as_ref().is_none_or(|b| b.size() < needed) {
            self.scatter_buf = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram_scatter_buf"),
                size: needed.next_power_of_two().max(65536),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let uni = TileSyncParamsUniform {
            fb_w: self.width,
            fb_h: self.height,
            num_rects: rects.len() as u32,
            mode: 2,
        };
        let (Some(rects_buf), Some(cur_buf), Some(ref_buf), Some(scatter_buf)) = (
            self.rects_buf.as_ref(),
            self.cur_buf.as_ref(),
            self.ref_buf.as_ref(),
            self.scatter_buf.as_ref(),
        ) else {
            return;
        };
        queue.write_buffer(&self.sync_uniform, 0, bytemuck::bytes_of(&uni));
        queue.write_buffer(rects_buf, 0, bytemuck::cast_slice(rects));
        queue.write_buffer(scatter_buf, 0, recon_tiles_bgra);

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vram_scatter_bg"),
            layout: &pipes.sync_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sync_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rects_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cur_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scatter_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vram_scatter_enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vram_scatter_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipes.sync_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(rects.len() as u32, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
}

/// Detect dominant vertical `(0, dy)` or horizontal `(dx, 0)` scroll MV from the GPU summary buffer.
fn detect_scroll_mv_from_summaries(words: &[u32], w: usize, h: usize) -> Option<(i16, i16)> {
    if words.len() < (h + w) * 4 {
        return None;
    }
    if let Some(dy) = detect_1d_scroll_shift(&words[0..h * 4], h) {
        if let Ok(dy16) = i16::try_from(dy) {
            return Some((0, dy16));
        }
    }
    if let Some(dx) = detect_1d_scroll_shift(&words[h * 4..(h + w) * 4], w) {
        if let Ok(dx16) = i16::try_from(dx) {
            return Some((dx16, 0));
        }
    }
    None
}

fn unpack_prof(p: u32) -> [i32; 4] {
    [
        (p & 0xFF) as i32,
        ((p >> 8) & 0xFF) as i32,
        ((p >> 16) & 0xFF) as i32,
        ((p >> 24) & 0xFF) as i32,
    ]
}

fn prof_sad(a: &[i32; 4], b: &[i32; 4]) -> i32 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() + (a[3] - b[3]).abs()
}

fn detect_1d_scroll_shift(summaries: &[u32], len: usize) -> Option<isize> {
    // 1. Exact hash vote pass
    let mut prev_map: HashMap<u32, usize> = HashMap::with_capacity(len);
    for i in 0..len {
        let ph = summaries[i * 4];
        if (ph & 1) != 0 {
            prev_map.entry(ph & !1).or_insert(i);
        }
    }
    let max_d = (len / 2) as isize;
    let mut votes: HashMap<isize, usize> = HashMap::new();
    for i in 0..len {
        let ph = summaries[i * 4];
        let ch = summaries[i * 4 + 1];
        if (ch & 1) == 0 || (ch & !1) == (ph & !1) {
            continue;
        }
        if let Some(&pi) = prev_map.get(&(ch & !1)) {
            let d = i as isize - pi as isize;
            if d != 0 && d.abs() <= max_d {
                *votes.entry(d).or_insert(0) += 1;
            }
        }
    }
    if let Some((best_d, best_v)) = votes.into_iter().max_by_key(|&(_, v)| v) {
        if best_v >= 24 {
            // Confirm contiguous run >= 48 lines
            let mut max_run = 0usize;
            let mut run = 0usize;
            for i in 0..len {
                let pi = i as isize - best_d;
                if pi >= 0 && (pi as usize) < len {
                    let ph = summaries[(pi as usize) * 4] & !1;
                    let ch = summaries[i * 4 + 1] & !1;
                    if ph == ch {
                        run += 1;
                        max_run = max_run.max(run);
                    } else {
                        run = 0;
                    }
                } else {
                    run = 0;
                }
            }
            if max_run >= 48 {
                return Some(best_d);
            }
        }
    }

    // 2. Fuzzy 4-segment luma profile SAD pass (smooth / fractional scrolling)
    let max_fuzzy = (len / 3).min(240) as isize;
    let lo = (len / 8).max(16);
    let hi = len.saturating_sub(lo);
    if hi <= lo + 64 {
        return None;
    }

    let mut base_sad: i64 = 0;
    let mut changed = 0usize;
    for i in (lo..hi).step_by(2) {
        let pp = unpack_prof(summaries[i * 4 + 2]);
        let cp = unpack_prof(summaries[i * 4 + 3]);
        let d = prof_sad(&cp, &pp);
        base_sad += d as i64;
        if d > 6 {
            changed += 1;
        }
    }
    if changed < 16 || base_sad < 200 {
        return None;
    }

    let mut best_d: isize = 0;
    let mut best_sad: i64 = i64::MAX;
    for d in (-max_fuzzy..=max_fuzzy).step_by(2) {
        if d == 0 {
            continue;
        }
        let mut sad = 0i64;
        let mut cnt = 0i64;
        for i in (lo..hi).step_by(4) {
            let pi = i as isize - d;
            if pi >= 0 && (pi as usize) < len {
                let pp = unpack_prof(summaries[(pi as usize) * 4 + 2]);
                let cp = unpack_prof(summaries[i * 4 + 3]);
                sad += prof_sad(&cp, &pp) as i64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            let norm = sad * ((hi - lo) as i64 / 2) / cnt;
            if norm < best_sad {
                best_sad = norm;
                best_d = d;
            }
        }
    }

    let center = best_d;
    for d in (center - 2).max(-max_fuzzy)..=(center + 2).min(max_fuzzy) {
        if d == 0 {
            continue;
        }
        let mut sad = 0i64;
        let mut cnt = 0i64;
        for i in (lo..hi).step_by(2) {
            let pi = i as isize - d;
            if pi >= 0 && (pi as usize) < len {
                let pp = unpack_prof(summaries[(pi as usize) * 4 + 2]);
                let cp = unpack_prof(summaries[i * 4 + 3]);
                sad += prof_sad(&cp, &pp) as i64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            let norm = sad * ((hi - lo) as i64 / 2) / cnt;
            if norm < best_sad {
                best_sad = norm;
                best_d = d;
            }
        }
    }

    if best_d == 0 || best_sad * 100 > base_sad * 35 {
        return None;
    }

    let mut best_run = 0usize;
    let mut run = 0usize;
    for i in 0..len {
        let pi = i as isize - best_d;
        let ok = if pi >= 0 && (pi as usize) < len {
            let pp_s = unpack_prof(summaries[(pi as usize) * 4 + 2]);
            let pp_z = unpack_prof(summaries[i * 4 + 2]);
            let cp = unpack_prof(summaries[i * 4 + 3]);
            let ds = prof_sad(&cp, &pp_s);
            let dz = prof_sad(&cp, &pp_z);
            ds <= 12 && (ds <= dz || dz <= 2)
        } else {
            false
        };
        if ok {
            run += 1;
            best_run = best_run.max(run);
        } else {
            run = 0;
        }
    }
    if best_run >= 64 {
        let _ = CopyRectOp {
            dst_x: 0,
            dst_y: 0,
            w: 0,
            h: 0,
            src_x: 0,
            src_y: 0,
        };
        let _ = (MODE_FLAG_MV, MODE_FLAG_LOCAL_MV);
        Some(best_d)
    } else {
        None
    }
}
