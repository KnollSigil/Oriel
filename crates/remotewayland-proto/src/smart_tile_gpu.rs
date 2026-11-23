//! Hardware GPU Compute Shader (`wgpu` / Vulkan) Acceleration for `SmartTile`.
//!
//! Offloads parallel `SmartTile` compute stages to a hardware Vulkan/GLES GPU via WGSL compute shaders:
//! 1. **Dirty-tile detection & opacity classification**: compares 64x64 tiles between `ref_bgra` and `cur_bgra`
//!    in parallel across workgroups, outputting a compact per-tile bitmask (`bit 0 = dirty`, `bit 1 = opaque`).
//! 2. **Batched 64x64 `BGRA` -> `YCbCr 4:2:0` + 2D 8x8 forward `DCT` + Quantization + Dequantization +
//!    2D 8x8 inverse `IDCT` + `YCbCr 4:2:0` -> `BGRA` reference reconstruction**:
//!    Executes the exact same 32-bit fixed-point integer arithmetic (`C1..C7`, `>> 11`, `>> 17`, BT.601 `>> 8`)
//!    as the CPU scalar and AVX2/NEON implementations so encoder and decoder reference buffers remain
//!    **100% bit-identical** with zero drift.
//! 3. **Batched 8x8 inverse `IDCT` + `YCbCr 4:2:0` -> `BGRA` tile decoding** on the client.
//!
//! Automatically selects a hardware discrete or integrated GPU (`DeviceType::DiscreteGpu` or
//! `DeviceType::IntegratedGpu`), and gracefully falls back to the AVX2/NEON SIMD path on headless
//! systems with only software `llvmpipe` or when tile counts are small.

#![allow(clippy::too_many_arguments)]

use std::sync::{Mutex, OnceLock};

use bytemuck::{Pod, Zeroable};

/// Minimum number of dirty lossy tiles in a frame to dispatch the batched GPU DCT/IDCT pipeline
/// (amortizes PCIe/DMA transfer overhead over parallel shader execution).
pub const GPU_MIN_BATCH_TILES: usize = 32;

/// WGSL compute shader implementing bit-exact `SmartTile` integer pipelines.
const SMART_TILE_WGSL: &str = r#"
struct GridParams {
    fb_w: u32,
    fb_h: u32,
    cols: u32,
    rows: u32,
    num_tiles: u32,
    is_inter: u32,
    _pad0: u32,
    _pad1: u32,
}

struct TileDesc {
    x0: u32,
    y0: u32,
    tw: u32,
    th: u32,
    pred_x: u32,
    pred_y: u32,
    has_alpha: u32,
    use_inter: u32,
}

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

// ---------------------------------------------------------------------------
// Kernel 1: Parallel 64x64 Dirty-Tile & Opacity Scanner
// Workgroup size: 64 threads per tile. Each thread scans 64 pixels (one row).
// Output flags per tile: bit 0 = dirty, bit 1 = opaque.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> scan_params: GridParams;
@group(0) @binding(1) var<storage, read> scan_ref_bgra: array<u32>;
@group(0) @binding(2) var<storage, read> scan_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read_write> scan_tile_flags: array<u32>;

var<workgroup> wg_dirty: atomic<u32>;
var<workgroup> wg_non_opaque: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn scan_dirty_tiles(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tidx = wid.x;
    if (tidx >= scan_params.cols * scan_params.rows) {
        return;
    }
    if (lid.x == 0u) {
        atomicStore(&wg_dirty, 0u);
        atomicStore(&wg_non_opaque, 0u);
    }
    workgroupBarrier();

    let tx = tidx % scan_params.cols;
    let ty = tidx / scan_params.cols;
    let x0 = tx * 64u;
    let y0 = ty * 64u;
    let tw = min(scan_params.fb_w - x0, 64u);
    let th = min(scan_params.fb_h - y0, 64u);

    let py = lid.x;
    if (py < th) {
        let row_base = (y0 + py) * scan_params.fb_w + x0;
        var row_dirty = 0u;
        var row_non_opaque = 0u;
        for (var px = 0u; px < tw; px = px + 1u) {
            let idx = row_base + px;
            let cur_px = scan_cur_bgra[idx];
            let ref_px = scan_ref_bgra[idx];
            if (cur_px != ref_px) {
                row_dirty = 1u;
            }
            if ((cur_px & 0xFF000000u) != 0xFF000000u) {
                row_non_opaque = 1u;
            }
        }
        if (row_dirty != 0u) {
            atomicOr(&wg_dirty, 1u);
        }
        if (row_non_opaque != 0u) {
            atomicOr(&wg_non_opaque, 1u);
        }
    }
    workgroupBarrier();

    if (lid.x == 0u) {
        let d = atomicLoad(&wg_dirty);
        let opaque = select(2u, 0u, atomicLoad(&wg_non_opaque) != 0u);
        scan_tile_flags[tidx] = d | opaque;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Batched Tile 2D 8x8 DCT + Quantize + Dequantize + IDCT
// Each workgroup of 64 threads (8x8) processes one 8x8 block within a plane batch.
// Shared workgroup memory holds the 8x8 block for conflict-free row & column butterflies.
// ---------------------------------------------------------------------------

struct DctBatchParams {
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    quant: array<vec4<i32>, 16>,
}

@group(0) @binding(0) var<uniform> dct_params: DctBatchParams;
@group(0) @binding(1) var<storage, read> dct_in_blocks: array<i32>;
@group(0) @binding(2) var<storage, read_write> dct_out_qblocks: array<i32>;
@group(0) @binding(3) var<storage, read_write> dct_out_idct: array<i32>;

var<workgroup> wg_blk: array<i32, 64>;
var<workgroup> wg_tmp: array<i32, 64>;

fn get_quant(idx: u32) -> i32 {
    let v = dct_params.quant[idx >> 2u];
    let lane = idx & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn dct_1d_elem(x0: i32, x1: i32, x2: i32, x3: i32, x4: i32, x5: i32, x6: i32, x7: i32, k: u32, shift: u32) -> i32 {
    let round = 1 << (shift - 1u);
    let s0 = x0 + x7;
    let s1 = x1 + x6;
    let s2 = x2 + x5;
    let s3 = x3 + x4;
    let d0 = x0 - x7;
    let d1 = x1 - x6;
    let d2 = x2 - x5;
    let d3 = x3 - x4;

    var sum = 0;
    switch (k) {
        case 0u: { sum = C4 * (s0 + s1 + s2 + s3); }
        case 1u: { sum = C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3; }
        case 2u: { sum = C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3; }
        case 3u: { sum = C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3; }
        case 4u: { sum = C4 * (s0 - s1 - s2 + s3); }
        case 5u: { sum = C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3; }
        case 6u: { sum = C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3; }
        default: { sum = C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3; }
    }
    return (sum + round) >> shift;
}

fn idct_1d_elem(x0: i32, x1: i32, x2: i32, x3: i32, x4: i32, x5: i32, x6: i32, x7: i32, k: u32, shift: u32) -> i32 {
    let round = 1 << (shift - 1u);
    let a0 = C4 * (x0 + x4);
    let a1 = C4 * (x0 - x4);
    let b0 = C2 * x2 + C6 * x6;
    let b1 = C6 * x2 - C2 * x6;

    let e0 = a0 + b0;
    let e1 = a1 + b1;
    let e2 = a1 - b1;
    let e3 = a0 - b0;

    let o0 = C1 * x1 + C3 * x3 + C5 * x5 + C7 * x7;
    let o1 = C3 * x1 - C7 * x3 - C1 * x5 - C5 * x7;
    let o2 = C5 * x1 - C1 * x3 + C7 * x5 + C3 * x7;
    let o3 = C7 * x1 - C5 * x3 + C3 * x5 - C1 * x7;

    var sum = 0;
    switch (k) {
        case 0u: { sum = e0 + o0; }
        case 1u: { sum = e1 + o1; }
        case 2u: { sum = e2 + o2; }
        case 3u: { sum = e3 + o3; }
        case 4u: { sum = e3 - o3; }
        case 5u: { sum = e2 - o2; }
        case 6u: { sum = e1 - o1; }
        default: { sum = e0 - o0; }
    }
    return (sum + round) >> shift;
}

// ---------------------------------------------------------------------------
// Kernel 4: Batched Full-Res 64x64 Tile Encoder Pipeline (Y + Cb + Cr)
// Workgroup size: 64 threads (8x8). Each workgroup processes ONE 64x64 tile:
// 1. Extracts 64x64 BGRA from `cur_bgra` (and `pred_bgra` if `use_inter != 0`),
//    converts to BT.601 YCbCr 4:2:0, and computes the 64x64 Y + 32x32 Cb + 32x32 Cr residuals.
// 2. Loops over all 80 8x8 DCT blocks (64 Y + 16 Cb + 16 Cr):
//    - Forward 2D 8x8 DCT -> Quantize -> store to `tile_qblocks[tidx * 96 + blk]`
//    - Dequantize -> Inverse 2D 8x8 IDCT -> Reconstruct clamped [0..255] Y, Cb, Cr planes
// 3. Converts reconstructed 64x64 Y + 32x32 Cb/Cr back to `recon_bgra` in-place!
// ---------------------------------------------------------------------------

struct TileBatchParams {
    fb_w: u32,
    fb_h: u32,
    num_tiles: u32,
    _pad0: u32,
    luma_q: array<vec4<i32>, 16>,
    chroma_q: array<vec4<i32>, 16>,
}

@group(0) @binding(0) var<uniform> tb_params: TileBatchParams;
@group(0) @binding(1) var<storage, read> tb_descs: array<TileDesc>;
@group(0) @binding(2) var<storage, read> tb_cur_bgra: array<u32>;
@group(0) @binding(3) var<storage, read> tb_pred_bgra: array<u32>;
@group(0) @binding(4) var<storage, read_write> tb_qblocks: array<i32>;
@group(0) @binding(5) var<storage, read_write> tb_recon_bgra: array<u32>;

// Pack two 16-bit signed integers `[-255, 255]` into one `i32` (`low 16 bits = v0`, `high 16 bits = v1`).
fn pack_i16x2(v0: i32, v1: i32) -> i32 {
    return bitcast<i32>((u32(v0) & 0xFFFFu) | ((u32(v1) & 0xFFFFu) << 16u));
}

fn unpack_i16_lo(packed: i32) -> i32 {
    return (packed << 16u) >> 16u;
}

fn unpack_i16_hi(packed: i32) -> i32 {
    return packed >> 16u;
}

// Total workgroup shared memory:
// - `wg_y_packed`: 4096 i32s (16 KB) -> low 16 bits = cur_y, high 16 bits = ref_y
// - `wg_c_packed`: 1024 i32s (4 KB)  -> byte 0 = cur_cb, byte 1 = cur_cr, byte 2 = ref_cb, byte 3 = ref_cr
// - `wg_multi_blk` + `wg_multi_tmp` + `wg_multi_ref`: 3 * 512 i32s (6 KB)
// - `wg_inter_sum` + `wg_inter_sad` + `wg_intra_sad`: 3 * 64 i32s (0.75 KB)
// Total = 26.75 KB -> fits 2 workgroups per RDNA2 LDS (64 KB), doubling CU occupancy!
var<workgroup> wg_y_packed: array<i32, 4096>;
var<workgroup> wg_c_packed: array<u32, 1024>;
var<workgroup> wg_inter_sum: array<i32, 64>;
var<workgroup> wg_inter_sad: array<i32, 64>;
var<workgroup> wg_intra_sad: array<i32, 64>;
var<workgroup> wg_use_inter: u32;
var<workgroup> wg_multi_blk: array<i32, 512>;
var<workgroup> wg_multi_tmp: array<i32, 512>;
var<workgroup> wg_multi_ref: array<i32, 512>;

fn get_luma_q(idx: u32) -> i32 {
    let v = tb_params.luma_q[idx >> 2u];
    let lane = idx & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn get_chroma_q(idx: u32) -> i32 {
    let v = tb_params.chroma_q[idx >> 2u];
    let lane = idx & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

@compute @workgroup_size(64, 1, 1)
fn encode_fullres_tiles_batch(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tidx = wid.x;
    if (tidx >= tb_params.num_tiles) {
        return;
    }
    let desc = tb_descs[tidx];
    let lane = lid.x;
    let max_x = desc.tw - 1u;
    let max_y = desc.th - 1u;

    // 1. Extract 64x64 Y and 32x32 Cb/Cr from cur_bgra & pred_bgra in a single pass!
    var my_sum_y = 0;
    var my_inter_sad = 0;

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

        let p00 = tb_cur_bgra[sy0 * tb_params.fb_w + sx0];
        let p01 = tb_cur_bgra[sy0 * tb_params.fb_w + sx1];
        let p10 = tb_cur_bgra[sy1 * tb_params.fb_w + sx0];
        let p11 = tb_cur_bgra[sy1 * tb_params.fb_w + sx1];

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

        let rsy0 = desc.pred_y + min(py0, max_y);
        let rsy1 = desc.pred_y + min(py1, max_y);
        let rsx0 = desc.pred_x + min(px0, max_x);
        let rsx1 = desc.pred_x + min(px1, max_x);
        let rp00 = tb_pred_bgra[rsy0 * tb_params.fb_w + rsx0];
        let rp01 = tb_pred_bgra[rsy0 * tb_params.fb_w + rsx1];
        let rp10 = tb_pred_bgra[rsy1 * tb_params.fb_w + rsx0];
        let rp11 = tb_pred_bgra[rsy1 * tb_params.fb_w + rsx1];

        let rb00 = i32(rp00 & 0xFFu); let rg00 = i32((rp00 >> 8u) & 0xFFu); let rr00 = i32((rp00 >> 16u) & 0xFFu);
        let rb01 = i32(rp01 & 0xFFu); let rg01 = i32((rp01 >> 8u) & 0xFFu); let rr01 = i32((rp01 >> 16u) & 0xFFu);
        let rb10 = i32(rp10 & 0xFFu); let rg10 = i32((rp10 >> 8u) & 0xFFu); let rr10 = i32((rp10 >> 16u) & 0xFFu);
        let rb11 = i32(rp11 & 0xFFu); let rg11 = i32((rp11 >> 8u) & 0xFFu); let rr11 = i32((rp11 >> 16u) & 0xFFu);

        let ry00 = rgb_to_y(rr00, rg00, rb00);
        let ry01 = rgb_to_y(rr01, rg01, rb01);
        let ry10 = rgb_to_y(rr10, rg10, rb10);
        let ry11 = rgb_to_y(rr11, rg11, rb11);

        wg_y_packed[py0 * 64u + px0] = pack_i16x2(y00, ry00);
        wg_y_packed[py0 * 64u + px1] = pack_i16x2(y01, ry01);
        wg_y_packed[py1 * 64u + px0] = pack_i16x2(y10, ry10);
        wg_y_packed[py1 * 64u + px1] = pack_i16x2(y11, ry11);

        let cb_ref = (rgb_to_cb(rr00, rg00, rb00) + rgb_to_cb(rr01, rg01, rb01) + rgb_to_cb(rr10, rg10, rb10) + rgb_to_cb(rr11, rg11, rb11) + 2) >> 2u;
        let cr_ref = (rgb_to_cr(rr00, rg00, rb00) + rgb_to_cr(rr01, rg01, rb01) + rgb_to_cr(rr10, rg10, rb10) + rgb_to_cr(rr11, rg11, rb11) + 2) >> 2u;
        wg_c_packed[qidx] = u32(cb_cur) | (u32(cr_cur) << 8u) | (u32(cb_ref) << 16u) | (u32(cr_ref) << 24u);

        if ((cx & 1u) == 0u) {
            my_sum_y = my_sum_y + y00 + y10;
            my_inter_sad = my_inter_sad + abs(y00 - ry00) + abs(y10 - ry10);
        }
    }
    wg_inter_sum[lane] = my_sum_y;
    wg_inter_sad[lane] = my_inter_sad;
    workgroupBarrier();

    // Reduce sum_y across 64 threads using a 6-step parallel tree reduction.
    for (var stride_r = 32u; stride_r > 0u; stride_r = stride_r >> 1u) {
        if (lane < stride_r) {
            wg_inter_sum[lane] = wg_inter_sum[lane] + wg_inter_sum[lane + stride_r];
            wg_inter_sad[lane] = wg_inter_sad[lane] + wg_inter_sad[lane + stride_r];
        }
        workgroupBarrier();
    }
    let mean_y = wg_inter_sum[0] / 1024;
    let total_inter_sad = wg_inter_sad[0];

    var my_intra_sad = 0;
    for (var sidx = lane; sidx < 1024u; sidx = sidx + 64u) {
        let cur_y_val = unpack_i16_lo(wg_y_packed[sidx * 4u]);
        my_intra_sad = my_intra_sad + abs(cur_y_val - mean_y);
    }
    wg_intra_sad[lane] = my_intra_sad;
    workgroupBarrier();

    for (var stride_r = 32u; stride_r > 0u; stride_r = stride_r >> 1u) {
        if (lane < stride_r) {
            wg_intra_sad[lane] = wg_intra_sad[lane] + wg_intra_sad[lane + stride_r];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let total_intra_sad = wg_intra_sad[0];
        let use_inter = select(0u, 1u, (desc.use_inter != 0u) && (total_inter_sad * 10 <= total_intra_sad * 9));
        wg_use_inter = use_inter;
    }
    workgroupBarrier();

    let is_inter = wg_use_inter != 0u;
    let luma_bx = (desc.tw + 7u) >> 3u;
    let luma_by = (desc.th + 7u) >> 3u;
    let chroma_w = (desc.tw + 1u) >> 1u;
    let chroma_h = (desc.th + 1u) >> 1u;
    let chroma_bx = (chroma_w + 7u) >> 3u;
    let chroma_by = (chroma_h + 7u) >> 3u;
    let num_luma = luma_bx * luma_by;
    let num_chroma = chroma_bx * chroma_by;
    let total_active_blocks = num_luma + 2u * num_chroma;

    let qout_tile_base = tidx * 3073u;
    if (lane == 0u) {
        tb_qblocks[qout_tile_base] = select(0, 1, is_inter);
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
                    let p_idx = (by * 8u + br) * 64u + (bx * 8u + bc);
                    let yp = wg_y_packed[p_idx];
                    let cur_y = unpack_i16_lo(yp);
                    let rv = select(128, unpack_i16_hi(yp), is_inter);
                    wg_multi_ref[elem] = rv;
                    wg_multi_blk[elem] = cur_y - rv;
                } else {
                    let c_seq = select(seq_blk - num_luma - num_chroma, seq_blk - num_luma, is_cb);
                    let by = c_seq / chroma_bx;
                    let bx = c_seq % chroma_bx;
                    let p_idx = (by * 8u + br) * 32u + (bx * 8u + bc);
                    let cp = wg_c_packed[p_idx];
                    let cur_c = i32(select((cp >> 8u) & 0xFFu, cp & 0xFFu, is_cb));
                    let ref_c = i32(select((cp >> 24u) & 0xFFu, (cp >> 16u) & 0xFFu, is_cb));
                    let rv = select(128, ref_c, is_inter);
                    wg_multi_ref[elem] = rv;
                    wg_multi_blk[elem] = cur_c - rv;
                }
            }
        }
        workgroupBarrier();

        // Forward 1D DCT along rows (shift = 11): 64 threads = 8 blocks * 8 rows/block!
        let local_blk = lane >> 3u;
        let line_idx = lane & 7u;
        let seq_blk_lane = b_batch * 8u + local_blk;
        let is_y_lane = seq_blk_lane < num_luma;
        let is_cb_lane = !is_y_lane && (seq_blk_lane < num_luma + num_chroma);

        if (seq_blk_lane < total_active_blocks) {
            let ro = local_blk * 64u + line_idx * 8u;
            let x0 = wg_multi_blk[ro];
            let x1 = wg_multi_blk[ro + 1u];
            let x2 = wg_multi_blk[ro + 2u];
            let x3 = wg_multi_blk[ro + 3u];
            let x4 = wg_multi_blk[ro + 4u];
            let x5 = wg_multi_blk[ro + 5u];
            let x6 = wg_multi_blk[ro + 6u];
            let x7 = wg_multi_blk[ro + 7u];

            let s0 = x0 + x7;
            let s1 = x1 + x6;
            let s2 = x2 + x5;
            let s3 = x3 + x4;
            let d0 = x0 - x7;
            let d1 = x1 - x6;
            let d2 = x2 - x5;
            let d3 = x3 - x4;

            wg_multi_tmp[ro]      = (C4 * (s0 + s1 + s2 + s3) + 1024) >> 11u;
            wg_multi_tmp[ro + 1u] = (C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3 + 1024) >> 11u;
            wg_multi_tmp[ro + 2u] = (C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3 + 1024) >> 11u;
            wg_multi_tmp[ro + 3u] = (C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3 + 1024) >> 11u;
            wg_multi_tmp[ro + 4u] = (C4 * (s0 - s1 - s2 + s3) + 1024) >> 11u;
            wg_multi_tmp[ro + 5u] = (C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3 + 1024) >> 11u;
            wg_multi_tmp[ro + 6u] = (C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3 + 1024) >> 11u;
            wg_multi_tmp[ro + 7u] = (C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3 + 1024) >> 11u;
        }
        workgroupBarrier();

        // Forward 1D DCT along columns (shift = 17): 64 threads = 8 blocks * 8 columns/block!
        if (seq_blk_lane < total_active_blocks) {
            let co = local_blk * 64u + line_idx;
            let x0 = wg_multi_tmp[co];
            let x1 = wg_multi_tmp[co + 8u];
            let x2 = wg_multi_tmp[co + 16u];
            let x3 = wg_multi_tmp[co + 24u];
            let x4 = wg_multi_tmp[co + 32u];
            let x5 = wg_multi_tmp[co + 40u];
            let x6 = wg_multi_tmp[co + 48u];
            let x7 = wg_multi_tmp[co + 56u];

            let s0 = x0 + x7;
            let s1 = x1 + x6;
            let s2 = x2 + x5;
            let s3 = x3 + x4;
            let d0 = x0 - x7;
            let d1 = x1 - x6;
            let d2 = x2 - x5;
            let d3 = x3 - x4;

            wg_multi_blk[co]       = (C4 * (s0 + s1 + s2 + s3) + 65536) >> 17u;
            wg_multi_blk[co + 8u]  = (C1 * d0 + C3 * d1 + C5 * d2 + C7 * d3 + 65536) >> 17u;
            wg_multi_blk[co + 16u] = (C2 * s0 + C6 * s1 - C6 * s2 - C2 * s3 + 65536) >> 17u;
            wg_multi_blk[co + 24u] = (C3 * d0 - C7 * d1 - C1 * d2 - C5 * d3 + 65536) >> 17u;
            wg_multi_blk[co + 32u] = (C4 * (s0 - s1 - s2 + s3) + 65536) >> 17u;
            wg_multi_blk[co + 40u] = (C5 * d0 - C1 * d1 + C7 * d2 + C3 * d3 + 65536) >> 17u;
            wg_multi_blk[co + 48u] = (C6 * s0 - C2 * s1 + C2 * s2 - C6 * s3 + 65536) >> 17u;
            wg_multi_blk[co + 56u] = (C7 * d0 - C5 * d1 + C3 * d2 - C1 * d3 + 65536) >> 17u;
        }
        workgroupBarrier();

        // Quantize + pack pairs to `tb_qblocks` + dequantize in-place!
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
                let q0_step = select(get_chroma_q(k0), get_luma_q(k0), is_y);
                let q1_step = select(get_chroma_q(k1), get_luma_q(k1), is_y);
                let q0 = quantize_div(wg_multi_blk[e0], q0_step);
                let q1 = quantize_div(wg_multi_blk[e1], q1_step);
                let packed = (u32(q0) & 0xFFFFu) | ((u32(q1) & 0xFFFFu) << 16u);
                tb_qblocks[qout_blocks_base + seq_blk * 32u + p_k] = bitcast<i32>(packed);
                wg_multi_blk[e0] = clamp(q0 * q0_step, -32768, 32768);
                wg_multi_blk[e1] = clamp(q1 * q1_step, -32768, 32768);
            }
        }
        workgroupBarrier();

        // Inverse 1D IDCT along columns (shift = 11): 64 threads = 8 blocks * 8 columns/block!
        if (seq_blk_lane < total_active_blocks) {
            let co = local_blk * 64u + line_idx;
            let x0 = wg_multi_blk[co];
            let x1 = wg_multi_blk[co + 8u];
            let x2 = wg_multi_blk[co + 16u];
            let x3 = wg_multi_blk[co + 24u];
            let x4 = wg_multi_blk[co + 32u];
            let x5 = wg_multi_blk[co + 40u];
            let x6 = wg_multi_blk[co + 48u];
            let x7 = wg_multi_blk[co + 56u];

            let a0 = C4 * (x0 + x4);
            let a1 = C4 * (x0 - x4);
            let b0 = C2 * x2 + C6 * x6;
            let b1 = C6 * x2 - C2 * x6;
            let e0 = a0 + b0;
            let e1 = a1 + b1;
            let e2 = a1 - b1;
            let e3 = a0 - b0;

            let o0 = C1 * x1 + C3 * x3 + C5 * x5 + C7 * x7;
            let o1 = C3 * x1 - C7 * x3 - C1 * x5 - C5 * x7;
            let o2 = C5 * x1 - C1 * x3 + C7 * x5 + C3 * x7;
            let o3 = C7 * x1 - C5 * x3 + C3 * x5 - C1 * x7;

            wg_multi_tmp[co]       = (e0 + o0 + 1024) >> 11u;
            wg_multi_tmp[co + 8u]  = (e1 + o1 + 1024) >> 11u;
            wg_multi_tmp[co + 16u] = (e2 + o2 + 1024) >> 11u;
            wg_multi_tmp[co + 24u] = (e3 + o3 + 1024) >> 11u;
            wg_multi_tmp[co + 32u] = (e3 - o3 + 1024) >> 11u;
            wg_multi_tmp[co + 40u] = (e2 - o2 + 1024) >> 11u;
            wg_multi_tmp[co + 48u] = (e1 - o1 + 1024) >> 11u;
            wg_multi_tmp[co + 56u] = (e0 - o0 + 1024) >> 11u;
        }
        workgroupBarrier();

        // Inverse 1D IDCT along rows (shift = 17) + Reconstruct plane: 64 threads = 8 blocks * 8 rows/block!
        if (seq_blk_lane < total_active_blocks) {
            let ro = local_blk * 64u + line_idx * 8u;
            let x0 = wg_multi_tmp[ro];
            let x1 = wg_multi_tmp[ro + 1u];
            let x2 = wg_multi_tmp[ro + 2u];
            let x3 = wg_multi_tmp[ro + 3u];
            let x4 = wg_multi_tmp[ro + 4u];
            let x5 = wg_multi_tmp[ro + 5u];
            let x6 = wg_multi_tmp[ro + 6u];
            let x7 = wg_multi_tmp[ro + 7u];

            let a0 = C4 * (x0 + x4);
            let a1 = C4 * (x0 - x4);
            let b0 = C2 * x2 + C6 * x6;
            let b1 = C6 * x2 - C2 * x6;
            let e0 = a0 + b0;
            let e1 = a1 + b1;
            let e2 = a1 - b1;
            let e3 = a0 - b0;

            let o0 = C1 * x1 + C3 * x3 + C5 * x5 + C7 * x7;
            let o1 = C3 * x1 - C7 * x3 - C1 * x5 - C5 * x7;
            let o2 = C5 * x1 - C1 * x3 + C7 * x5 + C3 * x7;
            let o3 = C7 * x1 - C5 * x3 + C3 * x5 - C1 * x7;

            var y_out: array<i32, 8>;
            y_out[0] = clamp(wg_multi_ref[ro]      + ((e0 + o0 + 65536) >> 17u), 0, 255);
            y_out[1] = clamp(wg_multi_ref[ro + 1u] + ((e1 + o1 + 65536) >> 17u), 0, 255);
            y_out[2] = clamp(wg_multi_ref[ro + 2u] + ((e2 + o2 + 65536) >> 17u), 0, 255);
            y_out[3] = clamp(wg_multi_ref[ro + 3u] + ((e3 + o3 + 65536) >> 17u), 0, 255);
            y_out[4] = clamp(wg_multi_ref[ro + 4u] + ((e3 - o3 + 65536) >> 17u), 0, 255);
            y_out[5] = clamp(wg_multi_ref[ro + 5u] + ((e2 - o2 + 65536) >> 17u), 0, 255);
            y_out[6] = clamp(wg_multi_ref[ro + 6u] + ((e1 - o1 + 65536) >> 17u), 0, 255);
            y_out[7] = clamp(wg_multi_ref[ro + 7u] + ((e0 - o0 + 65536) >> 17u), 0, 255);

            if (is_y_lane) {
                let by = seq_blk_lane / luma_bx;
                let bx = seq_blk_lane % luma_bx;
                let dst_base = (by * 8u + line_idx) * 64u + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    wg_y_packed[dst_base + c] = y_out[c];
                }
            } else if (is_cb_lane) {
                let c_seq = seq_blk_lane - num_luma;
                let by = c_seq / chroma_bx;
                let bx = c_seq % chroma_bx;
                let dst_base = (by * 8u + line_idx) * 32u + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    let c_idx = dst_base + c;
                    wg_c_packed[c_idx] = (wg_c_packed[c_idx] & 0xFFFFFF00u) | u32(y_out[c]);
                }
            } else {
                let c_seq = seq_blk_lane - num_luma - num_chroma;
                let by = c_seq / chroma_bx;
                let bx = c_seq % chroma_bx;
                let dst_base = (by * 8u + line_idx) * 32u + bx * 8u;
                for (var c = 0u; c < 8u; c = c + 1u) {
                    let c_idx = dst_base + c;
                    wg_c_packed[c_idx] = (wg_c_packed[c_idx] & 0xFFFF00FFu) | (u32(y_out[c]) << 8u);
                }
            }
        }
        workgroupBarrier();
    }

    // 3. Reconstruct 64x64 BGRA tile into `tb_recon_bgra[tidx * 4096 + py * 64 + px]`
    let tile_px_base = tidx * 4096u;
    for (var idx = lane; idx < 4096u; idx = idx + 64u) {
        let py = idx >> 6u;
        let px = idx & 63u;
        if (py < desc.th && px < desc.tw) {
            let y_val = wg_y_packed[idx];
            let c_idx = (py >> 1u) * 32u + (px >> 1u);
            let cp = wg_c_packed[c_idx];
            let cb_val = i32(cp & 0xFFu);
            let cr_val = i32((cp >> 8u) & 0xFFu);
            tb_recon_bgra[tile_px_base + idx] = ycbcr_to_bgra(y_val, cb_val, cr_val, 255);
        }
    }
}


@compute @workgroup_size(64, 1, 1)
fn batch_dct_quant_idct(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bidx = wid.x;
    if (bidx >= dct_params.num_blocks) {
        return;
    }
    let lane = lid.x;
    let r = lane >> 3u;
    let c = lane & 7u;
    let base = bidx * 64u;

    wg_blk[lane] = dct_in_blocks[base + lane];
    workgroupBarrier();

    // 1. Forward 1D DCT along rows (shift = 11)
    let ro = r * 8u;
    wg_tmp[lane] = dct_1d_elem(
        wg_blk[ro], wg_blk[ro + 1u], wg_blk[ro + 2u], wg_blk[ro + 3u],
        wg_blk[ro + 4u], wg_blk[ro + 5u], wg_blk[ro + 6u], wg_blk[ro + 7u],
        c, 11u
    );
    workgroupBarrier();

    // 2. Forward 1D DCT along columns (shift = 17) + Quantize + Dequantize
    let dct_val = dct_1d_elem(
        wg_tmp[c], wg_tmp[8u + c], wg_tmp[16u + c], wg_tmp[24u + c],
        wg_tmp[32u + c], wg_tmp[40u + c], wg_tmp[48u + c], wg_tmp[56u + c],
        r, 17u
    );
    let q = get_quant(lane);
    let qv = quantize_div(dct_val, q);
    dct_out_qblocks[base + lane] = qv;
    wg_blk[lane] = clamp(qv * q, -32768, 32768);
    workgroupBarrier();

    // 3. Inverse 1D IDCT along columns (shift = 11)
    wg_tmp[lane] = idct_1d_elem(
        wg_blk[c], wg_blk[8u + c], wg_blk[16u + c], wg_blk[24u + c],
        wg_blk[32u + c], wg_blk[40u + c], wg_blk[48u + c], wg_blk[56u + c],
        r, 11u
    );
    workgroupBarrier();

    // 4. Inverse 1D IDCT along rows (shift = 17)
    let idct_val = idct_1d_elem(
        wg_tmp[ro], wg_tmp[ro + 1u], wg_tmp[ro + 2u], wg_tmp[ro + 3u],
        wg_tmp[ro + 4u], wg_tmp[ro + 5u], wg_tmp[ro + 6u], wg_tmp[ro + 7u],
        c, 17u
    );
    dct_out_idct[base + lane] = idct_val;
}

// ---------------------------------------------------------------------------
// Kernel 3: Batched Inverse 2D 8x8 IDCT (Decoder)
// Input in `dct_in_blocks` is already dequantized & clamped to [-32768, 32768].
// ---------------------------------------------------------------------------

@compute @workgroup_size(64, 1, 1)
fn batch_idct_only(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bidx = wid.x;
    if (bidx >= dct_params.num_blocks) {
        return;
    }
    let lane = lid.x;
    let r = lane >> 3u;
    let c = lane & 7u;
    let base = bidx * 64u;

    wg_blk[lane] = dct_in_blocks[base + lane];
    workgroupBarrier();

    // 1. Inverse 1D IDCT along columns (shift = 11)
    wg_tmp[lane] = idct_1d_elem(
        wg_blk[c], wg_blk[8u + c], wg_blk[16u + c], wg_blk[24u + c],
        wg_blk[32u + c], wg_blk[40u + c], wg_blk[48u + c], wg_blk[56u + c],
        r, 11u
    );
    workgroupBarrier();

    // 2. Inverse 1D IDCT along rows (shift = 17)
    let ro = r * 8u;
    let idct_val = idct_1d_elem(
        wg_tmp[ro], wg_tmp[ro + 1u], wg_tmp[ro + 2u], wg_tmp[ro + 3u],
        wg_tmp[ro + 4u], wg_tmp[ro + 5u], wg_tmp[ro + 6u], wg_tmp[ro + 7u],
        c, 17u
    );
    dct_out_idct[base + lane] = idct_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParamsUniform {
    fb_w: u32,
    fb_h: u32,
    cols: u32,
    rows: u32,
    num_tiles: u32,
    is_inter: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DctBatchParamsUniform {
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    quant: [i32; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuTileDesc {
    pub x0: u32,
    pub y0: u32,
    pub tw: u32,
    pub th: u32,
    pub pred_x: u32,
    pub pred_y: u32,
    pub has_alpha: u32,
    pub use_inter: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileBatchParamsUniform {
    fb_w: u32,
    fb_h: u32,
    num_tiles: u32,
    _pad0: u32,
    luma_q: [i32; 64],
    chroma_q: [i32; 64],
}

/// Persistent GPU compute context and reusable staging/storage buffers.
pub struct SmartTileGpuEngine {
    pub adapter_name: String,
    pub device_type: wgpu::DeviceType,
    device: wgpu::Device,
    queue: wgpu::Queue,
    scan_bgl: wgpu::BindGroupLayout,
    scan_pipeline: wgpu::ComputePipeline,
    dct_bgl: wgpu::BindGroupLayout,
    dct_pipeline: wgpu::ComputePipeline,
    idct_pipeline: wgpu::ComputePipeline,
    tile_batch_bgl: wgpu::BindGroupLayout,
    tile_batch_pipeline: wgpu::ComputePipeline,
    /// Last timing breakdown `(upload_time, gpu_dispatch_and_readback_time, cpu_rle_and_scatter_time)`.
    pub last_timing: Mutex<(
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    )>,
    buffers: Mutex<GpuScratchBuffers>,
}

#[derive(Default)]
struct GpuScratchBuffers {
    frame_capacity_bytes: u64,
    ref_buf: Option<wgpu::Buffer>,
    cur_buf: Option<wgpu::Buffer>,
    flags_capacity_u32: u64,
    flags_storage: Option<wgpu::Buffer>,
    flags_staging: Option<wgpu::Buffer>,
    scan_uniform: Option<wgpu::Buffer>,

    blocks_capacity: usize,
    dct_uniform: Option<wgpu::Buffer>,
    dct_in_buf: Option<wgpu::Buffer>,
    dct_q_buf: Option<wgpu::Buffer>,
    dct_idct_buf: Option<wgpu::Buffer>,
    dct_q_staging: Option<wgpu::Buffer>,
    dct_idct_staging: Option<wgpu::Buffer>,

    tile_batch_capacity: usize,
    tb_uniform: Option<wgpu::Buffer>,
    tb_descs_buf: Option<wgpu::Buffer>,
    tb_qblocks_buf: Option<wgpu::Buffer>,
    tb_qblocks_staging: Option<wgpu::Buffer>,
    tb_recon_buf: Option<wgpu::Buffer>,
    tb_recon_staging: Option<wgpu::Buffer>,
}

static GLOBAL_GPU_ENGINE: OnceLock<Option<SmartTileGpuEngine>> = OnceLock::new();

/// Obtain a reference to the singleton hardware GPU compute engine if a hardware Vulkan/GLES
/// adapter (`DiscreteGpu` or `IntegratedGpu`) is available.
pub fn global_gpu_engine() -> Option<&'static SmartTileGpuEngine> {
    GLOBAL_GPU_ENGINE
        .get_or_init(SmartTileGpuEngine::try_new_hardware)
        .as_ref()
}

impl SmartTileGpuEngine {
    /// Initialize the `wgpu` compute pipelines on a hardware GPU (skipping CPU software adapters like `llvmpipe`).
    pub fn try_new_hardware() -> Option<Self> {
        Self::try_new_internal(false)
    }

    /// Initialize the `wgpu` compute pipelines, allowing fallback CPU adapters (used in tests if no physical GPU is exposed).
    pub fn try_new_any() -> Option<Self> {
        Self::try_new_internal(true)
    }

    fn try_new_internal(allow_cpu_adapter: bool) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapters = pollster::block_on(
            instance.enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::GL),
        );
        let adapter = adapters
            .into_iter()
            .filter(|a| {
                let info = a.get_info();
                allow_cpu_adapter
                    || matches!(
                        info.device_type,
                        wgpu::DeviceType::DiscreteGpu
                            | wgpu::DeviceType::IntegratedGpu
                            | wgpu::DeviceType::VirtualGpu
                    )
            })
            .min_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                _ => 3,
            })?;

        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("SmartTileGpuEngine"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;

        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("smart_tile.wgsl"),
                    source: wgpu::ShaderSource::Wgsl(SMART_TILE_WGSL.into()),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };

        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let scan_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan_pl"),
            bind_group_layouts: &[Some(&scan_bgl)],
            immediate_size: 0,
        });

        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scan_dirty_tiles_pipeline"),
            layout: Some(&scan_pl),
            module: &shader,
            entry_point: Some("scan_dirty_tiles"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dct_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dct_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let dct_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct_pl"),
            bind_group_layouts: &[Some(&dct_bgl)],
            immediate_size: 0,
        });

        let dct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_dct_quant_idct_pipeline"),
            layout: Some(&dct_pl),
            module: &shader,
            entry_point: Some("batch_dct_quant_idct"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let idct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_idct_only_pipeline"),
            layout: Some(&dct_pl),
            module: &shader,
            entry_point: Some("batch_idct_only"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let tile_batch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_batch_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tile_batch_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_batch_pl"),
            bind_group_layouts: &[Some(&tile_batch_bgl)],
            immediate_size: 0,
        });

        let tile_batch_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("encode_fullres_tiles_batch_pipeline"),
                layout: Some(&tile_batch_pl),
                module: &shader,
                entry_point: Some("encode_fullres_tiles_batch"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        tracing::info!(
            adapter = %info.name,
            device_type = ?info.device_type,
            backend = ?info.backend,
            "Initialized SmartTile wgpu compute shader engine"
        );

        Some(Self {
            adapter_name: info.name,
            device_type: info.device_type,
            device,
            queue,
            scan_bgl,
            scan_pipeline,
            dct_bgl,
            dct_pipeline,
            idct_pipeline,
            tile_batch_bgl,
            tile_batch_pipeline,
            last_timing: Mutex::new((
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
            )),
            buffers: Mutex::new(GpuScratchBuffers::default()),
        })
    }

    /// Scan all 64x64 tiles of a `width x height` frame on the GPU in parallel, returning a per-tile flag
    /// vector where `bit 0 = dirty` (`ref_bgra != cur_bgra`) and `bit 1 = opaque` (all alphas == 255).
    pub fn scan_dirty_and_opaque_tiles(
        &self,
        ref_bgra: &[u8],
        cur_bgra: &[u8],
        width: u32,
        height: u32,
    ) -> Option<Vec<u32>> {
        let byte_len = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if ref_bgra.len() != byte_len || cur_bgra.len() != byte_len || byte_len == 0 {
            return None;
        }
        let cols = width.div_ceil(64);
        let rows = height.div_ceil(64);
        let total_tiles = (cols * rows) as usize;
        if total_tiles == 0 || total_tiles > 65535 {
            return None;
        }

        let mut sc = self.buffers.lock().ok()?;
        let frame_u64 = byte_len as u64;
        if sc.frame_capacity_bytes < frame_u64 {
            let cap = frame_u64.next_power_of_two().max(4 * 1024 * 1024);
            sc.ref_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_ref_buf"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.cur_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_cur_buf"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.frame_capacity_bytes = cap;
        }

        let flags_bytes = (total_tiles * 4) as u64;
        if sc.flags_capacity_u32 < total_tiles as u64 {
            let cap_tiles = (total_tiles as u64).next_power_of_two().max(1024);
            let cap_bytes = cap_tiles * 4;
            sc.flags_storage = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_flags_storage"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            sc.flags_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_flags_staging"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.flags_capacity_u32 = cap_tiles;
        }
        if sc.scan_uniform.is_none() {
            sc.scan_uniform = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_uniform"),
                size: std::mem::size_of::<GridParamsUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let uniform = GridParamsUniform {
            fb_w: width,
            fb_h: height,
            cols,
            rows,
            num_tiles: total_tiles as u32,
            is_inter: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let u_buf = sc.scan_uniform.as_ref()?;
        let ref_buf = sc.ref_buf.as_ref()?;
        let cur_buf = sc.cur_buf.as_ref()?;
        let flags_storage = sc.flags_storage.as_ref()?;
        let flags_staging = sc.flags_staging.as_ref()?;

        self.queue
            .write_buffer(u_buf, 0, bytemuck::bytes_of(&uniform));
        self.queue.write_buffer(ref_buf, 0, ref_bgra);
        self.queue.write_buffer(cur_buf, 0, cur_bgra);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_bg"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
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
                    resource: flags_storage.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scan_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scan_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(total_tiles as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(flags_storage, 0, flags_staging, 0, flags_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = flags_staging.slice(0..flags_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;

        let out: Option<Vec<u32>> = slice
            .get_mapped_range()
            .ok()
            .map(|mapped| bytemuck::cast_slice(&mapped).to_vec());
        flags_staging.unmap();
        out
    }

    fn ensure_dct_buffers(&self, sc: &mut GpuScratchBuffers, num_blocks: usize) {
        if sc.dct_uniform.is_none() {
            sc.dct_uniform = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_uniform"),
                size: std::mem::size_of::<DctBatchParamsUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if sc.blocks_capacity < num_blocks {
            let cap_blocks = num_blocks.next_power_of_two().max(1024);
            let cap_bytes = (cap_blocks * 64 * std::mem::size_of::<i32>()) as u64;
            sc.dct_in_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_in_buf"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.dct_q_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_q_buf"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            sc.dct_idct_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_idct_buf"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            sc.dct_q_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_q_staging"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.dct_idct_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dct_idct_staging"),
                size: cap_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.blocks_capacity = cap_blocks;
        }
    }

    /// Execute batched 2D 8x8 forward DCT + quantization + dequantization + inverse 2D 8x8 IDCT on the GPU.
    /// Each block in `in_blocks` has 64 `i32` spatial residual elements (`[-255, 255]`).
    /// Writes quantized coefficients into `out_q_blocks` and reconstructed spatial residuals into `out_idct_blocks`.
    /// 100% bit-exact with `dct_8x8`, `quantize_div`, and `idct_8x8`.
    pub fn batch_dct_quant_idct(
        &self,
        in_blocks: &[[i32; 64]],
        quant: &[i32; 64],
        out_q_blocks: &mut [[i32; 64]],
        out_idct_blocks: &mut [[i32; 64]],
    ) -> bool {
        let num_blocks = in_blocks.len();
        if num_blocks == 0
            || num_blocks > 65535
            || out_q_blocks.len() != num_blocks
            || out_idct_blocks.len() != num_blocks
        {
            return false;
        }
        let Ok(mut sc) = self.buffers.lock() else {
            return false;
        };
        self.ensure_dct_buffers(&mut sc, num_blocks);

        let byte_len = (num_blocks * 64 * std::mem::size_of::<i32>()) as u64;
        let uniform = DctBatchParamsUniform {
            num_blocks: num_blocks as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            quant: *quant,
        };

        let u_buf = sc.dct_uniform.as_ref().unwrap();
        let in_buf = sc.dct_in_buf.as_ref().unwrap();
        let q_buf = sc.dct_q_buf.as_ref().unwrap();
        let idct_buf = sc.dct_idct_buf.as_ref().unwrap();
        let q_staging = sc.dct_q_staging.as_ref().unwrap();
        let idct_staging = sc.dct_idct_staging.as_ref().unwrap();

        self.queue
            .write_buffer(u_buf, 0, bytemuck::bytes_of(&uniform));
        self.queue
            .write_buffer(in_buf, 0, bytemuck::cast_slice(in_blocks));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dct_bg"),
            layout: &self.dct_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: idct_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dct_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dct_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.dct_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(num_blocks as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(q_buf, 0, q_staging, 0, byte_len);
        encoder.copy_buffer_to_buffer(idct_buf, 0, idct_staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let q_slice = q_staging.slice(0..byte_len);
        let idct_slice = idct_staging.slice(0..byte_len);
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        q_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx1.send(res);
        });
        idct_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx2.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if rx1.recv().ok().and_then(|r| r.ok()).is_none()
            || rx2.recv().ok().and_then(|r| r.ok()).is_none()
        {
            q_staging.unmap();
            idct_staging.unmap();
            return false;
        }

        let copied = match (q_slice.get_mapped_range(), idct_slice.get_mapped_range()) {
            (Ok(q_mapped), Ok(idct_mapped)) => {
                bytemuck::cast_slice_mut::<[i32; 64], u8>(out_q_blocks).copy_from_slice(&q_mapped);
                bytemuck::cast_slice_mut::<[i32; 64], u8>(out_idct_blocks)
                    .copy_from_slice(&idct_mapped);
                true
            }
            _ => false,
        };
        q_staging.unmap();
        idct_staging.unmap();
        copied
    }

    /// Execute batched 2D 8x8 inverse IDCT on the GPU for client decoding.
    /// Each block in `dequant_blocks` has 64 dequantized `i32` coefficients (`[-32768, 32768]`).
    /// 100% bit-exact with `idct_8x8`.
    pub fn batch_idct(
        &self,
        dequant_blocks: &[[i32; 64]],
        out_idct_blocks: &mut [[i32; 64]],
    ) -> bool {
        let num_blocks = dequant_blocks.len();
        if num_blocks == 0 || num_blocks > 65535 || out_idct_blocks.len() != num_blocks {
            return false;
        }
        let Ok(mut sc) = self.buffers.lock() else {
            return false;
        };
        self.ensure_dct_buffers(&mut sc, num_blocks);

        let byte_len = (num_blocks * 64 * std::mem::size_of::<i32>()) as u64;
        let uniform = DctBatchParamsUniform {
            num_blocks: num_blocks as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            quant: [1; 64],
        };

        let u_buf = sc.dct_uniform.as_ref().unwrap();
        let in_buf = sc.dct_in_buf.as_ref().unwrap();
        let q_buf = sc.dct_q_buf.as_ref().unwrap();
        let idct_buf = sc.dct_idct_buf.as_ref().unwrap();
        let idct_staging = sc.dct_idct_staging.as_ref().unwrap();

        self.queue
            .write_buffer(u_buf, 0, bytemuck::bytes_of(&uniform));
        self.queue
            .write_buffer(in_buf, 0, bytemuck::cast_slice(dequant_blocks));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("idct_bg"),
            layout: &self.dct_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: idct_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("idct_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idct_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.idct_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(num_blocks as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(idct_buf, 0, idct_staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let idct_slice = idct_staging.slice(0..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        idct_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            idct_staging.unmap();
            return false;
        }

        let copied = match idct_slice.get_mapped_range() {
            Ok(idct_mapped) => {
                bytemuck::cast_slice_mut::<[i32; 64], u8>(out_idct_blocks)
                    .copy_from_slice(&idct_mapped);
                true
            }
            Err(_) => false,
        };
        idct_staging.unmap();
        copied
    }

    /// Encode and reconstruct a batch of full-resolution opaque 64x64 tiles on the GPU in a single
    /// unified compute dispatch (`BGRA -> YCbCr420 -> Inter/Intra decision -> 2D 8x8 DCT -> Quantize
    /// -> Dequantize -> 2D 8x8 IDCT -> YCbCr420 -> BGRA`), invoking `consume(qblocks_flat, recon_pixels_flat)`
    /// directly on the mapped staging buffers without intermediate heap allocations.
    pub fn encode_fullres_tiles_batch_with<R, S, F: FnOnce(&mut [u8], S, &[i32], &[u8]) -> R>(
        &self,
        cur_bgra: &[u8],
        ref_bgra: &mut [u8],
        prev_ref_full: Option<&[u8]>,
        width: u32,
        height: u32,
        descs: &mut [GpuTileDesc],
        luma_q: &[i32; 64],
        chroma_q: &[i32; 64],
        prepare_descs: impl FnOnce(&[u8], &mut [GpuTileDesc]) -> S,
        on_submitted: impl FnOnce(),
        consume: F,
    ) -> Option<R> {
        let num_tiles = descs.len();
        if num_tiles == 0 || num_tiles > 65535 {
            return None;
        }
        let frame_bytes = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        let pred_bgra = prev_ref_full.unwrap_or(&*ref_bgra);
        if cur_bgra.len() != frame_bytes || pred_bgra.len() != frame_bytes {
            return None;
        }

        let mut sc = self.buffers.lock().ok()?;
        let frame_u64 = frame_bytes as u64;
        if sc.frame_capacity_bytes < frame_u64 {
            let cap = frame_u64.next_power_of_two().max(4 * 1024 * 1024);
            sc.ref_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_ref_buf"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.cur_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan_cur_buf"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.frame_capacity_bytes = cap;
        }

        if sc.tb_uniform.is_none() {
            sc.tb_uniform = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_uniform"),
                size: std::mem::size_of::<TileBatchParamsUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Layout in `tb_qblocks` per tile: `3073` i32s per tile (`1 + 96 * 32`):
        // - word 0 (`qout_tile_base + 0`): `is_inter` flag (`0` or `1`)
        // - words `1..3073` (`qout_tile_base + 1 + seq_blk * 32`): up to 96 quantized 8x8 blocks (64 i16s = 32 i32s per block)
        if sc.tile_batch_capacity < num_tiles {
            let cap_tiles = num_tiles.next_power_of_two().max(128);
            let descs_cap = (cap_tiles * std::mem::size_of::<GpuTileDesc>()) as u64;
            let qblocks_cap = (cap_tiles * 3073 * std::mem::size_of::<i32>()) as u64;
            let recon_cap = (cap_tiles * 4096 * 4) as u64;

            sc.tb_descs_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_descs_buf"),
                size: descs_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.tb_qblocks_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_qblocks_buf"),
                size: qblocks_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            sc.tb_qblocks_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_qblocks_staging"),
                size: qblocks_cap,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.tb_recon_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_recon_buf"),
                size: recon_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            sc.tb_recon_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tb_recon_staging"),
                size: recon_cap,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            sc.tile_batch_capacity = cap_tiles;
        }

        let uniform = TileBatchParamsUniform {
            fb_w: width,
            fb_h: height,
            num_tiles: num_tiles as u32,
            _pad0: 0,
            luma_q: *luma_q,
            chroma_q: *chroma_q,
        };

        let u_buf = sc.tb_uniform.as_ref()?;
        let descs_buf = sc.tb_descs_buf.as_ref()?;
        let cur_buf = sc.cur_buf.as_ref()?;
        let pred_buf = sc.ref_buf.as_ref()?;
        let q_buf = sc.tb_qblocks_buf.as_ref()?;
        let q_staging = sc.tb_qblocks_staging.as_ref()?;
        let recon_buf = sc.tb_recon_buf.as_ref()?;
        let recon_staging = sc.tb_recon_staging.as_ref()?;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tb_bg"),
            layout: &self.tile_batch_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: descs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cur_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: recon_buf.as_entire_binding(),
                },
            ],
        });

        let q_bytes = (num_tiles * 3073 * std::mem::size_of::<i32>()) as u64;
        let recon_bytes = (num_tiles * 4096 * 4) as u64;

        let t_up0 = std::time::Instant::now();
        self.queue
            .write_buffer(u_buf, 0, bytemuck::bytes_of(&uniform));
        self.queue.write_buffer(cur_buf, 0, cur_bgra);
        let any_inter = descs.iter().any(|d| d.use_inter != 0);
        if any_inter {
            self.queue.write_buffer(pred_buf, 0, pred_bgra);
        }
        // Kick off PCIe DMA transfer of the 16.5 MB framebuffers while the CPU executes 2D motion estimation!
        self.queue.submit(std::iter::empty());
        let d_up = t_up0.elapsed();

        let prep_state = prepare_descs(&*ref_bgra, descs);
        self.queue
            .write_buffer(descs_buf, 0, bytemuck::cast_slice(descs));

        let t_sub0 = std::time::Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tb_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tb_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.tile_batch_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(num_tiles as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(q_buf, 0, q_staging, 0, q_bytes);
        encoder.copy_buffer_to_buffer(recon_buf, 0, recon_staging, 0, recon_bytes);
        self.queue.submit(Some(encoder.finish()));

        let q_slice = q_staging.slice(0..q_bytes);
        let recon_slice = recon_staging.slice(0..recon_bytes);
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        q_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx1.send(res);
        });
        recon_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx2.send(res);
        });
        // Flush async work to the driver before overlapping CPU work!
        let _ = self.device.poll(wgpu::PollType::Poll);
        on_submitted();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if rx1.recv().ok().and_then(|r| r.ok()).is_none()
            || rx2.recv().ok().and_then(|r| r.ok()).is_none()
        {
            q_staging.unmap();
            recon_staging.unmap();
            return None;
        }
        let d_gpu = t_sub0.elapsed();

        let t_rle0 = std::time::Instant::now();
        let res = match (q_slice.get_mapped_range(), recon_slice.get_mapped_range()) {
            (Ok(q_mapped), Ok(recon_mapped)) => {
                let q_i32: &[i32] = bytemuck::cast_slice(&q_mapped);
                Some(consume(ref_bgra, prep_state, q_i32, &recon_mapped))
            }
            _ => None,
        };
        let d_rle = t_rle0.elapsed();
        if let Ok(mut lt) = self.last_timing.lock() {
            *lt = (d_up, d_gpu, d_rle);
        }
        q_staging.unmap();
        recon_staging.unmap();
        res
    }
}
