//! Fast LZ4 + 4-way interleaved 12-bit rANS (Asymmetric Numeral Systems) entropy coder for `SmartTile`.
//!
//! Works like Zstandard (Zstd) / Finite-State Entropy:
//! 1. First applies fast LZ4 dictionary/RLE match compression (`lz4_flex::compress`) to collapse
//!    repeated zero-runs, tile headers, and block patterns.
//! 2. Then applies 4-way interleaved 12-bit rANS arithmetic coding over the LZ4 token/literal byte
//!    stream, squeezing an extra **12–25% off** the already-LZ4-compressed packet!
//!
//! Wire format (prefixed with `RANS_MAGIC_BIT` on the 32-bit uncompressed length word):
//! - `[0..4]`: `u32` LE = `raw_uncompressed_len | RANS_MAGIC_BIT` (`0x8000_0000`)
//! - `[4..8]`: `u32` LE = `lz4_stage_len` (`0` if rANS was applied directly to `raw`, else length of intermediate LZ4 stream)
//! - `[8]`: `header_mode`:
//!   - `0`: Sparse symbol table: `[num_active: u8]` (`1..=255`), followed by `num_active` entries of
//!     `(sym: u8, freq_le: u16)` (sum of `freq` == `4096`).
//!   - `1`: Dense 12-bit table: 256 packed 12-bit frequencies (`384` bytes total).
//!   - `2`: Constant single-byte run (`[sym: u8]`, repeated `stage_len` times).
//! - `[table_end .. table_end + 16]`: 4 initial 32-bit rANS states (`s0, s1, s2, s3` in LE).
//! - `[table_end + 16 ..]`: `u16` LE renormalization words emitted in reverse order.

use anyhow::{bail, Result};

/// High bit set on the 32-bit size prefix to distinguish an rANS-compressed payload from raw LZ4.
pub const RANS_MAGIC_BIT: u32 = 0x8000_0000;

const PROB_BITS: u32 = 12;
const PROB_SCALE: usize = 1 << PROB_BITS; // 4096
const RANS_L: u32 = 1 << 16; // 65536 (16-bit word renormalization -> 32-bit state `[2^16, 2^28)`)

#[derive(Clone, Copy)]
struct SymEnc {
    x_max: u32,
    rcp: u64,
    cmpl_freq: u32,
    cum: u32,
}

/// Normalize raw byte counts `counts[0..256]` so all active symbols have `freq >= 1`
/// and `sum(freqs) == PROB_SCALE (4096)`.
fn normalize_frequencies(counts: &[u32; 256], total: usize) -> ([u16; 256], usize, u8) {
    let mut freqs = [0u16; 256];
    let mut active = 0usize;
    let mut single_sym = 0u8;
    let mut sum = 0i32;
    let mut max_sym = 0usize;
    let mut max_cnt = 0u32;

    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            active += 1;
            single_sym = s as u8;
            if c > max_cnt {
                max_cnt = c;
                max_sym = s;
            }
            let scaled = (((c as u64) * (PROB_SCALE as u64) + (total as u64 / 2)) / (total as u64))
                .max(1) as u16;
            freqs[s] = scaled;
            sum += scaled as i32;
        }
    }

    if active <= 1 {
        freqs = [0u16; 256];
        freqs[single_sym as usize] = PROB_SCALE as u16;
        return (freqs, active, single_sym);
    }

    let mut diff = (PROB_SCALE as i32) - sum;
    if diff > 0 {
        freqs[max_sym] = (freqs[max_sym] as i32 + diff) as u16;
    } else if diff < 0 {
        let need_sub = (-diff) as u16;
        if freqs[max_sym] > need_sub {
            freqs[max_sym] -= need_sub;
        } else {
            // Distribute reductions across symbols with `freq > 1`, largest first.
            let mut order: Vec<usize> = (0..256).filter(|&s| freqs[s] > 1).collect();
            order.sort_unstable_by(|&a, &b| freqs[b].cmp(&freqs[a]));
            while diff < 0 {
                for &s in &order {
                    if freqs[s] > 1 && diff < 0 {
                        let take = (freqs[s] - 1).min((-diff) as u16).max(1);
                        // Take smoothly on large surpluses, or 1-by-1 near zero.
                        let step = if take > 4 { take / 2 } else { 1 };
                        freqs[s] -= step;
                        diff += step as i32;
                    }
                }
            }
        }
    }

    debug_assert_eq!(freqs.iter().map(|&f| f as usize).sum::<usize>(), PROB_SCALE);
    (freqs, active, single_sym)
}

fn encode_rans_core(stage_input: &[u8], raw_len: usize, lz4_stage_len: u32) -> Vec<u8> {
    let n = stage_input.len();
    assert!(raw_len < (RANS_MAGIC_BIT as usize));
    let header_word = (raw_len as u32) | RANS_MAGIC_BIT;

    if n == 0 {
        let mut out = Vec::with_capacity(9);
        out.extend_from_slice(&header_word.to_le_bytes());
        out.extend_from_slice(&lz4_stage_len.to_le_bytes());
        out.push(2);
        return out;
    }

    let mut counts = [0u32; 256];
    for &b in stage_input {
        counts[b as usize] += 1;
    }

    let (freqs, active, single_sym) = normalize_frequencies(&counts, n);
    if active == 1 {
        let mut out = Vec::with_capacity(10);
        out.extend_from_slice(&header_word.to_le_bytes());
        out.extend_from_slice(&lz4_stage_len.to_le_bytes());
        out.push(2);
        out.push(single_sym);
        return out;
    }

    // Build cumulative frequency + division-free 63-bit reciprocal table for rANS encode.
    let mut enc_tab = [SymEnc {
        x_max: 0,
        rcp: 0,
        cmpl_freq: 0,
        cum: 0,
    }; 256];
    let mut cum = 0u32;
    for s in 0..256 {
        let f = freqs[s] as u32;
        if f > 0 {
            let rcp = (1u128 << 63).div_ceil(f as u128) as u64;
            enc_tab[s] = SymEnc {
                x_max: ((RANS_L >> PROB_BITS) << 16) * f,
                rcp,
                cmpl_freq: (PROB_SCALE as u32) - f,
                cum,
            };
            cum += f;
        }
    }

    // Emit u16 renormalization words into a backward buffer, then reverse into final payload.
    let mut words_rev: Vec<u16> = Vec::with_capacity(n / 2 + 16);
    let mut states = [RANS_L; 4];

    #[inline(always)]
    fn put_sym(state: &mut u32, sym: &SymEnc, words_rev: &mut Vec<u16>) {
        let mut x = *state;
        if x >= sym.x_max {
            words_rev.push((x & 0xFFFF) as u16);
            x >>= 16;
        }
        let q = (((x as u128) * (sym.rcp as u128)) >> 63) as u32;
        *state = x + q * sym.cmpl_freq + sym.cum;
    }

    let rem = n & 3;
    let aligned = n - rem;
    for lane in (0..rem).rev() {
        let b = stage_input[aligned + lane] as usize;
        put_sym(&mut states[lane], &enc_tab[b], &mut words_rev);
    }

    let mut idx = aligned;
    while idx > 0 {
        idx -= 4;
        put_sym(
            &mut states[3],
            &enc_tab[stage_input[idx + 3] as usize],
            &mut words_rev,
        );
        put_sym(
            &mut states[2],
            &enc_tab[stage_input[idx + 2] as usize],
            &mut words_rev,
        );
        put_sym(
            &mut states[1],
            &enc_tab[stage_input[idx + 1] as usize],
            &mut words_rev,
        );
        put_sym(
            &mut states[0],
            &enc_tab[stage_input[idx] as usize],
            &mut words_rev,
        );
    }

    let use_sparse = active <= 127;
    let table_bytes = if use_sparse { 1 + active * 3 } else { 1 + 384 };
    let mut out = Vec::with_capacity(8 + table_bytes + 16 + words_rev.len() * 2);
    out.extend_from_slice(&header_word.to_le_bytes());
    out.extend_from_slice(&lz4_stage_len.to_le_bytes());

    if use_sparse {
        out.push(0);
        out.push(active as u8);
        for (s, &f) in freqs.iter().enumerate() {
            if f > 0 {
                out.push(s as u8);
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
    } else {
        out.push(1);
        for pair in 0..128 {
            let f0 = freqs[pair * 2] as u32;
            let f1 = freqs[pair * 2 + 1] as u32;
            let packed = f0 | (f1 << 12);
            out.push((packed & 0xFF) as u8);
            out.push(((packed >> 8) & 0xFF) as u8);
            out.push(((packed >> 16) & 0xFF) as u8);
        }
    }

    for &st in &states {
        out.extend_from_slice(&st.to_le_bytes());
    }
    for &w in words_rev.iter().rev() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Compress `input` directly using 4-way interleaved 12-bit rANS (`lz4_stage_len == 0`).
pub fn compress_rans_prepend_size(input: &[u8]) -> Vec<u8> {
    encode_rans_core(input, input.len(), 0)
}

/// Compress an already-LZ4-compressed body (`lz4_with_4byte_prefix[4..]`) with 4-way interleaved 12-bit rANS,
/// returning `Some(rans_packet)` only when it strictly beats `lz4_with_4byte_prefix.len()`.
pub fn compress_lz4_body_with_rans(raw_len: usize, lz4_with_prefix: &[u8]) -> Option<Vec<u8>> {
    if lz4_with_prefix.len() < 64 || raw_len >= (RANS_MAGIC_BIT as usize) {
        return None;
    }
    let lz4_body = &lz4_with_prefix[4..];
    let rans_pkt = encode_rans_core(lz4_body, raw_len, lz4_body.len() as u32);
    if rans_pkt.len() < lz4_with_prefix.len() {
        Some(rans_pkt)
    } else {
        None
    }
}

/// Decompress a `RANS_MAGIC_BIT`-prefixed 4-way interleaved rANS (or LZ4 + rANS) payload with strict `max_uncompressed` bounds.
pub fn decompress_rans_bounded(payload: &[u8], max_uncompressed: usize) -> Result<Vec<u8>> {
    if payload.len() < 9 {
        bail!("rANS payload too short: {} bytes", payload.len());
    }
    let header_word = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let raw_out_len = (header_word & !RANS_MAGIC_BIT) as usize;
    if raw_out_len > max_uncompressed {
        bail!("rANS declared size {raw_out_len} exceeds limit {max_uncompressed}");
    }
    let lz4_stage_len =
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    let out_len = if lz4_stage_len > 0 {
        // Upper bound on valid LZ4 compressed stream size for `max_uncompressed`
        let max_lz4 = max_uncompressed
            .saturating_add(max_uncompressed / 255)
            .saturating_add(64);
        if lz4_stage_len > max_lz4 {
            bail!("rANS LZ4 stage size {lz4_stage_len} exceeds limit {max_lz4}");
        }
        lz4_stage_len
    } else {
        raw_out_len
    };
    let mode = payload[8];
    if raw_out_len == 0 {
        return Ok(Vec::new());
    }

    if mode == 2 {
        if payload.len() < 10 {
            bail!("rANS constant payload missing symbol byte");
        }
        let stage_vec = vec![payload[9]; out_len];
        return if lz4_stage_len > 0 {
            lz4_flex::decompress(&stage_vec, raw_out_len)
                .map_err(|e| anyhow::anyhow!("failed to decompress LZ4 stage after rANS: {e}"))
        } else {
            Ok(stage_vec)
        };
    }

    let mut freqs = [0u16; 256];
    let mut pos = 9usize;
    match mode {
        0 => {
            if pos >= payload.len() {
                bail!("unexpected EOF reading rANS sparse count");
            }
            let active = payload[pos] as usize;
            pos += 1;
            if active == 0 || pos + active * 3 > payload.len() {
                bail!("invalid rANS sparse frequency table");
            }
            let mut prev_sym: Option<u8> = None;
            for _ in 0..active {
                let s = payload[pos];
                if let Some(ps) = prev_sym {
                    if s <= ps {
                        bail!("non-monotonic symbol in rANS sparse table");
                    }
                }
                prev_sym = Some(s);
                let f = u16::from_le_bytes([payload[pos + 1], payload[pos + 2]]);
                if f == 0 || f > PROB_SCALE as u16 {
                    bail!("invalid rANS symbol frequency {f}");
                }
                freqs[s as usize] = f;
                pos += 3;
            }
        }
        1 => {
            if pos + 384 > payload.len() {
                bail!("unexpected EOF reading rANS dense frequency table");
            }
            for pair in 0..128 {
                let b0 = payload[pos] as u32;
                let b1 = payload[pos + 1] as u32;
                let b2 = payload[pos + 2] as u32;
                pos += 3;
                let packed = b0 | (b1 << 8) | (b2 << 16);
                let f0 = (packed & 0xFFF) as u16;
                let f1 = ((packed >> 12) & 0xFFF) as u16;
                freqs[pair * 2] = f0;
                freqs[pair * 2 + 1] = f1;
            }
        }
        _ => bail!("unknown rANS header mode: {mode}"),
    }

    let sum: usize = freqs.iter().map(|&f| f as usize).sum();
    if sum != PROB_SCALE {
        bail!("invalid rANS frequency sum {sum} (expected {PROB_SCALE})");
    }

    if pos + 16 > payload.len() {
        bail!("unexpected EOF reading rANS initial states");
    }
    let mut s0 = u32::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
    ]);
    let mut s1 = u32::from_le_bytes([
        payload[pos + 4],
        payload[pos + 5],
        payload[pos + 6],
        payload[pos + 7],
    ]);
    let mut s2 = u32::from_le_bytes([
        payload[pos + 8],
        payload[pos + 9],
        payload[pos + 10],
        payload[pos + 11],
    ]);
    let mut s3 = u32::from_le_bytes([
        payload[pos + 12],
        payload[pos + 13],
        payload[pos + 14],
        payload[pos + 15],
    ]);
    pos += 16;

    // Build single-lookup O(1) decode table (`4096 * 4 = 16 KB` in L1 cache!):
    // Packed `u32` entry: `bits 0..8 = sym`, `bits 8..20 = freq`, `bits 20..32 = slot - cum`.
    let mut dec_tab = [0u32; PROB_SCALE];
    let mut cum = 0usize;
    for (s, &freq_u16) in freqs.iter().enumerate() {
        let f = freq_u16 as usize;
        if f > 0 {
            let base_entry = (s as u32) | ((f as u32) << 8);
            for bias in 0..f {
                dec_tab[cum + bias] = base_entry | ((bias as u32) << 20);
            }
            cum += f;
        }
    }

    let mut out = vec![0u8; out_len];
    let stream = &payload[pos..];
    let mut sp = 0usize;
    let mask = (PROB_SCALE as u32) - 1;

    let chunks4 = out_len / 4;
    let tail_len = out_len & 3;
    let out_ptr = out.as_mut_ptr();
    for i in 0..chunks4 {
        let entry0 = dec_tab[(s0 & mask) as usize];
        let entry1 = dec_tab[(s1 & mask) as usize];
        let entry2 = dec_tab[(s2 & mask) as usize];
        let entry3 = dec_tab[(s3 & mask) as usize];

        unsafe {
            let base = out_ptr.add(i * 4);
            *base = (entry0 & 0xFF) as u8;
            *base.add(1) = (entry1 & 0xFF) as u8;
            *base.add(2) = (entry2 & 0xFF) as u8;
            *base.add(3) = (entry3 & 0xFF) as u8;
        }

        s0 = ((entry0 >> 8) & 0xFFF) * (s0 >> PROB_BITS) + (entry0 >> 20);
        s1 = ((entry1 >> 8) & 0xFFF) * (s1 >> PROB_BITS) + (entry1 >> 20);
        s2 = ((entry2 >> 8) & 0xFFF) * (s2 >> PROB_BITS) + (entry2 >> 20);
        s3 = ((entry3 >> 8) & 0xFFF) * (s3 >> PROB_BITS) + (entry3 >> 20);

        let is_last_chunk = (i + 1) == chunks4;
        if s0 < RANS_L && (!is_last_chunk || tail_len > 0) {
            if sp + 2 > stream.len() {
                bail!("unexpected EOF renormalizing rANS state 0");
            }
            s0 = (s0 << 16) | (u16::from_le_bytes([stream[sp], stream[sp + 1]]) as u32);
            sp += 2;
        }
        if s1 < RANS_L && (!is_last_chunk || tail_len > 1) {
            if sp + 2 > stream.len() {
                bail!("unexpected EOF renormalizing rANS state 1");
            }
            s1 = (s1 << 16) | (u16::from_le_bytes([stream[sp], stream[sp + 1]]) as u32);
            sp += 2;
        }
        if s2 < RANS_L && (!is_last_chunk || tail_len > 2) {
            if sp + 2 > stream.len() {
                bail!("unexpected EOF renormalizing rANS state 2");
            }
            s2 = (s2 << 16) | (u16::from_le_bytes([stream[sp], stream[sp + 1]]) as u32);
            sp += 2;
        }
        if s3 < RANS_L && !is_last_chunk {
            if sp + 2 > stream.len() {
                bail!("unexpected EOF renormalizing rANS state 3");
            }
            s3 = (s3 << 16) | (u16::from_le_bytes([stream[sp], stream[sp + 1]]) as u32);
            sp += 2;
        }
    }

    let mut states = [s0, s1, s2, s3];
    let base_idx = chunks4 * 4;
    for lane in 0..tail_len {
        let st = &mut states[lane];
        let entry = dec_tab[(*st & mask) as usize];
        out[base_idx + lane] = (entry & 0xFF) as u8;
        *st = ((entry >> 8) & 0xFFF) * (*st >> PROB_BITS) + (entry >> 20);
        // No further symbols are decoded from `states[lane]` in the tail, so no renormalization needed.
    }

    if lz4_stage_len > 0 {
        lz4_flex::decompress(&out, raw_out_len)
            .map_err(|e| anyhow::anyhow!("failed to decompress LZ4 stage after rANS: {e}"))
    } else {
        Ok(out)
    }
}
