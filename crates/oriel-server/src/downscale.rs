//! Area-average (box) downscaling of premultiplied `BGRA` frames.
//!
//! Each output pixel is the average of the source pixels its footprint covers, weighted by how
//! much of each it covers. Premultiplied channels average directly, so translucent edges keep
//! their colour. The filter is separable: source rows are weighted into a 16-bit accumulator row
//! (which vectorizes on baseline x86-64), which is then weighted across columns.

use smithay::utils::{Physical, Rectangle};

/// Fixed-point bits of the vertical weights: `255 << Y_BITS` fits the `u16` accumulator.
const Y_BITS: u32 = 8;
/// Fixed-point bits of the horizontal weights.
const X_BITS: u32 = 12;
const SHIFT: u32 = X_BITS + Y_BITS;
const HALF: u32 = 1 << (SHIFT - 1);

/// The source pixels each output pixel along one axis covers, and how much of each. Every output
/// pixel gets the same number of taps (`taps`), padded with zero weights, so the inner loops have
/// a fixed length.
#[derive(Default)]
struct AxisWeights {
    src_len: u32,
    dst_len: u32,
    taps: usize,
    /// Per output pixel: the first source pixel of its `taps` taps.
    firsts: Vec<u32>,
    /// `taps` weights per output pixel, summing to `1 << bits`.
    weights: Vec<u32>,
}

impl AxisWeights {
    fn update(&mut self, src_len: u32, dst_len: u32, bits: u32) {
        if self.src_len == src_len && self.dst_len == dst_len && !self.firsts.is_empty() {
            return;
        }
        self.src_len = src_len;
        self.dst_len = dst_len;
        let one = 1u32 << bits;
        let ratio = f64::from(src_len) / f64::from(dst_len);
        let mut spans: Vec<(u32, Vec<u32>)> = Vec::with_capacity(dst_len as usize);
        for o in 0..dst_len {
            let lo = f64::from(o) * ratio;
            let hi = (f64::from(o + 1) * ratio).min(f64::from(src_len));
            let first = (lo.floor() as u32).min(src_len - 1);
            let end = (hi.ceil() as u32).clamp(first + 1, src_len);
            let mut weights: Vec<u32> = (first..end)
                .map(|i| {
                    let cover = (f64::from(i + 1).min(hi) - f64::from(i).max(lo)).max(0.0);
                    (cover / ratio * f64::from(one)).round() as u32
                })
                .collect();
            // Give the rounding error to the largest weight so each pixel's weights sum to one.
            let sum: u32 = weights.iter().sum();
            let largest = (0..weights.len()).max_by_key(|&k| weights[k]).unwrap_or(0);
            weights[largest] = (weights[largest] + one).saturating_sub(sum);
            spans.push((first, weights));
        }

        self.taps = spans.iter().map(|(_, w)| w.len()).max().unwrap_or(1);
        self.firsts.clear();
        self.weights.clear();
        for (first, weights) in spans {
            // Pad after the span, or before it where the span ends at the source's edge.
            let padded_first = first.min(src_len - self.taps as u32);
            let lead = (first - padded_first) as usize;
            self.firsts.push(padded_first);
            self.weights.extend(std::iter::repeat_n(0, lead));
            self.weights.extend_from_slice(&weights);
            self.weights.resize(self.firsts.len() * self.taps, 0);
        }
    }

    /// The source range output pixels `range` read.
    fn source_range(&self, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        self.firsts[range.start] as usize..self.firsts[range.end - 1] as usize + self.taps
    }
}

/// A reusable box downscaler: keeps its weight tables and accumulator between frames of the
/// same sizes.
#[derive(Default)]
pub struct BoxDownscaler {
    x: AxisWeights,
    y: AxisWeights,
    acc: Vec<u16>,
}

impl BoxDownscaler {
    /// Downscale the tightly packed `src_w`x`src_h` `BGRA` frame `src` into `dst`, which holds
    /// `dst_w`x`dst_h` pixels. The output must not be larger than the input along either axis.
    ///
    /// # Panics
    /// If a size is zero, the output is larger than the input, or a buffer is too short.
    pub fn downscale(
        &mut self,
        src: &[u8],
        (src_w, src_h): (u32, u32),
        dst: &mut [u8],
        (dst_w, dst_h): (u32, u32),
    ) {
        let full = Rectangle::from_size((dst_w as i32, dst_h as i32).into());
        self.downscale_rects(src, (src_w, src_h), dst, (dst_w, dst_h), &[full]);
    }

    /// Like [`Self::downscale`], but only writes the output pixels inside `rects` (clipped to
    /// the output); the rest of `dst` is left as it is.
    pub fn downscale_rects(
        &mut self,
        src: &[u8],
        (src_w, src_h): (u32, u32),
        dst: &mut [u8],
        (dst_w, dst_h): (u32, u32),
        rects: &[Rectangle<i32, Physical>],
    ) {
        assert!(dst_w > 0 && dst_h > 0 && dst_w <= src_w && dst_h <= src_h);
        let src_stride = src_w as usize * 4;
        let dst_stride = dst_w as usize * 4;
        assert!(src.len() >= src_stride * src_h as usize);
        assert!(dst.len() >= dst_stride * dst_h as usize);

        self.x.update(src_w, dst_w, X_BITS);
        self.y.update(src_h, dst_h, Y_BITS);
        self.acc.resize(src_stride, 0);

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            unsafe { self.filter_rects_avx2(src, src_stride, dst, (dst_w, dst_h), rects) };
            return;
        }
        self.filter_rects(src, src_stride, dst, (dst_w, dst_h), rects);
    }

    /// [`Self::filter_rects`] compiled for AVX2, which roughly halves its time.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn filter_rects_avx2(
        &mut self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_size: (u32, u32),
        rects: &[Rectangle<i32, Physical>],
    ) {
        self.filter_rects(src, src_stride, dst, dst_size, rects);
    }

    #[inline(always)]
    fn filter_rects(
        &mut self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        (dst_w, dst_h): (u32, u32),
        rects: &[Rectangle<i32, Physical>],
    ) {
        let dst_stride = dst_w as usize * 4;
        let frame = Rectangle::<i32, Physical>::from_size((dst_w as i32, dst_h as i32).into());
        for rect in rects.iter().filter_map(|r| r.intersection(frame)) {
            let (x0, x1) = (rect.loc.x as usize, (rect.loc.x + rect.size.w) as usize);
            let (y0, y1) = (rect.loc.y as usize, (rect.loc.y + rect.size.h) as usize);
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            // Source columns (in bytes) the output columns read.
            let cols = self.x.source_range(x0..x1);
            let cols = cols.start * 4..cols.end * 4;
            for oy in y0..y1 {
                self.vertical(src, src_stride, oy, cols.clone());
                let out = &mut dst[oy * dst_stride + x0 * 4..oy * dst_stride + x1 * 4];
                let (firsts, weights) = (
                    &self.x.firsts[x0..x1],
                    &self.x.weights[x0 * self.x.taps..x1 * self.x.taps],
                );
                match self.x.taps {
                    1 => horizontal::<1>(&self.acc, firsts, weights, out),
                    2 => horizontal::<2>(&self.acc, firsts, weights, out),
                    3 => horizontal::<3>(&self.acc, firsts, weights, out),
                    4 => horizontal::<4>(&self.acc, firsts, weights, out),
                    taps => horizontal_any(&self.acc, firsts, weights, taps, out),
                }
            }
        }
    }

    /// Weight the source rows output row `oy` covers into the accumulator, for the source
    /// bytes `cols` of each row.
    #[inline(always)]
    fn vertical(&mut self, src: &[u8], stride: usize, oy: usize, cols: std::ops::Range<usize>) {
        let taps = self.y.taps;
        let first = self.y.firsts[oy] as usize;
        let acc = &mut self.acc[cols.clone()];
        let mut filled = false;
        for (k, &w) in self.y.weights[oy * taps..][..taps].iter().enumerate() {
            if w == 0 {
                continue;
            }
            let w = w as u16;
            let row = &src[(first + k) * stride..][cols.clone()];
            if filled {
                for (a, &p) in acc.iter_mut().zip(row) {
                    *a += w * u16::from(p);
                }
            } else {
                for (a, &p) in acc.iter_mut().zip(row) {
                    *a = w * u16::from(p);
                }
                filled = true;
            }
        }
    }
}

/// Weight the accumulator row `acc` across columns into `out`, `N` taps per output pixel.
#[inline(always)]
fn horizontal<const N: usize>(acc: &[u16], firsts: &[u32], weights: &[u32], out: &mut [u8]) {
    for ((px, &first), weights) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(firsts)
        .zip(weights.as_chunks::<N>().0)
    {
        let cols = &acc[first as usize * 4..][..N * 4];
        let mut sum = [HALF; 4];
        for (&w, col) in weights.iter().zip(cols.as_chunks::<4>().0) {
            for c in 0..4 {
                sum[c] += w * u32::from(col[c]);
            }
        }
        *px = sum.map(|s| (s >> SHIFT) as u8);
    }
}

#[inline(always)]
fn horizontal_any(acc: &[u16], firsts: &[u32], weights: &[u32], taps: usize, out: &mut [u8]) {
    for ((px, &first), weights) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(firsts)
        .zip(weights.chunks_exact(taps))
    {
        let cols = &acc[first as usize * 4..][..taps * 4];
        let mut sum = [HALF; 4];
        for (&w, col) in weights.iter().zip(cols.as_chunks::<4>().0) {
            for c in 0..4 {
                sum[c] += w * u32::from(col[c]);
            }
        }
        *px = sum.map(|s| (s >> SHIFT) as u8);
    }
}

/// The frame size, in pixels, of a `logical_w`x`logical_h` surface shown at `scale`.
pub fn scaled_size(logical_w: u32, logical_h: u32, scale: f64) -> (u32, u32) {
    (
        ((f64::from(logical_w) * scale).round() as u32).max(1),
        ((f64::from(logical_h) * scale).round() as u32).max(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn downscale(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
        let mut out = vec![0u8; (dw * dh * 4) as usize];
        BoxDownscaler::default().downscale(src, (sw, sh), &mut out, (dw, dh));
        out
    }

    fn solid(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
        px.repeat((w * h) as usize)
    }

    #[test]
    fn weights_sum_to_one_for_every_output_pixel() {
        for (src, dst) in [(2, 1), (10, 6), (2400, 1440), (2001, 1201), (7, 7), (13, 3)] {
            let mut axis = AxisWeights::default();
            axis.update(src, dst, X_BITS);
            assert_eq!(axis.firsts.len(), dst as usize);
            assert_eq!(axis.weights.len(), dst as usize * axis.taps);
            for (o, w) in axis.weights.chunks_exact(axis.taps).enumerate() {
                assert_eq!(w.iter().sum::<u32>(), 1 << X_BITS, "{src}->{dst}");
                assert!(axis.firsts[o] + axis.taps as u32 <= src);
            }
            // Every source pixel contributes to some output pixel.
            let mut covered = vec![false; src as usize];
            for (o, w) in axis.weights.chunks_exact(axis.taps).enumerate() {
                for (k, &w) in w.iter().enumerate() {
                    if w > 0 {
                        covered[axis.firsts[o] as usize + k] = true;
                    }
                }
            }
            assert!(covered.iter().all(|&c| c), "{src}->{dst}");
        }
    }

    #[test]
    fn halving_averages_each_2x2_block_exactly() {
        // 4x2 source: two 2x2 blocks.
        #[rustfmt::skip]
        let src: Vec<u8> = [
            [10, 20, 30, 40], [30, 40, 50, 60], [0, 0, 0, 0],     [100, 100, 100, 100],
            [50, 60, 70, 80], [70, 80, 90, 100], [200, 200, 200, 200], [0, 0, 0, 0],
        ]
        .concat();
        let out = downscale(&src, 4, 2, 2, 1);
        assert_eq!(out, vec![40, 50, 60, 70, 75, 75, 75, 75]);
    }

    #[test]
    fn identity_size_copies_the_frame() {
        let src: Vec<u8> = (0..5 * 3 * 4).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(downscale(&src, 5, 3, 5, 3), src);
    }

    #[test]
    fn two_to_one_point_two_covers_partial_pixels() {
        // 10 source pixels -> 6 output pixels: footprints are 5/3 px wide.
        let mut src = Vec::new();
        for v in [0u8, 30, 60, 90, 120, 150, 180, 210, 240, 255] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let out = downscale(&src, 10, 1, 6, 1);
        let values: Vec<u8> = out.chunks(4).map(|p| p[0]).collect();
        // Output 1 covers source [5/3, 10/3): 1/3 of px1, all of px2, 1/3 of px3.
        // (30/3 + 60 + 90/3) / (5/3) = 60.
        let expected: Vec<f64> = (0..6)
            .map(|o| {
                let (lo, hi) = (o as f64 * 10.0 / 6.0, (o + 1) as f64 * 10.0 / 6.0);
                let vals = [0., 30., 60., 90., 120., 150., 180., 210., 240., 255.];
                (0..10)
                    .map(|i| {
                        let cover = ((i + 1) as f64).min(hi) - (i as f64).max(lo);
                        cover.max(0.0) * vals[i]
                    })
                    .sum::<f64>()
                    / (hi - lo)
            })
            .collect();
        for (got, want) in values.iter().zip(&expected) {
            assert!(
                (f64::from(*got) - want).abs() <= 0.6,
                "{values:?} vs {expected:?}"
            );
        }
        assert_eq!(values[1], 60);
        // Alpha stays opaque.
        assert!(out.chunks(4).all(|p| p[3] == 255));
    }

    #[test]
    fn non_integer_ratio_sizes_and_solid_colour() {
        let (dw, dh) = scaled_size(1000, 700, 1.2);
        assert_eq!((dw, dh), (1200, 840));
        let src = solid(2000, 1400, [12, 34, 56, 78]);
        let out = downscale(&src, 2000, 1400, dw, dh);
        assert_eq!(out.len(), (dw * dh * 4) as usize);
        assert!(out.chunks(4).all(|p| p == [12, 34, 56, 78]));
    }

    #[test]
    fn premultiplied_alpha_averages_and_stays_valid() {
        // Opaque red next to fully transparent: half-covered red at half alpha.
        let src: Vec<u8> = [[0, 0, 255, 255], [0, 0, 0, 0]].concat();
        assert_eq!(downscale(&src, 2, 1, 1, 1), vec![0, 0, 128, 128]);

        // Random premultiplied pixels: no channel may exceed alpha after averaging.
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let (sw, sh) = (37, 23);
        let src: Vec<u8> = (0..sw * sh)
            .flat_map(|_| {
                let a = (next() % 256) as u8;
                let c = |n: u32| (n % (u32::from(a) + 1)) as u8;
                [c(next()), c(next()), c(next()), a]
            })
            .collect();
        let out = downscale(&src, sw, sh, 22, 14);
        for px in out.chunks(4) {
            assert!(px[0] <= px[3] && px[1] <= px[3] && px[2] <= px[3], "{px:?}");
        }
    }

    #[test]
    fn reuses_tables_across_sizes() {
        let mut ds = BoxDownscaler::default();
        let src = solid(8, 8, [1, 2, 3, 4]);
        let mut out = vec![0u8; 4 * 4 * 4];
        ds.downscale(&src, (8, 8), &mut out, (4, 4));
        let mut out2 = vec![0u8; 5 * 3 * 4];
        ds.downscale(&src, (8, 8), &mut out2, (5, 3));
        assert!(out2.chunks(4).all(|p| p == [1, 2, 3, 4]));
        ds.downscale(&src, (8, 8), &mut out, (4, 4));
        assert!(out.chunks(4).all(|p| p == [1, 2, 3, 4]));
    }

    #[test]
    fn rects_match_a_full_downscale_and_leave_the_rest() {
        let (sw, sh, dw, dh) = (50u32, 40u32, 30u32, 24u32);
        let src: Vec<u8> = (0..sw * sh * 4).map(|i| (i * 31 % 251) as u8).collect();
        let full = downscale(&src, sw, sh, dw, dh);
        let mut out = vec![7u8; (dw * dh * 4) as usize];
        let rects = [
            Rectangle::new((3, 4).into(), (5, 6).into()),
            Rectangle::new((25, 20).into(), (10, 10).into()), // clipped at the edge
        ];
        BoxDownscaler::default().downscale_rects(&src, (sw, sh), &mut out, (dw, dh), &rects);
        for y in 0..dh as i32 {
            for x in 0..dw as i32 {
                let i = ((y * dw as i32 + x) * 4) as usize;
                let inside = rects.iter().any(|r| r.contains((x, y)));
                let want = if inside {
                    &full[i..i + 4]
                } else {
                    &[7u8; 4][..]
                };
                assert_eq!(&out[i..i + 4], want, "({x}, {y})");
            }
        }
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_downscale_2x_to_1_2x() {
        let (lw, lh) = (1280, 800);
        let src = solid(lw * 2, lh * 2, [10, 20, 30, 255]);
        let (dw, dh) = scaled_size(lw, lh, 1.2);
        let mut out = vec![0u8; (dw * dh * 4) as usize];
        let mut ds = BoxDownscaler::default();
        ds.downscale(&src, (lw * 2, lh * 2), &mut out, (dw, dh));
        let n = 20;
        let start = std::time::Instant::now();
        for _ in 0..n {
            ds.downscale(&src, (lw * 2, lh * 2), &mut out, (dw, dh));
        }
        eprintln!(
            "{}x{} -> {dw}x{dh}: {:.2} ms/frame",
            lw * 2,
            lh * 2,
            start.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    }
}
