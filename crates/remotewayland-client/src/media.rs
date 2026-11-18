//! Pure-Rust XOR-delta + LZ4 video decoder (`VideoCodec::Lz4Delta`).

use anyhow::{bail, Context, Result};

/// A decoded BGRA (`Argb8888`) video frame ready for Wayland SHM presentation.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
}

/// Per-window XOR-delta + LZ4 video decoder.
#[derive(Default)]
pub struct VideoDecoder {
    width: u32,
    height: u32,
    prev_bgra: Vec<u8>,
}

impl VideoDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode an LZ4 keyframe or XOR-delta frame into a full `BGRA` (`Argb8888`) buffer.
    pub fn decode(
        &mut self,
        width: u32,
        height: u32,
        is_keyframe: bool,
        data: &[u8],
    ) -> Result<DecodedFrame> {
        let decompressed = lz4_flex::decompress_size_prepended(data)
            .context("failed to decompress LZ4 video frame")?;
        let expected_len = (width as usize) * (height as usize) * 4;
        if decompressed.len() != expected_len {
            bail!(
                "decoded frame size mismatch: expected {expected_len}, got {}",
                decompressed.len()
            );
        }

        if is_keyframe || width != self.width || height != self.height || self.prev_bgra.len() != expected_len {
            self.width = width;
            self.height = height;
            self.prev_bgra = decompressed;
        } else {
            for (prev, &delta) in self.prev_bgra.iter_mut().zip(decompressed.iter()) {
                *prev ^= delta;
            }
        }

        Ok(DecodedFrame {
            width,
            height,
            bgra: self.prev_bgra.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_keyframe_and_xor_delta() {
        let mut dec = VideoDecoder::new();
        let raw1 = vec![10u8; 16 * 16 * 4];
        let kf = lz4_flex::compress_prepend_size(&raw1);
        let d1 = dec.decode(16, 16, true, &kf).unwrap();
        assert_eq!(d1.bgra, raw1);

        // XOR delta flipping every byte from 10 -> 26 (10 ^ 16 = 26)
        let delta = vec![16u8; 16 * 16 * 4];
        let df = lz4_flex::compress_prepend_size(&delta);
        let d2 = dec.decode(16, 16, false, &df).unwrap();
        assert!(d2.bgra.iter().all(|&b| b == 26));
    }
}
