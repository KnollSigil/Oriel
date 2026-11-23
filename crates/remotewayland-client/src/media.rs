//! Pure-Rust video decoders (`VideoCodec::Lz4Delta` and `VideoCodec::SmartTile`).

use std::sync::Arc;

use anyhow::{bail, Result};
use remotewayland_proto::smart_tile::{
    decompress_lz4_bounded, validate_frame_dims, xor_decode_row, SmartTileDecoder,
};
use remotewayland_proto::VideoCodec;

/// A decoded BGRA (`Argb8888`) video frame ready for Wayland SHM presentation.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub bgra: Arc<Vec<u8>>,
    /// Damaged physical pixel rectangles `(x, y, w, h)` for partial compositor damage.
    /// Empty means full-frame damage.
    pub damage: Vec<(i32, i32, i32, i32)>,
}

/// Codec-polymorphic per-window video decoder.
pub enum WindowVideoDecoder {
    Lz4Delta(VideoDecoder),
    SmartTile(SmartTileDecoder),
}

impl WindowVideoDecoder {
    pub fn new(codec: VideoCodec) -> Self {
        match codec {
            VideoCodec::SmartTile => Self::SmartTile(SmartTileDecoder::new()),
            _ => Self::Lz4Delta(VideoDecoder::new()),
        }
    }

    pub fn decode(
        &mut self,
        width: u32,
        height: u32,
        is_keyframe: bool,
        data: &[u8],
    ) -> Result<DecodedFrame> {
        match self {
            Self::Lz4Delta(dec) => dec.decode(width, height, is_keyframe, data),
            Self::SmartTile(dec) => {
                let (bgra, damage) = dec.decode_with_damage(width, height, data)?;
                Ok(DecodedFrame {
                    width,
                    height,
                    bgra,
                    damage,
                })
            }
        }
    }
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
        let expected_len = validate_frame_dims(width, height)?;
        let decompressed = decompress_lz4_bounded(data, expected_len)?;
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
            xor_decode_row(&mut self.prev_bgra, &decompressed);
        }

        Ok(DecodedFrame {
            width,
            height,
            bgra: Arc::new(self.prev_bgra.clone()),
            damage: Vec::new(),
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
        assert_eq!(*d1.bgra, raw1);

        // XOR delta flipping every byte from 10 -> 26 (10 ^ 16 = 26)
        let delta = vec![16u8; 16 * 16 * 4];
        let df = lz4_flex::compress_prepend_size(&delta);
        let d2 = dec.decode(16, 16, false, &df).unwrap();
        assert!(d2.bgra.iter().all(|&b| b == 26));
    }
}
