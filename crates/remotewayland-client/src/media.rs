//! Pure-Rust video decoders (`VideoCodec::Lz4Delta` and `VideoCodec::SmartTile`).

use anyhow::{bail, Context, Result};
use remotewayland_proto::smart_tile::SmartTileDecoder;
use remotewayland_proto::VideoCodec;

/// A decoded BGRA (`Argb8888`) video frame ready for Wayland SHM presentation.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
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
                let bgra = dec.decode(width, height, data)?;
                Ok(DecodedFrame {
                    width,
                    height,
                    bgra,
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
