//! Pure-Rust video encoders (`VideoCodec::Lz4Delta` and `VideoCodec::SmartTile`).
//!
//! - `Lz4Delta`: lossless XOR-delta + LZ4 on full `Argb8888` (`BGRA`) frames (legacy fallback).
//! - `SmartTile`: per-tile motion compensation, 64x64 dirty-tile skipping, adaptive lossless/DCT
//!   (Intra I-tiles & Inter residual P-tiles with alpha support), and Golomb-Rice lossless idle refinement.

use anyhow::Result;
use remotewayland_proto::smart_tile::{
    xor_encode_row, SmartTileEncoder, DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES,
};
use remotewayland_proto::VideoCodec;

use crate::gpu::FramePixels;

/// An encoded video frame (keyframe or delta/tile packet).
pub struct EncodedFrame {
    pub pts_ns: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub used_lossy: bool,
    pub had_copy_rect: bool,
    pub dirty_tiles: usize,
    pub data: Vec<u8>,
}

/// Codec-polymorphic per-window video encoder.
pub enum WindowVideoEncoder {
    Lz4Delta(VideoEncoder),
    SmartTile(SmartTileEncoder),
}

impl WindowVideoEncoder {
    pub fn new(codec: VideoCodec) -> Self {
        match codec {
            VideoCodec::SmartTile => Self::SmartTile(SmartTileEncoder::new()),
            _ => Self::Lz4Delta(VideoEncoder::new()),
        }
    }

    pub fn encode_frame(
        &mut self,
        width: u32,
        height: u32,
        pts_ns: u64,
        pixels: &FramePixels,
        quality: u8,
    ) -> Result<EncodedFrame> {
        match (self, pixels) {
            (Self::SmartTile(enc), FramePixels::Gpu(gpu_handle)) => {
                let pkt = enc.encode_gpu_texture(&gpu_handle.as_texture_frame(), quality)?;
                Ok(EncodedFrame {
                    pts_ns,
                    width,
                    height,
                    is_keyframe: pkt.is_keyframe,
                    used_lossy: pkt.used_lossy,
                    had_copy_rect: pkt.had_copy_rect,
                    dirty_tiles: pkt.dirty_tiles,
                    data: pkt.payload,
                })
            }
            (Self::SmartTile(enc), FramePixels::Cpu(bgra)) => {
                let pkt = enc.encode(width, height, bgra, quality);
                Ok(EncodedFrame {
                    pts_ns,
                    width,
                    height,
                    is_keyframe: pkt.is_keyframe,
                    used_lossy: pkt.used_lossy,
                    had_copy_rect: pkt.had_copy_rect,
                    dirty_tiles: pkt.dirty_tiles,
                    data: pkt.payload,
                })
            }
            (Self::Lz4Delta(enc), pixels) => {
                let bgra = pixels.to_cpu_bgra();
                Ok(enc.encode_bgra(width, height, pts_ns, &bgra))
            }
        }
    }

    pub fn encode_bgra(
        &mut self,
        width: u32,
        height: u32,
        pts_ns: u64,
        bgra: &[u8],
        quality: u8,
    ) -> EncodedFrame {
        match self {
            Self::Lz4Delta(enc) => enc.encode_bgra(width, height, pts_ns, bgra),
            Self::SmartTile(enc) => {
                let pkt = enc.encode(width, height, bgra, quality);
                EncodedFrame {
                    pts_ns,
                    width,
                    height,
                    is_keyframe: pkt.is_keyframe,
                    used_lossy: pkt.used_lossy,
                    had_copy_rect: pkt.had_copy_rect,
                    dirty_tiles: pkt.dirty_tiles,
                    data: pkt.payload,
                }
            }
        }
    }

    pub fn has_pending_lossless_refresh(&self) -> bool {
        match self {
            Self::Lz4Delta(_) => false,
            Self::SmartTile(enc) => enc.has_pending_lossless_refresh(),
        }
    }

    pub fn encode_lossless_refresh(
        &mut self,
        width: u32,
        height: u32,
        pts_ns: u64,
    ) -> Option<EncodedFrame> {
        match self {
            Self::Lz4Delta(_) => None,
            Self::SmartTile(enc) => enc
                .encode_lossless_refresh(DEFAULT_REFRESH_CHUNK_TILES, DEFAULT_REFRESH_MAX_BYTES)
                .map(|pkt| EncodedFrame {
                    pts_ns,
                    width,
                    height,
                    is_keyframe: false,
                    used_lossy: false,
                    had_copy_rect: false,
                    dirty_tiles: pkt.dirty_tiles,
                    data: pkt.payload,
                }),
        }
    }
}

/// Per-window XOR-delta + LZ4 video encoder.
#[derive(Default)]
pub struct VideoEncoder {
    width: u32,
    height: u32,
    prev_bgra: Vec<u8>,
    scratch_delta: Vec<u8>,
}

impl VideoEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode a `BGRA` (`Argb8888`) frame as either an LZ4 keyframe or an LZ4 XOR-delta frame.
    pub fn encode_bgra(
        &mut self,
        width: u32,
        height: u32,
        pts_ns: u64,
        bgra: &[u8],
    ) -> EncodedFrame {
        let is_keyframe =
            width != self.width || height != self.height || self.prev_bgra.len() != bgra.len();

        let data = if is_keyframe {
            self.width = width;
            self.height = height;
            self.prev_bgra.clear();
            self.prev_bgra.extend_from_slice(bgra);
            lz4_flex::compress_prepend_size(bgra)
        } else {
            self.scratch_delta.clear();
            xor_encode_row(&mut self.prev_bgra, bgra, &mut self.scratch_delta);
            lz4_flex::compress_prepend_size(&self.scratch_delta)
        };

        EncodedFrame {
            pts_ns,
            width,
            height,
            is_keyframe,
            used_lossy: false,
            had_copy_rect: false,
            dirty_tiles: 0,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_and_delta_encoding() {
        let mut enc = VideoEncoder::new();
        let frame1 = vec![42u8; 64 * 64 * 4];
        let out1 = enc.encode_bgra(64, 64, 100, &frame1);
        assert!(out1.is_keyframe);

        // Identical frame2 (16 KiB) should compress to < 128 bytes via LZ4 run-length encoding.
        let out2 = enc.encode_bgra(64, 64, 200, &frame1);
        assert!(!out2.is_keyframe);
        assert!(out2.data.len() < 128);

        // Resize should trigger a new keyframe automatically.
        let frame3 = vec![99u8; 80 * 50 * 4];
        let out3 = enc.encode_bgra(80, 50, 300, &frame3);
        assert!(out3.is_keyframe);
    }
}
