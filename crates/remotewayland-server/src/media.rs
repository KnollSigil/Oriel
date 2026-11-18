//! Pure-Rust XOR-delta + LZ4 video encoder (`VideoCodec::Lz4Delta`).
//!
//! Compresses full `Argb8888` (`BGRA`) frames with zero chroma subsampling,
//! full alpha-channel preservation, and instant zero-stall window resizing.

/// An encoded video frame (keyframe or XOR delta compressed with LZ4).
pub struct EncodedFrame {
    pub pts_ns: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub data: Vec<u8>,
}

/// Per-window XOR-delta + LZ4 video encoder.
#[derive(Default)]
pub struct VideoEncoder {
    width: u32,
    height: u32,
    prev_bgra: Vec<u8>,
    scratch_delta: Vec<u8>,
    frame_idx: u64,
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
        let is_keyframe = width != self.width
            || height != self.height
            || self.prev_bgra.len() != bgra.len()
            || self.frame_idx % 120 == 0;

        let data = if is_keyframe {
            self.width = width;
            self.height = height;
            self.prev_bgra.clear();
            self.prev_bgra.extend_from_slice(bgra);
            lz4_flex::compress_prepend_size(bgra)
        } else {
            self.scratch_delta.resize(bgra.len(), 0);
            for ((d, &cur), prev) in self
                .scratch_delta
                .iter_mut()
                .zip(bgra.iter())
                .zip(self.prev_bgra.iter_mut())
            {
                *d = cur ^ *prev;
                *prev = cur;
            }
            lz4_flex::compress_prepend_size(&self.scratch_delta)
        };

        self.frame_idx = self.frame_idx.wrapping_add(1);
        EncodedFrame {
            pts_ns,
            width,
            height,
            is_keyframe,
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
