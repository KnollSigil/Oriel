//! GStreamer video decoding pipeline with hardware probe and software fallback.

use anyhow::{bail, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use remotewayland_proto::VideoCodec;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// A decoded BGRA (`Xrgb8888`) video frame ready for Wayland SHM presentation.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
}

/// Initialize GStreamer once per process.
pub fn init_gstreamer() -> Result<()> {
    gst::init().context("failed to initialize GStreamer")
}

/// Per-window GStreamer video decoder pipeline.
pub struct VideoDecoder {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
}

impl VideoDecoder {
    /// Create a new GStreamer decoder pipeline and return it with a receiver for decoded RGBA frames.
    pub fn new(codec: VideoCodec) -> Result<(Self, mpsc::UnboundedReceiver<DecodedFrame>)> {
        let (factory, dec_chain) = select_decoder_spec(codec)?;
        info!(?codec, decoder = factory, "creating GStreamer video decoder");

        let pipeline_str = format!(
            "appsrc name=src is-live=true format=time ! {dec_chain} ! videoconvert ! video/x-raw,format=BGRA ! appsink name=sink sync=false drop=true max-buffers=2"
        );
        let element = gst::parse::launch(&pipeline_str)
            .with_context(|| format!("failed to parse GStreamer decoder pipeline: {pipeline_str}"))?;
        let pipeline = element
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("parsed element is not a gst::Pipeline"))?;

        let appsrc = pipeline
            .by_name("src")
            .context("missing appsrc")?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| anyhow::anyhow!("src is not an AppSrc"))?;

        let caps = match codec {
            VideoCodec::H264 => gst::Caps::builder("video/x-h264")
                .field("stream-format", "byte-stream")
                .field("alignment", "au")
                .build(),
            VideoCodec::Hevc => gst::Caps::builder("video/x-h265")
                .field("stream-format", "byte-stream")
                .field("alignment", "au")
                .build(),
            VideoCodec::Av1 => gst::Caps::builder("video/x-av1").build(),
        };
        appsrc.set_caps(Some(&caps));

        let appsink = pipeline
            .by_name("sink")
            .context("missing appsink")?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("sink is not an AppSink"))?;

        pipeline
            .set_state(gst::State::Playing)
            .context("failed to set decoder pipeline to Playing")?;

        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("rw-gst-dec-pull".into())
            .spawn(move || {
                while let Ok(sample) = appsink.pull_sample() {
                    if let (Some(caps), Some(buf)) = (sample.caps(), sample.buffer()) {
                        if let Ok(info) = gst_video::VideoInfo::from_caps(caps) {
                            let width = info.width();
                            let height = info.height();
                            let stride = info.stride()[0].max(0) as usize;
                            if let Ok(map) = buf.map_readable() {
                                let src = map.as_slice();
                                let row_bytes = (width * 4) as usize;
                                let mut bgra = vec![0u8; row_bytes * height as usize];
                                for y in 0..(height as usize) {
                                    let s_off = y * stride;
                                    let d_off = y * row_bytes;
                                    if s_off + row_bytes <= src.len() {
                                        bgra[d_off..d_off + row_bytes]
                                            .copy_from_slice(&src[s_off..s_off + row_bytes]);
                                    }
                                }
                                if tx
                                    .send(DecodedFrame {
                                        width,
                                        height,
                                        bgra,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                warn!("GStreamer decoder pull thread stopped");
            })
            .context("failed to spawn decoder pull thread")?;

        Ok((Self { pipeline, appsrc }, rx))
    }

    /// Push an encoded access unit into the decoder.
    pub fn push_encoded(&self, pts_ns: u64, data: &[u8]) -> Result<()> {
        let mut buffer = gst::Buffer::with_size(data.len())
            .context("failed to allocate GStreamer encoded buffer")?;
        {
            let buf_mut = buffer.get_mut().context("failed to get mutable buffer")?;
            buf_mut.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            let mut map = buf_mut
                .map_writable()
                .context("failed to map GStreamer buffer writable")?;
            map.copy_from_slice(data);
        }

        self.appsrc
            .push_buffer(buffer)
            .map_err(|e| anyhow::anyhow!("appsrc push_buffer failed: {e:?}"))?;

        Ok(())
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        let _ = self.appsrc.end_of_stream();
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn select_decoder_spec(codec: VideoCodec) -> Result<(&'static str, &'static str)> {
    let candidates: &[(&str, &str)] = match codec {
        VideoCodec::H264 => &[
            ("vah264dec", "h264parse ! vah264dec"),
            ("nvh264dec", "h264parse ! nvh264dec"),
            ("v4l2h264dec", "h264parse ! v4l2h264dec"),
            ("avdec_h264", "h264parse ! avdec_h264"),
            ("openh264dec", "h264parse ! openh264dec"),
        ],
        VideoCodec::Hevc => &[
            ("vah265dec", "h265parse ! vah265dec"),
            ("nvh265dec", "h265parse ! nvh265dec"),
            ("v4l2h265dec", "h265parse ! v4l2h265dec"),
            ("avdec_h265", "h265parse ! avdec_h265"),
        ],
        VideoCodec::Av1 => &[
            ("vaav1dec", "av1parse ! vaav1dec"),
            ("nvav1dec", "av1parse ! nvav1dec"),
            ("dav1ddec", "av1parse ! dav1ddec"),
        ],
    };

    for (factory, chain) in candidates {
        if gst::ElementFactory::find(factory).is_some() {
            return Ok((factory, chain));
        }
    }

    bail!("no GStreamer decoder plugin found for {codec:?}");
}
