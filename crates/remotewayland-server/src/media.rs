//! GStreamer video encoding pipeline with hardware probe and software fallback.

use anyhow::{bail, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use remotewayland_proto::VideoCodec;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// An encoded video access unit produced by GStreamer.
pub struct EncodedFrame {
    pub pts_ns: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub data: Vec<u8>,
}

/// Initialize GStreamer once per process.
pub fn init_gstreamer() -> Result<()> {
    gst::init().context("failed to initialize GStreamer")
}

/// Per-window GStreamer video encoder pipeline.
pub struct VideoEncoder {
    codec: VideoCodec,
    width: u32,
    height: u32,
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    frame_tx: mpsc::UnboundedSender<EncodedFrame>,
}

impl VideoEncoder {
    /// Create a new GStreamer encoder pipeline and return it along with a receiver for encoded frames.
    pub fn new(
        codec: VideoCodec,
        width: u32,
        height: u32,
    ) -> Result<(Self, mpsc::UnboundedReceiver<EncodedFrame>)> {
        let (enc_w, enc_h) = even_dims(width, height);
        let (frame_tx, frame_rx) = mpsc::unbounded_channel();
        let (pipeline, appsrc) =
            build_encoder_pipeline(codec, enc_w, enc_h, frame_tx.clone())?;
        Ok((
            Self {
                codec,
                width: enc_w,
                height: enc_h,
                pipeline,
                appsrc,
                frame_tx,
            },
            frame_rx,
        ))
    }

    /// Push a raw RGBA frame into the encoder, rebuilding the pipeline if dimensions changed.
    pub fn push_rgba(&mut self, width: u32, height: u32, pts_ns: u64, rgba: &[u8]) -> Result<()> {
        let (enc_w, enc_h) = even_dims(width, height);
        if enc_w != self.width || enc_h != self.height {
            debug!(
                old_w = self.width,
                old_h = self.height,
                new_w = enc_w,
                new_h = enc_h,
                "rebuilding GStreamer encoder pipeline for resize"
            );
            let _ = self.appsrc.end_of_stream();
            let _ = self.pipeline.set_state(gst::State::Null);
            let (pipeline, appsrc) =
                build_encoder_pipeline(self.codec, enc_w, enc_h, self.frame_tx.clone())?;
            self.pipeline = pipeline;
            self.appsrc = appsrc;
            self.width = enc_w;
            self.height = enc_h;
        }

        let padded;
        let payload = if enc_w == width && enc_h == height {
            rgba
        } else {
            padded = pad_rgba(rgba, width, height, enc_w, enc_h);
            &padded
        };

        let mut buffer = gst::Buffer::with_size(payload.len())
            .context("failed to allocate GStreamer input buffer")?;
        {
            let buf_mut = buffer.get_mut().context("failed to get mutable buffer")?;
            buf_mut.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            let mut map = buf_mut
                .map_writable()
                .context("failed to map GStreamer buffer writable")?;
            map.copy_from_slice(payload);
        }

        self.appsrc
            .push_buffer(buffer)
            .map_err(|e| anyhow::anyhow!("appsrc push_buffer failed: {e:?}"))?;

        Ok(())
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        let _ = self.appsrc.end_of_stream();
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn even_dims(w: u32, h: u32) -> (u32, u32) {
    let w = w.max(2);
    let h = h.max(2);
    ((w + 1) & !1, (h + 1) & !1)
}

fn pad_rgba(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    let src_row = (src_w * 4) as usize;
    let dst_row = (dst_w * 4) as usize;
    for y in 0..(src_h as usize) {
        let s_off = y * src_row;
        let d_off = y * dst_row;
        if s_off + src_row <= src.len() {
            out[d_off..d_off + src_row].copy_from_slice(&src[s_off..s_off + src_row]);
        }
    }
    out
}

fn select_encoder_spec(codec: VideoCodec) -> Result<(&'static str, &'static str)> {
    let candidates: &[(&str, &str)] = match codec {
        VideoCodec::H264 => &[
            ("vah264lpenc", "vah264lpenc ! h264parse config-interval=-1"),
            ("vah264enc", "vah264enc ! h264parse config-interval=-1"),
            (
                "nvh264enc",
                "nvh264enc zerolatency=true ! h264parse config-interval=-1",
            ),
            ("v4l2h264enc", "v4l2h264enc ! h264parse config-interval=-1"),
            (
                "x264enc",
                "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=60 ! h264parse config-interval=-1",
            ),
            ("openh264enc", "openh264enc ! h264parse config-interval=-1"),
        ],
        VideoCodec::Hevc => &[
            ("vah265lpenc", "vah265lpenc ! h265parse config-interval=-1"),
            ("vah265enc", "vah265enc ! h265parse config-interval=-1"),
            (
                "nvh265enc",
                "nvh265enc zerolatency=true ! h265parse config-interval=-1",
            ),
            ("v4l2h265enc", "v4l2h265enc ! h265parse config-interval=-1"),
            (
                "x265enc",
                "x265enc tune=zerolatency speed-preset=ultrafast ! h265parse config-interval=-1",
            ),
        ],
        VideoCodec::Av1 => &[
            ("vaav1lpenc", "vaav1lpenc ! av1parse"),
            ("vaav1enc", "vaav1enc ! av1parse"),
            ("nvav1enc", "nvav1enc ! av1parse"),
            ("svtav1enc", "svtav1enc ! av1parse"),
            ("rav1enc", "rav1enc ! av1parse"),
        ],
    };

    for (factory, chain) in candidates {
        if gst::ElementFactory::find(factory).is_some() {
            return Ok((factory, chain));
        }
    }

    bail!("no GStreamer encoder plugin found for {codec:?}");
}

fn build_encoder_pipeline(
    codec: VideoCodec,
    width: u32,
    height: u32,
    frame_tx: mpsc::UnboundedSender<EncodedFrame>,
) -> Result<(gst::Pipeline, gst_app::AppSrc)> {
    let (factory, enc_chain) = select_encoder_spec(codec)?;
    info!(
        ?codec,
        encoder = factory,
        width,
        height,
        "creating GStreamer video encoder"
    );

    let pipeline_str = format!(
        "appsrc name=src is-live=true format=time ! videoconvert ! video/x-raw,format=NV12 ! {enc_chain} ! video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=sink sync=false drop=false"
    );
    let pipeline_str = if codec == VideoCodec::H264 {
        pipeline_str
    } else {
        format!(
            "appsrc name=src is-live=true format=time ! videoconvert ! {enc_chain} ! appsink name=sink sync=false drop=false"
        )
    };

    let element = gst::parse::launch(&pipeline_str)
        .with_context(|| format!("failed to parse GStreamer encoder pipeline: {pipeline_str}"))?;
    let pipeline = element
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow::anyhow!("parsed element is not a gst::Pipeline"))?;

    let appsrc = pipeline
        .by_name("src")
        .context("missing appsrc")?
        .downcast::<gst_app::AppSrc>()
        .map_err(|_| anyhow::anyhow!("src is not an AppSrc"))?;

    let video_info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgba, width, height)
        .fps(gst::Fraction::new(0, 1))
        .build()
        .context("failed to build VideoInfo")?;
    appsrc.set_caps(Some(&video_info.to_caps()?));

    let appsink = pipeline
        .by_name("sink")
        .context("missing appsink")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow::anyhow!("sink is not an AppSink"))?;

    pipeline
        .set_state(gst::State::Playing)
        .context("failed to set encoder pipeline to Playing")?;

    std::thread::Builder::new()
        .name("rw-gst-enc-pull".into())
        .spawn(move || {
            while let Ok(sample) = appsink.pull_sample() {
                if let Some(buf) = sample.buffer() {
                    let is_keyframe = !buf.flags().contains(gst::BufferFlags::DELTA_UNIT);
                    let pts_ns = buf.pts().map(|p| p.nseconds()).unwrap_or(0);
                    if let Ok(map) = buf.map_readable() {
                        if frame_tx
                            .send(EncodedFrame {
                                pts_ns,
                                width,
                                height,
                                is_keyframe,
                                data: map.as_slice().to_vec(),
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            warn!("GStreamer encoder pull thread stopped");
        })
        .context("failed to spawn encoder pull thread")?;

    Ok((pipeline, appsrc))
}
