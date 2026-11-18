//! Client-side PipeWire/PulseAudio <-> Opus audio bridge via GStreamer.

use anyhow::{bail, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// An Opus-encoded microphone packet captured from the client's audio session.
pub struct EncodedMicPacket {
    pub pts_ns: u64,
    pub data: Vec<u8>,
}

/// Plays incoming 48kHz stereo Opus audio packets from the server onto the client's audio output.
pub struct AudioPlaybackSink {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
}

impl AudioPlaybackSink {
    pub fn new() -> Result<Self> {
        let caps = gst::Caps::builder("audio/x-opus")
            .field("rate", 48000i32)
            .field("channels", 2i32)
            .field("channel-mapping-family", 0i32)
            .field("stream-count", 1i32)
            .field("coupled-count", 1i32)
            .build();

        for sink_el in ["pulsesink", "pipewiresink", "autoaudiosink"] {
            if gst::ElementFactory::find(sink_el).is_none() {
                continue;
            }
            let pipeline_str = format!(
                "appsrc name=src is-live=true format=time ! opusdec ! audioconvert ! audioresample ! {sink_el} sync=false"
            );
            let Ok(elem) = gst::parse::launch(&pipeline_str) else {
                continue;
            };
            let Ok(pipeline) = elem.downcast::<gst::Pipeline>() else {
                continue;
            };
            let Ok(appsrc) = pipeline
                .by_name("src")
                .context("missing appsrc")
                .and_then(|e| {
                    e.downcast::<gst_app::AppSrc>()
                        .map_err(|_| anyhow::anyhow!("not AppSrc"))
                })
            else {
                continue;
            };
            appsrc.set_caps(Some(&caps));
            if pipeline.set_state(gst::State::Playing).is_ok() {
                info!(sink = sink_el, "starting client audio playback pipeline");
                return Ok(Self { pipeline, appsrc });
            }
            let _ = pipeline.set_state(gst::State::Null);
        }

        bail!("no GStreamer audio playback plugin could enter Playing state")
    }

    pub fn push_opus(&self, pts_ns: u64, data: &[u8]) -> Result<()> {
        let mut buffer =
            gst::Buffer::with_size(data.len()).context("failed to allocate audio buffer")?;
        {
            let buf_mut = buffer.get_mut().context("failed to get mutable buffer")?;
            buf_mut.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            let mut map = buf_mut
                .map_writable()
                .context("failed to map audio buffer writable")?;
            map.copy_from_slice(data);
        }
        self.appsrc
            .push_buffer(buffer)
            .map_err(|e| anyhow::anyhow!("audio appsrc push_buffer failed: {e:?}"))?;
        Ok(())
    }
}

impl Drop for AudioPlaybackSink {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Start capturing the client's microphone as 48kHz mono Opus packets to send to the server.
pub fn start_mic_capture() -> Result<mpsc::UnboundedReceiver<EncodedMicPacket>> {
    let mut chosen = None;
    for src_el in ["pulsesrc", "pipewiresrc", "autoaudiosrc"] {
        if gst::ElementFactory::find(src_el).is_none() {
            continue;
        }
        let pipeline_str = format!(
            "{src_el} ! audioconvert ! audioresample ! audio/x-raw,rate=48000,channels=1 ! opusenc frame-size=20 bitrate=48000 ! appsink name=sink sync=false drop=true max-buffers=8"
        );
        let Ok(elem) = gst::parse::launch(&pipeline_str) else {
            continue;
        };
        let Ok(pipeline) = elem.downcast::<gst::Pipeline>() else {
            continue;
        };
        if pipeline.set_state(gst::State::Playing).is_err() {
            let _ = pipeline.set_state(gst::State::Null);
            continue;
        }
        info!(src = src_el, "starting client mic capture pipeline");
        chosen = Some(pipeline);
        break;
    }

    let pipeline = chosen.context("no client microphone source available")?;
    let appsink = pipeline
        .by_name("sink")
        .context("missing appsink")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow::anyhow!("sink is not an AppSink"))?;

    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("rw-cli-mic-cap".into())
        .spawn(move || {
            while let Ok(sample) = appsink.pull_sample() {
                if let Some(buf) = sample.buffer() {
                    let pts_ns = buf.pts().map(|p| p.nseconds()).unwrap_or(0);
                    if let Ok(map) = buf.map_readable() {
                        if tx
                            .send(EncodedMicPacket {
                                pts_ns,
                                data: map.as_slice().to_vec(),
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            let _ = pipeline.set_state(gst::State::Null);
            warn!("client mic capture thread stopped");
        })
        .context("failed to spawn client mic capture thread")?;

    Ok(rx)
}
