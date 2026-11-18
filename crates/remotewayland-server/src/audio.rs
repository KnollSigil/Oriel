//! Server-side PipeWire/PulseAudio <-> Opus audio bridge via GStreamer.

use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Name of the isolated virtual sink created for remote application speaker output.
pub const VIRTUAL_SINK_NAME: &str = "remotewayland_sink";
/// Name of the isolated virtual sink whose `.monitor` acts as the remote microphone source.
pub const VIRTUAL_MIC_NAME: &str = "remotewayland_mic";

/// An Opus-encoded audio packet captured from the server's audio session.
pub struct EncodedAudioPacket {
    pub pts_ns: u64,
    pub data: Vec<u8>,
}

/// Ensure virtual PulseAudio/PipeWire null-sinks exist for isolated speaker capture and mic input.
pub fn ensure_virtual_audio_devices() {
    let mut sinks_out = Command::new("pactl")
        .args(["list", "short", "sinks"])
        .stderr(Stdio::null())
        .output();

    if !matches!(&sinks_out, Ok(out) if out.status.success()) {
        // Attempt to start a user PipeWire + WirePlumber + PipeWire-Pulse daemon set if not running.
        for bin in ["pipewire", "wireplumber", "pipewire-pulse"] {
            let _ = Command::new(bin)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
        }
        for _ in 0..10 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            sinks_out = Command::new("pactl")
                .args(["list", "short", "sinks"])
                .stderr(Stdio::null())
                .output();
            if matches!(&sinks_out, Ok(out) if out.status.success()) {
                break;
            }
        }
    }

    let existing = match sinks_out {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        _ => {
            warn!("pactl could not connect to PulseAudio/PipeWire server; virtual sink setup skipped");
            return;
        }
    };

    if !existing.contains(VIRTUAL_SINK_NAME) {
        let status = Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                &format!("sink_name={VIRTUAL_SINK_NAME}"),
                "sink_properties=device.description=RemoteWayland_Speaker",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(s) if s.success()) {
            info!(sink = VIRTUAL_SINK_NAME, "created isolated virtual speaker sink");
        }
    }

    if !existing.contains(VIRTUAL_MIC_NAME) {
        let status = Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                &format!("sink_name={VIRTUAL_MIC_NAME}"),
                "sink_properties=device.description=RemoteWayland_Mic",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(s) if s.success()) {
            info!(sink = VIRTUAL_MIC_NAME, "created isolated virtual microphone sink");
        }
    }
}

/// Start capturing server audio (for playback on the client) as 48kHz stereo Opus packets.
pub fn start_playback_capture() -> Result<mpsc::UnboundedReceiver<EncodedAudioPacket>> {
    ensure_virtual_audio_devices();

    let candidates: Vec<(&str, String)> = vec![
        (
            "pulsesrc",
            format!("pulsesrc device={VIRTUAL_SINK_NAME}.monitor"),
        ),
        (
            "pipewiresrc",
            format!(
                "pipewiresrc target-object={VIRTUAL_SINK_NAME} stream-properties=\"props,stream.capture.sink=true\""
            ),
        ),
        ("pulsesrc", "pulsesrc".to_string()),
        ("autoaudiosrc", "autoaudiosrc".to_string()),
    ];

    let mut chosen = None;
    for (factory, src_spec) in candidates {
        if gst::ElementFactory::find(factory).is_none() {
            continue;
        }
        let pipeline_str = format!(
            "{src_spec} ! audioconvert ! audioresample ! audio/x-raw,rate=48000,channels=2 ! opusenc frame-size=20 bitrate=96000 ! appsink name=sink sync=false drop=true max-buffers=8"
        );
        let Ok(elem) = gst::parse::launch(&pipeline_str) else {
            continue;
        };
        let Ok(pipeline) = elem.downcast::<gst::Pipeline>() else {
            continue;
        };
        if pipeline.set_state(gst::State::Playing).is_err() {
            warn!(src = %src_spec, "audio capture source failed to enter Playing state; trying fallback");
            let _ = pipeline.set_state(gst::State::Null);
            continue;
        }
        info!(src = %src_spec, "starting server audio capture pipeline");
        chosen = Some(pipeline);
        break;
    }

    let pipeline = chosen.context("all server audio capture sources failed to start")?;
    let appsink = pipeline
        .by_name("sink")
        .context("missing appsink")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow::anyhow!("sink is not an AppSink"))?;

    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("rw-srv-audio-cap".into())
        .spawn(move || {
            while let Ok(sample) = appsink.pull_sample() {
                if let Some(buf) = sample.buffer() {
                    let pts_ns = buf.pts().map(|p| p.nseconds()).unwrap_or(0);
                    if let Ok(map) = buf.map_readable() {
                        if tx
                            .send(EncodedAudioPacket {
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
            warn!("server audio capture thread stopped");
        })
        .context("failed to spawn server audio capture thread")?;

    Ok(rx)
}

/// Receives Opus microphone packets from the client and injects them into the server's virtual mic.
pub struct AudioMicInjector {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
}

impl AudioMicInjector {
    pub fn new() -> Result<Self> {
        ensure_virtual_audio_devices();

        let candidates: Vec<(&str, String)> = vec![
            ("pulsesink", format!("pulsesink device={VIRTUAL_MIC_NAME}")),
            ("pipewiresink", "pipewiresink".to_string()),
            ("pulsesink", "pulsesink".to_string()),
            ("autoaudiosink", "autoaudiosink".to_string()),
        ];

        let caps = gst::Caps::builder("audio/x-opus")
            .field("rate", 48000i32)
            .field("channels", 1i32)
            .field("channel-mapping-family", 0i32)
            .field("stream-count", 1i32)
            .field("coupled-count", 0i32)
            .build();

        for (factory, sink_spec) in candidates {
            if gst::ElementFactory::find(factory).is_none() {
                continue;
            }
            let pipeline_str = format!(
                "appsrc name=src is-live=true format=time ! opusdec ! audioconvert ! audioresample ! {sink_spec} sync=false"
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
                info!(sink = %sink_spec, "starting server mic injection pipeline");
                return Ok(Self { pipeline, appsrc });
            }
            let _ = pipeline.set_state(gst::State::Null);
        }

        bail!("no GStreamer audio sink could be started for server mic injection")
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

impl Drop for AudioMicInjector {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
