//! Server-side PipeWire/PulseAudio audio bridge (48kHz S16LE PCM compressed with LZ4).

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Name of the isolated virtual sink created for remote application speaker output.
pub const VIRTUAL_SINK_NAME: &str = "remotewayland_sink";
/// Name of the isolated virtual sink whose `.monitor` acts as the remote microphone source.
pub const VIRTUAL_MIC_NAME: &str = "remotewayland_mic";

/// 20ms of 48kHz stereo 16-bit PCM = 960 frames * 4 bytes = 3840 bytes.
const PLAYBACK_CHUNK_BYTES: usize = 3840;

/// An LZ4-compressed PCM audio packet captured from the server's virtual speaker sink.
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

/// Start capturing server audio from `remotewayland_sink.monitor` as LZ4-compressed 48kHz stereo S16LE packets.
pub fn start_playback_capture() -> Result<mpsc::UnboundedReceiver<EncodedAudioPacket>> {
    ensure_virtual_audio_devices();

    let monitor_dev = format!("{VIRTUAL_SINK_NAME}.monitor");
    let mut child = Command::new("parec")
        .args([
            "--raw",
            "--format=s16le",
            "--rate=48000",
            "--channels=2",
            "--latency-msec=20",
            &format!("--device={monitor_dev}"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn parec for server audio capture")?;

    let mut stdout = child.stdout.take().context("missing parec stdout")?;
    info!(device = %monitor_dev, "started server audio capture (parec)");

    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("rw-srv-audio-cap".into())
        .spawn(move || {
            let start = Instant::now();
            let mut buf = [0u8; PLAYBACK_CHUNK_BYTES];
            let mut silent_chunks = 0u32;
            while stdout.read_exact(&mut buf).is_ok() {
                if buf.iter().all(|&b| b == 0) {
                    silent_chunks = silent_chunks.saturating_add(1);
                    if silent_chunks > 4 {
                        continue;
                    }
                } else {
                    silent_chunks = 0;
                }
                let compressed = lz4_flex::compress_prepend_size(&buf);
                let pts_ns = start.elapsed().as_nanos() as u64;
                if tx
                    .send(EncodedAudioPacket {
                        pts_ns,
                        data: compressed,
                    })
                    .is_err()
                {
                    break;
                }
            }
            let _ = child.kill();
            let _ = child.wait();
            warn!("server audio capture thread stopped");
        })
        .context("failed to spawn server audio capture thread")?;

    Ok(rx)
}

/// Receives LZ4-compressed 48kHz mono S16LE microphone packets from the client and feeds `remotewayland_mic`.
pub struct AudioMicInjector {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
}

impl AudioMicInjector {
    pub fn new() -> Result<Self> {
        ensure_virtual_audio_devices();

        let mut child = Command::new("pacat")
            .args([
                "--raw",
                "--format=s16le",
                "--rate=48000",
                "--channels=1",
                "--latency-msec=20",
                &format!("--device={VIRTUAL_MIC_NAME}"),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn pacat for server mic injection")?;

        let stdin = child.stdin.take().context("missing pacat stdin")?;
        info!(device = VIRTUAL_MIC_NAME, "started server mic injection (pacat)");

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
        })
    }

    pub fn push_audio(&self, _pts_ns: u64, data: &[u8]) -> Result<()> {
        let pcm = lz4_flex::decompress_size_prepended(data)
            .context("failed to decompress mic audio packet")?;
        if let Ok(mut guard) = self.stdin.lock() {
            guard.write_all(&pcm)?;
        }
        Ok(())
    }
}

impl Drop for AudioMicInjector {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
