//! Client-side PipeWire/PulseAudio audio bridge (48kHz S16LE PCM compressed with LZ4).

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// 20ms of 48kHz mono 16-bit PCM = 960 frames * 2 bytes = 1920 bytes.
const MIC_CHUNK_BYTES: usize = 1920;

/// An LZ4-compressed microphone packet captured from the client's audio input.
pub struct EncodedMicPacket {
    pub pts_ns: u64,
    pub data: Vec<u8>,
}

/// Plays incoming LZ4-compressed 48kHz stereo S16LE audio packets onto the client's default output.
pub struct AudioPlaybackSink {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
}

impl AudioPlaybackSink {
    pub fn new() -> Result<Self> {
        let mut child = Command::new("pacat")
            .args([
                "--raw",
                "--format=s16le",
                "--rate=48000",
                "--channels=2",
                "--latency-msec=20",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn pacat for client audio playback")?;

        let stdin = child.stdin.take().context("missing pacat stdin")?;
        info!("started client audio playback (pacat)");

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
        })
    }

    pub fn push_audio(&self, _pts_ns: u64, data: &[u8]) -> Result<()> {
        let pcm = lz4_flex::decompress_size_prepended(data)
            .context("failed to decompress playback audio packet")?;
        if let Ok(mut guard) = self.stdin.lock() {
            guard.write_all(&pcm)?;
        }
        Ok(())
    }
}

impl Drop for AudioPlaybackSink {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Start capturing the client's default microphone as LZ4-compressed 48kHz mono S16LE packets.
pub fn start_mic_capture() -> Result<mpsc::UnboundedReceiver<EncodedMicPacket>> {
    let mut child = Command::new("parec")
        .args([
            "--raw",
            "--format=s16le",
            "--rate=48000",
            "--channels=1",
            "--latency-msec=20",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn parec for client mic capture")?;

    let mut stdout = child.stdout.take().context("missing parec stdout")?;
    info!("started client mic capture (parec)");

    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("rw-cli-mic-cap".into())
        .spawn(move || {
            let start = Instant::now();
            let mut buf = [0u8; MIC_CHUNK_BYTES];
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
                    .send(EncodedMicPacket {
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
            warn!("client mic capture thread stopped");
        })
        .context("failed to spawn client mic capture thread")?;

    Ok(rx)
}
