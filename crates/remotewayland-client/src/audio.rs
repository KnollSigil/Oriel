//! Client-side native PipeWire + Opus audio bridge.
//!
//! Connects directly to the user's host PipeWire session via `pipewire-rs` for
//! low-latency 48kHz stereo speaker playback and 48kHz mono microphone capture,
//! compressed with 20ms Opus frames.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use pipewire as pw;
use pw::{properties::properties, spa, types::ObjectType};
use spa::pod::Pod;
use tokio::sync::{mpsc, watch};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_SAMPLES: usize = 960; // 20ms @ 48kHz
const PLAYBACK_CHANNELS: u32 = 2;
const MIC_CHANNELS: u32 = 1;
const MAX_OPUS_PACKET_BYTES: usize = 4000;
/// Cap client playback jitter buffer at ~200ms of 48kHz stereo S16LE (38,400 bytes).
const MAX_PLAYBACK_RINGBUF_BYTES: usize = (SAMPLE_RATE as usize) * 2 * 2 / 5;

/// An Opus-encoded 48kHz mono microphone packet captured from the client's default source.
pub struct EncodedMicPacket {
    pub pts_ns: u64,
    pub data: Vec<u8>,
}

fn build_pcm_format_pod(channels: u32) -> Result<Vec<u8>> {
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::S16LE);
    audio_info.set_rate(SAMPLE_RATE);
    audio_info.set_channels(channels);
    let mut position = [0; spa::param::audio::MAX_CHANNELS];
    if channels == 2 {
        position[0] = spa::sys::SPA_AUDIO_CHANNEL_FL;
        position[1] = spa::sys::SPA_AUDIO_CHANNEL_FR;
    } else if channels == 1 {
        position[0] = spa::sys::SPA_AUDIO_CHANNEL_MONO;
    }
    audio_info.set_position(position);

    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };

    let (cursor, _) = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| anyhow::anyhow!("failed to serialize SPA audio format pod: {e:?}"))?;

    Ok(cursor.into_inner())
}

/// Plays incoming Opus-encoded 48kHz stereo audio packets onto the client's PipeWire output.
pub struct AudioPlaybackSink {
    ringbuf: Arc<Mutex<VecDeque<u8>>>,
    decoder: Mutex<opus::Decoder>,
    quit_tx: pw::channel::Sender<()>,
}

impl Drop for AudioPlaybackSink {
    fn drop(&mut self) {
        let _ = self.quit_tx.send(());
    }
}

impl AudioPlaybackSink {
    pub fn new() -> Result<Self> {
        Self::new_for_app("RemoteWayland", "RemoteWayland Audio", "", "Music")
    }

    pub fn new_for_app(
        app_name: &str,
        media_name: &str,
        icon_name: &str,
        media_role: &str,
    ) -> Result<Self> {
        let decoder = opus::Decoder::new(SAMPLE_RATE, opus::Channels::Stereo)
            .context("failed to create client Opus stereo decoder")?;
        let ringbuf = Arc::new(Mutex::new(VecDeque::with_capacity(
            MAX_PLAYBACK_RINGBUF_BYTES,
        )));
        let thread_ringbuf = Arc::clone(&ringbuf);
        let (quit_tx, quit_rx) = pw::channel::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let app_name = app_name.to_string();
        let media_name = media_name.to_string();
        let icon_name = icon_name.to_string();
        let media_role = media_role.to_string();

        std::thread::Builder::new()
            .name("pw-client-playback".to_string())
            .spawn(move || {
                if let Err(err) = run_playback_loop(
                    thread_ringbuf,
                    quit_rx,
                    &app_name,
                    &media_name,
                    &icon_name,
                    &media_role,
                    &ready_tx,
                ) {
                    tracing::warn!(error = %err, "client PipeWire playback loop exited");
                }
            })
            .context("failed to spawn pw-client-playback thread")?;

        ready_rx
            .recv()
            .context("pw-client-playback thread exited before initializing")??;

        Ok(Self {
            ringbuf,
            decoder: Mutex::new(decoder),
            quit_tx,
        })
    }

    pub fn push_audio(&self, _pts_ns: u64, data: &[u8]) -> Result<()> {
        let mut pcm = [0i16; FRAME_SAMPLES * PLAYBACK_CHANNELS as usize * 2];
        let decoded_frames = {
            let mut dec = self
                .decoder
                .lock()
                .map_err(|_| anyhow::anyhow!("playback decoder lock poisoned"))?;
            dec.decode(data, &mut pcm, false)
                .context("Opus playback decode failed")?
        };

        let total_samples = decoded_frames * PLAYBACK_CHANNELS as usize;
        let mut buf = self
            .ringbuf
            .lock()
            .map_err(|_| anyhow::anyhow!("playback ringbuf lock poisoned"))?;
        for &sample in &pcm[..total_samples] {
            let le = sample.to_le_bytes();
            buf.push_back(le[0]);
            buf.push_back(le[1]);
        }
        while buf.len() > MAX_PLAYBACK_RINGBUF_BYTES {
            buf.pop_front();
        }
        Ok(())
    }
}

fn run_playback_loop(
    ringbuf: Arc<Mutex<VecDeque<u8>>>,
    quit_rx: pw::channel::Receiver<()>,
    app_name: &str,
    media_name: &str,
    icon_name: &str,
    media_role: &str,
    ready_tx: &std::sync::mpsc::Sender<Result<()>>,
) -> Result<()> {
    pw::init();

    let mainloop = match pw::main_loop::MainLoopRc::new(None) {
        Ok(m) => m,
        Err(e) => {
            let err = anyhow::anyhow!("PipeWire MainLoopRc::new (playback) failed: {e}");
            let _ = ready_tx.send(Err(anyhow::anyhow!("{err}")));
            return Err(err);
        }
    };

    let ml_quit_chan = mainloop.clone();
    let _quit_receiver = quit_rx.attach(mainloop.loop_(), move |_| {
        ml_quit_chan.quit();
    });

    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| anyhow::anyhow!("PipeWire ContextRc::new (playback) failed: {e}"))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| anyhow::anyhow!("PipeWire context.connect_rc (playback) failed: {e}"))?;

    let app_label = if app_name.is_empty() {
        "Remote App"
    } else {
        app_name
    };
    let media_label = if media_name.is_empty() {
        app_label
    } else {
        media_name
    };
    let role_label = if media_role.is_empty() {
        "Music"
    } else {
        media_role
    };

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => role_label,
        *pw::keys::APP_NAME => app_label,
        *pw::keys::MEDIA_NAME => media_label,
        *pw::keys::NODE_DESCRIPTION => app_label,
        *pw::keys::AUDIO_CHANNELS => "2",
        "audio.position" => "FL,FR",
    };
    if !icon_name.is_empty() {
        props.insert(*pw::keys::APP_ICON_NAME, icon_name);
    }

    let stream = pw::stream::StreamBox::new(&core, app_label, props)
        .map_err(|e| anyhow::anyhow!("PipeWire Stream::new (playback) failed: {e}"))?;

    let ml_quit = mainloop.clone();
    let _listener = stream
        .add_local_listener_with_user_data(ringbuf)
        .process(move |stream, rb| {
            if Arc::strong_count(rb) == 1 {
                ml_quit.quit();
                return;
            }
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let stride = std::mem::size_of::<i16>() * PLAYBACK_CHANNELS as usize;
            let data = &mut datas[0];
            let written_bytes = if let Some(slice) = data.data() {
                let target_bytes = (FRAME_SAMPLES * stride).min(slice.len());
                let mut guard = rb.lock().ok();
                for byte in &mut slice[..target_bytes] {
                    *byte = guard.as_mut().and_then(|q| q.pop_front()).unwrap_or(0);
                }
                target_bytes
            } else {
                0
            };
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = stride as i32;
            *chunk.size_mut() = written_bytes as u32;
        })
        .register()
        .map_err(|e| anyhow::anyhow!("PipeWire stream register (playback) failed: {e}"))?;

    let pod_bytes = build_pcm_format_pod(PLAYBACK_CHANNELS)?;
    let mut params = [Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid SPA Pod bytes for playback"))?];

    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| anyhow::anyhow!("PipeWire stream.connect (playback) failed: {e}"))?;

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}

/// Live host microphone routing & mute configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MicConfig {
    pub target: Option<String>,
    pub muted: bool,
}

static MIC_CONFIG_TX: OnceLock<watch::Sender<MicConfig>> = OnceLock::new();

fn mic_config_sender() -> &'static watch::Sender<MicConfig> {
    MIC_CONFIG_TX.get_or_init(|| {
        let (tx, _rx) = watch::channel(MicConfig::default());
        tx
    })
}

/// Subscribe to live host microphone routing & mute changes.
pub fn subscribe_mic_config() -> watch::Receiver<MicConfig> {
    mic_config_sender().subscribe()
}

/// Sets the preferred host PipeWire microphone source (node name or serial).
/// Pass `None` to use the system default microphone.
pub fn set_mic_target(target: Option<String>) {
    let target = target.filter(|s| !s.is_empty());
    mic_config_sender().send_modify(|cfg| {
        cfg.target = target;
    });
}

/// Mutes or unmutes host microphone capture for remote applications.
pub fn set_mic_muted(muted: bool) {
    mic_config_sender().send_modify(|cfg| {
        cfg.muted = muted;
    });
}

fn default_mic_target() -> Option<String> {
    mic_config_sender().borrow().target.clone()
}

/// A host PipeWire microphone / audio input source discovered via the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMicSource {
    pub node_name: String,
    pub description: String,
}

/// Enumerates available host PipeWire input sources (`Audio/Source` / `Audio/Source/Virtual`).
pub fn list_host_mic_sources() -> Vec<HostMicSource> {
    std::thread::Builder::new()
        .name("pw-list-mics".to_string())
        .spawn(list_host_mic_sources_sync)
        .ok()
        .and_then(|h| h.join().ok())
        .unwrap_or_default()
}

fn list_host_mic_sources_sync() -> Vec<HostMicSource> {
    pw::init();
    let Ok(mainloop) = pw::main_loop::MainLoopRc::new(None) else {
        return Vec::new();
    };
    let Ok(context) = pw::context::ContextRc::new(&mainloop, None) else {
        return Vec::new();
    };
    let Ok(core) = context.connect_rc(None) else {
        return Vec::new();
    };
    let Ok(registry) = core.get_registry_rc() else {
        return Vec::new();
    };
    let Ok(pending) = core.sync(0) else {
        return Vec::new();
    };

    let sources: Rc<RefCell<Vec<HostMicSource>>> = Rc::new(RefCell::new(Vec::new()));
    let sources_reg = Rc::clone(&sources);
    let ml_quit = mainloop.clone();
    let ml_timeout = mainloop.clone();

    let _timer = mainloop.loop_().add_timer(move |_| {
        ml_timeout.quit();
    });
    let _ = _timer.update_timer(Some(Duration::from_secs(1)), None);

    let _core_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == pending {
                ml_quit.quit();
            }
        })
        .register();

    let _reg_listener = registry
        .add_listener_local()
        .global(move |obj| {
            if obj.type_ != ObjectType::Node {
                return;
            }
            let Some(props) = obj.props else {
                return;
            };
            let media_class = props.get("media.class").unwrap_or("");
            if media_class != "Audio/Source" && media_class != "Audio/Source/Virtual" {
                return;
            }
            let Some(node_name) = props.get("node.name").filter(|s| !s.is_empty()) else {
                return;
            };
            let description = props
                .get("node.description")
                .filter(|s| !s.is_empty())
                .or_else(|| props.get("node.nick").filter(|s| !s.is_empty()))
                .unwrap_or(node_name)
                .to_string();
            sources_reg.borrow_mut().push(HostMicSource {
                node_name: node_name.to_string(),
                description,
            });
        })
        .register();

    mainloop.run();

    let mut out = sources.borrow().clone();
    out.sort_by(|a, b| a.description.cmp(&b.description));
    out
}

/// Handle keeping the client's PipeWire microphone capture stream alive.
/// Dropping this handle immediately closes the host microphone stream.
pub struct MicCaptureHandle {
    quit_tx: pw::channel::Sender<()>,
}

impl Drop for MicCaptureHandle {
    fn drop(&mut self) {
        let _ = self.quit_tx.send(());
    }
}

/// Captures 48kHz mono audio from the client's PipeWire microphone source
/// (optionally pinned to `target_object` node name/serial, or [`set_mic_target`])
/// and yields 20ms Opus packets.
pub fn start_mic_capture(
    target_object: Option<&str>,
) -> Result<(MicCaptureHandle, mpsc::UnboundedReceiver<EncodedMicPacket>)> {
    let (tx, rx) = mpsc::unbounded_channel();
    let (quit_tx, quit_rx) = pw::channel::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    let target = target_object
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(default_mic_target);

    std::thread::Builder::new()
        .name("pw-client-mic".to_string())
        .spawn(move || {
            if let Err(err) = run_mic_capture_loop(tx, quit_rx, target.as_deref(), &ready_tx) {
                tracing::warn!(error = %err, "client PipeWire mic capture loop exited");
            }
        })
        .context("failed to spawn pw-client-mic thread")?;

    ready_rx
        .recv()
        .context("pw-client-mic thread exited before initializing")??;

    Ok((MicCaptureHandle { quit_tx }, rx))
}

fn run_mic_capture_loop(
    tx: mpsc::UnboundedSender<EncodedMicPacket>,
    quit_rx: pw::channel::Receiver<()>,
    target_object: Option<&str>,
    ready_tx: &std::sync::mpsc::Sender<Result<()>>,
) -> Result<()> {
    pw::init();

    let mainloop = match pw::main_loop::MainLoopRc::new(None) {
        Ok(m) => m,
        Err(e) => {
            let err = anyhow::anyhow!("PipeWire MainLoopRc::new (client mic) failed: {e}");
            let _ = ready_tx.send(Err(anyhow::anyhow!("{err}")));
            return Err(err);
        }
    };

    let ml_quit_chan = mainloop.clone();
    let _quit_receiver = quit_rx.attach(mainloop.loop_(), move |_| {
        ml_quit_chan.quit();
    });

    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| anyhow::anyhow!("PipeWire ContextRc::new (client mic) failed: {e}"))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| anyhow::anyhow!("PipeWire context.connect_rc (client mic) failed: {e}"))?;

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::APP_NAME => "RemoteWayland",
        *pw::keys::MEDIA_NAME => "RemoteWayland Microphone",
        *pw::keys::NODE_DESCRIPTION => "RemoteWayland Microphone",
    };
    if let Some(target) = target_object.filter(|s| !s.is_empty()) {
        props.insert("target.object", target);
    }

    let stream = pw::stream::StreamBox::new(&core, "RemoteWayland Microphone", props)
        .map_err(|e| anyhow::anyhow!("PipeWire Stream::new (client mic) failed: {e}"))?;

    let encoder = opus::Encoder::new(SAMPLE_RATE, opus::Channels::Mono, opus::Application::Voip)
        .context("failed to create client Opus mono encoder")?;

    struct MicCaptureState {
        encoder: opus::Encoder,
        pcm_accum: Vec<i16>,
        opus_buf: Vec<u8>,
        start: Instant,
        tx: mpsc::UnboundedSender<EncodedMicPacket>,
        mainloop: pw::main_loop::MainLoopRc,
    }

    let state = MicCaptureState {
        encoder,
        pcm_accum: Vec::with_capacity(FRAME_SAMPLES * 2),
        opus_buf: vec![0u8; MAX_OPUS_PACKET_BYTES],
        start: Instant::now(),
        tx,
        mainloop: mainloop.clone(),
    };

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .process(|stream, st| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let valid_bytes = data.chunk().size() as usize;
            let Some(bytes) = data.data() else {
                return;
            };
            let slice = &bytes[..valid_bytes.min(bytes.len())];
            for &chunk in slice.as_chunks::<2>().0 {
                st.pcm_accum.push(i16::from_le_bytes(chunk));
            }

            while st.pcm_accum.len() >= FRAME_SAMPLES {
                let mut send_failed = false;
                {
                    let frame = &st.pcm_accum[..FRAME_SAMPLES];
                    if !frame.iter().all(|&s| s.abs() <= 16) {
                        if let Ok(encoded_len) = st.encoder.encode(frame, &mut st.opus_buf) {
                            let pkt = EncodedMicPacket {
                                pts_ns: st.start.elapsed().as_nanos() as u64,
                                data: st.opus_buf[..encoded_len].to_vec(),
                            };
                            send_failed = st.tx.send(pkt).is_err();
                        }
                    }
                }
                st.pcm_accum.drain(..FRAME_SAMPLES);
                if send_failed {
                    st.mainloop.quit();
                    return;
                }
            }
        })
        .register()
        .map_err(|e| anyhow::anyhow!("PipeWire stream register (client mic) failed: {e}"))?;

    let pod_bytes = build_pcm_format_pod(MIC_CHANNELS)?;
    let mut params = [Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid SPA Pod bytes for client mic"))?];

    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| anyhow::anyhow!("PipeWire stream.connect (client mic) failed: {e}"))?;

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}
