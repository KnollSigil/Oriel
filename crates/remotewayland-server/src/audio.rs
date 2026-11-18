//! Server-side isolated PipeWire + Opus audio bridge.
//!
//! Spawns a private, session-scoped `pipewire` + `wireplumber` + `pipewire-pulse`
//! stack under `$XDG_RUNTIME_DIR/remotewayland-pw-<pid>` so remote GUI applications
//! never touch the host user's desktop audio. Exposes native `pipewire-rs` virtual
//! `Audio/Sink` and `Audio/Source` nodes compressed with 48kHz Opus.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use pipewire as pw;
use pw::{properties::properties, spa, types::ObjectType};
use spa::pod::Pod;
use tokio::sync::mpsc;

/// Name of the isolated virtual sink created for remote application speaker output.
pub const VIRTUAL_SINK_NAME: &str = "remotewayland_sink";
/// Name of the isolated virtual source created for remote application microphone input.
pub const VIRTUAL_MIC_NAME: &str = "remotewayland_mic";

const SAMPLE_RATE: u32 = 48_000;
const FRAME_SAMPLES: usize = 960; // 20ms @ 48kHz
const SINK_CHANNELS: u32 = 2;
const MIC_CHANNELS: u32 = 2;
const MAX_OPUS_PACKET_BYTES: usize = 4000;
/// Cap server mic jitter buffer at ~200ms of 48kHz stereo S16LE (38,400 bytes).
const MAX_MIC_RINGBUF_BYTES: usize = (SAMPLE_RATE as usize) * 2 * 2 / 5;

static MIC_RINGBUF: OnceLock<Arc<Mutex<VecDeque<u8>>>> = OnceLock::new();

fn shared_mic_ringbuf() -> &'static Arc<Mutex<VecDeque<u8>>> {
    MIC_RINGBUF.get_or_init(|| Arc::new(Mutex::new(VecDeque::with_capacity(MAX_MIC_RINGBUF_BYTES))))
}

struct IsolatedPwEnv {
    runtime_dir: PathBuf,
    _daemons: Mutex<Vec<Child>>,
}

static ISOLATED_PW: OnceLock<std::result::Result<IsolatedPwEnv, String>> = OnceLock::new();

/// Returns the isolated `PIPEWIRE_RUNTIME_DIR` path once initialized.
pub fn isolated_pw_runtime_dir() -> Option<&'static Path> {
    ISOLATED_PW
        .get()
        .and_then(|res| res.as_ref().ok())
        .map(|env| env.runtime_dir.as_path())
}

/// Ensures the isolated private PipeWire + WirePlumber + pipewire-pulse stack is running.
pub fn ensure_virtual_audio_devices() -> Result<()> {
    let res = ISOLATED_PW.get_or_init(init_isolated_pipewire);
    match res {
        Ok(_) => Ok(()),
        Err(err) => Err(anyhow::anyhow!("{err}")),
    }
}

fn init_isolated_pipewire() -> std::result::Result<IsolatedPwEnv, String> {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let runtime_dir = base.join(format!("remotewayland-pw-{}", std::process::id()));
    let pulse_dir = runtime_dir.join("pulse");

    std::fs::create_dir_all(&pulse_dir)
        .map_err(|e| format!("failed to create {}: {e}", pulse_dir.display()))?;

    // Point this process's libpipewire client connections at the isolated daemon.
    std::env::set_var("PIPEWIRE_RUNTIME_DIR", &runtime_dir);
    std::env::set_var("PULSE_RUNTIME_PATH", &pulse_dir);

    let mut daemons = Vec::new();
    for bin in ["pipewire", "wireplumber", "pipewire-pulse"] {
        let mut cmd = Command::new(bin);
        cmd.env("PIPEWIRE_RUNTIME_DIR", &runtime_dir)
            .env("PULSE_RUNTIME_PATH", &pulse_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn isolated {bin}: {e}"))?;
        daemons.push(child);
    }

    let pw_socket = runtime_dir.join("pipewire-0");
    let deadline = Instant::now() + Duration::from_secs(3);
    while !pw_socket.exists() {
        if Instant::now() >= deadline {
            for child in &mut daemons {
                let _ = child.kill();
            }
            return Err(format!(
                "timed out waiting for isolated PipeWire socket at {}",
                pw_socket.display()
            ));
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    tracing::info!(
        runtime_dir = %runtime_dir.display(),
        "isolated PipeWire + WirePlumber + pipewire-pulse session started"
    );

    Ok(IsolatedPwEnv {
        runtime_dir,
        _daemons: Mutex::new(daemons),
    })
}

/// An Opus-encoded 48kHz stereo audio packet captured from the server's virtual speaker sink.
pub struct EncodedAudioPacket {
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

/// A newly appeared remote application audio stream with its per-app Opus packet receiver.
pub struct PlaybackStreamEvent {
    pub stream_id: u32,
    pub app_name: String,
    pub media_name: String,
    pub icon_name: String,
    pub media_role: String,
    pub packets: mpsc::UnboundedReceiver<EncodedAudioPacket>,
}

/// Guard that shuts down the server's PipeWire playback/registry loop when dropped.
pub struct PlaybackBridgeGuard {
    quit_tx: pw::channel::Sender<()>,
}

impl Drop for PlaybackBridgeGuard {
    fn drop(&mut self) {
        let _ = self.quit_tx.send(());
    }
}

/// Channels produced by the server's PipeWire audio registry monitor.
pub struct PlaybackBridgeChannels {
    pub streams: mpsc::UnboundedReceiver<PlaybackStreamEvent>,
    pub mic_active: mpsc::UnboundedReceiver<bool>,
    pub guard: PlaybackBridgeGuard,
}

/// Starts the isolated default virtual speaker sink and watches the PipeWire registry
/// for per-application `Stream/Output/Audio` nodes (playback) and `Stream/Input/Audio`
/// nodes (microphone demand).
pub fn start_per_app_playback_capture() -> Result<PlaybackBridgeChannels> {
    ensure_virtual_audio_devices()?;

    let (tx, rx) = mpsc::unbounded_channel();
    let (mic_tx, mic_rx) = mpsc::unbounded_channel();
    let (quit_tx, quit_rx) = pw::channel::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

    std::thread::Builder::new()
        .name("pw-server-sink".to_string())
        .spawn(move || {
            if let Err(err) = run_per_app_playback_loop(tx, mic_tx, quit_rx, &ready_tx) {
                tracing::warn!(error = %err, "server PipeWire playback capture loop exited");
            }
        })
        .context("failed to spawn pw-server-sink thread")?;

    ready_rx
        .recv()
        .context("pw-server-sink thread exited before initializing")??;

    Ok(PlaybackBridgeChannels {
        streams: rx,
        mic_active: mic_rx,
        guard: PlaybackBridgeGuard { quit_tx },
    })
}

struct AppCaptureUserData {
    encoder: opus::Encoder,
    pcm_accum: Vec<i16>,
    opus_buf: Vec<u8>,
    start: Instant,
    tx: mpsc::UnboundedSender<EncodedAudioPacket>,
}

struct AppCaptureEntry {
    stream: pw::stream::StreamRc,
    _listener: pw::stream::StreamListener<AppCaptureUserData>,
    out_ports: Vec<(u32, String)>,
    cap_in_ports: Vec<(u32, String)>,
    linked_in_ports: HashSet<u32>,
    links: Vec<pw::link::Link>,
}

fn try_link_app(core: &pw::core::CoreRc, app_node_id: u32, entry: &mut AppCaptureEntry) {
    if entry.out_ports.is_empty() || entry.cap_in_ports.is_empty() {
        return;
    }
    let cap_node_id = entry.stream.node_id();
    if cap_node_id == u32::MAX {
        return;
    }

    let has_mono = entry.out_ports.iter().any(|(_, ch)| ch == "MONO");
    for (idx, (cap_port_id, cap_ch)) in entry.cap_in_ports.iter().enumerate() {
        if entry.linked_in_ports.contains(cap_port_id) {
            continue;
        }

        // Prefer exact channel match (FL->FL, FR->FR), then MONO->both, and only
        // fall back to positional indexing once all output channels are present.
        let out_port_id = entry
            .out_ports
            .iter()
            .find(|(_, ch)| !cap_ch.is_empty() && ch == cap_ch)
            .or_else(|| {
                if has_mono {
                    entry.out_ports.iter().find(|(_, ch)| ch == "MONO")
                } else if entry.out_ports.len() >= 2 {
                    entry.out_ports.get(idx % entry.out_ports.len())
                } else {
                    None
                }
            })
            .map(|(id, _)| *id);

        let Some(out_port_id) = out_port_id else {
            continue;
        };

        match core.create_object::<pw::link::Link>(
            "link-factory",
            &properties! {
                "link.output.node" => app_node_id.to_string(),
                "link.output.port" => out_port_id.to_string(),
                "link.input.node" => cap_node_id.to_string(),
                "link.input.port" => cap_port_id.to_string(),
                "object.linger" => "false",
            },
        ) {
            Ok(link) => {
                entry.links.push(link);
                entry.linked_in_ports.insert(*cap_port_id);
            }
            Err(err) => {
                tracing::warn!(
                    app_node_id,
                    cap_node_id,
                    error = %err,
                    "failed to link remote app audio port to per-app capture"
                );
            }
        }
    }
}

fn run_per_app_playback_loop(
    event_tx: mpsc::UnboundedSender<PlaybackStreamEvent>,
    mic_active_tx: mpsc::UnboundedSender<bool>,
    quit_rx: pw::channel::Receiver<()>,
    ready_tx: &std::sync::mpsc::Sender<Result<()>>,
) -> Result<()> {
    pw::init();

    let mainloop = match pw::main_loop::MainLoopRc::new(None) {
        Ok(m) => m,
        Err(e) => {
            let err = anyhow::anyhow!("PipeWire MainLoopRc::new failed: {e}");
            let _ = ready_tx.send(Err(anyhow::anyhow!("{err}")));
            return Err(err);
        }
    };

    let ml_quit_chan = mainloop.clone();
    let _quit_receiver = quit_rx.attach(mainloop.loop_(), move |_| {
        ml_quit_chan.quit();
    });

    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| anyhow::anyhow!("PipeWire ContextRc::new failed: {e}"))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| anyhow::anyhow!("PipeWire context.connect_rc failed: {e}"))?;

    // Keep a silent default Audio/Sink alive so WirePlumber always has a sink clock
    // driving all remote apps even while per-app capture streams attach/detach.
    let dummy_sink = pw::stream::StreamBox::new(
        &core,
        "remotewayland-sink",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Duplex",
            *pw::keys::MEDIA_CLASS => "Audio/Sink",
            *pw::keys::NODE_NAME => VIRTUAL_SINK_NAME,
            *pw::keys::NODE_DESCRIPTION => "RemoteWayland Speaker",
            *pw::keys::AUDIO_CHANNELS => "2",
            "audio.position" => "FL,FR",
        },
    )
    .map_err(|e| anyhow::anyhow!("PipeWire Stream::new (sink) failed: {e}"))?;

    let _dummy_listener = dummy_sink
        .add_local_listener()
        .process(|stream, _: &mut ()| {
            let _ = stream.dequeue_buffer();
        })
        .register()
        .map_err(|e| anyhow::anyhow!("PipeWire stream register (sink) failed: {e}"))?;

    let pod_bytes = build_pcm_format_pod(SINK_CHANNELS)?;
    let mut params = [Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid SPA Pod bytes for sink"))?];

    dummy_sink
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::MAP_BUFFERS | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| anyhow::anyhow!("PipeWire stream.connect (sink) failed: {e}"))?;

    // Keep the virtual Audio/Source (remotewayland_mic) alive for the entire session so
    // apps like Audacity see a valid 48kHz source immediately at startup, even before the
    // client activates host mic capture on demand.
    let mic_source = pw::stream::StreamBox::new(
        &core,
        "remotewayland-mic",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Duplex",
            *pw::keys::MEDIA_CLASS => "Audio/Source",
            *pw::keys::NODE_NAME => VIRTUAL_MIC_NAME,
            *pw::keys::NODE_DESCRIPTION => "RemoteWayland Microphone",
            *pw::keys::AUDIO_CHANNELS => "2",
            "audio.position" => "FL,FR",
        },
    )
    .map_err(|e| anyhow::anyhow!("PipeWire Stream::new (mic) failed: {e}"))?;

    let mic_rb = Arc::clone(shared_mic_ringbuf());
    let _mic_listener = mic_source
        .add_local_listener_with_user_data(mic_rb)
        .process(|stream, rb| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let stride = std::mem::size_of::<i16>() * MIC_CHANNELS as usize;
            let data = &mut datas[0];
            let written_bytes = if let Some(slice) = data.data() {
                let target_bytes = (FRAME_SAMPLES * stride).min(slice.len());
                let mut guard = rb.lock().ok();
                for byte in &mut slice[..target_bytes] {
                    *byte = guard
                        .as_mut()
                        .and_then(|q| q.pop_front())
                        .unwrap_or(0);
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
        .map_err(|e| anyhow::anyhow!("PipeWire stream register (mic) failed: {e}"))?;

    let mic_pod_bytes = build_pcm_format_pod(MIC_CHANNELS)?;
    let mut mic_params = [Pod::from_bytes(&mic_pod_bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid SPA Pod bytes for mic"))?];

    mic_source
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::MAP_BUFFERS | pw::stream::StreamFlags::RT_PROCESS,
            &mut mic_params,
        )
        .map_err(|e| anyhow::anyhow!("PipeWire stream.connect (mic) failed: {e}"))?;

    let registry = core
        .get_registry_rc()
        .map_err(|e| anyhow::anyhow!("PipeWire get_registry_rc failed: {e}"))?;

    let apps: Rc<RefCell<HashMap<u32, AppCaptureEntry>>> = Rc::new(RefCell::new(HashMap::new()));
    let apps_global = Rc::clone(&apps);
    let apps_remove = Rc::clone(&apps);
    let mic_consumers: Rc<RefCell<HashSet<u32>>> = Rc::new(RefCell::new(HashSet::new()));
    let mic_global = Rc::clone(&mic_consumers);
    let mic_remove = Rc::clone(&mic_consumers);
    let mic_tx_global = mic_active_tx.clone();
    let mic_tx_remove = mic_active_tx;
    let core_global = core.clone();
    let ml_quit = mainloop.clone();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |obj| {
            let Some(props) = obj.props else {
                return;
            };

            match obj.type_ {
                ObjectType::Node => {
                    let media_class = props.get("media.class").unwrap_or("");
                    let node_name = props.get("node.name").unwrap_or("");
                    if node_name.starts_with("rw-cap-") || node_name.starts_with("remotewayland") {
                        return;
                    }
                    if media_class == "Stream/Input/Audio" {
                        let mut set = mic_global.borrow_mut();
                        let was_empty = set.is_empty();
                        if set.insert(obj.id) && was_empty {
                            tracing::info!(
                                stream_id = obj.id,
                                app = ?props.get("application.name"),
                                "remote app opened microphone; requesting host mic capture"
                            );
                            let _ = mic_tx_global.send(true);
                        }
                        return;
                    }
                    if media_class != "Stream/Output/Audio" {
                        return;
                    }

                    let app_name = props
                        .get("application.name")
                        .filter(|s| !s.is_empty())
                        .or_else(|| props.get("node.name").filter(|s| !s.is_empty()))
                        .unwrap_or("Remote App")
                        .to_string();
                    let media_name = props.get("media.name").unwrap_or("").to_string();
                    let icon_name = props
                        .get("application.icon-name")
                        .filter(|s| !s.is_empty())
                        .or_else(|| props.get("media.icon-name").filter(|s| !s.is_empty()))
                        .or_else(|| props.get("application.process.binary").filter(|s| !s.is_empty()))
                        .unwrap_or("")
                        .to_string();
                    let media_role = props
                        .get("media.role")
                        .filter(|s| !s.is_empty())
                        .unwrap_or("Music")
                        .to_string();

                    let Ok(encoder) = opus::Encoder::new(
                        SAMPLE_RATE,
                        opus::Channels::Stereo,
                        opus::Application::Audio,
                    ) else {
                        return;
                    };
                    let Ok(pod_bytes) = build_pcm_format_pod(SINK_CHANNELS) else {
                        return;
                    };
                    let Some(pod) = Pod::from_bytes(&pod_bytes) else {
                        return;
                    };

                    let cap_name = format!("rw-cap-{}", obj.id);
                    let Ok(cap_stream) = pw::stream::StreamRc::new(
                        core_global.clone(),
                        &cap_name,
                        properties! {
                            *pw::keys::MEDIA_TYPE => "Audio",
                            *pw::keys::MEDIA_CATEGORY => "Capture",
                            *pw::keys::MEDIA_ROLE => "Music",
                            *pw::keys::NODE_NAME => cap_name.as_str(),
                            *pw::keys::AUDIO_CHANNELS => "2",
                            "audio.position" => "FL,FR",
                            "node.autoconnect" => "false",
                            "node.passive" => "true",
                        },
                    ) else {
                        return;
                    };

                    let (pkt_tx, pkt_rx) = mpsc::unbounded_channel();
                    let user_data = AppCaptureUserData {
                        encoder,
                        pcm_accum: Vec::with_capacity(FRAME_SAMPLES * SINK_CHANNELS as usize * 2),
                        opus_buf: vec![0u8; MAX_OPUS_PACKET_BYTES],
                        start: Instant::now(),
                        tx: pkt_tx,
                    };

                    let Ok(listener) = cap_stream
                        .add_local_listener_with_user_data(user_data)
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
                            for chunk in slice.chunks_exact(2) {
                                st.pcm_accum
                                    .push(i16::from_le_bytes([chunk[0], chunk[1]]));
                            }

                            let frame_total_samples = FRAME_SAMPLES * SINK_CHANNELS as usize;
                            while st.pcm_accum.len() >= frame_total_samples {
                                let mut send_failed = false;
                                {
                                    let frame = &st.pcm_accum[..frame_total_samples];
                                    if !frame.iter().all(|&s| s.abs() <= 16) {
                                        if let Ok(encoded_len) =
                                            st.encoder.encode(frame, &mut st.opus_buf)
                                        {
                                            let pkt = EncodedAudioPacket {
                                                pts_ns: st.start.elapsed().as_nanos() as u64,
                                                data: st.opus_buf[..encoded_len].to_vec(),
                                            };
                                            send_failed = st.tx.send(pkt).is_err();
                                        }
                                    }
                                }
                                st.pcm_accum.drain(..frame_total_samples);
                                if send_failed {
                                    break;
                                }
                            }
                        })
                        .register()
                    else {
                        return;
                    };

                    let mut cap_params = [pod];
                    if cap_stream
                        .connect(
                            spa::utils::Direction::Input,
                            None,
                            pw::stream::StreamFlags::MAP_BUFFERS
                                | pw::stream::StreamFlags::RT_PROCESS,
                            &mut cap_params,
                        )
                        .is_err()
                    {
                        return;
                    }

                    if event_tx
                        .send(PlaybackStreamEvent {
                            stream_id: obj.id,
                            app_name: app_name.clone(),
                            media_name: media_name.clone(),
                            icon_name: icon_name.clone(),
                            media_role: media_role.clone(),
                            packets: pkt_rx,
                        })
                        .is_err()
                    {
                        ml_quit.quit();
                        return;
                    }

                    tracing::info!(
                        stream_id = obj.id,
                        app_name,
                        media_name,
                        icon_name,
                        media_role,
                        "registered per-app PipeWire audio capture"
                    );

                    apps_global.borrow_mut().insert(
                        obj.id,
                        AppCaptureEntry {
                            stream: cap_stream,
                            _listener: listener,
                            out_ports: Vec::new(),
                            cap_in_ports: Vec::new(),
                            linked_in_ports: HashSet::new(),
                            links: Vec::new(),
                        },
                    );
                }
                ObjectType::Port => {
                    let Some(node_id) = props.get("node.id").and_then(|s| s.parse::<u32>().ok())
                    else {
                        return;
                    };
                    let dir = props.get("port.direction");
                    let channel = props.get("audio.channel").unwrap_or("").to_string();

                    let mut map = apps_global.borrow_mut();
                    if dir == Some("out") {
                        if let Some(entry) = map.get_mut(&node_id) {
                            entry.out_ports.push((obj.id, channel));
                            try_link_app(&core_global, node_id, entry);
                        }
                    } else if dir == Some("in") {
                        for (&app_node_id, entry) in map.iter_mut() {
                            if entry.stream.node_id() == node_id {
                                entry.cap_in_ports.push((obj.id, channel));
                                try_link_app(&core_global, app_node_id, entry);
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        })
        .global_remove(move |id| {
            if apps_remove.borrow_mut().remove(&id).is_some() {
                tracing::info!(stream_id = id, "removed per-app PipeWire audio capture");
            }
            let mut set = mic_remove.borrow_mut();
            if set.remove(&id) && set.is_empty() {
                tracing::info!(
                    stream_id = id,
                    "no remote apps using microphone; stopping host mic capture"
                );
                let _ = mic_tx_remove.send(false);
            }
        })
        .register();

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}

/// Receives Opus-encoded microphone audio packets from the client and feeds them
/// into the isolated PipeWire `Audio/Source` node (`remotewayland_mic`).
pub struct AudioMicInjector {
    ringbuf: Arc<Mutex<VecDeque<u8>>>,
    decoder: Mutex<opus::Decoder>,
}

impl AudioMicInjector {
    pub fn new() -> Result<Self> {
        ensure_virtual_audio_devices()?;

        let decoder = opus::Decoder::new(SAMPLE_RATE, opus::Channels::Mono)
            .context("failed to create server Opus mono decoder")?;

        Ok(Self {
            ringbuf: Arc::clone(shared_mic_ringbuf()),
            decoder: Mutex::new(decoder),
        })
    }

    pub fn push_audio(&self, _pts_ns: u64, data: &[u8]) -> Result<()> {
        let mut pcm = [0i16; FRAME_SAMPLES * 2];
        let decoded_samples = {
            let mut dec = self
                .decoder
                .lock()
                .map_err(|_| anyhow::anyhow!("mic decoder lock poisoned"))?;
            dec.decode(data, &mut pcm, false)
                .context("Opus mic decode failed")?
        };

        let mut buf = self
            .ringbuf
            .lock()
            .map_err(|_| anyhow::anyhow!("mic ringbuf lock poisoned"))?;
        for &sample in &pcm[..decoded_samples] {
            let le = sample.to_le_bytes();
            // Duplicate mono mic sample to FL + FR for universal stereo/mono app compatibility.
            buf.push_back(le[0]);
            buf.push_back(le[1]);
            buf.push_back(le[0]);
            buf.push_back(le[1]);
        }
        while buf.len() > MAX_MIC_RINGBUF_BYTES {
            buf.pop_front();
        }
        Ok(())
    }
}
