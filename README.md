# RemoteWayland

Rootless, low-latency remote GUI Linux application streamer with video and bidirectional PipeWire audio over QUIC, targeting `x86_64` and `aarch64` Linux clients.

## Architecture

- **`remotewayland-proto`**: Shared wire protocol, stream framing, and ephemeral TLS/token handshake types.
- **`remotewayland-server`**: Headless Wayland compositor (`smithay`), pure-Rust XOR-delta + LZ4 video encoder, isolated PipeWire/PulseAudio virtual sink/source bridge, and QUIC server.
- **`remotewayland-client`**: Rootless multi-window & unbounded `xdg_popup` Wayland client (`smithay-client-toolkit`), pure-Rust LZ4 delta decoder, and local PipeWire/PulseAudio bridge.

## System Dependencies

### Fedora / RHEL
```bash
sudo dnf install wayland-devel libxkbcommon-devel pulseaudio-utils
```

### Ubuntu / Debian
```bash
sudo apt install libwayland-dev libxkbcommon-dev pulseaudio-utils
```

## Building

```bash
# Build all crates
cargo build --workspace

# Run protocol tests
cargo test -p remotewayland-proto
```

### Cross-compiling for `aarch64-unknown-linux-gnu`

Using [`cross`](https://github.com/cross-rs/cross) or a sysroot with `aarch64` GStreamer/PipeWire/Wayland dev libraries:

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu -p remotewayland-client
```
