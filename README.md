# RemoteWayland

Rootless, low-latency remote GUI Linux application streamer with video and bidirectional PipeWire audio over QUIC, targeting `x86_64` and `aarch64` Linux clients.

## Architecture

- **`remotewayland-proto`**: Shared wire protocol, stream framing, and ephemeral TLS/token handshake types.
- **`remotewayland-server`**: Headless Wayland compositor (`smithay`), GStreamer video/audio encoder, isolated PipeWire virtual sink/source bridge, and QUIC server.
- **`remotewayland-client`**: Rootless multi-window client (`winit` + `wgpu`) for Wayland and X11, GStreamer decoder, and local PipeWire bridge.

## System Dependencies

### Fedora / RHEL
```bash
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel \
  gstreamer1-plugins-good gstreamer1-plugins-bad-free \
  pipewire-devel wayland-devel libxkbcommon-devel libdrm-devel mesa-libgbm-devel
```

### Ubuntu / Debian
```bash
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
  libpipewire-0.3-dev libwayland-dev libxkbcommon-dev libdrm-dev libgbm-dev
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
