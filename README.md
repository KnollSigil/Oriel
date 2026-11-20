# RemoteWayland

Rootless, low-latency remote GUI Linux application streamer with video and bidirectional PipeWire audio over QUIC, targeting `x86_64` and `aarch64` Linux clients.

## Architecture

- **`remotewayland-proto`**: Shared wire protocol, stream framing, and ephemeral TLS/token handshake types.
- **`remotewayland-server`**: Headless Wayland compositor (`smithay`) with embedded `Xwayland` (`X11Wm`) for remote X11 apps, pure-Rust XOR-delta + LZ4 video encoder, isolated PipeWire/PulseAudio virtual sink/source bridge, and QUIC server.
- **`remotewayland-client`**: Rootless multi-window & unbounded `xdg_popup` Wayland client (`smithay-client-toolkit`), pure-Rust LZ4 delta decoder, and local PipeWire/PulseAudio bridge.

## System Dependencies

### Fedora / RHEL
```bash
sudo dnf install wayland-devel libxkbcommon-devel pulseaudio-utils xorg-x11-server-Xwayland
```

### Ubuntu / Debian
```bash
sudo apt install libwayland-dev libxkbcommon-dev pulseaudio-utils xwayland
```

## Building

```bash
# Build all crates
cargo build --workspace

# Run protocol tests
cargo test -p remotewayland-proto
```

### Cross-compiling for `aarch64-unknown-linux-gnu`

Using `podman` (or `docker`) with `qemu-aarch64-static` via [`scripts/build-aarch64.sh`](scripts/build-aarch64.sh):

```bash
# Build workspace with GTK4 launcher (Ubuntu 24.04 arm64, default)
./scripts/build-aarch64.sh

# Build client with Qt Quick/QML launcher (Fedora 44 arm64 for Qt 6.10+)
./scripts/build-aarch64.sh --qt --client-only

# Build without GUI launcher (Ubuntu 24.04 arm64)
./scripts/build-aarch64.sh --no-gui --client-only
```

Outputs are written to `target/aarch64-unknown-linux-gnu/release/` (Ubuntu 24.04) or `target/aarch64-fedora/aarch64-unknown-linux-gnu/release/` (Fedora 44).
