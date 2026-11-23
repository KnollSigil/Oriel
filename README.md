# RemoteWayland

Rootless, low-latency remote GUI Linux application streamer with video and bidirectional PipeWire audio over QUIC, targeting `x86_64` and `aarch64` Linux clients.

## Architecture

- **`remotewayland-proto`**: Shared wire protocol, stream framing, and ephemeral TLS/token handshake types.
- **`remotewayland-server`**: Headless Wayland compositor (`smithay`) with embedded `Xwayland` (`X11Wm`) for remote X11 apps, pure-Rust XOR-delta + LZ4 video encoder, isolated PipeWire/PulseAudio virtual sink/source bridge, and QUIC server.
- **`remotewayland-client`**: Rootless multi-window & unbounded `xdg_popup` Wayland client (`smithay-client-toolkit`), pure-Rust LZ4 delta decoder, local PipeWire/PulseAudio bridge, and a GTK4/libadwaita launcher. Launcher connection logic lives in the toolkit-independent `session` module; the GTK front-end only renders it.

## System Dependencies

### Fedora / RHEL
```bash
sudo dnf install wayland-devel libxkbcommon-devel pulseaudio-utils xorg-x11-server-Xwayland
```

The client launcher is built with GTK4 and libadwaita (1.5+); build with
`--no-default-features` for a terminal-only launcher instead:
```bash
sudo dnf install libadwaita-devel
```

### Ubuntu / Debian
```bash
sudo apt install libwayland-dev libxkbcommon-dev pulseaudio-utils xwayland
# GTK4/libadwaita launcher
sudo apt install libgtk-4-dev libadwaita-1-dev
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
# Build workspace with GTK4/libadwaita launcher (default)
./scripts/build-aarch64.sh

# Build client without GUI launcher
./scripts/build-aarch64.sh --no-gui --client-only
```

Builds run in an Ubuntu 24.04 arm64 container; outputs are written to `target/aarch64-unknown-linux-gnu/release/`.

### Flatpak

The client ships as the `dev.knoll.Oriel` Flatpak on the GNOME 51 runtime, via
[`scripts/build-flatpak.sh`](scripts/build-flatpak.sh):

```bash
# Install the GNOME runtime/SDK and Rust/LLVM extensions once, then build and install
./scripts/build-flatpak.sh --install-deps --install
```
