# Oriel

Rootless, low-latency remote GUI Linux application streamer with video and bidirectional PipeWire audio over QUIC, targeting `x86_64` and `aarch64` Linux clients.

## Architecture

- **`oriel-proto`**: Shared wire protocol, stream framing, and ephemeral TLS/token handshake types.
- **`oriel-server`**: Headless Wayland compositor (`smithay`) with embedded `Xwayland` (`X11Wm`) for remote X11 apps, pure-Rust XOR-delta + LZ4 video encoder, isolated PipeWire/PulseAudio virtual sink/source bridge, and QUIC server. Apps get GPU acceleration (OpenGL, Vulkan, Xwayland glamor) through `linux-dmabuf` on the first hardware render node EGL reports; pick another with `--render-node /dev/dri/renderD129` or turn it off with `--no-gpu` (both also accepted by `attach` when it starts a new server). Explicit sync (`linux-drm-syncobj`) is offered when the kernel driver supports it. Without `libEGL` or a usable GPU, apps fall back to software rendering.
- **`oriel-client`**: Rootless multi-window & unbounded `xdg_popup` Wayland client (`smithay-client-toolkit`), pure-Rust LZ4 delta decoder, local PipeWire/PulseAudio bridge, and a GTK4/libadwaita launcher. Launcher connection logic lives in the toolkit-independent `session` module; the GTK front-end only renders it.

## System Dependencies

### Fedora / RHEL
```bash
sudo dnf install wayland-devel libxkbcommon-devel pulseaudio-utils xorg-x11-server-Xwayland
# GPU acceleration for apps on the server (loaded at runtime)
sudo dnf install mesa-libEGL
```

The client launcher is built with GTK4 and libadwaita (1.5+); build with
`--no-default-features` for a terminal-only launcher instead:
```bash
sudo dnf install libadwaita-devel
```

### Ubuntu / Debian
```bash
sudo apt install libwayland-dev libxkbcommon-dev pulseaudio-utils xwayland
# GPU acceleration for apps on the server (loaded at runtime)
sudo apt install libegl1
# GTK4/libadwaita launcher
sudo apt install libgtk-4-dev libadwaita-1-dev
```

## Building

```bash
# Build all crates
cargo build --workspace

# Run all tests, including the loopback client/server tests in crates/oriel-server/tests
cargo test --workspace
```

`oriel-proto` builds the SmartTile encoder's GPU compute paths (wgpu, Vulkan) only with its
`gpu-encoder` feature, which the server enables. Built on its own
(`cargo build -p oriel-client`, as the Flatpak does), the client leaves them out; a
workspace build shares one `oriel-proto`, so its client includes them.

### Cross-compiling for `aarch64-unknown-linux-gnu`

Using the Rust/GCC aarch64 cross toolchain in a `podman` (or `docker`) container via [`scripts/build-aarch64.sh`](scripts/build-aarch64.sh); no `qemu` emulation is needed:

```bash
# Build workspace with GTK4/libadwaita launcher (default)
./scripts/build-aarch64.sh

# Build client without GUI launcher
./scripts/build-aarch64.sh --no-gui --client-only
```

Builds run natively in an x86_64 Ubuntu 24.04 container that uses arm64 multiarch `-dev` packages as the target sysroot; outputs are written to `target/aarch64-unknown-linux-gnu/release/`.

### Flatpak

The client ships as the `dev.knoll.Oriel` Flatpak on the GNOME 51 runtime, via
[`scripts/build-flatpak.sh`](scripts/build-flatpak.sh):

```bash
# Install the GNOME runtime/SDK and Rust/LLVM extensions once, then build and install
./scripts/build-flatpak.sh --install-deps --install
```

## Usage

Install `oriel-server` on the machine whose apps you want to use, and `oriel-client`
(or the Flatpak) where you sit. The client reaches the server over SSH once, to get a
connection URI, and then talks to it directly over QUIC (UDP).

```bash
# GUI launcher: add the host (an SSH target such as user@host) and connect
oriel-client

# From a terminal, optionally launching an app right away
oriel-client connect user@host -- gnome-text-editor

# A remote shell, without Wayland or audio
oriel-client connect --terminal user@host

# Without the client's SSH support: get a URI yourself and pass it on stdin
ssh user@host oriel-server attach --host host | oriel-client --uri -
```

### The server daemon

`oriel-server attach` starts a per-user server daemon on first use and keeps it, with its
apps, running between connections; later `attach` calls reuse it. Its log is
`$XDG_RUNTIME_DIR/oriel-server.log` on the server.

- `oriel-server stop` stops the daemon and closes its apps.
- After upgrading, a daemon from the old install may speak another protocol version than
  the new client. `attach` refuses to hand out a URI for a daemon that reports another
  version (daemons from before this check only get a warning, since they can't say);
  `oriel-server attach --replace` stops it and starts the installed version.
- `--listen IP[:PORT[-END]]` on `attach` (or `oriel-client connect --port`) picks the
  address and UDP port(s) a new daemon listens on; by default it is any free UDP port on
  all interfaces.

### Security model

- Every connection needs a single-use token that `attach` mints over SSH. Tokens expire
  after 60 seconds and, when the client asks for it (the launcher and `connect` always
  do), only work from the client's TLS certificate.
- The client pins the server's ephemeral certificate by the SHA-256 fingerprint in the
  URI, so it only ever talks to the server that SSH vouched for.
- Tokens never appear in process arguments: the client takes URIs on stdin or from
  `$ORIEL_URI`.
- The server listens on all interfaces by default. A peer that has not authenticated must
  do so within 10 seconds, may send at most 4 KiB before it has, and at most 32 such peers
  are served at a time. Restrict `--listen` or firewall the port if the server should not
  be reachable from the network.
- An authenticated client can do whatever the server's user can: it launches apps and
  opens shells.
