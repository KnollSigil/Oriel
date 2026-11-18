# RemoteWayland Implementation Plan

## Phase 1 (Completed)
- Pure-Rust Cargo workspace (`remotewayland-proto`, `remotewayland-server`, `remotewayland-client`) targeting `x86_64` and `aarch64` Linux.
- Headless `smithay` Wayland compositor with `xdg_toplevel`, unbounded `xdg_popup`, subsurfaces, `wp_cursor_shape_v1`, `zxdg_decoration_manager_v1`, and bidirectional `wl_data_device` clipboard.
- Native `smithay-client-toolkit` (`sctk`) + `SlotPool` (`Argb8888`) multi-window & popup client.
- Lossless XOR-delta + `lz4_flex` video/popup streaming and `parec`/`pacat` + `lz4_flex` audio bridge over `quinn` QUIC (TLS 1.3 + SHA-256 cert pinning + ephemeral token).

## Phase 2 — Launcher, Persistent Sessions, SSH Bootstrap & Remote PTYs

### 1. Protocol Extensions (`remotewayland-proto`)
- **App Discovery & Launch:**
  - `ControlMessage::ListApps` -> `ControlMessage::AppList { apps: Vec<RemoteAppEntry> }` (`id`, `name`, `exec`, `categories`, `icon_mime`, `icon_data`).
  - `ControlMessage::LaunchApp { exec: String }` (spawns GUI app inside the server's `WAYLAND_DISPLAY` with lazy compositor/audio init).
- **Remote PTY Streams (Zero Wayland / PipeWire):**
  - `StreamHeader::Pty { cols: u16, rows: u16, command: Option<String> }` on a bidirectional QUIC stream.
  - Framed PTY messages (`PtyMessage::Data(Vec<u8>)`, `PtyMessage::Resize { cols: u16, rows: u16 }`, `PtyMessage::Exit(i32)`).

### 2. Persistent Server & SSH Bootstrap (`remotewayland-server`)
- **`remotewayland-server attach` subcommand:**
  - Checks `$XDG_RUNTIME_DIR/remotewayland.sock`.
  - If a persistent server is already running, connects to the Unix socket, mints a fresh ephemeral auth token, and prints the `remotewayland://...` URI to stdout.
  - If no server is running, spawns `remotewayland-server --daemon` (detached from SSH stdio), waits for `$XDG_RUNTIME_DIR/remotewayland.sock`, mints a token, and prints the URI.
- **Lazy / Optional Wayland & PipeWire:**
  - Headless compositor starts lazily on first `LaunchApp` (or immediately if `-- <cmd>` was passed); terminals work even if PipeWire/PulseAudio is absent.
- **Remote PTY Handler:**
  - Accepts `StreamHeader::Pty` bidirectional QUIC streams, allocates a PTY (`openpty`), spawns `$SHELL` (or requested command), and bridges stdin/stdout/resize/exit over the QUIC stream.
- **XDG `.desktop` Scanner:**
  - Scans `/usr/share/applications`, `/usr/local/share/applications`, and `~/.local/share/applications`, resolves PNG/SVG icons from standard icon theme paths, and serves them in `AppList`.

### 3. Client SSH Bootstrap, PTY Bridge & GTK4 Launcher (`remotewayland-client`)
- **SSH Bootstrap & Host Profiles:**
  - Host profiles stored in `~/.config/remotewayland/hosts.json` (`label`, `ssh_target`, `server_bin`, `audio_enabled`) + quick-connect `user@host` input.
  - Runs `ssh <target> <server_bin> attach` to obtain the `remotewayland://...` URI automatically (substituting the SSH hostname/IP when the server binds `0.0.0.0`).
- **PTY Local Terminal Bridge:**
  - `remotewayland-client pty-attach --uri ...` connects to a QUIC `StreamHeader::Pty` stream, puts local stdin in raw mode, forwards `SIGWINCH` resizes, and is spawned inside the user's preferred local terminal emulator (`$TERMINAL`, `konsole`, `alacritty`, `kitty`, `foot`, `gnome-terminal`, `xterm`).
- **Native GTK4 Launcher Window (feature-gated `gtk4`):**
  - Opens when `remotewayland-client` is launched without `--uri` or `pty-attach`, offering host picker/manager, **New Terminal** button, searchable graphical app icon grid, and custom command launcher, running alongside the `sctk` rootless window/popup presenter.
