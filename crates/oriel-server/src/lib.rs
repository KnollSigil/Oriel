//! Oriel server: a headless Wayland compositor that streams rootless windows, audio and
//! remote PTYs over QUIC. The `oriel-server` binary is a thin entry point over these
//! modules; they are a library so integration tests can drive them.

pub mod apps;
pub mod audio;
pub mod compositor;
pub mod damage;
pub mod downscale;
pub mod flatpak_bwrap;
pub mod gpu;
pub mod media;
pub mod pty;
pub mod quic;
pub mod session;
pub mod settings_portal;
pub mod shell_env;
pub mod x11_selection;
