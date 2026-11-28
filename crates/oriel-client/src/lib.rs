//! Oriel client: renders remote windows streamed by an Oriel server, bridges audio,
//! clipboard and files, and hosts the launcher. The `oriel-client` binary is a thin entry
//! point over these modules; they are a library so integration tests can drive them.

#[cfg(feature = "gtk4")]
pub mod adw_launcher;
pub mod audio;
pub mod hosts;
pub mod icon;
pub mod launcher;
pub mod media;
pub mod pty;
pub mod quic;
#[cfg(any(feature = "gtk4", test))]
pub mod search;
#[cfg(any(feature = "gtk4", test))]
pub mod session;
#[cfg(feature = "gtk4")]
pub mod symbolic_icon;
pub mod traffic;
pub mod window;
pub mod xfer;
