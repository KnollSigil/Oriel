//! Session-level isolation for RemoteWayland server.
//!
//! Validates the host `XDG_RUNTIME_DIR` (refusing any `/tmp` fallback), creates a
//! private `0700` session `XDG_RUNTIME_DIR` hosting the Wayland, PipeWire,
//! PulseAudio, and D-Bus sockets, manages private `dbus-daemon` and
//! `xdg-desktop-portal` processes, and applies a unified isolated environment
//! to both GUI applications and remote PTYs.

use std::ffi::{OsStr, OsString};
use std::fs::{self, DirBuilder, Permissions};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

/// Snapshot of isolated session paths shared with `compositor.rs` and `pty.rs`.
#[derive(Debug, Clone)]
pub struct SessionEnvInfo {
    pub runtime_dir: PathBuf,
    pub dbus_address: Option<String>,
    pub wayland_socket: Option<OsString>,
}

static SESSION_INFO: OnceLock<Mutex<SessionEnvInfo>> = OnceLock::new();

/// Validate and return the host `XDG_RUNTIME_DIR`.
///
/// Fails fast if `XDG_RUNTIME_DIR` is unset, is not a real directory, is not
/// owned by the current effective UID, or is accessible by group/other.
pub fn validate_host_runtime_dir() -> Result<PathBuf> {
    let raw = std::env::var_os("XDG_RUNTIME_DIR").context(
        "XDG_RUNTIME_DIR is not set; refusing to fall back to /tmp (ensure a user runtime directory is configured)",
    )?;
    let path = PathBuf::from(&raw);
    if !path.is_absolute() {
        bail!(
            "XDG_RUNTIME_DIR ({}) must be an absolute path",
            path.display()
        );
    }

    let meta = fs::symlink_metadata(&path).with_context(|| {
        format!(
            "XDG_RUNTIME_DIR ({}) does not exist or cannot be inspected",
            path.display()
        )
    })?;
    if !meta.file_type().is_dir() {
        bail!(
            "XDG_RUNTIME_DIR ({}) must be a directory (not a symlink or file)",
            path.display()
        );
    }

    let euid = unsafe { libc::geteuid() };
    if meta.uid() != euid {
        bail!(
            "XDG_RUNTIME_DIR ({}) is owned by uid {}, expected current uid {}",
            path.display(),
            meta.uid(),
            euid
        );
    }

    let mode = meta.permissions().mode() & 0o777;
    if (mode & 0o077) != 0 {
        bail!(
            "XDG_RUNTIME_DIR ({}) has insecure permissions {:04o}; expected 0700",
            path.display(),
            mode
        );
    }

    Ok(path)
}

/// Path to the daemon control socket in the validated host `XDG_RUNTIME_DIR`.
pub fn control_socket_path(host_runtime_dir: &Path) -> PathBuf {
    host_runtime_dir.join("remotewayland.sock")
}

/// Path to the daemon log file in the validated host `XDG_RUNTIME_DIR`.
pub fn daemon_log_path(host_runtime_dir: &Path) -> PathBuf {
    host_runtime_dir.join("remotewayland-server.log")
}

/// Explicitly set Unix permissions on a path (bypassing process umask).
pub fn set_mode(path: &Path, mode: u32) -> Result<()> {
    fs::set_permissions(path, Permissions::from_mode(mode))
        .with_context(|| format!("failed to chmod {:04o} {}", mode, path.display()))
}

/// Create a directory with strict `0700` permissions.
pub fn create_private_dir(path: &Path) -> Result<()> {
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .with_context(|| format!("failed to create directory {}", path.display()))?;
    set_mode(path, 0o700)?;
    Ok(())
}

/// Owns the isolated session runtime directory and session-scoped daemons
/// (`dbus-daemon`, `xdg-desktop-portal`). Cleans them up on drop.
pub struct SessionGuard {
    runtime_dir: PathBuf,
    dbus_address: Option<String>,
    children: Vec<Child>,
}

impl SessionGuard {
    /// Create `<host_runtime_dir>/remotewayland-session-<pid>` (mode `0700`),
    /// switch this process's `XDG_RUNTIME_DIR` to it so Smithay & PipeWire bind
    /// their sockets inside the isolated session, and start a private D-Bus
    /// session bus.
    pub fn init(host_runtime_dir: &Path) -> Result<Self> {
        let runtime_dir =
            host_runtime_dir.join(format!("remotewayland-session-{}", std::process::id()));
        if runtime_dir.exists() {
            let _ = fs::remove_dir_all(&runtime_dir);
        }
        create_private_dir(&runtime_dir)?;
        create_private_dir(&runtime_dir.join("pulse"))?;

        // Point this server process (Smithay ListeningSocketSource, libpipewire)
        // at the isolated session runtime directory.
        std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
        std::env::remove_var("DISPLAY");
        std::env::remove_var("XAUTHORITY");

        let mut children = Vec::new();
        let dbus_address = match start_private_dbus(&runtime_dir) {
            Ok((addr, child)) => {
                children.push(child);
                std::env::set_var("DBUS_SESSION_BUS_ADDRESS", &addr);
                Some(addr)
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "private dbus-daemon unavailable; unsetting DBUS_SESSION_BUS_ADDRESS for session isolation"
                );
                std::env::remove_var("DBUS_SESSION_BUS_ADDRESS");
                None
            }
        };

        let info = SessionEnvInfo {
            runtime_dir: runtime_dir.clone(),
            dbus_address: dbus_address.clone(),
            wayland_socket: None,
        };
        let _ = SESSION_INFO.set(Mutex::new(info));

        info!(
            runtime_dir = %runtime_dir.display(),
            dbus = ?dbus_address,
            "isolated RemoteWayland session runtime initialized"
        );

        Ok(Self {
            runtime_dir,
            dbus_address,
            children,
        })
    }

    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    /// Record the Wayland socket name once the headless compositor is ready and
    /// best-effort launch `xdg-desktop-portal` inside the isolated session.
    pub fn set_wayland_socket_and_start_portal(&mut self, socket_name: &OsStr) {
        if let Some(lock) = SESSION_INFO.get() {
            if let Ok(mut guard) = lock.lock() {
                guard.wayland_socket = Some(socket_name.to_os_string());
            }
        }

        if self.dbus_address.is_some() {
            match start_xdg_desktop_portal(&self.runtime_dir, socket_name) {
                Ok(child) => self.children.push(child),
                Err(err) => {
                    warn!(
                        error = %err,
                        "xdg-desktop-portal not started inside isolated session"
                    );
                }
            }
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
        crate::audio::shutdown_isolated_pipewire();
        if let Err(err) = fs::remove_dir_all(&self.runtime_dir) {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    path = %self.runtime_dir.display(),
                    error = %err,
                    "failed to remove session runtime directory"
                );
            }
        }
    }
}

fn start_private_dbus(runtime_dir: &Path) -> Result<(String, Child)> {
    let bus_path = runtime_dir.join("bus");
    let address = format!("unix:path={}", bus_path.display());

    let mut cmd = Command::new("dbus-daemon");
    cmd.arg("--session")
        .arg(format!("--address={address}"))
        .arg("--nofork")
        .arg("--nopidfile")
        .env("XDG_RUNTIME_DIR", runtime_dir)
        .env_remove("DBUS_SESSION_BUS_ADDRESS")
        .env_remove("DISPLAY")
        .env_remove("XAUTHORITY")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    let mut child = cmd.spawn().context("failed to spawn dbus-daemon")?;

    let deadline = Instant::now() + Duration::from_secs(2);
    while !bus_path.exists() {
        if let Ok(Some(status)) = child.try_wait() {
            bail!("dbus-daemon exited prematurely with {status}");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("timed out waiting for private D-Bus socket at {}", bus_path.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let _ = set_mode(&bus_path, 0o600);
    info!(address = %address, "private D-Bus session daemon started");
    Ok((address, child))
}

fn find_portal_binary() -> Option<PathBuf> {
    for candidate in [
        "/usr/libexec/xdg-desktop-portal",
        "/usr/lib/xdg-desktop-portal",
        "/usr/lib/xdg-desktop-portal/xdg-desktop-portal",
    ] {
        let p = Path::new(candidate);
        if p.is_file() {
            return Some(p.to_path_buf());
        }
    }
    None
}

fn start_xdg_desktop_portal(runtime_dir: &Path, wayland_socket: &OsStr) -> Result<Child> {
    let Some(bin) = find_portal_binary() else {
        bail!("xdg-desktop-portal binary not found in /usr/libexec or /usr/lib");
    };

    let mut cmd = Command::new(&bin);
    cmd.arg("--replace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_session_env(&mut cmd, Some(wayland_socket));
    cmd.env("XDG_RUNTIME_DIR", runtime_dir);

    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", bin.display()))?;
    info!(bin = %bin.display(), pid = child.id(), "isolated xdg-desktop-portal started");
    Ok(child)
}

/// Apply the unified isolated session environment (Wayland, D-Bus, PipeWire,
/// PulseAudio, and XDG_RUNTIME_DIR, with host DISPLAY/XAUTHORITY stripped)
/// to a child command (both GUI apps and remote PTYs).
pub fn apply_session_env(cmd: &mut Command, wayland_override: Option<&OsStr>) {
    cmd.env_remove("DISPLAY")
        .env_remove("XAUTHORITY")
        .env("PULSE_SINK", crate::audio::VIRTUAL_SINK_NAME)
        .env("PULSE_SOURCE", crate::audio::VIRTUAL_MIC_NAME)
        .env("GDK_BACKEND", "wayland")
        .env("QT_QPA_PLATFORM", "wayland")
        .env("MOZ_ENABLE_WAYLAND", "1")
        .env("SDL_VIDEODRIVER", "wayland");

    if let Some(lock) = SESSION_INFO.get() {
        if let Ok(info) = lock.lock() {
            cmd.env("XDG_RUNTIME_DIR", &info.runtime_dir);
            if let Some(ws) = wayland_override.or(info.wayland_socket.as_deref()) {
                cmd.env("WAYLAND_DISPLAY", ws);
            }
            if let Some(addr) = &info.dbus_address {
                cmd.env("DBUS_SESSION_BUS_ADDRESS", addr);
            } else {
                cmd.env_remove("DBUS_SESSION_BUS_ADDRESS");
            }
        }
    } else if let Some(ws) = wayland_override {
        cmd.env("WAYLAND_DISPLAY", ws);
    }

    if let Some(pw_dir) = crate::audio::isolated_pw_runtime_dir() {
        cmd.env("PIPEWIRE_RUNTIME_DIR", pw_dir)
            .env("PULSE_RUNTIME_PATH", pw_dir.join("pulse"))
            .env(
                "PULSE_SERVER",
                format!("unix:{}", pw_dir.join("pulse/native").display()),
            );
    }
}
