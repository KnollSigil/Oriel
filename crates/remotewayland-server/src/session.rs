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
    pub x11_display: Option<String>,
    pub xauthority: Option<PathBuf>,
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
            x11_display: None,
            xauthority: None,
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

/// Return the isolated session runtime directory if initialized.
pub fn session_runtime_dir() -> Option<PathBuf> {
    SESSION_INFO
        .get()
        .and_then(|lock| lock.lock().ok())
        .map(|info| info.runtime_dir.clone())
}

/// Record the active XWayland `DISPLAY` and `XAUTHORITY` paths so subsequent
/// GUI applications and remote PTYs automatically inherit them.
pub fn set_x11_env(display: String, xauthority: Option<PathBuf>) {
    if let Some(lock) = SESSION_INFO.get() {
        if let Ok(mut info) = lock.lock() {
            info.x11_display = Some(display);
            info.xauthority = xauthority;
        }
    }
}

/// Clear `DISPLAY` and `XAUTHORITY` from the session environment (e.g., if
/// XWayland exits unexpectedly).
pub fn clear_x11_env() {
    if let Some(lock) = SESSION_INFO.get() {
        if let Ok(mut info) = lock.lock() {
            info.x11_display = None;
            info.xauthority = None;
        }
    }
}

/// Fill `buf` with cryptographically secure random bytes via `getrandom(2)`.
pub fn fill_random_bytes(buf: &mut [u8]) -> Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let ret = unsafe {
            libc::getrandom(
                buf[filled..].as_mut_ptr().cast::<libc::c_void>(),
                buf.len() - filled,
                0,
            )
        };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err).context("getrandom(2) failed");
        }
        filled += ret as usize;
    }
    Ok(())
}

/// Locate the real `Xwayland` executable on `PATH` (excluding any session shim dir).
pub fn find_xwayland_binary(exclude_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            if let Some(excl) = exclude_dir {
                if dir == excl {
                    continue;
                }
            }
            let candidate = dir.join("Xwayland");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    for fallback in ["/usr/bin/Xwayland", "/usr/local/bin/Xwayland"] {
        let p = Path::new(fallback);
        if p.is_file() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Write a per-session `MIT-MAGIC-COOKIE-1` Xauthority file (mode `0600`) for
/// `display_num` using 128 bits of CSPRNG entropy from `getrandom(2)`.
pub fn write_xauthority_file(path: &Path, display_num: u32) -> Result<()> {
    let mut cookie = [0u8; 16];
    fill_random_bytes(&mut cookie).context("failed to generate Xauthority cookie")?;

    let hostname = read_hostname();
    let display_str = display_num.to_string();
    let mut data = Vec::with_capacity(128);
    // FamilyLocal (256) with local hostname
    if !hostname.is_empty() {
        append_xauth_record(&mut data, 256, hostname.as_bytes(), display_str.as_bytes(), &cookie);
    }
    // FamilyWild (65535) so local clients connecting via unix socket match regardless of hostname
    append_xauth_record(&mut data, 65535, &[], display_str.as_bytes(), &cookie);

    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let _ = fs::remove_file(path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("failed to create Xauthority file {}", path.display()))?;
    file.write_all(&data)
        .with_context(|| format!("failed to write Xauthority file {}", path.display()))?;
    set_mode(path, 0o600)?;
    Ok(())
}

fn append_xauth_record(
    out: &mut Vec<u8>,
    family: u16,
    address: &[u8],
    display_num: &[u8],
    cookie: &[u8],
) {
    const AUTH_NAME: &[u8] = b"MIT-MAGIC-COOKIE-1";
    out.extend_from_slice(&family.to_be_bytes());
    out.extend_from_slice(&(address.len() as u16).to_be_bytes());
    out.extend_from_slice(address);
    out.extend_from_slice(&(display_num.len() as u16).to_be_bytes());
    out.extend_from_slice(display_num);
    out.extend_from_slice(&(AUTH_NAME.len() as u16).to_be_bytes());
    out.extend_from_slice(AUTH_NAME);
    out.extend_from_slice(&(cookie.len() as u16).to_be_bytes());
    out.extend_from_slice(cookie);
}

fn read_hostname() -> String {
    let mut buf = [0u8; 256];
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) };
    if rc != 0 {
        return String::new();
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

fn sh_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Create a `0700` wrapper script `<runtime_dir>/bin/Xwayland` that execs the
/// real `Xwayland` binary with `-auth <xauth_path>` so Smithay's `XWayland::spawn`
/// enforces MIT-MAGIC-COOKIE-1 authentication.
pub fn write_xwayland_auth_shim(
    runtime_dir: &Path,
    real_xwayland: &Path,
    xauth_path: &Path,
) -> Result<PathBuf> {
    let bin_dir = runtime_dir.join("bin");
    create_private_dir(&bin_dir)?;
    let shim_path = bin_dir.join("Xwayland");
    let real_str = real_xwayland
        .to_str()
        .context("Xwayland binary path must be valid UTF-8")?;
    let xauth_str = xauth_path
        .to_str()
        .context("Xauthority path must be valid UTF-8")?;
    let script = format!(
        "#!/bin/sh\nexec {} -auth {} \"$@\"\n",
        sh_single_quote(real_str),
        sh_single_quote(xauth_str),
    );
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let _ = fs::remove_file(&shim_path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(&shim_path)
        .with_context(|| format!("failed to create Xwayland auth shim {}", shim_path.display()))?;
    file.write_all(script.as_bytes())?;
    set_mode(&shim_path, 0o700)?;
    Ok(bin_dir)
}

/// Apply the unified isolated session environment (Wayland, XWayland DISPLAY/XAUTHORITY,
/// D-Bus, PipeWire, PulseAudio, and XDG_RUNTIME_DIR) to a child command (both GUI apps
/// and remote PTYs) without forcing toolkit backend overrides.
pub fn apply_session_env(cmd: &mut Command, wayland_override: Option<&OsStr>) {
    cmd.env_remove("DISPLAY")
        .env_remove("XAUTHORITY")
        .env("PULSE_SINK", crate::audio::VIRTUAL_SINK_NAME)
        .env("PULSE_SOURCE", crate::audio::VIRTUAL_MIC_NAME);

    if let Some(lock) = SESSION_INFO.get() {
        if let Ok(info) = lock.lock() {
            cmd.env("XDG_RUNTIME_DIR", &info.runtime_dir);
            if let Some(ws) = wayland_override.or(info.wayland_socket.as_deref()) {
                cmd.env("WAYLAND_DISPLAY", ws);
            }
            if let Some(disp) = &info.x11_display {
                cmd.env("DISPLAY", disp);
            }
            if let Some(xauth) = &info.xauthority {
                cmd.env("XAUTHORITY", xauth);
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        fn new(prefix: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            create_private_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn sh_single_quote_escapes_shell_metacharacters() {
        assert_eq!(sh_single_quote("/usr/bin/Xwayland"), "'/usr/bin/Xwayland'");
        assert_eq!(sh_single_quote("a'b"), "'a'\\''b'");
        assert_eq!(sh_single_quote("dir with spaces/$HOME/`id`"), "'dir with spaces/$HOME/`id`'");
    }

    #[test]
    fn xauthority_file_has_valid_records_and_0600_mode() {
        let dir = TestTempDir::new("rw-xauth-test");
        let xauth = dir.0.join("Xauthority");

        write_xauthority_file(&xauth, 12).unwrap();

        let meta = fs::metadata(&xauth).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);

        let bytes = fs::read(&xauth).unwrap();
        // Parse at least one MIT-MAGIC-COOKIE-1 record for display "12" with a 16-byte cookie.
        let mut cursor = 0;
        let mut records = 0;
        while cursor < bytes.len() {
            let family = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]);
            cursor += 2;
            let addr_len = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
            cursor += 2 + addr_len;
            let num_len = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
            cursor += 2;
            assert_eq!(&bytes[cursor..cursor + num_len], b"12");
            cursor += num_len;
            let name_len = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
            cursor += 2;
            assert_eq!(&bytes[cursor..cursor + name_len], b"MIT-MAGIC-COOKIE-1");
            cursor += name_len;
            let data_len = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
            cursor += 2;
            assert_eq!(data_len, 16);
            cursor += data_len;
            assert!(family == 256 || family == 65535);
            records += 1;
        }
        assert!(records >= 1);
    }

    #[test]
    fn xwayland_auth_shim_passes_auth_flag_via_child_path() {
        let dir = TestTempDir::new("rw-xshim-test");
        let xauth = dir.0.join("Xauthority");
        write_xauthority_file(&xauth, 7).unwrap();

        let fake_xwayland = Path::new("/bin/echo");
        let bin_dir = write_xwayland_auth_shim(&dir.0, fake_xwayland, &xauth).unwrap();

        let output = Command::new("Xwayland")
            .arg(":7")
            .arg("-rootless")
            .env_clear()
            .env("PATH", &bin_dir)
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        let expected = format!("-auth {} :7 -rootless\n", xauth.display());
        assert_eq!(stdout, expected);
    }
}
