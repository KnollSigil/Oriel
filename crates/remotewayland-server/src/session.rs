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
    pub portal_config_dir: PathBuf,
    pub dbus_address: Option<String>,
    pub wayland_socket: Option<OsString>,
    pub x11_display: Option<String>,
    pub xauthority: Option<PathBuf>,
    pub xdg_current_desktop: Option<String>,
    pub gtk_modules: Option<String>,
    pub color_scheme: Option<u32>,
    /// Client-synced `org.gnome.desktop.wm.preferences` `button-layout`.
    pub button_layout: Option<String>,
    /// Client-synced `org.freedesktop.appearance` `accent-color` (`None`: no preference).
    pub accent_color: Option<[f64; 3]>,
    /// Client-synced `org.freedesktop.appearance` `contrast`.
    pub contrast: Option<u32>,
}

/// The desktop settings a client synced, in the form [`write_desktop_settings`] applies.
struct SyncedSettings {
    scheme: Option<u32>,
    button_layout: String,
    accent_color: Option<[f64; 3]>,
    contrast: Option<u32>,
}

impl SessionEnvInfo {
    fn synced_settings(&self) -> SyncedSettings {
        SyncedSettings {
            scheme: self.color_scheme,
            button_layout: self
                .button_layout
                .clone()
                .unwrap_or_else(|| DEFAULT_BUTTON_LAYOUT.to_string()),
            accent_color: self.accent_color,
            contrast: self.contrast,
        }
    }
}

/// Serializes writers of the isolated portal config files (GSettings keyfile, kdeglobals).
static PORTAL_CONFIG_LOCK: Mutex<()> = Mutex::new(());

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

        let portal_config_dir = runtime_dir.join("portal-config");
        create_private_dir(&portal_config_dir)?;

        let inherited_desktop = std::env::var("XDG_CURRENT_DESKTOP")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let detected_backend = detect_portal_backend(inherited_desktop.as_deref());
        let xdg_current_desktop = inherited_desktop.or_else(|| {
            detected_backend.as_ref().map(|b| match b {
                PortalBackend::Kde(_) => "KDE".to_string(),
                PortalBackend::Gtk(_) => "GNOME".to_string(),
            })
        });
        let inherited_gtk_modules = std::env::var("GTK_MODULES").ok();
        let gtk_modules = build_gtk_modules(
            inherited_gtk_modules.as_deref(),
            has_gtk3_module("libwindow-decorations-gtk-module.so"),
            has_gtk3_module("libcolorreload-gtk-module.so"),
        );
        prepare_portal_config_dir(&portal_config_dir, detected_backend.as_ref());

        let info = SessionEnvInfo {
            runtime_dir: runtime_dir.clone(),
            portal_config_dir,
            dbus_address: dbus_address.clone(),
            wayland_socket: None,
            x11_display: None,
            xauthority: None,
            xdg_current_desktop: xdg_current_desktop.clone(),
            gtk_modules: gtk_modules.clone(),
            color_scheme: None,
            button_layout: None,
            accent_color: None,
            contrast: None,
        };
        let _ = SESSION_INFO.set(Mutex::new(info));

        info!(
            runtime_dir = %runtime_dir.display(),
            dbus = ?dbus_address,
            desktop = ?xdg_current_desktop,
            gtk_modules = ?gtk_modules,
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

    /// Record the Wayland socket name once the headless compositor is ready,
    /// sync the isolated activation environment into the private D-Bus daemon,
    /// and best-effort launch the detected portal backends + `xdg-desktop-portal`.
    pub fn set_wayland_socket_and_start_portal(&mut self, socket_name: &OsStr) {
        let portal_config_dir = if let Some(lock) = SESSION_INFO.get() {
            if let Ok(mut guard) = lock.lock() {
                guard.wayland_socket = Some(socket_name.to_os_string());
                guard.portal_config_dir.clone()
            } else {
                self.runtime_dir.join("portal-config")
            }
        } else {
            self.runtime_dir.join("portal-config")
        };

        if self.dbus_address.is_some() {
            sync_dbus_activation_env();

            let mut backends_to_start: Vec<PortalBackend> = Vec::new();
            if let Some(p) = find_kde_portal_backend() {
                backends_to_start.push(PortalBackend::Kde(p));
            }
            if let Some(p) = find_gtk_portal_backend() {
                backends_to_start.push(PortalBackend::Gtk(p));
            }

            // Spawn all detected backends first, then wait against a shared deadline
            // so startup latency does not multiply per backend.
            let mut pending_names: Vec<(&'static str, usize)> = Vec::new();
            for backend in &backends_to_start {
                match start_portal_backend(
                    backend,
                    &self.runtime_dir,
                    &portal_config_dir,
                    socket_name,
                ) {
                    Ok(child) => {
                        let idx = self.children.len();
                        self.children.push(child);
                        let bus_name = match backend {
                            PortalBackend::Kde(_) => "org.freedesktop.impl.portal.desktop.kde",
                            PortalBackend::Gtk(_) => "org.freedesktop.impl.portal.desktop.gtk",
                        };
                        pending_names.push((bus_name, idx));
                    }
                    Err(err) => {
                        warn!(
                            error = %err,
                            "portal backend not started inside isolated session"
                        );
                    }
                }
            }
            wait_for_portal_backends(
                &mut self.children,
                &pending_names,
                Duration::from_millis(800),
            );

            // Our own Settings backend must own its bus name before the frontend starts.
            // It serves what `write_desktop_settings` last wrote.
            let portal_dir = self.dbus_address.as_deref().and_then(|addr| {
                match crate::settings_portal::start(addr) {
                    Ok(()) => prepare_portal_definitions_dir(&portal_config_dir),
                    Err(err) => {
                        warn!(error = %err, "settings portal backend not started");
                        None
                    }
                }
            });

            match start_xdg_desktop_portal(
                &self.runtime_dir,
                &portal_config_dir,
                socket_name,
                portal_dir.as_deref(),
            ) {
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
            bail!(
                "timed out waiting for private D-Bus socket at {}",
                bus_path.display()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let _ = set_mode(&bus_path, 0o600);
    info!(address = %address, "private D-Bus session daemon started");
    Ok((address, child))
}

#[derive(Debug, Clone)]
enum PortalBackend {
    Kde(PathBuf),
    Gtk(PathBuf),
}

fn find_first_file(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(Path::new)
        .find(|p| p.is_file())
        .map(Path::to_path_buf)
}

fn find_kde_portal_backend() -> Option<PathBuf> {
    find_first_file(&[
        "/usr/libexec/xdg-desktop-portal-kde",
        "/usr/lib/xdg-desktop-portal-kde",
        "/usr/lib/x86_64-linux-gnu/xdg-desktop-portal-kde",
        "/usr/lib/aarch64-linux-gnu/xdg-desktop-portal-kde",
    ])
}

fn find_gtk_portal_backend() -> Option<PathBuf> {
    find_first_file(&[
        "/usr/libexec/xdg-desktop-portal-gtk",
        "/usr/lib/xdg-desktop-portal-gtk",
        "/usr/lib/x86_64-linux-gnu/xdg-desktop-portal-gtk",
        "/usr/lib/aarch64-linux-gnu/xdg-desktop-portal-gtk",
    ])
}

fn detect_portal_backend(preferred_desktop: Option<&str>) -> Option<PortalBackend> {
    let kde = find_kde_portal_backend();
    let gtk = find_gtk_portal_backend();

    if let Some(desk) = preferred_desktop {
        let lower = desk.to_ascii_lowercase();
        if lower.contains("kde") {
            if let Some(p) = kde {
                return Some(PortalBackend::Kde(p));
            }
        } else if lower.contains("gnome") || lower.contains("gtk") {
            if let Some(p) = gtk {
                return Some(PortalBackend::Gtk(p));
            }
        }
    }

    kde.map(PortalBackend::Kde)
        .or_else(|| gtk.map(PortalBackend::Gtk))
}

fn has_gtk3_module(filename: &str) -> bool {
    [
        "/usr/lib64/gtk-3.0/modules",
        "/usr/lib/gtk-3.0/modules",
        "/usr/lib/x86_64-linux-gnu/gtk-3.0/modules",
        "/usr/lib/aarch64-linux-gnu/gtk-3.0/modules",
    ]
    .iter()
    .any(|dir| Path::new(dir).join(filename).is_file())
}

fn build_gtk_modules(
    inherited: Option<&str>,
    has_window_decorations: bool,
    has_colorreload: bool,
) -> Option<String> {
    let mut parts: Vec<String> = inherited
        .unwrap_or("")
        .split(':')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if has_window_decorations && !parts.iter().any(|m| m == "window-decorations-gtk-module") {
        parts.push("window-decorations-gtk-module".to_string());
    }
    if has_colorreload && !parts.iter().any(|m| m == "colorreload-gtk-module") {
        parts.push("colorreload-gtk-module".to_string());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(":"))
    }
}

fn dbus_name_has_owner(bus_name: &str) -> bool {
    let mut cmd = Command::new("gdbus");
    apply_session_env(&mut cmd, None);
    cmd.args([
        "call",
        "--session",
        "--timeout",
        "2",
        "--dest",
        "org.freedesktop.DBus",
        "--object-path",
        "/org/freedesktop/DBus",
        "--method",
        "org.freedesktop.DBus.NameHasOwner",
        bus_name,
    ])
    .stdin(Stdio::null())
    .stderr(Stdio::null())
    .output()
    .is_ok_and(|out| out.status.success() && String::from_utf8_lossy(&out.stdout).contains("true"))
}

fn wait_for_portal_backends(
    children: &mut [Child],
    pending: &[(&'static str, usize)],
    timeout: Duration,
) {
    if pending.is_empty() {
        return;
    }
    let mut remaining: Vec<(&'static str, usize)> = pending.to_vec();
    let deadline = Instant::now() + timeout;
    while !remaining.is_empty() && Instant::now() < deadline {
        remaining.retain(|&(bus_name, child_idx)| {
            if let Some(child) = children.get_mut(child_idx) {
                if let Ok(Some(status)) = child.try_wait() {
                    warn!(bus_name, %status, "portal backend exited before owning D-Bus name");
                    return false;
                }
            }
            !dbus_name_has_owner(bus_name)
        });
        if !remaining.is_empty() {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    for (bus_name, _) in remaining {
        warn!(
            bus_name,
            "timed out waiting for portal backend to own D-Bus name"
        );
    }
}

/// Atomically write `contents` to `path` via a unique sibling temp file and `fs::rename`
/// so file monitors (`GFileMonitor`/inotify) never observe a truncated write or collide
/// across concurrent callers.
fn write_file_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{file_name}.{pid}.{seq}.tmp"));
    fs::write(&tmp, contents)
        .with_context(|| format!("failed to write temp file {}", tmp.display()))?;
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("failed to rename {} -> {}", tmp.display(), path.display()));
    }
    Ok(())
}

fn find_xdg_data_path(rel_path: &str) -> Option<PathBuf> {
    let data_dirs = std::env::var_os("XDG_DATA_DIRS")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| OsString::from("/usr/local/share:/usr/share"));
    for dir in std::env::split_paths(&data_dirs) {
        let candidate = dir.join(rel_path);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let fallback = Path::new("/usr/share").join(rel_path);
    fallback.exists().then_some(fallback)
}

fn host_config_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
}

fn prepare_portal_config_dir(portal_config_dir: &Path, backend: Option<&PortalBackend>) {
    // Seed portal-config/kdeglobals (and kdeglobals.orig for scheme=0 reset) from host
    // ~/.config/kdeglobals if present, without ever mutating the host config.
    if let Some(host_cfg) = host_config_dir() {
        let src_kdeglobals = host_cfg.join("kdeglobals");
        if src_kdeglobals.is_file() {
            if let Err(err) = fs::copy(&src_kdeglobals, portal_config_dir.join("kdeglobals")) {
                warn!(error = %err, "failed to seed isolated kdeglobals");
            }
            let _ = fs::copy(&src_kdeglobals, portal_config_dir.join("kdeglobals.orig"));
        }
    }

    // Seed the titlebar button layout before the portal starts so apps get minimize and
    // maximize even if the client never syncs its own layout. The color scheme and theme
    // stay unset (schema defaults) until the client syncs one.
    write_desktop_settings(
        portal_config_dir,
        &SyncedSettings {
            scheme: None,
            button_layout: DEFAULT_BUTTON_LAYOUT.to_string(),
            accent_color: None,
            contrast: None,
        },
    );

    let portal_conf_dir = portal_config_dir.join("xdg-desktop-portal");
    if create_private_dir(&portal_conf_dir).is_ok() {
        let pref = match backend {
            Some(PortalBackend::Gtk(_)) => "gtk;kde",
            _ => "kde;gtk",
        };
        // Our own Settings backend comes first: xdg-desktop-portal takes each key from the
        // first backend that has it, so the settings RemoteWayland decides (color scheme,
        // titlebar buttons, font rendering) are the same on every server (see
        // `settings_portal`). gtk then kde serve every other key, while KDE handles
        // FileChooser/OpenURI/etc.
        let settings_pref = if find_gtk_portal_backend().is_some() {
            "gtk;kde"
        } else {
            pref
        };
        let ours = crate::settings_portal::PORTAL_ID;
        let conf = format!(
            "[preferred]\ndefault={pref}\norg.freedesktop.impl.portal.Settings={ours};{settings_pref}\n"
        );
        if let Err(err) = write_file_atomic(&portal_conf_dir.join("portals.conf"), conf.as_bytes())
        {
            warn!(error = %err, "failed to write isolated portals.conf");
        }
    }
}

fn sync_dbus_activation_env() {
    let Some(lock) = SESSION_INFO.get() else {
        return;
    };
    let Ok(info) = lock.lock() else {
        return;
    };
    if info.dbus_address.is_none() {
        return;
    }
    let mut cmd = Command::new("dbus-update-activation-environment");
    let mut vars = vec![
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE",
        "XDG_CONFIG_DIRS",
        "GTK_USE_PORTAL",
        "PULSE_SINK",
        "PULSE_SOURCE",
    ];
    if crate::audio::isolated_pw_runtime_dir().is_some() {
        vars.extend_from_slice(&["PIPEWIRE_RUNTIME_DIR", "PULSE_RUNTIME_PATH", "PULSE_SERVER"]);
    }
    if info.wayland_socket.is_some() {
        vars.push("WAYLAND_DISPLAY");
    }
    if info.x11_display.is_some() {
        vars.push("DISPLAY");
    }
    if info.xauthority.is_some() {
        vars.push("XAUTHORITY");
    }
    if let Some(ref desk) = info.xdg_current_desktop {
        vars.push("XDG_CURRENT_DESKTOP");
        if desk.eq_ignore_ascii_case("kde") {
            vars.push("QT_QPA_PLATFORMTHEME");
        }
    }
    if info.gtk_modules.is_some() {
        vars.push("GTK_MODULES");
    }
    drop(info);

    apply_session_env(&mut cmd, None);
    cmd.args(&vars)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _ = cmd.status();
}

fn spawn_portal_process(
    bin: &Path,
    runtime_dir: &Path,
    portal_config_dir: &Path,
    wayland_socket: &OsStr,
    extra_env: &[(&str, &str)],
    label: &str,
) -> Result<Child> {
    let mut cmd = Command::new(bin);
    cmd.arg("--replace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_session_env(&mut cmd, Some(wayland_socket));
    cmd.env("XDG_RUNTIME_DIR", runtime_dir)
        .env("XDG_CONFIG_HOME", portal_config_dir);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }

    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", bin.display()))?;
    info!(bin = %bin.display(), pid = child.id(), "{label}");
    Ok(child)
}

fn start_portal_backend(
    backend: &PortalBackend,
    runtime_dir: &Path,
    portal_config_dir: &Path,
    wayland_socket: &OsStr,
) -> Result<Child> {
    let (bin, extra): (&Path, &[(&str, &str)]) = match backend {
        PortalBackend::Kde(p) => (p, &[]),
        PortalBackend::Gtk(p) => (p, &[("GSETTINGS_BACKEND", "keyfile")]),
    };
    spawn_portal_process(
        bin,
        runtime_dir,
        portal_config_dir,
        wayland_socket,
        extra,
        "isolated portal backend started",
    )
}

fn find_portal_binary() -> Option<PathBuf> {
    find_first_file(&[
        "/usr/libexec/xdg-desktop-portal",
        "/usr/lib/xdg-desktop-portal",
        "/usr/lib/xdg-desktop-portal/xdg-desktop-portal",
    ])
}

/// Directories to search for system `.portal` backend definitions, most preferred first:
/// the data dir of the xdg-desktop-portal install that will run (`<prefix>/share`, derived
/// from its binary path), then `XDG_DATA_DIRS` (default `/usr/local/share:/usr/share`).
fn portal_definition_search_dirs(
    portal_binary: Option<&Path>,
    xdg_data_dirs: Option<&str>,
) -> Vec<PathBuf> {
    let prefix_share = portal_binary.and_then(|bin| {
        bin.ancestors()
            .find(|a| {
                matches!(
                    a.file_name().and_then(|n| n.to_str()),
                    Some("libexec" | "lib" | "lib64")
                )
            })?
            .parent()
            .map(|prefix| prefix.join("share"))
    });
    let data_dirs = xdg_data_dirs
        .filter(|v| !v.is_empty())
        .unwrap_or("/usr/local/share:/usr/share");
    let mut dirs: Vec<PathBuf> = Vec::new();
    for share in prefix_share.into_iter().chain(
        data_dirs
            .split(':')
            .filter(|d| !d.is_empty())
            .map(PathBuf::from),
    ) {
        let dir = share.join("xdg-desktop-portal/portals");
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    dirs
}

/// Build an isolated `.portal` definitions dir (for `XDG_DESKTOP_PORTAL_DIR`): links to every
/// system backend definition, our own settings backend, and a copy of `portals.conf`.
/// Returns `None` (leave `XDG_DESKTOP_PORTAL_DIR` unset) if no system definitions were
/// found, since the frontend would then lose every other backend.
fn prepare_portal_definitions_dir(portal_config_dir: &Path) -> Option<PathBuf> {
    let dir = portal_config_dir.join("portals");
    if let Err(err) = create_private_dir(&dir) {
        warn!(error = %err, "failed to create isolated portal definitions dir");
        return None;
    }
    let search_dirs = portal_definition_search_dirs(
        find_portal_binary().as_deref(),
        std::env::var("XDG_DATA_DIRS").ok().as_deref(),
    );
    let mut linked = std::collections::HashSet::new();
    for search_dir in &search_dirs {
        let Ok(entries) = fs::read_dir(search_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            if path.extension().is_none_or(|e| e != "portal") || linked.contains(&name) {
                continue;
            }
            let link = dir.join(&name);
            let _ = fs::remove_file(&link);
            match std::os::unix::fs::symlink(&path, &link) {
                Ok(()) => {
                    linked.insert(name);
                }
                Err(err) => {
                    warn!(error = %err, path = %path.display(), "failed to link portal definition");
                }
            }
        }
    }
    if linked.is_empty() {
        warn!(
            ?search_dirs,
            "no system portal definitions found; settings portal backend not registered"
        );
        return None;
    }
    // With XDG_DESKTOP_PORTAL_DIR set, xdg-desktop-portal reads `portals.conf` only from
    // that dir (not XDG_CONFIG_HOME), so the backend preferences must be copied there.
    let conf = portal_config_dir.join("xdg-desktop-portal/portals.conf");
    if let Err(err) = fs::copy(&conf, dir.join("portals.conf")) {
        warn!(error = %err, "failed to copy portals.conf into portal definitions dir");
        return None;
    }
    let ours = dir.join(format!("{}.portal", crate::settings_portal::PORTAL_ID));
    if let Err(err) = write_file_atomic(&ours, crate::settings_portal::PORTAL_FILE.as_bytes()) {
        warn!(error = %err, "failed to write settings portal definition");
        return None;
    }
    Some(dir)
}

fn start_xdg_desktop_portal(
    runtime_dir: &Path,
    portal_config_dir: &Path,
    wayland_socket: &OsStr,
    portal_definitions_dir: Option<&Path>,
) -> Result<Child> {
    let Some(bin) = find_portal_binary() else {
        bail!("xdg-desktop-portal binary not found in /usr/libexec or /usr/lib");
    };
    let definitions_dir = portal_definitions_dir.map(|d| d.to_string_lossy().into_owned());
    let extra_env: Vec<(&str, &str)> = definitions_dir
        .as_deref()
        .map(|d| ("XDG_DESKTOP_PORTAL_DIR", d))
        .into_iter()
        .collect();
    spawn_portal_process(
        &bin,
        runtime_dir,
        portal_config_dir,
        wayland_socket,
        &extra_env,
        "isolated xdg-desktop-portal started",
    )
}

/// GNOME's `color-scheme` and `gtk-theme` for a client color scheme (`0` = no preference,
/// `1` = prefer dark, `2` = prefer light).
fn gnome_appearance(scheme: u32, has_breeze_gtk: bool) -> (&'static str, &'static str) {
    match (scheme, has_breeze_gtk) {
        (1, true) => ("prefer-dark", "Breeze-Dark"),
        (1, false) => ("prefer-dark", "Adwaita-dark"),
        (2, true) => ("prefer-light", "Breeze"),
        (2, false) => ("prefer-light", "Adwaita"),
        (_, true) => ("default", "Breeze"),
        (_, false) => ("default", "Adwaita"),
    }
}

/// Build the isolated GSettings keyfile. `scheme` is `None` until the client syncs a
/// color scheme, in which case the interface keys are left at their schema defaults.
fn format_gsettings_keyfile(
    scheme: Option<u32>,
    has_breeze_gtk: bool,
    button_layout: &str,
) -> String {
    let mut out = String::new();
    if let Some(scheme) = scheme {
        let (cs, theme) = gnome_appearance(scheme, has_breeze_gtk);
        out.push_str(&format!(
            "[org/gnome/desktop/interface]\ncolor-scheme='{cs}'\ngtk-theme='{theme}'\n\n"
        ));
    }
    out.push_str(&format!(
        "[org/gnome/desktop/wm/preferences]\nbutton-layout='{button_layout}'\n"
    ));
    out
}

/// Titlebar button layout served to GTK/Chromium/Firefox through the isolated portal
/// until the client syncs its own. GNOME's schema default is `appmenu:close`, which
/// hides minimize and maximize; RemoteWayland windows support both, so show them.
const DEFAULT_BUTTON_LAYOUT: &str = "appmenu:minimize,maximize,close";

/// Apply the color scheme and titlebar button layout to everything that carries them:
/// - our settings portal backend, which serves them to portal-using apps (GTK3 with
///   `GTK_USE_PORTAL=1`, sandboxed apps, Chromium, Firefox) ahead of the other backends;
/// - the GSettings keyfile that xdg-desktop-portal-gtk serves; it must agree with our
///   backend, since xdg-desktop-portal forwards both backends' change signals;
/// - `gtk-3.0`/`gtk-4.0` `settings.ini` (the dir is on apps' `XDG_CONFIG_DIRS`) for GTK
///   apps that read settings directly instead of through the portal.
fn write_desktop_settings(portal_config_dir: &Path, synced: &SyncedSettings) {
    let SyncedSettings {
        scheme,
        ref button_layout,
        accent_color,
        contrast,
    } = *synced;
    let has_breeze_gtk = find_xdg_data_path("themes/Breeze-Dark").is_some();
    crate::settings_portal::update(&crate::settings_portal::DesktopSettings {
        appearance: scheme.map(|scheme| {
            let (gnome_color_scheme, gtk_theme) = gnome_appearance(scheme, has_breeze_gtk);
            crate::settings_portal::Appearance {
                color_scheme: scheme,
                gnome_color_scheme,
                gtk_theme,
            }
        }),
        button_layout,
        accent_color,
        contrast,
    });
    let glib_dir = portal_config_dir.join("glib-2.0/settings");
    if create_private_dir(&glib_dir).is_ok() {
        let keyfile = format_gsettings_keyfile(scheme, has_breeze_gtk, button_layout);
        if let Err(err) = write_file_atomic(&glib_dir.join("keyfile"), keyfile.as_bytes()) {
            warn!(error = %err, "failed to update isolated GSettings keyfile");
        }
    }
    let settings_ini = format!("[Settings]\ngtk-decoration-layout={button_layout}\n");
    for gtk_dir in ["gtk-3.0", "gtk-4.0"] {
        let dir = portal_config_dir.join(gtk_dir);
        if create_private_dir(&dir).is_ok() {
            if let Err(err) = write_file_atomic(&dir.join("settings.ini"), settings_ini.as_bytes())
            {
                warn!(error = %err, gtk_dir, "failed to update isolated GTK settings.ini");
            }
        }
    }
}

/// Update `[General]` `ColorScheme` and `[Colors:Window]` `BackgroundNormal` / `ForegroundNormal`
/// inside an isolated `kdeglobals` buffer so `xdg-desktop-portal-kde` reports the synced
/// `org.freedesktop.appearance` `color-scheme` (`1` = dark, `2` = light).
fn update_kdeglobals_color_scheme(existing: &str, scheme: u32) -> String {
    let (scheme_name, bg, fg) = match scheme {
        1 => ("BreezeDark", "35,38,41", "239,240,241"),
        2 => ("BreezeLight", "239,240,241", "35,38,41"),
        _ => return existing.to_string(),
    };

    let mut out = String::with_capacity(existing.len() + 128);
    let mut section = String::new();
    let mut saw_general = false;
    let mut wrote_scheme = false;
    let mut saw_colors_window = false;
    let mut wrote_bg = false;
    let mut wrote_fg = false;

    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if section == "General" && !wrote_scheme {
                out.push_str(&format!("ColorScheme={scheme_name}\n"));
                wrote_scheme = true;
            } else if section == "Colors:Window" {
                if !wrote_bg {
                    out.push_str(&format!("BackgroundNormal={bg}\n"));
                    wrote_bg = true;
                }
                if !wrote_fg {
                    out.push_str(&format!("ForegroundNormal={fg}\n"));
                    wrote_fg = true;
                }
            }
            section = trimmed[1..trimmed.len() - 1].to_string();
            if section == "General" {
                saw_general = true;
            } else if section == "Colors:Window" {
                saw_colors_window = true;
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }

        if section == "General" && trimmed.starts_with("ColorScheme=") {
            out.push_str(&format!("ColorScheme={scheme_name}\n"));
            wrote_scheme = true;
            continue;
        }
        if section == "Colors:Window" && trimmed.starts_with("BackgroundNormal=") {
            out.push_str(&format!("BackgroundNormal={bg}\n"));
            wrote_bg = true;
            continue;
        }
        if section == "Colors:Window" && trimmed.starts_with("ForegroundNormal=") {
            out.push_str(&format!("ForegroundNormal={fg}\n"));
            wrote_fg = true;
            continue;
        }

        out.push_str(line);
        out.push('\n');
    }

    if section == "General" && !wrote_scheme {
        out.push_str(&format!("ColorScheme={scheme_name}\n"));
    } else if section == "Colors:Window" {
        if !wrote_bg {
            out.push_str(&format!("BackgroundNormal={bg}\n"));
        }
        if !wrote_fg {
            out.push_str(&format!("ForegroundNormal={fg}\n"));
        }
    }

    if !saw_general {
        out.push_str(&format!("\n[General]\nColorScheme={scheme_name}\n"));
    }
    if !saw_colors_window {
        out.push_str(&format!(
            "\n[Colors:Window]\nBackgroundNormal={bg}\nForegroundNormal={fg}\n"
        ));
    }
    out
}

/// KDE `KGlobalSettings::ChangeType` values emitted on `/KGlobalSettings`.
const KGLOBAL_PALETTE_CHANGED: &str = "0";
const KGLOBAL_SETTINGS_CHANGED: &str = "4";

/// Record a client-synced setting with `update`, then write every isolated settings target
/// (see [`write_desktop_settings`]) from the full synced state. Returns the portal config
/// dir, whether the session has a D-Bus, and the synced state.
fn update_synced_settings(
    update: impl FnOnce(&mut SessionEnvInfo),
) -> Option<(PathBuf, bool, SyncedSettings)> {
    let (portal_config_dir, has_dbus, synced) = {
        let lock = SESSION_INFO.get()?;
        let mut info = lock.lock().ok()?;
        update(&mut info);
        (
            info.portal_config_dir.clone(),
            info.dbus_address.is_some(),
            info.synced_settings(),
        )
    };
    write_desktop_settings(&portal_config_dir, &synced);
    Some((portal_config_dir, has_dbus, synced))
}

/// Apply the client-synced `org.freedesktop.appearance` `color-scheme`
/// (`0` = no preference, `1` = prefer dark, `2` = prefer light) to the isolated
/// portal configuration and notify the isolated portal backend live.
pub fn apply_color_scheme(scheme: u32) {
    if scheme > 2 {
        warn!(scheme, "ignoring invalid client color-scheme value");
        return;
    }
    let _apply_guard = PORTAL_CONFIG_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Serves `color-scheme` and `gtk-theme` from our settings portal backend (live over
    // org.freedesktop.portal.Settings) and the GSettings keyfile for xdg-desktop-portal-gtk.
    let Some((portal_config_dir, has_dbus, synced)) =
        update_synced_settings(|info| info.color_scheme = Some(scheme))
    else {
        return;
    };

    // Update isolated kdeglobals for xdg-desktop-portal-kde and Qt/KDE apps. Notify them
    // even if the file is unchanged (a color scheme switch back to the same palette).
    rewrite_kdeglobals(&portal_config_dir, &synced);
    if has_dbus {
        notify_kde_settings_changed();
    }

    info!(
        scheme,
        "applied synced client color-scheme to isolated portal"
    );
}

/// Apply the client-synced `org.freedesktop.appearance` `accent-color` (`None` = no
/// preference: the server's own accent stands) to our settings portal backend and the
/// isolated `kdeglobals`, which xdg-desktop-portal-kde and Qt/KDE apps read.
pub fn apply_accent_color(rgb: Option<[f64; 3]>) {
    if rgb.is_some_and(|rgb| !remotewayland_proto::is_valid_accent_color(&rgb)) {
        warn!(?rgb, "ignoring invalid client accent-color value");
        return;
    }
    let _apply_guard = PORTAL_CONFIG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((portal_config_dir, has_dbus, synced)) =
        update_synced_settings(|info| info.accent_color = rgb)
    else {
        return;
    };
    // Mirror the accent into kdeglobals: the KDE backend re-announces its accent (the
    // palette highlight) on every KGlobalSettings change, and xdg-desktop-portal forwards
    // that to apps unfiltered, so it must report the same accent as ours.
    if rewrite_kdeglobals(&portal_config_dir, &synced) && has_dbus {
        notify_kde_settings_changed();
    }
    info!(
        ?rgb,
        "applied synced client accent-color to isolated portal"
    );
}

/// Apply the client-synced `org.freedesktop.appearance` `contrast` (`0` = no preference,
/// `1` = higher contrast) to our settings portal backend.
pub fn apply_contrast(contrast: u32) {
    if contrast > 1 {
        warn!(contrast, "ignoring invalid client contrast value");
        return;
    }
    let _apply_guard = PORTAL_CONFIG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if update_synced_settings(|info| info.contrast = Some(contrast)).is_some() {
        info!(
            contrast,
            "applied synced client contrast to isolated portal"
        );
    }
}

/// Tell the KDE portal backend and Qt/KDE apps on the isolated D-Bus that `kdeglobals`
/// (palette and general settings) changed.
fn notify_kde_settings_changed() {
    for change_type in [KGLOBAL_PALETTE_CHANGED, KGLOBAL_SETTINGS_CHANGED] {
        let mut cmd = Command::new("gdbus");
        apply_session_env(&mut cmd, None);
        let res = cmd
            .args([
                "emit",
                "--session",
                "--object-path",
                "/KGlobalSettings",
                "--signal",
                "org.kde.KGlobalSettings.notifyChange",
                change_type,
                "0",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Err(err) = res {
            warn!(error = %err, change_type, "failed to emit KGlobalSettings.notifyChange");
        }
    }
}

/// Apply the client-synced titlebar `button-layout` (`None` = the client has no preference,
/// use [`DEFAULT_BUTTON_LAYOUT`]) to the isolated GTK settings and our settings portal
/// backend, which notify GTK/Chromium/Firefox live.
pub fn apply_button_layout(layout: Option<&str>) {
    if let Some(layout) = layout {
        if !remotewayland_proto::is_valid_button_layout(layout) {
            warn!(layout, "ignoring invalid client button-layout value");
            return;
        }
    }
    let _apply_guard = PORTAL_CONFIG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if update_synced_settings(|info| info.button_layout = layout.map(str::to_string)).is_some() {
        info!(
            layout = layout.unwrap_or(DEFAULT_BUTTON_LAYOUT),
            from_client = layout.is_some(),
            "applied titlebar button-layout to isolated portal"
        );
    }
}

/// Rebuild the isolated `kdeglobals` from the synced color scheme and accent: the color
/// scheme's palette (or the host's own `kdeglobals` without a color scheme preference),
/// with the accent as the selection color. Returns whether the file changed.
fn rewrite_kdeglobals(portal_config_dir: &Path, synced: &SyncedSettings) -> bool {
    let kdeglobals_path = portal_config_dir.join("kdeglobals");
    let existing = fs::read_to_string(&kdeglobals_path).ok();
    let host = || fs::read_to_string(portal_config_dir.join("kdeglobals.orig")).ok();
    let base = match synced.scheme {
        None => host().or_else(|| existing.clone()),
        Some(0) => host().or_else(|| {
            find_xdg_data_path("color-schemes/BreezeLight.colors")
                .and_then(|p| fs::read_to_string(p).ok())
        }),
        Some(scheme) => {
            let rel = if scheme == 1 {
                "color-schemes/BreezeDark.colors"
            } else {
                "color-schemes/BreezeLight.colors"
            };
            let base = find_xdg_data_path(rel)
                .and_then(|p| fs::read_to_string(p).ok())
                .or_else(|| existing.clone())
                .unwrap_or_default();
            Some(update_kdeglobals_color_scheme(&base, scheme))
        }
    };
    let Some(mut updated) = base else {
        return false;
    };
    if let Some(rgb) = synced.accent_color {
        updated = set_kdeglobals_accent(&updated, rgb);
    }
    if existing.as_deref() == Some(updated.as_str()) {
        return false;
    }
    if let Err(err) = write_file_atomic(&kdeglobals_path, updated.as_bytes()) {
        warn!(error = %err, "failed to update isolated kdeglobals");
        return false;
    }
    true
}

/// Apply an accent color to a `kdeglobals` buffer the way KDE does: as the selection
/// background (the palette highlight, which xdg-desktop-portal-kde reports as
/// `accent-color`), plus `[General]` `AccentColor`.
fn set_kdeglobals_accent(existing: &str, rgb: [f64; 3]) -> String {
    let [r, g, b] = rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    let value = format!("{r},{g},{b}");
    let out = set_ini_value(existing, "Colors:Selection", "BackgroundNormal", &value);
    set_ini_value(&out, "General", "AccentColor", &value)
}

/// Set `key=value` in `[section]` of an INI buffer, replacing an existing entry or adding
/// it (and the section) if missing.
fn set_ini_value(existing: &str, section: &str, key: &str, value: &str) -> String {
    let header = format!("[{section}]");
    let prefix = format!("{key}=");
    let entry = format!("{key}={value}");
    let mut out = String::with_capacity(existing.len() + entry.len() + header.len() + 4);
    let mut in_section = false;
    let mut written = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if in_section && !written {
                out.push_str(&entry);
                out.push('\n');
                written = true;
            }
            in_section = trimmed == header;
        } else if in_section && trimmed.starts_with(&prefix) {
            if !written {
                out.push_str(&entry);
                out.push('\n');
                written = true;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !written {
        if !in_section {
            if !out.is_empty() && !out.ends_with("\n\n") {
                out.push('\n');
            }
            out.push_str(&header);
            out.push('\n');
        }
        out.push_str(&entry);
        out.push('\n');
    }
    out
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
    sync_dbus_activation_env();
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

/// Apply the active XWayland `DISPLAY` and `XAUTHORITY` to `cmd`.
/// Returns `true` if an XWayland display is currently active.
pub fn apply_x11_env(cmd: &mut Command) -> bool {
    let Some(lock) = SESSION_INFO.get() else {
        return false;
    };
    let Ok(info) = lock.lock() else {
        return false;
    };
    let Some(disp) = &info.x11_display else {
        return false;
    };
    cmd.env("DISPLAY", disp);
    if let Some(xauth) = &info.xauthority {
        cmd.env("XAUTHORITY", xauth);
    } else {
        cmd.env_remove("XAUTHORITY");
    }
    true
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
        append_xauth_record(
            &mut data,
            256,
            hostname.as_bytes(),
            display_str.as_bytes(),
            &cookie,
        );
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
        .with_context(|| {
            format!(
                "failed to create Xwayland auth shim {}",
                shim_path.display()
            )
        })?;
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
        .env("XDG_SESSION_TYPE", "wayland")
        .env("GTK_USE_PORTAL", "1")
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
            if let Some(desk) = &info.xdg_current_desktop {
                cmd.env("XDG_CURRENT_DESKTOP", desk);
                if desk.eq_ignore_ascii_case("kde")
                    && std::env::var_os("QT_QPA_PLATFORMTHEME").is_none()
                {
                    cmd.env("QT_QPA_PLATFORMTHEME", "kde");
                }
            }
            if let Some(mods) = &info.gtk_modules {
                cmd.env("GTK_MODULES", mods);
            }
            // Prepend our isolated portal-config dir to XDG_CONFIG_DIRS so Qt/KDE apps
            // pick up the synced BreezeDark/BreezeLight kdeglobals without mutating ~/.config.
            let mut config_dirs = OsString::from(&info.portal_config_dir);
            config_dirs.push(":");
            match std::env::var_os("XDG_CONFIG_DIRS").filter(|v| !v.is_empty()) {
                Some(existing) => config_dirs.push(existing),
                None => config_dirs.push("/etc/xdg"),
            }
            cmd.env("XDG_CONFIG_DIRS", config_dirs);
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
        assert_eq!(
            sh_single_quote("dir with spaces/$HOME/`id`"),
            "'dir with spaces/$HOME/`id`'"
        );
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

    #[test]
    fn build_gtk_modules_preserves_and_deduplicates() {
        assert_eq!(
            build_gtk_modules(None, true, true).as_deref(),
            Some("window-decorations-gtk-module:colorreload-gtk-module")
        );
        assert_eq!(build_gtk_modules(None, false, false), None);
        assert_eq!(
            build_gtk_modules(Some("colorreload-gtk-module"), true, true).as_deref(),
            Some("colorreload-gtk-module:window-decorations-gtk-module")
        );
        assert_eq!(
            build_gtk_modules(
                Some("window-decorations-gtk-module:colorreload-gtk-module"),
                true,
                true
            )
            .as_deref(),
            Some("window-decorations-gtk-module:colorreload-gtk-module")
        );
    }

    #[test]
    fn set_ini_value_replaces_or_adds_entry() {
        assert_eq!(
            set_ini_value(
                "[General]\nAccentColor=1,2,3\nFont=x\n[KDE]\nA=1\n",
                "General",
                "AccentColor",
                "9"
            ),
            "[General]\nAccentColor=9\nFont=x\n[KDE]\nA=1\n"
        );
        assert_eq!(
            set_ini_value(
                "[General]\nFont=x\n\n[KDE]\nA=1\n",
                "General",
                "AccentColor",
                "9"
            ),
            "[General]\nFont=x\n\nAccentColor=9\n[KDE]\nA=1\n"
        );
        assert_eq!(
            set_ini_value("[KDE]\nA=1\n", "General", "AccentColor", "9"),
            "[KDE]\nA=1\n\n[General]\nAccentColor=9\n"
        );
        assert_eq!(
            set_ini_value("", "General", "AccentColor", "9"),
            "[General]\nAccentColor=9\n"
        );
        // Keys of the same name in other sections are left alone.
        assert_eq!(
            set_ini_value(
                "[Colors:View]\nBackgroundNormal=1\n",
                "Colors:Selection",
                "BackgroundNormal",
                "2"
            ),
            "[Colors:View]\nBackgroundNormal=1\n\n[Colors:Selection]\nBackgroundNormal=2\n"
        );
    }

    #[test]
    fn set_kdeglobals_accent_sets_selection_and_accent() {
        let out = set_kdeglobals_accent(
            "[Colors:Selection]\nBackgroundNormal=61,174,233\n",
            [0.2, 0.4, 1.0],
        );
        assert!(out.contains("[Colors:Selection]\nBackgroundNormal=51,102,255\n"));
        assert!(out.contains("[General]\nAccentColor=51,102,255\n"));
    }

    #[test]
    fn update_kdeglobals_color_scheme_sets_dark_and_light() {
        let base = "[General]\nFont=Noto Sans,10\nColorScheme=BreezeLight\n\n[Colors:Window]\nBackgroundNormal=239,240,241\n";
        let dark = update_kdeglobals_color_scheme(base, 1);
        assert!(dark.contains("ColorScheme=BreezeDark"));
        assert!(dark.contains("BackgroundNormal=35,38,41"));
        assert!(dark.contains("ForegroundNormal=239,240,241"));
        assert!(dark.contains("Font=Noto Sans,10"));

        let light = update_kdeglobals_color_scheme(&dark, 2);
        assert!(light.contains("ColorScheme=BreezeLight"));
        assert!(light.contains("BackgroundNormal=239,240,241"));

        // Empty input appends both missing sections cleanly.
        let from_empty = update_kdeglobals_color_scheme("", 1);
        assert!(from_empty.contains("[General]\nColorScheme=BreezeDark"));
        assert!(from_empty.contains("[Colors:Window]\nBackgroundNormal=35,38,41"));

        // scheme=0 leaves existing content unchanged.
        assert_eq!(update_kdeglobals_color_scheme(base, 0), base);
    }

    #[test]
    fn format_gsettings_keyfile_selects_expected_themes() {
        const WM: &str =
            "[org/gnome/desktop/wm/preferences]\nbutton-layout='appmenu:minimize,maximize,close'\n";
        let cases = [
            (
                1,
                true,
                "color-scheme='prefer-dark'\ngtk-theme='Breeze-Dark'",
            ),
            (
                1,
                false,
                "color-scheme='prefer-dark'\ngtk-theme='Adwaita-dark'",
            ),
            (2, true, "color-scheme='prefer-light'\ngtk-theme='Breeze'"),
            (0, false, "color-scheme='default'\ngtk-theme='Adwaita'"),
        ];
        for (scheme, breeze, interface) in cases {
            assert_eq!(
                format_gsettings_keyfile(Some(scheme), breeze, DEFAULT_BUTTON_LAYOUT),
                format!("[org/gnome/desktop/interface]\n{interface}\n\n{WM}")
            );
        }
    }

    #[test]
    fn format_gsettings_keyfile_without_scheme_leaves_theme_unset() {
        assert_eq!(
            format_gsettings_keyfile(None, true, "icon:minimize,maximize,close"),
            "[org/gnome/desktop/wm/preferences]\nbutton-layout='icon:minimize,maximize,close'\n"
        );
    }

    #[test]
    fn portal_definition_search_dirs_prefers_binary_prefix() {
        let dirs = portal_definition_search_dirs(
            Some(Path::new("/opt/xdp/libexec/xdg-desktop-portal")),
            Some("/usr/share:/opt/xdp/share::/usr/local/share"),
        );
        assert_eq!(
            dirs,
            [
                "/opt/xdp/share/xdg-desktop-portal/portals",
                "/usr/share/xdg-desktop-portal/portals",
                "/usr/local/share/xdg-desktop-portal/portals",
            ]
            .map(PathBuf::from)
        );
        let dirs = portal_definition_search_dirs(
            Some(Path::new("/usr/lib/xdg-desktop-portal/xdg-desktop-portal")),
            None,
        );
        assert_eq!(
            dirs,
            [
                "/usr/share/xdg-desktop-portal/portals",
                "/usr/local/share/xdg-desktop-portal/portals",
            ]
            .map(PathBuf::from)
        );
    }

    #[test]
    fn write_file_atomic_replaces_target_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!("rw-atomic-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("keyfile");

        write_file_atomic(&target, b"first").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"first");

        write_file_atomic(&target, b"second").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"second");

        let entries: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();
        assert_eq!(entries, vec![OsString::from("keyfile")]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_session_env_sets_xdg_session_type_wayland() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg("printf '%s' \"$XDG_SESSION_TYPE\"")
            .env("XDG_SESSION_TYPE", "tty");
        apply_session_env(&mut cmd, Some(OsStr::new("wayland-test")));
        let output = cmd.output().unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "wayland");
    }
}
