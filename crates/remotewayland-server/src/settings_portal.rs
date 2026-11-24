//! A minimal `org.freedesktop.impl.portal.Settings` backend on the isolated session bus.
//!
//! It serves the settings RemoteWayland decides itself, so every server presents the same
//! values to apps whichever other portal backends it has:
//!
//! - `org.gnome.desktop.wm.preferences` `button-layout`, synced from the client. Without
//!   it, GTK apps that read settings through the portal (GTK3 with `GTK_USE_PORTAL=1`,
//!   Flatpak apps such as Firefox and Chromium) fall back to GTK's `menu:close` titlebar
//!   on servers without xdg-desktop-portal-gtk, since the KDE portal serves no
//!   `org.gnome.*` keys.
//! - `org.gnome.desktop.interface` font rendering keys ([`FONT_SETTINGS`]). Without them,
//!   the same apps draw UI text with no antialiasing at all (Chromium's tab titles).
//! - Once the client syncs a color scheme, `org.freedesktop.appearance` `color-scheme` and
//!   the matching `org.gnome.desktop.interface` `color-scheme` and `gtk-theme`.
//! - Once the client syncs them, `org.freedesktop.appearance` `accent-color` (only when the
//!   client has an accent, so the server's own default stands otherwise) and `contrast`.
//!
//! It is listed first for the Settings interface. xdg-desktop-portal merges `ReadAll`
//! per key with the first-listed backend winning, and `Read`/`ReadOne` return the first
//! backend that has the key, so the KDE and GTK backends still serve every other key.
//! xdg-desktop-portal forwards `SettingChanged` from every backend unfiltered, so the
//! session writes the same values to the isolated GSettings keyfile that
//! xdg-desktop-portal-gtk serves (see `session::write_desktop_settings`).

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex, OnceLock};

use anyhow::{Context, Result};
use tracing::{info, warn};
use zbus::blocking::Connection;
use zbus::zvariant::{OwnedValue, Str, Value};

/// Portal id used in `portals.conf` and the `.portal` file name.
pub const PORTAL_ID: &str = "remotewayland";
const BUS_NAME: &str = "org.freedesktop.impl.portal.desktop.remotewayland";
const OBJECT_PATH: &str = "/org/freedesktop/portal/desktop";
const INTERFACE: &str = "org.freedesktop.impl.portal.Settings";
const APPEARANCE_NAMESPACE: &str = "org.freedesktop.appearance";
const INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";
const WM_NAMESPACE: &str = "org.gnome.desktop.wm.preferences";

/// `org.gnome.desktop.interface` font rendering keys and the values served: GNOME's
/// defaults, with grayscale antialiasing since subpixel (`rgba`) rendering would not
/// match the client's display once frames are streamed and scaled.
const FONT_SETTINGS: [(&str, &str); 3] = [
    ("font-antialiasing", "grayscale"),
    ("font-hinting", "slight"),
    ("font-rgba-order", "rgb"),
];

/// Contents of the `.portal` file that makes xdg-desktop-portal use this backend.
pub const PORTAL_FILE: &str = "[portal]\n\
     DBusName=org.freedesktop.impl.portal.desktop.remotewayland\n\
     Interfaces=org.freedesktop.impl.portal.Settings;\n";

/// The desktop settings the session decides, as written to the isolated GSettings keyfile.
pub struct DesktopSettings<'a> {
    /// `None` until the client syncs a color scheme; the other backends' values stand.
    pub appearance: Option<Appearance<'a>>,
    pub button_layout: &'a str,
    /// sRGB components in `[0, 1]`; `None` leaves the accent to the other backends.
    pub accent_color: Option<[f64; 3]>,
    /// `0` = no preference, `1` = higher contrast; `None` until the client syncs one.
    pub contrast: Option<u32>,
}

/// A client-synced color scheme in the forms apps read it.
pub struct Appearance<'a> {
    /// `org.freedesktop.appearance` `color-scheme`: `0` no preference, `1` dark, `2` light.
    pub color_scheme: u32,
    /// `org.gnome.desktop.interface` `color-scheme` (`default`, `prefer-dark`, ...).
    pub gnome_color_scheme: &'a str,
    pub gtk_theme: &'a str,
}

#[derive(Debug, Clone, PartialEq)]
enum Setting {
    Str(String),
    U32(u32),
    /// `(ddd)`, as `org.freedesktop.appearance` `accent-color`.
    Rgb([f64; 3]),
}

impl Setting {
    fn to_value(&self) -> Value<'_> {
        match self {
            Setting::Str(s) => Value::from(Str::from(s.as_str())),
            Setting::U32(v) => Value::from(*v),
            Setting::Rgb([r, g, b]) => Value::from((*r, *g, *b)),
        }
    }

    fn to_owned_value(&self) -> OwnedValue {
        match self {
            Setting::Str(s) => OwnedValue::from(Str::from(s.clone())),
            Setting::U32(v) => OwnedValue::from(*v),
            Setting::Rgb(_) => OwnedValue::try_from(self.to_value())
                .expect("a (ddd) structure holds no file descriptors"),
        }
    }
}

type Key = (&'static str, &'static str);

/// The served settings by `(namespace, key)`.
#[derive(Debug, Default)]
struct SettingsStore {
    settings: BTreeMap<Key, Setting>,
}

impl SettingsStore {
    fn desired(desktop: &DesktopSettings) -> BTreeMap<Key, Setting> {
        let mut out = BTreeMap::new();
        for (key, value) in FONT_SETTINGS {
            out.insert((INTERFACE_NAMESPACE, key), Setting::Str(value.to_string()));
        }
        out.insert(
            (WM_NAMESPACE, "button-layout"),
            Setting::Str(desktop.button_layout.to_string()),
        );
        if let Some(a) = &desktop.appearance {
            out.insert(
                (APPEARANCE_NAMESPACE, "color-scheme"),
                Setting::U32(a.color_scheme),
            );
            out.insert(
                (INTERFACE_NAMESPACE, "color-scheme"),
                Setting::Str(a.gnome_color_scheme.to_string()),
            );
            out.insert(
                (INTERFACE_NAMESPACE, "gtk-theme"),
                Setting::Str(a.gtk_theme.to_string()),
            );
        }
        if let Some(rgb) = desktop.accent_color {
            out.insert((APPEARANCE_NAMESPACE, "accent-color"), Setting::Rgb(rgb));
        }
        if let Some(contrast) = desktop.contrast {
            out.insert((APPEARANCE_NAMESPACE, "contrast"), Setting::U32(contrast));
        }
        out
    }

    /// Replace the served settings. Returns the keys whose value changed or appeared, and
    /// the keys no longer served (the other backends' value now applies).
    fn update(&mut self, desktop: &DesktopSettings) -> (Vec<(Key, Setting)>, Vec<Key>) {
        let desired = Self::desired(desktop);
        let changed = desired
            .iter()
            .filter(|(key, value)| self.settings.get(*key) != Some(*value))
            .map(|(key, value)| (*key, value.clone()))
            .collect();
        let removed = self
            .settings
            .keys()
            .filter(|key| !desired.contains_key(*key))
            .copied()
            .collect();
        self.settings = desired;
        (changed, removed)
    }

    fn read_all(&self, namespaces: &[String]) -> HashMap<String, HashMap<String, OwnedValue>> {
        let mut out: HashMap<String, HashMap<String, OwnedValue>> = HashMap::new();
        for ((namespace, key), value) in &self.settings {
            if namespace_matches(namespaces, namespace) {
                out.entry(namespace.to_string())
                    .or_default()
                    .insert(key.to_string(), value.to_owned_value());
            }
        }
        out
    }

    fn read(&self, namespace: &str, key: &str) -> Option<OwnedValue> {
        self.settings
            .iter()
            .find(|((ns, k), _)| *ns == namespace && *k == key)
            .map(|(_, value)| value.to_owned_value())
    }
}

/// Served settings; the session fills them in with [`update`] before the portal starts.
static STORE: LazyLock<Mutex<SettingsStore>> = LazyLock::new(Mutex::default);
static CONNECTION: OnceLock<Connection> = OnceLock::new();

fn store() -> std::sync::MutexGuard<'static, SettingsStore> {
    STORE.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Debug, zbus::DBusError)]
#[zbus(prefix = "org.freedesktop.portal.Error")]
enum PortalError {
    #[zbus(error)]
    ZBus(zbus::Error),
    NotFound(String),
}

struct SettingsPortal;

/// Whether `namespace` matches one of the `ReadAll` filters: an empty list or an empty
/// pattern matches everything, and a trailing `*` matches by prefix.
fn namespace_matches(patterns: &[String], namespace: &str) -> bool {
    patterns.is_empty()
        || patterns.iter().any(|p| match p.strip_suffix('*') {
            _ if p.is_empty() => true,
            Some(prefix) => namespace.starts_with(prefix),
            None => p == namespace,
        })
}

#[zbus::interface(name = "org.freedesktop.impl.portal.Settings")]
impl SettingsPortal {
    fn read_all(&self, namespaces: Vec<String>) -> HashMap<String, HashMap<String, OwnedValue>> {
        store().read_all(&namespaces)
    }

    fn read(&self, namespace: &str, key: &str) -> Result<OwnedValue, PortalError> {
        store().read(namespace, key).ok_or_else(|| {
            PortalError::NotFound(format!("Requested setting {namespace}.{key} not found"))
        })
    }

    #[zbus(property)]
    fn version(&self) -> u32 {
        2
    }
}

/// Claim the backend's bus name on the isolated session bus and start serving the
/// settings last passed to [`update`]. Must run before xdg-desktop-portal starts.
pub fn start(dbus_address: &str) -> Result<()> {
    if CONNECTION.get().is_some() {
        // A second connection would be dropped right away (releasing the bus name);
        // keep serving from the first one.
        warn!("settings portal backend already running; not starting another");
        return Ok(());
    }
    let conn = zbus::blocking::connection::Builder::address(dbus_address)
        .context("invalid isolated D-Bus address")?
        .name(BUS_NAME)?
        .serve_at(OBJECT_PATH, SettingsPortal)?
        .build()
        .context("failed to start the RemoteWayland settings portal backend")?;
    if CONNECTION.set(conn).is_err() {
        warn!("settings portal backend started concurrently; keeping the first connection");
    }
    info!(
        bus_name = BUS_NAME,
        "isolated settings portal backend started"
    );
    Ok(())
}

/// Serve `desktop` and notify xdg-desktop-portal (and through it, running apps) of every
/// key that changed, including keys no longer served (see [`value_now_in_effect`]).
pub fn update(desktop: &DesktopSettings) {
    // The store lock is released here: the frontend reads from us while we query it below.
    let (changed, removed) = store().update(desktop);
    let Some(conn) = CONNECTION.get() else {
        return;
    };
    let emit = |namespace: &str, key: &str, value: &Value<'_>| {
        if let Err(err) = conn.emit_signal(
            None::<()>,
            OBJECT_PATH,
            INTERFACE,
            "SettingChanged",
            &(namespace, key, value),
        ) {
            warn!(error = %err, namespace, key, "failed to emit settings portal SettingChanged");
        }
    };
    for ((namespace, key), value) in changed {
        emit(namespace, key, &value.to_value());
    }
    for (namespace, key) in removed {
        if let Some(value) = value_now_in_effect(conn, namespace, key) {
            emit(namespace, key, &value);
        }
    }
}

/// The value of a key we stopped serving, as apps now read it: the other backends' value
/// through the xdg-desktop-portal frontend, or for `accent-color`, "no accent" (the spec's
/// out-of-range triple) when no backend has one. Queried only while the frontend runs, so
/// the call cannot D-Bus-activate a frontend without the session's configuration.
fn value_now_in_effect(conn: &Connection, namespace: &str, key: &str) -> Option<Value<'static>> {
    const FRONTEND: &str = "org.freedesktop.portal.Desktop";
    let frontend_running = zbus::blocking::fdo::DBusProxy::new(conn)
        .ok()
        .and_then(|dbus| dbus.name_has_owner(FRONTEND.try_into().ok()?).ok())
        .unwrap_or(false);
    let from_frontend = frontend_running
        .then(|| {
            conn.call_method(
                Some(FRONTEND),
                OBJECT_PATH,
                Some("org.freedesktop.portal.Settings"),
                "ReadOne",
                &(namespace, key),
            )
            .ok()?
            .body()
            .deserialize::<OwnedValue>()
            .ok()
        })
        .flatten()
        .map(Value::from);
    from_frontend.or_else(|| {
        (namespace == APPEARANCE_NAMESPACE && key == "accent-color")
            .then(|| Value::from((-1.0f64, -1.0f64, -1.0f64)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_value(value: &str) -> OwnedValue {
        OwnedValue::from(Str::from(value))
    }

    const LAYOUT: &str = "appmenu:minimize,maximize,close";

    #[test]
    fn namespace_matches_supports_globs() {
        let p = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(namespace_matches(&[], WM_NAMESPACE));
        assert!(namespace_matches(&p(&[""]), WM_NAMESPACE));
        assert!(namespace_matches(&p(&["org.gnome.*"]), WM_NAMESPACE));
        assert!(namespace_matches(&p(&[WM_NAMESPACE]), WM_NAMESPACE));
        assert!(!namespace_matches(
            &p(&["org.freedesktop.appearance"]),
            WM_NAMESPACE
        ));
        assert!(!namespace_matches(&p(&["org.kde.*"]), WM_NAMESPACE));
    }

    #[test]
    fn serves_fonts_and_layout_before_a_color_scheme_syncs() {
        let mut store = SettingsStore::default();
        store.update(&DesktopSettings {
            appearance: None,
            button_layout: LAYOUT,
            accent_color: None,
            contrast: None,
        });
        let all = store.read_all(&[INTERFACE_NAMESPACE.to_string()]);
        assert_eq!(
            all[INTERFACE_NAMESPACE]["font-antialiasing"],
            str_value("grayscale")
        );
        assert!(!all[INTERFACE_NAMESPACE].contains_key("gtk-theme"));
        assert!(!all.contains_key(WM_NAMESPACE));
        assert_eq!(
            store.read(WM_NAMESPACE, "button-layout"),
            Some(str_value(LAYOUT))
        );
        assert_eq!(
            store.read(INTERFACE_NAMESPACE, "font-hinting"),
            Some(str_value("slight"))
        );
        assert_eq!(store.read(APPEARANCE_NAMESPACE, "color-scheme"), None);
        assert_eq!(store.read(APPEARANCE_NAMESPACE, "accent-color"), None);
        assert_eq!(store.read(INTERFACE_NAMESPACE, "font-name"), None);
    }

    #[test]
    fn serves_synced_accent_color_and_contrast() {
        let mut store = SettingsStore::default();
        let (changed, _) = store.update(&DesktopSettings {
            appearance: None,
            button_layout: LAYOUT,
            accent_color: Some([0.25, 0.5, 1.0]),
            contrast: Some(1),
        });
        assert!(changed.contains(&(
            (APPEARANCE_NAMESPACE, "accent-color"),
            Setting::Rgb([0.25, 0.5, 1.0])
        )));
        assert_eq!(
            store.read(APPEARANCE_NAMESPACE, "accent-color"),
            Some(OwnedValue::try_from(Value::from((0.25, 0.5, 1.0))).unwrap())
        );
        assert_eq!(
            store.read(APPEARANCE_NAMESPACE, "contrast"),
            Some(OwnedValue::from(1u32))
        );
        let all = store.read_all(&[APPEARANCE_NAMESPACE.to_string()]);
        assert_eq!(all[APPEARANCE_NAMESPACE].len(), 2);

        // Clearing the accent reports it as removed, so its new value gets announced.
        let (changed, removed) = store.update(&DesktopSettings {
            appearance: None,
            button_layout: LAYOUT,
            accent_color: None,
            contrast: Some(1),
        });
        assert!(changed.is_empty());
        assert_eq!(removed, vec![(APPEARANCE_NAMESPACE, "accent-color")]);
        assert_eq!(store.read(APPEARANCE_NAMESPACE, "accent-color"), None);
    }

    #[test]
    fn update_reports_only_changed_keys() {
        let mut store = SettingsStore::default();
        let dark = DesktopSettings {
            appearance: Some(Appearance {
                color_scheme: 1,
                gnome_color_scheme: "prefer-dark",
                gtk_theme: "Breeze-Dark",
            }),
            button_layout: LAYOUT,
            accent_color: None,
            contrast: None,
        };
        assert_eq!(store.update(&dark).0.len(), 7);
        assert_eq!(
            store.read(APPEARANCE_NAMESPACE, "color-scheme"),
            Some(OwnedValue::from(1u32))
        );
        assert!(store.update(&dark).0.is_empty());

        let light = DesktopSettings {
            appearance: Some(Appearance {
                color_scheme: 2,
                gnome_color_scheme: "prefer-light",
                gtk_theme: "Breeze",
            }),
            button_layout: LAYOUT,
            accent_color: None,
            contrast: None,
        };
        let changed: Vec<Key> = store.update(&light).0.into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            changed,
            vec![
                (APPEARANCE_NAMESPACE, "color-scheme"),
                (INTERFACE_NAMESPACE, "color-scheme"),
                (INTERFACE_NAMESPACE, "gtk-theme"),
            ]
        );
    }
}
