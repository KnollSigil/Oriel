//! A minimal `org.freedesktop.impl.portal.Settings` backend on the isolated session bus.
//!
//! It serves the `org.gnome.*` keys GTK apps need that the KDE portal does not, for
//! servers without xdg-desktop-portal-gtk. GTK apps that read settings through the portal
//! (GTK3 with `GTK_USE_PORTAL=1`, Flatpak apps such as Firefox and Chromium) otherwise:
//!
//! - fall back to GTK's built-in `menu:close` titlebar (close button only) without
//!   `org.gnome.desktop.wm.preferences` `button-layout`;
//! - draw UI text with no antialiasing at all (Chromium's tab titles, for one) without the
//!   `org.gnome.desktop.interface` font rendering keys ([`FONT_SETTINGS`]).
//!
//! It is listed last for the Settings interface, so a running xdg-desktop-portal-gtk
//! still takes precedence.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use tracing::{info, warn};
use zbus::blocking::Connection;
use zbus::zvariant::{OwnedValue, Value};

/// Portal id used in `portals.conf` and the `.portal` file name.
pub const PORTAL_ID: &str = "remotewayland";
const BUS_NAME: &str = "org.freedesktop.impl.portal.desktop.remotewayland";
const OBJECT_PATH: &str = "/org/freedesktop/portal/desktop";
const INTERFACE: &str = "org.freedesktop.impl.portal.Settings";
const WM_NAMESPACE: &str = "org.gnome.desktop.wm.preferences";
const BUTTON_LAYOUT_KEY: &str = "button-layout";
const INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";

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

static BUTTON_LAYOUT: Mutex<String> = Mutex::new(String::new());
static CONNECTION: OnceLock<Connection> = OnceLock::new();

#[derive(Debug, zbus::DBusError)]
#[zbus(prefix = "org.freedesktop.portal.Error")]
enum PortalError {
    #[zbus(error)]
    ZBus(zbus::Error),
    NotFound(String),
}

struct SettingsPortal;

fn button_layout_value() -> OwnedValue {
    let layout = BUTTON_LAYOUT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    str_value(&layout)
}

fn str_value(value: &str) -> OwnedValue {
    OwnedValue::from(zbus::zvariant::Str::from(value))
}

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
        let mut out = HashMap::new();
        if namespace_matches(&namespaces, WM_NAMESPACE) {
            out.insert(
                WM_NAMESPACE.to_string(),
                HashMap::from([(BUTTON_LAYOUT_KEY.to_string(), button_layout_value())]),
            );
        }
        if namespace_matches(&namespaces, INTERFACE_NAMESPACE) {
            out.insert(
                INTERFACE_NAMESPACE.to_string(),
                FONT_SETTINGS
                    .iter()
                    .map(|&(key, value)| (key.to_string(), str_value(value)))
                    .collect(),
            );
        }
        out
    }

    fn read(&self, namespace: &str, key: &str) -> Result<OwnedValue, PortalError> {
        let font_value = (namespace == INTERFACE_NAMESPACE)
            .then(|| FONT_SETTINGS.iter().find(|(k, _)| *k == key))
            .flatten();
        if namespace == WM_NAMESPACE && key == BUTTON_LAYOUT_KEY {
            Ok(button_layout_value())
        } else if let Some(&(_, value)) = font_value {
            Ok(str_value(value))
        } else {
            Err(PortalError::NotFound(format!(
                "Requested setting {namespace}.{key} not found"
            )))
        }
    }

    #[zbus(property)]
    fn version(&self) -> u32 {
        2
    }
}

/// Claim the backend's bus name on the isolated session bus and start serving
/// `button_layout`. Must run before xdg-desktop-portal starts.
pub fn start(dbus_address: &str, button_layout: &str) -> Result<()> {
    if CONNECTION.get().is_some() {
        // A second connection would be dropped right away (releasing the bus name);
        // keep serving from the first one.
        warn!("settings portal backend already running; not starting another");
        return Ok(());
    }
    *BUTTON_LAYOUT.lock().unwrap_or_else(|e| e.into_inner()) = button_layout.to_string();
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

/// Update the served `button-layout` and notify xdg-desktop-portal (and through it,
/// running apps) of the change.
pub fn set_button_layout(layout: &str) {
    {
        let mut current = BUTTON_LAYOUT.lock().unwrap_or_else(|e| e.into_inner());
        if *current == layout {
            return;
        }
        *current = layout.to_string();
    }
    let Some(conn) = CONNECTION.get() else {
        return;
    };
    if let Err(err) = conn.emit_signal(
        None::<()>,
        OBJECT_PATH,
        INTERFACE,
        "SettingChanged",
        &(WM_NAMESPACE, BUTTON_LAYOUT_KEY, Value::from(layout)),
    ) {
        warn!(error = %err, "failed to emit settings portal SettingChanged");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn serves_font_rendering_keys() {
        let all = SettingsPortal.read_all(vec![INTERFACE_NAMESPACE.to_string()]);
        assert_eq!(
            all[INTERFACE_NAMESPACE]["font-antialiasing"],
            str_value("grayscale")
        );
        assert!(!all.contains_key(WM_NAMESPACE));
        assert_eq!(
            SettingsPortal
                .read(INTERFACE_NAMESPACE, "font-hinting")
                .unwrap(),
            str_value("slight")
        );
        assert!(SettingsPortal
            .read(INTERFACE_NAMESPACE, "font-name")
            .is_err());
    }
}
