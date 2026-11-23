//! XDG `.desktop` application and icon discovery for the RemoteWayland launcher.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use remotewayland_proto::RemoteAppEntry;

const MAX_ICON_BYTES: usize = 256 * 1024;

/// Scan standard XDG `.desktop` directories and return sorted `RemoteAppEntry` items.
pub fn discover_apps() -> Vec<RemoteAppEntry> {
    let app_dirs = xdg_application_dirs();
    let icon_dirs = xdg_icon_dirs();

    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut apps: Vec<RemoteAppEntry> = Vec::new();

    for dir in app_dirs {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("desktop") {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let id = file_name.to_string();
            if seen_ids.contains(&id) {
                continue;
            }
            if let Some(app) = parse_desktop_file(&path, &id, &icon_dirs) {
                seen_ids.insert(id);
                apps.push(app);
            }
        }
    }

    apps.sort_by_key(|a| a.name.to_lowercase());
    apps
}

fn xdg_application_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(data_home) = std::env::var_os("XDG_DATA_HOME") {
        dirs.push(PathBuf::from(data_home).join("applications"));
    } else if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".local/share/applications"));
    }
    if let Some(data_dirs) = std::env::var_os("XDG_DATA_DIRS") {
        for part in std::env::split_paths(&data_dirs) {
            dirs.push(part.join("applications"));
        }
    } else {
        dirs.push(PathBuf::from("/usr/local/share/applications"));
        dirs.push(PathBuf::from("/usr/share/applications"));
    }
    dirs
}

fn xdg_icon_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut base_dirs = Vec::new();
    if let Some(data_home) = std::env::var_os("XDG_DATA_HOME") {
        base_dirs.push(PathBuf::from(data_home));
    } else if let Some(home) = std::env::var_os("HOME") {
        base_dirs.push(PathBuf::from(home).join(".local/share"));
    }
    if let Some(data_dirs) = std::env::var_os("XDG_DATA_DIRS") {
        base_dirs.extend(std::env::split_paths(&data_dirs));
    } else {
        base_dirs.push(PathBuf::from("/usr/local/share"));
        base_dirs.push(PathBuf::from("/usr/share"));
    }

    for base in &base_dirs {
        for sub in [
            "icons/hicolor/48x48/apps",
            "icons/hicolor/64x64/apps",
            "icons/hicolor/128x128/apps",
            "icons/hicolor/scalable/apps",
            "icons/breeze/apps/48",
            "icons/breeze/apps/64",
            "icons/Adwaita/48x48/apps",
            "icons/Adwaita/scalable/apps",
            "pixmaps",
        ] {
            let p = base.join(sub);
            if p.is_dir() {
                dirs.push(p);
            }
        }
    }
    dirs
}

fn parse_desktop_file(path: &Path, id: &str, icon_dirs: &[PathBuf]) -> Option<RemoteAppEntry> {
    let content = fs::read_to_string(path).ok()?;
    parse_desktop_content(&content, id, icon_dirs)
}

fn parse_semicolon_list(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn parse_desktop_content(content: &str, id: &str, icon_dirs: &[PathBuf]) -> Option<RemoteAppEntry> {
    let mut in_desktop_entry = false;
    let mut fields: HashMap<String, String> = HashMap::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_desktop_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_desktop_entry {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            fields
                .entry(k.trim().to_string())
                .or_insert_with(|| v.trim().to_string());
        }
    }

    if fields.get("Type").map(|s| s.as_str()) != Some("Application") {
        return None;
    }
    if fields.get("NoDisplay").map(|s| s.as_str()) == Some("true")
        || fields.get("Hidden").map(|s| s.as_str()) == Some("true")
    {
        return None;
    }
    if fields.get("Terminal").map(|s| s.as_str()) == Some("true") {
        // Keep terminal apps out or include them? They still launch fine inside a terminal or skipped;
        // let's include them if they have Name and Exec.
    }

    let name = fields.get("Name")?.trim().to_string();
    let raw_exec = fields.get("Exec")?.trim();
    let exec = strip_exec_field_codes(raw_exec);
    if name.is_empty() || exec.is_empty() {
        return None;
    }

    let generic_name = fields
        .get("GenericName")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let keywords = fields
        .get("Keywords")
        .map(|s| parse_semicolon_list(s))
        .unwrap_or_default();

    let categories = fields
        .get("Categories")
        .map(|s| parse_semicolon_list(s))
        .unwrap_or_default();

    let (icon_mime, icon_data) = fields
        .get("Icon")
        .and_then(|icon| load_icon_bytes(icon, icon_dirs))
        .map(|(mime, data)| (Some(mime), data))
        .unwrap_or((None, Vec::new()));

    Some(RemoteAppEntry {
        id: id.to_string(),
        name,
        generic_name,
        keywords,
        exec,
        categories,
        icon_mime,
        icon_data,
    })
}

fn strip_exec_field_codes(raw: &str) -> String {
    let mut out = Vec::new();
    for tok in raw.split_whitespace() {
        if matches!(
            tok,
            "%f" | "%F"
                | "%u"
                | "%U"
                | "%d"
                | "%D"
                | "%n"
                | "%N"
                | "%i"
                | "%c"
                | "%k"
                | "%v"
                | "%m"
        ) {
            continue;
        }
        out.push(tok);
    }
    out.join(" ")
}

fn load_icon_bytes(icon: &str, icon_dirs: &[PathBuf]) -> Option<(String, Vec<u8>)> {
    let icon_path = Path::new(icon);
    if icon_path.is_absolute() && icon_path.is_file() {
        return read_icon_file(icon_path);
    }

    for dir in icon_dirs {
        for ext in ["png", "svg"] {
            let candidate = dir.join(format!("{icon}.{ext}"));
            if candidate.is_file() {
                if let Some(res) = read_icon_file(&candidate) {
                    return Some(res);
                }
            }
        }
    }
    None
}

fn read_icon_file(path: &Path) -> Option<(String, Vec<u8>)> {
    let meta = fs::metadata(path).ok()?;
    if meta.len() as usize > MAX_ICON_BYTES {
        return None;
    }
    let data = fs::read(path).ok()?;
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("svg") | Some("svgz") => "image/svg+xml",
        Some("png") => "image/png",
        _ => return None,
    };
    Some((mime.to_string(), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_desktop_content_extracts_generic_name_keywords_categories_and_exec() {
        let content = "\
            [Desktop Entry]
            Type=Application
            Name=Firefox
            GenericName=Web Browser
            Keywords=internet; www ; ;web;
            Categories=Network;WebBrowser;
            Exec=/usr/bin/firefox %u
        ";
        let app = parse_desktop_content(content, "firefox.desktop", &[]).unwrap();
        assert_eq!(app.id, "firefox.desktop");
        assert_eq!(app.name, "Firefox");
        assert_eq!(app.generic_name.as_deref(), Some("Web Browser"));
        assert_eq!(app.keywords, vec!["internet", "www", "web"]);
        assert_eq!(app.categories, vec!["Network", "WebBrowser"]);
        assert_eq!(app.exec, "/usr/bin/firefox");
    }

    #[test]
    fn parse_desktop_content_handles_missing_optional_fields_and_hidden() {
        let minimal = "[Desktop Entry]\nType=Application\nName=XTerm\nExec=xterm %f\n";
        let app = parse_desktop_content(minimal, "xterm.desktop", &[]).unwrap();
        assert_eq!(app.generic_name, None);
        assert!(app.keywords.is_empty());
        assert_eq!(app.exec, "xterm");

        let hidden = "[Desktop Entry]\nType=Application\nName=Hidden\nExec=true\nNoDisplay=true\n";
        assert!(parse_desktop_content(hidden, "hidden.desktop", &[]).is_none());
    }
}
