//! XDG `.desktop` application and icon discovery for the Oriel launcher.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use oriel_proto::RemoteAppEntry;

const MAX_ICON_BYTES: usize = 256 * 1024;

/// Oriel's desktop name: first entry of apps' `XDG_CURRENT_DESKTOP`, and the name matched
/// against `.desktop` files' `OnlyShowIn=`/`NotShowIn=` lists (ignoring case).
pub const DESKTOP_NAME: &str = "ORIEL";

/// Scan standard XDG `.desktop` directories and return sorted `RemoteAppEntry` items.
pub fn discover_apps() -> Vec<RemoteAppEntry> {
    // Read every `.desktop` file first, so the icon index only covers the names they use.
    let mut files: Vec<(String, String)> = Vec::new();
    for dir in xdg_application_dirs() {
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
            if let Ok(content) = fs::read_to_string(&path) {
                files.push((file_name.to_string(), content));
            }
        }
    }

    let wanted: HashSet<String> = files
        .iter()
        .flat_map(|(_, content)| content.lines())
        .filter_map(|line| line.trim().strip_prefix("Icon="))
        .map(|icon| icon_name(icon.trim()).to_string())
        .collect();
    let icon_index = IconIndex::build(&wanted);

    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut apps: Vec<RemoteAppEntry> = Vec::new();
    for (id, content) in files {
        if seen_ids.contains(&id) {
            continue;
        }
        if let Some(app) = parse_desktop_content(&content, &id, &icon_index) {
            seen_ids.insert(id);
            apps.push(app);
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

/// Icon themes searched, in priority order. `hicolor` comes first because that is where
/// applications install their own icons; the others supply generic names like `printer`.
const ICON_THEMES: [&str; 4] = ["hicolor", "Adwaita", "breeze", "AdwaitaLegacy"];

/// Launcher icon size the lookup aims for, in pixels.
const TARGET_ICON_SIZE: u32 = 64;

/// Index of the non-symbolic PNG/SVG icons in [`ICON_THEMES`] (all sizes and contexts) and
/// in `pixmaps`, across the XDG data dirs, keyed by icon name.
#[derive(Default)]
pub(crate) struct IconIndex {
    /// Candidates per name, best first.
    by_name: HashMap<String, Vec<PathBuf>>,
}

impl IconIndex {
    fn build(wanted: &HashSet<String>) -> Self {
        // (theme rank, size penalty, path) per name; theme rank `ICON_THEMES.len()` is pixmaps.
        let mut found: HashMap<String, Vec<(usize, u32, PathBuf)>> = HashMap::new();
        for base in xdg_data_dirs() {
            for (rank, theme) in ICON_THEMES.iter().enumerate() {
                let root = base.join("icons").join(theme);
                let mut files = Vec::new();
                collect_icon_files(&root, 3, wanted, &mut files);
                for path in files {
                    let rel = path.strip_prefix(&root).unwrap_or(&path);
                    if let Some((name, penalty)) = classify_icon(rel) {
                        found.entry(name).or_default().push((rank, penalty, path));
                    }
                }
            }
            let mut files = Vec::new();
            collect_icon_files(&base.join("pixmaps"), 0, wanted, &mut files);
            for path in files {
                if let Some(name) = icon_stem(&path) {
                    found
                        .entry(name)
                        .or_default()
                        .push((ICON_THEMES.len(), 0, path));
                }
            }
        }
        let by_name = found
            .into_iter()
            .map(|(name, mut cands)| {
                // Stable sort keeps XDG data dir order among equal candidates.
                cands.sort_by_key(|(rank, penalty, _)| (*rank, *penalty));
                (name, cands.into_iter().map(|(_, _, p)| p).collect())
            })
            .collect();
        Self { by_name }
    }

    fn candidates(&self, name: &str) -> &[PathBuf] {
        self.by_name.get(name).map(Vec::as_slice).unwrap_or(&[])
    }
}

fn xdg_data_dirs() -> Vec<PathBuf> {
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
    base_dirs
}

/// Collect `.png`/`.svg` files named in `wanted` under `dir`, descending at most `depth`
/// levels (symlinks followed; the depth bound keeps link cycles finite).
fn collect_icon_files(dir: &Path, depth: u32, wanted: &HashSet<String>, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        // Filter by name before any stat: themes like breeze hold tens of thousands of files.
        let is_wanted_icon = || icon_stem(&path).is_some_and(|stem| wanted.contains(&stem));
        if file_type.is_dir() {
            if depth > 0 {
                collect_icon_files(&path, depth - 1, wanted, out);
            }
        } else if file_type.is_file() {
            if is_wanted_icon() {
                out.push(path);
            }
        } else if file_type.is_symlink() {
            let Ok(meta) = fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                if depth > 0 {
                    collect_icon_files(&path, depth - 1, wanted, out);
                }
            } else if meta.is_file() && is_wanted_icon() {
                out.push(path);
            }
        }
    }
}

/// Icon name of a `.png`/`.svg` file, or `None` for other files and symbolic icons.
fn icon_stem(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?;
    if ext != "png" && ext != "svg" {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    if stem.ends_with("-symbolic") {
        return None;
    }
    Some(stem.to_string())
}

/// Name and size penalty (lower is better) of an icon at `rel` inside a theme directory,
/// whose size comes from a `48x48`, `48x48@2`, `48`, `48@2x` or `scalable` component.
/// Symbolic icons are skipped.
fn classify_icon(rel: &Path) -> Option<(String, u32)> {
    let name = icon_stem(rel)?;
    let mut size = None;
    for comp in rel.parent()?.components() {
        let comp = comp.as_os_str().to_str()?;
        if comp.contains("symbolic") {
            return None;
        }
        if size.is_none() {
            size = parse_size_dir(comp);
        }
    }
    let penalty = match size? {
        // Scalable renders crisply at any size: rank it just after an exact 64-128 px PNG.
        IconSize::Scalable => 32,
        IconSize::Pixels(px) if px >= TARGET_ICON_SIZE => px - TARGET_ICON_SIZE,
        // Too small: upscaling blurs, so only use it when nothing larger exists.
        IconSize::Pixels(px) => 10_000 + (TARGET_ICON_SIZE - px),
    };
    Some((name, penalty))
}

#[derive(Debug, PartialEq)]
enum IconSize {
    Scalable,
    Pixels(u32),
}

fn parse_size_dir(comp: &str) -> Option<IconSize> {
    if comp == "scalable" {
        return Some(IconSize::Scalable);
    }
    let (dims, scale) = match comp.split_once('@') {
        Some((d, s)) => (d, s.trim_end_matches('x').parse::<u32>().ok()?),
        None => (comp, 1),
    };
    let px = match dims.split_once('x') {
        Some((w, h)) if w == h => w.parse::<u32>().ok()?,
        Some(_) => return None,
        None => dims.parse::<u32>().ok()?,
    };
    Some(IconSize::Pixels(px.saturating_mul(scale)))
}

fn parse_semicolon_list(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn parse_desktop_content(
    content: &str,
    id: &str,
    icon_index: &IconIndex,
) -> Option<RemoteAppEntry> {
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
    // `OnlyShowIn=` restricts an entry to the listed desktops (an empty list restricts
    // nothing); `NotShowIn=` hides it on the listed ones.
    let lists_us = |key: &str| {
        fields.get(key).map(|s| {
            let names = parse_semicolon_list(s);
            let ours = names.iter().any(|d| d.eq_ignore_ascii_case(DESKTOP_NAME));
            (names.is_empty(), ours)
        })
    };
    if let Some((false, false)) = lists_us("OnlyShowIn") {
        return None;
    }
    if let Some((_, true)) = lists_us("NotShowIn") {
        return None;
    }

    let terminal = fields.get("Terminal").map(|s| s.as_str()) == Some("true");

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
        .and_then(|icon| load_icon_bytes(icon, icon_index))
        .map(|(mime, data)| (Some(mime), data))
        .unwrap_or((None, Vec::new()));

    Some(RemoteAppEntry {
        id: id.to_string(),
        name,
        generic_name,
        keywords,
        exec,
        categories,
        terminal,
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

/// Icon name from an `Icon=` value, minus the extension some entries include (`foo.png`).
fn icon_name(icon: &str) -> &str {
    icon.strip_suffix(".png")
        .or_else(|| icon.strip_suffix(".svg"))
        .unwrap_or(icon)
}

fn load_icon_bytes(icon: &str, icon_index: &IconIndex) -> Option<(String, Vec<u8>)> {
    let icon_path = Path::new(icon);
    if icon_path.is_absolute() {
        return icon_path
            .is_file()
            .then(|| read_icon_file(icon_path))
            .flatten();
    }
    icon_index
        .candidates(icon_name(icon))
        .iter()
        .find_map(|p| read_icon_file(p))
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
        let app = parse_desktop_content(content, "firefox.desktop", &IconIndex::default()).unwrap();
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
        let app = parse_desktop_content(minimal, "xterm.desktop", &IconIndex::default()).unwrap();
        assert_eq!(app.generic_name, None);
        assert!(app.keywords.is_empty());
        assert_eq!(app.exec, "xterm");
        assert!(!app.terminal);

        let tui = "[Desktop Entry]\nType=Application\nName=htop\nExec=htop\nTerminal=true\n";
        assert!(
            parse_desktop_content(tui, "htop.desktop", &IconIndex::default())
                .unwrap()
                .terminal
        );

        let hidden = "[Desktop Entry]\nType=Application\nName=Hidden\nExec=true\nNoDisplay=true\n";
        assert!(parse_desktop_content(hidden, "hidden.desktop", &IconIndex::default()).is_none());
    }

    #[test]
    fn parse_desktop_content_honors_only_show_in_and_not_show_in() {
        let shown = |line: &str| {
            let content =
                format!("[Desktop Entry]\nType=Application\nName=App\nExec=app\n{line}\n");
            parse_desktop_content(&content, "a.desktop", &IconIndex::default()).is_some()
        };
        assert!(!shown("OnlyShowIn=KDE;"));
        assert!(shown("OnlyShowIn=GNOME;ORIEL;"));
        assert!(shown("OnlyShowIn=Oriel;"));
        assert!(shown("OnlyShowIn="));
        assert!(!shown("NotShowIn=KDE;ORIEL;"));
        assert!(shown("NotShowIn=KDE;"));
        assert!(shown("NotShowIn="));
    }
}

#[cfg(test)]
mod icon_tests {
    use super::*;

    #[test]
    fn parse_size_dir_handles_theme_layouts() {
        assert_eq!(parse_size_dir("scalable"), Some(IconSize::Scalable));
        assert_eq!(parse_size_dir("48x48"), Some(IconSize::Pixels(48)));
        assert_eq!(parse_size_dir("48x48@2"), Some(IconSize::Pixels(96)));
        assert_eq!(parse_size_dir("64"), Some(IconSize::Pixels(64)));
        assert_eq!(parse_size_dir("32@2x"), Some(IconSize::Pixels(64)));
        assert_eq!(parse_size_dir("apps"), None);
    }

    #[test]
    fn classify_icon_ranks_sizes_and_skips_symbolic() {
        let p = |s: &str| classify_icon(Path::new(s));
        let exact = p("64x64/apps/a.png").unwrap().1;
        let scalable = p("scalable/apps/a.svg").unwrap().1;
        let large = p("512x512/apps/a.png").unwrap().1;
        let small = p("16x16/apps/a.png").unwrap().1;
        assert!(exact < scalable && scalable < large && large < small);
        // Context directories other than `apps`, and breeze's context/size layout.
        assert_eq!(p("devices/64/printer.svg"), Some(("printer".into(), 0)));
        assert!(p("symbolic/apps/a.svg").is_none());
        assert!(p("scalable/apps/a-symbolic.svg").is_none());
        assert!(p("apps/a.png").is_none());
    }
}
