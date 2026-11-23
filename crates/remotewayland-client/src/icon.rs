//! Remote application icon matching and decoding for Wayland `xdg-toplevel-icon-v1`.

use std::path::Path;

/// Maximum allowed raw icon payload size (4 MiB) before decoding.
const MAX_ICON_BYTES: usize = 4 * 1024 * 1024;

/// Lightweight app metadata + raw icon bytes (`image/png` or `image/svg+xml`)
/// received from the remote server's `.desktop` catalog (`ControlMessage::AppList`).
#[derive(Debug, Clone)]
pub struct AppIconEntry {
    pub id: String,
    pub name: String,
    pub exec: String,
    pub icon_data: Vec<u8>,
}

fn normalize_desktop_id(id: &str) -> String {
    let trimmed = id.trim();
    trimmed
        .strip_suffix(".desktop")
        .unwrap_or(trimmed)
        .to_lowercase()
}

fn last_rdns_segment(id: &str) -> &str {
    id.rsplit('.').next().unwrap_or(id)
}

fn strip_matching_quotes(tok: &str) -> &str {
    tok.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| tok.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(tok)
}

fn exec_basename(exec: &str) -> Option<String> {
    let mut tokens = exec.split_whitespace().peekable();
    while let Some(&raw_tok) = tokens.peek() {
        let tok = strip_matching_quotes(raw_tok);
        let base = Path::new(tok)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(tok);
        if base == "env" {
            tokens.next();
            while let Some(&next) = tokens.peek() {
                let unquoted = strip_matching_quotes(next);
                if unquoted.starts_with('-') || unquoted.contains('=') {
                    tokens.next();
                } else {
                    break;
                }
            }
            continue;
        }
        if base == "flatpak" {
            tokens.next();
            if tokens.peek().map(|t| strip_matching_quotes(t)) == Some("run") {
                tokens.next();
                if let Some(app_id) = tokens
                    .map(strip_matching_quotes)
                    .find(|t| !t.starts_with('-'))
                {
                    return Some(app_id.to_lowercase());
                }
            }
            return None;
        }
        if tok.contains('=') && !tok.starts_with('/') {
            tokens.next();
            continue;
        }
        return Some(base.to_lowercase());
    }
    None
}

/// Find the best matching [`AppIconEntry`] (with non-empty `icon_data`) for a Wayland `app_id`.
pub fn match_app_icon<'a>(app_id: &str, catalog: &'a [AppIconEntry]) -> Option<&'a AppIconEntry> {
    let target = normalize_desktop_id(app_id);
    if target.is_empty() {
        return None;
    }
    let target_seg = last_rdns_segment(&target);

    // Tier 1: exact desktop ID match (e.g. `org.kde.dolphin` == `org.kde.dolphin.desktop`)
    if let Some(entry) = catalog
        .iter()
        .filter(|e| !e.icon_data.is_empty())
        .find(|e| normalize_desktop_id(&e.id) == target)
    {
        return Some(entry);
    }

    // Tier 2: reverse-DNS last-segment match (e.g. `dolphin` <-> `org.kde.dolphin.desktop`
    // or `org.mozilla.firefox` <-> `firefox.desktop`)
    if let Some(entry) = catalog.iter().filter(|e| !e.icon_data.is_empty()).find(|e| {
        let eid = normalize_desktop_id(&e.id);
        let eseg = last_rdns_segment(&eid);
        eseg == target || eid == target_seg || eseg == target_seg
    }) {
        return Some(entry);
    }

    // Tier 3: exec binary basename match (e.g. `code` <-> `/usr/share/code/code`)
    if let Some(entry) = catalog.iter().filter(|e| !e.icon_data.is_empty()).find(|e| {
        exec_basename(&e.exec).is_some_and(|bin| {
            let bseg = last_rdns_segment(&bin);
            bin == target || bin == target_seg || bseg == target_seg
        })
    }) {
        return Some(entry);
    }

    // Tier 4: exact display name match (case-insensitive)
    catalog.iter().filter(|e| !e.icon_data.is_empty()).find(|e| {
        let name = e.name.trim().to_lowercase();
        !name.is_empty() && (name == target || name == target_seg)
    })
}

/// Decode raw PNG or SVG icon bytes into a square `target_size x target_size`
/// premultiplied `ARGB8888` pixel buffer (`wl_shm::Format::Argb8888`).
pub fn decode_icon_argb8888(data: &[u8], target_size: u32) -> Option<Vec<u8>> {
    if data.is_empty() || data.len() > MAX_ICON_BYTES || target_size == 0 || target_size > 512 {
        return None;
    }
    decode_icon_argb8888_impl(data, target_size)
}

#[cfg(feature = "gtk4")]
fn decode_icon_argb8888_impl(data: &[u8], target_size: u32) -> Option<Vec<u8>> {
    use gtk4::gdk_pixbuf::{prelude::PixbufLoaderExt, InterpType, PixbufLoader};

    let dim = target_size as i32;
    let loader = PixbufLoader::new();
    // Preserve aspect ratio during decode (especially for SVG rasterization).
    loader.connect_size_prepared(move |loader, orig_w, orig_h| {
        if orig_w > 0 && orig_h > 0 {
            let scale = (dim as f64 / orig_w as f64).min(dim as f64 / orig_h as f64);
            let new_w = ((orig_w as f64 * scale).round() as i32).clamp(1, dim);
            let new_h = ((orig_h as f64 * scale).round() as i32).clamp(1, dim);
            loader.set_size(new_w, new_h);
        }
    });
    if loader.write(data).is_err() {
        let _ = loader.close();
        return None;
    }
    loader.close().ok()?;
    let pixbuf = loader.pixbuf()?;

    let orig_w = pixbuf.width();
    let orig_h = pixbuf.height();
    if orig_w <= 0 || orig_h <= 0 || pixbuf.bits_per_sample() != 8 {
        return None;
    }
    let scaled = if orig_w > dim || orig_h > dim || (orig_w != dim && orig_h != dim) {
        let scale = (dim as f64 / orig_w as f64).min(dim as f64 / orig_h as f64);
        let new_w = ((orig_w as f64 * scale).round() as i32).clamp(1, dim);
        let new_h = ((orig_h as f64 * scale).round() as i32).clamp(1, dim);
        pixbuf.scale_simple(new_w, new_h, InterpType::Bilinear)?
    } else {
        pixbuf
    };

    let w = scaled.width() as usize;
    let h = scaled.height() as usize;
    let channels = scaled.n_channels() as usize;
    if channels != 3 && channels != 4 {
        return None;
    }
    let rowstride = scaled.rowstride() as usize;
    let pixels = unsafe { scaled.pixels() };
    rgba_to_centered_argb8888(pixels, w, h, channels, rowstride, target_size as usize)
}

#[cfg(not(feature = "gtk4"))]
fn decode_icon_argb8888_impl(_data: &[u8], _target_size: u32) -> Option<Vec<u8>> {
    None
}

#[cfg(any(feature = "gtk4", test))]
fn rgba_to_centered_argb8888(
    src: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    rowstride: usize,
    target_size: usize,
) -> Option<Vec<u8>> {
    if (channels != 3 && channels != 4) || target_size == 0 {
        return None;
    }
    let rows = h.min(target_size);
    let cols = w.min(target_size);
    let row_bytes = cols.checked_mul(channels)?;
    if rowstride < row_bytes {
        return None;
    }
    if rows > 0 {
        let min_src_len = (rows - 1).checked_mul(rowstride)?.checked_add(row_bytes)?;
        if src.len() < min_src_len {
            return None;
        }
    }
    let mut out = vec![0u8; target_size.checked_mul(target_size)?.checked_mul(4)?];
    let x_off = (target_size.saturating_sub(cols)) / 2;
    let y_off = (target_size.saturating_sub(rows)) / 2;
    for y in 0..rows {
        let src_row = &src[y * rowstride..y * rowstride + row_bytes];
        let dst_row_start = ((y + y_off) * target_size + x_off) * 4;
        for x in 0..cols {
            let si = x * channels;
            let r = src_row[si] as u16;
            let g = src_row[si + 1] as u16;
            let b = src_row[si + 2] as u16;
            let a = if channels == 4 {
                src_row[si + 3] as u16
            } else {
                255u16
            };
            let pr = ((r * a + 127) / 255) as u8;
            let pg = ((g * a + 127) / 255) as u8;
            let pb = ((b * a + 127) / 255) as u8;
            let di = dst_row_start + x * 4;
            // Native-endian ARGB8888 on little-endian is [B, G, R, A].
            let px = ((a as u32) << 24) | ((pr as u32) << 16) | ((pg as u32) << 8) | (pb as u32);
            out[di..di + 4].copy_from_slice(&px.to_ne_bytes());
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, name: &str, exec: &str, has_icon: bool) -> AppIconEntry {
        AppIconEntry {
            id: id.to_string(),
            name: name.to_string(),
            exec: exec.to_string(),
            icon_data: if has_icon { b"icon".to_vec() } else { Vec::new() },
        }
    }

    #[test]
    fn match_app_icon_handles_exact_rdns_exec_and_name() {
        let catalog = vec![
            entry("org.kde.dolphin.desktop", "Dolphin", "dolphin", true),
            entry("firefox.desktop", "Firefox", "/usr/lib/firefox/firefox", true),
            entry("code.desktop", "Visual Studio Code", "env FOO=1 \"/usr/share/code/code\" %F", true),
            entry("fp-app.desktop", "Flatpak App", "flatpak run --branch=stable org.example.Tool", true),
            entry("custom.desktop", "MyCustomViewer", "/opt/launcher/run-it", true),
            entry("empty.desktop", "Empty", "empty", false),
        ];

        assert_eq!(
            match_app_icon("org.kde.dolphin", &catalog).map(|e| e.id.as_str()),
            Some("org.kde.dolphin.desktop")
        );
        assert_eq!(
            match_app_icon("dolphin", &catalog).map(|e| e.id.as_str()),
            Some("org.kde.dolphin.desktop")
        );
        assert_eq!(
            match_app_icon("org.mozilla.firefox", &catalog).map(|e| e.id.as_str()),
            Some("firefox.desktop")
        );
        assert_eq!(
            match_app_icon("Code", &catalog).map(|e| e.id.as_str()),
            Some("code.desktop")
        );
        assert_eq!(
            match_app_icon("org.example.Tool", &catalog).map(|e| e.id.as_str()),
            Some("fp-app.desktop")
        );
        assert_eq!(
            match_app_icon("mycustomviewer", &catalog).map(|e| e.id.as_str()),
            Some("custom.desktop")
        );
        assert_eq!(match_app_icon("empty", &catalog).map(|e| e.id.as_str()), None);
        assert_eq!(match_app_icon("unknown-app", &catalog).map(|e| e.id.as_str()), None);
    }

    #[test]
    fn rgba_to_centered_argb8888_premultiplies_centers_and_handles_rgb() {
        // 2x1 RGBA: opaque red, 50%-alpha green
        let src = [255, 0, 0, 255, 0, 255, 0, 128];
        let out = rgba_to_centered_argb8888(&src, 2, 1, 4, 8, 4).unwrap();
        assert_eq!(out.len(), 4 * 4 * 4);
        // Centered: x_off = 1, y_off = 1
        let px0 = u32::from_ne_bytes(out[(4 + 1) * 4..(4 + 2) * 4].try_into().unwrap());
        assert_eq!(px0, 0xFFFF_0000);
        let px1 = u32::from_ne_bytes(out[(4 + 2) * 4..(4 + 3) * 4].try_into().unwrap());
        assert_eq!(px1, 0x8000_8000);
        // Padding pixel at (0, 0) is transparent
        let pad = u32::from_ne_bytes(out[0..4].try_into().unwrap());
        assert_eq!(pad, 0);

        // 1x1 3-channel RGB: blue
        let rgb = [0, 0, 255];
        let out_rgb = rgba_to_centered_argb8888(&rgb, 1, 1, 3, 3, 2).unwrap();
        let px_rgb = u32::from_ne_bytes(out_rgb[0..4].try_into().unwrap());
        assert_eq!(px_rgb, 0xFF00_00FF);

        // Truncated source buffer rejected safely
        assert!(rgba_to_centered_argb8888(&[255, 0], 2, 1, 4, 8, 4).is_none());
    }

    #[cfg(feature = "gtk4")]
    #[test]
    fn decode_icon_argb8888_decodes_png_and_preserves_aspect_ratio() {
        // Minimal valid 1x1 opaque red (#FF0000) PNG
        const RED_1X1_PNG: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let argb = decode_icon_argb8888(RED_1X1_PNG, 16).expect("PNG icon decode failed");
        assert_eq!(argb.len(), 16 * 16 * 4);
        let center =
            u32::from_ne_bytes(argb[(8 * 16 + 8) * 4..(8 * 16 + 9) * 4].try_into().unwrap());
        assert_eq!(center, 0xFFFF_0000);
        assert!(decode_icon_argb8888(b"not-an-image", 16).is_none());

        // Minimal valid 2x1 opaque red (#FF0000) PNG -> must scale to 16x8 centered vertically
        // so row 0 is transparent padding and row 8 is opaque red.
        const RED_2X1_PNG: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x7B, 0x40, 0xE8, 0xDD, 0x00, 0x00, 0x00, 0x0F, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x9F, 0x81, 0x01, 0x00, 0x08, 0xFE, 0x01, 0xFF,
            0x48, 0xC5, 0x66, 0x90, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
            0x60, 0x82,
        ];
        let wide = decode_icon_argb8888(RED_2X1_PNG, 16).expect("2x1 PNG decode failed");
        let top_pad = u32::from_ne_bytes(wide[0..4].try_into().unwrap());
        assert_eq!(top_pad, 0, "non-square icon should have transparent top padding");
        let mid_px =
            u32::from_ne_bytes(wide[(8 * 16 + 8) * 4..(8 * 16 + 9) * 4].try_into().unwrap());
        assert_eq!(mid_px, 0xFFFF_0000);

        // Oversized input rejected before decoding
        let oversized = vec![0u8; MAX_ICON_BYTES + 1];
        assert!(decode_icon_argb8888(&oversized, 16).is_none());
    }
}
