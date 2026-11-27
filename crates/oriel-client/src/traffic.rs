//! Bytes received per remote window and per app audio stream, so the launcher can show
//! which windows use the connection.
//!
//! A window's traffic is its video stream plus the frames of its popups (menus,
//! tooltips), which are counted for the toplevel they belong to.

use std::collections::HashMap;
use std::sync::Mutex;

use oriel_proto::WindowId;

/// What received bytes are counted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Window(WindowId),
    /// A per-app audio playback stream, by stream id.
    Audio(u32),
}

/// The bytes one source has received so far, and what to call it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTotal {
    pub source: Source,
    /// The window title or the audio stream's app name.
    pub name: String,
    /// The window's app id, or what the audio stream plays.
    pub detail: String,
    pub bytes: u64,
}

#[derive(Default)]
struct State {
    /// Toplevel windows: title and app id, as last announced.
    windows: HashMap<WindowId, (String, String)>,
    /// Popup -> the window it belongs to (another popup or a toplevel).
    popup_parent: HashMap<WindowId, WindowId>,
    /// Audio stream -> app name and media name.
    audio: HashMap<u32, (String, String)>,
    bytes: HashMap<Source, u64>,
}

impl State {
    /// The toplevel a window or popup belongs to.
    fn toplevel(&self, mut id: WindowId) -> WindowId {
        // Bounded: a malformed parent cycle must not hang the receive loop.
        for _ in 0..32 {
            match self.popup_parent.get(&id) {
                Some(&parent) => id = parent,
                None => break,
            }
        }
        id
    }
}

/// Per-connection byte counts, fed by the receive loops and read once a second.
#[derive(Default)]
pub struct TrafficMeter {
    state: Mutex<State>,
}

impl TrafficMeter {
    /// A toplevel was announced, or its title or app id changed.
    pub fn window_named(&self, window_id: WindowId, title: &str, app_id: &str) {
        let mut st = self.state.lock().unwrap();
        let entry = st.windows.entry(window_id).or_default();
        if !title.is_empty() {
            entry.0 = title.to_string();
        }
        if !app_id.is_empty() {
            entry.1 = app_id.to_string();
        }
    }

    pub fn popup_created(&self, popup: WindowId, parent: WindowId) {
        self.state
            .lock()
            .unwrap()
            .popup_parent
            .insert(popup, parent);
    }

    /// A window or popup closed: forget it.
    pub fn window_closed(&self, window_id: WindowId) {
        let mut st = self.state.lock().unwrap();
        st.windows.remove(&window_id);
        st.popup_parent.remove(&window_id);
        st.bytes.remove(&Source::Window(window_id));
    }

    /// `n` bytes of frames for window or popup `window_id`.
    pub fn add_window(&self, window_id: WindowId, n: usize) {
        let mut st = self.state.lock().unwrap();
        let top = st.toplevel(window_id);
        *st.bytes.entry(Source::Window(top)).or_default() += n as u64;
    }

    pub fn audio_opened(&self, stream_id: u32, app_name: &str, media_name: &str) {
        self.state
            .lock()
            .unwrap()
            .audio
            .insert(stream_id, (app_name.to_string(), media_name.to_string()));
    }

    pub fn audio_closed(&self, stream_id: u32) {
        let mut st = self.state.lock().unwrap();
        st.audio.remove(&stream_id);
        st.bytes.remove(&Source::Audio(stream_id));
    }

    /// `n` bytes of audio stream `stream_id`.
    pub fn add_audio(&self, stream_id: u32, n: usize) {
        *self
            .state
            .lock()
            .unwrap()
            .bytes
            .entry(Source::Audio(stream_id))
            .or_default() += n as u64;
    }

    /// Every source that has received anything, with its running total.
    pub fn totals(&self) -> Vec<SourceTotal> {
        let st = self.state.lock().unwrap();
        st.bytes
            .iter()
            .map(|(&source, &bytes)| {
                let (name, detail) = match source {
                    Source::Window(id) => match st.windows.get(&id) {
                        Some((title, app_id)) if !title.is_empty() => {
                            (title.clone(), app_id.clone())
                        }
                        Some((_, app_id)) if !app_id.is_empty() => (app_id.clone(), app_id.clone()),
                        _ => (format!("Window #{id}"), String::new()),
                    },
                    Source::Audio(id) => match st.audio.get(&id) {
                        Some((app, media)) => (format!("{app} (audio)"), media.clone()),
                        None => ("Audio".to_string(), String::new()),
                    },
                };
                SourceTotal {
                    source,
                    name,
                    detail,
                    bytes,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(meter: &TrafficMeter, source: Source) -> Option<(String, u64)> {
        meter
            .totals()
            .into_iter()
            .find(|t| t.source == source)
            .map(|t| (t.name, t.bytes))
    }

    #[test]
    fn popups_count_for_their_toplevel() {
        let meter = TrafficMeter::default();
        meter.window_named(1, "", "org.example.Editor");
        meter.add_window(1, 100);
        meter.popup_created(2, 1);
        meter.popup_created(3, 2);
        meter.add_window(3, 50);
        assert_eq!(
            bytes_of(&meter, Source::Window(1)),
            Some(("org.example.Editor".into(), 150))
        );
        meter.window_named(1, "notes.txt", "");
        assert_eq!(
            bytes_of(&meter, Source::Window(1)),
            Some(("notes.txt".into(), 150))
        );
        meter.window_closed(1);
        assert_eq!(bytes_of(&meter, Source::Window(1)), None);
    }

    #[test]
    fn audio_streams_are_named_by_app() {
        let meter = TrafficMeter::default();
        meter.audio_opened(7, "Firefox", "Video");
        meter.add_audio(7, 10);
        assert_eq!(
            bytes_of(&meter, Source::Audio(7)),
            Some(("Firefox (audio)".into(), 10))
        );
        meter.audio_closed(7);
        assert!(meter.totals().is_empty());
    }
}
