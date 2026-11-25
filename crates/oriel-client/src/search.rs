//! Tiered application search matching (Kickoff/KRunner style):
//! exact name prefix -> word prefix -> substring -> generic name / keyword ->
//! word-anchored fuzzy subsequence / 1-typo Damerau-Levenshtein -> sanitized exec key.

use oriel_proto::RemoteAppEntry;

/// Match quality of an application, best first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchTier {
    NamePrefix,
    NameWordPrefix,
    NameSubstring,
    GenericOrKeyword,
    Fuzzy,
    Exec,
}

/// Pre-lowercased searchable fields derived from a [`RemoteAppEntry`].
#[derive(Debug, Clone)]
pub struct SearchIndexEntry {
    pub name_lower: String,
    pub generic_name_lower: Option<String>,
    pub keywords_lower: Vec<String>,
    pub exec_key: String,
}

impl SearchIndexEntry {
    pub fn from_app(app: &RemoteAppEntry) -> Self {
        Self {
            name_lower: app.name.to_lowercase(),
            generic_name_lower: app
                .generic_name
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_lowercase),
            keywords_lower: app
                .keywords
                .iter()
                .map(|k| k.trim().to_lowercase())
                .filter(|k| !k.is_empty())
                .collect(),
            exec_key: exec_key(&app.exec),
        }
    }
}

/// If `query` starts with `>` or `!`, return the trimmed command to run directly
/// (skipping application matching). Returns `Some("")` if only the prefix was typed.
pub fn explicit_command(query: &str) -> Option<&str> {
    let trimmed = query.trim();
    trimmed
        .strip_prefix('>')
        .or_else(|| trimmed.strip_prefix('!'))
        .map(str::trim)
}

/// The searchable part of a command: the program's basename, or the app ID for `flatpak run`.
/// Leading `env VAR=value` wrappers are skipped.
pub fn exec_key(exec: &str) -> String {
    let mut tokens = exec
        .split_whitespace()
        .skip_while(|tok| tok.rsplit('/').next() == Some("env") || tok.contains('='));
    let Some(program) = tokens.next() else {
        return String::new();
    };
    let program = program.rsplit('/').next().unwrap_or(program);
    if program == "flatpak" && tokens.next() == Some("run") {
        if let Some(app_id) = tokens.find(|tok| !tok.starts_with('-')) {
            return app_id.to_lowercase();
        }
    }
    program.to_lowercase()
}

fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
}

/// Return the best [`MatchTier`] for `entry` against an already-trimmed, lowercased `q`.
/// Returns `None` if `q` is empty or does not match.
pub fn match_tier(entry: &SearchIndexEntry, q: &str) -> Option<MatchTier> {
    if q.is_empty() || explicit_command(q).is_some() {
        return None;
    }
    if entry.name_lower.starts_with(q) {
        Some(MatchTier::NamePrefix)
    } else if words(&entry.name_lower).any(|w| w.starts_with(q)) {
        Some(MatchTier::NameWordPrefix)
    } else if entry.name_lower.contains(q) {
        Some(MatchTier::NameSubstring)
    } else if entry
        .generic_name_lower
        .as_deref()
        .is_some_and(|g| g.contains(q))
        || entry.keywords_lower.iter().any(|k| k.starts_with(q))
    {
        Some(MatchTier::GenericOrKeyword)
    } else if fuzzy_matches(&entry.name_lower, q) {
        Some(MatchTier::Fuzzy)
    } else if !entry.exec_key.is_empty() && entry.exec_key.contains(q) {
        Some(MatchTier::Exec)
    } else {
        None
    }
}

/// Subsequence match starting at a word ("ffx" -> "firefox", but not "fire" -> "libreof*f*ice
/// w*r*it*e*r"), or for queries of 4+ characters one typo against the start of the name or of one
/// of its words ("firfox" -> "firefox").
fn fuzzy_matches(name: &str, q: &str) -> bool {
    let q_chars: Vec<char> = q.chars().collect();
    if q_chars.len() >= 2 {
        let mut prev_alnum = false;
        for (byte_idx, ch) in name.char_indices() {
            let is_alnum = ch.is_alphanumeric();
            if is_alnum && !prev_alnum && ch == q_chars[0] {
                let mut rest = name[byte_idx..].chars();
                if q_chars.iter().all(|&qc| rest.any(|nc| nc == qc)) {
                    return true;
                }
            }
            prev_alnum = is_alnum;
        }
    }
    if q_chars.len() < 4 {
        return false;
    }
    let name_chars: Vec<char> = name.chars().collect();
    let matches_prefix = |slice: &[char]| {
        // Compare against prefixes one shorter, equal, and one longer than the query.
        (q_chars.len().saturating_sub(1)..=q_chars.len() + 1)
            .filter(|&len| len <= slice.len())
            .any(|len| edit_distance_at_most_one(&q_chars, &slice[..len]))
    };
    if matches_prefix(&name_chars) {
        return true;
    }
    let mut start = 0usize;
    while start < name_chars.len() {
        while start < name_chars.len() && !name_chars[start].is_alphanumeric() {
            start += 1;
        }
        let mut end = start;
        while end < name_chars.len() && name_chars[end].is_alphanumeric() {
            end += 1;
        }
        if start < end && matches_prefix(&name_chars[start..end]) {
            return true;
        }
        start = end;
    }
    false
}

/// `true` if `a` and `b` differ by at most one insertion, deletion, substitution, or adjacent swap.
fn edit_distance_at_most_one(a: &[char], b: &[char]) -> bool {
    if a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    let prefix = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => a.len() + b.len() <= 1,
        _ if a.len() == b.len() => {
            a[1..] == b[1..] || (a.len() >= 2 && a[0] == b[1] && a[1] == b[0] && a[2..] == b[2..])
        }
        _ if a.len() > b.len() => a[1..] == *b,
        _ => *a == b[1..],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, generic: Option<&str>, keywords: &[&str], exec: &str) -> SearchIndexEntry {
        let app = RemoteAppEntry {
            id: format!("{name}.desktop"),
            name: name.to_string(),
            generic_name: generic.map(String::from),
            keywords: keywords.iter().map(|s| (*s).to_string()).collect(),
            exec: exec.to_string(),
            categories: Vec::new(),
            icon_mime: None,
            icon_data: Vec::new(),
        };
        SearchIndexEntry::from_app(&app)
    }

    fn rank<'a>(entries: &'a [SearchIndexEntry], query: &str) -> Vec<&'a str> {
        let q = query.trim().to_lowercase();
        let mut hits: Vec<(MatchTier, &str)> = entries
            .iter()
            .filter_map(|e| Some((match_tier(e, &q)?, e.name_lower.as_str())))
            .collect();
        hits.sort();
        hits.into_iter().map(|(_, name)| name).collect()
    }

    #[test]
    fn tiers_rank_prefix_word_substring_generic_keyword_fuzzy_exec() {
        let apps = vec![
            entry(
                "Firefox",
                Some("Web Browser"),
                &["internet", "www"],
                "firefox",
            ),
            entry("Konsole", Some("Terminal"), &["shell", "cli"], "konsole"),
            entry("Visual Studio Code", Some("Text Editor"), &["ide"], "code"),
            entry(
                "Firewall",
                Some("Firewall Settings"),
                &[],
                "firewall-config",
            ),
            entry(
                "Discord",
                Some("Messenger"),
                &[],
                "/usr/bin/flatpak run --branch=stable com.discordapp.Discord",
            ),
            entry("Kate", Some("Text Editor"), &[], "kate"),
            entry("Ark", Some("Archiver"), &[], "ark"),
        ];

        assert_eq!(rank(&apps, "fire"), vec!["firefox", "firewall"]);
        assert_eq!(rank(&apps, "code")[0], "visual studio code");
        assert_eq!(rank(&apps, "terminal"), vec!["konsole"]);
        assert_eq!(rank(&apps, "shell"), vec!["konsole"]);
        assert_eq!(rank(&apps, "editor"), vec!["kate", "visual studio code"]);
        assert_eq!(rank(&apps, "firfox")[0], "firefox");
        assert_eq!(rank(&apps, "ffx")[0], "firefox");
        assert_eq!(rank(&apps, "ar")[0], "ark");
        assert!(rank(&apps, "run").is_empty());
        assert_eq!(rank(&apps, "discordapp"), vec!["discord"]);
    }

    #[test]
    fn fuzzy_subsequence_requires_word_start_and_min_length() {
        let apps = vec![entry(
            "LibreOffice Writer",
            Some("Word Processor"),
            &[],
            "lowriter",
        )];
        assert!(rank(&apps, "fire").is_empty());
        assert!(rank(&apps, "z").is_empty());
    }

    #[test]
    fn explicit_command_prefixes_skip_app_matching() {
        assert_eq!(explicit_command(">foot -e htop"), Some("foot -e htop"));
        assert_eq!(explicit_command("! firefox"), Some("firefox"));
        assert_eq!(explicit_command(">"), Some(""));
        assert_eq!(explicit_command("firefox"), None);
    }

    #[test]
    fn edit_distance_at_most_one_handles_all_single_edits() {
        let d = |a: &str, b: &str| {
            edit_distance_at_most_one(
                &a.chars().collect::<Vec<_>>(),
                &b.chars().collect::<Vec<_>>(),
            )
        };
        assert!(d("firefox", "firefox"));
        assert!(d("firfox", "firefox"));
        assert!(d("firefoxx", "firefox"));
        assert!(d("fierfox", "firefox"));
        assert!(d("firebox", "firefox"));
        assert!(!d("fxrefxx", "firefox"));
        assert!(!d("fire", "firefox"));
    }

    #[test]
    fn exec_key_strips_env_and_flatpak_wrappers() {
        assert_eq!(
            exec_key("env FOO=1 /usr/bin/firefox --new-window"),
            "firefox"
        );
        assert_eq!(
            exec_key("/usr/bin/env FOO=1 BAR=2 /usr/bin/firefox --new-window"),
            "firefox"
        );
        assert_eq!(
            exec_key("/usr/bin/flatpak run --branch=stable --arch=x86_64 org.chromium.Chromium"),
            "org.chromium.chromium"
        );
    }

    #[test]
    fn fuzzy_matches_unicode_app_names() {
        let apps = vec![entry("Über Editor", None, &[], "uber")];
        assert_eq!(rank(&apps, "über"), vec!["über editor"]);
        assert_eq!(rank(&apps, "übr"), vec!["über editor"]);
        assert_eq!(rank(&apps, "ubex"), Vec::<&str>::new());
        assert_eq!(rank(&apps, "übar"), vec!["über editor"]);
    }
}
