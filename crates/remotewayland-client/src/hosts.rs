//! Host profile persistence (`~/.config/remotewayland/hosts.json`) and SSH bootstrapping.

use std::fs;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use remotewayland_proto::PortRange;
use serde::{Deserialize, Serialize};

/// Maximum render scale cap for remote HiDPI windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderScaleCap {
    #[default]
    Uncapped,
    Scale200,
    Scale150,
    Scale100,
}

impl RenderScaleCap {
    pub const ALL: [RenderScaleCap; 4] = [
        RenderScaleCap::Uncapped,
        RenderScaleCap::Scale200,
        RenderScaleCap::Scale150,
        RenderScaleCap::Scale100,
    ];

    pub fn as_max_factor(self) -> Option<f64> {
        match self {
            Self::Uncapped => None,
            Self::Scale200 => Some(2.0),
            Self::Scale150 => Some(1.5),
            Self::Scale100 => Some(1.0),
        }
    }

    pub fn apply(self, raw_scale: f64) -> f64 {
        let s = if raw_scale.is_finite() && raw_scale >= 1.0 {
            raw_scale
        } else {
            1.0
        };
        match self.as_max_factor() {
            Some(cap) => s.min(cap).max(1.0),
            None => s.min(4.0),
        }
    }

    pub fn to_index(self) -> usize {
        match self {
            Self::Uncapped => 0,
            Self::Scale200 => 1,
            Self::Scale150 => 2,
            Self::Scale100 => 3,
        }
    }

    pub fn from_index(idx: usize) -> Self {
        match idx {
            1 => Self::Scale200,
            2 => Self::Scale150,
            3 => Self::Scale100,
            _ => Self::Uncapped,
        }
    }

    /// Encode an optional per-host override (`None` = Use Global Default) as a combo index `0..=4`.
    pub fn option_to_index(opt: Option<Self>) -> usize {
        match opt {
            None => 0,
            Some(cap) => cap.to_index() + 1,
        }
    }

    /// Decode a per-host combo index `0..=4` back to `Option<RenderScaleCap>`.
    pub fn option_from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => None,
            n => Some(Self::from_index(n - 1)),
        }
    }

    pub fn parse_cli(s: &str) -> Result<Self> {
        let norm = s.trim().to_ascii_lowercase();
        match norm.as_str() {
            "uncapped" | "none" | "auto" | "0" => Ok(Self::Uncapped),
            "2" | "2.0" | "2x" | "2.0x" | "200%" => Ok(Self::Scale200),
            "1.5" | "1.5x" | "150%" => Ok(Self::Scale150),
            "1" | "1.0" | "1x" | "1.0x" | "100%" => Ok(Self::Scale100),
            _ => bail!("invalid --max-scale '{s}': expected uncapped, 2.0, 1.5, or 1.0"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientSettings {
    #[serde(default)]
    pub max_scale: RenderScaleCap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProfile {
    pub label: String,
    pub ssh_target: String,
    #[serde(default)]
    pub server_bin: Option<String>,
    /// UDP port or port range the server may listen on; `None` lets it use any free port.
    /// Older clients saved a plain `port` number, which loads as that single port.
    #[serde(default, alias = "port", skip_serializing_if = "Option::is_none")]
    pub ports: Option<PortRange>,
    #[serde(default = "default_true")]
    pub audio_enabled: bool,
    #[serde(default)]
    pub max_scale: Option<RenderScaleCap>,
}

fn default_true() -> bool {
    true
}

pub fn settings_config_path() -> PathBuf {
    if let Some(cfg) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(cfg).join("remotewayland/settings.json")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/remotewayland/settings.json")
    } else {
        PathBuf::from("remotewayland-settings.json")
    }
}

pub fn load_client_settings() -> ClientSettings {
    let path = settings_config_path();
    let Ok(data) = fs::read_to_string(path) else {
        return ClientSettings::default();
    };
    serde_json::from_str(&data).unwrap_or_default()
}

pub fn save_client_settings(settings: &ClientSettings) -> Result<()> {
    let path = settings_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(settings)?;
    fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn hosts_config_path() -> PathBuf {
    if let Some(cfg) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(cfg).join("remotewayland/hosts.json")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/remotewayland/hosts.json")
    } else {
        PathBuf::from("remotewayland-hosts.json")
    }
}

pub fn load_hosts() -> Vec<HostProfile> {
    let path = hosts_config_path();
    let Ok(data) = fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str(&data).unwrap_or_default()
}

pub fn save_hosts(hosts: &[HostProfile]) -> Result<()> {
    let path = hosts_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(hosts)?;
    fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Extract the hostname portion of `user@host` (or `host`).
pub fn ssh_hostname(ssh_target: &str) -> &str {
    match ssh_target.rsplit_once('@') {
        Some((_, host)) => host,
        None => ssh_target,
    }
}

/// Run `ssh <ssh_target> <server_bin> attach` to start or attach to a persistent
/// `remotewayland-server` session and return the `remotewayland://...` connect URI.
/// Start (or attach to) the user's server on `ssh_target` and return its connect URI. With
/// `ports`, the server must listen on one of them; otherwise it may use any free port.
pub fn ssh_bootstrap(
    ssh_target: &str,
    server_bin: Option<&str>,
    ports: Option<PortRange>,
) -> Result<String> {
    let host_part = ssh_hostname(ssh_target);
    let connect_host = if host_part == "localhost" || host_part == "127.0.0.1" {
        "127.0.0.1"
    } else {
        host_part
    };

    let bin = server_bin
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("remotewayland-server");

    let listen = match ports {
        Some(ports) => format!("0.0.0.0:{ports}"),
        None => "0.0.0.0".to_string(),
    };
    let remote_cmd = format!(
        "{} attach --listen {} --host {}",
        shell_escape(bin),
        shell_escape(&listen),
        shell_escape(connect_host),
    );

    let output = Command::new("ssh")
        .arg(ssh_target)
        .arg(remote_cmd)
        .output()
        .with_context(|| format!("failed to invoke ssh to {ssh_target}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ssh attach to {ssh_target} failed ({}): {}",
            output.status,
            stderr.trim()
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let uri = stdout
        .lines()
        .rev()
        .find(|line| line.trim().starts_with("remotewayland://"))
        .map(|l| l.trim().to_string())
        .context("remote attach command did not output a remotewayland:// URI")?;

    Ok(uri)
}

/// Parse `remotewayland://host:port?fp=...&token=...`, resolving hostnames if needed.
pub fn parse_uri(uri: &str) -> Result<(SocketAddr, String, String)> {
    let rest = uri
        .strip_prefix("remotewayland://")
        .context("URI must start with remotewayland://")?;
    let (addr_str, query) = rest
        .split_once('?')
        .context("URI must include ?fp=...&token=...")?;

    let addr: SocketAddr = if let Ok(sa) = addr_str.parse::<SocketAddr>() {
        sa
    } else {
        addr_str
            .to_socket_addrs()
            .with_context(|| format!("failed to resolve server address '{addr_str}'"))?
            .next()
            .with_context(|| format!("no addresses resolved for '{addr_str}'"))?
    };

    let mut fp = None;
    let mut token = None;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            match k {
                "fp" => fp = Some(v.to_string()),
                "token" => token = Some(v.to_string()),
                _ => {}
            }
        }
    }

    match (fp, token) {
        (Some(fp), Some(token)) => Ok((addr, fp, token)),
        _ => bail!("URI missing fp or token query parameter"),
    }
}

fn shell_escape(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':' | '~'))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_scale_cap_apply_and_cli_parsing() {
        assert_eq!(RenderScaleCap::Uncapped.apply(1.5), 1.5);
        assert_eq!(RenderScaleCap::Uncapped.apply(3.0), 3.0);
        assert_eq!(RenderScaleCap::Uncapped.apply(0.5), 1.0);
        assert_eq!(RenderScaleCap::Scale200.apply(2.5), 2.0);
        assert_eq!(RenderScaleCap::Scale200.apply(1.25), 1.25);
        assert_eq!(RenderScaleCap::Scale150.apply(2.0), 1.5);
        assert_eq!(RenderScaleCap::Scale100.apply(2.0), 1.0);

        assert_eq!(
            RenderScaleCap::parse_cli("uncapped").unwrap(),
            RenderScaleCap::Uncapped
        );
        assert_eq!(
            RenderScaleCap::parse_cli("2.0x").unwrap(),
            RenderScaleCap::Scale200
        );
        assert_eq!(
            RenderScaleCap::parse_cli("1.5").unwrap(),
            RenderScaleCap::Scale150
        );
        assert_eq!(
            RenderScaleCap::parse_cli("100%").unwrap(),
            RenderScaleCap::Scale100
        );
        assert!(RenderScaleCap::parse_cli("3.5").is_err());

        for cap in RenderScaleCap::ALL {
            assert_eq!(RenderScaleCap::from_index(cap.to_index()), cap);
            assert_eq!(
                RenderScaleCap::option_from_index(RenderScaleCap::option_to_index(Some(cap))),
                Some(cap)
            );
        }
        assert_eq!(RenderScaleCap::option_from_index(0), None);
    }

    #[test]
    fn host_profile_deserializes_without_max_scale() {
        let json = r#"{"label":"Box","ssh_target":"u@h","port":7443,"audio_enabled":true}"#;
        let profile: HostProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile.max_scale, None);
    }

    #[test]
    fn host_profile_ports_default_to_any_and_keep_legacy_port() {
        let any: HostProfile =
            serde_json::from_str(r#"{"label":"Box","ssh_target":"u@h"}"#).unwrap();
        assert_eq!(any.ports, None);
        assert!(!serde_json::to_string(&any).unwrap().contains("port"));

        let legacy: HostProfile =
            serde_json::from_str(r#"{"label":"Box","ssh_target":"u@h","port":7443}"#).unwrap();
        assert_eq!(legacy.ports, Some("7443".parse().unwrap()));

        let mut ranged = legacy;
        ranged.ports = Some("7443-7462".parse().unwrap());
        let json = serde_json::to_string(&ranged).unwrap();
        assert!(json.contains(r#""ports":"7443-7462""#), "{json}");
        let back: HostProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ports, ranged.ports);
    }
}
