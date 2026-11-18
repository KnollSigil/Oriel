//! Host profile persistence (`~/.config/remotewayland/hosts.json`) and SSH bootstrapping.

use std::fs;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProfile {
    pub label: String,
    pub ssh_target: String,
    #[serde(default)]
    pub server_bin: Option<String>,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_true")]
    pub audio_enabled: bool,
}

fn default_port() -> u16 {
    7443
}

fn default_true() -> bool {
    true
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
pub fn ssh_bootstrap(
    ssh_target: &str,
    server_bin: Option<&str>,
    listen_port: u16,
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

    let remote_cmd = format!(
        "{} attach --listen 0.0.0.0:{} --host {}",
        shell_escape(bin),
        listen_port,
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
