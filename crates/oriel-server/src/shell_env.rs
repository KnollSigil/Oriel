//! Resolve the user's login-shell environment, the way VS Code does when it is started
//! outside a terminal.
//!
//! The server is usually started by `attach` run as a one-off SSH command
//! (`ssh host oriel-server attach`), but may also be started by hand, from a
//! service or from cron. None of these need have a login shell's environment, so the
//! server would miss everything users and distros set in their profiles (`PATH`
//! additions, Flatpak's `XDG_DATA_DIRS`, locale, ...). `attach` resolves it before starting
//! the daemon (so failures reach its stderr) and tells the daemon not to repeat it; a
//! server started any other way resolves it itself at startup. Either way it runs the
//! user's shell as an interactive login shell that runs this binary's hidden `print-shell-env`
//! subcommand, which prints the environment (as `env -0` would, so every byte survives)
//! between random markers so anything the profile itself prints is ignored. If that fails (an rc file that `exec`s tmux or
//! another shell, or waits for input), it retries as a non-interactive login shell.
//!
//! The shell starts from a clean login baseline (the account's `HOME`/`USER`/`LOGNAME`/
//! `SHELL`, a default `PATH`, and the `pam_env` files `/etc/environment` and
//! `/etc/default/locale`), and its result replaces the server's environment entirely, apart
//! from `XDG_RUNTIME_DIR` and `RUST_LOG`. Nothing from the SSH connection or terminal the
//! server was started from carries over: no forwarded agent socket, no client locale. If
//! the shell fails, the server runs with the baseline alone.
//!
//! Profiles can detect this with `ORIEL_RESOLVING_ENVIRONMENT=1` and skip slow or
//! interactive setup. Set `ORIEL_SKIP_SHELL_ENV=1` to turn resolution off and keep
//! the inherited environment as it is.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const RESOLVING_VAR: &str = "ORIEL_RESOLVING_ENVIRONMENT";
const SKIP_VAR: &str = "ORIEL_SKIP_SHELL_ENV";
const EXE_VAR: &str = "ORIEL_SHELL_ENV_EXE";
const MARKER_VAR: &str = "ORIEL_SHELL_ENV_MARKER";
/// Set by `attach` on the daemon it starts: the login environment was already resolved
/// (or tried), so the daemon must not run the shell again.
pub const ATTACH_RESOLVED_VAR: &str = "ORIEL_SHELL_ENV_RESOLVED";
const MAX_OUTPUT: usize = 4 * 1024 * 1024;

/// Environment variables by name, kept as raw bytes.
pub type EnvMap = BTreeMap<OsString, OsString>;

/// Hidden subcommand name `attach` runs inside the login shell.
pub const PRINT_SUBCOMMAND: &str = "print-shell-env";

/// Variables that describe the terminal or display a shell runs in, not the user's
/// environment, dropped from the login shell's output. The server starts without them and
/// sets its own where it needs them. `XDG_RUNTIME_DIR` is handled by [`carry_over`].
fn is_session_specific(name: &str) -> bool {
    const EXACT: &[&str] = &[
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "DBUS_SESSION_BUS_ADDRESS",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "TERM",
        "COLORTERM",
        "COLUMNS",
        "LINES",
        "TTY",
        "GPG_TTY",
        "WINDOWID",
        "VTE_VERSION",
        "TMUX",
        "TMUX_PANE",
        "STY",
        // Set by sshd. The shell starts from a clean baseline, so these can only come from
        // a profile, but they would still describe a connection that isn't the server's.
        // Other `SSH_*` variables a profile sets (an agent socket on this host,
        // `SSH_ASKPASS`) are kept.
        "SSH_CONNECTION",
        "SSH_CLIENT",
        "SSH_TTY",
        "SSH_ORIGINAL_COMMAND",
    ];
    const PREFIXES: &[&str] = &[
        "ORIEL_",
        "TERM_PROGRAM",
        "ZELLIJ",
        "KITTY_",
        "WEZTERM_",
        "ALACRITTY_",
        "KONSOLE_",
    ];
    EXACT.contains(&name) || PREFIXES.iter().any(|p| name.starts_with(p))
}

/// `print-shell-env`: print this process's environment as `<marker>K=V\0K=V\0...<marker>`.
pub fn print_env() -> Result<()> {
    use std::io::Write;
    let marker = std::env::var(MARKER_VAR).context("print-shell-env is for internal use")?;
    let mut out = std::io::stdout().lock();
    out.write_all(&env_payload(&marker, std::env::vars_os()))?;
    out.flush()?;
    Ok(())
}

fn env_payload(marker: &str, vars: impl Iterator<Item = (OsString, OsString)>) -> Vec<u8> {
    let mut out = marker.as_bytes().to_vec();
    for (key, value) in vars {
        // The marker must not appear inside the payload it delimits.
        if [MARKER_VAR, EXE_VAR, RESOLVING_VAR]
            .iter()
            .any(|k| k.as_bytes() == key.as_bytes())
        {
            continue;
        }
        out.extend_from_slice(key.as_bytes());
        out.push(b'=');
        out.extend_from_slice(value.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(marker.as_bytes());
    out
}

/// Parse `K=V\0` entries (the format of `env -0`).
fn parse_payload(payload: &[u8]) -> EnvMap {
    payload
        .split(|&b| b == 0)
        .filter_map(|entry| {
            let eq = entry.iter().position(|&b| b == b'=').filter(|&i| i > 0)?;
            Some((
                OsString::from_vec(entry[..eq].to_vec()),
                OsString::from_vec(entry[eq + 1..].to_vec()),
            ))
        })
        .collect()
}

/// The flags and command string that make `shell` run `"$EXE_VAR" print-shell-env` as a
/// login shell, interactive or not. Mirrors VS Code's per-shell handling. Returns `None`
/// when the shell has no distinct invocation for that mode.
fn shell_invocation(shell: &str, interactive: bool) -> Option<(Vec<&'static str>, String)> {
    let name = Path::new(shell)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let posix_cmd = || format!("\"${EXE_VAR}\" {PRINT_SUBCOMMAND}");
    Some(match (name, interactive) {
        // csh only accepts `-l` as its sole argument, so it can't be a login shell with
        // `-c`: it reads `.cshrc` (in either mode) but never `.login`.
        ("csh" | "tcsh", true) => (vec!["-i", "-c"], posix_cmd()),
        ("csh" | "tcsh", false) => (vec!["-c"], posix_cmd()),
        ("nu", true) => (
            vec!["-i", "-l", "-c"],
            format!("^$env.{EXE_VAR} {PRINT_SUBCOMMAND}"),
        ),
        ("nu", false) => (
            vec!["-l", "-c"],
            format!("^$env.{EXE_VAR} {PRINT_SUBCOMMAND}"),
        ),
        ("pwsh" | "pwsh-preview", true) => (
            vec!["-Login", "-Command"],
            format!("& $env:{EXE_VAR} {PRINT_SUBCOMMAND}"),
        ),
        ("pwsh" | "pwsh-preview", false) => return None,
        (_, true) => (vec!["-i", "-l", "-c"], posix_cmd()),
        (_, false) => (vec!["-l", "-c"], posix_cmd()),
    })
}

/// Finds the payload between the first two markers in a stream read in chunks, scanning
/// each byte only once (plus a marker-length overlap for markers split across chunks).
struct MarkerScanner<'a> {
    marker: &'a [u8],
    buf: Vec<u8>,
    scanned: usize,
    payload_start: Option<usize>,
}

impl<'a> MarkerScanner<'a> {
    fn new(marker: &'a str) -> Self {
        Self {
            marker: marker.as_bytes(),
            buf: Vec::new(),
            scanned: 0,
            payload_start: None,
        }
    }

    fn find(&self, from: usize) -> Option<usize> {
        self.buf[from..]
            .windows(self.marker.len())
            .position(|w| w == self.marker)
            .map(|i| from + i)
    }

    /// Feed more output; returns the payload once the closing marker has arrived.
    fn push(&mut self, chunk: &[u8]) -> Option<&[u8]> {
        self.buf.extend_from_slice(chunk);
        let from = self.scanned.saturating_sub(self.marker.len() - 1);
        self.scanned = self.buf.len();
        if self.payload_start.is_none() {
            self.payload_start = self.find(from).map(|i| i + self.marker.len());
        }
        let start = self.payload_start?;
        let end = self.find(start.max(from))?;
        Some(&self.buf[start..end])
    }
}

/// Kills a process group on drop, unless disarmed.
struct GroupKiller(Option<i32>);

impl Drop for GroupKiller {
    fn drop(&mut self) {
        if let Some(pgid) = self.0.filter(|&p| p > 0) {
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
    }
}

struct Options {
    interactive_timeout: Duration,
    login_timeout: Duration,
    /// The whole environment the shell starts with; nothing else is inherited.
    base_env: EnvMap,
}

impl Options {
    fn new(base_env: EnvMap) -> Self {
        Self {
            interactive_timeout: Duration::from_secs(10),
            login_timeout: Duration::from_secs(5),
            base_env,
        }
    }
}

/// The current user's account entry from the password database.
struct Account {
    name: OsString,
    home: OsString,
    shell: OsString,
}

fn account() -> Option<Account> {
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    let mut buf = vec![0 as libc::c_char; 16 * 1024];
    let rc = unsafe {
        libc::getpwuid_r(
            libc::getuid(),
            &mut pwd,
            buf.as_mut_ptr(),
            buf.len(),
            &mut result,
        )
    };
    if rc != 0 || result.is_null() {
        return None;
    }
    let field = |ptr: *const libc::c_char| -> OsString {
        if ptr.is_null() {
            return OsString::new();
        }
        let bytes = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes();
        OsString::from_vec(bytes.to_vec())
    };
    Some(Account {
        name: field(pwd.pw_name),
        home: field(pwd.pw_dir),
        shell: field(pwd.pw_shell),
    })
}

/// Shells that refuse logins (service accounts): there is no profile to read.
fn refuses_login(shell: &str) -> bool {
    matches!(
        Path::new(shell.trim()).file_name().and_then(|n| n.to_str()),
        Some("nologin" | "false")
    )
}

/// The user's login shell: the account's shell from the password database, or `$SHELL`
/// if that can't be read. The account comes first because `$SHELL` need not be the login
/// shell: cron, for one, sets it to `/bin/sh`. `Err` says why there is none to run.
fn login_shell() -> std::result::Result<String, String> {
    let shell = account()
        .and_then(|a| a.shell.into_string().ok())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("SHELL").ok().filter(|s| !s.trim().is_empty()))
        .ok_or_else(|| "the account has no shell and $SHELL is unset".to_string())?;
    if refuses_login(&shell) {
        return Err(format!("the account's shell is {shell}"));
    }
    Ok(shell)
}

/// `PATH` before `/etc/environment` or the profile change it (systemd's default).
const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// Files a login session's environment comes from before any user startup file runs:
/// `/etc/environment` (`pam_env`; Ubuntu's `PATH`), and the system locale from systemd's
/// `/etc/locale.conf` (read by `profile.d` on Fedora, Arch and SteamOS) or Debian's
/// `/etc/default/locale` (`pam_env`). All use plain `KEY=VALUE` lines.
const BASELINE_ENV_FILES: &[&str] = &[
    "/etc/environment",
    "/etc/locale.conf",
    "/etc/default/locale",
];

/// Parse `pam_env`-style `KEY=VALUE` lines: no expansion, optional `export ` and quotes.
fn parse_pam_env_file(content: &str) -> Vec<(OsString, OsString)> {
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
            let (key, value) = line.split_once('=')?;
            let valid_key = key
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !valid_key {
                return None;
            }
            let value = value.trim();
            let value = ['"', '\'']
                .iter()
                .find_map(|q| value.strip_prefix(*q)?.strip_suffix(*q))
                .unwrap_or(value);
            Some((key.into(), value.into()))
        })
        .collect()
}

/// Variables the server itself needs from the environment it was started with: its
/// session runtime dir, and its own log filter. An inherited value wins over the
/// profile's; the profile's is kept only where none was inherited. For `XDG_RUNTIME_DIR`
/// that matters: `attach` looks for a running daemon's control socket in its own
/// (inherited) runtime dir, so the daemon must use that same dir.
const CARRIED_OVER: &[&str] = &["XDG_RUNTIME_DIR", "RUST_LOG"];

fn carry_over(mut env: EnvMap) -> EnvMap {
    for key in CARRIED_OVER {
        if let Some(value) = std::env::var_os(key) {
            env.insert(key.into(), value);
        }
    }
    env
}

/// The environment a fresh login would have before any shell startup file runs: the
/// account's `HOME`/`USER`/`LOGNAME`/`SHELL`, a default `PATH`, and the `pam_env` files.
/// Nothing from the SSH connection or terminal the server was started from gets in, so
/// the result doesn't depend on which client started it (no forwarded agent socket, no
/// client locale from `SendEnv`).
fn baseline_env() -> EnvMap {
    let mut env = EnvMap::new();
    env.insert("PATH".into(), DEFAULT_PATH.into());
    for file in BASELINE_ENV_FILES {
        if let Ok(content) = std::fs::read_to_string(file) {
            env.extend(parse_pam_env_file(&content));
        }
    }
    match account() {
        Some(a) => {
            env.insert("HOME".into(), a.home);
            env.insert("USER".into(), a.name.clone());
            env.insert("LOGNAME".into(), a.name);
            if !a.shell.is_empty() {
                env.insert("SHELL".into(), a.shell);
            }
        }
        None => {
            for key in ["HOME", "USER", "LOGNAME", "SHELL"] {
                if let Some(value) = std::env::var_os(key) {
                    env.insert(key.into(), value);
                }
            }
        }
    }
    carry_over(env)
}

/// Whether `ORIEL_SKIP_SHELL_ENV` turns resolution off.
fn skip_requested() -> bool {
    std::env::var_os(SKIP_VAR).is_some_and(|v| !v.is_empty() && v != "0")
}

/// The environment the server should run with.
pub enum Resolution {
    /// `ORIEL_SKIP_SHELL_ENV`: keep the inherited environment as it is.
    Skipped,
    /// No login shell to run (e.g. a `nologin` service account): keep the inherited
    /// environment, since there is no profile to replace it with.
    NoLoginShell(String),
    /// The login shell's environment.
    Resolved(EnvMap),
    /// The login shell failed: the clean baseline, without the profile's additions.
    Fallback { env: EnvMap, error: anyhow::Error },
}

/// Run the user's login shell from a clean login baseline and return its environment,
/// which then replaces the server's inherited one entirely.
pub async fn resolve(exe: &Path) -> Resolution {
    if skip_requested() {
        return Resolution::Skipped;
    }
    let shell = match login_shell() {
        Ok(shell) => shell,
        Err(reason) => return Resolution::NoLoginShell(reason),
    };
    let base = baseline_env();
    match resolve_with(&shell, exe, &Options::new(base.clone())).await {
        Ok(env) => Resolution::Resolved(carry_over(env)),
        Err(error) => Resolution::Fallback { env: base, error },
    }
}

/// What [`apply_at_startup`] did, for logging once logging is set up.
pub enum StartupOutcome {
    /// `attach` resolved (or tried to resolve) the environment before starting this server.
    ResolvedByAttach,
    Skipped,
    NoLoginShell(String),
    Applied(usize),
    Fallback(usize, anyhow::Error),
    /// Nothing was changed: the inherited environment is kept.
    NotApplied(anyhow::Error),
}

/// Replace this process's environment with the login one, unless `attach` already did.
/// Must run before any other thread exists: changing the environment while other threads
/// may read it is undefined behavior.
pub fn apply_at_startup() -> StartupOutcome {
    if std::env::var_os(ATTACH_RESOLVED_VAR).is_some() {
        std::env::remove_var(ATTACH_RESOLVED_VAR);
        return StartupOutcome::ResolvedByAttach;
    }
    if skip_requested() {
        return StartupOutcome::Skipped;
    }
    // Logging isn't set up yet, and this can take several seconds.
    eprintln!("oriel-server: reading the login shell environment ({SKIP_VAR}=1 skips this)");
    let resolution = std::env::current_exe()
        .context("failed to locate oriel-server binary")
        .and_then(|exe| {
            Ok(tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("failed to start a runtime")?
                .block_on(resolve(&exe)))
        })
        .unwrap_or_else(|error| Resolution::Fallback {
            env: baseline_env(),
            error,
        });
    let (env, error) = match resolution {
        Resolution::Skipped => return StartupOutcome::Skipped,
        Resolution::NoLoginShell(reason) => return StartupOutcome::NoLoginShell(reason),
        Resolution::Resolved(env) => (env, None),
        Resolution::Fallback { env, error } => (env, Some(error)),
    };
    // The temporary runtime is gone, but a helper thread (tokio's, or one NSS started for
    // getpwuid_r) may still be exiting; give it a moment before giving up.
    let thread_count = || std::fs::read_dir("/proc/self/task").map_or(1, |d| d.count());
    let mut threads = thread_count();
    for _ in 0..20 {
        if threads == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
        threads = thread_count();
    }
    if threads > 1 {
        return StartupOutcome::NotApplied(anyhow::anyhow!(
            "the process already has {threads} threads"
        ));
    }
    let count = env.len();
    for (key, _) in std::env::vars_os() {
        if !env.contains_key(&key) {
            std::env::remove_var(key);
        }
    }
    for (key, value) in env {
        std::env::set_var(key, value);
    }
    match error {
        None => StartupOutcome::Applied(count),
        Some(error) => StartupOutcome::Fallback(count, error),
    }
}

async fn resolve_with(shell: &str, exe: &Path, opts: &Options) -> Result<EnvMap> {
    let interactive_err = match run_shell(shell, exe, true, opts.interactive_timeout, opts).await {
        Ok(vars) => return Ok(vars),
        Err(err) => err,
    };
    let login = match shell_invocation(shell, false) {
        Some(_) => run_shell(shell, exe, false, opts.login_timeout, opts).await,
        None => Err(anyhow::anyhow!("no non-interactive login mode")),
    };
    login.map_err(|login_err| {
        anyhow::anyhow!(
            "could not read the login environment from {shell} (interactive: {interactive_err:#}; \
             login only: {login_err:#}); make sure your shell startup files don't exec another \
             program or wait for input when {RESOLVING_VAR}=1 is set"
        )
    })
}

async fn run_shell(
    shell: &str,
    exe: &Path,
    interactive: bool,
    timeout: Duration,
    opts: &Options,
) -> Result<EnvMap> {
    let mut nonce = [0u8; 16];
    crate::session::fill_random_bytes(&mut nonce)?;
    let marker: String = std::iter::once("__ORIEL_ENV_".to_string())
        .chain(nonce.iter().map(|b| format!("{b:02x}")))
        .chain(std::iter::once("__".to_string()))
        .collect();

    let (flags, command) = shell_invocation(shell, interactive).context("unsupported mode")?;
    let mut cmd = Command::new(shell);
    cmd.env_clear()
        .envs(&opts.base_env)
        .args(flags)
        .arg(command)
        .env(RESOLVING_VAR, "1")
        .env(EXE_VAR, exe)
        .env(MARKER_VAR, &marker)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // A new session: no controlling terminal, so an interactive shell started from one
    // (`attach` run by hand, `ssh -t`) can't stop itself with SIGTTIN waiting to own it,
    // and the shell leads its own process group, which a failure kills as a whole.
    // That also keeps the terminal's Ctrl-C from reaching it, and a fatal signal skips
    // `GroupKiller`, so have the kernel kill the shell if this process dies first. The
    // death signal follows the spawning thread: here that is the main thread (startup) or
    // a tokio worker (`attach`), both of which live until the process exits.
    let parent = unsafe { libc::getpid() };
    unsafe {
        cmd.pre_exec(move || {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            // The parent may have died before the death signal was set.
            if libc::getppid() != parent {
                // No allocation here: another thread may have held the malloc lock at fork time.
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to run {shell}"))?;
    let mut killer = GroupKiller(child.id().map(|id| id as i32));
    let mut stdout = child.stdout.take().context("shell stdout missing")?;

    // Read until the closing marker rather than EOF: a background process the profile
    // started may keep the pipe open long after the shell has exited.
    let read = async {
        let mut scanner = MarkerScanner::new(&marker);
        let mut chunk = [0u8; 8192];
        loop {
            let n = stdout.read(&mut chunk).await?;
            if n == 0 {
                bail!("exited without printing its environment");
            }
            if let Some(payload) = scanner.push(&chunk[..n]) {
                return Ok(payload.to_vec());
            }
            if scanner.buf.len() > MAX_OUTPUT {
                bail!("printed too much output");
            }
        }
    };
    let payload = match tokio::time::timeout(timeout, read).await {
        Ok(res) => res?,
        Err(_) => bail!(
            "did not print its environment within {}s",
            timeout.as_secs_f32()
        ),
    };
    // Success: the shell exits right after printing, and whatever the profile started in
    // the background is left alone. tokio reaps the shell if this process outlives it.
    killer.0 = None;
    drop(child);

    Ok(parse_payload(&payload)
        .into_iter()
        // A name that isn't UTF-8 can't be one of the session-specific ones.
        .filter(|(k, _)| !k.to_str().is_some_and(is_session_specific))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_scanner_finds_payload_across_chunks() {
        let m = "__ORIEL_ENV_ab__";
        let out = "motd\n__ORIEL_ENV_ab__{\"A\":\"1\"}__ORIEL_ENV_ab__bye";
        // Every split point, including inside either marker.
        for split in 0..=out.len() {
            let mut s = MarkerScanner::new(m);
            let first = s.push(&out.as_bytes()[..split]).map(<[u8]>::to_vec);
            let got = first.or_else(|| s.push(&out.as_bytes()[split..]).map(<[u8]>::to_vec));
            assert_eq!(got.as_deref(), Some(&b"{\"A\":\"1\"}"[..]), "split {split}");
        }
        // Byte by byte.
        let mut s = MarkerScanner::new(m);
        let mut got = None;
        for b in out.as_bytes() {
            if let Some(p) = s.push(&[*b]) {
                got = Some(p.to_vec());
                break;
            }
        }
        assert_eq!(got.as_deref(), Some(&b"{\"A\":\"1\"}"[..]));

        let mut s = MarkerScanner::new(m);
        assert_eq!(s.push(b"motd\n__ORIEL_ENV_ab__{\"A\""), None);
        let mut s = MarkerScanner::new(m);
        assert_eq!(s.push(b"nothing"), None);
    }

    #[test]
    fn env_payload_round_trips_without_its_own_marker() {
        let m = "__ORIEL_ENV_ab__";
        let raw = OsString::from_vec(b"/h\xf6me".to_vec());
        let vars = vec![
            (MARKER_VAR.into(), m.into()),
            ("PATH".into(), "/x".into()),
            ("EQ".into(), "a=b".into()),
            ("RAW".into(), raw.clone()),
            (EXE_VAR.into(), "/bin/oriel".into()),
        ];
        let out = env_payload(m, vars.into_iter());
        let mut s = MarkerScanner::new(m);
        let parsed = parse_payload(s.push(&out).unwrap());
        assert_eq!(parsed.len(), 3, "{parsed:?}");
        assert_eq!(parsed[OsStr::new("PATH")], "/x");
        assert_eq!(parsed[OsStr::new("EQ")], "a=b");
        assert_eq!(parsed[OsStr::new("RAW")], raw);
    }

    #[test]
    fn parses_pam_env_files() {
        let parsed = parse_pam_env_file(
            "# comment\n\nPATH=\"/usr/local/bin:/usr/bin\"\nexport LANG='en_US.UTF-8'\n\
             LC_TIME=C\nBAD-KEY=x\n=novalue\nnoequals\nEMPTY=\nSPACED = a b \n",
        );
        let expect = [
            ("PATH", "/usr/local/bin:/usr/bin"),
            ("LANG", "en_US.UTF-8"),
            ("LC_TIME", "C"),
            ("EMPTY", ""),
        ];
        assert_eq!(
            parsed,
            expect
                .iter()
                .map(|(k, v)| (OsString::from(k), OsString::from(v)))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn baseline_has_account_and_default_path_but_nothing_inherited() {
        let base = baseline_env();
        for key in ["HOME", "USER", "LOGNAME", "PATH"] {
            assert!(
                base.contains_key(OsStr::new(key)),
                "{key} missing: {base:?}"
            );
        }
        // Only these may come from the process's own environment.
        let allowed = [
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "XDG_RUNTIME_DIR",
            "RUST_LOG",
        ];
        let from_files: Vec<OsString> = BASELINE_ENV_FILES
            .iter()
            .filter_map(|f| std::fs::read_to_string(f).ok())
            .flat_map(|c| parse_pam_env_file(&c).into_iter().map(|(k, _)| k))
            .collect();
        for key in base.keys() {
            let ok = key == "PATH"
                || from_files.contains(key)
                || key.to_str().is_some_and(|k| allowed.contains(&k));
            assert!(ok, "unexpected baseline variable {key:?}");
        }
    }

    #[test]
    fn session_specific_vars_are_dropped() {
        for name in [
            "SSH_CONNECTION",
            "SHLVL",
            "ORIEL_X",
            "GPG_TTY",
            "KITTY_WINDOW_ID",
            "TERM_PROGRAM_VERSION",
            "ZELLIJ_SESSION_NAME",
        ] {
            assert!(is_session_specific(name), "{name}");
        }
        for name in [
            "PATH",
            "XDG_DATA_DIRS",
            "LANG",
            "HOME",
            "TERMINFO_DIRS",
            "SSH_AUTH_SOCK",
            "SSH_ASKPASS",
            "XDG_RUNTIME_DIR",
        ] {
            assert!(!is_session_specific(name), "{name}");
        }
    }

    #[test]
    fn nologin_shells_refuse_login() {
        for shell in [
            "/usr/sbin/nologin",
            "/sbin/nologin",
            "/bin/false",
            "/usr/bin/false",
        ] {
            assert!(refuses_login(shell), "{shell}");
        }
        for shell in ["/bin/bash", "/usr/bin/fish", "/bin/sh"] {
            assert!(!refuses_login(shell), "{shell}");
        }
    }

    #[test]
    fn shell_invocation_per_shell() {
        let flags = |s, i| shell_invocation(s, i).map(|(f, _)| f);
        assert_eq!(flags("/bin/bash", true), Some(vec!["-i", "-l", "-c"]));
        assert_eq!(flags("/bin/bash", false), Some(vec!["-l", "-c"]));
        assert_eq!(flags("/usr/bin/fish", true), Some(vec!["-i", "-l", "-c"]));
        assert_eq!(flags("/bin/tcsh", true), Some(vec!["-i", "-c"]));
        assert_eq!(flags("/bin/tcsh", false), Some(vec!["-c"]));
        assert_eq!(
            flags("/usr/bin/pwsh", true),
            Some(vec!["-Login", "-Command"])
        );
        assert_eq!(flags("/usr/bin/pwsh", false), None);
        assert!(shell_invocation("/usr/bin/nu", true)
            .unwrap()
            .1
            .starts_with("^$env."));
    }

    /// A temp `HOME` holding `profile` as `.profile`, and a stand-in for this binary's
    /// `print-shell-env` that prints the variables the test cares about.
    struct ShellFixture {
        dir: PathBuf,
        exe: PathBuf,
    }

    use std::path::PathBuf;

    impl ShellFixture {
        fn new(name: &str, profile: &str) -> Self {
            use std::os::unix::fs::PermissionsExt;
            let dir =
                std::env::temp_dir().join(format!("oriel-shell-env-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(".profile"), profile).unwrap();
            let exe = dir.join("fake-exe");
            std::fs::write(
                &exe,
                format!(
                    "#!/bin/sh\necho noise\nprintf %s \"${MARKER_VAR}\"\n\
                     env -u {MARKER_VAR} -u {EXE_VAR} -u {RESOLVING_VAR} -0\n\
                     printf %s \"${MARKER_VAR}\"\n"
                ),
            )
            .unwrap();
            std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
            Self { dir, exe }
        }

        fn opts(&self) -> Options {
            Options {
                interactive_timeout: Duration::from_secs(5),
                login_timeout: Duration::from_secs(5),
                base_env: [
                    ("HOME", self.dir.as_os_str()),
                    ("PATH", OsStr::new("/usr/bin:/bin")),
                    ("SSH_TTY", OsStr::new("/dev/pts/9")),
                ]
                .into_iter()
                .map(|(k, v)| (k.into(), v.to_owned()))
                .collect(),
            }
        }
    }

    impl Drop for ShellFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    use std::ffi::OsStr;

    fn get<'a>(vars: &'a EnvMap, key: &str) -> Option<&'a str> {
        vars.get(OsStr::new(key)).and_then(|v| v.to_str())
    }

    #[tokio::test]
    async fn resolves_profile_through_a_real_login_shell() {
        let fx = ShellFixture::new(
            "profile",
            "echo welcome\nexport SHELLENV_TEST=from-profile\nexport SHELLENV_BYTES=\"$(printf '\\377')\"\n",
        );
        let vars = resolve_with("/bin/sh", &fx.exe, &fx.opts()).await.unwrap();
        assert_eq!(get(&vars, "SHELLENV_TEST"), Some("from-profile"));
        assert_eq!(
            vars.get(OsStr::new("SHELLENV_BYTES")).map(|v| v.as_bytes()),
            Some(&[0xff][..])
        );
        assert!(!vars.contains_key(OsStr::new("SSH_TTY")));
    }

    #[tokio::test]
    async fn falls_back_to_login_only_when_interactive_startup_exits() {
        // Like an rc file that execs tmux or another shell when interactive.
        let fx = ShellFixture::new(
            "fallback",
            "export SHELLENV_TEST=login-only\ncase $- in *i*) exit 0;; esac\n",
        );
        let vars = resolve_with("/bin/sh", &fx.exe, &fx.opts()).await.unwrap();
        assert_eq!(get(&vars, "SHELLENV_TEST"), Some("login-only"));
    }

    #[tokio::test]
    async fn falls_back_to_login_only_when_interactive_startup_hangs() {
        let fx = ShellFixture::new(
            "hang",
            "export SHELLENV_TEST=after-hang\ncase $- in *i*) sleep 30;; esac\n",
        );
        let mut opts = fx.opts();
        opts.interactive_timeout = Duration::from_millis(300);
        let started = std::time::Instant::now();
        let vars = resolve_with("/bin/sh", &fx.exe, &opts).await.unwrap();
        assert_eq!(get(&vars, "SHELLENV_TEST"), Some("after-hang"));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Run by `shell_dies_with_the_server` in a child process, which it then kills.
    #[tokio::test]
    #[ignore = "helper for shell_dies_with_the_server"]
    async fn pdeathsig_helper() {
        // Does nothing under `--include-ignored`; only the parent test sets this.
        let Some(dir) = std::env::var_os("ORIEL_PDEATHSIG_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let opts = Options {
            interactive_timeout: Duration::from_secs(60),
            login_timeout: Duration::from_secs(60),
            base_env: [
                ("HOME", dir.as_os_str()),
                ("PATH", OsStr::new("/usr/bin:/bin")),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), v.to_owned()))
            .collect(),
        };
        let _ = resolve_with("/bin/sh", &dir.join("fake-exe"), &opts).await;
    }

    #[test]
    fn shell_dies_with_the_server() {
        let fx = ShellFixture::new("pdeathsig", "echo $$ > \"$HOME/shell.pid\"\nsleep 60\n");
        let pid_file = fx.dir.join("shell.pid");
        let mut helper = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "shell_env::tests::pdeathsig_helper"])
            .env("ORIEL_PDEATHSIG_DIR", &fx.dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let started = std::time::Instant::now();
        let pid = loop {
            if let Some(pid) = std::fs::read_to_string(&pid_file)
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
            {
                break pid;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "shell never started"
            );
            std::thread::sleep(Duration::from_millis(20));
        };
        // Like SIGKILL or an unhandled SIGINT/SIGTERM: no destructors run.
        helper.kill().unwrap();
        helper.wait().unwrap();

        let gone = |pid: i32| {
            std::fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
                stat.rsplit(')')
                    .next()
                    .is_some_and(|s| s.trim_start().starts_with('Z'))
            })
        };
        let started = std::time::Instant::now();
        while !gone(pid) && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let died = gone(pid);
        // The shell led its own process group; clean up what the profile started.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        assert!(died, "login shell {pid} outlived the server");
    }

    #[tokio::test]
    async fn reports_both_failures_with_a_profile_hint() {
        let fx = ShellFixture::new("fail", "exit 0\n");
        let err = resolve_with("/bin/sh", &fx.exe, &fx.opts())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("interactive:"), "{err}");
        assert!(err.contains("login only:"), "{err}");
        assert!(err.contains(RESOLVING_VAR), "{err}");
    }
}
