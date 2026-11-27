//! A `FLATPAK_BWRAP` wrapper that shows each Flatpak app its own documents at the session
//! document portal's mount path.
//!
//! The session runs its own xdg-document-portal, mounted at `<session runtime dir>/doc`.
//! The FileTransfer portal hands sandboxed apps paths under that mount
//! (`<mount>/<doc id>/<name>`), but Flatpak only binds the app's view of it
//! (`<mount>/by-app/<app id>`) at `/run/flatpak/doc`, so those paths don't exist in the
//! sandbox. Exposing the mount itself (`--filesystem=<mount>`) would let every app read
//! every document in the store: the mount's top level does not check who reads.
//!
//! Flatpak runs its sandboxes through the program in `FLATPAK_BWRAP`. Ours is a symlink to
//! this binary: it finds Flatpak's `--bind <mount>/by-app/<app id> /run/flatpak/doc` in
//! the bundled arguments and binds the same view at `<mount>` too, then runs the real
//! bwrap. Calls without such a bind (Flatpak's D-Bus proxy) and apps of a normal desktop
//! session (whose mount is already `/run/user/<uid>/doc`) pass through unchanged.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

/// File name prefix of the wrapper symlink; `main` dispatches on it.
const WRAPPER_PREFIX: &str = "oriel-flatpak-bwrap";

/// The bwrap to run when `FLATPAK_BWRAP` was set before we replaced it.
const REAL_BWRAP_ENV: &str = "ORIEL_REAL_BWRAP";

/// Where Flatpak binds the app's view of the document portal.
const SANDBOX_DOC: &[u8] = b"/run/flatpak/doc";

/// Run as the wrapper when started through its symlink; returns otherwise.
pub fn run_if_wrapper() {
    let mut args = std::env::args_os();
    let is_wrapper = args
        .next()
        .as_deref()
        .and_then(|arg0| Path::new(arg0).file_name())
        .is_some_and(|name| name.as_bytes().starts_with(WRAPPER_PREFIX.as_bytes()));
    if !is_wrapper {
        return;
    }
    let args: Vec<OsString> = args.collect();
    let real = real_bwrap();
    let err = std::process::Command::new(&real)
        .args(wrapped_args(args, read_bundled_args))
        .exec();
    eprintln!("{WRAPPER_PREFIX}: cannot run {}: {err}", real.display());
    std::process::exit(127);
}

/// The bwrap Flatpak would run without us.
fn real_bwrap() -> PathBuf {
    std::env::var_os(REAL_BWRAP_ENV)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // Distributions that build Flatpak with its bundled bwrap install it here.
            let bundled = Path::new("/usr/libexec/flatpak-bwrap");
            if bundled.exists() {
                bundled.into()
            } else {
                PathBuf::from("bwrap")
            }
        })
}

/// The contents of the `--args` file descriptor `fd`, read through `/proc` so the
/// descriptor's offset stays where bwrap expects it.
fn read_bundled_args(fd: &OsStr) -> Option<Vec<u8>> {
    let fd: u32 = fd.to_str()?.parse().ok()?;
    std::fs::read(format!("/proc/self/fd/{fd}")).ok()
}

/// `args` with the app's document view also bound at the portal's mount path, right after
/// Flatpak's bundled `--args <fd>` (so it lands on top of the sandbox's runtime directory).
fn wrapped_args(
    args: Vec<OsString>,
    read: impl FnOnce(&OsStr) -> Option<Vec<u8>>,
) -> Vec<OsString> {
    let bind = match args.as_slice() {
        [flag, fd, ..] if flag == "--args" => {
            read(fd).and_then(|bundled| doc_view_bind(&bundled, unsafe { libc::getuid() }))
        }
        _ => None,
    };
    let Some((view, mount)) = bind else {
        return args;
    };
    let mut out = Vec::with_capacity(args.len() + 3);
    let mut args = args.into_iter();
    out.extend(args.by_ref().take(2));
    out.extend([OsString::from("--bind"), view, mount]);
    out.extend(args);
    out
}

/// Flatpak's bind of the app's document view (`--bind <mount>/by-app/<app id>
/// /run/flatpak/doc`) in its bundled, NUL-terminated arguments: `(view, mount)`, unless the
/// mount is the sandbox's own `/run/user/<uid>/doc` (a normal desktop session).
fn doc_view_bind(bundled: &[u8], uid: u32) -> Option<(OsString, OsString)> {
    let args: Vec<&[u8]> = bundled.split(|&b| b == 0).collect();
    let view = args
        .windows(3)
        .find(|w| w[0] == b"--bind" && w[2] == SANDBOX_DOC)
        .map(|w| Path::new(OsStr::from_bytes(w[1])))?;
    let by_app = view.parent()?;
    if by_app.file_name()? != "by-app" || view.file_name().is_none() {
        return None;
    }
    let mount = by_app.parent()?;
    if !mount.is_absolute() || mount == Path::new(&format!("/run/user/{uid}/doc")) {
        return None;
    }
    Some((view.into(), mount.into()))
}

/// Point Flatpak at the wrapper for this process and everything it starts (set before the
/// session's private bus starts, so D-Bus activated apps get it too). The wrapper is a
/// symlink to this binary in `host_runtime_dir`, outside the session directory: processes
/// that outlive the session keep `FLATPAK_BWRAP`, and it must still run. `false` when it
/// cannot be set up; sandboxed apps then can't open files passed through the portal.
pub fn install(host_runtime_dir: &Path) -> bool {
    match try_install(host_runtime_dir) {
        Ok(()) => true,
        Err(err) => {
            tracing::warn!(
                error = format!("{err:#}"),
                "Flatpak bwrap wrapper unavailable; Flatpak apps cannot open pasted or dropped files"
            );
            false
        }
    }
}

fn try_install(host_runtime_dir: &Path) -> anyhow::Result<()> {
    use anyhow::Context;
    let exe = std::env::current_exe().context("cannot find this executable")?;
    // One link per binary: replacing another binary's link could point running sessions
    // at a build without the wrapper.
    let hash = exe
        .as_os_str()
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
    let link = host_runtime_dir.join(format!("{WRAPPER_PREFIX}-{hash:016x}"));
    if std::fs::read_link(&link).ok().as_deref() != Some(exe.as_path()) {
        let tmp = host_runtime_dir.join(format!("{WRAPPER_PREFIX}.tmp-{}", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        std::os::unix::fs::symlink(&exe, &tmp).context("cannot create the wrapper link")?;
        std::fs::rename(&tmp, &link).context("cannot place the wrapper link")?;
    }

    // An inherited wrapper of ours (another server's session) is not the real bwrap.
    if let Some(inherited) = std::env::var_os("FLATPAK_BWRAP").filter(|p| !p.is_empty()) {
        let ours = Path::new(&inherited)
            .file_name()
            .is_some_and(|name| name.as_bytes().starts_with(WRAPPER_PREFIX.as_bytes()));
        if !ours {
            std::env::set_var(REAL_BWRAP_ENV, &inherited);
        }
    }

    // Make sure it runs (and finds the real bwrap) before every Flatpak launch depends on it.
    let status = std::process::Command::new(&link)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .context("cannot run the wrapper")?;
    anyhow::ensure!(status.success(), "`{} --version` failed", link.display());
    std::env::set_var("FLATPAK_BWRAP", &link);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(args: &[&str]) -> Vec<u8> {
        args.iter()
            .flat_map(|a| [a.as_bytes(), b"\0"].concat())
            .collect()
    }

    #[test]
    fn finds_the_app_view_of_a_session_portal() {
        let bundled = bundle(&[
            "--bind",
            "/run/user/7/.flatpak/org.x.Y/xdg-run",
            "/run/user/7",
            "--bind",
            "/run/user/7/oriel-session-9/doc/by-app/org.x.Y",
            "/run/flatpak/doc",
            "--unshare-net",
        ]);
        assert_eq!(
            doc_view_bind(&bundled, 7),
            Some((
                "/run/user/7/oriel-session-9/doc/by-app/org.x.Y".into(),
                "/run/user/7/oriel-session-9/doc".into()
            ))
        );
        let args = |v: &[&str]| v.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(
            wrapped_args(args(&["--args", "78", "--", "app"]), |fd| {
                assert_eq!(fd, "78");
                Some(bundled.clone())
            }),
            args(&[
                "--args",
                "78",
                "--bind",
                "/run/user/7/oriel-session-9/doc/by-app/org.x.Y",
                "/run/user/7/oriel-session-9/doc",
                "--",
                "app"
            ])
        );
    }

    #[test]
    fn leaves_other_sandboxes_alone() {
        // A normal desktop session: the sandbox already shows the view at that path.
        let desktop = bundle(&[
            "--bind",
            "/run/user/7/doc/by-app/org.x.Y",
            "/run/flatpak/doc",
        ]);
        assert_eq!(doc_view_bind(&desktop, 7), None);
        // Flatpak's D-Bus proxy, and binds that are not an app view.
        assert_eq!(doc_view_bind(&bundle(&["--ro-bind", "/a", "/a"]), 7), None);
        let odd = bundle(&["--bind", "/s/doc/org.x.Y", "/run/flatpak/doc"]);
        assert_eq!(doc_view_bind(&odd, 7), None);
        let args = vec![OsString::from("--version")];
        assert_eq!(wrapped_args(args.clone(), |_| unreachable!()), args);
        let args = vec![OsString::from("--args"), OsString::from("5")];
        assert_eq!(wrapped_args(args.clone(), |_| None), args);
    }
}
