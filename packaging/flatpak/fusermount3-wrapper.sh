#!/bin/sh
# Installed as /app/bin/fusermount3. A FUSE mount made inside the sandbox would stay in
# its private mount namespace, invisible to host apps such as the file manager, so run the
# host's fusermount3 instead.
#
# Inside the sandbox `$XDG_RUNTIME_DIR/app/$FLATPAK_ID` resolves to
# `/run/flatpak/app/$FLATPAK_ID`, a bind mount of the host's `$XDG_RUNTIME_DIR/app/$FLATPAK_ID`;
# translate canonicalized paths back to the host's.
host_app_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/app/"
for arg do
    shift
    case "$arg" in
        /run/flatpak/app/*) arg="${host_app_dir}${arg#/run/flatpak/app/}" ;;
    esac
    set -- "$@" "$arg"
done
# On mount, the caller passes the socket that receives the /dev/fuse fd in $_FUSE_COMMFD;
# forward that fd to the host process.
if [ -n "${_FUSE_COMMFD:-}" ]; then
    exec flatpak-spawn --host --forward-fd="${_FUSE_COMMFD}" \
        --env=_FUSE_COMMFD="${_FUSE_COMMFD}" fusermount3 "$@"
fi
exec flatpak-spawn --host fusermount3 "$@"
