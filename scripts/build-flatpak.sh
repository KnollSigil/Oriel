#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/packaging/flatpak/dev.knoll.Oriel.yml"
APP_ID="dev.knoll.Oriel"
ARCH="$(uname -m)"
INSTALL=0
BUNDLE=1
INSTALL_DEPS=0

usage() {
    cat <<'EOF'
Usage: scripts/build-flatpak.sh [OPTIONS]

Build the Oriel (RemoteWayland client, GTK4/libadwaita) Flatpak.

Options:
  --arch <x86_64|aarch64>  Target architecture (default: host). Non-host arches
                           need qemu-user binfmt (e.g. qemu-user-static) and are slow.
  --install                Also install the result for the current user
  --no-bundle              Skip creating the single-file .flatpak bundle
  --install-deps           Install required runtimes/SDKs for --arch (user scope, needs network)
  -h, --help               Show this help message

Output: build/flatpak/<app-id>-<arch>.flatpak
Cargo artifacts are cached in build/flatpak/cargo-target-<arch>/ (delete to force a clean build).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --install) INSTALL=1; shift ;;
        --no-bundle) BUNDLE=0; shift ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "${ARCH}" in
    x86_64|aarch64) ;;
    arm64) ARCH="aarch64" ;;
    *) echo "Unsupported arch: ${ARCH}" >&2; exit 1 ;;
esac

# cargo-sources.json is a generated artifact (git-ignored); rebuild it when missing or stale.
if [[ ! -f "${REPO_ROOT}/packaging/flatpak/cargo-sources.json" || "${REPO_ROOT}/Cargo.lock" -nt "${REPO_ROOT}/packaging/flatpak/cargo-sources.json" ]]; then
    "${REPO_ROOT}/scripts/flatpak-sources.sh"
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
    for ref in org.gnome.Platform//51 org.gnome.Sdk//51 \
               org.freedesktop.Sdk.Extension.rust-stable//26.08 \
               org.freedesktop.Sdk.Extension.llvm22//26.08; do
        flatpak install --user -y --noninteractive --arch="${ARCH}" flathub "${ref}"
    done
fi

BUILD_ROOT="${REPO_ROOT}/build/flatpak"
mkdir -p "${BUILD_ROOT}"

# Persistent per-arch Cargo target dir. flatpak-builder only caches a module when
# none of its inputs change, so any source edit would otherwise recompile every
# dependency. We expose a host dir to the build sandbox and point CARGO_TARGET_DIR
# at it so only changed crates are rebuilt. (Local/CI use only; not Flathub-valid.)
CARGO_CACHE="${BUILD_ROOT}/cargo-target-${ARCH}"
mkdir -p "${CARGO_CACHE}"
GEN_MANIFEST="${REPO_ROOT}/packaging/flatpak/.generated-${ARCH}.yml"
python3 - "${MANIFEST}" "${GEN_MANIFEST}" "${CARGO_CACHE}" <<'PY'
import sys
src, dst, cache = sys.argv[1:4]
s = open(src).read()
s = s.replace("build-options:\n  append-path:",
              f"build-options:\n  build-args:\n    - --filesystem={cache}\n  append-path:", 1)
s = s.replace("    CARGO_HOME:", f"    CARGO_TARGET_DIR: {cache}\n    CARGO_HOME:", 1)
open(dst, "w").write(s)
PY
MANIFEST="${GEN_MANIFEST}"

echo "==> Building ${APP_ID} for ${ARCH}..."
flatpak-builder \
    --arch="${ARCH}" \
    --force-clean \
    --delete-build-dirs \
    --user \
    --disable-rofiles-fuse \
    --state-dir="${BUILD_ROOT}/state-${ARCH}" \
    --repo="${BUILD_ROOT}/repo" \
    $([[ "${INSTALL}" -eq 1 ]] && echo --install) \
    "${BUILD_ROOT}/build-${ARCH}" \
    "${MANIFEST}"

if [[ "${BUNDLE}" -eq 1 ]]; then
    OUT="${BUILD_ROOT}/${APP_ID}-${ARCH}.flatpak"
    flatpak build-bundle --arch="${ARCH}" "${BUILD_ROOT}/repo" "${OUT}" "${APP_ID}"
    echo "==> Bundle: ${OUT}"
fi
