#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GUI="gtk4"
TARGET_SCOPE="workspace"
PROFILE="release"
REBUILD_IMAGE=0

usage() {
    cat <<'EOF'
Usage: scripts/build-aarch64.sh [OPTIONS]

Cross-compile RemoteWayland for aarch64-unknown-linux-gnu with the Rust/GCC aarch64
cross toolchain inside a native (x86_64) Ubuntu 24.04 container. No qemu emulation.

Options:
  --gtk4             Build client with GTK4/libadwaita launcher (default)
  --no-gui           Build client without GUI launcher (--no-default-features)
  --client-only      Build only remotewayland-client
  --server-only      Build only remotewayland-server
  --debug            Build debug profile instead of release
  --rebuild-image    Rebuild the cross-compilation builder container image
  -h, --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gtk4) GUI="gtk4"; shift ;;
        --no-gui) GUI="none"; shift ;;
        --client-only) TARGET_SCOPE="client"; shift ;;
        --server-only) TARGET_SCOPE="server"; shift ;;
        --debug) PROFILE="debug"; shift ;;
        --rebuild-image) REBUILD_IMAGE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

IMAGE_NAME="localhost/remotewayland-aarch64-cross-builder:latest"
CONTAINERFILE="${REPO_ROOT}/scripts/Containerfile.aarch64"
TARGET_DIR="${REPO_ROOT}/target"

if command -v podman >/dev/null 2>&1; then
    RUNTIME=(podman --cgroup-manager=cgroupfs)
elif command -v docker >/dev/null 2>&1; then
    RUNTIME=(docker)
else
    echo "Error: neither podman nor docker is installed." >&2
    exit 1
fi

# Rebuild the image whenever the Containerfile changes, so new build
# dependencies are picked up without needing --rebuild-image.
CONTAINERFILE_HASH="$(sha256sum "${CONTAINERFILE}" | cut -d' ' -f1)"
IMAGE_HASH="$("${RUNTIME[@]}" image inspect --format '{{ index .Config.Labels "containerfile-sha256" }}' "${IMAGE_NAME}" 2>/dev/null || true)"
if [[ -n "${IMAGE_HASH}" && "${IMAGE_HASH}" != "${CONTAINERFILE_HASH}" ]]; then
    echo "==> ${CONTAINERFILE##*/} changed since the image was built; rebuilding."
    REBUILD_IMAGE=1
fi

if [[ "${REBUILD_IMAGE}" -eq 1 ]] || [[ -z "${IMAGE_HASH}" ]]; then
    echo "==> Building container image ${IMAGE_NAME}..."
    "${RUNTIME[@]}" build --arch amd64 --label "containerfile-sha256=${CONTAINERFILE_HASH}" \
        -t "${IMAGE_NAME}" -f "${CONTAINERFILE}" "${REPO_ROOT}/scripts"
fi

PROFILE_FLAG=()
if [[ "${PROFILE}" == "release" ]]; then
    PROFILE_FLAG=(--release)
fi

CLIENT_FEAT_FLAGS=()
case "${GUI}" in
    gtk4) CLIENT_FEAT_FLAGS=() ;;
    none) CLIENT_FEAT_FLAGS=(--no-default-features) ;;
esac

CARGO_MOUNT=()
if [[ -d "${HOME}/.cargo/registry" ]]; then
    CARGO_MOUNT=(-v "${HOME}/.cargo/registry:/root/.cargo/registry")
fi

CONTAINER_TARGET_DIR="/work/${TARGET_DIR#${REPO_ROOT}/}"
# Keep intermediates (build scripts, proc-macros, deps) out of target/release:
# host builds use the same unit hashes, so sharing it makes native and
# container builds clobber each other and rebuild from scratch every time.
CONTAINER_BUILD_DIR="${CONTAINER_TARGET_DIR}/aarch64-builder"

run_cargo() {
    "${RUNTIME[@]}" run --rm --arch amd64 \
        -v "${REPO_ROOT}:/work" \
        "${CARGO_MOUNT[@]}" \
        -w /work \
        -e CARGO_BUILD_BUILD_DIR="${CONTAINER_BUILD_DIR}" \
        "${IMAGE_NAME}" \
        cargo "$@" --target-dir "${CONTAINER_TARGET_DIR}"
}

if [[ "${TARGET_SCOPE}" == "workspace" && "${GUI}" == "gtk4" ]]; then
    echo "==> Building workspace (gtk4, ${PROFILE}) for aarch64-unknown-linux-gnu..."
    run_cargo build "${PROFILE_FLAG[@]}" --workspace --target aarch64-unknown-linux-gnu
else
    if [[ "${TARGET_SCOPE}" != "server" ]]; then
        echo "==> Building remotewayland-client (gui=${GUI}, ${PROFILE}) for aarch64-unknown-linux-gnu..."
        run_cargo build "${PROFILE_FLAG[@]}" -p remotewayland-client "${CLIENT_FEAT_FLAGS[@]}" --target aarch64-unknown-linux-gnu
    fi
    if [[ "${TARGET_SCOPE}" != "client" ]]; then
        echo "==> Building remotewayland-server (${PROFILE}) for aarch64-unknown-linux-gnu..."
        run_cargo build "${PROFILE_FLAG[@]}" -p remotewayland-server --target aarch64-unknown-linux-gnu
    fi
fi

echo "==> Done. Output binaries in: ${TARGET_DIR}/aarch64-unknown-linux-gnu/${PROFILE}/"
