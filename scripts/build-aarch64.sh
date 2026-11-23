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

Build RemoteWayland for aarch64-unknown-linux-gnu inside an arm64 Ubuntu 24.04 container.

Options:
  --gtk4             Build client with GTK4/libadwaita launcher (default)
  --no-gui           Build client without GUI launcher (--no-default-features)
  --client-only      Build only remotewayland-client
  --server-only      Build only remotewayland-server
  --debug            Build debug profile instead of release
  --rebuild-image    Rebuild the selected arm64 builder container image
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

IMAGE_NAME="localhost/remotewayland-aarch64-builder:latest"
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

if [[ "${REBUILD_IMAGE}" -eq 1 ]] || ! "${RUNTIME[@]}" image exists "${IMAGE_NAME}" 2>/dev/null; then
    echo "==> Building container image ${IMAGE_NAME}..."
    "${RUNTIME[@]}" build --arch arm64 -t "${IMAGE_NAME}" -f "${CONTAINERFILE}" "${REPO_ROOT}/scripts"
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

run_cargo() {
    "${RUNTIME[@]}" run --rm --arch arm64 \
        -v "${REPO_ROOT}:/work" \
        "${CARGO_MOUNT[@]}" \
        -w /work \
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
