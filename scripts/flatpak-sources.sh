#!/usr/bin/env bash
# Regenerate packaging/flatpak/cargo-sources.json from Cargo.lock (needs network).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/target/flatpak-venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --quiet aiohttp tomlkit
fi

"${VENV}/bin/python" "${REPO_ROOT}/packaging/flatpak/flatpak-cargo-generator.py" \
    "${REPO_ROOT}/Cargo.lock" \
    -o "${REPO_ROOT}/packaging/flatpak/cargo-sources.json"

echo "==> Wrote packaging/flatpak/cargo-sources.json"
