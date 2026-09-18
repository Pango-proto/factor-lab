#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"

mkdir -p "${UV_CACHE_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"

echo "[bootstrap] syncing python deps into project .venv (locked + dev extras)..."
UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv" \
  uv sync --locked --extra dev

echo "[bootstrap] installing node deps..."
if [ -f "${ROOT_DIR}/package-lock.json" ]; then
  (cd "${ROOT_DIR}" && npm ci)
else
  (cd "${ROOT_DIR}" && npm install)
fi

echo "[bootstrap] done"
echo "[bootstrap] quick check:"
echo "  UV_PROJECT_ENVIRONMENT=${ROOT_DIR}/.venv"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"
