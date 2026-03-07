#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  echo "venv not found. Run bin/setup_linux.sh first." >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR/src"
exec python -m cfb_tix run --no-gui
