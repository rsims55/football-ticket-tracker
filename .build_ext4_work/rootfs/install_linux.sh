#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CFB Ticket Price Tracker"
PKG_NAME="cfb-tix"
INSTALL_BASE="${HOME}/.local/share/${PKG_NAME}"
VENV_DIR="${INSTALL_BASE}/venv"
APP_DIR="${INSTALL_BASE}/app"
ICON_DEST_BASE="${HOME}/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="${HOME}/.local/share/applications"

# source locations inside mounted image
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SRC="${SCRIPT_DIR}/app"
ICON_SRC="${SCRIPT_DIR}/assets/icons/cfb-tix.svg"

AUTOSTART=1
SYNC_TIME="${SYNC_TIME:-06:10}"

usage() {
  cat <<EOF
${APP_NAME} — Linux installer

Usage:
  ./install_linux.sh [--no-autostart] [--sync-time HH:MM]

Options:
  --no-autostart   Install without enabling login autostart (systemd --user)
  --sync-time      Daily time for CSV sync (default 06:10)
EOF
}

while (( "$#" )); do
  case "$1" in
    --no-autostart) AUTOSTART=0; shift ;;
    --sync-time) SYNC_TIME="${2:-06:10}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "${INSTALL_BASE}" "${DESKTOP_DIR}" "${ICON_DEST_BASE}"

# copy app payload
if [[ -d "${APP_SRC}" ]]; then
  rsync -a --delete "${APP_SRC}/" "${APP_DIR}/" \
    --exclude ".git" --exclude "packaging/dist" --exclude "dist" --exclude "build" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude "*.egg-info" || true
else
  echo "⚠️  Source app dir not found at ${APP_SRC}"
fi

# venv
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# editable install
if [[ -f "${APP_DIR}/pyproject.toml" || -f "${APP_DIR}/setup.py" ]]; then
  pushd "${APP_DIR}" >/dev/null
  "${VENV_DIR}/bin/pip" install -e .
  popd >/dev/null
else
  echo "❌ ${APP_DIR} missing pyproject.toml/setup.py"
  exit 1
fi

# autostart via CLI
if (( AUTOSTART == 1 )); then
  if "${VENV_DIR}/bin/cfb-tix" autostart --enable; then
    echo "✅ Autostart enabled."
  else
    echo "⚠️ Autostart not enabled (systemd --user unavailable?)."
  fi
else
  echo "ℹ️ Skipped autostart."
fi

# desktop launcher
if [[ -f "${ICON_SRC}" ]]; then
  install -m 0644 "${ICON_SRC}" "${ICON_DEST_BASE}/cfb-tix.svg"
fi
DESKTOP_FILE="${DESKTOP_DIR}/cfb-tix-gui.desktop"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=CFB Tickets (GUI)
Comment=Open the CFB Ticket Price Tracker GUI
Exec=${VENV_DIR}/bin/cfb-tix-gui
Icon=cfb-tix
Terminal=false
Categories=Utility;Education;
StartupNotify=true
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "${DESKTOP_DIR}" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f "${HOME}/.local/share/icons" || true

# CSV sync install + first-time pull
REPO_DIR="${APP_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python}"
PYTHON_BIN="${VENV_DIR}/bin/python"
RUN_TIME="${SYNC_TIME}"
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" RUN_TIME="$RUN_TIME" bash "${SCRIPT_DIR}/install_sync.sh" || true

# extra best-effort first pull
"${VENV_DIR}/bin/python" "${APP_DIR}/scripts/sync_snapshots.py" pull || true

echo "✅ ${APP_NAME} installed."
echo "• App dir: ${APP_DIR}"
echo "• Venv:    ${VENV_DIR}"
echo "• CSV Sync: Daily at ${SYNC_TIME} (systemd --user)"
echo "   Token file for uploads: \$HOME/.config/cfb-tix/env  (set GH_TOKEN there if not prompted)"
