#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CFB Ticket Price Tracker"
PKG_NAME="cfb-tix"
INSTALL_BASE="${HOME}/.local/share/${PKG_NAME}"
VENV_DIR="${INSTALL_BASE}/venv"
APP_DIR="${INSTALL_BASE}/app"
ICON_DEST_BASE="${HOME}/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="${HOME}/.local/share/applications"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_NAME="${PKG_NAME}.service"

# --- IMPORTANT: copy from the mounted image's own dir ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SRC="${SCRIPT_DIR}/app"
ICON_SRC="${SCRIPT_DIR}/assets/icons/cfb-tix.svg"

mkdir -p "${INSTALL_BASE}" "${ICON_DEST_BASE}" "${DESKTOP_DIR}" "${SYSTEMD_USER_DIR}"

# Copy/refresh app to user space (no root needed)
if [[ -d "${APP_SRC}" ]]; then
  rsync -a --delete "${APP_SRC}/" "${APP_DIR}/" \
    --exclude ".git" --exclude "dist" --exclude "build" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude "*.egg-info" || true
else
  echo "⚠️  Source app directory not found at ${APP_SRC} — continuing (maybe already installed)."
fi

# venv
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
# upgrade pip, wheel, setuptools (quiet)
"${VENV_DIR}/bin/python" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# editable install (works for src-layout if pyproject.toml present)
if [[ -f "${APP_DIR}/pyproject.toml" || -f "${APP_DIR}/setup.py" ]]; then
  pushd "${APP_DIR}" >/dev/null
  "${VENV_DIR}/bin/pip" install -e .
  popd >/dev/null
else
  echo "❌ ${APP_DIR} does not contain pyproject.toml or setup.py"
  echo "   Make sure the image includes your repo under /app."
  exit 1
fi

# systemd user service for headless daemon
UNIT_PATH="${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
cat > "${UNIT_PATH}" <<EOF
[Unit]
Description=${APP_NAME} background scheduler (user)
After=default.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=${VENV_DIR}/bin/python -m cfb_tix --no-gui
WorkingDirectory=${INSTALL_BASE}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload || true
systemctl --user enable "${SERVICE_NAME}" || true
systemctl --user start "${SERVICE_NAME}" || true

# Desktop launcher for GUI
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

# Try to refresh caches if available (best-effort)
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "${DESKTOP_DIR}" || true
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f "${HOME}/.local/share/icons" || true
fi

echo "✅ ${APP_NAME} installed."
echo "• Background service: ${SERVICE_NAME} (systemd --user)"
echo "• GUI launcher: Applications menu → 'CFB Tickets (GUI)'"
echo "• Venv: ${VENV_DIR}"
