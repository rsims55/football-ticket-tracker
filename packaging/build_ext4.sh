#!/usr/bin/env bash
# packaging/build_ext4.sh
# Build a self-contained ext4 image with:
#  - /app         → your repo (clean copy, excludes junk)
#  - /assets/...  → linux icon
#  - /install_linux.sh → idempotent installer (venv + autostart toggle + desktop launcher)
#
# Output: dist/cfb-tix.ext4
set -euo pipefail

#---- repo roots ----#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
WORK_DIR="${REPO_ROOT}/.build_ext4_work"
IMG_PATH="${DIST_DIR}/cfb-tix.ext4"

#---- clean work ----#
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${DIST_DIR}"

#---- stage tree that becomes image root ----#
STAGE="${WORK_DIR}/rootfs"
mkdir -p "${STAGE}/app" "${STAGE}/assets/icons" "${STAGE}"

# rsync the repo into /app with excludes
rsync -a "${REPO_ROOT}/" "${STAGE}/app/" \
  --delete \
  --exclude ".git" \
  --exclude ".github" \
  --exclude ".gitignore" \
  --exclude "dist" \
  --exclude "build" \
  --exclude "*.egg-info" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".venv" \
  --exclude ".mypy_cache" \
  --exclude ".pytest_cache" \
  --exclude ".DS_Store"

# copy Linux icon
if [[ -f "${REPO_ROOT}/assets/icons/cfb-tix.svg" ]]; then
  install -m 0644 "${REPO_ROOT}/assets/icons/cfb-tix.svg" "${STAGE}/assets/icons/cfb-tix.svg"
fi

# write installer script into image root (uses relative paths from mount point)
cat > "${STAGE}/install_linux.sh" <<'EOSH'
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CFB Ticket Price Tracker"
PKG_NAME="cfb-tix"
INSTALL_BASE="${HOME}/.local/share/${PKG_NAME}"
VENV_DIR="${INSTALL_BASE}/venv"
APP_DIR="${INSTALL_BASE}/app"
ICON_DEST_BASE="${HOME}/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="${HOME}/.local/share/applications"

# --- source locations inside mounted image ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SRC="${SCRIPT_DIR}/app"
ICON_SRC="${SCRIPT_DIR}/assets/icons/cfb-tix.svg"

# --- flags ---
AUTOSTART=1
usage() {
  cat <<EOF
${APP_NAME} — Linux installer

Usage:
  ./install_linux.sh [--no-autostart]

Options:
  --no-autostart   Install without enabling login autostart (systemd --user)
EOF
}

for arg in "${@:-}"; do
  case "$arg" in
    --no-autostart) AUTOSTART=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 2 ;;
  esac
done

# --- ensure dirs ---
mkdir -p "${INSTALL_BASE}" "${DESKTOP_DIR}" "${ICON_DEST_BASE}"

# --- copy app payload ---
if [[ -d "${APP_SRC}" ]]; then
  rsync -a --delete "${APP_SRC}/" "${APP_DIR}/" \
    --exclude ".git" --exclude "dist" --exclude "build" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude "*.egg-info" || true
else
  echo "⚠️  Source app directory not found at ${APP_SRC} — continuing (maybe already installed)."
fi

# --- venv ---
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# --- editable install of the app ---
if [[ -f "${APP_DIR}/pyproject.toml" || -f "${APP_DIR}/setup.py" ]]; then
  pushd "${APP_DIR}" >/dev/null
  "${VENV_DIR}/bin/pip" install -e .
  popd >/dev/null
else
  echo "❌ ${APP_DIR} does not contain pyproject.toml or setup.py"
  echo "   Make sure the image includes your repo under /app."
  exit 1
fi

# --- autostart (systemd --user) via app CLI; default ON, opt-out with --no-autostart ---
if (( AUTOSTART == 1 )); then
  if "${VENV_DIR}/bin/cfb-tix" autostart --enable; then
    echo "✅ Autostart enabled (systemd --user)."
  else
    echo "⚠️ Autostart not enabled (systemd --user unavailable?). You can run:"
    echo "   ${VENV_DIR}/bin/cfb-tix autostart --enable"
  fi
else
  echo "ℹ️ Skipped autostart (use later: ${VENV_DIR}/bin/cfb-tix autostart --enable)"
fi

# --- desktop launcher for GUI ---
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
echo "• App dir: ${APP_DIR}"
echo "• Venv:    ${VENV_DIR}"
echo "• GUI:     Applications → 'CFB Tickets (GUI)'"
echo "• Autostart: $( ((AUTOSTART==1)) && echo Enabled || echo Skipped )"
EOSH
chmod +x "${STAGE}/install_linux.sh"

#---- size the image (repo size + slack) ----#
BYTES_USED=$(du -sb "${STAGE}" | awk '{print $1}')
# add 60% overhead + 64MB minimum slack
SLACK=$((64 * 1024 * 1024))
TARGET_SIZE=$(( BYTES_USED + BYTES_USED/2 + SLACK ))

# round up to nearest 64MB
ROUND=$((64 * 1024 * 1024))
TARGET_SIZE=$(( ( (TARGET_SIZE + ROUND - 1) / ROUND ) * ROUND ))

#---- create ext4 image ----#
rm -f "${IMG_PATH}"
dd if=/dev/zero of="${IMG_PATH}" bs=1 count=0 seek="${TARGET_SIZE}"
mkfs.ext4 -F -L "cfb-tix" "${IMG_PATH}"

#---- mount, copy files, unmount ----#
MNT="${WORK_DIR}/mnt"
mkdir -p "${MNT}"
sudo mount -o loop "${IMG_PATH}" "${MNT}"
sudo rsync -a "${STAGE}/" "${MNT}/"
sync
sudo umount "${MNT}"

echo "🎉 Built ${IMG_PATH}"
