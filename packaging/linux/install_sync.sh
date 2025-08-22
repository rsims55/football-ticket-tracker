#!/usr/bin/env bash
# packaging/linux/build_ext4.sh
# Build a self-contained ext4 image with:
#  - /app         → your repo (clean copy, excludes junk)
#  - /assets/...  → linux icon
#  - /install_linux.sh → idempotent installer (venv + autostart + desktop launcher)
#  - /install_sync.sh  → installs daily CSV sync timer + first-time pull
#
# Output: dist/cfb-tix.ext4
set -euo pipefail
# === Prompt for GH token if not already set ===
ENV_FILE="$HOME/.config/cfb-tix/env"
mkdir -p "$(dirname "$ENV_FILE")"

if [[ ! -s "$ENV_FILE" ]]; then
  echo "🔑 No GitHub token found."
  read -r -s -p "Paste your GitHub access token (leave blank to skip): " GH_TOKEN_INPUT
  echo
  if [[ -n "${GH_TOKEN_INPUT:-}" ]]; then
    printf 'GH_TOKEN=%s\n' "$GH_TOKEN_INPUT" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "✅ Saved token to $ENV_FILE"
  else
    printf 'GH_TOKEN=\n' > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "⚠️ Skipping token setup. Uploads will be disabled."
  fi
fi

#---- repo roots ----#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

# --- write CSV sync installer (runs as user via systemd --user) ---
cat > "${STAGE}/install_sync.sh" <<'EOSH'
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/.local/share/cfb-tix/app}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
RUN_TIME="${RUN_TIME:-07:00}"  # local time

SYSTEMD_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="cfb-tix-sync"
SERVICE_FILE="$SYSTEMD_DIR/${SERVICE_NAME}.service"
TIMER_FILE="$SYSTEMD_DIR/${SERVICE_NAME}.timer"

mkdir -p "$SYSTEMD_DIR"

# Write service
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Upload merged price_snapshots.csv to GitHub Release

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
# GH_TOKEN can be read from $REPO_DIR/.env by scripts/sync_snapshots.py
ExecStart=$PYTHON_BIN scripts/sync_snapshots.py pull_push
EOF

# Write timer (daily at RUN_TIME)
cat > "$TIMER_FILE" <<EOF
[Unit]
Description=Run cfb-tix CSV sync every morning

[Timer]
OnCalendar=*-*-* ${RUN_TIME}
Persistent=true
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}.timer" || true

# First-time pull now (download only; don't fail install if it errors)
"$PYTHON_BIN" scripts/sync_snapshots.py pull || true

echo "✅ Installed user timer ${SERVICE_NAME}.timer at ${RUN_TIME} daily."
echo "   Repo: $REPO_DIR"
echo "   Tip: put GH_TOKEN=... in $REPO_DIR/.env to enable uploads."
EOSH
chmod +x "${STAGE}/install_sync.sh"

# --- write main installer (venv + app + autostart + desktop + call sync installer) ---
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

# --- CSV SYNC: install timer + first-time pull ---
REPO_DIR="${APP_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
RUN_TIME="${SYNC_TIME}"

if bash "${SCRIPT_DIR}/install_sync.sh"; then
  echo "✅ CSV daily sync installed."
else
  echo "⚠️ CSV sync installer failed; you can run it later with:"
  echo "   REPO_DIR=\"${REPO_DIR}\" PYTHON_BIN=\"${PYTHON_BIN}\" RUN_TIME=\"${RUN_TIME}\" bash \"${SCRIPT_DIR}/install_sync.sh\""
fi

# One extra belt-and-suspenders pull via venv python (best-effort)
"${VENV_DIR}/bin/python" "${APP_DIR}/scripts/sync_snapshots.py" pull || true

echo "✅ ${APP_NAME} installed."
echo "• App dir: ${APP_DIR}"
echo "• Venv:    ${VENV_DIR}"
echo "• GUI:     Applications → 'CFB Tickets (GUI)'"
echo "• Autostart: $( ((AUTOSTART==1)) && echo Enabled || echo Skipped )"
echo "• CSV Sync: Daily at ${SYNC_TIME} (systemd --user)"
echo "   Tip: Put GH_TOKEN in ${APP_DIR}/.env to enable uploads."
EOSH
chmod +x "${STAGE}/install_linux.sh"

#---- size the image (repo size + slack) ----#
BYTES_USED=$(du -sb "${STAGE}" | awk '{print $1}')
SLACK=$((64 * 1024 * 1024))              # 64MB minimum slack
TARGET_SIZE=$(( BYTES_USED + BYTES_USED/2 + SLACK ))  # +60% overhead

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
