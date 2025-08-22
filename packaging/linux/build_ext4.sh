#!/usr/bin/env bash
# packaging/linux/build_ext4.sh
# Builds a self-contained ext4 image with:
#  - /app                → your repo (clean copy, excludes junk)
#  - /assets/...         → linux icon (if present)
#  - /install_linux.sh   → idempotent app installer (venv + autostart + .desktop)
#  - /install_sync.sh    → installs daily CSV sync timer + does first-time pull (asks for GH token)
#
# Output: packaging/dist/cfb-tix.ext4

set -euo pipefail

#---- repo roots ----#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${REPO_ROOT}/packaging/dist"
WORK_DIR="${REPO_ROOT}/.build_ext4_work"
IMG_PATH="${DIST_DIR}/cfb-tix.ext4"

#---- clean work ----#
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${DIST_DIR}"

#---- stage tree that becomes image root ----#
STAGE="${WORK_DIR}/rootfs"
mkdir -p "${STAGE}/app" "${STAGE}/assets/icons"

# rsync the repo into /app with excludes
rsync -a "${REPO_ROOT}/" "${STAGE}/app/" \
  --delete \
  --exclude ".git" \
  --exclude ".github" \
  --exclude ".gitignore" \
  --exclude "packaging/dist" \
  --exclude ".build_ext4_work" \
  --exclude "dist" \
  --exclude "build" \
  --exclude "*.egg-info" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".venv" \
  --exclude ".mypy_cache" \
  --exclude ".pytest_cache" \
  --exclude ".DS_Store"

# copy Linux icon if present
if [[ -f "${REPO_ROOT}/assets/icons/cfb-tix.svg" ]]; then
  install -m 0644 "${REPO_ROOT}/assets/icons/cfb-tix.svg" "${STAGE}/assets/icons/cfb-tix.svg"
fi

# --- CSV sync installer (user systemd timer + first-time pull + GH token prompt) ---
cat > "${STAGE}/install_sync.sh" <<'EOSH'
#!/usr/bin/env bash
set -euo pipefail

# ---- Inputs / defaults ----
REPO_DIR="${REPO_DIR:-$HOME/.local/share/cfb-tix/app}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.local/share/cfb-tix/venv/bin/python}"
RUN_TIME="${RUN_TIME:-06:10}"  # local time

SYSTEMD_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="cfb-tix-sync"
SERVICE_FILE="$SYSTEMD_DIR/${SERVICE_NAME}.service"
TIMER_FILE="$SYSTEMD_DIR/${SERVICE_NAME}.timer"

# ---- Prompt for GH token (once) and store securely ----
ENV_FILE="$HOME/.config/cfb-tix/env"
mkdir -p "$(dirname "$ENV_FILE")"
if [[ ! -s "$ENV_FILE" ]]; then
  echo "🔑 No GitHub token found."
  read -r -s -p "Paste your GitHub access token (leave blank to skip uploads): " GH_TOKEN_INPUT
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

mkdir -p "$SYSTEMD_DIR"

# ---- Write service (reads token from EnvironmentFile) ----
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Upload merged price_snapshots.csv to GitHub Release

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=%h/.config/cfb-tix/env
ExecStart=$PYTHON_BIN scripts/sync_snapshots.py pull_push
EOF

# ---- Write timer (daily at RUN_TIME) ----
cat > "$TIMER_FILE" <<EOF
[Unit]
Description=Run cfb-tix CSV sync every morning

[Timer]
OnCalendar=*-*-* $RUN_TIME
Persistent=true
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}.timer" || true

# ---- First-time pull (download only; non-fatal) ----
"$PYTHON_BIN" scripts/sync_snapshots.py pull || true

echo "✅ Installed user timer ${SERVICE_NAME}.timer at ${RUN_TIME} daily."
echo "   Repo: $REPO_DIR"
echo "   Token file: $ENV_FILE"
EOSH
chmod +x "${STAGE}/install_sync.sh"

# --- main Linux installer (calls install_sync.sh) ---
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
EOSH
chmod +x "${STAGE}/install_linux.sh"

#---- size the image (repo size + slack) ----#
BYTES_USED=$(du -sb "${STAGE}" | awk '{print $1}')
SLACK=$((64 * 1024 * 1024))                       # 64MB slack
TARGET_SIZE=$(( BYTES_USED + BYTES_USED/2 + SLACK ))  # +60% overhead
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
