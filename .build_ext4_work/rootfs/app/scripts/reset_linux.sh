#!/usr/bin/env bash
set -euo pipefail

SERVICE="cfb-tix.service"
MNT="$HOME/mnt/cfb-tix"

echo "→ Stopping user service (if present)…"
systemctl --user stop "$SERVICE" 2>/dev/null || true
systemctl --user disable "$SERVICE" 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/$SERVICE" || true
systemctl --user daemon-reload || true

echo "→ Removing local install…"
rm -rf "$HOME/.local/share/cfb-tix"

echo "→ Removing desktop launcher & icon…"
rm -f "$HOME/.local/share/applications/cfb-tix-gui.desktop"
rm -f "$HOME/.local/share/icons/hicolor/scalable/apps/cfb-tix.svg"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
gtk-update-icon-cache -f "$HOME/.local/share/icons" 2>/dev/null || true

echo "→ Unmounting image (if mounted)…"
sudo umount "$MNT" 2>/dev/null || true
rm -rf "$MNT"

echo "→ Cleaning build artifacts…"
rm -rf dist/ .build_ext4_work/

echo "✅ Reset complete."
