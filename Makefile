# Makefile for CFB Ticket Price Tracker

# ========= OS detection & common vars =========
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
  PYTHON ?= python
  PS      := powershell -NoProfile -ExecutionPolicy Bypass -Command
  PSFILE  := powershell -NoProfile -ExecutionPolicy Bypass -File
else
  IS_WINDOWS := 0
  PYTHON ?= python3
endif

# ========= Default =========
ifeq ($(IS_WINDOWS),1)
  .PHONY: all
  all: win-zip
else
  .PHONY: all
  all: ext4
endif

.PHONY: help \
        ext4 install-linux mount umount clean reset smoke \
        sync-install sync-now sync-status sync-logs sync-uninstall \
        data-pull data-push \
        win-zip win-install win-reset win-smoke \
        win-sync-install win-sync-now win-sync-status win-sync-uninstall

# ========= Common paths =========
REPO_DIR   ?= $(shell pwd)
IMG        ?= packaging/dist/cfb-tix.ext4
MOUNTPOINT ?= $(HOME)/mnt/cfb-tix
SYNC_TIME  ?= 07:10

# Windows artifact path
WIN_ZIP    ?= packaging/dist/cfb-tix-win.zip

help:
	@echo "Targets:"
	@echo "  Default 'all': builds OS-specific artifact (ext4 on Linux/macOS, zip on Windows)"
	@echo "  Linux:  ext4 install-linux mount umount smoke reset clean"
	@echo "          sync-install sync-now sync-status sync-logs sync-uninstall"
	@echo "  Both:   data-pull data-push"
	@echo "  Win:    win-zip win-install win-reset win-smoke"
	@echo "          win-sync-install win-sync-now win-sync-status win-sync-uninstall"

# ========= Linux packaging flow =========
ext4:
	chmod +x packaging/linux/build_ext4.sh
	bash packaging/linux/build_ext4.sh

install-linux: ext4 mount
	bash "$(MOUNTPOINT)/install_linux.sh" --sync-time $(SYNC_TIME)
	$(MAKE) umount
	@echo "✅ Install complete. Service status:"
	systemctl --user status cfb-tix.service --no-pager || true
	@echo "💡 Tip: run 'make sync-status' to check the CSV sync timer."

mount:
	mkdir -p "$(MOUNTPOINT)"
	sudo mount -o loop,ro "$(IMG)" "$(MOUNTPOINT)"

umount:
	- sudo umount "$(MOUNTPOINT)"
	- rmdir "$(MOUNTPOINT)"

smoke:
	@echo "→ Checking service and CLIs…"
	systemctl --user status cfb-tix.service --no-pager || true
	"$(HOME)/.local/share/cfb-tix/venv/bin/cfb-tix" --no-gui --help || true
	"$(HOME)/.local/share/cfb-tix/venv/bin/cfb-tix-gui" --help 2>/dev/null || true
	@echo "→ Recent logs:"
	journalctl --user -u cfb-tix.service -n 25 --no-pager || true

reset:
	chmod +x scripts/reset_linux.sh
	bash scripts/reset_linux.sh

clean:
	rm -rf dist/ .build_ext4_work/ .build_win_work/ packaging/windows/Output/ packaging/dist/

# ===== CSV sync helpers (Linux) =====
sync-install:
	REPO_DIR="$(HOME)/.local/share/cfb-tix/app" \
	PYTHON_BIN="$(HOME)/.local/share/cfb-tix/venv/bin/python" \
	RUN_TIME="$(SYNC_TIME)" \
	bash "$(MOUNTPOINT)/install_sync.sh"

sync-now:
	systemctl --user start cfb-tix-sync.service || true

sync-status:
	systemctl --user status cfb-tix-sync.service --no-pager || true
	systemctl --user status cfb-tix-sync.timer --no-pager || true

sync-logs:
	journalctl --user -u cfb-tix-sync.service -n 50 --no-pager || true

sync-uninstall:
	- systemctl --user disable --now cfb-tix-sync.timer
	- rm -f $(HOME)/.config/systemd/user/cfb-tix-sync.service
	- rm -f $(HOME)/.config/systemd/user/cfb-tix-sync.timer
	- systemctl --user daemon-reload
	@echo "🧹 Removed cfb-tix-sync user timer."

# ===== Cross-OS local sync (runs in current repo/venv) =====
data-pull:
	$(PYTHON) scripts/sync_snapshots.py pull

data-push:
	$(PYTHON) scripts/sync_snapshots.py pull_push

# ========= Windows helpers =========
# Build distributable zip (Windows analog of ext4 image)
win-zip:
	$(PSFILE) .\packaging\windows\build_zip.ps1

# Install into %LOCALAPPDATA%\cfb-tix using your canonical installer
win-install:
	$(PSFILE) .\packaging\windows\install_win.ps1 -AppDir "$(REPO_DIR)"

# Uninstall/reset Windows install (kills tasks, removes payload & shortcuts)
win-reset:
	$(PSFILE) .\scripts\reset_windows.ps1

# Quick status checks on Windows
win-smoke:
	@echo "→ Checking Scheduled Tasks (cfb-tix & sync)…"
	- $(PS) "Get-ScheduledTask -TaskName 'CFB Tickets','cfb-tix-sync' | Format-Table -AutoSize" || true
	@echo "→ Checking GUI/daemon stubs exist…"
	- $(PS) "Get-ChildItem -Path $$env:LOCALAPPDATA\cfb-tix\venv\Scripts\ -Filter 'cfb-tix*' | Select-Object Name,Length" || true

# Windows: register daily sync via scripts/register_sync.ps1
win-sync-install:
	$(PSFILE) .\scripts\register_sync.ps1 -At "$(SYNC_TIME)"

win-sync-now:
	$(PS) "Start-ScheduledTask -TaskName 'cfb-tix-sync'" || true

win-sync-status:
	$(PS) "Get-ScheduledTask -TaskName 'cfb-tix-sync' | Format-List *" || true
	$(PS) "Get-ScheduledTaskInfo -TaskName 'cfb-tix-sync'" || true

win-sync-uninstall:
	- $(PSFILE) .\scripts\register_sync.ps1 -Unregister || true
	@echo "🧹 Removed cfb-tix-sync scheduled task."
