# Makefile for CFB Ticket Price Tracker

.PHONY: all ext4 install-linux mount umount clean reset smoke \
        sync-install sync-now sync-status sync-logs sync-uninstall \
        data-pull data-push windows-sync-install windows-sync-now

REPO_DIR   ?= $(shell pwd)
IMG        ?= packaging/dist/cfb-tix.ext4
MOUNTPOINT ?= $(HOME)/mnt/cfb-tix
SYNC_TIME  ?= 07:10
PYTHON     ?= python3

all: ext4

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

clean:
	rm -rf dist/ .build_ext4_work/ packaging/windows/Output/ packaging/dist/

reset:
	chmod +x scripts/reset_linux.sh
	bash scripts/reset_linux.sh

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

# ===== cross-OS local sync =====
data-pull:
	$(PYTHON) scripts/sync_snapshots.py pull

data-push:
	$(PYTHON) scripts/sync_snapshots.py pull_push

# ===== Windows helpers =====
windows-sync-install:
	powershell -ExecutionPolicy Bypass -File .\packaging\windows\register_sync.ps1 -Repo "$(REPO_DIR)" -At "$(SYNC_TIME)"

windows-sync-now:
	powershell Start-ScheduledTask -TaskName "CFB-Tix Snapshot Sync"
