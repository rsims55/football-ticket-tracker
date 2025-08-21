# Makefile for CFB Ticket Price Tracker

.PHONY: all ext4 install-linux mount umount clean reset smoke

all: ext4

# Build Linux ext4 image
ext4:
	chmod +x packaging/build_ext4.sh
	bash packaging/build_ext4.sh

# Mount, run installer from image, then unmount (fresh install flow)
install-linux: ext4 mount
	bash "$(HOME)/mnt/cfb-tix/install_linux.sh"
	$(MAKE) umount
	@echo "✅ Install complete. Service status:"
	systemctl --user status cfb-tix.service --no-pager || true

# Mount helper
mount:
	mkdir -p "$(HOME)/mnt/cfb-tix"
	sudo mount -o loop,ro dist/cfb-tix.ext4 "$(HOME)/mnt/cfb-tix"

# Unmount helper
umount:
	-sudo umount "$(HOME)/mnt/cfb-tix"
	-rmdir "$(HOME)/mnt/cfb-tix"

# Quick smoke test after install
smoke:
	@echo "→ Checking service and CLIs…"
	systemctl --user status cfb-tix.service --no-pager || true
	"$(HOME)/.local/share/cfb-tix/venv/bin/cfb-tix" --no-gui --help || true
	"$(HOME)/.local/share/cfb-tix/venv/bin/cfb-tix-gui" --help 2>/dev/null || true
	@echo "→ Upcoming jobs:"
	journalctl --user -u cfb-tix.service -n 25 --no-pager || true

# Clean build artifacts
clean:
	rm -rf dist/ .build_ext4_work/ packaging/windows/Output/

# Full reset (calls the script)
reset:
	chmod +x scripts/reset_linux.sh
	bash scripts/reset_linux.sh
