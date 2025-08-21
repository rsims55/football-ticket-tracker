# Makefile for CFB Ticket Price Tracker installers

.PHONY: all ext4 windows clean

all: ext4 windows

# Build Linux ext4 image
ext4:
	chmod +x packaging/build_ext4.sh
	bash packaging/build_ext4.sh

# Build Windows installer (requires Inno Setup / iscc in PATH)
windows:
	"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging/windows/installer.iss /DAppVersion=0.0.0

# Clean build artifacts
clean:
	rm -rf dist/ .build_ext4_work/ packaging/windows/Output/

