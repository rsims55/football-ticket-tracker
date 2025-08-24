# 🎟️ College Football Ticket Price Tracker

A cross-platform toolkit to track, model, and visualize college football ticket prices. Priority platform: **Windows** (silent background daemon, Start Menu/Desktop shortcuts, and automatic icon packaging). Linux quickstart included.

---

## ✨ What you get

- **GUI app**: “Ticket Price Predictor” (PyQt5), launched windowlessly via a Start Menu/Desktop shortcut.
- **Background daemon**: “CFB Ticket Tracker (Background)” runs **silently** (no console window) and:
  - Pulls the latest `data/daily/price_snapshots.csv` at startup.
  - Runs a **daily sync at 7:00 AM** local time (pull → push if you have a token).
- **Icons & shortcuts**: Installed via `cfb-tix-shortcuts` (no PowerShell popups).
- **One source of truth** for snapshots: a GitHub Release asset (`snapshots-latest` → `price_snapshots.csv`) with safe ETag caching.

---

## 📦 Project layout (key parts)

- `src/gui/` — GUI (`ticket_predictor_gui.py`)
- `src/cfb_tix/daemon.py` — headless daemon (`cfb-tix run --no-gui`), schedules the 7:00 AM sync
- `src/cfb_tix/launchers.py` — windowless Windows launcher for the daemon
- `src/cfb_tix/windows/create_shortcuts.py` — creates Start Menu, Desktop, Startup shortcuts (icons, no console)
- `src/cfb_tix/windows/data_sync.py` — pull/push `price_snapshots.csv` against GitHub Releases; can prompt for token
- `assets/icons/` — source icons; packaged copies live under `src/cfb_tix/assets/icons/`
- `data/` — local data folder (daemon/GUI read & write here)

---

## 🪟 Windows Quickstart (recommended)

1) **Install prerequisites**
   - Install Git for Windows.
   - Install **Python 3.11 (64-bit)** and check “Add python.exe to PATH".

2) **Clone & create a virtual environment**
   - `git clone https://github.com/rsims55/football-ticket-tracker.git`
   - `cd football-ticket-tracker`
   - `python -m venv venv`
   - `.\venv\Scripts\Activate` (PowerShell) or run the equivalent in your shell.

3) **Install the package (editable)**
   - `python -m pip install --upgrade pip wheel setuptools`
   - `pip install -e .`

4) **Create shortcuts (icons, no console)**
   - `cfb-tix-shortcuts`
   - This adds:
     - Start Menu → **CFB Ticket Tracker**
       - **Ticket Price Predictor** (GUI)
       - **CFB Ticket Tracker (Background)** (daemon, silent)
     - Desktop → **Ticket Price Predictor**
     - Startup → **CFB-Ticket-Tracker** (daemon auto-starts every login)

5) **First-run token (optional but recommended)**
   - If you installed via the Windows installer, it already prompted you for a GitHub token.
   - If you installed manually, you can add it now:
     - `.\venv\Scripts\pythonw.exe -m cfb_tix.windows.data_sync ensure_token`
     - This stores `GITHUB_TOKEN=<your token>` in `.env` at the repo root.
   - The token is **required only for uploads**; downloads work without it (subject to GitHub rate limits).

6) **Run**
   - GUI: Start Menu → **Ticket Price Predictor**.
   - Background daemon (silent): Start Menu → **CFB Ticket Tracker (Background)**. It also auto-starts at login.

7) **Data locations**
   - Snapshots live at `data/daily/price_snapshots.csv`.
   - The daemon performs an initial pull at start, then a **daily sync at 7:00 AM** local time.

---

## 🐧 Linux Quickstart

1) **Prereqs**: Git, Python 3.11.
2) **Clone & venv**:
   - `git clone https://github.com/rsims55/football-ticket-tracker.git`
   - `cd football-ticket-tracker`
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3) **Install**:
   - `pip install --upgrade pip`
   - `pip install -e .`
4) **Run**:
   - GUI: `python -m gui.ticket_predictor_gui`
   - Daemon (foreground): `python -m cfb_tix run --no-gui`
5) **Optional autostart (systemd user)**:
   - Copy `packaging/linux/cfb-tix.service` to `~/.config/systemd/user/cfb-tix.service`
   - `systemctl --user enable --now cfb-tix.service`
6) **Token prompt** (optional for uploads):
   - `python -m cfb_tix.windows.data_sync ensure_token` (works on Linux too; stores to `.env`)

---

## 🔁 Data sync model

- **Remote source**: GitHub Release tag `snapshots-latest`, asset name `price_snapshots.csv`.
- **On start**: the daemon pulls the latest file using conditional requests (ETag / Last-Modified).
- **Daily at 7:00 AM**: pull latest → push local (push requires token).
- **Token**: read from `SNAP_GH_TOKEN` or `GITHUB_TOKEN` env vars, or `.env` at repo root.
- **Manual commands** (all platforms):
  - Ensure token: `python -m cfb_tix.windows.data_sync ensure_token`
  - Pull only: `python -m cfb_tix.windows.data_sync pull`
  - Push only: `python -m cfb_tix.windows.data_sync push`
  - Pull → Push: `python -m cfb_tix.windows.data_sync pull_push`

---

## 🧰 Commands & entry points

- GUI (module): `python -m gui.ticket_predictor_gui`
- Daemon (module): `python -m cfb_tix run --no-gui`
- GUI desktop app (Windows shim): Start Menu → **Ticket Price Predictor**
- Background daemon (Windows shim): Start Menu → **CFB Ticket Tracker (Background)**
- Shortcut creator: `cfb-tix-shortcuts`

---

## 🔧 Configuration (env)

- `SNAP_OWNER` (default `rsims55`)
- `SNAP_REPO` (default `football-ticket-tracker`)
- `SNAP_TAG` (default `snapshots-latest`)
- `SNAP_ASSET` (default `price_snapshots.csv`)
- `SNAP_DEST` (default `data/daily/price_snapshots.csv`)
- `SNAP_GH_TOKEN` or `GITHUB_TOKEN` (required for uploads; optional for downloads)

You can set these in a `.env` file in the repo root (the token prompt writes here automatically).

---

## 🖼 Icons & packaging

- Icons are included in wheels and sdists:
  - `src/cfb_tix/assets/icons/cfb-tix_gui.ico`
  - `src/cfb_tix/assets/icons/cfb-tix_daemon.ico`
- `cfb-tix-shortcuts` assigns these icons to Start Menu/Desktop/Startup shortcuts on Windows.
- Launchers use **GUI-style executables** (via `[project.gui-scripts]`) to avoid console windows.

---

## 🧪 Troubleshooting

- **No shortcuts created**: Ensure the venv is active and run `cfb-tix-shortcuts` again.
- **Icons missing on shortcuts**: Confirm the `.ico` files exist under `src/cfb_tix/assets/icons/` and reinstall `pip install -e .`.
- **No uploads**: Add a token via `pythonw -m cfb_tix.windows.data_sync ensure_token` or set `GITHUB_TOKEN`.
- **GUI won’t start**: Try `python -m gui.ticket_predictor_gui` from the venv to see errors in the console.
- **Daemon not syncing**: Check `logs/cfb_tix.log`.

---

## 📝 Development notes

- GUI lives under `src/gui/` (not under `cfb_tix`). Entry points are configured accordingly.
- Daemon honors `--with-gui` if you want to route through the same process (dev convenience).
- Packaging uses:
  - `[project.gui-scripts]` for windowless Windows shims.
  - `include-package-data = true` and `[tool.setuptools.package-data] cfb_tix = ["assets/icons/*.ico"]`.
  - `MANIFEST.in` to include icons in sdists.

---

## 📄 License

Proprietary (see `pyproject.toml`).
