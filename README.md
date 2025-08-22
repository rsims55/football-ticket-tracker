# 🏈 College Football Ticket Price Forecasting

Forecasts college football ticket prices and identifies the **optimal date & time to buy**. The app merges schedules, rankings, stadium capacity, rivalry flags, and live price snapshots, then trains a model to predict price trajectories and surface the best purchase window.

---

## 🔧 Features

- ✅ **Data builders**: schedules, rankings (CFD + Wikipedia fallback), stadiums, rivalries  
- ⏱ **Snapshots 4× daily**: logs lowest & average prices + listings from TickPick  
- 🧠 **Daily modeling**: trains & predicts price trajectories; writes optimal purchase rows  
- 📨 **Weekly report**: accuracy summary emailed Sundays  
- 🖥️ **GUI (PyQt5)**: pick a matchup, see predictions, countdown to the optimal time  
- 🏃 **Daemon**: background scheduler keeps running even if the GUI is closed

---

## 📁 Project Structure (repo)

```text
cfb-ticket-tracker/
├── assets/
│   └── icons/                         # cfb-tix.svg, cfb-tix.ico
├── data/
│   ├── annual/                        # stadiums_YYYY.csv, rivalries_YYYY.csv
│   ├── weekly/                        # full_YYYY_schedule.csv, wiki_rankings_YYYY.csv
│   ├── daily/                         # price_snapshots.csv
│   ├── predicted/                     # predicted_prices_optimal.csv
│   └── permanent/                     # team_aliases.json, tickpick_teams.txt
├── logs/                              # local logs (dev runs)
├── models/                            # trained model(s) (dev runs)
├── packaging/
│   ├── linux/
│   │   └── build_ext4.sh              # Linux ext4 image builder
│   └── windows/
│       ├── build_zip.ps1              # Windows zip builder
│       ├── install_win.ps1            # Windows installer (per-user venv + autostart)
│       └── installer.iss              # Inno Setup script
├── reports/
│   └── weekly/
├── scripts/
│   ├── reset_linux.sh
│   ├── reset_windows.ps1
│   └── register_sync.ps1
├── src/
│   ├── builders/                      # annual + weekly setup & daily snapshot
│   ├── cfb_tix/                       # daemon + entry points
│   ├── fetchers/                      # API fetchers
│   ├── gui/                           # PyQt5 GUI
│   ├── modeling/                      # ML model training & prediction
│   ├── reports/                       # report generation & email
│   └── scrapers/                      # web scrapers (stadiums, rivalries, TickPick)
├── .env (optional)
├── pyproject.toml
└── README.md
```

> **Installed app (Linux)** lives under `~/.local/share/cfb-tix/app/` with **data** in `~/.local/share/cfb-tix/app/data/` and **logs** in `~/.local/share/cfb-tix/app/logs/`.  
> **Installed app (Windows)** lives under `%LOCALAPPDATA%\cfb-tix\app\` with logs and data alongside.

---

## 💾 Downloading & Installing

### 🔧 Linux (ext4 image)

The app ships as a self-contained **ext4 image** that installs into `~/.local/share/cfb-tix/`.

```bash
# Build the ext4 image
make ext4

# Mount and run the installer from the image
make install-linux
```

This will:
- Copy the app into `~/.local/share/cfb-tix/app`
- Create a Python venv at `~/.local/share/cfb-tix/venv`
- Install the app (editable mode)
- Enable autostart (systemd --user)
- Install the desktop launcher for the GUI
- **Install the CSV sync timer (daily at 06:10 local)**  
- **Do a first-time pull of `price_snapshots.csv`**

### 🪟 Windows (installer script)

On Windows, install using the provided PowerShell script:

```powershell
# Install into %LOCALAPPDATA%\cfb-tix
powershell -ExecutionPolicy Bypass -File .\packaging\windows\install_win.ps1 -AppDir "$PWD"
```

This will:
- Copy the app into `%LOCALAPPDATA%\cfb-tix\app`
- Create a Python venv in `%LOCALAPPDATA%\cfb-tix\venv`
- Install the app (editable mode)
- Register a background **Task Scheduler job** to run `cfb-tix --no-gui` at logon
- Create a Start Menu shortcut: **“CFB Tickets (GUI)”**
- Register a daily **CSV sync** task at `06:10` (runs `scripts\sync_snapshots.py pull_push`)
- Do a first-time pull of `price_snapshots.csv`

Uninstall/reset:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reset_windows.ps1
```

Managing the sync job:

```powershell
# Run the sync now
Start-ScheduledTask -TaskName "CFB-Tix Snapshot Sync"

# Check last run result
Get-ScheduledTaskInfo -TaskName "CFB-Tix Snapshot Sync"

# Delete the task
Unregister-ScheduledTask -TaskName "CFB-Tix Snapshot Sync" -Confirm:$false
```

Manual sync:

```powershell
# Pull only
python scripts/sync_snapshots.py pull

# Merge + upload (requires GH_TOKEN in .env)
python scripts/sync_snapshots.py pull_push
```

## 📊 Shared CSV Sync

We keep a single shared `price_snapshots.csv` on the repo’s **GitHub Release** (`snapshots` tag).  

- **Linux**: a `systemd --user` timer runs daily at 06:10 (`cfb-tix-sync.timer`).  
- **Windows**: a Task Scheduler job (`cfb-tix-sync`) runs daily at 06:10.

Both do:
- Download the shared CSV
- Merge with your local copy
- Upload the merged version back to GitHub Release (requires `GH_TOKEN`)

### Managing the sync job (Linux)

```bash
make sync-status      # status of service + timer
make sync-now         # run now
make sync-logs        # recent logs
make sync-uninstall   # disable + remove timer
```

### Managing the sync job (Windows)

```powershell
# Register task (default 06:10)
powershell -ExecutionPolicy Bypass -File .\scripts\register_sync.ps1 -At "06:10"

# Run it now
Start-ScheduledTask -TaskName "cfb-tix-sync"

# Check status
Get-ScheduledTaskInfo -TaskName "cfb-tix-sync"

# Unregister
powershell -ExecutionPolicy Bypass -File .\scripts\register_sync.ps1 -Unregister
```

### Manual sync (both OSes)

```bash
# Pull latest CSV only
make data-pull

# Merge + upload (requires GH_TOKEN in .env)
make data-push
```

---

## 🔑 Authentication for Uploads

- **Downloading works without a token** if the repo is public.  
- **Uploading requires a GitHub token.**

Create `.env` in the installed app dir with:

```
GH_TOKEN=ghp_yourtokenhere
```

---

## ▶️ Running

**Entry points**
- `cfb-tix` — run daemon (scheduler). Launches GUI unless `--no-gui`  
- `cfb-tix --no-gui` — headless scheduler service  
- `cfb-tix-gui` — GUI only (no scheduler)

**Quick checks (Linux):**
```bash
pgrep -fa 'cfb[-_]tix'                           # confirm the process is running
tail -n 200 ~/.local/share/cfb-tix/app/logs/cfb_tix.log
```

**Quick checks (Windows):**
```powershell
Get-ScheduledTask -TaskName 'CFB Tickets','cfb-tix-sync'
Get-Content $env:LOCALAPPDATA\cfb-tix\app\logs\cfb_tix.log -Tail 200
```

---

## ⏰ Built-in Schedules (America/New_York)

- **Annual check**: daily @ **00:30** (ensure current-year files; drop prior-year after May 1)  
- **Weekly refresh**: **Wed 06:00** (schedule + rankings)  
- **Daily snapshots**: **00:00, 06:00, 12:00, 18:00** (prices)  
- **Daily modeling**: **06:00** (train → predict)  
- **Sunday report**: **Sun 06:30** (evaluate + weekly report + email)

---

## 🧪 Dev Quickstart

```bash
# From repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e .  # installs `cfb-tix` & deps via pyproject

# Optional .env for external APIs
# CFD_API_KEY=...
# SEATGEEK_CLIENT_ID=...
# SEATGEEK_CLIENT_SECRET=...

# Manual run (dev)
python -m cfb_tix --no-gui
```

**Key runtime deps** (packaged): `joblib`, `scikit-learn` (plus transitive `scipy`, `threadpoolctl`).

---

## 🔍 Troubleshooting

**Linux**
```bash
pgrep -fa 'cfb[-_]tix' || echo "cfb-tix not running"
tail -f ~/.local/share/cfb-tix/app/logs/cfb_tix.log
```

**Windows**
```powershell
Get-ScheduledTask -TaskName 'cfb-tix-sync'
Get-Content $env:LOCALAPPDATA\cfb-tix\app\logs\cfb_tix.log -Tail 50 -Wait
```

---

## 📜 License

© 2025 Randi Sims. All rights reserved.
