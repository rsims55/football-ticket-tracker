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
│   ├── build_ext4.sh                  # Linux ext4 image builder
│   └── windows/
│       ├── install_win.ps1
│       └── installer.iss
├── reports/
│   └── weekly/
├── scripts/
│   └── reset_linux.sh
├── src/
│   ├── builders/                      # annual + weekly setup & daily snapshot
│   │   ├── annual_setup.py
│   │   ├── daily_snapshot.py
│   │   └── weekly_update.py
│   ├── cfb_tix/                       # daemon + entry points
│   │   ├── daemon.py                  # `cfb-tix` (daemon) / `cfb-tix-gui` (GUI only)
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── fetchers/
│   │   ├── fetch_ncaa_events.py
│   │   ├── rankings_fetcher.py
│   │   └── schedule_fetcher.py
│   ├── gui/
│   │   └── ticket_predictor_gui.py
│   ├── modeling/
│   │   ├── train_price_model.py
│   │   ├── predict_price.py
│   │   └── evaluate_predictions.py
│   ├── reports/
│   │   ├── generate_weekly_report.py
│   │   └── send_email.py
│   └── scrapers/
│       ├── rivalry_scraper.py
│       ├── stadium_scraper.py
│       └── tickpick_pricer.py
├── .env (optional)
├── pyproject.toml
└── README.md
```

> **Installed app (Linux)** lives under `~/.local/share/cfb-tix/app/` with **data** in `~/.local/share/cfb-tix/app/data/` and **logs** in `~/.local/share/cfb-tix/app/logs/`.

---

## 💾 Downloading & Installing

## 🔧 Building & Installing (Linux)

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

---

## 📊 Shared CSV Sync

We keep a single shared `price_snapshots.csv` on the repo’s **GitHub Release** (`snapshots` tag).  
The installer sets up a **systemd user timer** (`cfb-tix-sync.timer`) that:

- Every morning at 06:10 local time:
  - Downloads the current CSV from GitHub
  - Merges with your local copy
  - Uploads the merged version back to the Release (requires `GH_TOKEN`)

### Managing the sync job

```bash
# Check status of sync service + timer
make sync-status

# Run sync right now (merge + upload)
make sync-now

# View recent logs
make sync-logs

# Remove the timer if you don’t want daily sync
make sync-uninstall
```

### Manual sync

```bash
# Pull latest CSV only
make data-pull

# Merge + upload (requires GH_TOKEN in ~/.local/share/cfb-tix/app/.env)
make data-push
```

---

## 🔑 Authentication for Uploads

- **Downloading works without a token** if the repo is public.  
- **Uploading requires a GitHub token.**

Create `~/.local/share/cfb-tix/app/.env` with:

```
GH_TOKEN=ghp_yourtokenhere
```

**What you get (Linux):**
- **Daemon service (user)**: `~/.config/systemd/user/cfb-tix.service` → runs `cfb-tix --no-gui`  
- **GUI launcher**: Applications menu → **“CFB Tickets (GUI)”**  
- **Autostart toggle**: `cfb-tix autostart --enable|--disable|--status`

## 🪟 Installing & CSV Sync (Windows)

On Windows, we use a **Task Scheduler job** to keep `price_snapshots.csv` synced daily.

### Install the scheduled task

From PowerShell (run once):

```powershell
cd $HOME\cfb-ticket-tracker

# Register the daily sync at 06:10 local time
powershell -ExecutionPolicy Bypass -File .\packaging\windows\register_sync.ps1 -Repo "$PWD" -At "06:10"
```

This will:
- Create a Task Scheduler job named **CFB-Tix Snapshot Sync**
- Run every day at 06:10 local time
- Run `scripts/sync_snapshots.py pull_push` using your Python
- Do a first-time pull of `price_snapshots.csv` immediately

### Managing the sync job

```powershell
# Run the sync right now
Start-ScheduledTask -TaskName "CFB-Tix Snapshot Sync"

# Check the last run result
Get-ScheduledTaskInfo -TaskName "CFB-Tix Snapshot Sync"

# Delete the task if you no longer want daily sync
Unregister-ScheduledTask -TaskName "CFB-Tix Snapshot Sync" -Confirm:$false
```

### Manual sync

```powershell
# Pull latest CSV only
python scripts/sync_snapshots.py pull

# Merge + upload (requires GH_TOKEN in .env)
python scripts/sync_snapshots.py pull_push
```

### 🔑 Authentication for Uploads

- **Downloading works without a token** if the repo is public.  
- **Uploading requires a GitHub token.**

Create `.env` in your repo root with:

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

**Autostart (Linux/systemd user service):**
```bash
cfb-tix autostart --enable
cfb-tix autostart --disable
cfb-tix autostart --status
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

```bash
# Process alive?
pgrep -fa 'cfb[-_]tix' || echo "cfb-tix not running"

# Tail the daemon log
tail -f ~/.local/share/cfb-tix/app/logs/cfb_tix.log

# Re-run model steps from the installed app
APP=~/.local/share/cfb-tix/app
PY=~/.local/share/cfb-tix/venv/bin/python
cd "$APP" && "$PY" -m modeling.train_price_model && "$PY" -m modeling.predict_price
```

---

## 📜 License

© 2025 Randi Sims. All rights reserved.
