# 🎟️ College Football Ticket Price Tracker

A cross-platform toolkit to **collect**, **model**, and **visualize** college football ticket prices with a fully automated pipeline.

---

## 🚀 Quickstart

### Windows

```powershell
# 1. Clone the repo
git clone https://github.com/rsims55/football-ticket-tracker.git
cd football-ticket-tracker

# 2. Create .env and fill in secrets (see below)
copy .env.example .env

# 3. Set up Python virtual environment
.\bin\setup_windows.ps1

# 4. Install the daemon as a startup item (runs automatically at every login)
wscript.exe "C:\path\to\repo\bin\cfb_daemon.vbs"

# Or run the daemon manually in the foreground:
.\bin\run_daemon_windows.ps1
```

### Linux

```bash
# 1. Clone the repo
git clone https://github.com/rsims55/football-ticket-tracker.git
cd football-ticket-tracker

# 2. Create .env and fill in secrets (see below)
cp .env.example .env

# 3. Set up venv and run the daemon
bin/setup_linux.sh
bin/run_daemon_linux.sh
```

> **Note:** Windows uses `.venv_win\`, Linux uses `.venv/`. Neither is committed to git.

---

## 🔐 Required `.env` Secrets

```env
CFBD_API_KEY=
GMAIL_ADDRESS=
GMAIL_APP_PASSWORD=
TO_EMAIL=
```

Optional:
```env
SEASON_YEAR=2026
ALLOW_OFFSEASON_SCRAPE=0
```

---

## 🤖 Daemon

The daemon runs all jobs on a schedule and auto-starts on login.

### Auto-start (Windows)

A VBScript is placed in your Windows Startup folder:
```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\cfb_daemon.vbs
```
This launches the daemon hidden in the background every time you log in. No terminal needs to be open.

### Auto-start (Linux)

Use the provided systemd user service:
```bash
mkdir -p ~/.config/systemd/user
cp configs/systemd/cfb-ticket-tracker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cfb-ticket-tracker.service
```

### Manual start

```powershell
# Windows
.\bin\run_daemon_windows.ps1

# Linux
bin/run_daemon_linux.sh
```

### Check if running

```powershell
# Windows — lock file exists while daemon is alive
Test-Path "$env:LOCALAPPDATA\cfb-tix\Logs\daemon.lock"

# Tail the log
Get-Content "$env:LOCALAPPDATA\cfb-tix\Logs\cfb_tix.log" -Tail 20 -Wait
```

---

## 🗓️ Pipeline Schedule (America/New_York)

### First-run kickoff (runs once on daemon startup, in order)

Each step completes before the next begins:

1. **Annual Setup** — builds venues, rivalries, stadiums CSVs for the season
2. **Weekly Update** — refreshes full schedule and rankings
3. **Daily Snapshot** — full scrape of all teams from TickPick
4. **Model Training** — retrains CatBoost on all available data
5. **Weekly Report + Email** — generates report and sends to `TO_EMAIL`

### Recurring schedule

| Time (ET) | Job |
|---|---|
| 12:00 AM, 6:00 AM, 12:00 PM, 6:00 PM | Daily snapshot (full scrape) |
| 6:45 AM, 6:45 PM | Train model + predict prices |
| 8:00 AM daily | Evaluate predictions |
| 8:15 AM daily | Generate weekly report |
| 8:20 AM daily | Send report email |
| Wednesday 5:30 AM | Weekly update |
| May 1, 5:00 AM | Annual setup |
| :15 past every hour | Hourly git sync (push only) |

---

## 📧 Weekly Report Email

Sent automatically at **8:20 AM ET daily**. Includes:

- **CatBoost Training Summary** — rows, Price MAE/RMSE, timing accuracy
- **Pipeline Status** — success/fail for each step with timestamps
- **Best Predictors** — feature importances from the model
- **Season State & Data Freshness**
- **Model Accuracy (Past 7 Days)** — when in-season games are available

---

## 🛠️ Manual Commands

```bash
# Set PYTHONPATH first (from repo root)
export PYTHONPATH=src  # Linux
$env:PYTHONPATH="src"  # Windows PowerShell

# Daily snapshot
python src/builders/daily_snapshot.py

# Test run (3 random teams)
DAILY_TEST_MODE=1 TEAMS_LIMIT=3 python src/builders/daily_snapshot.py

# Weekly update
python src/builders/weekly_update.py

# Annual setup
python src/builders/annual_setup.py

# Train model
python src/modeling/train_catboost_min.py

# Generate report
python src/reports/generate_weekly_report.py

# Send report email
python src/reports/send_email.py

# Health check
python scripts/health_check.py

# Launch GUI
python src/gui/ticket_predictor_gui.py
```

---

## 📂 Project Structure

```
bin/                    launcher scripts (setup, daemon, run) for Windows + Linux
data/
  annual/               venues, stadiums, rivalries (rebuilt each season)
  daily/                price_snapshots_YYYY.csv, archives, backups
  modeling/             combined training rows
  permanent/            pipeline_status.json, team aliases/conferences
  weekly/               full schedule
models/                 catboost_price_min.cbm (trained model)
reports/
  catboost_*.csv        training metrics and feature importances
  weekly/               weekly report markdown files
scripts/                health_check.py and maintenance utilities
src/
  builders/             annual_setup.py, weekly_update.py, daily_snapshot.py
  cfb_tix/              daemon, launcher, Windows startup integration
  fetchers/             schedule, rankings, venues fetchers
  gui/                  ticket_predictor_gui.py
  modeling/             train_catboost_min.py
  reports/              generate_weekly_report.py, send_email.py
  utils/                logging, status, http helpers
```

---

## 🧠 Model

Trains a **CatBoost regressor** to predict the future minimum ticket price from a live snapshot.

**Target:** `gap_pct` — the % drop from the current lowest price to the future minimum.

**Model file:** `models/catboost_price_min.cbm` (retrained on all years each run).

**All features:**

| Feature | Type | Notes |
|---|---|---|
| `homeTeam` | categorical | Strongest signal (~24%) — pinned |
| `hours_until_game` | numeric | Time to kickoff; monotonic constraint applied (~12%) — pinned |
| `awayTeam` | categorical | Critical for marquee matchups (~11%) — pinned |
| `capacity` | numeric | Stadium size (~11%) |
| `week` | numeric | Season week number (~9%) |
| `kickoff_hour` | numeric | Hour of kickoff in local time (~9%) |
| `away_last_point_diff_at_snapshot` | numeric | Away team recent form — point differential at snapshot time (~9%) |
| `home_last_point_diff_at_snapshot` | numeric | Home team recent form — point differential at snapshot time (~8%) |
| `homeConference` | categorical | Conference-level pricing effects (~6%) — pinned |
| `awayConference` | categorical | Away team's conference |
| `homeTeamRank` | numeric | AP poll rank for home team; missing indicator included — pinned |
| `awayTeamRank` | numeric | AP poll rank for away team (optional, off by default) |
| `season_year` | numeric | Year-over-year price trends — pinned |
| `neutralSite` | numeric (bool) | Game played at neutral venue |
| `isRivalry` | numeric (bool) | Rivalry game flag |
| `isRankedMatchup` | numeric (bool) | Both teams ranked |
| `conferenceGame` | numeric (bool) | In-conference game (optional, off by default) |
| `kickoff_dayofweek` | numeric | Day of week (0=Mon … 6=Sun) |
| `stadium` | categorical | Specific stadium name (optional, off by default) |

Pinned features are never dropped by the importance pruner regardless of score.

---

## 🩺 Health Check

Validates required columns, data freshness, and date format integrity:
```bash
python scripts/health_check.py
```
Results are written to `data/permanent/pipeline_status.json` and included in the weekly report.

---

## ⚠️ Postseason Rules

- Postseason games are **excluded** from model training and GUI display.
- They can still be scraped but are filtered downstream.

---

## 🔄 Updating

```bash
git pull
```

Re-run setup if dependencies changed:
- Windows: `.\bin\setup_windows.ps1`
- Linux: `bin/setup_linux.sh`

---

## 🔧 Troubleshooting

**Daemon won't start (lock file stuck)**
```powershell
Remove-Item "$env:LOCALAPPDATA\cfb-tix\Logs\daemon.lock" -Force
```

**No report email**
Check `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, and `TO_EMAIL` in `.env`.

**Snapshot not updating**
Ensure `DAILY_TEST_MODE` is not set to `1` in your environment. The daemon sets it to `0` automatically.

**Unicode errors on Windows**
The daemon sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` for all child scripts automatically.
