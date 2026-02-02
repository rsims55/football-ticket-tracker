# 🎟️ College Football Ticket Price Tracker

A cross‑platform toolkit to **collect**, **model**, and **visualize** college football ticket prices with a clean, automated pipeline.  
Primary goals: daily price snapshots, robust modeling, and a GUI predictor.

---

## ✅ Quickstart (Windows + Linux)

1) **Clone repo**
```
git clone https://github.com/rsims55/football-ticket-tracker.git
cd football-ticket-tracker
```

2) **Create `.env`**
```
cp configs/.env.example .env
```
Fill in required secrets in `.env` (see below).

3) **Setup environment**
- **Linux**
```
bin/setup_linux.sh
```
- **Windows**
```
bin/setup_windows.cmd
```

4) **Run everything (scheduler + GUI ready)**
- **Linux**
```
bin/run_all_linux.sh
```
- **Windows**
```
bin/run_all_windows.cmd
```

5) **Launch GUI only**
- **Linux**: `bin/run_gui_linux.sh`
- **Windows**: `bin/run_gui_windows.cmd`

---

## 🔐 Required `.env` secrets

Edit `.env` at repo root:

```
CFD_API_KEY=
TICKPICK_EMAIL=
TICKPICK_PASSWORD=
GMAIL_ADDRESS=
GMAIL_APP_PASSWORD=
TO_EMAIL=
```

Optional:
```
# SEASON_YEAR=2026
ALLOW_OFFSEASON_SCRAPE=0
```

---

## 🧠 Pipeline Schedule (EDT)

All time rules use **America/New_York (EDT/EST)**.

### 1) Annual Setup  
**Runs Mar 1 @ 4:30 AM**  
Also runs later in March+ if annual files are missing.

### 2) Weekly Update  
**Mondays @ 5:30 AM**  
Uses **current year** if March+; else prior year.  
Overwrites same‑year weekly outputs.

### 3) Daily Scraper  
**4 random runs/day**, ≥4 hours apart  
Only runs if **March+** in current year.  
Skips any game **after kickoff**.

### 4) Model Training  
**Mondays @ 7:00 AM**

### 5) Weekly Report + Email  
Runs after model training; emailed to `TO_EMAIL`.

---

## ⚠️ Postseason Rules (Current)

- **Postseason games are excluded from:**
  - Model training
  - GUI display
- **Postseason can still be scraped** but is ignored downstream.
- We’ll revisit postseason logic after next season.

---

## ✅ Status Reporting

Each pipeline step writes to:
```
data/permanent/pipeline_status.json
```

The **weekly report email** includes the latest success/skip/fail status for:
- annual_setup  
- weekly_update  
- daily_snapshot  
- model_train  
- weekly_report  
 - health_check

---

## 🩺 Health Check

Weekly health check validates required columns and basic freshness:
```
python scripts/health_check.py
```
Results are written to `data/permanent/pipeline_status.json` and included in the weekly report.

---

## 🗄️ Backups

`price_snapshots.csv` is backed up automatically before each write:
```
data/daily/backups/price_snapshots.csv.YYYYMMDD_HHMMSS.bak
```
The last **7** backups are kept.

---

## 🛠️ System Daemons (optional)

### Linux (systemd user)
```
mkdir -p ~/.config/systemd/user
cp configs/systemd/cfb-ticket-tracker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cfb-ticket-tracker.service
```

### Windows (Task Scheduler)
1) Edit `configs/windows/cfb-ticket-tracker.xml`  
   Replace `C:\path\to\repo` with your repo location.
2) Import the task in Task Scheduler.

### Keep Awake
- **Linux**: systemd-inhibit is built into the service (prevents sleep while running).
- **Windows**: daemon runner uses `SetThreadExecutionState` to prevent sleep.

---

## ✅ Daemon Activation (VERY CLEAR)

When the system daemon starts, it **immediately runs all pipeline steps once**:
- annual setup
- weekly update
- daily snapshot (will skip in offseason)
- model training
- weekly report
- email send

It then continues running the scheduler loop.

### Linux activation (one‑time)
```
mkdir -p ~/.config/systemd/user
cp configs/systemd/cfb-ticket-tracker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cfb-ticket-tracker.service
```

### Windows activation (one‑time)
1) Edit `configs/windows/cfb-ticket-tracker.xml`  
   Replace `C:\path\to\repo` with your repo path.
2) Import into Task Scheduler.

### Manual activation (if needed)
- Linux: `bin/run_daemon_linux.sh`
- Windows: `bin/run_daemon_windows.cmd`

---

## 🔄 Update / Upgrade

```
git pull
```
Then re-run setup if dependencies changed:
- Linux: `bin/setup_linux.sh`
- Windows: `bin/setup_windows.cmd`

---

## 📂 Project Structure (clean)

```
assets/                 icons, UI assets
bin/                    cross‑platform launcher scripts
configs/                .env.example + templates
data/                   all data outputs + permanent state
docs/                   documentation
reports/                weekly reports + model outputs
scripts/                one‑off maintenance scripts
src/                    main codebase (builders, models, GUI)
```

---

## ✅ Core Commands (manual)

**Daily snapshot**
```
python src/builders/daily_snapshot.py
```

**Weekly update**
```
python src/builders/weekly_update.py
```

**Train model**
```
python src/modeling/train_catboost_min.py
```

**Generate weekly report**
```
python src/reports/generate_weekly_report.py
```

**Run GUI**
```
python src/gui/ticket_predictor_gui.py
```

---

## 🧩 Notes

- Daily scraping **won’t run in offseason** (March+ rule).
- If no events exist, the daily script prints a clear “skipped” message.
- All scripts emit **clean success/failure logs**.

---

## ✅ Troubleshooting

**GUI crashes on datetime compare**  
Make sure `startDateEastern` is present; GUI now coerces mixed timezones safely.

**No report email**
Check `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, and `TO_EMAIL` in `.env`.

**No daily scraping**
Likely offseason rule or not a collection window; check console output.

---

## 👇 Next steps (optional)

If you want system‑level daemon registration:
- **Linux:** systemd user service  
- **Windows:** Task Scheduler XML  

Just say the word and we’ll generate those.
