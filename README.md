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

### Linux (ext4 image)

1) Download **`cfb-tix.ext4`** from the latest GitHub Release.  
2) Mount and run the installer (autostart enabled by default):
```bash
mkdir -p ~/mnt/cfb-tix
sudo mount -o loop,ro cfb-tix.ext4 ~/mnt/cfb-tix
bash ~/mnt/cfb-tix/install_linux.sh
# or opt out of autostart:
bash ~/mnt/cfb-tix/install_linux.sh --no-autostart
sudo umount ~/mnt/cfb-tix && rmdir ~/mnt/cfb-tix
```

If the GUI needs Qt/XCB libs on your distro:
```bash
sudo apt-get update && sudo apt-get install -y \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-shm0
```

**What you get (Linux):**
- **Daemon service (user)**: `~/.config/systemd/user/cfb-tix.service` → runs `cfb-tix --no-gui`  
- **GUI launcher**: Applications menu → **“CFB Tickets (GUI)”**  
- **Autostart toggle**: `cfb-tix autostart --enable|--disable|--status`

### Windows (.exe)

1) Download **`cfb-tix-setup.exe`** from the latest GitHub Release and run it.  
2) The installer places the app in `%LocalAppData%\cfb-tix\app`, creates `%LocalAppData%\cfb-tix\venv`, registers a **Task Scheduler** job (daemon on logon), and adds a **Start Menu** shortcut “CFB Tickets (GUI)”.

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
