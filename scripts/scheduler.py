#!/usr/bin/env python3
"""Unified scheduler for annual/weekly/daily/model/report jobs."""
import json
import os
import random
import subprocess
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PERM_DIR = DATA_DIR / "permanent"
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = PERM_DIR / "scheduler_state.json"

ANNUAL_TIME = time(4, 30)
WEEKLY_TIME = time(5, 30)
MODEL_TIME = time(7, 0)

DAILY_RUNS = 4
MIN_GAP_HOURS = 4
RUN_WINDOW_MIN = 10  # minutes


def _now() -> datetime:
    return datetime.now(TZ)


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _run(cmd: list[str], env: dict | None = None) -> None:
    subprocess.run(cmd, cwd=str(ROOT), env=env or os.environ.copy(), check=True)


def _annual_files_missing(year: int) -> bool:
    annual_dir = DATA_DIR / "annual"
    expected = [
        annual_dir / f"stadiums_{year}.csv",
        annual_dir / f"rivalries_{year}.csv",
        annual_dir / f"venues_{year}.csv",
    ]
    return any(not p.exists() for p in expected)


def _archive_year_folders(prev_year: int) -> None:
    for section in ("annual", "weekly"):
        base = DATA_DIR / section
        archive = base / "archive" / str(prev_year)
        archive.mkdir(parents=True, exist_ok=True)
        for p in base.glob(f"*{prev_year}*"):
            if p.is_dir() or p.name.startswith("archive"):
                continue
            p.rename(archive / p.name)


def _should_run_at(target: time, now: datetime) -> bool:
    window_start = datetime.combine(now.date(), target, tzinfo=TZ)
    window_end = window_start + timedelta(minutes=RUN_WINDOW_MIN)
    return window_start <= now < window_end


def _ensure_daily_times(state: dict, today: str) -> list[str]:
    day_key = f"daily_times_{today}"
    if day_key in state:
        return state[day_key]

    # Generate 4 random times with >=4h spacing and avoid weekly/annual windows
    times = []
    while len(times) < DAILY_RUNS:
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        t = time(hour, minute)
        if t == ANNUAL_TIME or t == WEEKLY_TIME:
            continue
        if any(abs((datetime.combine(_now().date(), t) - datetime.combine(_now().date(), pt)).total_seconds()) < MIN_GAP_HOURS * 3600 for pt in times):
            continue
        times.append(t)
    times = sorted(times)
    state[day_key] = [f"{t.hour:02d}:{t.minute:02d}" for t in times]
    _save_state(state)
    return state[day_key]


def _daily_due(now: datetime, state: dict) -> bool:
    today = now.strftime("%Y-%m-%d")
    times = _ensure_daily_times(state, today)
    run_key = f"daily_ran_{today}"
    ran = set(state.get(run_key, []))
    for tstr in times:
        hh, mm = map(int, tstr.split(":"))
        t = time(hh, mm)
        win_start = datetime.combine(now.date(), t, tzinfo=TZ)
        win_end = win_start + timedelta(minutes=RUN_WINDOW_MIN)
        if win_start <= now < win_end and tstr not in ran:
            ran.add(tstr)
            state[run_key] = sorted(ran)
            _save_state(state)
            return True
    return False


def main() -> None:
    now = _now()
    year = now.year
    month = now.month
    state = _load_state()

    # Annual setup: March 1st 04:30 or anytime after March if missing annual files
    if month >= 3:
        annual_key = f"annual_done_{year}"
        if _should_run_at(ANNUAL_TIME, now) and state.get(annual_key) != str(now.date()):
            _archive_year_folders(year - 1)
            _run(["python", "src/builders/annual_setup.py"])
            state[annual_key] = str(now.date())
            _save_state(state)
        elif _annual_files_missing(year) and state.get(annual_key) != str(now.date()):
            _archive_year_folders(year - 1)
            _run(["python", "src/builders/annual_setup.py"])
            state[annual_key] = str(now.date())
            _save_state(state)

    # Weekly setup: Monday 05:30, year = current if month>=3 else previous
    if now.weekday() == 0:
        weekly_year = year if month >= 3 else year - 1
        weekly_key = f"weekly_done_{now.strftime('%Y-%m-%d')}"
        if _should_run_at(WEEKLY_TIME, now) and state.get(weekly_key) != str(now.date()):
            _run(["python", "src/builders/weekly_update.py"], env={**os.environ, "DEFAULT_YEAR": str(weekly_year)})
            state[weekly_key] = str(now.date())
            _save_state(state)

    # Daily scraper: 4 random times/day, only if month>=3
    if month >= 3:
        if _daily_due(now, state):
            _run(["python", "src/builders/daily_snapshot.py"])

    # Models: Monday 07:00
    if now.weekday() == 0:
        model_key = f"model_done_{now.strftime('%Y-%m-%d')}"
        if _should_run_at(MODEL_TIME, now) and state.get(model_key) != str(now.date()):
            _run(["python", "src/modeling/train_catboost_min.py"], env={**os.environ, "PRUNE_FEATURES": "0"})
            state[model_key] = str(now.date())
            _save_state(state)

            # Weekly report + email
            _run(["python", "src/reports/generate_weekly_report.py"], env={**os.environ, "SEASON_YEAR": str(year)})
            _run(["python", "src/reports/send_email.py"])
            _run(["python", "scripts/health_check.py"])


if __name__ == "__main__":
    main()
