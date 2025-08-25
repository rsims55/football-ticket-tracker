from __future__ import annotations

import argparse
import ctypes
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from platformdirs import user_log_dir
from zoneinfo import ZoneInfo

# ---------- Paths & logging ----------

APP_NAME = "cfb-tix"
TZ = ZoneInfo("America/New_York")

@dataclass
class Paths:
    repo_root: Path
    app_root: Path
    logs_dir: Path
    log_file: Path
    py_exe: Path

def detect_paths() -> Paths:
    """Resolve paths whether running from source or from installed copy."""
    here = Path(__file__).resolve()
    # Source layout: .../src/cfb_tix/daemon.py  -> repo root is here.parents[3] or [2]
    # Installed layout: %LocalAppData%/cfb-tix/app/src/cfb_tix/daemon.py
    # Walk up until we find pyproject.toml or a top-level 'src' sibling.
    cur = here
    repo = None
    for p in [cur.parents[i] for i in range(1, 6)]:
        if (p / "pyproject.toml").exists() or (p / "src").exists():
            repo = p if (p / "pyproject.toml").exists() else p
            break
    if repo is None:
        # Fallback to LocalAppData layout
        repo = Path(os.getenv("LOCALAPPDATA", "")) / APP_NAME / "app"

    app_root = repo
    logs_dir = Path(user_log_dir(APP_NAME, APP_NAME))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "cfb_tix.log"
    py_exe = Path(sys.executable)
    return Paths(repo, app_root, logs_dir, log_file, py_exe)

def setup_logging(log_file: Path) -> None:
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        log.addHandler(fh)

# ---------- helpers ----------

def notify(title: str, msg: str) -> None:
    """Best-effort user notification."""
    try:
        if sys.platform == "win32":
            MB_TOPMOST = 0x00040000
            ctypes.windll.user32.MessageBoxW(0, msg, title, 0x00000040 | MB_TOPMOST)
        elif sys.platform.startswith("linux"):
            subprocess.run(["notify-send", title, msg], check=False)
    except Exception:
        pass

def run_py_script(script_rel: str, cwd: Path, env: Optional[dict] = None) -> int:
    """Run a Python script via the current interpreter."""
    script = cwd / script_rel
    if not script.exists():
        logging.info("Script missing, skipping: %s", script_rel)
        return 0
    cmd = [sys.executable, str(script)]
    logging.info("Running: %s", " ".join(cmd))
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), env=env or os.environ.copy())
        return cp.returncode
    except Exception as e:
        logging.exception("Error running %s: %s", script_rel, e)
        notify("CFB-Tix job failed", f"{script_rel}\n{e}")
        return 1

# ---------- jobs ----------

def job_daily_snapshot(paths: Paths) -> None:
    run_py_script("src/builders/daily_snapshot.py", paths.app_root)

def job_train_model(paths: Paths) -> None:
    # adjust if your training script lives elsewhere
    run_py_script("src/modeling/train_model.py", paths.app_root)

def job_predict_price(paths: Paths) -> None:
    run_py_script("src/modeling/predict_price.py", paths.app_root)

def job_weekly_update(paths: Paths) -> None:
    # prefer builders/weekly_update.py; fallback to reports/generate_weekly_report.py
    if (paths.app_root / "src/builders/weekly_update.py").exists():
        run_py_script("src/builders/weekly_update.py", paths.app_root)
    else:
        run_py_script("src/reports/generate_weekly_report.py", paths.app_root)

def job_annual_setup(paths: Paths) -> None:
    run_py_script("src/builders/annual_setup.py", paths.app_root)

def job_daily_pull_push(paths: Paths) -> None:
    # Optional GitHub sync
    try:
        from cfb_tix.windows.data_sync import pull_then_push  # type: ignore
        updated, pushed = pull_then_push(verbose=False)
        logging.info("Daily pull/push -> updated=%s, pushed=%s", updated, pushed)
    except Exception as e:
        logging.warning("Daily pull/push skipped or failed: %s", e)

# ---------- main ----------

def schedule_all(sched: BackgroundScheduler, paths: Paths) -> None:
    # 00:00, 06:00, 12:00, 18:00
    sched.add_job(lambda: job_daily_snapshot(paths),
                  CronTrigger(hour="0,6,12,18", minute="0", timezone=TZ),
                  id="job_daily_snapshot", name="job_daily_snapshot", replace_existing=True)

    # 06:45 and 18:45
    sched.add_job(lambda: job_train_model(paths),
                  CronTrigger(hour="6,18", minute="45", timezone=TZ),
                  id="job_train_model", name="job_train_model", replace_existing=True)

    sched.add_job(lambda: job_predict_price(paths),
                  CronTrigger(hour="6,18", minute="45", timezone=TZ),
                  id="job_predict_price", name="job_predict_price", replace_existing=True)

    # Weekly: Wednesday 05:30
    sched.add_job(lambda: job_weekly_update(paths),
                  CronTrigger(day_of_week="wed", hour=5, minute=30, timezone=TZ),
                  id="job_weekly_update", name="job_weekly_update", replace_existing=True)

    # Annual: May 1 at 05:00
    sched.add_job(lambda: job_annual_setup(paths),
                  CronTrigger(month=5, day=1, hour=5, minute=0, timezone=TZ),
                  id="job_annual_setup", name="job_annual_setup", replace_existing=True)

    # Optional: Daily GH sync at 07:10
    sched.add_job(lambda: job_daily_pull_push(paths),
                  CronTrigger(hour=7, minute=10, timezone=TZ),
                  id="job_daily_pull_push", name="job_daily_pull_push", replace_existing=True)

def log_next_runs(sched: BackgroundScheduler) -> None:
    for j in sched.get_jobs():
        logging.info("JOB %-17s next: %s", j.id.replace("job_", ""), j.next_run_time.astimezone(TZ) if j.next_run_time else "n/a")

def main(argv: Optional[list[str]] = None) -> int:
    args = argparse.ArgumentParser(prog="cfb-tix")
    sub = args.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run", help="Run daemon (with --no-gui for headless)")
    p_run.add_argument("--no-gui", action="store_true", help="Do not launch the GUI")
    # default: run
    parsed = args.parse_args(argv)

    paths = detect_paths()
    setup_logging(paths.log_file)
    logging.info("CFB-Tix daemon starting up (root=%s)", paths.app_root)

    # Optional initial sync
    try:
        from cfb_tix.windows.data_sync import pull_then_push  # type: ignore
        updated, _ = pull_then_push(verbose=False)
        logging.info("Initial pull complete (updated=%s)", updated)
    except Exception as e:
        logging.info("Initial pull skipped: %s", e)

    sched = BackgroundScheduler(timezone=TZ)
    schedule_all(sched, paths)
    logging.info("Adding job tentatively -- it will be properly scheduled when the scheduler starts")
    sched.start()
    logging.info("Scheduler started")
    log_next_runs(sched)

    # Optionally launch GUI unless --no-gui
    if parsed.cmd == "run" and not p_run.parse_known_args(sys.argv[2:])[0].no_gui:
        gui_script = paths.app_root / "src/gui/ticket_predictor_gui.py"
        if gui_script.exists():
            try:
                subprocess.Popen([sys.executable, str(gui_script)], cwd=str(paths.app_root))
            except Exception as e:
                logging.warning("GUI failed to launch: %s", e)

    try:
        # Keep process alive
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        sched.shutdown(wait=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
