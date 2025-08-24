# src/cfb_tix/daemon.py
from __future__ import annotations
import argparse, logging, os, signal, sys, threading
from pathlib import Path

import subprocess
from contextlib import suppress
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None
with suppress(Exception):
    import ctypes  # Windows MessageBox
with suppress(Exception):
    import tkinter as _tk
    from tkinter import messagebox as _mb


def _setup_logging():
    log_dir = Path("logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cfb_tix.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("CFB-Tix daemon starting up")

# --- Failure pop-up & runner helpers (non-breaking additions) ---
def notify_failure(title: str, message: str) -> None:
    """Best-effort pop-up for failures; falls back to logging."""
    # Windows MessageBox
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
        return
    except Exception:
        pass
    # Linux notify-send
    try:
        if os.getenv("DISPLAY"):
            subprocess.run(["notify-send", title, message], check=False)
            return
    except Exception:
        pass
    # Tkinter dialog
    try:
        if os.getenv("DISPLAY") or sys.platform == "win32":
            root = _tk.Tk(); root.withdraw()
            _mb.showerror(title, message)
            root.destroy()
            return
    except Exception:
        pass
    logging.error("POPUP FAILED [%s]: %s", title, message)

def _proj_root() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "..").resolve())

def _py() -> str:
    return sys.executable or "python"

def run_script(script_rel: str, *args: str) -> None:
    """Run a repo script; pop-up + raise on failure."""
    proj = _proj_root()
    script = os.path.join(proj, script_rel)
    cmd = [_py(), script, *args]
    logging.info("Running: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, cwd=proj, capture_output=True, text=True)
        if res.stdout:
            logging.info(res.stdout.strip())
        if res.stderr:
            logging.warning(res.stderr.strip())
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()[:2000]
            notify_failure("CFB-Tix task failed", f"{os.path.basename(script)} exited {res.returncode}\n\n{msg}")
            raise RuntimeError(f"{script_rel} failed ({res.returncode}): {msg}")
    except Exception as e:
        notify_failure("CFB-Tix task crashed", f"{os.path.basename(script)}\n\n{e}")
        raise


def _headless_loop():
    """
    Keeps the process alive in headless mode if scheduler can't be initialized.
    """
    _block_until_signal()

def _block_until_signal():
    stop = threading.Event()
    def handler(*_): stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass
    stop.wait()

def _launch_gui():
    """
    Launches the PyQt5 GUI if requested.
    """
    try:
        # GUI lives at src/gui/ in your repo
        from cfb_tix.daemon import gui  # if you expose it here
    except Exception:
        # Fallback to running the GUI module by path
        proj = _proj_root()
        gui_path = os.path.join(proj, "src", "gui", "ticket_predictor_gui.py")
        if not os.path.exists(gui_path):
            raise RuntimeError(f"GUI not found at {gui_path}")
        subprocess.Popen([_py(), gui_path], cwd=proj)

# --- Job wrappers ---
DAILY_SNAPSHOT = "src/builders/daily_snapshot.py"
TRAIN_MODEL    = "src/modeling/train_price_model.py"
PREDICT_PRICE  = "src/modeling/predict_price.py"
WEEKLY_SCRIPT  = "src/reports/weekly_update.py"
if not os.path.exists(os.path.join(_proj_root(), WEEKLY_SCRIPT)):
    WEEKLY_SCRIPT = "src/reports/generate_weekly_report.py"
ANNUAL_SETUP   = "src/annual_setup.py"
if not os.path.exists(os.path.join(_proj_root(), ANNUAL_SETUP)):
    ANNUAL_SETUP = "src/fetchers/annual_setup.py"

def job_daily_snapshot():
    run_script(DAILY_SNAPSHOT)

def job_train_model():
    run_script(TRAIN_MODEL)

def job_predict_price():
    run_script(PREDICT_PRICE)

def job_weekly_update():
    run_script(WEEKLY_SCRIPT)

def job_annual_setup():
    run_script(ANNUAL_SETUP)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception as e:
        logging.exception("apscheduler not available: %s", e)
        # keep a passive loop so Startup shortcut still holds the process
        _block_until_signal()
        return 0

    sched = BackgroundScheduler(timezone=_TZ or "America/New_York")

    # — Initial sync at startup (pull latest; upload if token exists)
    def initial_sync():
        try:
            # Windows-specific helper lives here per your tree
            from cfb_tix.windows.data_sync import pull_then_push, sync_down_latest_snapshots
            # Pull first so local has freshest
            pulled = sync_down_latest_snapshots(verbose=False)
            logging.info("Initial pull complete (updated=%s)", pulled)
            pushed = False
            if os.getenv("SNAP_GH_TOKEN") or os.getenv("GITHUB_TOKEN"):
                pulled, pushed = pull_then_push(verbose=False)
                logging.info("Initial pull_then_push done (pulled=%s, pushed=%s)", pulled, pushed)
        except Exception:
            logging.exception("Initial sync failed")

    # — Daily at 06:10 local time (matches your PS default)
    def daily_pull_push():
        try:
            from cfb_tix.windows.data_sync import pull_then_push
            pulled, pushed = pull_then_push(verbose=False)
            logging.info("Daily pull_then_push (pulled=%s, pushed=%s)", pulled, pushed)
        except Exception:
            logging.exception("daily_pull_push failed")

    parser = argparse.ArgumentParser(prog="cfb-tix")
    sub = parser.add_subparsers(dest="cmd")
    g = sub.add_parser("run", help="Run scheduled tasks")
    g.add_argument("--no-gui", action="store_true", help="Run headless (default)")
    g.add_argument("--with-gui", action="store_true", help="Launch the GUI instead")

    # Allow `cfb-tix` with no subcommand to behave like `cfb-tix run --no-gui`
    if not argv:
        argv = ["run", "--no-gui"]
    args = parser.parse_args(argv)

    if args.cmd != "run":
        # default again if someone passes odd args
        args = parser.parse_args(["run", "--no-gui"])

    if args.with_gui and not args.no_gui:
        _launch_gui()
        return 0

    # ---- Existing schedules you already had (preserved) ----
    # Initial sync:
    initial_sync()
    # Your existing example: daily at 07:10 (adjust if you changed earlier)
    try:
        from apscheduler.triggers.cron import CronTrigger
        sched.add_job(daily_pull_push, CronTrigger(hour=7, minute=10))
    except Exception:
        logging.exception("Failed to schedule daily_pull_push")

    # -------- NEW requested schedules --------
    # 4x daily snapshots: 06:00, 12:00, 18:00, 00:00
    try:
        sched.add_job(job_daily_snapshot, CronTrigger(hour="0,6,12,18", minute=0, timezone=_TZ or "America/New_York"),
                      id="daily_snapshot", replace_existing=True, coalesce=True, misfire_grace_time=3600)
    except Exception as e:
        logging.exception("Failed to schedule daily_snapshot: %s", e)

    # Modeling twice daily: 06:45 and 18:45
    for jid, fn in (("train_model", job_train_model), ("predict_price", job_predict_price)):
        try:
            sched.add_job(fn, CronTrigger(hour="6,18", minute=45, timezone=_TZ or "America/New_York"),
                          id=jid, replace_existing=True, coalesce=True, misfire_grace_time=3600)
        except Exception as e:
            logging.exception("Failed to schedule %s: %s", jid, e)

    # Weekly update: Wednesday 05:30
    try:
        sched.add_job(job_weekly_update, CronTrigger(day_of_week="wed", hour=5, minute=30, timezone=_TZ or "America/New_York"),
                      id="weekly_update", replace_existing=True, coalesce=True, misfire_grace_time=7200)
    except Exception as e:
        logging.exception("Failed to schedule weekly_update: %s", e)

    # Annual setup: May 1 at 05:00
    try:
        sched.add_job(job_annual_setup, CronTrigger(month=5, day=1, hour=5, minute=0, timezone=_TZ or "America/New_York"),
                      id="annual_setup", replace_existing=True, coalesce=True, misfire_grace_time=86400)
    except Exception as e:
        logging.exception("Failed to schedule annual_setup: %s", e)

    # Start scheduler
    sched.start()

    # Run two tasks at launch (in addition to your existing initial sync)
    try:
        logging.info("Running daily_snapshot immediately at launch…")
        job_daily_snapshot()
    except Exception:
        logging.exception("daily_snapshot failed at launch")
    try:
        logging.info("Running weekly_update immediately at launch…")
        job_weekly_update()
    except Exception:
        logging.exception("weekly_update failed at launch")

    try:
        _block_until_signal()
    finally:
        with suppress(Exception):
            sched.shutdown(wait=False)
        logging.info("CFB-Tix daemon shut down")
    return 0


def gui():
    """Entry point if you expose cfb-tix-gui to run GUI directly via argparse."""
    _launch_gui()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:
        logging.exception("Fatal error in daemon: %s", e)
        try:
            notify_failure("CFB-Tix daemon crashed", str(e))
        except Exception:
            pass
        raise
