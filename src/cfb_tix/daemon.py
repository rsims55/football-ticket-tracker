# src/cfb_tix/daemon.py
from __future__ import annotations

import os
import sys
import atexit
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    # Fallback if ever run on <3.9 with backport installed
    from backports.zoneinfo import ZoneInfo  # type: ignore

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from logging.handlers import RotatingFileHandler

# -------- Paths & logging --------
NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]  # project root (one above src/)
DATA = ROOT / "data"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

_file_handler = RotatingFileHandler(LOGS / "cfb_tix.log", maxBytes=5_000_000, backupCount=5)
_stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


# ===== Startup status announcer ==================================================
class StatusAnnouncer:
    """
    Prints/Logs a clear pre-GUI step-by-step status.
    Best-effort desktop notifications:
      • Linux: notify-send (if available)
      • Windows: PowerShell toast (if available)
    All failures are swallowed to avoid breaking startup.
    """
    def __init__(self, channel: str = "startup"):
        self.channel = channel

    def _fmt(self, msg: str) -> str:
        return f"[{self.channel}] {msg}"

    def _notify_linux(self, title: str, body: str) -> None:
        try:
            # Use notify-send if available
            if _which("notify-send"):
                subprocess.run(["notify-send", "--app-name=CFB Tickets", title, body],
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _notify_windows(self, title: str, body: str) -> None:
        try:
            # Try PowerShell toast (Windows 10+). Non-fatal if it fails.
            ps = r"""
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
            $xml = @"
            <toast>
              <visual>
                <binding template="ToastGeneric">
                  <text>{0}</text>
                  <text>{1}</text>
                </binding>
              </visual>
            </toast>
"@
            $xml = [string]::Format($xml, $args[0], $args[1])
            $doc = New-Object Windows.Data.Xml.Dom.XmlDocument
            $doc.LoadXml($xml)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("CFB Tickets")
            $notifier.Show($toast)
            """
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps, title, body],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def _notify(self, title: str, body: str) -> None:
        if sys.platform.startswith("linux"):
            self._notify_linux(title, body)
        elif os.name == "nt":
            self._notify_windows(title, body)
        # macOS not implemented — console/logs are sufficient

    def step(self, msg: str) -> None:
        text = self._fmt(f"▶ {msg}")
        print(text, flush=True)
        logger.info(text)
        self._notify("CFB Tickets: Starting…", msg)

    def ok(self, msg: str = "Done") -> None:
        text = self._fmt(f"✓ {msg}")
        print(text, flush=True)
        logger.info(text)

    def fail(self, msg: str) -> None:
        text = self._fmt(f"✗ {msg}")
        print(text, flush=True)
        logger.error(text)
        self._notify("CFB Tickets: Error", msg)

    def info(self, msg: str) -> None:
        text = self._fmt(msg)
        print(text, flush=True)
        logger.info(text)
# ================================================================================
def _which(name: str) -> Optional[str]:
    from shutil import which
    return which(name)


def run_module(mod: str, args: List[str] | None = None, cwd: Path | None = None) -> None:
    """
    Run a package module as: python -m <mod> [args...] with cwd defaulting to project root.
    Keeps the daemon alive on failure; logs exceptions.
    """
    args = args or []
    cwd = cwd or ROOT
    cmd = [sys.executable, "-m", mod] + args
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd))
    except subprocess.CalledProcessError:
        logger.exception("Module failed: %s", mod)
    except Exception:
        logger.exception("Unexpected error when running %s", mod)


def _year_now() -> int:
    return datetime.now(NY).year


# -------- Annual data --------
def check_and_build_annual() -> None:
    """
    Ensure data/annual has current-year rivalries_{Y}.csv and stadiums_{Y}.csv.
    If missing, run builders.annual_setup --year Y.
    After May 1 (ET) of the current year, delete prior year's copies if present.
    """
    year = _year_now()
    annual = DATA / "annual"
    annual.mkdir(parents=True, exist_ok=True)

    need = [annual / f"rivalries_{year}.csv", annual / f"stadiums_{year}.csv"]
    missing = [p for p in need if not p.exists()]
    if missing:
        logger.info("Annual missing: %s — running annual_setup", [p.name for p in missing])
        run_module("builders.annual_setup", ["--year", str(year)])

    # Roll previous year after May 1 of current year
    may1 = datetime(year, 5, 1, tzinfo=NY)
    if datetime.now(NY) >= may1:
        prev = year - 1
        for base in ("rivalries", "stadiums"):
            old = annual / f"{base}_{prev}.csv"
            if old.exists():
                try:
                    old.unlink()
                    logger.info("Removed previous year's %s", old.name)
                except Exception:
                    logger.exception("Failed removing %s", old)


# -------- Weekly data --------
def ensure_weekly_files_exist() -> None:
    """
    On startup only: if weekly files are missing, run builders.weekly_update once.
    """
    year = _year_now()
    weekly = DATA / "weekly"
    weekly.mkdir(parents=True, exist_ok=True)

    need = [weekly / f"full_{year}_schedule.csv", weekly / f"wiki_rankings_{year}.csv"]
    if not all(p.exists() for p in need):
        logger.info("Weekly files missing — running weekly_update once")
        run_module("builders.weekly_update")


def weekly_refresh() -> None:
    """
    Replace weekly files on schedule: every Wednesday at 06:00 ET.
    """
    logger.info("Weekly refresh (Wed 06:00 ET)")
    run_module("builders.weekly_update")


# -------- Daily snapshots --------
def run_daily_snapshot() -> None:
    logger.info("Daily snapshot")
    run_module("builders.daily_snapshot")


# -------- Modeling (daily) --------
def daily_model_update() -> None:
    logger.info("Daily model update: train -> predict")
    run_module("modeling.train_price_model")
    run_module("modeling.predict_price")


# -------- Sunday reporting --------
def sunday_report() -> None:
    logger.info("Sunday eval + weekly report + email")
    run_module("modeling.evaluate_predictions")
    run_module("reports.generate_weekly_report")
    run_module("reports.send_email")


# -------- GUI --------
def launch_gui() -> None:
    logger.info("Launching GUI…")
    # Spawn GUI separate so scheduler keeps running even if GUI closes/crashes
    subprocess.Popen([sys.executable, "-m", "gui.ticket_predictor_gui"], cwd=str(ROOT))


def _dump_schedule(sched: BackgroundScheduler) -> None:
    for job in sched.get_jobs():
        logger.info("JOB %-20s Next run: %s", job.id or job.name, job.next_run_time)


def main(no_gui: bool = False) -> None:
    # Ensure all relative paths resolve to the project root
    try:
        os.chdir(ROOT)
    except Exception:
        logger.exception("Failed to chdir to project root: %s", ROOT)

    # ---- Pre-GUI status announcer ----
    say = StatusAnnouncer()

    # ---- Startup actions (run immediately on program open) ----
    try:
        say.step("Annual data check/build")
        check_and_build_annual()
        say.ok("Annual data ready")
    except Exception:
        say.fail("Annual data check failed (continuing; see logs)")

    try:
        say.step("Ensure weekly files")
        ensure_weekly_files_exist()
        say.ok("Weekly files ready")
    except Exception:
        say.fail("Weekly ensure failed (continuing; see logs)")

    try:
        say.step("Daily snapshot")
        run_daily_snapshot()
        say.ok("Daily snapshot done")
    except Exception:
        say.fail("Daily snapshot failed (continuing; see logs)")

    try:
        say.step("Daily model update (train → predict)")
        daily_model_update()
        say.ok("Model updated")
    except Exception:
        say.fail("Model update failed (continuing; see logs)")

    # ---- Launch GUI (if requested) AFTER all steps announce ----
    if not no_gui:
        say.step("Launching GUI")
        launch_gui()
        say.ok("GUI launched")

    # ---- Scheduler wiring ----
    sched = BackgroundScheduler(
        timezone=NY,
        job_defaults={
            "coalesce": True,           # collapse missed runs into one
            "misfire_grace_time": 600,  # 10 minutes
            "max_instances": 1,
        },
    )

    # Annual check daily at 00:30 ET
    sched.add_job(check_and_build_annual, CronTrigger(hour=0, minute=30), id="annual_check")

    # Weekly refresh Wednesday 06:00 ET
    sched.add_job(weekly_refresh, CronTrigger(day_of_week="wed", hour=6, minute=0), id="weekly_refresh")

    # Daily snapshots: 00:00, 06:00, 12:00, 18:00 ET
    sched.add_job(run_daily_snapshot, CronTrigger(hour="0,6,12,18", minute=0), id="daily_snapshot")

    # Daily model: 06:00 ET
    sched.add_job(daily_model_update, CronTrigger(hour=6, minute=0), id="daily_model_update")

    # Sunday: 06:30 ET (after model & morning snapshot)
    sched.add_job(sunday_report, CronTrigger(day_of_week="sun", hour=6, minute=30), id="sunday_report")

    sched.start()
    atexit.register(lambda: sched.shutdown(wait=False))
    _dump_schedule(sched)

    # Keep process alive if GUI is disabled (GUI process keeps us alive otherwise)
    if no_gui:
        try:
            import time as _t
            while True:
                _t.sleep(3600)
        except KeyboardInterrupt:
            pass


def gui():
    """GUI-only launcher: no scheduler, no startup jobs."""
    subprocess.call([sys.executable, "-m", "gui.ticket_predictor_gui"], cwd=str(ROOT))


if __name__ == "__main__":
    main(no_gui=("--no-gui" in sys.argv))
