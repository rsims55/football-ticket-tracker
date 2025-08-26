# src/cfb_tix/daemon.py
from __future__ import annotations

import argparse
import ctypes
import logging
import os
import subprocess
import sys
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from platformdirs import user_log_dir
from zoneinfo import ZoneInfo

# ---------- Optional Sync Disable -------
def _sync_disabled() -> bool:
    return os.getenv("CFB_TIX_DISABLE_SYNC", "0") == "1"


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
    # Source layout: .../src/cfb_tix/daemon.py -> repo_root has pyproject.toml
    repo: Optional[Path] = None
    for i in range(1, 6):
        p = here.parents[i]
        if (p / "pyproject.toml").exists():
            repo = p
            break
    if repo is None:
        # Fallback to LocalAppData layout if pyproject isn't found
        repo = Path(os.getenv("LOCALAPPDATA", "")) / APP_NAME / "app"

    # Use appauthor=False so we DON'T get the doubled "cfb-tix\\cfb-tix\\Logs"
    logs_dir = Path(user_log_dir(APP_NAME, appauthor=False))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "cfb_tix.log"
    return Paths(
        repo_root=repo,
        app_root=repo,      # app_root == repo_root for this project
        logs_dir=logs_dir,
        log_file=log_file,
        py_exe=Path(sys.executable),
    )

def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        root.addHandler(fh)

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

def _popen(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str]:
    """Run a process and return (exitcode, combined_output)."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )
    out, _ = proc.communicate()
    return proc.returncode, out

def _commit_if_dirty(repo_root: Path) -> bool:
    """Stage and commit tracked changes if any; return True if a commit was made."""
    rc, out = _popen(["git", "status", "--porcelain"], cwd=repo_root)
    if rc != 0 or not out.strip():
        return False
    _popen(["git", "add", "-A"], cwd=repo_root)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _popen(["git", "commit", "-m", f"automated snapshot sync ({ts})"], cwd=repo_root)
    return True

def _child_env_for_repo(paths: Paths) -> dict:
    """Environment for child scripts: repo-locked and with src on PYTHONPATH."""
    env = os.environ.copy()
    # Force repo-lock defaults unless caller explicitly overrides
    env.setdefault("REPO_DATA_LOCK", "1")
    env.setdefault("REPO_ALLOW_NON_REPO_OUT", "0")
    # Helpful for scripts that import via repo/src
    src_dir = str(paths.repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    if src_dir not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (src_dir + (os.pathsep + existing if existing else ""))
    return env

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

# ---------- Git sync (prefer your data_sync; else fallback to plain git) ----------

def _git_sync_safe(repo_root: Path) -> None:
    """
    Robust git sync:
      - pull --rebase --autostash (handles unstaged changes cleanly)
      - stage/commit any changes
      - push (no-op if up-to-date)
    Respects .gitignore.
    """
    def run(args: list[str]) -> tuple[int, str]:
        rc, out = _popen(args, cwd=repo_root)
        if rc != 0:
            logging.warning("git %s failed (rc=%s)\n%s", " ".join(args), rc, out)
        return rc, out

    rc, _ = run(["git", "rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        logging.info("Not a git repo; skipping sync.")
        return

    run(["git", "fetch", "--all"])
    run(["git", "pull", "--rebase", "--autostash"])

    rc, status = run(["git", "status", "--porcelain"])
    if rc == 0 and status.strip():
        run(["git", "add", "-A"])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rc, out = run(["git", "commit", "-m", f"automated snapshot sync ({ts})"])
        if rc != 0:
            logging.warning("git commit returned %s\n%s", rc, out)

    rc, out = run(["git", "push"])
    if rc != 0:
        logging.error("git push failed:\n%s", out)
        notify("CFB-Tix sync failed", "git push failed — see daemon log.")
        raise RuntimeError("git push failed")
    logging.info("✅ git sync complete.")

def do_sync(paths: Paths, label: str) -> None:
    """
    Commit locally (if dirty) then try repo helper (src -> pkg), else fallback to plain git.
    This guarantees new artifacts under data/ (except data/permanent/) get committed + pushed.
    """
    if _sync_disabled():
        logging.info("[%s] sync skipped (CFB_TIX_DISABLE_SYNC=1)", label)
        return
    committed = _commit_if_dirty(paths.repo_root)
    try:
        from src.cfb_tix.windows.data_sync import pull_then_push  # type: ignore
        updated, pushed = pull_then_push(verbose=False)
        logging.info("[%s] pull_then_push (src) -> updated=%s, pushed=%s, committed=%s",
                     label, updated, pushed, committed)
        return
    except Exception as e1:
        try:
            from cfb_tix.windows.data_sync import pull_then_push  # type: ignore
            updated, pushed = pull_then_push(verbose=False)
            logging.info("[%s] pull_then_push (pkg) -> updated=%s, pushed=%s, committed=%s",
                         label, updated, pushed, committed)
            return
        except Exception as e2:
            logging.info("[%s] data_sync unavailable (%s / %s); using plain git.", label, e1, e2)
            _git_sync_safe(paths.repo_root)

# ---------- jobs ----------

def job_daily_snapshot(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        run_py_script("src/builders/daily_snapshot.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "daily_snapshot")

def job_train_model(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        run_py_script("src/modeling/train_price_model.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "train_model")

def job_predict_price(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        run_py_script("src/modeling/predict_price.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "predict_price")

def job_weekly_update(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        # Prefer your current location: src/preparation/weekly_update.py
        if (paths.app_root / "src/preparation/weekly_update.py").exists():
            run_py_script("src/preparation/weekly_update.py", paths.app_root, env=env)
        elif (paths.app_root / "src/builders/weekly_update.py").exists():
            run_py_script("src/builders/weekly_update.py", paths.app_root, env=env)
        else:
            # Last resort (older report generator)
            run_py_script("src/reports/generate_weekly_report.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "weekly_update")

def job_annual_setup(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        run_py_script("src/builders/annual_setup.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "annual_setup")

def job_daily_pull_push(paths: Paths) -> None:
    try:
        do_sync(paths, "daily_pull_push")
    except Exception as e:
        logging.warning("Daily pull/push failed: %s", e)

def job_hourly_sync(paths: Paths) -> None:
    try:
        do_sync(paths, "hourly_sync")
    except Exception as e:
        logging.warning("Hourly sync failed: %s", e)

# Optional: run evaluation after predictions (enable schedule if desired)
def job_evaluate_predictions(paths: Paths) -> None:
    env = _child_env_for_repo(paths)
    try:
        run_py_script("src/modeling/evaluate_predictions.py", paths.app_root, env=env)
    finally:
        do_sync(paths, "evaluate_predictions")

# ---------- main ----------

def schedule_all(sched: BackgroundScheduler, paths: Paths) -> None:
    # 00:00, 06:00, 12:00, 18:00
    sched.add_job(lambda: job_daily_snapshot(paths),
                  CronTrigger(hour="0,6,12,18", minute="0", timezone=TZ),
                  id="job_daily_snapshot", name="job_daily_snapshot", replace_existing=True)

    # 06:45 and 18:45 — train & predict are separate jobs
    sched.add_job(lambda: job_train_model(paths),
                  CronTrigger(hour="6,18", minute="45", timezone=TZ),
                  id="job_train_model", name="job_train_model", replace_existing=True)

    sched.add_job(lambda: job_predict_price(paths),
                  CronTrigger(hour="6,18", minute="45", timezone=TZ),
                  id="job_predict_price", name="job_predict_price", replace_existing=True)

    # Optional: evaluate right after predict (uncomment to enable)
    # sched.add_job(lambda: job_evaluate_predictions(paths),
    #               CronTrigger(hour="6,18", minute="50", timezone=TZ),
    #               id="job_evaluate_predictions", name="job_evaluate_predictions", replace_existing=True)

    # Weekly: Wednesday 05:30
    sched.add_job(lambda: job_weekly_update(paths),
                  CronTrigger(day_of_week="wed", hour=5, minute=30, timezone=TZ),
                  id="job_weekly_update", name="job_weekly_update", replace_existing=True)

    # Annual: May 1 at 05:00
    sched.add_job(lambda: job_annual_setup(paths),
                  CronTrigger(month=5, day=1, hour=5, minute=0, timezone=TZ),
                  id="job_annual_setup", name="job_annual_setup", replace_existing=True)

    # Daily GH sync at 07:10
    sched.add_job(lambda: job_daily_pull_push(paths),
                  CronTrigger(hour=7, minute=10, timezone=TZ),
                  id="job_daily_pull_push", name="job_daily_pull_push", replace_existing=True)

    # Hourly safety sync (quarter past)
    sched.add_job(lambda: job_hourly_sync(paths),
                  CronTrigger(minute=15, timezone=TZ),
                  id="job_hourly_sync", name="job_hourly_sync", replace_existing=True)

def log_next_runs(sched: BackgroundScheduler) -> None:
    for j in sched.get_jobs():
        next_run = j.next_run_time.astimezone(TZ) if j.next_run_time else "n/a"
        logging.info("JOB %-22s next: %s", j.id.replace("job_", ""), next_run)

def main(argv: Optional[list[str]] = None) -> int:
    args = argparse.ArgumentParser(prog="cfb-tix")
    sub = args.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run", help="Run daemon (with --no-gui for headless)")
    p_run.add_argument("--no-gui", action="store_true", help="Do not launch the GUI")
    parsed = args.parse_args(argv)

    paths = detect_paths()
    setup_logging(paths.log_file)
    logging.info("CFB-Tix daemon starting up (root=%s)", paths.app_root)

    # Initial pull to ensure we're up to date before first jobs
    try:
        do_sync(paths, "initial_pull")
        logging.info("Initial pull complete.")
    except Exception as e:
        logging.info("Initial pull skipped/fallback failed: %s", e)

    # Coalesce late runs, prevent overlap, allow small grace
    sched = BackgroundScheduler(
        timezone=TZ,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 900},
    )
    schedule_all(sched, paths)
    logging.info("Scheduling jobs…")
    sched.start()

    # Kickoff: run key jobs once after startup (staggered to avoid overlap)
    from apscheduler.triggers.date import DateTrigger
    base = datetime.datetime.now(TZ) + datetime.timedelta(seconds=5)
    sched.add_job(lambda: job_daily_snapshot(paths),
                  DateTrigger(run_date=base),
                  id="kickoff_daily_snapshot", replace_existing=True)
    sched.add_job(lambda: job_train_model(paths),
                  DateTrigger(run_date=base + datetime.timedelta(seconds=45)),
                  id="kickoff_train_model", replace_existing=True)
    sched.add_job(lambda: job_predict_price(paths),
                  DateTrigger(run_date=base + datetime.timedelta(seconds=90)),
                  id="kickoff_predict_price", replace_existing=True)
    sched.add_job(lambda: job_weekly_update(paths),
                  DateTrigger(run_date=base + datetime.timedelta(seconds=135)),
                  id="kickoff_weekly_update", replace_existing=True)
    logging.info("Kickoff jobs scheduled to run immediately after startup.")

    logging.info("Scheduler started")
    log_next_runs(sched)

    # Optionally launch GUI unless --no-gui
    if parsed.cmd == "run" and not p_run.parse_known_args(sys.argv[2:])[0].no_gui:
        gui_script = paths.app_root / "src/gui/ticket_predictor_gui.py"
        if gui_script.exists():
            try:
                env = _child_env_for_repo(paths)
                subprocess.Popen([sys.executable, str(gui_script)], cwd=str(paths.app_root), env=env)
            except Exception as e:
                logging.warning("GUI failed to launch: %s", e)

    try:
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
