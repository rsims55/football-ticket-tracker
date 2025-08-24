# src/cfb_tix/daemon.py
from __future__ import annotations
import argparse, logging, os, signal, sys, threading
from pathlib import Path

def _setup_logging():
    log_dir = Path("logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cfb_tix.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("CFB-Tix daemon starting up")

def _headless_loop():
    """Run background jobs; on Windows also do daily snapshots pull→push if token provided."""
    _setup_logging()
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception as e:
        logging.exception("apscheduler not available: %s", e)
        # keep a passive loop so Startup shortcut still holds the process
        _block_until_signal()
        return

    sched = BackgroundScheduler()
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
            logging.exception("Daily pull_push failed")

    initial_sync()
    sched.add_job(daily_pull_push, CronTrigger(hour=7, minute=10))
    sched.start()
    try:
        _block_until_signal()
    finally:
        sched.shutdown(wait=False)
        logging.info("CFB-Tix daemon shut down")

def _block_until_signal():
    stop = threading.Event()
    def handler(*_): stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, handler)
        except Exception: pass
    stop.wait()

def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="cfb-tix", add_help=True)
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the ticket tracker daemon")
    g = p_run.add_mutually_exclusive_group()
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
        # GUI lives at src/gui/ in your repo
        from gui.ticket_predictor_gui import main as gui_main
        gui_main()
    else:
        _headless_loop()

if __name__ == "__main__":
    main()
