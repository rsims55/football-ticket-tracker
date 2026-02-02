#!/usr/bin/env python3
"""Long-running daemon loop that executes the scheduler periodically."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.status import write_status, read_status
from reports.send_email import send_markdown_report, _validate_env
from scripts import scheduler

INTERVAL_SECONDS = int(os.getenv("DAEMON_LOOP_SECONDS", "60"))
RUN_ALL_ON_START = os.getenv("DAEMON_RUN_ALL_ON_START", "1") == "1"


def _run(cmd: list[str]) -> None:
    import subprocess
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_all_once() -> None:
    _run(["python", "src/builders/annual_setup.py"])
    _run(["python", "src/builders/weekly_update.py"])
    _run(["python", "src/builders/daily_snapshot.py"])
    _run(["python", "src/modeling/train_catboost_min.py"])
    _run(["python", "src/reports/generate_weekly_report.py"])
    _run(["python", "src/reports/send_email.py"])


def _send_activation_email() -> None:
    try:
        _validate_env()
    except Exception:
        return
    status = read_status()
    lines = [
        "# ✅ CFB Ticket Tracker Daemon Activated",
        f"**Activated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Latest Pipeline Status",
    ]
    if not status:
        lines.append("- No status data found yet.")
    else:
        for k, v in status.items():
            lines.append(f"- **{k}**: **{v.get('status','unknown')}** — {v.get('detail','')}")
    body = "\n".join(lines)
    send_markdown_report(body, "✅ CFB Ticket Tracker Daemon Activated")


def main() -> None:
    write_status("daemon", "running", f"Loop active every {INTERVAL_SECONDS}s")
    if RUN_ALL_ON_START:
        try:
            _run_all_once()
        except Exception as e:
            write_status("daemon", "failed", f"Startup run failed: {e}")
        _send_activation_email()
    while True:
        try:
            scheduler.main()
        except Exception as e:
            write_status("daemon", "failed", f"Loop error: {e}")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
