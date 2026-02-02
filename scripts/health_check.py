#!/usr/bin/env python3
"""Lightweight data health check for snapshots and schedule."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# ensure src import path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from utils.status import write_status

TZ = ZoneInfo("America/New_York")
DATA_DIR = ROOT / "data"
DAILY_SNAP = DATA_DIR / "daily" / "price_snapshots.csv"

REQUIRED_COLS = [
    "homeTeam",
    "awayTeam",
    "homeConference",
    "awayConference",
    "date_local",
    "time_local",
    "lowest_price",
    "days_until_game",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "date_collected",
    "time_collected",
    "home_last_point_diff_at_snapshot",
    "away_last_point_diff_at_snapshot",
    "week",
]


def _fmt_bool(v: bool) -> str:
    return "OK" if v else "FAIL"


def main() -> None:
    if not DAILY_SNAP.exists():
        write_status("health_check", "failed", "price_snapshots.csv not found")
        print("❌ price_snapshots.csv not found")
        return

    df = pd.read_csv(DAILY_SNAP, low_memory=False)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

    # Basic freshness: last collected within 7 days (if any rows)
    freshness_ok = True
    last_collected = None
    if "date_collected" in df.columns:
        last_collected = pd.to_datetime(df["date_collected"], errors="coerce").max()
        if pd.notna(last_collected):
            freshness_ok = last_collected >= (datetime.now(TZ).date() - timedelta(days=7))

    # Date/time format checks (spot-check parsing)
    date_ok = True
    time_ok = True
    if "date_local" in df.columns:
        date_ok = pd.to_datetime(df["date_local"], errors="coerce").notna().mean() > 0.95
    if "time_local" in df.columns:
        time_ok = pd.to_datetime(df["time_local"], errors="coerce").notna().mean() > 0.95

    status = "success" if (not missing_cols and freshness_ok and date_ok and time_ok) else "warning"
    detail = "Health check passed" if status == "success" else "Potential issues detected"

    write_status(
        "health_check",
        status,
        detail,
        {
            "missing_cols": missing_cols,
            "freshness_ok": freshness_ok,
            "date_format_ok": date_ok,
            "time_format_ok": time_ok,
            "rows": int(len(df)),
            "last_collected": str(last_collected) if last_collected is not None else "NA",
        },
    )

    print("🩺 Health Check")
    print(f"- Missing columns: {len(missing_cols)} -> {_fmt_bool(not missing_cols)}")
    if missing_cols:
        print(f"  {missing_cols}")
    print(f"- Freshness (<=7d): {_fmt_bool(freshness_ok)}")
    print(f"- date_local parse: {_fmt_bool(date_ok)}")
    print(f"- time_local parse: {_fmt_bool(time_ok)}")


if __name__ == "__main__":
    main()
