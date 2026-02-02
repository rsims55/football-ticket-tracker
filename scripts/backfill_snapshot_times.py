#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.builders.daily_snapshot import backfill_kickoff_times_from_schedule, WEEKLY_SCHEDULE_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill placeholder kickoff times (e.g., 3:00 AM) in price_snapshots.csv using weekly schedule."
    )
    parser.add_argument(
        "--snapshots",
        default="data/daily/price_snapshots.csv",
        help="Path to snapshots CSV (default: data/daily/price_snapshots.csv)",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        help="Path to weekly schedule CSV (default: auto-pick latest in data/weekly)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy next to the snapshots file before overwriting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print match diagnostics for schedule alignment.",
    )
    parser.add_argument(
        "--refresh-point-diff",
        action="store_true",
        help="Overwrite home/away last point diffs from the schedule.",
    )
    parser.add_argument(
        "--normalize-datetime",
        action="store_true",
        help="Standardize date_local/time_local using startDateEastern when available.",
    )
    args = parser.parse_args()

    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        raise SystemExit(f"Snapshots file not found: {snap_path}")

    df = pd.read_csv(snap_path, low_memory=False)
    before = df.get("time_local", pd.Series(dtype=str)).copy()

    schedule_path = args.schedule
    if schedule_path is None:
        weekly_dir = ROOT / "data" / "weekly"
        candidates = sorted(
            weekly_dir.glob("full_*_schedule*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"No schedule files found in {weekly_dir}")
        schedule_path = str(candidates[0])

    df = backfill_kickoff_times_from_schedule(df, schedule_path, debug=args.debug)
    if args.refresh_point_diff:
        from src.builders.daily_snapshot import refresh_point_diffs_from_schedule  # noqa: E402
        df = refresh_point_diffs_from_schedule(df, schedule_path, overwrite=True, debug=args.debug)
    if args.normalize_datetime:
        from src.builders.daily_snapshot import normalize_local_datetime  # noqa: E402
        df = normalize_local_datetime(df, overwrite=True)

    if args.backup:
        backup_path = snap_path.with_suffix(snap_path.suffix + ".bak")
        df.to_csv(backup_path, index=False)

    df.to_csv(snap_path, index=False)

    after = df.get("time_local", pd.Series(dtype=str))
    changed = (before != after).sum() if len(before) == len(after) else "n/a"
    print(f"Updated snapshots: {snap_path}")
    print(f"Rows with time_local changed: {changed}")


if __name__ == "__main__":
    main()
