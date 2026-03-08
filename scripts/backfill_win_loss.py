#!/usr/bin/env python3
"""Backfill home_wins_at_snapshot / home_losses_at_snapshot /
away_wins_at_snapshot / away_losses_at_snapshot columns into existing
price-snapshot CSVs.

Usage (examples):
    python scripts/backfill_win_loss.py
    python scripts/backfill_win_loss.py --years 2025 2026
    python scripts/backfill_win_loss.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fetchers.schedule_fetcher import _load_win_loss_records  # noqa: E402


WEEKLY_DIR = ROOT / "data" / "weekly"
DAILY_DIR = ROOT / "data" / "daily"
ARCHIVE_DIR = DAILY_DIR / "backups"
ALIASES_PATH = ROOT / "data" / "permanent" / "team_aliases.json"

WIN_LOSS_COLS = [
    "home_wins_at_snapshot",
    "home_losses_at_snapshot",
    "away_wins_at_snapshot",
    "away_losses_at_snapshot",
]


def _normalize(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _build_tickpick_to_cfbd() -> dict[str, str]:
    """Build reverse map: tickpick_name_lower -> cfbd_name_lower.

    team_aliases.json maps cfbd_name -> tickpick_name.
    """
    import json
    if not ALIASES_PATH.exists():
        return {}
    with open(ALIASES_PATH, encoding="utf-8") as f:
        aliases = json.load(f)
    return {_normalize(v): _normalize(k) for k, v in aliases.items()}


def build_combined_map(years: list[int]) -> dict[tuple[str, int], tuple[float, float]]:
    """Merge win/loss maps from multiple seasons into one."""
    combined: dict[tuple[str, int], tuple[float, float]] = {}
    for year in years:
        year_map = _load_win_loss_records(year)
        if year_map:
            print(f"  Loaded {len(year_map)} (team, week) records for {year}.")
            combined.update(year_map)
        else:
            print(f"  No completed-games data found for {year}; skipping.")
    return combined


def lookup_wins(
    team: str | None,
    week: int | None,
    rec_map: dict,
    tp_to_cfbd: dict[str, str],
) -> float | None:
    if not team or pd.isna(week):
        return None
    norm = _normalize(team)
    cfbd = tp_to_cfbd.get(norm, norm)
    rec = rec_map.get((cfbd, int(week)))
    return rec[0] if rec is not None else None


def lookup_losses(
    team: str | None,
    week: int | None,
    rec_map: dict,
    tp_to_cfbd: dict[str, str],
) -> float | None:
    if not team or pd.isna(week):
        return None
    norm = _normalize(team)
    cfbd = tp_to_cfbd.get(norm, norm)
    rec = rec_map.get((cfbd, int(week)))
    return rec[1] if rec is not None else None


def backfill_csv(
    csv_path: Path,
    rec_map: dict[tuple[str, int], tuple[float, float]],
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Fill win/loss columns into a single snapshot CSV.

    Returns the number of rows updated.
    """
    if not csv_path.exists():
        print(f"  File not found, skipping: {csv_path}")
        return 0

    df = pd.read_csv(csv_path, low_memory=False)

    if "week" not in df.columns:
        print(f"  No 'week' column in {csv_path.name}; skipping.")
        return 0

    if not force:
        already_filled = all(c in df.columns for c in WIN_LOSS_COLS)
        if already_filled and df[WIN_LOSS_COLS].notna().all(axis=None):
            print(f"  {csv_path.name}: all win/loss columns already populated; use --force to overwrite.")
            return 0

    # Ensure columns exist.
    for col in WIN_LOSS_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["_week_int"] = pd.to_numeric(df["week"], errors="coerce")

    home_teams = df.get("homeTeam", pd.Series(dtype=str))
    away_teams = df.get("awayTeam", pd.Series(dtype=str))
    weeks = df["_week_int"]

    tp_to_cfbd = _build_tickpick_to_cfbd()

    df["home_wins_at_snapshot"] = [
        lookup_wins(t, w, rec_map, tp_to_cfbd) for t, w in zip(home_teams, weeks)
    ]
    df["home_losses_at_snapshot"] = [
        lookup_losses(t, w, rec_map, tp_to_cfbd) for t, w in zip(home_teams, weeks)
    ]
    df["away_wins_at_snapshot"] = [
        lookup_wins(t, w, rec_map, tp_to_cfbd) for t, w in zip(away_teams, weeks)
    ]
    df["away_losses_at_snapshot"] = [
        lookup_losses(t, w, rec_map, tp_to_cfbd) for t, w in zip(away_teams, weeks)
    ]

    df.drop(columns=["_week_int"], inplace=True, errors="ignore")

    filled = df[WIN_LOSS_COLS].notna().any(axis=1).sum()
    print(f"  {csv_path.name}: {filled} rows received at least one win/loss value.")

    if dry_run:
        print(f"  [dry-run] Would write {csv_path}")
        return filled

    # Backup before overwriting.
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = ARCHIVE_DIR / f"{csv_path.name}.{ts}.bak"
    shutil.copy2(csv_path, bak)
    print(f"  Backup written to {bak.name}")

    df.to_csv(csv_path, index=False)
    print(f"  Wrote updated file: {csv_path}")
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cumulative win/loss records into price-snapshot CSVs."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help="Season years to process (default: 2025 2026).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without modifying files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite win/loss columns even if already populated.",
    )
    args = parser.parse_args()

    print(f"Building win/loss record map for years: {args.years}")
    rec_map = build_combined_map(args.years)

    if not rec_map:
        print("No win/loss data available. Nothing to backfill.")
        return

    snapshot_files = []
    for year in args.years:
        # Current-season file (no year suffix for the active season)
        current = DAILY_DIR / "price_snapshots.csv"
        if current.exists():
            snapshot_files.append(current)
            break  # only add once

    for year in args.years:
        year_file = DAILY_DIR / f"price_snapshots_{year}.csv"
        snapshot_files.append(year_file)
        archive_file = DAILY_DIR / "archives" / f"price_snapshots_{year}.csv"
        snapshot_files.append(archive_file)

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in snapshot_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    total_updated = 0
    for csv_path in unique_files:
        print(f"\nProcessing: {csv_path}")
        total_updated += backfill_csv(csv_path, rec_map, dry_run=args.dry_run, force=args.force)

    print(f"\nDone. {total_updated} rows updated across {len(unique_files)} files.")


if __name__ == "__main__":
    main()
