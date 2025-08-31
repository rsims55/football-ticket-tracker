# FILE: tools/repair_merge_temp_snapshots.py
# PURPOSE: Merge any stray _tmp/_temp snapshot CSVs, enrich with schedule/rankings,
#          then append+dedupe into data/daily/price_snapshots.csv.

from __future__ import annotations
import os, sys, shutil, glob
from pathlib import Path
from datetime import datetime

import pandas as pd

# --- Repo-root detection (works from anywhere) ---
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1] if (_THIS.name == "repair_merge_temp_snapshots.py") else Path.cwd()
# If launched outside repo, try to walk up until we see data/src
for p in [ROOT] + list(ROOT.parents):
    if (p / "data").exists() and (p / "src").exists():
        ROOT = p
        break

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Prefer the same paths used by daily_snapshot
DAILY_DIR       = ROOT / "data" / "daily"
SNAPSHOT_PATH   = DAILY_DIR / "price_snapshots.csv"
WEEKLY_SCHEDULE = ROOT / "data" / "weekly" / f"full_{datetime.now().year}_schedule.csv"

# --- Import only the helpers we need from your file ---
# NOTE: This relies on your existing functions in src/builders/daily_snapshot.py
try:
    from builders.daily_snapshot import (
        _apply_title_and_alias_fixes,
        _enrich_with_schedule_and_stadiums,
    )
except Exception as e:
    raise SystemExit(
        f"Couldn't import enrichment helpers from builders.daily_snapshot.\n"
        f"error = {e}\n"
        f"Tip: run from the repo root so 'src' is importable and scrapers/* is present."
    )

def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".__tmp__")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def main() -> None:
    os.makedirs(DAILY_DIR, exist_ok=True)

    # Find both patterns just in case (_tmp and _temp)
    patterns = [
        str(DAILY_DIR / "_tmp_snapshots_*.csv"),
        str(DAILY_DIR / "_temp_snapshots_*.csv"),
    ]
    temp_files = []
    for pat in patterns:
        temp_files.extend(glob.glob(pat))

    if not temp_files:
        print("No temp snapshot CSVs found. Nothing to do.")
        return

    print("Found temp snapshot CSVs:")
    for f in sorted(temp_files):
        print("  •", Path(f).name)

    # Read & concatenate all temp CSVs
    frames = []
    for f in sorted(temp_files):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  ⚠️ Skipping {f}: {e}")

    if not frames:
        print("No readable rows from temp files. Exiting.")
        return

    snap_all = pd.concat(frames, ignore_index=True)

    if snap_all.empty:
        print("Temp rows are empty after concat. Exiting.")
        return

    # Clean titles + alias fill (no new columns), then schedule/stadium/rankings enrichment
    print("→ Cleaning titles & applying aliases…")
    snap_all = _apply_title_and_alias_fixes(snap_all)

    print("→ Enriching with schedule/stadiums/rivalries/ranks…")
    snap_all = _enrich_with_schedule_and_stadiums(snap_all)

    # Build dedupe keys like daily_snapshot
    key_cols = []
    if "offer_url" in snap_all.columns:
        key_cols.append("offer_url")
    if "event_id" in snap_all.columns:
        key_cols.append("event_id")
    key_cols.extend(["date_collected", "time_collected"])
    key_cols = [c for c in key_cols if c in snap_all.columns]

    # Backup existing master, then append+dedupe
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if SNAPSHOT_PATH.exists():
        bak = SNAPSHOT_PATH.with_suffix(f".csv.bak-{ts}")
        shutil.copy2(SNAPSHOT_PATH, bak)
        print(f"📦 Backup saved → {bak.name}")

        existing = pd.read_csv(SNAPSHOT_PATH)
        combined = pd.concat([existing, snap_all], ignore_index=True)

        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            # Fallback: extremely safe dedupe on all columns
            combined = combined.drop_duplicates(keep="last")

        _write_csv_atomic(combined, SNAPSHOT_PATH)
        print(f"✅ Appended {len(snap_all)} rows; total now {len(combined)} → {SNAPSHOT_PATH.name}")
    else:
        _write_csv_atomic(snap_all, SNAPSHOT_PATH)
        print(f"✅ Wrote {len(snap_all)} rows → {SNAPSHOT_PATH.name}")

    # Move processed temps to an archive folder so you don’t reprocess
    archive_dir = DAILY_DIR / "archives"
    os.makedirs(archive_dir, exist_ok=True)
    for f in temp_files:
        src = Path(f)
        dst = archive_dir / f"{src.stem}__merged_{ts}{src.suffix}"
        try:
            src.rename(dst)
            print(f"🧹 Archived {src.name} → archives/{dst.name}")
        except Exception as e:
            print(f"⚠️ Could not archive {src.name}: {e}")

    # Freshness log (optional)
    try:
        df = pd.read_csv(SNAPSHOT_PATH)
        mx = pd.to_datetime(df.get("date_collected"), errors="coerce").max()
        print(f"🗓️  Max date_collected in master: {mx.date() if pd.notnull(mx) else 'NA'}")
    except Exception as e:
        print(f"⚠️ Freshness check failed: {e}")

if __name__ == "__main__":
    main()
