"""
Generate optimal ticket purchase predictions for all upcoming games.

Reuses the CatBoost gap_pct trajectory engine from gui/predict_trajectory.py.
For each upcoming game it scans a time grid from now to kickoff, then picks the
minimum predicted price as the floor and its timestamp as the optimal buy time.

Output: data/predicted/predicted_prices_optimal.csv
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on sys.path so utils and gui are importable
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.status import write_status

# ---- Repo-locked paths ----
def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parent.parent

_THIS = Path(__file__).resolve()
PROJ_DIR = _find_repo_root(_THIS)

YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
SNAPSHOT_DIR = PROJ_DIR / "data" / "daily"
OUTPUT_PATH = PROJ_DIR / "data" / "predicted" / "predicted_prices_optimal.csv"
SCAN_STEP_HOURS = int(os.getenv("PREDICT_STEP_HOURS", "6"))
SCAN_MAX_HOURS = int(os.getenv("PREDICT_MAX_HOURS", "2500"))

_POSTSEASON_RE = re.compile(
    r"\b(bowl|playoff|first round|quarterfinal|semifinal|final|championship|cfp)\b",
    flags=re.IGNORECASE,
)

# ---- Helpers ----

def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".__tmp__")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _is_postseason(df: pd.DataFrame) -> pd.Series:
    if "is_postseason" in df.columns:
        return df["is_postseason"].fillna(False).astype(bool)
    if "title" in df.columns:
        return df["title"].fillna("").str.contains(_POSTSEASON_RE)
    return pd.Series(False, index=df.index)


def _infer_kickoff(row: dict) -> pd.Timestamp:
    for key in ("startDateEastern", "date_local"):
        v = row.get(key)
        if v is not None and pd.notna(v):
            dt = pd.to_datetime(v, errors="coerce")
            if pd.notna(dt):
                try:
                    return dt.tz_localize(None)
                except Exception:
                    return dt
    return pd.NaT


def _infer_collected_dt(df: pd.DataFrame) -> pd.Series:
    for c in ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            if ts.notna().any():
                return ts
    if "date_collected" in df.columns:
        date_dt = pd.to_datetime(df["date_collected"], errors="coerce")
        if "time_collected" in df.columns:
            time_td = pd.to_timedelta(df["time_collected"].astype(str).str.strip(), errors="coerce")
            ts = date_dt.dt.normalize() + time_td
            if ts.notna().any():
                return ts
        return date_dt
    return pd.Series(pd.NaT, index=df.index)


def _load_snapshots() -> pd.DataFrame:
    year_path = SNAPSHOT_DIR / f"price_snapshots_{YEAR}.csv"
    combined_path = SNAPSHOT_DIR / "price_snapshots.csv"
    if year_path.exists():
        df = pd.read_csv(year_path, low_memory=False)
        needs_filter = False
    elif combined_path.exists():
        df = pd.read_csv(combined_path, low_memory=False)
        needs_filter = True
    else:
        raise FileNotFoundError(f"No snapshot CSV found in {SNAPSHOT_DIR}")

    df["lowest_price"] = pd.to_numeric(df.get("lowest_price"), errors="coerce")
    df["collected_dt"] = _infer_collected_dt(df)
    df = df[~_is_postseason(df)].copy()

    if needs_filter and "date_local" in df.columns:
        gd = pd.to_datetime(df["date_local"], errors="coerce")
        df = df[(gd >= f"{YEAR}-03-01") & (gd < f"{YEAR+1}-02-01")].copy()

    if "event_id" not in df.columns:
        raise ValueError("Snapshot CSV missing 'event_id' column.")
    df["event_id"] = df["event_id"].astype(str)
    return df


def _observed_min_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """Ever-observed minimum price per event with its timestamp."""
    tmp = df[df["lowest_price"].notna()].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["event_id", "observed_lowest_price_num", "observed_lowest_dt"])
    idx = tmp.groupby("event_id")["lowest_price"].idxmin()
    mins = tmp.loc[idx, ["event_id", "lowest_price", "collected_dt"]].copy()
    mins = mins.rename(columns={
        "lowest_price": "observed_lowest_price_num",
        "collected_dt": "observed_lowest_dt",
    })
    mins["observed_lowest_price_num"] = mins["observed_lowest_price_num"].round(2)
    mins["observed_lowest_dt"] = mins["observed_lowest_dt"].apply(
        lambda x: x.isoformat() if pd.notna(x) else ""
    )
    return mins.reset_index(drop=True)


def _latest_row_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event_id using the most recent snapshot (freshest win/loss, rank, price)."""
    return (
        df.sort_values("collected_dt", ascending=False)
          .drop_duplicates(subset="event_id", keep="first")
          .copy()
    )


# ---- Main ----

def main() -> None:
    from gui.predict_trajectory import predict_trajectory

    print(f"[predict_price] Loading {YEAR} snapshots …")
    df = _load_snapshots()
    print(f"[predict_price]   {len(df):,} rows, {df['event_id'].nunique():,} events")

    obs = _observed_min_per_event(df)
    games = _latest_row_per_event(df)

    now = pd.Timestamp.now()
    results = []
    skipped = 0

    for _, row in games.iterrows():
        row_dict = row.to_dict()
        kickoff = _infer_kickoff(row_dict)

        # Skip past games (completed more than 1 day ago) and games with no kickoff
        if pd.isna(kickoff) or kickoff < now - pd.Timedelta(days=1):
            skipped += 1
            continue

        # Skip if no current price (gap_pct can't be converted to absolute dollars)
        if pd.isna(row.get("lowest_price")):
            skipped += 1
            continue

        hours_to_kickoff = max(1.0, (kickoff - now).total_seconds() / 3600.0)
        hours_back = min(hours_to_kickoff, float(SCAN_MAX_HOURS))

        try:
            _, summary = predict_trajectory(
                row_dict,
                hours_back=int(hours_back),
                step_hours=SCAN_STEP_HOURS,
            )
        except Exception as e:
            skipped += 1
            print(f"[predict_price]   SKIP {row.get('homeTeam','?')} vs {row.get('awayTeam','?')}: {e}")
            continue

        pred_min = summary.get("predicted_min_price", np.nan)
        pred_time = summary.get("predicted_min_time", pd.NaT)

        try:
            if not np.isfinite(float(pred_min)):
                skipped += 1
                continue
        except (TypeError, ValueError):
            skipped += 1
            continue

        opt_date = pred_time.date().isoformat() if pd.notna(pred_time) else ""
        opt_time = pred_time.strftime("%H:%M") if pd.notna(pred_time) else ""

        results.append({
            "event_id": row.get("event_id"),
            "homeTeam": row.get("homeTeam"),
            "awayTeam": row.get("awayTeam"),
            "startDateEastern": str(kickoff),
            "week": row.get("week"),
            "homeConference": row.get("homeConference"),
            "awayConference": row.get("awayConference"),
            "predicted_lowest_price": round(float(pred_min), 2),
            "optimal_purchase_date": opt_date,
            "optimal_purchase_time": opt_time,
            "optimal_source": "catboost",
        })

    if not results:
        msg = "No predictions generated — no upcoming games with price data."
        print(f"[predict_price] ⚠️  {msg}")
        write_status("predict_price", "failed", msg)
        return

    out = pd.DataFrame(results)
    out = out.merge(obs, on="event_id", how="left")

    _write_atomic(out, OUTPUT_PATH)
    n = len(out)
    print(f"[predict_price] ✅ {n} predictions → {OUTPUT_PATH}")
    if skipped:
        print(f"[predict_price]    {skipped} skipped (past / no price / no kickoff)")

    write_status(
        "predict_price",
        "success",
        f"{n} predictions generated for {YEAR}",
        {"n_predicted": n, "n_skipped": skipped},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_status("predict_price", "failed", str(e))
        raise
