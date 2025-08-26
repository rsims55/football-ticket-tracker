# src/modeling/evaluate_predictions.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Repo-locked paths (runs from anywhere)
# -----------------------------
_THIS = Path(__file__).resolve()
SRC_DIR = _THIS.parents[2]         # .../src
PROJ_DIR = SRC_DIR.parent          # repo root

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

def _under_repo(p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(PROJ_DIR.resolve())
    except AttributeError:
        return str(p.resolve()).startswith(str(PROJ_DIR.resolve()))

def _resolve_file(env_name: str, default_rel: Path) -> Path:
    env_val = os.getenv(env_name)
    if REPO_DATA_LOCK or not env_val:
        return PROJ_DIR / default_rel
    p = Path(env_val).expanduser()
    if _under_repo(p) or ALLOW_ESCAPE:
        return p
    print(f"🚫 {env_name} resolves outside repo → {p} ; forcing repo path")
    return PROJ_DIR / default_rel

PREDICTIONS_PATH = _resolve_file("PREDICTIONS_PATH", Path("data") / "predicted" / "predicted_prices_optimal.csv")
SNAPSHOTS_PATH   = _resolve_file("SNAPSHOTS_PATH",   Path("data") / "daily"     / "price_snapshots.csv")
ERROR_LOG_PATH   = _resolve_file("ERROR_LOG_PATH",   Path("data") / "predicted" / "evaluation_metrics.csv")
MERGED_OUTPUT    = _resolve_file("MERGED_OUTPUT",    Path("data") / "predicted" / "merged_eval_results.csv")
TRAIN_SCRIPT     = PROJ_DIR / "src" / "modeling" / "train_price_model.py"  # repo-absolute for retrain

print("[evaluate_predictions] Paths resolved:")
print(f"  PROJ_DIR:         {PROJ_DIR}")
print(f"  PREDICTIONS_PATH: {PREDICTIONS_PATH}")
print(f"  SNAPSHOTS_PATH:   {SNAPSHOTS_PATH}")
print(f"  ERROR_LOG_PATH:   {ERROR_LOG_PATH}")
print(f"  MERGED_OUTPUT:    {MERGED_OUTPUT}")
print(f"  TRAIN_SCRIPT:     {TRAIN_SCRIPT}")

PERCENT_ERROR_THRESHOLD = 0.05  # 5%
ERROR_FRACTION_TRIGGER = 0.5    # Retrain if >50% of games exceed threshold

def _compose_start_datetime(row) -> pd.Timestamp:
    """Combine date_local and (optional) time_local into Timestamp."""
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    return pd.to_datetime(dt_str, errors="coerce")

def evaluate_predictions():
    # Existence checks
    if not PREDICTIONS_PATH.exists() or not SNAPSHOTS_PATH.exists():
        print("❌ Prediction or snapshot file not found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    snap_df = pd.read_csv(SNAPSHOTS_PATH)

    # Parse datetimes
    pred_df["startDateEastern"] = pd.to_datetime(pred_df.get("startDateEastern"), errors="coerce")
    if "startDateEastern" not in snap_df.columns:
        snap_df["startDateEastern"] = snap_df.apply(_compose_start_datetime, axis=1)
    else:
        snap_df["startDateEastern"] = pd.to_datetime(snap_df["startDateEastern"], errors="coerce")

    # Only evaluate completed games (midnight cutoff local system time)
    today = pd.Timestamp.today().normalize()
    pred_df = pred_df[pred_df["startDateEastern"].notna() & (pred_df["startDateEastern"] < today)]
    snap_df = snap_df[snap_df["startDateEastern"].notna() & (snap_df["startDateEastern"] < today)]

    if pred_df.empty or snap_df.empty:
        print("⚠️ No completed games to evaluate yet.")
        return

    # Build actuals from snapshots
    if "event_id" in snap_df.columns:
        actual_df = (
            snap_df.groupby("event_id", as_index=False)
                   .agg(actual_lowest_price=("lowest_price", "min"))
        )
    else:
        # Fallback join by teams+date if event_id isn't available
        actual_df = (
            snap_df.assign(startDateEasternDate=snap_df["startDateEastern"].dt.date)
                   .groupby(["homeTeam", "awayTeam", "startDateEasternDate"], as_index=False)
                   .agg(actual_lowest_price=("lowest_price", "min"))
                   .rename(columns={"startDateEasternDate": "startDateEastern"})
        )

    # Align predictions for join
    pred_df = pred_df.copy()
    pred_df["startDateEasternDate"] = pred_df["startDateEastern"].dt.date

    if "event_id" in pred_df.columns and "event_id" in actual_df.columns:
        merged = pd.merge(pred_df, actual_df, on="event_id", how="inner")
    else:
        merged = pd.merge(
            pred_df,
            actual_df,
            left_on=["homeTeam", "awayTeam", "startDateEasternDate"],
            right_on=["homeTeam", "awayTeam", "startDateEastern"],
            how="inner",
        )
        merged.drop(columns=["startDateEastern"], inplace=True)
        merged.rename(columns={"startDateEasternDate": "startDateEastern"}, inplace=True)

    if merged.empty:
        print("⚠️ No new completed games to evaluate.")
        return

    # Remove already-evaluated games
    if ERROR_LOG_PATH.exists():
        existing_log = pd.read_csv(ERROR_LOG_PATH)
        if "event_id" in merged.columns and "event_id" in existing_log.columns:
            existing_keys = set(existing_log["event_id"].dropna().astype(str))
            merged = merged[~merged["event_id"].astype(str).isin(existing_keys)]
        else:
            existing_keys = set(
                zip(
                    existing_log.get("homeTeam", pd.Series(dtype=str)),
                    existing_log.get("awayTeam", pd.Series(dtype=str)),
                    pd.to_datetime(existing_log.get("startDateEastern", pd.Series(dtype=str)), errors="coerce").dt.date,
                )
            )
            merged["key"] = list(
                zip(
                    merged.get("homeTeam", pd.Series(dtype=str)),
                    merged.get("awayTeam", pd.Series(dtype=str)),
                    merged.get("startDateEastern", pd.Series(dtype="datetime64[ns]")),
                )
            )
            merged = merged[~merged["key"].isin(existing_keys)]
            merged.drop(columns=["key"], inplace=True, errors="ignore")

    if merged.empty:
        print("ℹ️ All completed games already evaluated.")
        return

    # Calculate errors
    merged["abs_error"] = np.abs(merged["predicted_lowest_price"] - merged["actual_lowest_price"])
    merged["percent_error"] = np.where(
        merged["actual_lowest_price"] > 0,
        merged["abs_error"] / merged["actual_lowest_price"],
        np.nan,
    )
    mae = mean_absolute_error(merged["actual_lowest_price"], merged["predicted_lowest_price"])
    rmse = mean_squared_error(merged["actual_lowest_price"], merged["predicted_lowest_price"], squared=False)

    print(f"✅ Evaluated {len(merged)} new games — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

    # Save merged eval results
    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT, index=False)

    # Append to log
    merged["timestamp"] = datetime.now().isoformat()
    summary_log = merged[
        [
            "timestamp",
            "homeTeam",
            "awayTeam",
            "startDateEastern",
            "predicted_lowest_price",
            "actual_lowest_price",
            "abs_error",
            "percent_error",
        ]
    ]
    if ERROR_LOG_PATH.exists():
        existing = pd.read_csv(ERROR_LOG_PATH)
        full_log = pd.concat([existing, summary_log], ignore_index=True)
    else:
        full_log = summary_log

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_log.to_csv(ERROR_LOG_PATH, index=False)
    print(f"📊 Logged {len(summary_log)} new game evaluations.")

    # CI-based retraining trigger
    error_fraction = (merged["percent_error"] > PERCENT_ERROR_THRESHOLD).mean()
    if error_fraction > ERROR_FRACTION_TRIGGER:
        print(f"⚠️ {error_fraction:.0%} of games exceed {int(PERCENT_ERROR_THRESHOLD*100)}% error — retraining model.")
        # Use the same interpreter and an absolute path to the script
        subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True)
    else:
        print(f"✅ Only {error_fraction:.0%} of games exceeded {int(PERCENT_ERROR_THRESHOLD*100)}% error — no retraining needed.")

if __name__ == "__main__":
    evaluate_predictions()
