# =============================
# FILE: src/modeling/evaluate_predictions.py
# PURPOSE: Evaluate predicted optimal prices vs. actual lowest snapshot prices.
#          Enhancements:
#            • Robust numeric coercion
#            • Event-ID–based joining & de-dup
#            • Carry-through of predictor columns (optimal_source/date/time)
#            • Percent-error logging and optional retraining trigger
# =============================
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

def _find_repo_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parent.parent

_THIS    = Path(__file__).resolve()
PROJ_DIR = _find_repo_root(_THIS)
SRC_DIR  = PROJ_DIR / "src"

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
    print(f"🚫 {env_name} outside repo → {p} ; forcing repo path")
    return PROJ_DIR / default_rel


PREDICTIONS_PATH = _resolve_file("PREDICTIONS_PATH", Path("data") / "predicted" / "predicted_prices_optimal.csv")
SNAPSHOTS_PATH   = _resolve_file("SNAPSHOTS_PATH",   Path("data") / "daily"     / "price_snapshots.csv")
ERROR_LOG_PATH   = _resolve_file("ERROR_LOG_PATH",   Path("data") / "predicted" / "evaluation_metrics.csv")
MERGED_OUTPUT    = _resolve_file("MERGED_OUTPUT",    Path("data") / "predicted" / "merged_eval_results.csv")
TRAIN_SCRIPT     = PROJ_DIR / "src" / "modeling" / "train_price_model.py"

print("[evaluate_predictions] Paths resolved:")
print(f"  PROJ_DIR:         {PROJ_DIR}")
print(f"  PREDICTIONS_PATH: {PREDICTIONS_PATH}")
print(f"  SNAPSHOTS_PATH:   {SNAPSHOTS_PATH}")
print(f"  ERROR_LOG_PATH:   {ERROR_LOG_PATH}")
print(f"  MERGED_OUTPUT:    {MERGED_OUTPUT}")
print(f"  TRAIN_SCRIPT:     {TRAIN_SCRIPT}")


# Error thresholds
PERCENT_ERROR_THRESHOLD = float(os.getenv("EVAL_PERCENT_ERROR_THRESHOLD", 0.05))  # default 5%
ERROR_FRACTION_TRIGGER  = float(os.getenv("EVAL_ERROR_FRACTION_TRIGGER", 0.5))    # retrain if >50% exceed threshold


def _compose_start_datetime(row) -> pd.Timestamp:
    """Combine date_local and (optional) time_local into Timestamp."""
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    return pd.to_datetime(dt_str, errors="coerce")


def _coerce_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df.get(col), errors="coerce")
    return s


def evaluate_predictions():
    # Existence checks
    if not PREDICTIONS_PATH.exists() or not SNAPSHOTS_PATH.exists():
        print("❌ Prediction or snapshot file not found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    snap_df = pd.read_csv(SNAPSHOTS_PATH)

    # Basic sanity: event_id should exist in both for clean evaluation
    if "event_id" not in pred_df.columns:
        print("❌ 'event_id' missing from predictions; cannot evaluate reliably.")
        return
    if "event_id" not in snap_df.columns:
        print("❌ 'event_id' missing from snapshots; cannot evaluate reliably.")
        return

    # Parse datetimes (game start)
    pred_df["startDateEastern"] = pd.to_datetime(pred_df.get("startDateEastern"), errors="coerce")
    if "startDateEastern" not in snap_df.columns:
        snap_df["startDateEastern"] = snap_df.apply(_compose_start_datetime, axis=1)
    else:
        snap_df["startDateEastern"] = pd.to_datetime(snap_df["startDateEastern"], errors="coerce")

    # Filter to completed games (midnight local)
    today = pd.Timestamp.today().normalize()
    pred_df = pred_df[pred_df["startDateEastern"].notna() & (pred_df["startDateEastern"] < today)]
    snap_df = snap_df[snap_df["startDateEastern"].notna() & (snap_df["startDateEastern"] < today)]

    if pred_df.empty or snap_df.empty:
        print("⚠️ No completed games to evaluate yet.")
        return

    # Build actuals from snapshots (min observed price per event)
    snap_df["lowest_price_num"] = _coerce_numeric_col(snap_df, "lowest_price")
    actual_df = (
        snap_df.dropna(subset=["lowest_price_num"]) \
               .groupby("event_id", as_index=False) \
               .agg(actual_lowest_price=("lowest_price_num", "min"))
    )

    # Align predictions
    pred_df = pred_df.copy()
    pred_df["predicted_lowest_price_num"] = _coerce_numeric_col(pred_df, "predicted_lowest_price")

    merged = pd.merge(pred_df, actual_df, on="event_id", how="inner")

    if merged.empty:
        print("⚠️ No new completed games to evaluate.")
        return

    # Remove already-evaluated games (prefer event_id-based dedup)
    if ERROR_LOG_PATH.exists():
        try:
            existing_log = pd.read_csv(ERROR_LOG_PATH)
            if "event_id" in existing_log.columns:
                done_ids = set(existing_log["event_id"].dropna().astype(str))
                merged = merged[~merged["event_id"].astype(str).isin(done_ids)]
        except Exception as e:
            print(f"⚠️ Could not read prior log for dedup: {e}")

    if merged.empty:
        print("ℹ️ All completed games already evaluated.")
        return

    # Errors & metrics
    merged = merged.dropna(subset=["predicted_lowest_price_num", "actual_lowest_price"]).copy()
    if merged.empty:
        print("⚠️ Nothing to score after numeric coercion.")
        return

    merged["abs_error"] = (merged["predicted_lowest_price_num"] - merged["actual_lowest_price"]).abs()
    merged["percent_error"] = np.where(
        merged["actual_lowest_price"] > 0,
        merged["abs_error"] / merged["actual_lowest_price"],
        np.nan,
    )

    mae  = mean_absolute_error(merged["actual_lowest_price"], merged["predicted_lowest_price_num"])
    rmse = mean_squared_error(merged["actual_lowest_price"], merged["predicted_lowest_price_num"], squared=False)

    print(f"✅ Evaluated {len(merged)} new games — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

    # Save merged_eval_results with helpful columns preserved
    keep_cols = [
        "event_id", "startDateEastern", "week",
        "homeConference", "awayConference",
        "predicted_lowest_price", "predicted_lowest_price_num",
        "optimal_source", "optimal_purchase_date", "optimal_purchase_time",
        "actual_lowest_price", "abs_error", "percent_error",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_csv(MERGED_OUTPUT, index=False)

    # Append summary to ERROR_LOG_PATH (use event_id for future dedup)
    merged["timestamp"] = datetime.now().isoformat()
    summary_cols = [
        "timestamp", "event_id", "startDateEastern", "week",
        "homeConference", "awayConference",
        "predicted_lowest_price_num", "actual_lowest_price",
        "abs_error", "percent_error", "optimal_source",
        "optimal_purchase_date", "optimal_purchase_time",
    ]
    summary_log = merged[summary_cols].copy()

    if ERROR_LOG_PATH.exists():
        try:
            existing = pd.read_csv(ERROR_LOG_PATH)
            full_log = pd.concat([existing, summary_log], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Could not append to prior log ({e}); rewriting log.")
            full_log = summary_log
    else:
        full_log = summary_log

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_log.to_csv(ERROR_LOG_PATH, index=False)
    print(f"📊 Logged {len(summary_log)} new game evaluations.")

    # CI-based retraining trigger
    error_fraction = (merged["percent_error"] > PERCENT_ERROR_THRESHOLD).mean()
    if error_fraction > ERROR_FRACTION_TRIGGER:
        print(
            f"⚠️ {error_fraction:.0%} of games exceed {int(PERCENT_ERROR_THRESHOLD*100)}% error — retraining model."
        )
        try:
            subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Retrain failed: {e}")
    else:
        print(
            f"✅ Only {error_fraction:.0%} of games exceeded {int(PERCENT_ERROR_THRESHOLD*100)}% error — no retraining needed."
        )


if __name__ == "__main__":
    evaluate_predictions()
