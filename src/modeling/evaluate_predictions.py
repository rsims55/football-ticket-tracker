# =============================
# FILE: src/modeling/evaluate_predictions.py
# PURPOSE: Evaluate predicted optimal prices vs. actual lowest snapshot prices.
#          Enhancements:
#            • Robust numeric coercion
#            • Event-ID–based joining & de-dup
#            • Carry-through of predictor columns (optimal_source/date/time)
#            • Percent-error logging and optional retraining trigger
#            • Timing deltas (predicted optimal vs actual lowest timestamp)
#            • Tie-breaker: choose actual-lowest snapshot closest to predicted time
# =============================
from __future__ import annotations
from zoneinfo import ZoneInfo

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
PREDICT_SCRIPT     = PROJ_DIR / "src" / "modeling" / "predict_price.py"

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

# -----------------------------
# Helpers
# -----------------------------
def _compose_start_datetime(row) -> pd.Timestamp:
    """Combine date_local and (optional) time_local into Timestamp."""
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    return pd.to_datetime(dt_str, errors="coerce")

def _coerce_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")

def _compose_snapshot_ts(row) -> pd.Timestamp:
    dc = str(row.get("date_collected", "")).strip()
    tc = str(row.get("time_collected", "")).strip()
    dt_str = f"{dc} {tc}".strip()
    return pd.to_datetime(dt_str, errors="coerce")

def _compose_predicted_optimal_ts(row) -> pd.Timestamp:
    """
    Combine:
      - optimal_purchase_date
      - optimal_purchase_time
    into a single predicted timestamp.
    """
    d = str(row.get("optimal_purchase_date", "")).strip()
    t = str(row.get("optimal_purchase_time", "")).strip()
    if d and d.lower() != "nan":
        if t and t.lower() != "nan":
            return pd.to_datetime(f"{d} {t}", errors="coerce")
        return pd.to_datetime(d, errors="coerce")
    return pd.NaT

# -----------------------------
# Main
# -----------------------------
def evaluate_predictions():
    # Existence checks
    if not PREDICTIONS_PATH.exists() or not SNAPSHOTS_PATH.exists():
        print("❌ Prediction or snapshot file not found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    snap_df = pd.read_csv(SNAPSHOTS_PATH)

    # Must have event_id to align
    if "event_id" not in pred_df.columns:
        print("❌ 'event_id' missing from predictions; cannot evaluate reliably.")
        return
    if "event_id" not in snap_df.columns:
        print("❌ 'event_id' missing from snapshots; cannot evaluate reliably.")
        return

    # Parse game start (ET). Use supplied startDateEastern where present.
    pred_df["startDateEastern"] = pd.to_datetime(pred_df.get("startDateEastern"), errors="coerce")
    if "startDateEastern" not in snap_df.columns:
        snap_df["startDateEastern"] = snap_df.apply(_compose_start_datetime, axis=1)
    else:
        snap_df["startDateEastern"] = pd.to_datetime(snap_df["startDateEastern"], errors="coerce")

    # Snapshot collection timestamp from data/daily/price_snapshots.csv
    snap_df["snapshot_ts"] = snap_df.apply(_compose_snapshot_ts, axis=1)

    # Only evaluate games that have kicked off (<= now ET)
    now_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).tz_localize(None)
    pred_df = pred_df[pred_df["startDateEastern"].notna() & (pred_df["startDateEastern"] <= now_et)]
    snap_df = snap_df[snap_df["startDateEastern"].notna() & (snap_df["startDateEastern"] <= now_et)]

    if pred_df.empty or snap_df.empty:
        print("⚠️ No kicked-off games to evaluate yet.")
        return

    # Numeric coercions
    snap_df["lowest_price_num"] = _coerce_numeric_col(snap_df, "lowest_price")
    pred_df["predicted_lowest_price_num"] = _coerce_numeric_col(pred_df, "predicted_lowest_price")

    # Keep only snapshots strictly before kickoff
    pre_k_snap = snap_df[
        snap_df["snapshot_ts"].notna()
        & snap_df["lowest_price_num"].notna()
        & snap_df["startDateEastern"].notna()
        & (snap_df["snapshot_ts"] < snap_df["startDateEastern"])
    ].copy()

    if pre_k_snap.empty:
        print("⚠️ No pre-kickoff snapshots available yet to score against.")
        return

    # Prepare predicted optimal timestamp per event (we'll need it for tie-breaks)
    pred_df["predicted_optimal_dt"] = pred_df.apply(_compose_predicted_optimal_ts, axis=1)
    pred_predtime = pred_df[["event_id", "predicted_optimal_dt"]].copy()

    # >>> LOWEST-PRICE SELECTION WITH TIE-BREAKER (vectorized, no groupby.apply) <<<
    # 1) Find min pre-kickoff price per event
    min_price = (
        pre_k_snap.groupby("event_id", as_index=False)["lowest_price_num"]
                .min()
                .rename(columns={"lowest_price_num": "min_price"})
    )

    # 2) Keep only rows that match the per-event minimum
    candidates = pre_k_snap.merge(min_price, on="event_id", how="inner")
    candidates = candidates[candidates["lowest_price_num"] == candidates["min_price"]].copy()

    # 3) Attach predicted optimal dt to each candidate row
    pred_df["predicted_optimal_dt"] = pred_df.apply(_compose_predicted_optimal_ts, axis=1)
    candidates = candidates.merge(
        pred_df[["event_id", "predicted_optimal_dt"]],
        on="event_id", how="left"
    )

    # 4) Build distance to predicted; if prediction is NaT, use +inf so fallback picks earliest snapshot
    candidates["distance_hours"] = np.where(
        candidates["predicted_optimal_dt"].notna(),
        (candidates["snapshot_ts"] - candidates["predicted_optimal_dt"]).abs() / np.timedelta64(1, "h"),
        np.inf
    )

    # 5) Sort by: event_id, distance (closest first), then snapshot_ts (earliest as tie-break)
    candidates.sort_values(["event_id", "distance_hours", "snapshot_ts"], inplace=True)

    # 6) Pick the first candidate per event (closest among ties; earliest if no prediction)
    chosen_rows = candidates.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    # 7) Build actuals table
    actual_min = (
        chosen_rows[["event_id", "min_price", "snapshot_ts"]]
        .rename(columns={"min_price": "actual_lowest_price", "snapshot_ts": "actual_lowest_dt"})
        .copy()
    )

    # Align predictions to actuals by event_id
    merged = pd.merge(pred_df, actual_min, on="event_id", how="inner")
    if merged.empty:
        print("⚠️ No kicked-off games with pre-kickoff snapshots to evaluate.")
        return

    # Event-ID–based dedup against prior log
    if ERROR_LOG_PATH.exists():
        try:
            existing_log = pd.read_csv(ERROR_LOG_PATH)
            if "event_id" in existing_log.columns:
                done_ids = set(existing_log["event_id"].dropna().astype(str))
                merged = merged[~merged["event_id"].astype(str).isin(done_ids)]
        except Exception as e:
            print(f"⚠️ Could not read prior log for dedup: {e}")

    if merged.empty:
        print("ℹ️ All eligible kicked-off games already evaluated.")
        return

    # Price errors
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

    # Timing deltas (in hours): + means predicted happens AFTER actual; - BEFORE.
    merged["timing_signed_error_hours"] = (
        (merged["predicted_optimal_dt"] - merged["actual_lowest_dt"])
        / np.timedelta64(1, "h")
    )
    merged["timing_abs_error_hours"] = merged["timing_signed_error_hours"].abs()

    # Metrics (price)
    mae = mean_absolute_error(
        merged["actual_lowest_price"], merged["predicted_lowest_price_num"]
    )
    try:
        rmse = mean_squared_error(
            merged["actual_lowest_price"], merged["predicted_lowest_price_num"], squared=False
        )
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(
            merged["actual_lowest_price"], merged["predicted_lowest_price_num"]
        )))

    print(f"✅ Evaluated {len(merged)} new kicked-off games — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

    # -----------------------------
    # Write merged output (one row per newly-evaluated event)
    # -----------------------------
    keep_cols = [
        "event_id",
        "startDateEastern",
        "week", "homeConference", "awayConference",
        "predicted_lowest_price", "predicted_lowest_price_num",
        "optimal_source", "optimal_purchase_date", "optimal_purchase_time",
        "predicted_optimal_dt",                 # predicted
        "actual_lowest_price", "actual_lowest_dt",  # actual (closest among ties)
        "abs_error", "percent_error",
        "timing_abs_error_hours", "timing_signed_error_hours",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_csv(MERGED_OUTPUT, index=False)

    # Append summary log (evaluation_metrics.csv)
    merged["timestamp"] = datetime.now().isoformat()
    summary_cols = [
        "timestamp", "event_id", "startDateEastern",
        "predicted_lowest_price_num", "actual_lowest_price",
        "abs_error", "percent_error",
        "optimal_source", "optimal_purchase_date", "optimal_purchase_time",
        "predicted_optimal_dt", "actual_lowest_dt",
        "timing_abs_error_hours", "timing_signed_error_hours",
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

    # Retrain trigger (unchanged)
    error_fraction = (merged["percent_error"] > PERCENT_ERROR_THRESHOLD).mean()
    if error_fraction > ERROR_FRACTION_TRIGGER:
        print(
            f"⚠️ {error_fraction:.0%} of games exceed {int(PERCENT_ERROR_THRESHOLD*100)}% error — retraining model."
        )
        try:
            subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True)
            subprocess.run([sys.executable, str(PREDICT_SCRIPT)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Retrain failed: {e}")
    else:
        print(
            f"✅ Only {error_fraction:.0%} of games exceeded {int(PERCENT_ERROR_THRESHOLD*100)}% error — no retraining needed."
        )

if __name__ == "__main__":
    evaluate_predictions()
