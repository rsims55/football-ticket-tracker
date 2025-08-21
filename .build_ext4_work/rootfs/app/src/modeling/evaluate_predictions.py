import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import subprocess

PREDICTIONS_PATH = "data/predicted/predicted_prices_optimal.csv"
SNAPSHOTS_PATH = "data/daily/price_snapshots.csv"
ERROR_LOG_PATH = "data/predicted/evaluation_metrics.csv"
MERGED_OUTPUT = "data/predicted/merged_eval_results.csv"

PERCENT_ERROR_THRESHOLD = 0.05  # 5%
ERROR_FRACTION_TRIGGER = 0.5    # Retrain if >50% of games exceed threshold


def _compose_start_datetime(row) -> pd.Timestamp:
    """Combine date_local and (optional) time_local into Timestamp."""
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    if time_str and time_str.lower() != "nan":
        dt_str = f"{date_str} {time_str}"
    else:
        dt_str = date_str
    return pd.to_datetime(dt_str, errors="coerce")


def evaluate_predictions():
    if not os.path.exists(PREDICTIONS_PATH) or not os.path.exists(SNAPSHOTS_PATH):
        print("❌ Prediction or snapshot file not found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    snap_df = pd.read_csv(SNAPSHOTS_PATH)

    # Parse datetimes
    pred_df["startDateEastern"] = pd.to_datetime(
        pred_df["startDateEastern"], errors="coerce"
    )
    if "startDateEastern" not in snap_df.columns:
        snap_df["startDateEastern"] = snap_df.apply(
            _compose_start_datetime, axis=1
        )
    else:
        snap_df["startDateEastern"] = pd.to_datetime(
            snap_df["startDateEastern"], errors="coerce"
        )

    # Only evaluate completed games
    today = pd.Timestamp.today().normalize()
    pred_df = pred_df[
        pred_df["startDateEastern"].notna()
        & (pred_df["startDateEastern"] < today)
    ]
    snap_df = snap_df[
        snap_df["startDateEastern"].notna()
        & (snap_df["startDateEastern"] < today)
    ]

    if pred_df.empty or snap_df.empty:
        print("⚠️ No completed games to evaluate yet.")
        return

    # Build actuals from snapshots
    if "event_id" in snap_df.columns:
        actual_df = (
            snap_df.groupby("event_id")
            .agg(actual_lowest_price=("lowest_price", "min"))
            .reset_index()
        )
    else:
        actual_df = (
            snap_df.assign(startDateEasternDate=snap_df["startDateEastern"].dt.date)
            .groupby(["homeTeam", "awayTeam", "startDateEasternDate"])
            .agg(actual_lowest_price=("lowest_price", "min"))
            .reset_index()
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
        merged.rename(
            columns={"startDateEasternDate": "startDateEastern"}, inplace=True
        )

    if merged.empty:
        print("⚠️ No new completed games to evaluate.")
        return

    # Remove already-evaluated games
    if os.path.exists(ERROR_LOG_PATH):
        existing_log = pd.read_csv(ERROR_LOG_PATH)
        if "event_id" in merged.columns and "event_id" in existing_log.columns:
            existing_keys = set(existing_log["event_id"].dropna().astype(str))
            merged = merged[~merged["event_id"].astype(str).isin(existing_keys)]
        else:
            existing_keys = set(
                zip(
                    existing_log["homeTeam"],
                    existing_log["awayTeam"],
                    pd.to_datetime(existing_log["startDateEastern"]).dt.date,
                )
            )
            merged["key"] = list(
                zip(
                    merged["homeTeam"],
                    merged["awayTeam"],
                    merged["startDateEastern"],
                )
            )
            merged = merged[~merged["key"].isin(existing_keys)]
            merged.drop(columns=["key"], inplace=True)

    if merged.empty:
        print("ℹ️ All completed games already evaluated.")
        return

    # Calculate errors
    merged["abs_error"] = np.abs(
        merged["predicted_lowest_price"] - merged["actual_lowest_price"]
    )
    merged["percent_error"] = np.where(
        merged["actual_lowest_price"] > 0,
        merged["abs_error"] / merged["actual_lowest_price"],
        np.nan,
    )
    mae = mean_absolute_error(
        merged["actual_lowest_price"], merged["predicted_lowest_price"]
    )
    rmse = mean_squared_error(
        merged["actual_lowest_price"],
        merged["predicted_lowest_price"],
        squared=False,
    )

    print(
        f"✅ Evaluated {len(merged)} new games — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}"
    )

    # Save merged eval results
    os.makedirs(os.path.dirname(MERGED_OUTPUT), exist_ok=True)
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
    if os.path.exists(ERROR_LOG_PATH):
        existing = pd.read_csv(ERROR_LOG_PATH)
        full_log = pd.concat([existing, summary_log], ignore_index=True)
    else:
        full_log = summary_log

    full_log.to_csv(ERROR_LOG_PATH, index=False)
    print(f"📊 Logged {len(summary_log)} new game evaluations.")

    # CI-based retraining trigger
    error_fraction = (merged["percent_error"] > PERCENT_ERROR_THRESHOLD).mean()
    if error_fraction > ERROR_FRACTION_TRIGGER:
        print(
            f"⚠️ {error_fraction:.0%} of games exceed {PERCENT_ERROR_THRESHOLD:.0%} error — retraining model."
        )
        subprocess.run(
            ["python", "src/modeling/train_price_model.py"], check=True
        )
    else:
        print(
            f"✅ Only {error_fraction:.0%} of games exceeded {PERCENT_ERROR_THRESHOLD:.0%} error — no retraining needed."
        )


if __name__ == "__main__":
    evaluate_predictions()
