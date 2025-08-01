import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import subprocess

PREDICTIONS_PATH = "data/predicted_prices_optimal.csv"
SNAPSHOTS_PATH = "data/price_snapshots.csv"
ERROR_LOG_PATH = "data/evaluation_metrics.csv"
MERGED_OUTPUT = "data/merged_eval_results.csv"
PERCENT_ERROR_THRESHOLD = 0.05  # 5%
ERROR_FRACTION_TRIGGER = 0.5    # Retrain if over 50% of games exceed threshold

def evaluate_predictions():
    if not os.path.exists(PREDICTIONS_PATH) or not os.path.exists(SNAPSHOTS_PATH):
        print("❌ Prediction or snapshot file not found.")
        return

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    snap_df = pd.read_csv(SNAPSHOTS_PATH)

    pred_df["startDateEastern"] = pd.to_datetime(pred_df["startDateEastern"]).dt.date
    snap_df["startDateEastern"] = pd.to_datetime(snap_df["startDateEastern"]).dt.date

    today = datetime.now().date()
    pred_df = pred_df[pred_df["startDateEastern"] < today]
    snap_df = snap_df[snap_df["startDateEastern"] < today]

    actual_df = (
        snap_df.groupby(["homeTeam", "awayTeam", "startDateEastern"])
        .agg(actual_lowest_price=("lowest_price", "min"))
        .reset_index()
    )

    merged = pd.merge(
        pred_df,
        actual_df,
        on=["homeTeam", "awayTeam", "startDateEastern"],
        how="inner"
    )

    if merged.empty:
        print("⚠️ No new completed games to evaluate.")
        return

    # Remove already-evaluated games
    if os.path.exists(ERROR_LOG_PATH):
        existing_log = pd.read_csv(ERROR_LOG_PATH)
        existing_games = set(zip(
            existing_log["homeTeam"],
            existing_log["awayTeam"],
            pd.to_datetime(existing_log["startDateEastern"]).dt.date
        ))
        merged["key"] = list(zip(merged["homeTeam"], merged["awayTeam"], merged["startDateEastern"]))
        merged = merged[~merged["key"].isin(existing_games)]
        merged.drop(columns=["key"], inplace=True)

    if merged.empty:
        print("ℹ️ All completed games already evaluated.")
        return

    merged["abs_error"] = np.abs(merged["predicted_lowest_price"] - merged["actual_lowest_price"])
    merged["percent_error"] = merged["abs_error"] / merged["actual_lowest_price"]
    mae = mean_absolute_error(merged["actual_lowest_price"], merged["predicted_lowest_price"])
    rmse = mean_squared_error(merged["actual_lowest_price"], merged["predicted_lowest_price"], squared=False)

    print(f"✅ Evaluated {len(merged)} new games — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")
    merged.to_csv(MERGED_OUTPUT, index=False)

    # Save log
    merged["timestamp"] = datetime.now().isoformat()
    summary_log = merged[[
        "timestamp", "homeTeam", "awayTeam", "startDateEastern",
        "predicted_lowest_price", "actual_lowest_price", "abs_error", "percent_error"
    ]]

    if os.path.exists(ERROR_LOG_PATH):
        existing = pd.read_csv(ERROR_LOG_PATH)
        full_log = pd.concat([existing, summary_log], ignore_index=True)
    else:
        full_log = summary_log

    full_log.to_csv(ERROR_LOG_PATH, index=False)
    print(f"📊 Logged {len(summary_log)} new game evaluations.")

    # 🔁 CI-based retraining
    error_fraction = (merged["percent_error"] > PERCENT_ERROR_THRESHOLD).mean()
    if error_fraction > ERROR_FRACTION_TRIGGER:
        print(f"⚠️ {error_fraction:.0%} of games exceed {PERCENT_ERROR_THRESHOLD:.0%} error — retraining model.")
        subprocess.run(["python", "src/train_price_model.py"], check=True)
    else:
        print(f"✅ Only {error_fraction:.0%} of games exceeded {PERCENT_ERROR_THRESHOLD:.0%} error — no retraining needed.")

if __name__ == "__main__":
    evaluate_predictions()