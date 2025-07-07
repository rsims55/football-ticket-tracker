import pandas as pd
from datetime import datetime
from src.build_dataset import DatasetBuilder
import os

# Set up logging details
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
SNAPSHOT_PATH = "data/price_snapshots.csv"


def log_today_snapshots():
    now = datetime.now()
    current_time_str = now.strftime("%H:%M")
    current_date_str = now.strftime("%Y-%m-%d")

    # TEMPORARY: Always run for testing
    # if current_time_str not in COLLECTION_TIMES:
    #     return

    # Build today's enriched dataset (mock ticket prices)
    builder = DatasetBuilder(use_mock_tickets=True)
    df = builder.build()

    # Add metadata columns
    df["date_collected"] = current_date_str
    df["time_collected"] = current_time_str
    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"])
    df["days_until_game"] = (df["startDateEastern"].dt.date - now.date()).apply(lambda x: x.days)


    # Drop columns we don't want to log if needed
    df = df.drop(columns=["startDateEastern", "kickoffTimeStr"], errors="ignore")

    # Save or append to master snapshot log
    if os.path.exists(SNAPSHOT_PATH):
        existing = pd.read_csv(SNAPSHOT_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(SNAPSHOT_PATH, index=False)
    else:
        df.to_csv(SNAPSHOT_PATH, index=False)


if __name__ == "__main__":
    log_today_snapshots()