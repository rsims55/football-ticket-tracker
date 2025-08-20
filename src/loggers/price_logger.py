# src/loggers/price_logger.py
import os
import sys
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# ---- Project imports
try:
    from scrapers.tickpick_pricer import TickPickPricer
except ModuleNotFoundError as e:
    print("❌ Could not import TickPickPricer. Ensure fetchers/tickpick_pricer.py exists.", file=sys.stderr)
    raise e

try:
    from builders.build_dataset import DatasetBuilder  # prefer reading CSV; fallback allowed
except Exception:
    DatasetBuilder = None  # type: ignore

# ---- Config
TIMEZONE = ZoneInfo("America/New_York")
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
YEAR = int(os.getenv("SEASON_YEAR", datetime.now(TIMEZONE).year))
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", "data/price_snapshots.csv")
SCHEDULE_PATH = os.getenv("SCHEDULE_PATH", f"data/enriched_schedule_{YEAR}.csv")
ALWAYS_RUN_FOR_TESTING = os.getenv("ALWAYS_RUN_PRICE_LOGGER", "1") == "1"


def _load_schedule_df() -> pd.DataFrame:
    if os.path.exists(SCHEDULE_PATH):
        df = pd.read_csv(SCHEDULE_PATH)
    elif DatasetBuilder is not None:
        builder = DatasetBuilder(year=YEAR, use_mock_tickets=False, include_prices=False)
        df = builder.build()
        builder.save(f"data/enriched_schedule_{YEAR}.csv")
    else:
        raise FileNotFoundError(f"Missing schedule CSV at {SCHEDULE_PATH} and DatasetBuilder not available.")

    # Normalize startDateEastern
    if "startDateEastern" not in df.columns:
        raise KeyError("Schedule missing 'startDateEastern' column.")

    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce")
    # If naive -> localize to ET; else convert to ET
    if df["startDateEastern"].dt.tz is None:
        df["startDateEastern"] = df["startDateEastern"].dt.tz_localize(TIMEZONE)
    else:
        df["startDateEastern"] = df["startDateEastern"].dt.tz_convert(TIMEZONE)

    # Standardize expected names
    if "eventId" in df.columns and "event_id" not in df.columns:
        df = df.rename(columns={"eventId": "event_id"})
    # optional friendly title
    if "matchupTitle" not in df.columns:
        df["matchupTitle"] = df["homeTeam"].astype(str) + " vs " + df["awayTeam"].astype(str)

    return df


def _filter_future_games(df: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    return df[df["startDateEastern"] >= now_et].copy()


def _call_tickpick(pricer: "TickPickPricer", row: pd.Series) -> dict:
    home = row.get("homeTeam")
    away = row.get("awayTeam")
    start_dt = row.get("startDateEastern")
    event_id = row.get("event_id") or row.get("eventId")

    try_order = [
        {"event_id": event_id, "home_team": home, "away_team": away, "start_dt": start_dt},
        {"event_id": event_id},
        {"home_team": home, "away_team": away, "start_dt": start_dt},
        {"home_team": home, "away_team": away},
    ]

    last_err = None
    for kwargs in try_order:
        try:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            if not clean:
                continue
            res = pricer.get_summary(**clean)
            if isinstance(res, dict) and res:
                return res
        except Exception as e:
            last_err = e
            continue

    return {"_error": str(last_err) if last_err else "no_result"}


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def log_today_snapshots():
    now_et = datetime.now(TIMEZONE)
    current_time_str = now_et.strftime("%H:%M")
    current_date_str = now_et.strftime("%Y-%m-%d")

    if not ALWAYS_RUN_FOR_TESTING and current_time_str not in COLLECTION_TIMES:
        return

    schedule_df = _load_schedule_df()
    schedule_df = _filter_future_games(schedule_df, now_et)

    if schedule_df.empty:
        print("ℹ️ No future games to log. Exiting.")
        return

    schedule_df["days_until_game"] = (schedule_df["startDateEastern"].dt.date - now_et.date()).apply(lambda x: int(x.days))
    pricer = TickPickPricer(use_mock=False)

    rows = []
    for _, row in schedule_df.iterrows():
        try:
            px = _call_tickpick(pricer, row)
        except Exception as e:
            px = {"_error": str(e)}

        rows.append({
            # Identity
            "event_id": row.get("event_id"),
            "homeTeam": row.get("homeTeam"),
            "awayTeam": row.get("awayTeam"),
            "matchupTitle": row.get("matchupTitle"),
            "stadium": row.get("stadium"),
            "city": row.get("city"),
            "state": row.get("state"),
            "startDateEastern": row.get("startDateEastern"),
            "days_until_game": row.get("days_until_game"),
            # Prices
            "lowest_price": px.get("lowest_price"),
            "average_price": px.get("average_price"),
            "listing_count": px.get("listing_count"),
            "source_url": px.get("source_url"),
            # Metadata
            "date_collected": current_date_str,
            "time_collected": current_time_str,
            # Error (if any)
            "error": px.get("_error")
        })

    snap_df = pd.DataFrame.from_records(rows)

    # Types
    for c in ["lowest_price", "average_price"]:
        if c in snap_df.columns:
            snap_df[c] = pd.to_numeric(snap_df[c], errors="coerce")
    if "listing_count" in snap_df.columns:
        snap_df["listing_count"] = pd.to_numeric(snap_df["listing_count"], errors="coerce", downcast="integer")

    # Append with de-dupe
    _ensure_dir(SNAPSHOT_PATH)
    if os.path.exists(SNAPSHOT_PATH):
        existing = pd.read_csv(SNAPSHOT_PATH)
        combined = pd.concat([existing, snap_df], ignore_index=True)

        key_cols = [c for c in ["event_id", "startDateEastern", "date_collected", "time_collected"] if c in combined.columns]
        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last")

        combined.to_csv(SNAPSHOT_PATH, index=False)
    else:
        snap_df.to_csv(SNAPSHOT_PATH, index=False)

    print(f"✅ Logged {len(snap_df)} snapshots at {current_time_str} ET to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    log_today_snapshots()
