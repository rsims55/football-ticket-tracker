import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

MODEL_PATH = "models/ticket_price_model.pkl"
PRICE_PATH = "data/daily/price_snapshots.csv"
OUTPUT_PATH = "data/predicted/predicted_prices_optimal.csv"

COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
MAX_DAYS_OUT = 30

FEATURES = [
    "days_until_game", "capacity", "neutralSite", "conferenceGame",
    "isRivalry", "isRankedMatchup", "homeTeamRank", "awayTeamRank"
]

# Columns we need from the snapshot CSV to simulate predictions
MIN_REQUIRED_INPUT = [
    "event_id", "homeTeam", "awayTeam", "date_local"  # time_local optional
]

def _coerce_booleans(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = (
                    df[c].astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False})
                )
            df[c] = df[c].astype("boolean").astype(bool)
    return df

def _compose_start_datetime(row) -> pd.Timestamp:
    """Combine date_local and (optional) time_local into a single Timestamp."""
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    if time_str and time_str.lower() != "nan":
        dt_str = f"{date_str} {time_str}"
    else:
        dt_str = date_str
    # Parse without forcing timezone; treat as local/Eastern for our purposes
    return pd.to_datetime(dt_str, errors="coerce")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    return joblib.load(MODEL_PATH)

def _prep_games_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Check minimum required inputs
    missing = [c for c in MIN_REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for simulation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Coerce booleans to match training
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    # Build startDateEastern from date_local/time_local
    df["startDateEastern"] = df.apply(_compose_start_datetime, axis=1)

    # Drop rows where we can't determine the game datetime
    df = df.dropna(subset=["startDateEastern"])

    # Ensure all model features exist; fill missing with sentinel
    for f in FEATURES:
        if f not in df.columns:
            df[f] = -1
    df[FEATURES] = df[FEATURES].fillna(-1)

    # Deduplicate to one row per game using event_id (most reliable)
    df_games = (
        df.sort_values(by=["startDateEastern"])
          .drop_duplicates(subset=["event_id"], keep="first")
          .copy()
    )

    # Keep only what we need later (but keep week if present)
    keep_cols = ["event_id", "homeTeam", "awayTeam", "startDateEastern", "week"] + FEATURES
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    df_games = df_games[keep_cols]

    return df_games

def simulate_predictions(df_games: pd.DataFrame, model) -> pd.DataFrame:
    results = []

    for _, row in df_games.iterrows():
        best_price = float("inf")
        best_date = None
        best_time = None

        game_dt = row["startDateEastern"]
        game_date = game_dt.date()

        for delta in range(1, MAX_DAYS_OUT + 1):
            sim_date = game_date - timedelta(days=delta)
            days_until_game = delta

            for time_str in COLLECTION_TIMES:
                sim_time = datetime.strptime(time_str, "%H:%M").time()

                # Build model input exactly like training
                input_row = pd.DataFrame([{f: row.get(f, -1) for f in FEATURES}])
                input_row["days_until_game"] = days_until_game
                input_row = input_row.fillna(-1)

                price_pred = float(model.predict(input_row)[0])

                if price_pred < best_price:
                    best_price = price_pred
                    best_date = sim_date
                    best_time = sim_time

        results.append({
            "event_id": row.get("event_id"),
            "homeTeam": row.get("homeTeam"),
            "awayTeam": row.get("awayTeam"),
            "startDateEastern": row.get("startDateEastern").isoformat(),
            "week": row.get("week") if "week" in df_games.columns else None,
            "predicted_lowest_price": round(best_price, 2) if best_price != float("inf") else None,
            "optimal_purchase_date": best_date.isoformat() if best_date else None,
            "optimal_purchase_time": best_time.strftime("%H:%M") if best_time else None,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    if not os.path.exists(PRICE_PATH):
        raise FileNotFoundError(f"Snapshot CSV not found at '{PRICE_PATH}'")

    price_df = pd.read_csv(PRICE_PATH)
    games_df = _prep_games_frame(price_df)
    model = load_model()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    simulate_predictions(games_df, model).to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")
