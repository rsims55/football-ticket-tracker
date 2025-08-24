# src/modeling/predict_price.py
import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

MODEL_PATH = "models/ticket_price_model.pkl"
PRICE_PATH = "data/daily/price_snapshots.csv"
OUTPUT_PATH = "data/predicted/predicted_prices_optimal.csv"

# Sim grid
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
MAX_DAYS_OUT = 30

# --- FEATURES (must match training) ---
NUMERIC_FEATURES = [
    "days_until_game",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
    "week",  # ← NEW: add week as numeric predictor
]
CATEGORICAL_FEATURES = ["homeConference", "awayConference"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Minimum columns needed from snapshots
MIN_REQUIRED_INPUT = [
    "event_id",
    "date_local",                     # time_local optional
    "homeConference",
    "awayConference",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
    "week",                           # ← require week now that it's in snapshots
]

def _coerce_booleans(df, bool_cols=None):
    """
    Normalize boolean-like columns with NaNs/strings/0/1 safely.
    If bool_cols is None, use the default set used in this script.
    """
    import numpy as np
    import pandas as pd

    if bool_cols is None:
        bool_cols = ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]

    # Only keep existing columns
    bool_cols = [c for c in bool_cols if c in df.columns]

    truth_map = {
        True: True, False: False,
        "true": True, "false": False, "True": True, "False": False,
        "yes": True, "no": False, "YES": True, "NO": False,
        "y": True, "n": False, "Y": True, "N": False,
        1: True, 0: False, "1": True, "0": False,
        "t": True, "f": False, "T": True, "F": False,
        np.nan: np.nan
    }

    for c in bool_cols:
        s = df[c]
        if not pd.api.types.is_bool_dtype(s):
            s = s.map(truth_map).astype("boolean")
        df[c] = s.fillna(False).astype(bool)

    return df


def _compose_start_datetime(row) -> pd.Timestamp:
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    return pd.to_datetime(dt_str, errors="coerce")

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    return joblib.load(MODEL_PATH)

def _prep_games_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Schema check
    missing = [c for c in MIN_REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for simulation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Types aligned with training
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    # Coerce numeric columns that sometimes arrive as strings
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build datetime and drop unknowns
    df["startDateEastern"] = df.apply(_compose_start_datetime, axis=1)
    df = df.dropna(subset=["startDateEastern"])

    # One row per event (first seen snapshot)
    df_games = (
        df.sort_values(by=["startDateEastern"])
          .drop_duplicates(subset=["event_id"], keep="first")
          .copy()
    )

    # Ensure week is present and numeric (fallback 0 if NaN)
    if "week" in df_games.columns:
        df_games["week"] = pd.to_numeric(df_games["week"], errors="coerce").fillna(0).astype(int)
    else:
        df_games["week"] = 0  # safety

    # Keep just what we need
    keep_cols = ["event_id", "startDateEastern"] + FEATURES
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    return df_games[keep_cols]

def _simulate_one_game(game_row: pd.Series, model) -> dict:
    """Vectorized simulation for a single game: build a block of all (day, time) combos and predict once."""
    game_dt = pd.to_datetime(game_row["startDateEastern"])
    game_date = game_dt.date()

    # Build simulation grid (days x times)
    days = np.arange(1, MAX_DAYS_OUT + 1, dtype=int)
    times = COLLECTION_TIMES

    # Cartesian product
    sim_days = np.repeat(days, len(times))
    sim_times = np.tile(times, len(days))

    # Base feature dict from the game row (constant across grid except days_until_game)
    base_feats = {f: game_row.get(f, None) for f in FEATURES}

    # Build dataframe in one go
    sim_df = pd.DataFrame(base_feats, index=np.arange(len(sim_days)))
    sim_df["days_until_game"] = sim_days  # override per row

    # Predict in one call
    preds = model.predict(sim_df)  # shape (len(sim_df),)

    # Pick best
    best_idx = int(np.argmin(preds))
    best_price = float(preds[best_idx])
    best_delta = int(sim_days[best_idx])
    best_time_str = sim_times[best_idx]

    best_date = game_date - timedelta(days=best_delta)
    best_time_fmt = datetime.strptime(best_time_str, "%H:%M").time().strftime("%H:%M")

    return {
        "event_id": game_row.get("event_id"),
        "startDateEastern": pd.to_datetime(game_row.get("startDateEastern")).isoformat(),
        "week": int(game_row.get("week")) if pd.notna(game_row.get("week")) else None,
        # Keep conferences (no team names)
        "homeConference": game_row.get("homeConference"),
        "awayConference": game_row.get("awayConference"),
        "predicted_lowest_price": round(best_price, 2),
        "optimal_purchase_date": best_date.isoformat(),
        "optimal_purchase_time": best_time_fmt,
    }

def main():
    if not os.path.exists(PRICE_PATH):
        raise FileNotFoundError(f"Snapshot CSV not found at '{PRICE_PATH}'")

    price_df = pd.read_csv(PRICE_PATH)
    games_df = _prep_games_frame(price_df)
    model = _load_model()

    # Run batched per game (fast and memory-safe)
    results = [_simulate_one_game(row, model) for _, row in games_df.iterrows()]
    out = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
