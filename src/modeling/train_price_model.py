import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

SNAPSHOT_PATH = "data/daily/price_snapshots.csv"
MODEL_PATH = "models/ticket_price_model.pkl"

FEATURES = [
    "days_until_game",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
]

REQUIRED = FEATURES + ["lowest_price"]


def _coerce_booleans(df: pd.DataFrame, cols):
    """Ensure boolean-like columns are actually bools."""
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


def train_model():
    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError(f"Snapshot data not found at '{SNAPSHOT_PATH}'")

    df = pd.read_csv(SNAPSHOT_PATH)

    # Schema check
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Convert stringified booleans
    df = _coerce_booleans(
        df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]
    )

    # Keep only relevant cols
    df = df[REQUIRED].copy()

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=["lowest_price"])
    after = len(df)
    if after == 0:
        raise ValueError("No rows with 'lowest_price' after dropping NA.")

    # Fill missing features
    df[FEATURES] = df[FEATURES].fillna(-1)

    X = df[FEATURES]
    y = df["lowest_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(
        f"Rows read: {before} | Used: {after} | "
        f"Train: {len(X_train)} | Test: {len(X_test)}"
    )
    print(f"Test MSE: {mse:.2f} | R²: {r2:.3f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
