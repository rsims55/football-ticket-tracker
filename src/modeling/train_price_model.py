import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

SNAPSHOT_PATH = "data/daily/price_snapshots.csv"
MODEL_PATH = "models/ticket_price_model.pkl"

# -----------------------------
# Use conferences, not team names
# -----------------------------
NUMERIC_FEATURES = [
    "days_until_game",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
    "week"
]
CATEGORICAL_FEATURES = ["homeConference", "awayConference"]

TARGET = "lowest_price"
REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def _coerce_booleans(df, bool_cols=None):
    """
    Make boolean-like columns robust against NaN/strings/0/1 before casting.
    If bool_cols is not passed, it uses a default list.
    """
    import numpy as np
    import pandas as pd

    # Default list if none provided
    if bool_cols is None:
        bool_cols = [
            "neutral_site", "rivalry", "conference_game", "is_weeknight"
        ]

    # Keep only columns that actually exist
    bool_cols = [c for c in bool_cols if c in df.columns]

    truth_map = {
        True: True, False: False,
        "true": True, "false": False,
        "True": True, "False": False,
        "yes": True, "no": False,
        "YES": True, "NO": False,
        "y": True, "n": False,
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



def _coerce_numerics(df: pd.DataFrame, cols):
    """Coerce numerics; non-numeric -> NaN for imputation."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
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

    # Clean types
    df = _coerce_booleans(
        df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]
    )
    df = _coerce_numerics(
        df, ["days_until_game", "capacity", "homeTeamRank", "awayTeamRank"]
    )

    # Keep only relevant cols
    df = df[REQUIRED].copy()

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if after == 0:
        raise ValueError("No rows with 'lowest_price' after dropping NA.")

    # Split X/y
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Preprocessing: imputers + one-hot for conferences
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Handle both old/new sklearn APIs for OneHotEncoder
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(
        f"Rows read: {before} | Used: {after} | "
        f"Train: {len(X_train)} | Test: {len(X_test)}"
    )
    print(f"Test MSE: {mse:.2f} | R²: {r2:.3f}")

    # Save the whole pipeline (encoder + model)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
