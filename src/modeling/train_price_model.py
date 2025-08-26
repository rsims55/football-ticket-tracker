# src/modeling/train_price_model.py
from __future__ import annotations

import os
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Repo-locked paths (runs from anywhere)
# -----------------------------
_THIS = Path(__file__).resolve()
SRC_DIR = _THIS.parents[2]         # .../src
PROJ_DIR = SRC_DIR.parent          # repo root

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

def _under_repo(p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(PROJ_DIR.resolve())
    except AttributeError:
        return str(p.resolve()).startswith(str(PROJ_DIR.resolve()))

# Resolve snapshot path
_env_snap = os.getenv("SNAPSHOT_PATH")
if REPO_DATA_LOCK or not _env_snap:
    SNAPSHOT_PATH = PROJ_DIR / "data" / "daily" / "price_snapshots.csv"
else:
    SNAPSHOT_PATH = Path(_env_snap).expanduser()
    if not _under_repo(SNAPSHOT_PATH) and not ALLOW_ESCAPE:
        print(f"🚫 SNAPSHOT_PATH outside repo → {SNAPSHOT_PATH} ; forcing repo path")
        SNAPSHOT_PATH = PROJ_DIR / "data" / "daily" / "price_snapshots.csv"

# Resolve model path
_env_model = os.getenv("MODEL_PATH")
if REPO_DATA_LOCK or not _env_model:
    MODEL_PATH = PROJ_DIR / "models" / "ticket_price_model.pkl"
else:
    MODEL_PATH = Path(_env_model).expanduser()
    if not _under_repo(MODEL_PATH) and not ALLOW_ESCAPE:
        print(f"🚫 MODEL_PATH outside repo → {MODEL_PATH} ; forcing repo path")
        MODEL_PATH = PROJ_DIR / "models" / "ticket_price_model.pkl"

print("[train_price_model] Paths resolved:")
print(f"  PROJ_DIR:      {PROJ_DIR}")
print(f"  SNAPSHOT_PATH: {SNAPSHOT_PATH}")
print(f"  MODEL_PATH:    {MODEL_PATH}")

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
    "week",
]
CATEGORICAL_FEATURES = ["homeConference", "awayConference"]

TARGET = "lowest_price"
REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

def _coerce_booleans(df, bool_cols=None):
    """
    Make boolean-like columns robust against NaN/strings/0/1 before casting.
    If bool_cols is not passed, it uses a default list.
    (NaN-safe and accepts optional list.)
    """
    import numpy as np
    import pandas as pd

    if bool_cols is None:
        bool_cols = ["neutral_site", "rivalry", "conference_game", "is_weeknight"]

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
        np.nan: np.nan,
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
    if not SNAPSHOT_PATH.exists():
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

    # sklearn >=1.2 uses sparse_output; earlier uses sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
