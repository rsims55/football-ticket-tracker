# =============================
# FILE: src/modeling/train_price_model.py
# PURPOSE: Add time-of-day feature (collectionSlot) so optimal times vary
# =============================
from __future__ import annotations

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

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
def _find_repo_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: two levels up (src/..)
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

SNAPSHOT_PATH = _resolve_file("SNAPSHOT_PATH", Path("data") / "daily" / "price_snapshots.csv")
MODEL_PATH    = _resolve_file("MODEL_PATH",    Path("models") / "ticket_price_model.pkl")

print("[train_price_model] Paths resolved:")
print(f"  PROJ_DIR:      {PROJ_DIR}")
print(f"  SNAPSHOT_PATH: {SNAPSHOT_PATH}")
print(f"  MODEL_PATH:    {MODEL_PATH}")

# -----------------------------
# Time-of-day slot derivation (NEW)
# -----------------------------
SLOT_LABELS = ["00:00", "06:00", "12:00", "18:00"]

def _nearest_6h_slot(ts) -> str:
    if pd.isna(ts):
        return np.nan
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return np.nan
    minutes = t.hour * 60 + t.minute
    idx = int(np.round(minutes / 360.0)) % 4
    return SLOT_LABELS[idx]

def _ensure_collection_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Populate df['collectionSlot'] from any available timestamp/time columns.
    Priority:
      1) 'collected_at' or 'snapshot_datetime' (datetime)
      2) 'time_collected' / 'collection_time' / 'snapshot_time' (time)
      3) 'date_collected' + 'time_local' (fallback)
    If none are parseable, fill with '12:00'.
    """
    cand_dt = [c for c in ["collected_at", "snapshot_datetime"] if c in df.columns]
    cand_t  = [c for c in ["time_collected", "collection_time", "snapshot_time"] if c in df.columns]

    slot = None
    if cand_dt:
        slot = df[cand_dt[0]].apply(_nearest_6h_slot)
    elif cand_t:
        slot = pd.to_datetime("1970-01-01 " + df[cand_t[0]].astype(str), errors="coerce").apply(_nearest_6h_slot)
    elif "date_collected" in df.columns and "time_local" in df.columns:
        combo = df["date_collected"].astype(str).str.strip() + " " + df["time_local"].astype(str).str.strip()
        slot = pd.to_datetime(combo, errors="coerce").apply(_nearest_6h_slot)

    if slot is not None:
        df["collectionSlot"] = slot
    else:
        df["collectionSlot"] = "12:00"

    df["collectionSlot"] = df["collectionSlot"].fillna("12:00")
    return df

# -----------------------------
# Features
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
CATEGORICAL_FEATURES = ["homeConference", "awayConference", "collectionSlot"]  # NEW

TARGET = "lowest_price"
# REQUIRED: do not include derived 'collectionSlot'
REQUIRED = NUMERIC_FEATURES + ["homeConference", "awayConference", TARGET]

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

    # Derive collectionSlot BEFORE schema check (NEW)
    df = _ensure_collection_slot(df)

    # Schema check (without collectionSlot)
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
        df, ["days_until_game", "capacity", "homeTeamRank", "awayTeamRank", "home_last_point_diff_at_snapshot", "away_last_point_diff_at_snapshot"]
    )

    # Keep only relevant cols (includes derived collectionSlot via CATEGORICAL_FEATURES)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if after == 0:
        raise ValueError("No rows with 'lowest_price' after dropping NA.")

    # Split X/y
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Preprocessing: imputers + one-hot for conferences + time slot
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

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
