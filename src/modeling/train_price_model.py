# =============================
# FILE: src/modeling/train_price_model.py
# PURPOSE:
#   Train a model using CONTINUOUS time features derived from each snapshot:
#     - hours_until_game: (kickoff_ts - snapshot_ts) in hours (float > 0)
#     - days_until_game : hours_until_game / 24
#     - collection_hour_local: snapshot local clock time in hours [0,24)
#   Target: lowest_price
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
# Helpers: timestamps & coercions
# -----------------------------
# Candidate columns to reconstruct the snapshot timestamp
_DT_CANDIDATES = ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at"]
_TIME_ONLY     = ["time_collected", "collection_time", "snapshot_time"]

def _best_snapshot_ts(df: pd.DataFrame) -> pd.Series:
    """
    Build a best-effort snapshot timestamp per row using (in priority order):
      1) a datetime-like column (e.g., 'collected_at')
      2) 'date_collected' + (time-only column)
      3) 'date_collected' (midnight)
    Returns a tz-naive pandas.Timestamp Series (may contain NaT).
    """
    ts = None

    # 1) Direct datetime-like columns
    for c in _DT_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            if not ts.isna().all():
                break

    # 2) Combine date_collected + a time-only column
    if ts is None or ts.isna().all():
        tcol = next((c for c in _TIME_ONLY if c in df.columns), None)
        if "date_collected" in df.columns and tcol:
            combo = (
                df["date_collected"].astype(str).str.strip()
                + " "
                + df[tcol].astype(str).str.strip()
            )
            ts = pd.to_datetime(combo, errors="coerce")

    # 3) Fallback: just date_collected (midnight)
    if ts is None or ts.isna().all():
        if "date_collected" in df.columns:
            ts = pd.to_datetime(df["date_collected"], errors="coerce")

    if ts is None:
        ts = pd.Series(pd.NaT, index=df.index)

    # make tz-naive consistently
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass

    return ts

def _kickoff_ts(row) -> pd.Timestamp:
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    ts = pd.to_datetime(dt_str, errors="coerce")
    try:
        ts = ts.tz_localize(None)
    except Exception:
        pass
    return ts

def _coerce_booleans(df: pd.DataFrame, bool_cols=None) -> pd.DataFrame:
    if bool_cols is None:
        bool_cols = ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]
    truth_map = {
        True: True, False: False,
        "true": True, "false": False, "True": True, "False": False,
        "yes": True, "no": False, "YES": True, "NO": False,
        "y": True, "n": False, "Y": True, "N": False,
        1: True, 0: False, "1": True, "0": False,
        "t": True, "f": False, "T": True, "F": False,
        np.nan: np.nan,
    }
    for c in [c for c in bool_cols if c in df.columns]:
        s = df[c]
        if not pd.api.types.is_bool_dtype(s):
            s = s.map(truth_map).astype("boolean")
        df[c] = s.fillna(False).astype(bool)
    return df

# -----------------------------
# Feature schema
# -----------------------------
# Continuous time features
TIME_FEATURES = [
    "hours_until_game",      # continuous time-to-kickoff in hours
    "days_until_game",       # hours_until_game / 24
    "collection_hour_local", # local clock-time at collection (0-24)
]

NUMERIC_FEATURES = TIME_FEATURES + [
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

REQUIRED_BASE = [
    "event_id",
    "date_local",  # time_local optional but recommended
    "homeConference", "awayConference",
    "capacity",
    "neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup",
    "homeTeamRank", "awayTeamRank", "week",
    TARGET,
]

# -----------------------------
# Training
# -----------------------------
def train_model():
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot data not found at '{SNAPSHOT_PATH}'")

    df = pd.read_csv(SNAPSHOT_PATH)

    # Basic schema check
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Clean types
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week", TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build timestamps
    df["_kickoff_ts"] = df.apply(_kickoff_ts, axis=1)
    df["_snapshot_ts"] = _best_snapshot_ts(df)

    # Compute continuous features
    delta_hours = (df["_kickoff_ts"] - df["_snapshot_ts"]).dt.total_seconds() / 3600.0
    df["hours_until_game"] = pd.to_numeric(delta_hours, errors="coerce")

    df["days_until_game"] = df["hours_until_game"] / 24.0

    # local clock hour of the snapshot (e.g., 13.5 for ~1:30pm)
    # if time-only available, _best_snapshot_ts already handled that
    snap = pd.to_datetime(df["_snapshot_ts"], errors="coerce")
    df["collection_hour_local"] = (
        snap.dt.hour.fillna(0).astype(float)
        + (snap.dt.minute.fillna(0).astype(float) / 60.0)
        + (snap.dt.second.fillna(0).astype(float) / 3600.0)
    )

    # Filter to valid rows: target present, positive horizon, known kickoff/snapshot
    before = len(df)
    df = df.dropna(subset=[TARGET, "hours_until_game", "days_until_game", "collection_hour_local", "_kickoff_ts", "_snapshot_ts"]).copy()
    df = df[df["hours_until_game"] > 0]
    after = len(df)

    if after == 0:
        raise ValueError("No valid rows remain after filtering for continuous time features and target.")

    # Final training frame
    use_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    train_df = df[use_cols].copy()

    # Split X/y
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[TARGET]

    # Preprocessing
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
        n_estimators=300,
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

    # Save pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_schema": {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "target": TARGET,
            },
            "notes": "Continuous time features: hours_until_game, days_until_game, collection_hour_local",
        },
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
