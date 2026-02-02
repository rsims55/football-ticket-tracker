# =============================
# FILE: src/modeling/predict_price.py
# PURPOSE:
#   Predict optimal purchase (price/date/time) using a model trained on
#   CONTINUOUS time features:
#     - hours_until_game (float)
#     - days_until_game (float)
#     - collection_hour_local (float in [0,24))
#   We scan an hourly grid prior to kickoff and pick the argmin.
#   We also attach the *ever observed* minimum price per event to the CSV.
# =============================
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

# -----------------------------
# Repo-locked paths (runs from anywhere)
# -----------------------------
def _find_repo_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
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

MODEL_PATH  = _resolve_file("MODEL_PATH",  Path("models") / "ticket_price_model.pkl")
PRICE_PATH  = _resolve_file("PRICE_PATH",  Path("data") / "daily" / "price_snapshots.csv")
OUTPUT_PATH = _resolve_file("OUTPUT_PATH", Path("data") / "predicted" / "predicted_prices_optimal.csv")

print("[predict_price] Paths resolved:")
print(f"  PROJ_DIR:    {PROJ_DIR}")
print(f"  MODEL_PATH:  {MODEL_PATH}")
print(f"  PRICE_PATH:  {PRICE_PATH}")
print(f"  OUTPUT_PATH: {OUTPUT_PATH}")

# -----------------------------
# Write safety (atomic + guard)
# -----------------------------
def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".__tmp__")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

_FORBIDDEN = {
    str(PRICE_PATH.resolve()),
    str((PROJ_DIR / "data" / "daily" / "price_snapshots.csv").resolve()),
}

def _assert_not_snapshot(target: Path):
    t = str(target.resolve())
    assert t not in _FORBIDDEN, f"Refusing to overwrite snapshots CSV: {t}"

# -----------------------------
# Continuous-time simulation settings
# -----------------------------
MAX_DAYS_OUT: int = int(os.getenv("MAX_DAYS_OUT", "30"))  # horizon for search (days)
STEP_HOURS:   int = int(os.getenv("STEP_HOURS", "1"))     # grid resolution in hours (1h default)

# Feature names used by the trained pipeline
TIME_FEATURES = ["hours_until_game", "days_until_game", "collection_hour_local"]
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
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Required base inputs present in snapshots to define a game row
MIN_REQUIRED_INPUT = [
    "event_id",
    "date_local",  # time_local optional but recommended
    "homeConference",
    "awayConference",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
    "week",
]

# -----------------------------
# Helpers — timestamps & coercions
# -----------------------------
_DT_CANDIDATES = ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at"]
_TIME_ONLY     = ["time_collected", "collection_time", "snapshot_time"]

def _best_snapshot_ts(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort snapshot timestamp (for *observed min* only).
    Priority:
      1) datetime-like column
      2) date_collected + time-only column
      3) date_collected (midnight)
    Returns tz-naive Series; may contain NaT.
    """
    ts = None
    for c in _DT_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            if not ts.isna().all():
                break
    if ts is None or ts.isna().all():
        tcol = next((c for c in _TIME_ONLY if c in df.columns), None)
        if "date_collected" in df.columns and tcol:
            combo = df["date_collected"].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip()
            ts = pd.to_datetime(combo, errors="coerce")
    if ts is None or ts.isna().all():
        if "date_collected" in df.columns:
            ts = pd.to_datetime(df["date_collected"], errors="coerce")
    if ts is None:
        ts = pd.Series(pd.NaT, index=df.index)
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    return ts

def _kickoff_ts_from_row(row: pd.Series) -> pd.Timestamp:
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

def _observed_min_per_event(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ever-observed minimum 'lowest_price' per event, with best-effort timestamp.
    Returns: event_id, observed_lowest_price_num, observed_lowest_dt
    (Robust to groups with all-NaN prices and NaN event_ids.)
    """
    if "lowest_price" not in price_df.columns:
        raise ValueError("Snapshots CSV is missing 'lowest_price' column.")

    # Work on a copy and build snapshot ts (for the WHEN)
    tmp = price_df.copy()
    tmp["_snap_ts"] = _best_snapshot_ts(tmp)

    # Only keep rows with a valid event_id and a numeric price
    tmp = tmp[pd.notna(tmp["event_id"])].copy()
    tmp["lowest_price"] = pd.to_numeric(tmp["lowest_price"], errors="coerce")
    tmp_valid = tmp.dropna(subset=["lowest_price"])

    # If nothing valid, return empty with expected columns
    if tmp_valid.empty:
        return pd.DataFrame(columns=["event_id", "observed_lowest_price_num", "observed_lowest_dt"])

    # Find the row (per event_id) where lowest_price is minimal
    # Using idxmin on the filtered frame avoids NaN indices
    idx = tmp_valid.groupby("event_id")["lowest_price"].idxmin()
    mins = tmp_valid.loc[idx, ["event_id", "lowest_price", "_snap_ts"]].copy()

    mins = mins.rename(columns={
        "lowest_price": "observed_lowest_price_num",
        "_snap_ts": "observed_lowest_dt",
    })

    # Normalize types/format
    mins["observed_lowest_price_num"] = mins["observed_lowest_price_num"].astype(float).round(2)
    mins["observed_lowest_dt"] = mins["observed_lowest_dt"].apply(lambda x: x.isoformat() if pd.notna(x) else "")

    return mins.reset_index(drop=True)


# -----------------------------
# Prep games (1 row per event)
# -----------------------------
def _prep_games_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MIN_REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for simulation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    df = df.copy()
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # kickoff timestamp
    df["kickoff_ts"] = df.apply(_kickoff_ts_from_row, axis=1)
    df = df.dropna(subset=["kickoff_ts"])

    # One row per event_id (earliest defined row)
    df_games = (
        df.sort_values(by=["kickoff_ts"])
          .drop_duplicates(subset=["event_id"], keep="first")
          .copy()
    )

    # Rename for output parity
    df_games["startDateEastern"] = df_games["kickoff_ts"]

    keep_cols = ["event_id", "startDateEastern"] + ALL_FEATURES
    # Not all ALL_FEATURES exist yet (time features are created in sim), so filter
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    return df_games[keep_cols + ["kickoff_ts"]]

# -----------------------------
# Build hourly grid & predict
# -----------------------------
def _simulate_one_game(game_row: pd.Series, model) -> Dict[str, object]:
    """
    Build an hourly grid up to MAX_DAYS_OUT before kickoff, predict each hour,
    then pick the minimum.
    """
    kickoff_ts = pd.to_datetime(game_row["kickoff_ts"])
    horizon_hours = MAX_DAYS_OUT * 24
    hours = np.arange(1, horizon_hours + 1, STEP_HOURS, dtype=int)

    # Simulated snapshot timestamps (when we'd hypothetically buy)
    sim_ts = [kickoff_ts - pd.Timedelta(int(h), "h") for h in hours]

    # Continuous time features
    sim_hours_until = hours.astype(float)
    sim_days_until = sim_hours_until / 24.0
    sim_clock = np.array([t.hour + t.minute/60.0 + t.second/3600.0 for t in sim_ts], dtype=float)

    # Static features copied from the game row
    static = {k: game_row.get(k, np.nan) for k in [
        "capacity", "neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup",
        "homeTeamRank", "awayTeamRank", "week", "homeConference", "awayConference"
    ]}

    # Assemble feature frame
    feat = pd.DataFrame({
        "hours_until_game": sim_hours_until,
        "days_until_game":  sim_days_until,
        "collection_hour_local": sim_clock,
        **{k: [static[k]] * len(hours) for k in static}
    })

    # Model prediction
    yhat = model.predict(feat[ALL_FEATURES])
    yhat = np.maximum(yhat, 0.0)  # guard against negatives

    # Argmin
    best_idx = int(np.nanargmin(yhat))
    best_price = float(yhat[best_idx])
    best_dt = sim_ts[best_idx]

    return {
        "event_id": game_row.get("event_id"),
        "startDateEastern": pd.to_datetime(game_row.get("startDateEastern")).isoformat(),
        "week": int(game_row.get("week")) if pd.notna(game_row.get("week")) else None,
        "homeConference": game_row.get("homeConference"),
        "awayConference": game_row.get("awayConference"),
        "predicted_lowest_price": round(best_price, 2),
        "optimal_purchase_date": best_dt.date().isoformat(),
        "optimal_purchase_time": best_dt.strftime("%H:%M"),
        "optimal_source": "model",
    }

# -----------------------------
# Model loader
# -----------------------------
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    obj = joblib.load(MODEL_PATH)
    # Support either direct Pipeline or dict wrapper from the training script
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj

# -----------------------------
# Main
# -----------------------------
def main():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Snapshot CSV not found at '{PRICE_PATH}'")

    # Load snapshots
    price_df = pd.read_csv(PRICE_PATH)

    # Prepare per-game rows
    games_df = _prep_games_frame(price_df)

    # Load model (sklearn Pipeline)
    model = _load_model()

    # Predict per game
    results = [_simulate_one_game(row, model) for _, row in games_df.iterrows()]
    out = pd.DataFrame(results)

    # Attach ever-observed minima
    obs = _observed_min_per_event(price_df)
    out = out.merge(obs, on="event_id", how="left")

    # Write
    _assert_not_snapshot(OUTPUT_PATH)
    _write_csv_atomic(out, OUTPUT_PATH)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")
    print(f"   Columns: {list(out.columns)}")

if __name__ == "__main__":
    main()
