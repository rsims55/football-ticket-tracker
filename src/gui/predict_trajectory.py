# src/modeling/predict_trajectory.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Repo paths ----------
_THIS = Path(__file__).resolve()
PROJ_DIR = _THIS.parents[2]
DEFAULT_MODEL_PATH = PROJ_DIR / "models" / "ticket_price_model.pkl"

# ---- Parity with predict_price.py (do NOT change without changing predict_price) ----
# These names/values MUST match the training/prediction pipeline.
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
CATEGORICAL_FEATURES = ["homeConference", "awayConference", "collectionSlot"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# EXACT grid & order used by predict_price.py
COLLECTION_TIMES: List[str] = ["06:00", "12:00", "18:00", "00:00"]  # same order
MAX_DAYS_OUT: int = 30

# ---------- Robust model loader ----------
@lru_cache(maxsize=1)
def _load_model(model_path: str | Path | None = None):
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")

    last_err: Optional[Exception] = None

    # 1) joblib (preferred for sklearn)
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception as e:
        last_err = e

    # 2) cloudpickle
    try:
        import cloudpickle  # type: ignore
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        last_err = e

    # 3) stdlib pickle
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model at {path} with joblib/cloudpickle/pickle. "
            f"Last error: {type(e).__name__}: {e}"
        ) from last_err

# ---------- Helpers ----------
def _to_dt(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def _to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _coerce_booleans(df: pd.DataFrame, bool_cols=None) -> pd.DataFrame:
    """NaN-safe boolean coercion (parity with predict_price)."""
    import numpy as np
    if bool_cols is None:
        bool_cols = ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]
    bool_cols = [c for c in bool_cols if c in df.columns]
    truth_map = {
        True: True, False: False,
        "true": True, "false": False, "True": True, "False": False,
        "yes": True, "no": False, "YES": True, "NO": False,
        "y": True, "n": False, "Y": True, "N": False,
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

def _nearest_slot_label(ts: pd.Timestamp) -> str:
    """Map any timestamp to one of the EXACT labels the model knows."""
    if pd.isna(ts):
        return "12:00"
    minutes = ts.hour * 60 + ts.minute
    idx = int(np.round(minutes / 360.0)) % 4
    return ["00:00", "06:00", "12:00", "18:00"][idx]

def _kickoff_date(row: Dict) -> pd.Timestamp:
    # Accept a handful of common keys and coerce
    for key in ("startDateEastern", "kickoff_ts", "start_time", "startDate", "date_local"):
        if key in row and pd.notna(row[key]):
            dt = _to_dt(row[key])
            if pd.notna(dt):
                return dt
    return pd.NaT

def _alias_row_to_required(row: Dict) -> Dict:
    """Accept common aliases in your data and produce the exact keys FEATURES expect."""
    # Start with passthroughs or common alt names
    out = {
        "homeConference": row.get("homeConference", row.get("home_conference")),
        "awayConference": row.get("awayConference", row.get("away_conference")),
        "capacity":       _to_float(row.get("capacity")),
        "neutralSite":    row.get("neutralSite", row.get("neutral_site")),
        "conferenceGame": row.get("conferenceGame", row.get("conference_game")),
        "isRivalry":      row.get("isRivalry", row.get("rivalry", row.get("is_rivalry"))),
        "isRankedMatchup": row.get("isRankedMatchup", row.get("is_ranked_matchup")),
        "homeTeamRank":   _to_float(row.get("homeTeamRank", row.get("home_rank", row.get("homeRanking")))),
        "awayTeamRank":   _to_float(row.get("awayTeamRank", row.get("away_rank", row.get("awayRanking")))),
        "week":           int(_to_float(row.get("week"))) if row.get("week") is not None else 0,
    }
    # boolean coercion happens later on the DataFrame
    return out

def _build_sim_grid_like_predict_price(kickoff: pd.Timestamp) -> pd.DataFrame:
    """Produce the exact (days x 4 slots) grid used by predict_price.py for a single game."""
    if pd.isna(kickoff):
        raise ValueError("startDateEastern/kickoff is missing or invalid for trajectory simulation.")

    game_date = kickoff.date()
    days = np.arange(1, MAX_DAYS_OUT + 1, dtype=int)

    # Repeat days for each slot in the SAME ORDER as predict_price.py
    sim_days = np.repeat(days, len(COLLECTION_TIMES))
    sim_slots = np.tile(np.array(COLLECTION_TIMES, dtype=object), len(days))

    # For convenience, also carry the timestamp we are "simulating from"
    # (not used by the model, just helpful for plotting/inspection)
    sim_dates = [pd.Timestamp(game_date) - pd.Timedelta(int(d), "D") for d in sim_days]
    # Represent a wall-clock for plotting by combining date + slot (00:00/06:00/12:00/18:00)
    sim_ts = [pd.to_datetime(f"{d.date().isoformat()} {slot}") for d, slot in zip(sim_dates, sim_slots)]

    return pd.DataFrame(
        {
            "days_until_game": sim_days,
            "collectionSlot": sim_slots,
            "prediction_time": sim_ts,  # not used by the model; good for charts
        }
    )

def _assemble_feature_frame(base_row: Dict, kickoff: pd.Timestamp, sim_grid: pd.DataFrame) -> pd.DataFrame:
    """Combine static game fields with sim grid to match the model's FEATURES exactly."""
    static = _alias_row_to_required(base_row)
    static_df = pd.DataFrame({k: [v] * len(sim_grid) for k, v in static.items()})
    df = pd.concat([sim_grid.reset_index(drop=True), static_df.reset_index(drop=True)], axis=1)

    # Coerce numerics & booleans (parity with predict_price)
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    # Ensure columns and order; add any missing with NaN (pipeline imputes/one-hots)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df = df[FEATURES + ["prediction_time"]]  # keep prediction_time for caller
    return df

# ---------- Public APIs ----------
def predict_for_times(row: Dict, times: Iterable[pd.Timestamp], model_path: str | Path | None = None) -> List[float]:
    """
    Predict prices for arbitrary timestamps, mapped to the model's exact features.
    NOTE: For *parity* with predict_price.py's "lowest", use predict_grid_like_optimal().
    """
    model = _load_model(model_path)
    kickoff = _kickoff_date(row)
    if pd.isna(kickoff):
        raise ValueError("Row must include a valid 'startDateEastern' (or alias) to compute days_until_game.")

    # Build a small frame where each time maps to (days_until_game, collectionSlot)
    times = [pd.Timestamp(t) for t in times]
    days_until = [(kickoff.date() - t.date()).days for t in times]
    slots = [_nearest_slot_label(t) for t in times]

    base = _alias_row_to_required(row)
    df = pd.DataFrame(
        {
            "days_until_game": days_until,
            "collectionSlot": slots,
            **{k: base[k] for k in base},
        }
    )
    # Coercions and ordering
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df = df[FEATURES]

    yhat = model.predict(df)
    return pd.to_numeric(pd.Series(yhat), errors="coerce").astype(float).tolist()

def predict_grid_like_optimal(row: Dict, model_path: str | Path | None = None, max_days: int = MAX_DAYS_OUT) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build the SAME grid (days 1..max_days × slots 06/12/18/00) used by predict_price.py,
    predict every point, and return:
      - a DataFrame with columns: prediction_time, days_until_game, collectionSlot, yhat
      - a dict {predicted_lowest_price, optimal_purchase_date, optimal_purchase_time}
    Using this function ensures the argmin matches predict_price.py exactly.
    """
    # allow caller to shrink horizon if desired
    global MAX_DAYS_OUT
    original = MAX_DAYS_OUT
    MAX_DAYS_OUT = int(max_days)
    try:
        model = _load_model(model_path)
        kickoff = _kickoff_date(row)
        sim_grid = _build_sim_grid_like_predict_price(kickoff)
        feat_df = _assemble_feature_frame(row, kickoff, sim_grid)

        yhat = model.predict(feat_df[FEATURES])
        sim_grid = sim_grid.assign(yhat=pd.to_numeric(yhat, errors="coerce").astype(float))

        # Find argmin exactly like predict_price.py: over the (days, slots) flattened in this order
        best_idx = int(np.nanargmin(sim_grid["yhat"].values))
        best_row = sim_grid.iloc[best_idx]

        predicted_lowest_price = round(float(best_row["yhat"]), 2)
        # Convert "days_until_game" back to a calendar date (kickoff_date - delta_days)
        best_date = (kickoff.normalize() - pd.Timedelta(int(best_row["days_until_game"]), "D")).date().isoformat()
        best_time = str(best_row["collectionSlot"])

        summary = {
            "predicted_lowest_price": predicted_lowest_price,
            "optimal_purchase_date": best_date,
            "optimal_purchase_time": best_time,
            "optimal_source": "model",  # parity label
        }

        # Also expose a convenient trajectory frame for plotting
        # (Each row is one simulated timestamp’s prediction.)
        # Columns: prediction_time, days_until_game, collectionSlot, yhat
        out_df = sim_grid[["prediction_time", "days_until_game", "collectionSlot", "yhat"]].copy()
        return out_df, summary
    finally:
        MAX_DAYS_OUT = original

# Backwards-compatible short name
def predict_trajectory(row: Dict, model_path: str | Path | None = None, max_days: int = MAX_DAYS_OUT):
    """
    Convenience wrapper: returns (trajectory_df, summary_dict).
    The summary_dict fields align with predict_price.py output.
    """
    return predict_grid_like_optimal(row, model_path=model_path, max_days=max_days)
