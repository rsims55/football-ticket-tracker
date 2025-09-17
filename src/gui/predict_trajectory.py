# src/modeling/predict_trajectory.py — continuous-time version
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Repo paths ----------
_THIS = Path(__file__).resolve()
PROJ_DIR = _THIS.parents[2]
DEFAULT_MODEL_PATH = PROJ_DIR / "models" / "ticket_price_model.pkl"

# ---------- Feature schema (matches train/predict) ----------
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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------- Robust model loader ----------
@lru_cache(maxsize=1)
def _load_model(model_path: str | Path | None = None):
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")

    last_err: Optional[Exception] = None

    # 1) joblib (preferred)
    try:
        import joblib  # type: ignore
        obj = joblib.load(path)
        # unwrap dict wrapper from training
        if isinstance(obj, dict) and "pipeline" in obj:
            return obj["pipeline"]
        return obj
    except Exception as e:
        last_err = e

    # 2) cloudpickle
    try:
        import cloudpickle  # type: ignore
        with open(path, "rb") as f:
            obj = cloudpickle.load(f)
            if isinstance(obj, dict) and "pipeline" in obj:
                return obj["pipeline"]
            return obj
    except Exception as e:
        last_err = e

    # 3) stdlib pickle
    try:
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, dict) and "pipeline" in obj:
                return obj["pipeline"]
            return obj
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
    """NaN-safe boolean coercion."""
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

def _kickoff_ts(row: Dict) -> pd.Timestamp:
    """Accept several common keys for kickoff; return tz-naive Timestamp or NaT."""
    for key in ("startDateEastern", "kickoff_ts", "start_time", "startDate", "date_local"):
        if key in row and pd.notna(row[key]):
            dt = _to_dt(row[key])
            if pd.notna(dt):
                try:
                    return dt.tz_localize(None)
                except Exception:
                    return dt
    return pd.NaT

def _alias_row_to_required(row: Dict) -> Dict:
    """Normalize incoming row keys to model feature keys (except time features)."""
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
    return out

# ---------- Public API ----------
def predict_for_times(row: Dict, times: Iterable[pd.Timestamp], model_path: str | Path | None = None) -> List[float]:
    """
    Predict prices for arbitrary timestamps using the CONTINUOUS-TIME feature schema:
      - hours_until_game = (kickoff - t) in hours
      - days_until_game  = hours_until_game / 24
      - collection_hour_local = local clock-hour of t (fractional)
    """
    model = _load_model(model_path)
    kickoff = _kickoff_ts(row)
    if pd.isna(kickoff):
        raise ValueError("Row must include a valid 'startDateEastern' (or alias) to compute time until game.")

    # Ensure times are pandas Timestamps (tz-naive ok)
    times = [pd.Timestamp(t) for t in times]

    # Continuous time features
    hours_until = np.array([(kickoff - t).total_seconds() / 3600.0 for t in times], dtype=float)
    days_until = hours_until / 24.0
    clock_hour = np.array([t.hour + t.minute/60.0 + t.second/3600.0 for t in times], dtype=float)

    # Static features
    base = _alias_row_to_required(row)
    n = len(times)
    df = pd.DataFrame({
        "hours_until_game": hours_until,
        "days_until_game":  days_until,
        "collection_hour_local": clock_hour,
        **{k: [base[k]] * n for k in base},
    })

    # Coercions and ordering
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    # Ensure all expected cols exist (pipeline will impute / one-hot as needed)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df = df[FEATURES]

    yhat = model.predict(df)
    yhat = np.maximum(pd.to_numeric(pd.Series(yhat), errors="coerce").astype(float), 0.0)
    return yhat.tolist()


# Optional convenience: generate an hourly grid back from kickoff
def predict_trajectory(row: Dict, hours_back: int = 24 * 30, step_hours: int = 6, model_path: str | Path | None = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build a grid of times: kickoff - [1h..hours_back] and predict each.
    Returns (df, summary) where df includes columns: when, yhat.
    """
    model = _load_model(model_path)
    kickoff = _kickoff_ts(row)
    if pd.isna(kickoff):
        raise ValueError("Missing/invalid kickoff timestamp for trajectory simulation.")

    hours = np.arange(1, int(hours_back) + 1, int(step_hours), dtype=int)
    ts_grid = [kickoff - pd.Timedelta(int(h), "h") for h in hours]
    y = predict_for_times(row, ts_grid, model_path=model_path)

    df = pd.DataFrame({"when": ts_grid, "yhat": y})
    best_idx = int(np.nanargmin(df["yhat"].values))
    best_row = df.iloc[best_idx]
    summary = {
        "predicted_lowest_price": round(float(best_row["yhat"]), 2),
        "optimal_purchase_date": pd.Timestamp(best_row["when"]).date().isoformat(),
        "optimal_purchase_time": pd.Timestamp(best_row["when"]).strftime("%H:%M"),
        "optimal_source": "model",
    }
    return df, summary
