# src/modeling/predict_trajectory.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Iterable, Optional

import numpy as np
import pandas as pd

# ---------- Repo paths ----------
_THIS = Path(__file__).resolve()
PROJ_DIR = _THIS.parents[2]
DEFAULT_MODEL_PATH = PROJ_DIR / "models" / "ticket_price_model.pkl"

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
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def _to_float(x):
    try:
        if pd.isna(x): return np.nan
        return float(x)
    except Exception:
        return np.nan

def _time_of_day_bin(hour: int) -> str:
    # 4 bins aligned with your 4x daily cadence
    if 0 <= hour < 6:   return "00-06"
    if 6 <= hour < 12:  return "06-12"
    if 12 <= hour < 18: return "12-18"
    return "18-24"

def _synthesize_time_features(kickoff: pd.Timestamp, tstamp: pd.Timestamp) -> dict:
    # Raw values
    weekday = int(tstamp.weekday())               # 0=Mon
    hour    = int(tstamp.hour)
    month   = int(tstamp.month)
    is_weekend = 1 if weekday >= 5 else 0
    tod_bin = _time_of_day_bin(hour)

    # Horizons
    days_until  = np.nan
    hours_until = np.nan
    if pd.notna(kickoff):
        delta = (kickoff - tstamp)
        days_until  = delta.total_seconds() / 86400.0
        hours_until = delta.total_seconds() / 3600.0

    return {
        # numerics
        "days_until": days_until,
        "days_until_game": days_until,   # alias
        "hours_until": hours_until,
        "weekday": weekday,
        "day_of_week": weekday,          # alias
        "month": month,
        "hour": hour,
        "is_weekend": is_weekend,
        "is_night_game": 1 if hour >= 17 else 0,
        # categoricals
        "time_of_day_bin": tod_bin,
        "day_name": tstamp.strftime("%a"),
        "date_only": tstamp.date().isoformat(),
        # datetime passthrough (pipelines usually ignore or transform)
        "kickoff_ts": kickoff,
        "startDateEastern": kickoff,
        "prediction_time": tstamp,
    }

def _coerce_numeric_fields(feat: dict, keys: list[str]):
    for k in keys:
        if k in feat:
            feat[k] = _to_float(feat[k])

def _build_features_for_time(row: Dict, tstamp: pd.Timestamp) -> Dict:
    """Build a rich, model-agnostic feature set with common aliases."""
    kickoff = _to_dt(row.get("startDateEastern"))
    base = {
        # ids / labels (categoricals are fine; pipeline encoders will handle them)
        "event_id": row.get("event_id"),
        "homeTeam": row.get("homeTeam"),
        "awayTeam": row.get("awayTeam"),
        "home_team": row.get("homeTeam"),   # alias
        "away_team": row.get("awayTeam"),   # alias
        "week": row.get("week"),
        "stadium": row.get("stadium") or row.get("venue"),
        "venue": row.get("venue") or row.get("stadium"),
        "homeConference": row.get("homeConference"),
        "awayConference": row.get("awayConference"),
        # flags (bool/int)
        "neutral_site": row.get("neutral_site"),
        "rivalry": row.get("is_rivalry") if "is_rivalry" in row else row.get("rivalry"),
        "is_rivalry": row.get("is_rivalry") if "is_rivalry" in row else row.get("rivalry"),
        "is_conference_game": row.get("is_conference_game") if "is_conference_game" in row else row.get("conference_game"),
        "conference_game": row.get("conference_game") if "conference_game" in row else row.get("is_conference_game"),
        # ranks / capacity
        "home_rank": row.get("home_rank") if "home_rank" in row else row.get("homeRanking"),
        "away_rank": row.get("away_rank") if "away_rank" in row else row.get("awayRanking"),
        "capacity": row.get("capacity"),
        # kickoff passthroughs
        "startDateEastern": kickoff,
        "kickoff_ts": kickoff,
    }

    # numeric coercions
    _coerce_numeric_fields(base, ["week", "home_rank", "away_rank", "capacity"])

    # join time-derived features
    base.update(_synthesize_time_features(kickoff, pd.Timestamp(tstamp)))
    return base

def _align_to_model_requirements(model, df_feats: pd.DataFrame) -> pd.DataFrame:
    """If the model exposes expected feature names, add any missing ones (NaN) and order columns."""
    # Try to discover expected names
    expected = None
    # direct
    expected = getattr(model, "feature_names_in_", None)
    # pipeline last step
    if expected is None and hasattr(model, "named_steps"):
        try:
            last = list(model.named_steps.values())[-1]
            expected = getattr(last, "feature_names_in_", None)
        except Exception:
            pass
    # ColumnTransformer sometimes has get_feature_names_out AFTER fit; but final model
    # usually only knows the transformed array names, not raw feature names—skip that.

    if expected is None:
        # No explicit contract: return as-is; pipeline should handle unknowns.
        return df_feats

    expected = list(expected)

    # Provide common aliases if the model trained with slightly different keys
    alias_map = {
        "days_until_game": "days_until",
        "day_of_week": "weekday",
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "kickoff": "startDateEastern",
    }
    for exp in list(expected):
        if exp not in df_feats.columns and exp in alias_map and alias_map[exp] in df_feats.columns:
            df_feats[exp] = df_feats[alias_map[exp]]

    # Add any still-missing columns as NaN so predict() won’t error
    for exp in expected:
        if exp not in df_feats.columns:
            df_feats[exp] = np.nan

    # Order columns to match model
    df_feats = df_feats.reindex(columns=expected, fill_value=np.nan)
    return df_feats

# ---------- Public API ----------
def predict_for_times(row: Dict, times: Iterable[pd.Timestamp], model_path: str | Path | None = None) -> List[float]:
    """Return predicted prices for the given timestamps using your saved model.

    - Builds a robust feature frame with common aliases (days_until & days_until_game, etc.).
    - If the model declares expected raw feature names, missing ones are added as NaN and ordered.
    """
    model = _load_model(model_path)
    feats = [_build_features_for_time(row, t) for t in times]
    df_feats = pd.DataFrame(feats)

    df_feats = _align_to_model_requirements(model, df_feats)

    yhat = model.predict(df_feats)
    return pd.to_numeric(pd.Series(yhat), errors="coerce").astype(float).tolist()
