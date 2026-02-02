"""
Predict price trajectories using the CatBoost gap_pct model.

The model predicts gap_pct (log1p) and we convert to min price via:
  predicted_min = current_price * (1 - gap_pct_pred)
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except Exception as e:  # pragma: no cover
    raise ImportError("CatBoost is required. Install with `pip install catboost`.") from e

# ---------- Repo paths ----------
_THIS = Path(__file__).resolve()
PROJ_DIR = _THIS.parents[2]
YEAR = int(os.getenv("SEASON_YEAR", "2025"))
DEFAULT_MODEL_PATH = PROJ_DIR / "models" / "best" / f"catboost_price_min_{YEAR}.cbm"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = PROJ_DIR / "models" / f"catboost_price_min_{YEAR}.cbm"


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


def _kickoff_ts(row: Dict) -> pd.Timestamp:
    for key in ("startDateEastern", "kickoff_ts", "start_time", "startDate", "date_local"):
        if key in row and pd.notna(row[key]):
            dt = _to_dt(row[key])
            if pd.notna(dt):
                try:
                    return dt.tz_localize(None)
                except Exception:
                    return dt
    return pd.NaT


def _coerce_booleans(df: pd.DataFrame, bool_cols=None) -> pd.DataFrame:
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


@lru_cache(maxsize=1)
def _load_model(model_path: str | Path | None = None) -> CatBoostRegressor:
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"CatBoost model not found at: {path}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def _feature_names(model: CatBoostRegressor) -> List[str]:
    try:
        return model.feature_names_
    except Exception:
        return model.get_feature_names()


def _build_feature_frame(
    row: Dict, times: Iterable[pd.Timestamp], model: CatBoostRegressor
) -> pd.DataFrame:
    feat_names = _feature_names(model)
    kickoff = _kickoff_ts(row)
    if pd.isna(kickoff):
        raise ValueError("Missing/invalid kickoff timestamp for trajectory prediction.")

    times = [pd.Timestamp(t) for t in times]
    hours_until = np.array([(kickoff - t).total_seconds() / 3600.0 for t in times], dtype=float)

    base = {
        "capacity": _to_float(row.get("capacity")),
        "neutralSite": row.get("neutralSite"),
        "conferenceGame": row.get("conferenceGame"),
        "isRivalry": row.get("isRivalry"),
        "isRankedMatchup": row.get("isRankedMatchup"),
        "homeTeamRank": _to_float(row.get("homeTeamRank")),
        "awayTeamRank": _to_float(row.get("awayTeamRank")),
        "week": _to_float(row.get("week")),
        "home_last_point_diff_at_snapshot": _to_float(row.get("home_last_point_diff_at_snapshot")),
        "away_last_point_diff_at_snapshot": _to_float(row.get("away_last_point_diff_at_snapshot")),
        "kickoff_hour": kickoff.hour if pd.notna(kickoff) else np.nan,
        "kickoff_dayofweek": kickoff.dayofweek if pd.notna(kickoff) else np.nan,
        "homeTeam": row.get("homeTeam"),
        "awayTeam": row.get("awayTeam"),
        "homeConference": row.get("homeConference"),
        "awayConference": row.get("awayConference"),
        "stadium": row.get("stadium"),
    }

    n = len(times)
    df = pd.DataFrame({"hours_until_game": hours_until})
    for k, v in base.items():
        df[k] = [v] * n

    # Coerce numeric + booleans
    for col in df.columns:
        if col in ["capacity", "homeTeamRank", "awayTeamRank", "week",
                   "home_last_point_diff_at_snapshot", "away_last_point_diff_at_snapshot",
                   "kickoff_hour", "kickoff_dayofweek", "hours_until_game"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _coerce_booleans(df)

    # Missing indicators expected by model
    for feat in feat_names:
        if feat.endswith("_missing"):
            base_col = feat[:-8]
            if base_col in df.columns:
                df[feat] = df[base_col].isna().astype(int)
            else:
                df[feat] = 1

    # Ensure all expected features exist
    for f in feat_names:
        if f not in df.columns:
            df[f] = np.nan

    # Ensure categoricals are strings
    for f in feat_names:
        if df[f].dtype == object:
            df[f] = df[f].astype(str).fillna("NA")

    return df[feat_names]


def predict_for_times(
    row: Dict, times: Iterable[pd.Timestamp], model_path: str | Path | None = None
) -> List[float]:
    """
    Predict minimum price at each time in `times`.
    Uses the CatBoost gap_pct model and converts using current snapshot price.
    """
    model = _load_model(model_path)
    df = _build_feature_frame(row, times, model)
    preds = model.predict(df)
    gap_pct = np.expm1(pd.to_numeric(pd.Series(preds), errors="coerce")).clip(0, 1)

    current_price = _to_float(row.get("lowest_price"))
    if not np.isfinite(current_price):
        # fallback: use observed lowest price if available
        current_price = _to_float(row.get("observed_lowest_price_num"))
    if not np.isfinite(current_price):
        current_price = np.nan

    if np.isnan(current_price):
        return [np.nan] * len(gap_pct)
    return (current_price * (1.0 - gap_pct)).tolist()


def predict_trajectory(
    row: Dict,
    hours_back: int = 24 * 30,
    step_hours: int = 6,
    model_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    kickoff = _kickoff_ts(row)
    if pd.isna(kickoff):
        raise ValueError("Missing/invalid kickoff timestamp for trajectory simulation.")

    hours = np.arange(1, int(hours_back) + 1, int(step_hours), dtype=int)
    ts_grid = [kickoff - pd.Timedelta(int(h), "h") for h in hours]
    prices = predict_for_times(row, ts_grid, model_path=model_path)

    df = pd.DataFrame({"when": ts_grid, "yhat": prices})
    summary = {
        "predicted_min_price": float(np.nanmin(df["yhat"])) if df["yhat"].notna().any() else np.nan,
        "predicted_min_time": df.loc[df["yhat"].idxmin(), "when"] if df["yhat"].notna().any() else pd.NaT,
    }
    return df, summary
