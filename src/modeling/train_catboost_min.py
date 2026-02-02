"""
Train CatBoost models to predict:
  1) future minimum price (from a snapshot)
  2) time-to-min in hours

Uses group-time split by event_id (no leakage) and per-event sample weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.status import write_status
import re

try:
    from catboost import CatBoostRegressor
except Exception as e:  # pragma: no cover
    raise ImportError(
        "CatBoost is required. Install with `pip install catboost`."
    ) from e


# -----------------------------
# Repo-locked paths
# -----------------------------
def _find_repo_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parent.parent


_THIS = Path(__file__).resolve()
PROJ_DIR = _find_repo_root(_THIS)

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"


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


YEAR = int(os.getenv("SEASON_YEAR", "2025"))
SNAPSHOT_PATH = _resolve_file("SNAPSHOT_PATH", Path("data") / "daily" / f"price_snapshots_{YEAR}.csv")
SNAPSHOT_FALLBACK = _resolve_file("SNAPSHOT_PATH_FALLBACK", Path("data") / "daily" / "price_snapshots.csv")
MODEL_DIR = _resolve_file("MODEL_DIR", Path("models"))
REPORT_PATH = _resolve_file("MODEL_REPORT_PATH", Path("reports") / f"catboost_report_{YEAR}.csv")
DATASET_PATH = _resolve_file("TRAIN_DATASET_PATH", Path("data") / "modeling" / f"train_rows_{YEAR}.csv")
FEATURES_PATH = _resolve_file("MODEL_FEATURES_PATH", Path("reports") / f"catboost_features_{YEAR}.csv")
PRICE_GAP_CLIP_PCT = float(os.getenv("PRICE_GAP_CLIP_PCT", "99.5"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.1"))
PRICE_WEIGHT_MIN = float(os.getenv("PRICE_WEIGHT_MIN", "10"))
PRICE_WEIGHT_ALPHA = float(os.getenv("PRICE_WEIGHT_ALPHA", "1.5"))
PRUNE_FEATURES = os.getenv("PRUNE_FEATURES", "1") == "1"
PRUNE_IMPORTANCE_CUTOFF = float(os.getenv("PRUNE_IMPORTANCE_CUTOFF", "0.95"))


# -----------------------------
# Feature schema (with toggles)
# -----------------------------
# Default to "on" so all features are included unless explicitly disabled.
USE_NEUTRAL_SITE = os.getenv("USE_NEUTRAL_SITE", "1") == "1"
USE_CONFERENCE_GAME = os.getenv("USE_CONFERENCE_GAME", "0") == "1"
USE_RIVALRY = os.getenv("USE_RIVALRY", "1") == "1"
USE_RANKED_MATCHUP = os.getenv("USE_RANKED_MATCHUP", "1") == "1"
USE_AWAY_RANK = os.getenv("USE_AWAY_RANK", "0") == "1"
USE_KICKOFF_DAY = os.getenv("USE_KICKOFF_DAY", "1") == "1"
USE_AWAY_TEAM = os.getenv("USE_AWAY_TEAM", "1") == "1"
USE_STADIUM = os.getenv("USE_STADIUM", "0") == "1"

NUMERIC_FEATURES = [
    "hours_until_game",
    "capacity",
    "week",
    "kickoff_hour",
    "home_last_point_diff_at_snapshot",
    "away_last_point_diff_at_snapshot",
]

if USE_NEUTRAL_SITE:
    NUMERIC_FEATURES.append("neutralSite")
if USE_CONFERENCE_GAME:
    NUMERIC_FEATURES.append("conferenceGame")
if USE_RIVALRY:
    NUMERIC_FEATURES.append("isRivalry")
if USE_RANKED_MATCHUP:
    NUMERIC_FEATURES.append("isRankedMatchup")
if USE_AWAY_RANK:
    NUMERIC_FEATURES.append("awayTeamRank")
if USE_KICKOFF_DAY:
    NUMERIC_FEATURES.append("kickoff_dayofweek")

CATEGORICAL_FEATURES = [
    "homeTeam",
    "homeConference",
    "awayConference",
]
if USE_AWAY_TEAM:
    CATEGORICAL_FEATURES.append("awayTeam")
if USE_STADIUM:
    CATEGORICAL_FEATURES.append("stadium")

FUTURE_MIN_PRICE = "future_min_price"
TARGET_PRICE = "gap_pct"
TARGET_TIME = "time_to_min_hours"
TARGET_PRICE_LOG = "gap_pct_log"
TARGET_TIME_LOG = "time_to_min_hours_log"

EVENT_ID_RE = re.compile(r"/(\d{6,})/?$")
POSTSEASON_RE = re.compile(
    r"\b(bowl|playoff|first round|quarterfinal|semifinal|final|championship|cfp)\b",
    flags=re.IGNORECASE,
)

META_COLUMNS = ["lowest_price"]


def _prune_by_importance(
    numeric_features: List[str],
    categorical_features: List[str],
    cutoff: float,
) -> Tuple[List[str], List[str]]:
    """Prune features using cumulative importance from the last run."""
    if not FEATURES_PATH.exists():
        return numeric_features, categorical_features
    try:
        feats = pd.read_csv(FEATURES_PATH)
        if feats.empty or "feature" not in feats.columns or "importance" not in feats.columns:
            return numeric_features, categorical_features
        feats = feats.sort_values("importance", ascending=False).reset_index(drop=True)
        feats["cum_importance"] = feats["importance"].cumsum() / feats["importance"].sum()
        keep = feats[feats["cum_importance"] <= cutoff]["feature"].tolist()
        # Ensure we keep the exact elbow feature at the cutoff boundary
        if len(keep) < len(feats):
            keep.append(feats.loc[len(keep), "feature"])
        keep_set = set(keep)
        num_kept = [c for c in numeric_features if c in keep_set]
        cat_kept = [c for c in categorical_features if c in keep_set]
        return num_kept, cat_kept
    except Exception:
        return numeric_features, categorical_features

def _keep_columns(numeric_features: List[str], meta_columns: List[str]) -> List[str]:
    return (
        ["event_id", "_snapshot_ts", "_kickoff_ts"]
        + meta_columns
        + numeric_features
        + CATEGORICAL_FEATURES
        + [TARGET_PRICE, TARGET_TIME, TARGET_PRICE_LOG, TARGET_TIME_LOG]
    )


# -----------------------------
# Helpers
# -----------------------------
def _best_snapshot_ts(df: pd.DataFrame) -> pd.Series:
    candidates = ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at"]
    time_only = ["time_collected", "collection_time", "snapshot_time"]
    ts = pd.Series(pd.NaT, index=df.index)
    # Coalesce across all possible timestamp columns (not just the first hit).
    for c in candidates:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            ts = ts.fillna(parsed)
    tcol = next((c for c in time_only if c in df.columns), None)
    if "date_collected" in df.columns:
        date_raw = df["date_collected"].astype(str).str.strip()
        # Handle mixed formats by trying US m/d/Y first, then ISO Y-m-d.
        date_dt = pd.to_datetime(date_raw, errors="coerce", format="%m/%d/%Y")
        date_dt = date_dt.fillna(pd.to_datetime(date_raw, errors="coerce", format="%Y-%m-%d"))
        if tcol:
            time_raw = df[tcol].astype(str).str.strip()
            time_norm = time_raw.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
            time_td = pd.to_timedelta(time_norm, errors="coerce")
            ts = ts.fillna(date_dt.dt.normalize() + time_td)
        ts = ts.fillna(date_dt)
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    return ts


def _kickoff_ts(row: pd.Series) -> pd.Timestamp:
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    ts = pd.to_datetime(dt_str, errors="coerce")
    try:
        ts = ts.tz_localize(None)
    except Exception:
        pass
    return ts


def _coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    truth_map = {
        True: True, False: False,
        "true": True, "false": False, "True": True, "False": False,
        "yes": True, "no": False, "YES": True, "NO": False,
        "y": True, "n": False, "Y": True, "N": False,
        1: True, 0: False, "1": True, "0": False,
        "t": True, "f": False, "T": True, "F": False,
        np.nan: np.nan,
    }
    for c in ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]:
        if c in df.columns:
            s = df[c]
            if not pd.api.types.is_bool_dtype(s):
                s = s.map(truth_map).astype("boolean")
            df[c] = s.fillna(False).astype(bool)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "event_id" not in df.columns:
        df["event_id"] = np.nan

    if "offer_url" in df.columns:
        def _extract_id(url: object) -> Optional[str]:
            if not isinstance(url, str):
                return None
            m = EVENT_ID_RE.search(url.strip())
            return m.group(1) if m else None
        df["event_id"] = df["event_id"].fillna(df["offer_url"].map(_extract_id))

    if df["event_id"].isna().all():
        parts = []
        for col in ["homeTeam", "awayTeam", "date_local", "time_local", "team_slug", "title"]:
            if col in df.columns:
                parts.append(df[col].astype(str).fillna(""))
        if parts:
            df["event_id"] = ("event_" + parts[0]).astype(str)
            for p in parts[1:]:
                df["event_id"] = df["event_id"] + "|" + p
        else:
            raise ValueError("event_id missing and no fallback columns available.")

    df["event_id"] = df["event_id"].astype(str)
    return df


def _is_postseason_row(df: pd.DataFrame) -> pd.Series:
    if "is_postseason" in df.columns:
        return df["is_postseason"].fillna(False).astype(bool)
    if "title" in df.columns:
        return df["title"].fillna("").astype(str).str.contains(POSTSEASON_RE, regex=True)
    return pd.Series(False, index=df.index)


def _event_min_snapshot_ts(g: pd.DataFrame) -> pd.Timestamp:
    prices = pd.to_numeric(g["lowest_price"], errors="coerce")
    if prices.notna().any():
        min_price = prices.min()
        return g.loc[prices == min_price, "_snapshot_ts"].min()
    return pd.NaT


def _apply_completion_overrides(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    now_ts = pd.Timestamp.now().tz_localize(None)

    kickoff = df.groupby("event_id")["_kickoff_ts"].min()
    complete = kickoff < now_ts
    df["event_complete"] = df["event_id"].map(complete).fillna(False).astype(bool)

    event_min_price = df.groupby("event_id")["lowest_price"].min()
    try:
        min_ts = df.groupby("event_id", group_keys=False).apply(_event_min_snapshot_ts, include_groups=False)
    except TypeError:
        min_ts = df.groupby("event_id", group_keys=False).apply(_event_min_snapshot_ts)

    df["event_min_price"] = df["event_id"].map(event_min_price)
    df["event_min_snapshot_ts"] = df["event_id"].map(min_ts)

    df["event_min_date"] = df["event_min_snapshot_ts"].dt.date.astype(str)
    df["event_min_time"] = df["event_min_snapshot_ts"].dt.time.astype(str)
    df["min_already_passed"] = df["event_min_snapshot_ts"].notna() & (df["_snapshot_ts"] > df["event_min_snapshot_ts"])

    mask = df["event_complete"] & df["event_min_price"].notna()
    if mask.any():
        df.loc[mask, FUTURE_MIN_PRICE] = df.loc[mask, "event_min_price"]
        dt_hours = (df.loc[mask, "event_min_snapshot_ts"] - df.loc[mask, "_snapshot_ts"]).dt.total_seconds() / 3600.0
        df.loc[mask, TARGET_TIME] = dt_hours.clip(lower=0)

    gap_abs = (df["lowest_price"] - df[FUTURE_MIN_PRICE]).clip(lower=0)
    gap_pct = gap_abs / df["lowest_price"].replace(0, np.nan)
    df[TARGET_PRICE] = gap_pct.clip(lower=0, upper=1)
    # Robustness: clip extreme gaps to reduce outlier impact.
    if df[TARGET_PRICE].notna().any():
        clip_val = df[TARGET_PRICE].quantile(PRICE_GAP_CLIP_PCT / 100.0)
        df[TARGET_PRICE] = df[TARGET_PRICE].clip(upper=clip_val)
    # Log targets for stability
    df[TARGET_PRICE_LOG] = np.log1p(df[TARGET_PRICE])
    df[TARGET_TIME_LOG] = np.log1p(df[TARGET_TIME].clip(lower=0))
    return df


def _build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_event_id(df)
    df = _coerce_booleans(df)
    df = _build_targets(df)
    df = _apply_completion_overrides(df)
    return df


def _keep_model_columns(df: pd.DataFrame, numeric_features: List[str], meta_columns: List[str]) -> pd.DataFrame:
    keep_cols = _keep_columns(numeric_features, meta_columns)
    keep = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Missing expected columns (will be skipped): {', '.join(missing)}")
    return df[keep].copy()


def _add_missing_indicators(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    added = []
    for c in cols:
        if c in df.columns:
            miss_col = f"{c}_missing"
            df[miss_col] = df[c].isna().astype(int)
            added.append(miss_col)
    return df, added


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_snapshot_ts"] = _best_snapshot_ts(df)
    df["_kickoff_ts"] = df.apply(_kickoff_ts, axis=1)
    df["hours_until_game"] = (df["_kickoff_ts"] - df["_snapshot_ts"]).dt.total_seconds() / 3600.0
    # Kickoff time features
    df["kickoff_hour"] = df["_kickoff_ts"].dt.hour.astype("float")
    df["kickoff_dayofweek"] = df["_kickoff_ts"].dt.dayofweek.astype("float")

    # Build targets per event_id using future snapshots (including current)
    df["lowest_price"] = pd.to_numeric(df["lowest_price"], errors="coerce")
    df = df[df["event_id"].notna()].copy()
    df = df.sort_values(["event_id", "_snapshot_ts"])

    future_min_price = []
    time_to_min = []

    for _, g in df.groupby("event_id", sort=False):
        prices = g["lowest_price"].to_numpy()
        times = g["_snapshot_ts"].to_numpy()

        # compute suffix min ignoring NaN (treat NaN as +inf)
        prices_for_min = np.where(np.isnan(prices), np.inf, prices)
        suffix_min = np.minimum.accumulate(prices_for_min[::-1])[::-1]
        suffix_min = np.where(np.isinf(suffix_min), np.nan, suffix_min)

        # for time_to_min, we need index of first min occurrence in suffix
        min_idx = []
        current_min = np.inf
        current_idx = None
        for i in range(len(prices) - 1, -1, -1):
            p = prices_for_min[i]
            if np.isinf(p):
                min_idx.append(current_idx)
                continue
            if p <= current_min:
                current_min = p
                current_idx = i
            min_idx.append(current_idx)
        min_idx = list(reversed(min_idx))

        future_min_price.extend(suffix_min.tolist())
        # time to min
        for i, idx in enumerate(min_idx):
            if idx is None or pd.isna(times[i]) or pd.isna(times[idx]):
                time_to_min.append(np.nan)
            else:
                dt = (times[idx] - times[i]) / np.timedelta64(1, "h")
                time_to_min.append(float(dt))

    df[FUTURE_MIN_PRICE] = future_min_price
    df[TARGET_TIME] = time_to_min
    return df


def _add_price_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features using past snapshots only."""
    df = df.copy()
    df = df.sort_values(["event_id", "_snapshot_ts"])
    event_ids = df["event_id"].copy()

    # Ensure numeric
    for c in ["lowest_price", "average_price", "highest_price", "listing_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _per_event(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["price_prev"] = g["lowest_price"].shift(1)
        g["price_delta"] = g["lowest_price"] - g["price_prev"]
        g["price_delta_pct"] = g["price_delta"] / g["price_prev"].replace(0, np.nan)
        g["price_roll3_mean"] = g["lowest_price"].rolling(3, min_periods=1).mean()
        g["price_roll3_std"] = g["lowest_price"].rolling(3, min_periods=2).std()
        g["snapshots_so_far"] = np.arange(1, len(g) + 1)
        # time since last snapshot
        prev_ts = g["_snapshot_ts"].shift(1)
        g["time_since_last_snapshot_hours"] = (g["_snapshot_ts"] - prev_ts).dt.total_seconds() / 3600.0
        return g

    grouped = df.groupby("event_id", group_keys=False)
    try:
        df = grouped.apply(_per_event, include_groups=False)
        # include_groups=False drops the grouping column; reattach by index.
        df["event_id"] = event_ids.loc[df.index].values
    except TypeError:
        df = grouped.apply(_per_event)
    return df


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _group_time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Split:
    # Group by event, sort by kickoff time
    g = df.groupby("event_id")["_kickoff_ts"].min().dropna().sort_values()
    events = g.index.to_numpy()
    n_test = max(1, int(len(events) * test_frac))
    test_events = set(events[-n_test:])
    is_test = df["event_id"].isin(test_events)
    return Split(train_idx=np.where(~is_test)[0], test_idx=np.where(is_test)[0])


def _event_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df["event_id"].value_counts()
    w = df["event_id"].map(lambda x: 1.0 / counts.get(x, 1))
    return w.to_numpy()


def _evaluate_gap_pct(y_true: np.ndarray, y_pred: np.ndarray, tol_abs: float) -> dict:
    mae = np.nanmean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))
    within = np.nanmean(np.abs(y_true - y_pred) <= tol_abs)
    return {"mae": mae, "rmse": rmse, "within": within}


def _evaluate_price(y_true: np.ndarray, y_pred: np.ndarray, tol_frac: float) -> dict:
    mae = np.nanmean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))
    rel = np.abs(y_true - y_pred) / np.maximum(1e-6, np.abs(y_true))
    within = np.nanmean(rel <= tol_frac)
    return {"mae": mae, "rmse": rmse, "within": within}


def _bucketed_price_mae(y_true: np.ndarray, y_pred: np.ndarray, bins=None) -> dict:
    if bins is None:
        bins = [0, 20, 50, 100, 200, np.inf]
    labels = [f"${bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    out = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            out[f"price_mae_{labels[i]}"] = float(np.nanmean(np.abs(y_true[mask] - y_pred[mask])))
        else:
            out[f"price_mae_{labels[i]}"] = np.nan
    return out


def train():
    if SNAPSHOT_PATH.exists():
        df = pd.read_csv(SNAPSHOT_PATH)
    elif SNAPSHOT_FALLBACK.exists():
        df = pd.read_csv(SNAPSHOT_FALLBACK, low_memory=False)
    else:
        raise FileNotFoundError("No snapshot file found.")

    # Exclude postseason games from modeling
    df = df[~_is_postseason_row(df)].copy()

    df = _build_training_dataset(df)
    counts = df["event_id"].value_counts()
    df = df[df["event_id"].map(counts) > 1].copy()
    df, missing_cols = _add_missing_indicators(df, NUMERIC_FEATURES)
    numeric_features = NUMERIC_FEATURES + missing_cols
    if PRUNE_FEATURES:
        numeric_features, categorical_features = _prune_by_importance(
            numeric_features, CATEGORICAL_FEATURES, PRUNE_IMPORTANCE_CUTOFF
        )
    else:
        categorical_features = CATEGORICAL_FEATURES
    df = _keep_model_columns(df, numeric_features, META_COLUMNS)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    # Filter rows with valid targets and features
    req = numeric_features + categorical_features + [TARGET_PRICE, TARGET_TIME]
    # Only require essential time features and targets; allow other feature NaNs.
    df = df.dropna(subset=[TARGET_PRICE, "hours_until_game"])
    df = df[df["hours_until_game"] > 0]

    # Ensure numeric
    for c in numeric_features + [TARGET_PRICE, TARGET_TIME, TARGET_PRICE_LOG, TARGET_TIME_LOG]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Split
    split = _group_time_split(df, test_frac=0.2)
    train_df = df.iloc[split.train_idx].copy()
    test_df = df.iloc[split.test_idx].copy()

    # Validation split from training (event-grouped) for early stopping.
    val_split = _group_time_split(train_df, test_frac=VAL_FRAC)
    train_inner = train_df.iloc[val_split.train_idx].copy()
    val_df = train_df.iloc[val_split.test_idx].copy()

    # Separate datasets for price and time (time target may be missing for some rows)
    price_train = train_inner[train_inner[TARGET_PRICE_LOG].notna()].copy()
    price_val = val_df[val_df[TARGET_PRICE_LOG].notna()].copy()
    price_test = test_df[test_df[TARGET_PRICE_LOG].notna()].copy()
    time_train = train_inner[train_inner[TARGET_TIME_LOG].notna()].copy()
    time_val = val_df[val_df[TARGET_TIME_LOG].notna()].copy()
    time_test = test_df[test_df[TARGET_TIME_LOG].notna()].copy()

    # Sample weights: event-balance + extra emphasis on low-price games.
    w_price = _event_weights(price_train)
    if "lowest_price" in price_train.columns:
        lp = pd.to_numeric(price_train["lowest_price"], errors="coerce")
        lp = lp.fillna(lp.median())
        base = 1.0 / np.maximum(lp.to_numpy(), PRICE_WEIGHT_MIN)
        price_scale = np.power(base, PRICE_WEIGHT_ALPHA)
        w_price = w_price * price_scale
    w_time = _event_weights(time_train)

    # Build CatBoost pools
    X_price_train = price_train[numeric_features + categorical_features].copy()
    X_price_val = price_val[numeric_features + categorical_features].copy()
    X_price_test = price_test[numeric_features + categorical_features].copy()
    X_time_train = time_train[numeric_features + categorical_features].copy()
    X_time_val = time_val[numeric_features + categorical_features].copy()
    X_time_test = time_test[numeric_features + categorical_features].copy()

    # CatBoost requires categorical values to be strings (no NaN).
    for c in categorical_features:
        if c in X_price_train.columns:
            X_price_train.loc[:, c] = X_price_train[c].astype(str).fillna("NA")
            X_price_val.loc[:, c] = X_price_val[c].astype(str).fillna("NA")
            X_price_test.loc[:, c] = X_price_test[c].astype(str).fillna("NA")
            X_time_train.loc[:, c] = X_time_train[c].astype(str).fillna("NA")
            X_time_val.loc[:, c] = X_time_val[c].astype(str).fillna("NA")
            X_time_test.loc[:, c] = X_time_test[c].astype(str).fillna("NA")

    cat_features = [X_price_train.columns.get_loc(c) for c in categorical_features if c in X_price_train.columns]
    # Monotonic constraints: price gap should generally decrease as kickoff approaches.
    monotone = {c: 0 for c in X_price_train.columns}
    if "hours_until_game" in monotone:
        monotone["hours_until_game"] = 1  # higher hours -> higher gap expected
    if "days_until_game" in monotone:
        monotone["days_until_game"] = 1
    monotone_constraints = [monotone[c] for c in X_price_train.columns]

    # Price model
    price_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2000,
        depth=9,
        learning_rate=0.035,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        monotone_constraints=monotone_constraints,
        verbose=False,
    )
    price_model.fit(
        X_price_train,
        price_train[TARGET_PRICE_LOG],
        cat_features=cat_features,
        sample_weight=w_price,
        eval_set=(X_price_val, price_val[TARGET_PRICE_LOG]),
        use_best_model=True,
    )

    # Time model
    time_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2000,
        depth=9,
        learning_rate=0.035,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        verbose=False,
    )
    time_model.fit(
        X_time_train,
        time_train[TARGET_TIME_LOG],
        cat_features=cat_features,
        sample_weight=w_time,
        eval_set=(X_time_val, time_val[TARGET_TIME_LOG]),
        use_best_model=True,
    )

    # Evaluate
    y_pred_gap = np.expm1(price_model.predict(X_price_test)).clip(0, 1)
    y_pred_time = np.expm1(time_model.predict(X_time_test))

    price_metrics = _evaluate_gap_pct(price_test[TARGET_PRICE].to_numpy(), y_pred_gap, tol_abs=0.05)
    # Convert gap_pct back to predicted min price for reporting
    true_min_price = price_test["lowest_price"].to_numpy() * (1.0 - price_test[TARGET_PRICE].to_numpy())
    pred_min_price = price_test["lowest_price"].to_numpy() * (1.0 - y_pred_gap)
    price_abs_metrics = _evaluate_price(true_min_price, pred_min_price, tol_frac=0.05)
    price_bucket_metrics = _bucketed_price_mae(true_min_price, pred_min_price)
    time_mae = np.nanmean(np.abs(time_test[TARGET_TIME].to_numpy() - y_pred_time))
    time_within = np.nanmean(np.abs(time_test[TARGET_TIME].to_numpy() - y_pred_time) <= 24.0)

    report = pd.DataFrame([
        {
            "year": YEAR,
            "n_train_price": len(price_train),
            "n_test_price": len(price_test),
            "n_train_time": len(time_train),
            "n_test_time": len(time_test),
            "rows_total_used": len(df),
            "gap_pct_mae": price_metrics["mae"],
            "gap_pct_rmse": price_metrics["rmse"],
            "gap_pct_within_0p05": price_metrics["within"],
            "price_mae": price_abs_metrics["mae"],
            "price_rmse": price_abs_metrics["rmse"],
            "price_within_5pct": price_abs_metrics["within"],
            **price_bucket_metrics,
            "time_mae_hours": time_mae,
            "time_within_24h": time_within,
        }
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    price_model.save_model(str(MODEL_DIR / f"catboost_price_min_{YEAR}.cbm"))
    time_model.save_model(str(MODEL_DIR / f"catboost_time_to_min_{YEAR}.cbm"))
    # Save feature importances for debugging/pruning
    feat_importances = pd.DataFrame(
        {"feature": X_price_train.columns, "importance": price_model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_importances.to_csv(FEATURES_PATH, index=False)

    print("\nSTATUS: SUCCESS | CatBoost models trained")
    print("-" * 72)
    row = report.iloc[0].to_dict()
    print(f"year: {int(row['year'])}")
    print(f"rows_total_used: {int(row['rows_total_used'])}")
    print(f"n_train_price: {int(row['n_train_price'])}")
    print(f"n_test_price: {int(row['n_test_price'])}")
    print(f"n_train_time: {int(row['n_train_time'])}")
    print(f"n_test_time: {int(row['n_test_time'])}")
    print(f"gap_pct_mae: {row['gap_pct_mae']:.4f}")
    print(f"gap_pct_rmse: {row['gap_pct_rmse']:.4f}")
    print(f"gap_pct_within_0p05: {row['gap_pct_within_0p05']:.4f}")
    print(f"price_mae: {row['price_mae']:.4f}")
    print(f"price_rmse: {row['price_rmse']:.4f}")
    print(f"price_within_5pct: {row['price_within_5pct']:.4f}")
    for k in [k for k in row.keys() if k.startswith('price_mae_')]:
        v = row[k]
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"{k}: NA")
    print(f"time_mae_hours: {row['time_mae_hours']:.4f}")
    print(f"time_within_24h: {row['time_within_24h']:.4f}")
    print("-" * 72)

    write_status(
        "model_train",
        "success",
        f"CatBoost training complete for {YEAR}",
        row,
    )


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        write_status("model_train", "failed", str(e))
        raise
