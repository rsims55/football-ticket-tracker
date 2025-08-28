# =============================
# FILE: src/modeling/predict_price.py
# PURPOSE: (1) Use time-of-day feature 'collectionSlot' during simulation; (2) If any
# observed snapshot already has a lower price than the model's best future prediction,
# override the recommendation with that observed min (date/time/price).
# =============================
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

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
# Sim grid & features
# -----------------------------
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
MAX_DAYS_OUT = 30

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

# Minimum columns needed from snapshots
MIN_REQUIRED_INPUT = [
    "event_id",
    "date_local",  # time_local optional
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
    # NOTE: we will also try to infer a snapshot timestamp for observed-min logic
]

# -----------------------------
# Helpers — booleans, timestamps, observed-min
# -----------------------------

def _coerce_booleans(df, bool_cols=None):
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
        np.nan: np.nan
    }

    for c in bool_cols:
        s = df[c]
        if not pd.api.types.is_bool_dtype(s):
            s = s.map(truth_map).astype("boolean")
        df[c] = s.fillna(False).astype(bool)
    return df


def _compose_start_datetime(row) -> pd.Timestamp:
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    return pd.to_datetime(dt_str, errors="coerce")


# Build a best-effort snapshot timestamp for each row (when the snapshot was taken)
_SNAPSHOT_TS_CANDIDATES_DT = [
    "collected_at", "snapshot_datetime", "retrieved_at", "scraped_at",
]
_SNAPSHOT_TS_CANDIDATES_TIME = [
    "time_collected", "collection_time", "snapshot_time",
]

def _nearest_slot_from_ts(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "12:00"
    minutes = ts.hour * 60 + ts.minute
    idx = int(np.round(minutes / 360.0)) % 4
    return ["00:00", "06:00", "12:00", "18:00"][idx]


def _build_snapshot_ts(df: pd.DataFrame) -> pd.Series:
    ts = None

    # 1) direct datetime-like columns
    for c in _SNAPSHOT_TS_CANDIDATES_DT:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            break

    # 2) combine date_collected + a time-only column
    if ts is None or ts.isna().all():
        time_col = next((c for c in _SNAPSHOT_TS_CANDIDATES_TIME if c in df.columns), None)
        if "date_collected" in df.columns and time_col:
            ts = pd.to_datetime(df["date_collected"].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(), errors="coerce")

    # 3) fallback: just date_collected as midnight
    if ts is None or ts.isna().all():
        if "date_collected" in df.columns:
            ts = pd.to_datetime(df["date_collected"], errors="coerce")

    # 4) last resort: use file load time (non-ideal) — leave as NaT to avoid overrides when unknown
    if ts is None:
        ts = pd.Series(pd.NaT, index=df.index)

    return ts


def _observed_min_for_event(price_df: pd.DataFrame, event_id) -> tuple[float | None, pd.Timestamp | None]:
    sub = price_df[price_df.get("event_id").astype(str) == str(event_id)].copy()
    if sub.empty:
        return None, None

    # coerce price and drop NAs
    sub["lowest_price_num"] = pd.to_numeric(sub.get("lowest_price"), errors="coerce")
    sub = sub.dropna(subset=["lowest_price_num"])  # must have a numeric price
    if sub.empty:
        return None, None

    # build snapshot ts to recover date/time of observed min
    sub["snapshot_ts"] = _build_snapshot_ts(sub)

    # find idx of min price (first occurrence if ties)
    idx = sub["lowest_price_num"].idxmin()
    row = sub.loc[idx]
    return float(row["lowest_price_num"]), pd.to_datetime(row.get("snapshot_ts"), errors="coerce")

# -----------------------------
# Prep games
# -----------------------------

def _prep_games_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MIN_REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for simulation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["startDateEastern"] = df.apply(_compose_start_datetime, axis=1)
    df = df.dropna(subset=["startDateEastern"])

    df_games = (
        df.sort_values(by=["startDateEastern"])\
          .drop_duplicates(subset=["event_id"], keep="first")\
          .copy()
    )

    if "week" in df_games.columns:
        df_games["week"] = pd.to_numeric(df_games["week"], errors="coerce").fillna(0).astype(int)
    else:
        df_games["week"] = 0

    keep_cols = ["event_id", "startDateEastern"] + FEATURES
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    return df_games[keep_cols]

# -----------------------------
# Simulation per game (with observed-min override)
# -----------------------------

def _simulate_one_game(game_row: pd.Series, model, price_df_all: pd.DataFrame) -> dict:
    game_dt = pd.to_datetime(game_row["startDateEastern"])
    game_date = game_dt.date()

    days = np.arange(1, MAX_DAYS_OUT + 1, dtype=int)
    times = COLLECTION_TIMES

    sim_days = np.repeat(days, len(times))
    sim_times = np.tile(times, len(days))

    base_feats = {f: game_row.get(f, None) for f in FEATURES}
    sim_df = pd.DataFrame(base_feats, index=np.arange(len(sim_days)))
    sim_df["days_until_game"] = sim_days
    sim_df["collectionSlot"] = sim_times  # vary by time for the model

    preds = model.predict(sim_df)
    best_idx = int(np.argmin(preds))
    best_price_model = float(preds[best_idx])
    best_delta = int(sim_days[best_idx])
    best_time_str = sim_times[best_idx]

    best_date_model = game_date - timedelta(days=best_delta)

    # Observed minimum override logic
    obs_price, obs_ts = _observed_min_for_event(price_df_all, game_row.get("event_id"))

    if obs_price is not None and (np.isnan(best_price_model) or obs_price <= best_price_model):
        # Prefer observed if it is lower or equal
        use_price = round(float(obs_price), 2)
        if pd.isna(obs_ts):
            # if we cannot recover a timestamp, keep the model's time but observed price
            use_date = best_date_model.isoformat()
            use_time = best_time_str
        else:
            use_date = pd.to_datetime(obs_ts).date().isoformat()
            use_time = pd.to_datetime(obs_ts).strftime("%H:%M")
        source = "observed"
    else:
        use_price = round(best_price_model, 2)
        use_date = best_date_model.isoformat()
        use_time = best_time_str
        source = "model"

    return {
        "event_id": game_row.get("event_id"),
        "startDateEastern": pd.to_datetime(game_row.get("startDateEastern")).isoformat(),
        "week": int(game_row.get("week")) if pd.notna(game_row.get("week")) else None,
        "homeConference": game_row.get("homeConference"),
        "awayConference": game_row.get("awayConference"),
        "predicted_lowest_price": use_price,
        "optimal_purchase_date": use_date,
        "optimal_purchase_time": use_time,
        "optimal_source": source,
    }

# -----------------------------
# Main
# -----------------------------

def main():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Snapshot CSV not found at '{PRICE_PATH}'")

    price_df = pd.read_csv(PRICE_PATH)
    games_df = _prep_games_frame(price_df)
    model = _load_model()

    results = [_simulate_one_game(row, model, price_df) for _, row in games_df.iterrows()]
    out = pd.DataFrame(results)

    _assert_not_snapshot(OUTPUT_PATH)
    _write_csv_atomic(out, OUTPUT_PATH)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    main()
