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
MERGED_OUT  = _resolve_file("MERGED_OUT",  Path("data") / "predicted" / "predicted_with_context.csv")

print("[predict_price] Paths resolved:")
print(f"  PROJ_DIR:    {PROJ_DIR}")
print(f"  MODEL_PATH:  {MODEL_PATH}")
print(f"  PRICE_PATH:  {PRICE_PATH}")
print(f"  OUTPUT_PATH: {OUTPUT_PATH}")
print(f"  MERGED_OUT:  {MERGED_OUT}")

# -----------------------------
# Write safety (atomic + guard)
# -----------------------------
def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically to avoid partial/overwritten writes on Windows/OneDrive/AV."""
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
# Sim grid (unchanged)
# -----------------------------
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
MAX_DAYS_OUT = 30

# --- FEATURES (must match training) ---
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
]

def _coerce_booleans(df, bool_cols=None):
    """Normalize boolean-like columns robustly."""
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

def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    return joblib.load(MODEL_PATH)

def _prep_games_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Schema check
    missing = [c for c in MIN_REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for simulation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Types aligned with training
    df = _coerce_booleans(df, ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"])

    # Coerce numeric columns that sometimes arrive as strings
    for col in ["capacity", "homeTeamRank", "awayTeamRank", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build datetime and drop unknowns
    df["startDateEastern"] = df.apply(_compose_start_datetime, axis=1)
    df = df.dropna(subset=["startDateEastern"])

    # One row per event (first seen snapshot)
    df_games = (
        df.sort_values(by=["startDateEastern"])
          .drop_duplicates(subset=["event_id"], keep="first")
          .copy()
    )

    # Ensure week exists and is int
    if "week" in df_games.columns:
        df_games["week"] = pd.to_numeric(df_games["week"], errors="coerce").fillna(0).astype(int)
    else:
        df_games["week"] = 0

    keep_cols = ["event_id", "startDateEastern"] + FEATURES
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    return df_games[keep_cols]

def _simulate_one_game(game_row: pd.Series, model) -> dict:
    """Choose best (day, time) combo by minimizing predicted price."""
    game_dt = pd.to_datetime(game_row["startDateEastern"])
    game_date = game_dt.date()

    days = np.arange(1, MAX_DAYS_OUT + 1, dtype=int)
    times = COLLECTION_TIMES

    sim_days = np.repeat(days, len(times))
    sim_times = np.tile(times, len(days))

    base_feats = {f: game_row.get(f, None) for f in FEATURES}
    sim_df = pd.DataFrame(base_feats, index=np.arange(len(sim_days)))
    sim_df["days_until_game"] = sim_days

    preds = model.predict(sim_df)
    best_idx = int(np.argmin(preds))
    best_price = float(preds[best_idx])
    best_delta = int(sim_days[best_idx])
    best_time_str = sim_times[best_idx]

    best_date = game_date - timedelta(days=best_delta)
    best_time_fmt = datetime.strptime(best_time_str, "%H:%M").time().strftime("%H:%M")

    return {
        "event_id": game_row.get("event_id"),
        "startDateEastern": pd.to_datetime(game_row.get("startDateEastern")).isoformat(),
        "week": int(game_row.get("week")) if pd.notna(game_row.get("week")) else None,
        "homeConference": game_row.get("homeConference"),
        "awayConference": game_row.get("awayConference"),
        "predicted_lowest_price": round(best_price, 2),
        "optimal_purchase_date": best_date.isoformat(),
        "optimal_purchase_time": best_time_fmt,
    }

def main():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Snapshot CSV not found at '{PRICE_PATH}'")

    # READ-ONLY
    price_df = pd.read_csv(PRICE_PATH)
    games_df = _prep_games_frame(price_df)
    model = _load_model()

    results = [_simulate_one_game(row, model) for _, row in games_df.iterrows()]
    out = pd.DataFrame(results)

    # NEVER write to snapshots; assert safety and write atomically to /predicted
    _assert_not_snapshot(OUTPUT_PATH)
    _write_csv_atomic(out, OUTPUT_PATH)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")

    # Optional merged artifact for convenience (still read-only snapshot)
    if "event_id" in price_df.columns:
        merged = price_df.merge(out, on="event_id", how="left")
        _assert_not_snapshot(MERGED_OUT)
        _write_csv_atomic(merged, MERGED_OUT)
        print(f"📝 Merged snapshot+predictions saved to {MERGED_OUT}")

if __name__ == "__main__":
    main()
