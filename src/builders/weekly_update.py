#!/usr/bin/env python3
from __future__ import annotations

import sys, os, json, time, uuid
from datetime import datetime, date
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd

# --- Robust import path ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJ_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetchers.schedule_fetcher import ScheduleFetcher
from fetchers.rankings_fetcher import RankingsFetcher

YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

def _under_repo(p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(PROJ_DIR.resolve())
    except AttributeError:
        return str(p.resolve()).startswith(str(PROJ_DIR.resolve()))

def _resolve_dir(env_value: Optional[str], default: Path) -> Path:
    if REPO_DATA_LOCK or not env_value:
        return default
    p = Path(env_value).expanduser()
    if _under_repo(p) or ALLOW_ESCAPE:
        return p
    return default

DATA_DIR = _resolve_dir(os.getenv("DATA_DIR"), PROJ_DIR / "data")
if not _under_repo(DATA_DIR) and not ALLOW_ESCAPE:
    print(f"🚫 DATA_DIR resolved outside repo: {DATA_DIR} → forcing repo path")
    DATA_DIR = PROJ_DIR / "data"

WEEKLY_DIR    = DATA_DIR / "weekly"
ANNUAL_DIR    = DATA_DIR / "annual"
PERMANENT_DIR = DATA_DIR / "permanent"
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_JSON = PERMANENT_DIR / "team_aliases.json"
WEEKLY_SCHEDULE_OUT = WEEKLY_DIR / f"full_{YEAR}_schedule.csv"

STADIUM_CANDIDATES = [
    ANNUAL_DIR / f"stadiums_{YEAR}.csv",
    DATA_DIR   / f"stadiums_{YEAR}.csv",
]

print("[weekly_update] Paths resolved:")
print(f"  PROJ_DIR:            {PROJ_DIR}")
print(f"  DATA_DIR:            {DATA_DIR}")
print(f"  WEEKLY_DIR:          {WEEKLY_DIR}")
print(f"  ANNUAL_DIR:          {ANNUAL_DIR}")
print(f"  PERMANENT_DIR:       {PERMANENT_DIR}")
print(f"  ALIAS_JSON:          {ALIAS_JSON}")
print(f"  WEEKLY_SCHEDULE_OUT: {WEEKLY_SCHEDULE_OUT}")

if not ALIAS_JSON.exists():
    raise FileNotFoundError(
        f"Alias map not found at {ALIAS_JSON}. "
        "Please create data/permanent/team_aliases.json (a dict of {'Incoming Name': 'Canonical Name'})."
    )

with open(ALIAS_JSON, "r", encoding="utf-8") as f:
    alias_map: Dict[str, str] = json.load(f)

def get_alias(name: str) -> str:
    return alias_map.get(name, name)

def _canon(val):
    if pd.isna(val):
        return val
    return get_alias(str(val).strip())

def _pick_venue_column(df: pd.DataFrame) -> str:
    for c in ["venue", "stadium", "site", "venue_name"]:
        if c in df.columns:
            return c
    df["venue"] = pd.NA
    return "venue"

def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _norm_key(s) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return s.strip().lower()

def _to_date_yyyy_mm_dd(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    return dt.dt.date

def _atomic_csv_write(df: pd.DataFrame, path: Path, retries: int = 5, delay: float = 0.6) -> None:
    """Write via temp file + os.replace to avoid Windows sharing violations."""
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    for i in range(retries):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)  # atomic on Windows if target not locked
            return
        except PermissionError as e:
            print(f"⚠️ PermissionError writing {path}: attempt {i+1}/{retries} – {e}")
            time.sleep(delay)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
    alt = path.with_name(f"{path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    print(f"❌ Could not replace {path}. Writing fallback: {alt}")
    df.to_csv(alt, index=False)

def _ensure_week_column(df: pd.DataFrame) -> None:
    """Ensure schedule_df has a numeric 'week' column."""
    candidates = ["week", "weekNumber", "week_num", "gameWeek"]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise KeyError("schedule_df is missing a 'week' (or alias: weekNumber/week_num/gameWeek) column.")
    df["week"] = pd.to_numeric(df[found], errors="coerce").astype("Int64")

# ======================
# 0) Load previous snapshot (to preserve past weeks' ranks)
# ======================
prev_df: Optional[pd.DataFrame] = None
if WEEKLY_SCHEDULE_OUT.exists():
    try:
        prev_df = pd.read_csv(WEEKLY_SCHEDULE_OUT)
        prev_df["game_date"] = _to_date_yyyy_mm_dd(prev_df.get("startDateEastern"))
    except Exception as e:
        print(f"⚠️ Could not read previous weekly schedule: {e}")

# ======================
# 1) Fetch schedule
# ======================
print("📅 Fetching schedule…")
schedule_df = ScheduleFetcher(YEAR).fetch().copy()
if "startDateEastern" not in schedule_df.columns:
    raise KeyError("schedule_df is missing 'startDateEastern'")

# Canonicalize teams immediately
schedule_df["homeTeam"] = schedule_df["homeTeam"].apply(_canon)
schedule_df["awayTeam"] = schedule_df["awayTeam"].apply(_canon)

# Ensure 'week' exists (numeric) and compute the rankings week (schedule.week - 1, min 0)
_ensure_week_column(schedule_df)
schedule_df["rankings_week"] = (schedule_df["week"].fillna(1) - 1).clip(lower=0).astype(int)

schedule_df["game_date"] = _to_date_yyyy_mm_dd(schedule_df["startDateEastern"])
today_d = date.today()
is_past = schedule_df["game_date"] < today_d
is_now_or_future = schedule_df["game_date"] >= today_d

# ======================
# 2) Fetch latest rankings (Wikipedia strict order)
# ======================
print("📈 Fetching latest rankings (Wikipedia only; CFP→AP, current→prior)…")
rf = RankingsFetcher(YEAR)
rankings_df = rf.fetch_current_then_prior_cfp_ap()
print(f"   → rankings source: {rf.source} year={rf.source_year}")

if rankings_df is None or rankings_df.empty:
    raise RuntimeError("No Wikipedia rankings found in order: current CFP, current AP, prior CFP, prior AP.")

# Normalize rankings frame:
# - 'school' → canonical
# - ensure 'week' exists (int; default 0 if missing)
rankings_df = rankings_df.copy()
rankings_df = rankings_df.rename(columns={"school": "rank_school", "rank": "rank"}).copy()
rankings_df["rank_school_alias"] = rankings_df["rank_school"].apply(_canon)
if "week" not in rankings_df.columns:
    rankings_df["week"] = 0
rankings_df["week"] = pd.to_numeric(rankings_df["week"], errors="coerce").fillna(0).astype(int)

# Build a lookup for (rank_week, team) → rank
rank_map_df = rankings_df[["week", "rank_school_alias", "rank"]].copy()
rank_map_df["key"] = rank_map_df["week"].astype(str) + "||" + rank_map_df["rank_school_alias"]
rank_lookup: Dict[str, int] = dict(zip(rank_map_df["key"], rank_map_df["rank"]))

# ======================
# 3) Initialize/Preserve/Update ranks
# ======================
for col in ["homeTeamRank", "awayTeamRank"]:
    if col not in schedule_df.columns:
        schedule_df[col] = pd.NA

# 3a) Preserve past weeks from previous snapshot
if prev_df is not None and {"homeTeamRank", "awayTeamRank", "homeTeam", "awayTeam", "game_date"}.issubset(prev_df.columns):
    prev_past = prev_df[prev_df["game_date"] < today_d][["homeTeam", "awayTeam", "game_date", "homeTeamRank", "awayTeamRank"]]
    schedule_df = schedule_df.merge(
        prev_past,
        on=["homeTeam", "awayTeam", "game_date"],
        how="left",
        suffixes=("", "_prev")
    )
    if "homeTeamRank_prev" in schedule_df.columns:
        schedule_df["homeTeamRank"] = schedule_df["homeTeamRank"].mask(is_past, schedule_df["homeTeamRank_prev"])
    if "awayTeamRank_prev" in schedule_df.columns:
        schedule_df["awayTeamRank"] = schedule_df["awayTeamRank"].mask(is_past, schedule_df["awayTeamRank_prev"])
    schedule_df.drop(columns=["homeTeamRank_prev", "awayTeamRank_prev"], inplace=True, errors="ignore")

# 3b) Update current+future using week-offset rankings
# Build composite keys: "<rankings_week>||<canonical team>"
schedule_df["home_key"] = schedule_df["rankings_week"].astype(str) + "||" + schedule_df["homeTeam"]
schedule_df["away_key"] = schedule_df["rankings_week"].astype(str) + "||" + schedule_df["awayTeam"]

schedule_df["homeTeamRank_new"] = schedule_df["home_key"].map(rank_lookup)
schedule_df["awayTeamRank_new"] = schedule_df["away_key"].map(rank_lookup)

# Assign safely with mask (avoids shape/ndim AssertionError)
schedule_df["homeTeamRank"] = schedule_df["homeTeamRank"].mask(is_now_or_future, schedule_df["homeTeamRank_new"])
schedule_df["awayTeamRank"] = schedule_df["awayTeamRank"].mask(is_now_or_future, schedule_df["awayTeamRank_new"])

schedule_df.drop(columns=["homeTeamRank_new", "awayTeamRank_new", "home_key", "away_key"], inplace=True, errors="ignore")

# Boolean flag for ranked vs ranked
schedule_df["isRankedMatchup"] = schedule_df["homeTeamRank"].notna() & schedule_df["awayTeamRank"].notna()

# ======================
# 4) Stadium capacity merge
# ======================
print("🏟️  Merging stadium capacities…")
stadiums_path = _first_existing_path(STADIUM_CANDIDATES)
if stadiums_path and stadiums_path.exists():
    stad = pd.read_csv(stadiums_path).copy()
    if "stadium" not in stad.columns:
        raise KeyError(f"{stadiums_path.name} is missing required 'stadium' column")
    if "capacity" not in stad.columns:
        raise KeyError(f"{stadiums_path.name} is missing required 'capacity' column")

    if "school" in stad.columns:
        stad["school"] = stad["school"].apply(_canon)
    else:
        stad["school"] = pd.NA

    venue_col = _pick_venue_column(schedule_df)

    stad_lookup = dict(zip(stad["stadium"].map(_norm_key), stad["capacity"]))
    schedule_df["capacity"] = schedule_df[venue_col].map(lambda v: stad_lookup.get(_norm_key(v), pd.NA))

    need = schedule_df["capacity"].isna()
    if need.any():
        school_cap = (stad.dropna(subset=["school"])
                        .groupby("school", dropna=True, as_index=False)["capacity"].max()
                        .rename(columns={"capacity": "cap_by_home"}))
        schedule_df = schedule_df.merge(school_cap, left_on="homeTeam", right_on="school", how="left").drop(columns=["school"])
        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, "cap_by_home"]
        schedule_df = schedule_df.drop(columns=["cap_by_home"])

    need = schedule_df["capacity"].isna()
    if need.any():
        school_cap_away = (stad.dropna(subset=["school"])
                             .groupby("school", dropna=True, as_index=False)["capacity"].max()
                             .rename(columns={"capacity": "cap_by_away"}))
        schedule_df = schedule_df.merge(school_cap_away, left_on="awayTeam", right_on="school", how="left").drop(columns=["school"])
        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, "cap_by_away"]
        schedule_df = schedule_df.drop(columns=["cap_by_away"])
else:
    print("ℹ️ No stadiums file found; leaving capacity blank.")
    schedule_df["capacity"] = pd.NA

# ======================
# 5) Save outputs (atomic, Windows-safe)
# ======================
print("💾 Saving weekly schedule snapshot…")
_atomic_csv_write(schedule_df, WEEKLY_SCHEDULE_OUT)
print(f"✅ Weekly schedule saved: {WEEKLY_SCHEDULE_OUT}")

try:
    n = len(schedule_df)
    date_min = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").min()
    date_max = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").max()
    print(f"[weekly_update] rows={n}, date_range={date_min.date() if pd.notnull(date_min) else 'NA'}→{date_max.date() if pd.notnull(date_max) else 'NA'}")
except Exception as e:
    print(f"[weekly_update] post-write summary failed: {e}")
