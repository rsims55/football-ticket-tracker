# src/preparation/weekly_update.py
#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import json
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import re

# --- Robust import path so this runs from anywhere ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fetchers.schedule_fetcher import ScheduleFetcher
from fetchers.rankings_fetcher import RankingsFetcher

# ======================
# Constants & Paths
# ======================
YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))

DATA_DIR = "data"
WEEKLY_DIR = os.path.join(DATA_DIR, "weekly")
ANNUAL_DIR = os.path.join(DATA_DIR, "annual")
PERMANENT_DIR = os.path.join(DATA_DIR, "permanent")
ALIAS_JSON = os.path.join(PERMANENT_DIR, "team_aliases.json")

os.makedirs(WEEKLY_DIR, exist_ok=True)

WEEKLY_SCHEDULE_OUT = os.path.join(WEEKLY_DIR, f"full_{YEAR}_schedule.csv")

# Candidate stadium files (first existing wins)
STADIUM_CANDIDATES = [
    os.path.join(ANNUAL_DIR, f"stadiums_{YEAR}.csv"),
    os.path.join(DATA_DIR,     f"stadiums_{YEAR}.csv"),
]

# ======================
# Load alias map (required)
# ======================
if not os.path.exists(ALIAS_JSON):
    raise FileNotFoundError(
        f"Alias map not found at {ALIAS_JSON}. "
        "Please create data/permanent/team_aliases.json (a dict of {'Incoming Name': 'Canonical Name'})."
    )

with open(ALIAS_JSON, "r", encoding="utf-8") as f:
    alias_map: Dict[str, str] = json.load(f)

def get_alias(name: str) -> str:
    # Exact match aliasing; if no mapping, keep original
    return alias_map.get(name, name)

def _canon(val):
    # keep NaN as-is; otherwise alias if present, else original
    if pd.isna(val):
        return val
    return get_alias(str(val).strip())

# ----------------------
# Helpers
# ----------------------
def _pick_venue_column(df: pd.DataFrame) -> str:
    for c in ["venue", "stadium", "site", "venue_name"]:
        if c in df.columns:
            return c
    # create a blank column if none exists (so code downstream still works)
    df["venue"] = pd.NA
    return "venue"

def _first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _norm_key(s) -> str:
    """Lower + strip string for exact-but-case-insensitive matching."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return s.strip().lower()

# ======================
# 1) Fetch schedule
# ======================
print("📅 Fetching schedule...")
schedule_df = ScheduleFetcher(YEAR).fetch().copy()

if "startDateEastern" not in schedule_df.columns:
    raise KeyError("schedule_df is missing 'startDateEastern'")

# Canonicalize team names
schedule_df["homeTeam"] = schedule_df["homeTeam"].apply(_canon)
schedule_df["awayTeam"] = schedule_df["awayTeam"].apply(_canon)

# ======================
# 2) Fetch rankings and merge (using canonicalized names on BOTH sides)
# ======================
print("📈 Fetching rankings...")
rankings_df = RankingsFetcher(YEAR).fetch_and_load()

if rankings_df is not None and not rankings_df.empty:
    rankings_df = rankings_df.rename(columns={"school": "rank_school", "rank": "rank"}).copy()
    alias_df = rankings_df.assign(rank_school_alias=rankings_df["rank_school"].apply(_canon))

    # Merge for home
    schedule_df = (
        schedule_df.merge(
            alias_df[["rank_school_alias", "rank"]],
            left_on="homeTeam",
            right_on="rank_school_alias",
            how="left",
        )
        .rename(columns={"rank": "homeTeamRank"})
        .drop(columns=["rank_school_alias"])
    )

    # Merge for away
    schedule_df = (
        schedule_df.merge(
            alias_df[["rank_school_alias", "rank"]],
            left_on="awayTeam",
            right_on="rank_school_alias",
            how="left",
        )
        .rename(columns={"rank": "awayTeamRank"})
        .drop(columns=["rank_school_alias"])
    )

    schedule_df["isRankedMatchup"] = schedule_df["homeTeamRank"].notna() & schedule_df["awayTeamRank"].notna()
else:
    schedule_df["homeTeamRank"] = pd.NA
    schedule_df["awayTeamRank"] = pd.NA
    schedule_df["isRankedMatchup"] = False

# ======================
# 3) Stadium capacity merge (exact venue match, then school==homeTeam, then school==awayTeam)
# ======================
print("🏟️  Merging stadium capacities...")

stadiums_path = _first_existing_path(STADIUM_CANDIDATES)
if stadiums_path and os.path.exists(stadiums_path):
    stad = pd.read_csv(stadiums_path).copy()

    # Required columns
    if "stadium" not in stad.columns:
        raise KeyError(f"{os.path.basename(stadiums_path)} is missing required 'stadium' column")
    if "capacity" not in stad.columns:
        raise KeyError(f"{os.path.basename(stadiums_path)} is missing required 'capacity' column")

    # Canonicalize school for team-based matching
    if "school" in stad.columns:
        stad["school"] = stad["school"].apply(_canon)
    else:
        stad["school"] = pd.NA

    # Determine venue column on the schedule
    venue_col = _pick_venue_column(schedule_df)

    # --- 3.1 Direct exact match: venue == stadium (case/whitespace-insensitive)
    # Build a lookup from normalized stadium name -> capacity
    stad_lookup = dict(zip(stad["stadium"].map(_norm_key), stad["capacity"]))
    schedule_df["capacity"] = schedule_df[venue_col].map(lambda v: stad_lookup.get(_norm_key(v), pd.NA))

    # --- 3.2 If still missing, match school == homeTeam
    need = schedule_df["capacity"].isna()

    if need.any():
        # Collapse any duplicates in stadiums to one capacity per school (take max non-null just in case)
        school_cap = (
            stad.dropna(subset=["school"])
                .groupby("school", dropna=True, as_index=False)["capacity"]
                .max()
        )
        school_cap = school_cap.rename(columns={"capacity": "cap_by_home"})
        schedule_df = schedule_df.merge(
            school_cap, left_on="homeTeam", right_on="school", how="left"
        ).drop(columns=["school"])

        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, "cap_by_home"]
        schedule_df = schedule_df.drop(columns=["cap_by_home"])

    # --- 3.3 If still missing, match school == awayTeam
    need = schedule_df["capacity"].isna()
    if need.any():
        school_cap_away = (
            stad.dropna(subset=["school"])
                .groupby("school", dropna=True, as_index=False)["capacity"]
                .max()
                .rename(columns={"capacity": "cap_by_away"})
        )
        schedule_df = schedule_df.merge(
            school_cap_away, left_on="awayTeam", right_on="school", how="left"
        ).drop(columns=["school"])

        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, "cap_by_away"]
        schedule_df = schedule_df.drop(columns=["cap_by_away"])

else:
    print("ℹ️ No stadiums file found; leaving capacity blank.")
    schedule_df["capacity"] = pd.NA

# ======================
# 4) Save outputs
# ======================
print("💾 Saving weekly schedule snapshot...")
schedule_df.to_csv(WEEKLY_SCHEDULE_OUT, index=False)
print(f"✅ Weekly schedule saved: {WEEKLY_SCHEDULE_OUT}")
