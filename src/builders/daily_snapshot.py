# src/builders/daily_snapshot.py
#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd

# ---------------------------
# Paths & imports so this runs from anywhere
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))        # .../src
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))           # project root
for p in (SRC_DIR, PROJ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

TIMEZONE = ZoneInfo("America/New_York")
YEAR = int(os.getenv("SEASON_YEAR", datetime.now(TIMEZONE).year))

# ---------------------------
# Output config
# ---------------------------
DAILY_DIR = os.getenv("DAILY_DIR", os.path.join(PROJ_DIR, "data", "daily"))
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", os.path.join(DAILY_DIR, "price_snapshots.csv"))
os.makedirs(DAILY_DIR, exist_ok=True)

# ---------------------------
# Team list config
# ---------------------------
TEAMS_JSON_PATH = os.path.join(PROJ_DIR, "data", "tickpick_teams.txt")
TEAMS_LIMIT = int(os.getenv("TEAMS_LIMIT", "3"))  # default: first 3 for testing

# Collection windows (can be enforced by setting ALWAYS_RUN_DAILY=0)
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
ALWAYS_RUN_DAILY = os.getenv("ALWAYS_RUN_DAILY", "1") == "1"

# Politeness and caching for TickPickPricer
TP_POLITE_LOW = int(os.getenv("TP_POLITE_LOW", "1"))
TP_POLITE_HIGH = int(os.getenv("TP_POLITE_HIGH", "2"))
TP_RETRIES = int(os.getenv("TP_RETRIES", "3"))
TP_TIMEOUT = int(os.getenv("TP_TIMEOUT", "25"))
TP_USE_CACHE = os.getenv("TP_USE_CACHE", "0") == "1"
TP_CACHE_DIR = os.getenv("TP_CACHE_DIR", os.path.join(PROJ_DIR, "data", "_cache_tickpick_html"))
TP_VERBOSE = os.getenv("TP_VERBOSE", "0") == "1"

# Delete TickPickPricer exports after use
DELETE_TP_EXPORTS = os.getenv("DELETE_TP_EXPORTS", "1") == "1"

# Paths for enrichment sources
WEEKLY_SCHEDULE_PATH = os.path.join(PROJ_DIR, "data", "weekly", f"full_{YEAR}_schedule.csv")

# Rivalries path
RIVALRIES_PATH = os.path.join(PROJ_DIR, "data", "annual", f"rivalries_{YEAR}.csv")

# ---------------------------
# Import your working TickPickPricer
# ---------------------------
try:
    from scrapers.tickpick_pricer import TickPickPricer
except Exception as e:
    raise ImportError(f"Could not import scrapers.tickpick_pricer: {e!r}")

# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _load_teams_list() -> List[Dict[str, str]]:
    """Load team slugs/urls from data/tickpick_teams.txt (JSON array)."""
    if os.path.exists(TEAMS_JSON_PATH):
        try:
            with open(TEAMS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            out: List[Dict[str, str]] = []
            for item in data:
                if isinstance(item, dict):
                    slug = item.get("slug")
                    url = item.get("url")
                    if slug and url:
                        out.append({"slug": slug, "url": url})
            if out:
                return out[:TEAMS_LIMIT] if TEAMS_LIMIT > 0 else out
        except Exception:
            pass
    # Fallback 3
    return [
        {"slug": "boston-college-eagles-football",
         "url": "https://www.tickpick.com/ncaa-football/boston-college-eagles-football-tickets/"},
        {"slug": "california-golden-bears-football",
         "url": "https://www.tickpick.com/ncaa-football/california-golden-bears-football-tickets/"},
        {"slug": "clemson-tigers-football",
         "url": "https://www.tickpick.com/ncaa-football/clemson-tigers-football-tickets/"},
    ][:TEAMS_LIMIT or None]

def _titleize_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    tail = tail.replace("-tickets", "").replace("-", " ")
    return tail.title()

def _slug_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace("-tickets", "")

def _map_rows_to_snapshots(rows: List[Dict[str, Any]], now_et: datetime) -> pd.DataFrame:
    """Convert TickPickPricer rows to the base snapshot schema."""
    time_str = now_et.strftime("%H:%M")
    date_str = now_et.strftime("%Y-%m-%d")

    recs = []
    for r in rows:
        team_url = r.get("source_team_url")
        team_slug = _slug_from_url(team_url) if team_url else None
        team_name = _titleize_from_url(team_url) if team_url else None

        recs.append({
            "team_slug": team_slug,
            "team_name": team_name,
            "team_url": team_url,
            "event_id": r.get("event_id"),
            "title": r.get("title"),
            "home_team_guess": r.get("home_team_guess"),
            "away_team_guess": r.get("away_team_guess"),
            "date_local": r.get("date_local"),
            "time_local": r.get("time_local"),
            "offer_url": r.get("offer_url"),
            "lowest_price": r.get("low_price"),
            "highest_price": r.get("high_price"),
            "average_price": r.get("avg_price_from_page"),
            "listing_count": r.get("tickets_available_from_page"),
            "date_collected": date_str,
            "time_collected": time_str,
        })

    df = pd.DataFrame.from_records(recs) if recs else pd.DataFrame()
    for c in ("lowest_price", "highest_price", "average_price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "listing_count" in df.columns:
        df["listing_count"] = pd.to_numeric(df["listing_count"], errors="coerce", downcast="integer")
    return df

def _cleanup_pricer_exports(paths: Dict[str, str]) -> None:
    if not DELETE_TP_EXPORTS:
        return
    for k in ("json", "csv"):
        p = paths.get(k)
        if p and os.path.exists(p):
            try:
                os.remove(p)
                print(f"🧹 Removed temp {k.upper()} export: {p}")
            except Exception as e:
                print(f"⚠️ Could not delete {k} export {p}: {e}")

# ---------------------------
# Matching & enrichment helpers
# ---------------------------
_STOPWORDS = {"university","state","college","the","of","and","at","football","st","saint"}

def _normalize_team_name(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    x = s.lower()
    x = x.replace("&", " and ")
    for ch in "/.,-()[]{}'’":
        x = x.replace(ch, " ")
    toks = [t for t in x.split() if t and t not in _STOPWORDS]
    return " ".join(toks)

def _load_schedule() -> Optional[pd.DataFrame]:
    if not os.path.exists(WEEKLY_SCHEDULE_PATH):
        print(f"⚠️ Weekly schedule not found: {WEEKLY_SCHEDULE_PATH}")
        return None

    df = pd.read_csv(WEEKLY_SCHEDULE_PATH)

    # Helper: pick the first existing column name from candidates
    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Required fields (must exist in some form)
    required_map = {
        "startDateEastern": ["startDateEastern", "start_date_eastern", "startDate", "game_date", "date_local"],
        "homeTeam":         ["homeTeam", "home_team", "home"],
        "awayTeam":         ["awayTeam", "away_team", "away"],
    }

    rename_dict: Dict[str, str] = {}
    for canon, cands in required_map.items():
        src = pick(cands)
        if not src:
            print(f"⚠️ Weekly schedule missing required column for {canon} among {cands}")
            return None
        if src != canon:
            rename_dict[src] = canon

    # Optional fields (nice-to-have; fill with NA if absent)
    optional_map = {
        "stadium":          ["stadium", "venue", "venue_name", "stadium_name", "site"],
        "capacity":         ["capacity", "cap", "max_capacity"],
        "neutralSite":      ["neutralSite", "neutral_site", "neutral"],
        "conferenceGame":   ["conferenceGame", "conference_game", "isConferenceGame", "conference"],
        "isRivalry":        ["isRivalry", "is_rivalry", "rivalry"],
        "isRankedMatchup":  ["isRankedMatchup", "is_ranked_matchup", "ranked_matchup"],
        "homeTeamRank":     ["homeTeamRank", "home_rank", "homeRank", "home_team_rank"],
        "awayTeamRank":     ["awayTeamRank", "away_rank", "awayRank", "away_team_rank"],
    }

    for canon, cands in optional_map.items():
        src = pick(cands)
        if src and src != canon:
            rename_dict[src] = canon

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Ensure all optional columns exist (fill with NA if they weren't present)
    for canon in optional_map.keys():
        if canon not in df.columns:
            df[canon] = pd.NA

    # Standardize datetime
    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce")

    return df

def _prepare_schedule_for_join(sched: pd.DataFrame) -> pd.DataFrame:
    df = sched.copy()
    # date key
    df["date_key"] = pd.to_datetime(df["startDateEastern"], errors="coerce").dt.strftime("%Y-%m-%d")
    # normalized team keys
    df["home_key_sched"] = df["homeTeam"].map(_normalize_team_name)
    df["away_key_sched"] = df["awayTeam"].map(_normalize_team_name)
    return df

def _prepare_snapshots_for_join(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df["snap_idx"] = df.index
    df["date_key"] = pd.to_datetime(df["date_local"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home_key"] = df["home_team_guess"].map(_normalize_team_name)
    df["away_key"] = df["away_team_guess"].map(_normalize_team_name)
    return df

# ---- Rivalries loading & matching ----
def _choose_rivalry_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Pick two team-name columns from a rivalries CSV robustly."""
    cols_lower = {c.lower(): c for c in df.columns}

    preferred_pairs = [
        ("team1", "team2"),
        ("rival1", "rival2"),
        ("hometeam", "awayteam"),
        ("home", "away"),
        ("team_a", "team_b"),
        ("school1", "school2"),
    ]
    for a, b in preferred_pairs:
        if a in cols_lower and b in cols_lower:
            return (cols_lower[a], cols_lower[b])

    # Fallback: first two object/string-like columns
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(obj_cols) >= 2:
        return (obj_cols[0], obj_cols[1])

    return None

def _load_rivalries() -> Optional[Set[frozenset]]:
    """Return a set of frozenset({teamA_norm, teamB_norm}) from the rivalries CSV."""
    if not os.path.exists(RIVALRIES_PATH):
        print(f"ℹ️ Rivalries file not found: {RIVALRIES_PATH}")
        return None

    try:
        rdf = pd.read_csv(RIVALRIES_PATH)
    except Exception as e:
        print(f"⚠️ Could not read rivalries CSV: {e}")
        return None

    pair_cols = _choose_rivalry_columns(rdf)
    if not pair_cols:
        print("⚠️ Rivalries CSV has no recognizable team columns; skipping rivalry enrichment.")
        return None

    a_col, b_col = pair_cols
    pairs: Set[frozenset] = set()
    for _, row in rdf.iterrows():
        a = _normalize_team_name(row.get(a_col))
        b = _normalize_team_name(row.get(b_col))
        if a and b:
            pairs.add(frozenset({a, b}))
    if not pairs:
        print("ℹ️ Rivalries CSV produced no valid pairs after normalization.")
        return None
    return pairs

def _mark_rivalries(snap: pd.DataFrame, rivalry_pairs: Optional[Set[frozenset]]) -> pd.DataFrame:
    """Set isRivalry boolean by checking (home, away) pair against rivalry_pairs (order-agnostic, same-row constraint)."""
    if snap.empty:
        snap["isRivalry"] = False
        return snap
    if not rivalry_pairs:
        # ensure column exists; if schedule had a value already, keep it; else False
        if "isRivalry" not in snap.columns:
            snap["isRivalry"] = False
        return snap

    # Prefer schedule-derived teams; fallback to guesses
    home_series = snap["homeTeam"] if "homeTeam" in snap.columns else snap.get("home_team_guess")
    away_series = snap["awayTeam"] if "awayTeam" in snap.columns else snap.get("away_team_guess")

    # If still missing, create empty to avoid KeyErrors
    if home_series is None:
        home_series = pd.Series([""] * len(snap), index=snap.index)
    if away_series is None:
        away_series = pd.Series([""] * len(snap), index=snap.index)

    home_norm = home_series.map(_normalize_team_name)
    away_norm = away_series.map(_normalize_team_name)

    def is_rival(h: str, a: str) -> bool:
        if not h or not a:
            return False
        return frozenset({h, a}) in rivalry_pairs

    rivals = [is_rival(h, a) for h, a in zip(home_norm, away_norm)]
    snap["isRivalry"] = pd.Series(rivals, index=snap.index).astype(bool)
    return snap

def _enrich_with_schedule_and_stadiums(snap: pd.DataFrame) -> pd.DataFrame:
    """Join snapshots with the already-enriched weekly schedule to bring in stadium, capacity, ranks, etc."""
    if snap.empty:
        return snap

    # Compute days_until_game regardless of joins
    snap["days_until_game"] = (
        pd.to_datetime(snap["date_local"], errors="coerce") -
        pd.to_datetime(snap["date_collected"], errors="coerce")
    ).dt.days

    sched = _load_schedule()
    if sched is None or sched.empty:
        snap["stadium"] = pd.NA
        snap["capacity"] = pd.NA
        snap["neutralSite"] = pd.NA
        snap["conferenceGame"] = pd.NA
        # rivalry will be set below from file
        snap["isRankedMatchup"] = pd.NA
        snap["homeTeamRank"] = pd.NA
        snap["awayTeamRank"] = pd.NA
    else:
        sched_pre = _prepare_schedule_for_join(sched)
        snap_pre  = _prepare_snapshots_for_join(snap)

        # ----- direct (home, away) on same date -----
        m1 = snap_pre.merge(
            sched_pre,
            how="left",
            on="date_key",
            suffixes=("", "_sched"),
        )
        m1 = m1[(m1["home_key"] == m1["home_key_sched"]) & (m1["away_key"] == m1["away_key_sched"])].copy()
        m1 = m1.sort_values(["snap_idx"]).drop_duplicates(subset=["snap_idx"], keep="first")

        # ----- flipped (away, home) on same date -----
        sched_flip = sched_pre.rename(columns={
            "home_key_sched": "away_key_sched",
            "away_key_sched": "home_key_sched",
            "homeTeam": "awayTeam",
            "awayTeam": "homeTeam",
        })
        m2 = snap_pre.merge(
            sched_flip,
            how="left",
            on="date_key",
            suffixes=("", "_sched"),
        )
        m2 = m2[(m2["home_key"] == m2["home_key_sched"]) & (m2["away_key"] == m2["away_key_sched"])].copy()
        m2 = m2.sort_values(["snap_idx"]).drop_duplicates(subset=["snap_idx"], keep="first")

        # ----- choose direct match when available; else flipped -----
        matched = pd.merge(
            m1[["snap_idx","homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
                 "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank"]],
            m2[["snap_idx","homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
                 "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank"]],
            on="snap_idx",
            how="outer",
            suffixes=("_dir","_flip"),
        )

        def pick_pair(row, a, b):
            return row[a] if pd.notna(row[a]) else row[b]

        out = pd.DataFrame({"snap_idx": matched["snap_idx"]})
        for col in ["homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
                    "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank"]:
            out[col] = matched.apply(lambda r, c=col: pick_pair(r, f"{c}_dir", f"{c}_flip"), axis=1)

        snap = snap.merge(out, left_index=True, right_on="snap_idx", how="left").drop(columns=["snap_idx"])

    # Rivalries from rivalries_{YEAR}.csv override/ensure boolean
    rivalry_pairs = _load_rivalries()
    snap = _mark_rivalries(snap, rivalry_pairs)

    # Final column ordering
    col_order_front = [
        "team_slug","team_name","team_url",
        "event_id","title","home_team_guess","away_team_guess",
        "homeTeam","awayTeam",
        "date_local","time_local","offer_url",
        "lowest_price","highest_price","average_price","listing_count",
        "days_until_game","stadium","capacity",
        "neutralSite","conferenceGame","isRivalry","isRankedMatchup",
        "homeTeamRank","awayTeamRank",
        "date_collected","time_collected",
    ]
    final_cols = [c for c in col_order_front if c in snap.columns] + \
                 [c for c in snap.columns if c not in col_order_front]
    return snap[final_cols]

# ---------------------------
# Main snapshot logic (TEAM URLs -> TickPickPricer.run() -> export JSON -> map -> enrich)
# ---------------------------
def log_price_snapshot():
    now_et = datetime.now(TIMEZONE)
    time_str = now_et.strftime("%H:%M")

    if not ALWAYS_RUN_DAILY and time_str not in COLLECTION_TIMES:
        print(f"⏭️  Not a collection window ({time_str}).")
        return

    teams = _load_teams_list()
    if not teams:
        print("❌ No team URLs available.")
        return

    team_urls = [t["url"] for t in teams]

    pricer = TickPickPricer(
        team_urls=team_urls,
        output_dir=DAILY_DIR,
        polite_delay_range=(TP_POLITE_LOW, TP_POLITE_HIGH),
        retries=TP_RETRIES,
        timeout=TP_TIMEOUT,
        use_cache=TP_USE_CACHE,
        cache_dir=TP_CACHE_DIR,
        verbose=TP_VERBOSE,
    )

    export_paths = pricer.run()
    json_path = export_paths.get("json")
    if not json_path or not os.path.exists(json_path):
        print("❌ TickPickPricer did not produce a JSON export.")
        _cleanup_pricer_exports(export_paths)
        return

    # Load rows from TickPickPricer's JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    finally:
        _cleanup_pricer_exports(export_paths)

    if not isinstance(rows, list) or not rows:
        print("⚠️ No rows found in TickPickPricer JSON.")
        return

    snap = _map_rows_to_snapshots(rows, now_et)
    if snap.empty:
        print("⚠️ No snapshots collected after mapping.")
        return

    # Enrich with weekly schedule (includes capacity), rivalry flag, and days_until_game
    snap = _enrich_with_schedule_and_stadiums(snap)

    # Append/dedupe within same minute per (offer_url or event_id)
    _ensure_dir(SNAPSHOT_PATH)
    key_cols: List[str] = []
    if "offer_url" in snap.columns:
        key_cols.append("offer_url")
    if "event_id" in snap.columns:
        key_cols.append("event_id")
    key_cols.extend(["date_collected", "time_collected"])

    if os.path.exists(SNAPSHOT_PATH):
        existing = pd.read_csv(SNAPSHOT_PATH)
        combined = pd.concat([existing, snap], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in key_cols if c in combined.columns],
            keep="last",
        )
        combined.to_csv(SNAPSHOT_PATH, index=False)
        print(f"✅ Snapshot appended ({len(snap)} rows). Total now: {len(combined)}")
    else:
        snap.to_csv(SNAPSHOT_PATH, index=False)
        print(f"✅ Snapshot saved ({len(snap)} rows) to {SNAPSHOT_PATH}")

if __name__ == "__main__":
    log_price_snapshot()
