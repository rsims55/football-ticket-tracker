# src/builders/daily_snapshot.py
#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import re 
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

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
# Output config (FORCE repo paths by default)
# ---------------------------
# If REPO_DATA_LOCK=1 (default), ignore env overrides and force repo paths.
REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"

# Proposed (from env), but may be ignored if lock is ON
_env_daily = os.getenv("DAILY_DIR")
_env_snap  = os.getenv("SNAPSHOT_PATH")

if REPO_DATA_LOCK:
    DAILY_DIR = os.path.join(PROJ_DIR, "data", "daily")
    SNAPSHOT_PATH = os.path.join(DAILY_DIR, "price_snapshots.csv")
else:
    # Respect env or fall back to repo
    DAILY_DIR = _env_daily or os.path.join(PROJ_DIR, "data", "daily")
    SNAPSHOT_PATH = _env_snap or os.path.join(DAILY_DIR, "price_snapshots.csv")

# Hard safety rails: if someone did set envs and lock is ON, warn loudly.
if REPO_DATA_LOCK and (_env_daily or _env_snap):
    print("⚠️  REPO_DATA_LOCK=1 -> ignoring DAILY_DIR/SNAPSHOT_PATH env and writing to repo paths only.")

# Final guard: even with lock OFF, refuse paths that escape the repo unless explicitly allowed.
# (You can allow escapes by setting REPO_ALLOW_NON_REPO_OUT=1 for special cases.)
ALLOW_ESCAPE = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"
def _under_repo(p: str) -> bool:
    try:
        return Path(p).resolve().is_relative_to(Path(PROJ_DIR).resolve())
    except AttributeError:
        rp, rroot = Path(p).resolve(), Path(PROJ_DIR).resolve()
        return str(rp).startswith(str(rroot))

if not _under_repo(DAILY_DIR) or not _under_repo(SNAPSHOT_PATH):
    if not ALLOW_ESCAPE:
        print(f"🚫 Output resolved outside repo:\n  DAILY_DIR={DAILY_DIR}\n  SNAPSHOT={SNAPSHOT_PATH}\n  PROJ_DIR={PROJ_DIR}")
        print("    Forcing repo-relative paths. Set REPO_ALLOW_NON_REPO_OUT=1 to permit.")
        DAILY_DIR = os.path.join(PROJ_DIR, "data", "daily")
        SNAPSHOT_PATH = os.path.join(DAILY_DIR, "price_snapshots.csv")

os.makedirs(DAILY_DIR, exist_ok=True)

# Combined export behavior
KEEP_COMBINED_EXPORTS = os.getenv("KEEP_COMBINED_EXPORTS", "0") == "1"

# ---------------------------
# Team list config
# ---------------------------
TEAMS_JSON_PATH = os.getenv(
    "TEAMS_JSON_PATH",
    os.path.join(PROJ_DIR, "data", "permanent", "tickpick_teams.txt")
)
# 0 (or negative) means "all teams"
TEAMS_LIMIT = int(os.getenv("TEAMS_LIMIT", "0"))
# TEAMS_LIMIT = 3

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
def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically to avoid partial rewrites on Windows/OneDrive."""
    tmp = f"{path}.__tmp__"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _read_json_array(path: str) -> Optional[List[dict]]:
    """Read either a JSON array or JSON-Lines file into a list of dicts."""
    if not os.path.exists(path):
        print(f"⚠️  Teams file does not exist: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            print(f"⚠️  Teams file is empty: {path}")
            return None
        if txt.lstrip().startswith("["):
            try:
                data = json.loads(txt)
                if isinstance(data, list):
                    return data
                print("⚠️  Expected a JSON array at top-level.")
                return None
            except Exception as e:
                print(f"⚠️  JSON array parse error: {e}")
                return None
        items: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                    else:
                        print(f"⚠️  Line {ln} is not a JSON object; skipping.")
                except Exception as e:
                    print(f"⚠️  JSON-Lines parse error at line {ln}: {e}")
                    return None
        return items if items else None
    except Exception as e:
        print(f"⚠️  Failed to read {path}: {e}")
        return None

def _load_teams_list() -> List[Dict[str, str]]:
    """
    Load team slugs/urls from TEAMS_JSON_PATH (.txt or .json).
    Accepts JSON array or JSON-Lines.
    Respects TEAMS_LIMIT (0 => all).
    """
    data = _read_json_array(TEAMS_JSON_PATH)

    if data is None and TEAMS_JSON_PATH.endswith(".txt"):
        alt = TEAMS_JSON_PATH[:-4] + ".json"
        if os.path.exists(alt):
            print(f"ℹ️  Trying alternate teams file: {alt}")
            data = _read_json_array(alt)

    if data:
        out: List[Dict[str, str]] = []
        bad_count = 0
        for item in data:
            if isinstance(item, dict):
                url = item.get("url")
                slug = item.get("slug")
                if not slug and isinstance(url, str):
                    tail = url.rstrip("/").split("/")[-1]
                    slug = tail.replace("-tickets", "")
                if slug and url:
                    out.append({"slug": slug, "url": url})
                else:
                    bad_count += 1
            else:
                bad_count += 1
        if bad_count:
            print(f"ℹ️  Skipped {bad_count} invalid rows in teams file.")
        if out:
            if TEAMS_LIMIT and TEAMS_LIMIT > 0:
                print(f"📄 Loaded {len(out)} teams from file; using first {TEAMS_LIMIT}.")
                return out[:TEAMS_LIMIT]
            print(f"📄 Loaded {len(out)} teams from file; using ALL.")
            return out
        else:
            print("⚠️  Teams file contained no valid {slug,url} entries.")

    fallback = [
        {"slug": "boston-college-eagles-football",
         "url": "https://www.tickpick.com/ncaa-football/boston-college-eagles-football-tickets/"},
        {"slug": "california-golden-bears-football",
         "url": "https://www.tickpick.com/ncaa-football/california-golden-bears-football-tickets/"},
        {"slug": "clemson-tigers-football",
         "url": "https://www.tickpick.com/ncaa-football/clemson-tigers-football-tickets/"},
    ]
    print("🚨 Falling back to built-in 3-team list (teams file missing or invalid JSON).")
    if TEAMS_LIMIT and TEAMS_LIMIT > 0:
        return fallback[:TEAMS_LIMIT]
    return fallback

def _titleize_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    tail = tail.replace("-tickets", "").replace("-", " ")
    return tail.title()

def _slug_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace("-tickets", "")

def _map_rows_to_snapshots(rows: List[Dict[str, Any]], now_et: datetime) -> pd.DataFrame:
    """Convert TickPickPricer rows to the base snapshot schema (no enrichment)."""
    time_str = now_et.strftime("%H:%M:%S")  # seconds precision to avoid same-minute overwrite
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
            # Keep guesses TEMPORARILY for matching; we drop them before saving final
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

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

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

    optional_map = {
        "stadium":          ["stadium", "venue", "venue_name", "stadium_name", "site"],
        "capacity":         ["capacity", "cap", "max_capacity"],
        "neutralSite":      ["neutralSite", "neutral_site", "neutral"],
        "conferenceGame":   ["conferenceGame", "conference_game", "isConferenceGame", "conference"],
        "isRivalry":        ["isRivalry", "is_rivalry", "rivalry"],
        "isRankedMatchup":  ["isRankedMatchup", "is_ranked_matchup", "ranked_matchup"],
        "homeTeamRank":     ["homeTeamRank", "home_rank", "homeRank", "home_team_rank"],
        "awayTeamRank":     ["awayTeamRank", "away_rank", "awayRank", "away_team_rank"],
        "homeConference":   ["homeConference", "home_conference", "home_conf", "home_conf_name"],
        "awayConference":   ["awayConference", "away_conference", "away_conf", "away_conf_name"],
        "week":             ["week", "game_week", "week_num", "week_number"],
    }

    for canon, cands in optional_map.items():
        src = pick(cands)
        if src and src != canon:
            rename_dict[src] = canon

    if rename_dict:
        df = df.rename(columns=rename_dict)

    for canon in optional_map.keys():
        if canon not in df.columns:
            df[canon] = pd.NA

    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce")
    return df

def _prepare_schedule_for_join(sched: pd.DataFrame) -> pd.DataFrame:
    df = sched.copy()
    df["date_key"] = pd.to_datetime(df["startDateEastern"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home_key_sched"] = df["homeTeam"].map(_normalize_team_name)
    df["away_key_sched"] = df["awayTeam"].map(_normalize_team_name)
    return df

def _prepare_snapshots_for_join(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df["snap_idx"] = df.index
    df["date_key"] = pd.to_datetime(df["date_local"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Prefer fixed homeTeam/awayTeam; fallback to guess columns if needed
    def _pick_norm(src_primary: str, src_fallback: str) -> pd.Series:
        primary = df[src_primary] if src_primary in df.columns else pd.Series([""] * len(df), index=df.index)
        primary_norm = primary.fillna("").map(_normalize_team_name)
        if src_fallback in df.columns:
            fallback_norm = df[src_fallback].fillna("").map(_normalize_team_name)
            return primary_norm.where(primary_norm.astype(bool), fallback_norm)
        return primary_norm

    df["home_key"] = _pick_norm("homeTeam", "home_team_guess")
    df["away_key"] = _pick_norm("awayTeam", "away_team_guess")
    return df

def _choose_rivalry_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
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
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(obj_cols) >= 2:
        return (obj_cols[0], obj_cols[1])
    return None

def _load_rivalries() -> Optional[Set[frozenset]]:
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
    """Set isRivalry boolean by checking (home, away) pair against rivalry_pairs (order-agnostic)."""
    if snap.empty:
        snap["isRivalry"] = False
        return snap

    if not rivalry_pairs:
        if "isRivalry" not in snap.columns:
            snap["isRivalry"] = False
        return snap

    if "homeTeam" in snap.columns:
        home_series = snap["homeTeam"]
    else:
        home_series = pd.Series([""] * len(snap), index=snap.index)

    if "awayTeam" in snap.columns:
        away_series = snap["awayTeam"]
    else:
        away_series = pd.Series([""] * len(snap), index=snap.index)

    home_norm = home_series.fillna("").map(_normalize_team_name)
    away_norm = away_series.fillna("").map(_normalize_team_name)

    def is_rival(h: str, a: str) -> bool:
        return bool(h and a and (frozenset({h, a}) in rivalry_pairs))

    snap["isRivalry"] = [is_rival(h, a) for h, a in zip(home_norm, away_norm)]
    snap["isRivalry"] = snap["isRivalry"].astype(bool)
    return snap

# ---------------------------
# Title cleanup + alias + matchup fixes (no new columns)
# ---------------------------

ALIASES_PATH = os.path.join(PROJ_DIR, "data", "permanent", "team_aliases.json")

def _load_team_aliases() -> Dict[str, str]:
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️ team_aliases.json is not an object: {ALIASES_PATH}")
            return {}
        return {str(k).strip().lower(): str(v).strip()
                for k, v in data.items()
                if isinstance(k, str) and isinstance(v, str)}
    except FileNotFoundError:
        print(f"ℹ️ No team_aliases.json found at {ALIASES_PATH}")
        return {}
    except Exception as e:
        print(f"⚠️ Failed to read team_aliases.json: {e}")
        return {}

def _clean_event_title(title: Optional[str]) -> Optional[str]:
    if not isinstance(title, str):
        return title
    t = title.strip()
    if ":" in t:
        right = t.split(":", 1)[1].strip()
        if right:
            return right
    return t

# Match " vs " or " vs. " with ANY casing, flexible spacing, and NBSPs
_VS_RE = re.compile(r'\s+vs\.?\s+', flags=re.IGNORECASE)

def _split_matchup(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Prefer 'X vs Y' => (home=X, away=Y).
    Robust to:
      - 'Vs' / 'VS' / 'vs.'
      - non-breaking spaces
      - extra punctuation/spaces around team names
    """
    if not isinstance(text, str):
        return (None, None)
    # Normalize non-breaking spaces and stray dashes
    t = text.replace('\u00A0', ' ').strip()
    parts = _VS_RE.split(t, maxsplit=1)
    if len(parts) == 2:
        left = parts[0].strip(" \t-–—")
        right = parts[1].strip(" \t-–—")
        if left and right:
            return (left, right)
    return (None, None)

def _canonicalize_if_aliased(name: Optional[str], aliases: Dict[str, str]) -> Optional[str]:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    return aliases.get(key, name.strip())

def _apply_title_and_alias_fixes(snap: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean 'title' (drop prefix before ':').
    - If homeTeam/awayTeam exist, fill only their blanks from 'X vs Y' and apply aliases.
    - Else, if guess columns exist, fill only their blanks and apply aliases.
    - NEVER create new columns.
    """
    if snap.empty:
        return snap
    df = snap.copy()
    aliases = _load_team_aliases()

    if "title" in df.columns:
        df["title"] = df["title"].map(_clean_event_title)

    def _needs_fill(x) -> bool:
        return (not isinstance(x, str)) or (x.strip() == "")

    # Helper to produce inferred values from title without writing anything yet
    def _infer_pair_from_title(row):
        ih, ia = _split_matchup(row.get("title") or "")
        return ih, ia

    if ("homeTeam" in df.columns) or ("awayTeam" in df.columns):
        # Work on homeTeam
        if "homeTeam" in df.columns:
            inferred_home = df.apply(lambda r: _infer_pair_from_title(r)[0], axis=1)
            mask = df["homeTeam"].isna() | (df["homeTeam"].astype(str).str.strip() == "")
            df.loc[mask, "homeTeam"] = inferred_home[mask]
            df["homeTeam"] = df["homeTeam"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

        # Work on awayTeam
        if "awayTeam" in df.columns:
            inferred_away = df.apply(lambda r: _infer_pair_from_title(r)[1], axis=1)
            mask = df["awayTeam"].isna() | (df["awayTeam"].astype(str).str.strip() == "")
            df.loc[mask, "awayTeam"] = inferred_away[mask]
            df["awayTeam"] = df["awayTeam"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

    elif ("home_team_guess" in df.columns) or ("away_team_guess" in df.columns):
        # Fall back to fixing the guess columns (common pre-enrichment)
        if "home_team_guess" in df.columns:
            inferred_home = df.apply(lambda r: _infer_pair_from_title(r)[0], axis=1)
            mask = df["home_team_guess"].isna() | (df["home_team_guess"].astype(str).str.strip() == "")
            df.loc[mask, "home_team_guess"] = inferred_home[mask]
            df["home_team_guess"] = df["home_team_guess"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

        if "away_team_guess" in df.columns:
            inferred_away = df.apply(lambda r: _infer_pair_from_title(r)[1], axis=1)
            mask = df["away_team_guess"].isna() | (df["away_team_guess"].astype(str).str.strip() == "")
            df.loc[mask, "away_team_guess"] = inferred_away[mask]
            df["away_team_guess"] = df["away_team_guess"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

    return df


def _enrich_with_schedule_and_stadiums(snap: pd.DataFrame) -> pd.DataFrame:
    if snap.empty:
        return snap
    snap["days_until_game"] = (
        pd.to_datetime(snap["date_local"], errors="coerce") -
        pd.to_datetime(snap["date_collected"], errors="coerce")
    ).dt.days
    sched = _load_schedule()
    if sched is None or sched.empty:
        for col in [
            "stadium","capacity","neutralSite","conferenceGame",
            "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
            "homeConference","awayConference","homeTeam","awayTeam","week"
        ]:
            if col not in snap.columns:
                snap[col] = pd.NA
    else:
        sched_pre = _prepare_schedule_for_join(sched)
        snap_pre  = _prepare_snapshots_for_join(snap)
        m1 = snap_pre.merge(sched_pre, how="left", on="date_key", suffixes=("", "_sched"))
        m1 = m1[(m1["home_key"] == m1["home_key_sched"]) & (m1["away_key"] == m1["away_key_sched"])].copy()
        m1 = m1.sort_values(["snap_idx"]).drop_duplicates(subset=["snap_idx"], keep="first")
        sched_flip = sched_pre.rename(columns={
            "home_key_sched": "away_key_sched",
            "away_key_sched": "home_key_sched",
            "homeTeam": "awayTeam",
            "awayTeam": "homeTeam",
            "homeConference": "awayConference",
            "awayConference": "homeConference",
        })
        m2 = snap_pre.merge(sched_flip, how="left", on="date_key", suffixes=("", "_sched"))
        m2 = m2[(m2["home_key"] == m2["home_key_sched"]) & (m2["away_key"] == m2["away_key_sched"])].copy()
        m2 = m2.sort_values(["snap_idx"]).drop_duplicates(subset=["snap_idx"], keep="first")
        cols_to_carry = [
            "homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
            "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
            "homeConference","awayConference","week"
        ]
        matched = pd.merge(
            m1[["snap_idx"] + [c for c in cols_to_carry if c in m1.columns]],
            m2[["snap_idx"] + [c for c in cols_to_carry if c in m2.columns]],
            on="snap_idx", how="outer", suffixes=("_dir","_flip"),
        )
        def pick_pair(row, a, b):
            return row[a] if pd.notna(row.get(a)) else row.get(b)
        out = pd.DataFrame({"snap_idx": matched["snap_idx"]})
        for col in cols_to_carry:
            a, b = f"{col}_dir", f"{col}_flip"
            out[col] = matched.apply(lambda r, a=a, b=b: pick_pair(r, a, b), axis=1)
        snap = snap.merge(out, left_index=True, right_on="snap_idx", how="left").drop(columns=["snap_idx"])

        # --- Backfill home/away strictly from the title (no new columns) ---
        _VS_RE = re.compile(r'\s+vs\.?\s+', flags=re.IGNORECASE)

        def _clean_title_local(t):
            if not isinstance(t, str):
                return t
            t = t.replace('\u00A0', ' ').strip()  # normalize NBSPs
            if ":" in t:
                right = t.split(":", 1)[1].strip()  # keep only after colon
                if right:
                    t = right
            return t

        def _part_from_title(t, idx):
            t = _clean_title_local(t)
            if not isinstance(t, str):
                return None
            parts = _VS_RE.split(t, maxsplit=1)  # split on vs / Vs. / VS. with flexible spacing
            if len(parts) == 2:
                side = parts[idx].strip(" \t-–—")
                return side if side else None
            return None

        for col, idx in (("homeTeam", 0), ("awayTeam", 1)):
            if col in snap.columns and "title" in snap.columns:
                mask = snap[col].isna() | (snap[col].astype(str).str.strip() == "")
                parsed = snap["title"].apply(lambda s: _part_from_title(s, idx))
                fill = mask & parsed.notna() & (parsed.astype(str).str.strip() != "")
                snap.loc[fill, col] = parsed[fill]

    # --- Second-chance enrichment using newly backfilled home/away + date ---
    cols_to_carry = [
        "homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
        "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
        "homeConference","awayConference","week"
    ]

    # Build temp keys without leaving extra columns behind
    tmp = snap.copy()
    tmp["snap_idx2"] = tmp.index
    tmp["date_key"]   = pd.to_datetime(tmp["date_local"], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp["home_key_new"] = tmp.get("homeTeam", pd.Series([""]*len(tmp), index=tmp.index)).fillna("").map(_normalize_team_name)
    tmp["away_key_new"] = tmp.get("awayTeam", pd.Series([""]*len(tmp), index=tmp.index)).fillna("").map(_normalize_team_name)

    sched_pre = _prepare_schedule_for_join(sched)

    # direct (home vs away)
    m1b = tmp.merge(sched_pre, how="left", on="date_key", suffixes=("", "_sched"))
    m1b = m1b[(m1b["home_key_new"] == m1b["home_key_sched"]) & (m1b["away_key_new"] == m1b["away_key_sched"])].copy()
    m1b = m1b.sort_values("snap_idx2").drop_duplicates(subset=["snap_idx2"], keep="first")

    # flipped (away vs home)
    sched_flip = sched_pre.rename(columns={
        "home_key_sched": "away_key_sched",
        "away_key_sched": "home_key_sched",
        "homeTeam": "awayTeam",
        "awayTeam": "homeTeam",
        "homeConference": "awayConference",
        "awayConference": "homeConference",
        "homeTeamRank": "awayTeamRank",
        "awayTeamRank": "homeTeamRank",
    })
    m2b = tmp.merge(sched_flip, how="left", on="date_key", suffixes=("", "_sched"))
    m2b = m2b[(m2b["home_key_new"] == m2b["home_key_sched"]) & (m2b["away_key_new"] == m2b["away_key_sched"])].copy()
    m2b = m2b.sort_values("snap_idx2").drop_duplicates(subset=["snap_idx2"], keep="first")

    # build overlay preferring direct over flipped
    overlay = pd.merge(
        m1b[["snap_idx2"] + [c for c in cols_to_carry if c in m1b.columns]],
        m2b[["snap_idx2"] + [c for c in cols_to_carry if c in m2b.columns]],
        on="snap_idx2", how="outer", suffixes=("_dir","_flip")
    )

    def _pick(row, col):
        a = row.get(f"{col}_dir")
        b = row.get(f"{col}_flip")
        return a if pd.notna(a) else b

    over = pd.DataFrame({"snap_idx2": overlay["snap_idx2"]})
    for col in cols_to_carry:
        if f"{col}_dir" in overlay.columns or f"{col}_flip" in overlay.columns:
            over[col] = overlay.apply(lambda r, c=col: _pick(r, c), axis=1)
    over = over.set_index("snap_idx2")

    # Fill ONLY where missing/blank (never create new columns)
    for col in cols_to_carry:
        if col in snap.columns and col in over.columns:
            newvals = over[col].reindex(snap.index)
            if pd.api.types.is_numeric_dtype(snap[col]):
                mask = snap[col].isna() & newvals.notna()
            else:
                mask = (snap[col].isna() | (snap[col].astype(str).str.strip() == "")) & newvals.notna()
            snap.loc[mask, col] = newvals[mask]

    rivalry_pairs = _load_rivalries()
    snap = _mark_rivalries(snap, rivalry_pairs)
    if "capacity" in snap.columns:
        snap["capacity"] = pd.to_numeric(snap["capacity"], errors="coerce")
    drop_guess_cols = ["home_team_guess", "away_team_guess", "home_key", "away_key"]
    snap = snap.drop(columns=[c for c in drop_guess_cols if c in snap.columns], errors="ignore")
    col_order_front = [
        "team_slug","team_name","team_url",
        "event_id","title",
        "week",
        "homeTeam","awayTeam",
        "homeConference","awayConference",
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
# Main snapshot logic:
# per-team scrape → append to ONE temp JSONL + ONE temp CSV → enrich at end → append to master
# ---------------------------
def log_price_snapshot():
    now_et = datetime.now(TIMEZONE)
    time_str = now_et.strftime("%H:%M")
    ts = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    TMP_JSONL = os.path.join(DAILY_DIR, f"_tmp_tickpick_rows_{ts}.jsonl")
    TMP_CSV   = os.path.join(DAILY_DIR, f"_tmp_snapshots_{ts}.csv")

    if not ALWAYS_RUN_DAILY and time_str not in COLLECTION_TIMES:
        print(f"⏭️  Not a collection window ({time_str}).")
        return

    teams = _load_teams_list()
    if not teams:
        print("❌ No team URLs available.")
        return

    print(f"🔧 TEAMS file: {TEAMS_JSON_PATH}")
    print(f"🔧 TEAMS_LIMIT={TEAMS_LIMIT} (0=all)")
    print(f"🗂️  Temp JSONL: {TMP_JSONL}")
    print(f"🗂️  Temp CSV  : {TMP_CSV}")
    print(f"🚀 Starting TickPick scrape for {len(teams)} team pages...")

    all_rows_count = 0
    scraped_ok = 0
    scraped_fail = 0
    csv_exists = os.path.exists(TMP_CSV)

    for idx, t in enumerate(teams, start=1):
        url = t["url"]
        team_name = _titleize_from_url(url)
        print(f"  [{idx}/{len(teams)}] 🏈 Scraping {team_name} ...", flush=True)

        try:
            pricer = TickPickPricer(
                team_urls=[url],
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
                print(f"      ❌ No JSON produced for {team_name}")
                scraped_fail += 1
                _cleanup_pricer_exports(export_paths)
                continue

            # read rows from TickPickPricer export
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            finally:
                _cleanup_pricer_exports(export_paths)

            if isinstance(rows, list) and rows:
                # --- append raw rows to ONE JSONL file ---
                with open(TMP_JSONL, "a", encoding="utf-8") as jf:
                    for r in rows:
                        jf.write(json.dumps(r, ensure_ascii=False) + "\n")

                # --- map to snapshots and append to ONE CSV file (no enrichment yet) ---
                snap_chunk = _map_rows_to_snapshots(rows, now_et)
                if not snap_chunk.empty:
                    # write header only once
                    header = not csv_exists and not os.path.exists(TMP_CSV)
                    snap_chunk.to_csv(TMP_CSV, mode="a", header=header, index=False)
                    csv_exists = True

                all_rows_count += len(rows)
                print(f"      ✅ {team_name}: {len(rows)} events (total so far: {all_rows_count})")
                scraped_ok += 1
            else:
                print(f"      ⚠️ {team_name}: 0 events")
                scraped_ok += 1

        except Exception as e:
            print(f"      💥 Error scraping {team_name}: {e}")
            scraped_fail += 1

    print(f"📦 Aggregate: {all_rows_count} total event rows | ✅ {scraped_ok} ok | ❌ {scraped_fail} failed")
    if not os.path.exists(TMP_CSV):
        print("⚠️ No snapshot CSV produced. Exiting.")
        return

    # ---- read the single temp CSV, enrich once, then append to master ----
    try:
        snap_all = pd.read_csv(TMP_CSV)
    except Exception as e:
        print(f"❌ Failed to read temp CSV {TMP_CSV}: {e}")
        return

    if snap_all.empty:
        print("⚠️ Temp CSV is empty after reading. Exiting.")
        return
    
    # Normalize titles, infer missing home/away from title, and apply team aliases (no new columns)
    snap_all = _apply_title_and_alias_fixes(snap_all)
    snap_all = _enrich_with_schedule_and_stadiums(snap_all)

    # ⏰ NEW: filter out games whose kickoff has already passed
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if "startDateEastern" in snap_all.columns:
        snap_all["startDateEastern"] = pd.to_datetime(snap_all["startDateEastern"], errors="coerce")
        mask_known  = snap_all["startDateEastern"].notna()
        mask_future = snap_all["startDateEastern"] >= now_et  # includes equality at kickoff
        snap_all = pd.concat([
            snap_all[ mask_known &  mask_future],
            snap_all[~mask_known]  # keep rows with no kickoff time
        ], ignore_index=True)

    # Append/dedupe within same minute per (offer_url or event_id)
    _ensure_dir(SNAPSHOT_PATH)
    key_cols: List[str] = []
    if "offer_url" in snap_all.columns:
        key_cols.append("offer_url")
    if "event_id" in snap_all.columns:
        key_cols.append("event_id")
    key_cols.extend(["date_collected", "time_collected"])

    # Ensure missing team info is filled only if those columns ALREADY exist (do not create new columns)
    def _ensure_col_existing(df, target, candidates):
        if target not in df.columns:
            return df
        src = next((c for c in candidates if c in df.columns), None)
        if src is not None:
            # fill only NaNs/empties
            needs = df[target].isna() | (df[target].astype(str).str.strip() == "")
            df.loc[needs, target] = df.loc[needs, src]
        return df

    snap_all = _ensure_col_existing(snap_all, "homeTeam", ["home_team_name","homeTeam","home_team","team_name"])
    snap_all = _ensure_col_existing(snap_all, "awayTeam", ["away_team_name","awayTeam","away_team"])
    # do NOT touch slugs unless they already exist:
    snap_all = _ensure_col_existing(snap_all, "team_slug", ["home_team_slug","home_slug","homeTeamSlug","team_slug"])

    if os.path.exists(SNAPSHOT_PATH):
        existing = pd.read_csv(SNAPSHOT_PATH)
        combined = pd.concat([existing, snap_all], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in key_cols if c in combined.columns],
            keep="last",
        )
        _write_csv_atomic(combined, SNAPSHOT_PATH)
        print(f"✅ Snapshot appended ({len(snap_all)} new rows). Total now: {len(combined)}")
        print(f"[daily_snapshot] write complete → {SNAPSHOT_PATH}")
        # freshness log
        _df = combined
    else:
        _write_csv_atomic(snap_all, SNAPSHOT_PATH)
        print(f"✅ Snapshot saved ({len(snap_all)} rows) to {SNAPSHOT_PATH}")
        print(f"[daily_snapshot] write complete → {SNAPSHOT_PATH}")
        # freshness log
        _df = snap_all

    # Post-write freshness assertion (America/New_York)
    try:
        col = "date_collected"
        max_date = pd.to_datetime(_df[col], errors="coerce").max()
        now_d = datetime.now(ZoneInfo("America/New_York")).date()
        print(f"[daily_snapshot] rows={len(_df)}, max({col})={max_date.date() if pd.notnull(max_date) else 'NA'} (now={now_d})")
    except Exception as e:
        print(f"[daily_snapshot] freshness check failed: {e}")

    # ---- clean up single temp files ----
    if KEEP_COMBINED_EXPORTS:
        print(f"📝 Keeping temp files:\n  - {TMP_JSONL}\n  - {TMP_CSV}")
    else:
        cleanup_targets = [TMP_JSONL, TMP_CSV]

        # also nuke any stray .etag.json files in the daily dir
        for f in Path(DAILY_DIR).glob("*.etag.json"):
            cleanup_targets.append(str(f))

        for p in cleanup_targets:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"🧹 Removed temp file: {p}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file {p}: {e}")

if __name__ == "__main__":
    log_price_snapshot()
