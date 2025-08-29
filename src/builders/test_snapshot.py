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
REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"

_env_daily = os.getenv("DAILY_DIR")
_env_snap  = os.getenv("SNAPSHOT_PATH")

if REPO_DATA_LOCK:
    DAILY_DIR = os.path.join(PROJ_DIR, "data", "daily")
    SNAPSHOT_PATH = os.path.join(DAILY_DIR, "price_snapshots_test.csv")
else:
    DAILY_DIR = _env_daily or os.path.join(PROJ_DIR, "data", "daily")
    SNAPSHOT_PATH = _env_snap or os.path.join(DAILY_DIR, "price_snapshots.csv")

if REPO_DATA_LOCK and (_env_daily or _env_snap):
    print("⚠️  REPO_DATA_LOCK=1 -> ignoring DAILY_DIR/SNAPSHOT_PATH env and writing to repo paths only.")

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

KEEP_COMBINED_EXPORTS = os.getenv("KEEP_COMBINED_EXPORTS", "0") == "1"

# ---------------------------
# Team list config
# ---------------------------
TEAMS_JSON_PATH = os.getenv(
    "TEAMS_JSON_PATH",
    os.path.join(PROJ_DIR, "data", "permanent", "tickpick_teams.txt")
)
TEAMS_LIMIT = int(os.getenv("TEAMS_LIMIT", "0"))
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

DELETE_TP_EXPORTS = os.getenv("DELETE_TP_EXPORTS", "1") == "1"

# Paths for enrichment sources
WEEKLY_SCHEDULE_PATH = os.path.join(PROJ_DIR, "data", "weekly", f"full_{YEAR}_schedule.csv")
RIVALRIES_PATH = os.path.join(PROJ_DIR, "data", "annual", f"rivalries_{YEAR}.csv")

# ---------------------------
# Import TickPickPricer
# ---------------------------
try:
    from scrapers.tickpick_pricer import TickPickPricer
except Exception as e:
    raise ImportError(f"Could not import scrapers.tickpick_pricer: {e!r}")

# ---------------------------
# Helpers
# ---------------------------
def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.__tmp__"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _read_json_array(path: str) -> Optional[List[dict]]:
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
                return data if isinstance(data, list) else None
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
    data = _read_json_array(TEAMS_JSON_PATH)

    if data is None and TEAMS_JSON_PATH.endswith(".txt"):
        alt = TEAMS_JSON_PATH[:-4] + ".json"
        if os.path.exists(alt):
            print(f"ℹ️  Trying alternate teams file: {alt}")
            data = _read_json_array(alt)

    if data:
        out: List[Dict[str, str]] = []
        bad = 0
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
                    bad += 1
            else:
                bad += 1
        if bad:
            print(f"ℹ️  Skipped {bad} invalid rows in teams file.")
        if out:
            if TEAMS_LIMIT > 0:
                print(f"📄 Loaded {len(out)} teams from file; using first {TEAMS_LIMIT}.")
                return out[:TEAMS_LIMIT]
            print(f"📄 Loaded {len(out)} teams from file; using ALL.")
            return out

    fallback = [
        {"slug": "boston-college-eagles-football",
         "url": "https://www.tickpick.com/ncaa-football/boston-college-eagles-football-tickets/"},
        {"slug": "california-golden-bears-football",
         "url": "https://www.tickpick.com/ncaa-football/california-golden-bears-football-tickets/"},
        {"slug": "clemson-tigers-football",
         "url": "https://www.tickpick.com/ncaa-football/clemson-tigers-football-tickets/"},
    ]
    print("🚨 Falling back to built-in 3-team list (teams file missing or invalid JSON).")
    return fallback[:TEAMS_LIMIT] if TEAMS_LIMIT > 0 else fallback

def _titleize_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    tail = tail.replace("-tickets", "").replace("-", " ")
    return tail.title()

def _slug_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace("-tickets", "")

def _map_rows_to_snapshots(rows: List[Dict[str, Any]], now_et: datetime) -> pd.DataFrame:
    time_str = now_et.strftime("%H:%M:%S")
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

    # Prefer committed home/away if present; fallback to guesses
    def _pick_norm(primary: str, fallback: str) -> pd.Series:
        base = df.get(primary)
        if base is None:
            base = pd.Series([""] * len(df), index=df.index)
        base = base.fillna("")
        if fallback in df.columns:
            fb = df[fallback].fillna("")
            base = base.where(base.astype(bool), fb)
        return base.map(_normalize_team_name)

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
    if snap.empty:
        snap["isRivalry"] = False
        return snap
    if not rivalry_pairs:
        if "isRivalry" not in snap.columns:
            snap["isRivalry"] = False
        return snap

    home_norm = snap.get("homeTeam", pd.Series([""]*len(snap))).fillna("").map(_normalize_team_name)
    away_norm = snap.get("awayTeam", pd.Series([""]*len(snap))).fillna("").map(_normalize_team_name)

    snap["isRivalry"] = [
        bool(h and a and (frozenset({h, a}) in rivalry_pairs))
        for h, a in zip(home_norm, away_norm)
    ]
    snap["isRivalry"] = snap["isRivalry"].astype(bool)
    return snap

# ---------------------------
# Title cleanup + alias + matchup fixes (no new columns created)
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

_VS_RE = re.compile(r'\s+vs\.?\s+', flags=re.IGNORECASE)

def _split_matchup(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return (None, None)
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
    if snap.empty:
        return snap
    df = snap.copy()
    aliases = _load_team_aliases()

    if "title" in df.columns:
        df["title"] = df["title"].map(_clean_event_title)

    def _infer_pair_from_title(row):
        ih, ia = _split_matchup(row.get("title") or "")
        return ih, ia

    # Prefer filling canonical columns; else, fill the guess columns.
    has_home = "homeTeam" in df.columns
    has_away = "awayTeam" in df.columns

    if has_home:
        inferred_home = df.apply(lambda r: _infer_pair_from_title(r)[0], axis=1)
        mask = df["homeTeam"].isna() | (df["homeTeam"].astype(str).str.strip() == "")
        df.loc[mask, "homeTeam"] = inferred_home[mask]
        df["homeTeam"] = df["homeTeam"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)
    else:
        if "home_team_guess" in df.columns:
            inferred_home = df.apply(lambda r: _infer_pair_from_title(r)[0], axis=1)
            mask = df["home_team_guess"].isna() | (df["home_team_guess"].astype(str).str.strip() == "")
            df.loc[mask, "home_team_guess"] = inferred_home[mask]
            df["home_team_guess"] = df["home_team_guess"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

    if has_away:
        inferred_away = df.apply(lambda r: _infer_pair_from_title(r)[1], axis=1)
        mask = df["awayTeam"].isna() | (df["awayTeam"].astype(str).str.strip() == "")
        df.loc[mask, "awayTeam"] = inferred_away[mask]
        df["awayTeam"] = df["awayTeam"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)
    else:
        if "away_team_guess" in df.columns:
            inferred_away = df.apply(lambda r: _infer_pair_from_title(r)[1], axis=1)
            mask = df["away_team_guess"].isna() | (df["away_team_guess"].astype(str).str.strip() == "")
            df.loc[mask, "away_team_guess"] = inferred_away[mask]
            df["away_team_guess"] = df["away_team_guess"].apply(lambda x: _canonicalize_if_aliased(x, aliases) if isinstance(x, str) else x)

    return df

# ---------------------------
# Enrichment (schedule + stadiums + ranks + rivalry)
# ---------------------------
def _enrich_with_schedule_and_stadiums(snap: pd.DataFrame) -> pd.DataFrame:
    if snap.empty:
        return snap

    # Days until game
    snap["days_until_game"] = (
        pd.to_datetime(snap["date_local"], errors="coerce") -
        pd.to_datetime(snap["date_collected"], errors="coerce")
    ).dt.days

    # Load schedule once
    sched = _load_schedule()

    # If no schedule, ensure columns exist and bail early (rivalries still handled later)
    cols_to_carry = [
        "homeTeam","awayTeam","stadium","capacity","neutralSite","conferenceGame",
        "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
        "homeConference","awayConference","week"
    ]

    if sched is None or sched.empty:
        for col in cols_to_carry:
            if col not in snap.columns:
                snap[col] = pd.NA
        # Rivalries will be marked below based on any available home/away
        rivalry_pairs = _load_rivalries()
        snap = _mark_rivalries(snap, rivalry_pairs)
        if "capacity" in snap.columns:
            snap["capacity"] = pd.to_numeric(snap["capacity"], errors="coerce")
        # Drop guess columns if present
        snap = snap.drop(columns=[c for c in ("home_team_guess","away_team_guess") if c in snap.columns], errors="ignore")
        # tidy order
        return _finalize_columns_order(snap)

    # Build join keys from current best values (home/away if present; else guesses)
    sched_pre = _prepare_schedule_for_join(sched)
    snap_pre  = _prepare_snapshots_for_join(snap)

    # direct (home vs away)
    m1 = snap_pre.merge(sched_pre, how="left", on="date_key", suffixes=("", "_sched"))
    m1 = m1[(m1["home_key"] == m1["home_key_sched"]) & (m1["away_key"] == m1["away_key_sched"])].copy()
    m1 = m1.sort_values("snap_idx").drop_duplicates(subset=["snap_idx"], keep="first")

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
    m2 = snap_pre.merge(sched_flip, how="left", on="date_key", suffixes=("", "_sched"))
    m2 = m2[(m2["home_key"] == m2["home_key_sched"]) & (m2["away_key"] == m2["away_key_sched"])].copy()
    m2 = m2.sort_values("snap_idx").drop_duplicates(subset=["snap_idx"], keep="first")

    # overlay: prefer direct over flipped
    matched = pd.merge(
        m1[["snap_idx"] + [c for c in cols_to_carry if c in m1.columns]],
        m2[["snap_idx"] + [c for c in cols_to_carry if c in m2.columns]],
        on="snap_idx", how="outer", suffixes=("_dir","_flip"),
    )

    def pick(row, col):
        a = row.get(f"{col}_dir")
        b = row.get(f"{col}_flip")
        return a if pd.notna(a) else b

    overlay = pd.DataFrame({"snap_idx": matched["snap_idx"]})
    for col in cols_to_carry:
        if f"{col}_dir" in matched.columns or f"{col}_flip" in matched.columns:
            overlay[col] = matched.apply(lambda r, c=col: pick(r, c), axis=1)

    # --- Fill from overlay without merging; align by snap.index ---
    # Make sure target columns exist so we can fill them
    for col in cols_to_carry:
        if col not in snap.columns:
            snap[col] = pd.NA

    # overlay is keyed by snap_idx (which equals the original snap.index used to build snap_pre)
    ov_idx = overlay.set_index("snap_idx")

    for col in cols_to_carry:
        if col not in ov_idx.columns:
            continue
        # Map overlay values to snap by its index → perfectly aligned 1:1
        ov = snap.index.to_series().map(ov_idx[col])

        # Fill only where snap is missing/blank AND overlay has a value
        if pd.api.types.is_numeric_dtype(snap[col]):
            need = snap[col].isna()
        else:
            need = snap[col].isna() | (snap[col].astype(str).str.strip() == "")

        has = ov.notna()
        mask = need & has

        # Use .values on RHS to avoid any residual index alignment surprises
        snap.loc[mask, col] = ov[mask].values

    # Backfill home/away strictly from "title" if still missing (keeps existing values)
    def _clean_title_local(t):
        if not isinstance(t, str):
            return t
        t = t.replace('\u00A0', ' ').strip()
        if ":" in t:
            right = t.split(":", 1)[1].strip()
            if right:
                t = right
        return t

    def _part_from_title(t, idx):
        t = _clean_title_local(t)
        if not isinstance(t, str):
            return None
        parts = _VS_RE.split(t, maxsplit=1)
        if len(parts) == 2:
            side = parts[idx].strip(" \t-–—")
            return side if side else None
        return None

    if "title" in snap.columns:
        if "homeTeam" in snap.columns:
            mask = snap["homeTeam"].isna() | (snap["homeTeam"].astype(str).str.strip() == "")
            parsed = snap["title"].apply(lambda s: _part_from_title(s, 0))
            fill = mask & parsed.notna() & (parsed.astype(str).str.strip() != "")
            snap.loc[fill, "homeTeam"] = parsed[fill]

        if "awayTeam" in snap.columns:
            mask = snap["awayTeam"].isna() | (snap["awayTeam"].astype(str).str.strip() == "")
            parsed = snap["title"].apply(lambda s: _part_from_title(s, 1))
            fill = mask & parsed.notna() & (parsed.astype(str).str.strip() != "")
            snap.loc[fill, "awayTeam"] = parsed[fill]

    # Rivalries and final numeric coercions
    rivalry_pairs = _load_rivalries()
    snap = _mark_rivalries(snap, rivalry_pairs)
    if "capacity" in snap.columns:
        snap["capacity"] = pd.to_numeric(snap["capacity"], errors="coerce")

    # Drop guess columns to keep the master tidy
    snap = snap.drop(columns=[c for c in ("home_team_guess","away_team_guess","home_key","away_key") if c in snap.columns],
                     errors="ignore")

    return _finalize_columns_order(snap)

def _finalize_columns_order(snap: pd.DataFrame) -> pd.DataFrame:
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
# Main snapshot logic
# ---------------------------
def log_price_snapshot():
    now_et = datetime.now(TIMEZONE)
    time_str = now_et.strftime("%H:%M")
    ts = now_et.strftime("%Y%m%d_%H%M%S")
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

            # read rows -> map -> CLEAN (title/aliases/guesses) BEFORE writing temp CSV
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            finally:
                _cleanup_pricer_exports(export_paths)

            if isinstance(rows, list) and rows:
                # append raw rows to one JSONL
                with open(TMP_JSONL, "a", encoding="utf-8") as jf:
                    for r in rows:
                        jf.write(json.dumps(r, ensure_ascii=False) + "\n")

                # map to snapshots, then clean titles/aliases and infer guesses early
                snap_chunk = _map_rows_to_snapshots(rows, now_et)
                if not snap_chunk.empty:
                    snap_chunk = _apply_title_and_alias_fixes(snap_chunk)

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

    # Enrich with schedule/stadiums/ranks & rivalries
    snap_all = _enrich_with_schedule_and_stadiums(snap_all)

    WEEKLY_SCHEDULE_PATH = os.path.join(PROJ_DIR, "data", "weekly", f"full_{YEAR}_schedule.csv")
    snap_all = fallback_fill_unmatched_snapshots(snap_all, WEEKLY_SCHEDULE_PATH)

    # Append/dedupe within same minute per (offer_url or event_id)
    _ensure_dir(SNAPSHOT_PATH)
    key_cols: List[str] = []
    if "offer_url" in snap_all.columns:
        key_cols.append("offer_url")
    if "event_id" in snap_all.columns:
        key_cols.append("event_id")
    key_cols.extend(["date_collected", "time_collected"])

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
        _df = combined
    else:
        _write_csv_atomic(snap_all, SNAPSHOT_PATH)
        print(f"✅ Snapshot saved ({len(snap_all)} rows) to {SNAPSHOT_PATH}")
        print(f"[daily_snapshot] write complete → {SNAPSHOT_PATH}")
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
        for f in Path(DAILY_DIR).glob("*.etag.json"):
            cleanup_targets.append(str(f))
        for p in cleanup_targets:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"🧹 Removed temp file: {p}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file {p}: {e}")

# ---------------------------
# Fallback matcher: fill stubborn unmatched rows
# ---------------------------
from math import exp

_FM_STOP = {"university","state","college","the","of","and","at","football","st","saint"}

def _fm_norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    x = s.lower()
    x = x.replace("&", " and ")
    for ch in "/.,-()[]{}'’":
        x = x.replace(ch, " ")
    toks = [t for t in x.split() if t and t not in _FM_STOP]
    return " ".join(toks)

def _fm_tokens(s: str) -> set[str]:
    return set(_fm_norm(s).split()) if isinstance(s, str) else set()

def _fm_split_title(title: str) -> tuple[str|None, str|None]:
    if not isinstance(title, str) or not title.strip():
        return (None, None)
    t = title.strip()
    if ":" in t:
        t = t.split(":", 1)[1].strip() or t
    t = t.replace("\u00A0", " ")
    m = re.split(r"\s+vs\.?\s+", t, flags=re.IGNORECASE, maxsplit=1)
    if len(m) == 2:
        left, right = m[0].strip(" \t-–—"), m[1].strip(" \t-–—")
        return (left or None, right or None)
    return (None, None)

def _fm_to_time(dt_like) -> Optional[pd.Timestamp]:
    if pd.isna(dt_like):
        return None
    try:
        return pd.to_datetime(dt_like, errors="coerce")
    except Exception:
        return None

def _fm_row_is_matched(row: pd.Series) -> bool:
    # "Matched enough" means we got week and either stadium or conferences (tweak if you prefer stricter)
    has_week = pd.notna(row.get("week"))
    has_site = pd.notna(row.get("stadium")) or pd.notna(row.get("homeConference")) or pd.notna(row.get("awayConference"))
    return bool(has_week and has_site)

def _fm_score(snapshot_row: pd.Series, cand: pd.Series) -> tuple[float, bool, dict]:
    """
    Return (score, flipped, parts) where flipped=True if away/home orientation fits better.
    Score components:
      - team/title token overlap (heavier)
      - time proximity (lighter)
    """
    # Snapshot tokens
    sh = _fm_tokens(snapshot_row.get("homeTeam", ""))
    sa = _fm_tokens(snapshot_row.get("awayTeam", ""))
    th, ta = _fm_split_title(snapshot_row.get("title", "") or "")
    th = _fm_tokens(th or "")
    ta = _fm_tokens(ta or "")
    # Schedule tokens
    ch = _fm_tokens(cand.get("homeTeam", ""))
    ca = _fm_tokens(cand.get("awayTeam", ""))

    def jacc(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    # direct match (home→home, away→away)
    s_h_direct = max(jacc(sh, ch), jacc(th, ch))
    s_a_direct = max(jacc(sa, ca), jacc(ta, ca))
    direct = (s_h_direct + s_a_direct) / 2.0

    # flipped match (home→away, away→home)
    s_h_flip = max(jacc(sh, ca), jacc(th, ca))
    s_a_flip = max(jacc(sa, ch), jacc(ta, ch))
    flip = (s_h_flip + s_a_flip) / 2.0

    flipped = flip > direct
    team_score = max(direct, flip)  # 0..1

    # time proximity (minutes). Prefer closer, but keep it a small weight.
    t_snap_date = _fm_to_time(snapshot_row.get("date_local"))
    t_snap_time = snapshot_row.get("time_local")
    if isinstance(t_snap_time, str) and t_snap_time.strip():
        # combine date_local + time_local if time is given as HH:MM (local)
        try:
            t_snap = pd.to_datetime(f"{str(t_snap_date.date())} {t_snap_time}", errors="coerce")
        except Exception:
            t_snap = t_snap_date
    else:
        t_snap = t_snap_date

    t_sched = _fm_to_time(cand.get("startDateEastern"))
    time_score = 0.0
    if t_snap is not None and t_sched is not None:
        try:
            delta_min = abs((t_sched - t_snap).total_seconds()) / 60.0
            # 0 min -> 1.0 ; 60 min -> ~0.37 ; 180 min -> ~0.05
            time_score = exp(-delta_min / 60.0)
        except Exception:
            time_score = 0.0

    # weights: teams dominate
    score = 0.85 * team_score + 0.15 * time_score
    parts = {
        "team_score": round(team_score, 4),
        "time_score": round(time_score, 4),
        "direct": round(direct, 4),
        "flip": round(flip, 4),
    }
    return score, flipped, parts

def fallback_fill_unmatched_snapshots(snap: pd.DataFrame, schedule_csv_path: str) -> pd.DataFrame:
    """
    Heuristic pass for the last stubborn rows:
      1) If per-date there is exactly one unmatched snap and exactly one unused schedule row → assign.
      2) Else choose closest match by token overlap (+ light time proximity).
    Only fills missing values; never overwrites non-null fields.
    """
    if snap.empty:
        return snap.copy()

    # --- Load schedule with minimal normalization/columns ---
    if not os.path.exists(schedule_csv_path):
        print(f"[fallback] schedule not found: {schedule_csv_path}")
        return snap.copy()

    sched = pd.read_csv(schedule_csv_path)
    # harmonize required columns
    rename_try = {
        "startDateEastern": ["startDateEastern", "start_date_eastern", "startDate", "game_date", "date_local"],
        "homeTeam": ["homeTeam", "home_team", "home"],
        "awayTeam": ["awayTeam", "away_team", "away"],
    }
    for canon, cands in rename_try.items():
        if canon not in sched.columns:
            for c in cands:
                if c in sched.columns:
                    sched = sched.rename(columns={c: canon})
                    break
    if not {"startDateEastern","homeTeam","awayTeam"}.issubset(sched.columns):
        print("[fallback] schedule missing required columns after rename attempt")
        return snap.copy()

    optional = ["week","stadium","capacity","neutralSite","conferenceGame",
                "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
                "homeConference","awayConference"]
    for c in optional:
        if c not in sched.columns:
            sched[c] = pd.NA

    sched["startDateEastern"] = pd.to_datetime(sched["startDateEastern"], errors="coerce")
    sched["date_key"] = sched["startDateEastern"].dt.strftime("%Y-%m-%d")
    sched["__sched_id"] = sched.index  # stable id for "used" tracking

    # Build date keys on snapshots
    out = snap.copy()
    out["date_key"] = pd.to_datetime(out["date_local"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Which schedule rows are already "used"? If a snapshot clearly matches (both teams tokens equal), mark used.
    used_sched_ids: set[int] = set()
    for _, r in out.iterrows():
        if pd.isna(r.get("date_key")):
            continue
        sh, sa = _fm_tokens(r.get("homeTeam", "")), _fm_tokens(r.get("awayTeam", ""))
        if not sh or not sa:
            continue
        cand = sched[(sched["date_key"] == r["date_key"])]
        if cand.empty:
            continue
        # try direct
        m_direct = cand[(cand["homeTeam"].map(_fm_tokens) == sh) & (cand["awayTeam"].map(_fm_tokens) == sa)]
        m_flip   = cand[(cand["homeTeam"].map(_fm_tokens) == sa) & (cand["awayTeam"].map(_fm_tokens) == sh)]
        m = m_direct if not m_direct.empty else m_flip
        if not m.empty:
            used_sched_ids.add(int(m.iloc[0]["__sched_id"]))

    # Helper to fill from a schedule row into a snapshot row (only missing fields)
    carry_cols = ["week","stadium","capacity","neutralSite","conferenceGame",
                  "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
                  "homeConference","awayConference","homeTeam","awayTeam"]
    def _apply_fill(row: pd.Series, cand: pd.Series, flipped: bool) -> pd.Series:
        # choose home/away orientation
        src_home, src_away = cand["homeTeam"], cand["awayTeam"]
        if flipped:
            src_home, src_away = cand["awayTeam"], cand["homeTeam"]

        for col in carry_cols:
            if col in ("homeTeam","awayTeam"):
                src = src_home if col == "homeTeam" else src_away
            else:
                src = cand.get(col)
            if col not in row.index:
                continue
            if pd.isna(row[col]) or (isinstance(row[col], str) and row[col].strip() == ""):
                row[col] = src
        return row

    # Pass 1: per-date unique remainder
    for date_key, group_idx in out.groupby("date_key").groups.items():
        idxs = list(group_idx)
        if not idxs:
            continue
        g = out.loc[idxs]

        # snap needing fill
        need_mask = ~g.apply(_fm_row_is_matched, axis=1)
        need_ids = list(g[need_mask].index)

        # sched candidates not already used
        sched_cands = sched[(sched["date_key"] == date_key) & (~sched["__sched_id"].isin(used_sched_ids))]

        if len(need_ids) == 1 and len(sched_cands) == 1:
            i = need_ids[0]
            out.loc[i] = _apply_fill(out.loc[i], sched_cands.iloc[0], flipped=False)
            used_sched_ids.add(int(sched_cands.iloc[0]["__sched_id"]))

    # Pass 2: closest-match scoring for anything still unmatched
    still_need = out[~out.apply(_fm_row_is_matched, axis=1)].index.tolist()
    for i in still_need:
        r = out.loc[i]
        if pd.isna(r.get("date_key")):
            continue
        cands = sched[(sched["date_key"] == r["date_key"]) & (~sched["__sched_id"].isin(used_sched_ids))]
        if cands.empty:
            continue
        # score each candidate
        scored = []
        for _, c in cands.iterrows():
            score, flipped, _parts = _fm_score(r, c)
            scored.append((score, flipped, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        out.loc[i] = _apply_fill(r, best[2], flipped=best[1])
        used_sched_ids.add(int(best[2]["__sched_id"]))

    # Clean up
    out.drop(columns=["date_key"], inplace=True, errors="ignore")
    return out


if __name__ == "__main__":
    log_price_snapshot()
