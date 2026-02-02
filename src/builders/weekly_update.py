#!/usr/bin/env python3
"""Weekly update: schedule refresh + rankings merge + stadium capacity merge."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# --- Robust import path (run from anywhere) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJ_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetchers.schedule_fetcher import ScheduleFetcher
from fetchers.rankings_fetcher import RankingsFetchers
from fetchers.rankings_api_fetcher import RankingsApiFetcher
from fetchers.results_fetcher import save_completed_fbs_games
from fetchers.venues_fetcher import save_venues
from utils.logging_utils import get_logger
from utils.status import write_status

DEFAULT_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

log = get_logger("weekly_update")
VERBOSE = os.getenv("WEEKLY_VERBOSE", "0") == "1"
QUIET = os.getenv("WEEKLY_QUIET", "1") == "1"
KEEP_ONLY = os.getenv("WEEKLY_KEEP_ONLY", "1") == "1"


# ----------------------------- Paths & helpers -----------------------------
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
    log.warning("DATA_DIR resolved outside repo: %s → forcing repo path", DATA_DIR)
    DATA_DIR = PROJ_DIR / "data"

WEEKLY_DIR = DATA_DIR / "weekly"
ANNUAL_DIR = DATA_DIR / "annual"
PERMANENT_DIR = DATA_DIR / "permanent"
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_JSON = PERMANENT_DIR / "team_aliases.json"
if not QUIET:
    if VERBOSE:
        log.info("[weekly_update] Paths resolved:")
        log.info("  PROJ_DIR:            %s", PROJ_DIR)
        log.info("  DATA_DIR:            %s", DATA_DIR)
        log.info("  WEEKLY_DIR:          %s", WEEKLY_DIR)
        log.info("  ANNUAL_DIR:          %s", ANNUAL_DIR)
        log.info("  PERMANENT_DIR:       %s", PERMANENT_DIR)
        log.info("  ALIAS_JSON:          %s", ALIAS_JSON)
        log.info("  WEEKLY_SCHEDULE_OUT: <computed after schedule fetch>")
    else:
        log.info("[weekly_update] Paths: PROJ_DIR=%s DATA_DIR=%s WEEKLY_DIR=%s", PROJ_DIR, DATA_DIR, WEEKLY_DIR)

if QUIET:
    import logging
    for name in [
        "schedule_fetcher",
        "rankings_fetcher",
        "rankings_api_fetcher",
        "results_fetcher",
        "venues_fetcher",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    log.setLevel(logging.ERROR)


def _load_alias_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Alias map not found at {path}. "
            "Please create data/permanent/team_aliases.json (a dict of {'Incoming Name': 'Canonical Name'})."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


ALIAS_MAP = _load_alias_map(ALIAS_JSON)


def _canon(name: object) -> object:
    if pd.isna(name):
        return name
    s = str(name).strip()
    if s in ALIAS_MAP:
        return ALIAS_MAP[s]
    stripped = re.sub(r"\s*\([^)]*\)", "", s).strip().rstrip(".")
    return ALIAS_MAP.get(stripped, stripped)


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


def _find_stadiums_file(schedule_year: int) -> Optional[Path]:
    """Pick the closest available stadiums_<YEAR>.csv from annual/data directories."""
    candidates: list[tuple[int, int, int, Path]] = []
    # (abs_delta, is_future, priority, path)
    for priority, base in enumerate([ANNUAL_DIR, DATA_DIR]):
        for p in base.glob("stadiums_*.csv"):
            m = re.search(r"stadiums_(\d{4})\.csv$", p.name)
            if not m:
                continue
            year = int(m.group(1))
            abs_delta = abs(year - schedule_year)
            is_future = 1 if year > schedule_year else 0
            candidates.append((abs_delta, is_future, priority, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return candidates[0][3]


def _find_venues_file(schedule_year: int) -> Optional[Path]:
    """Pick the closest available venues_<YEAR>.csv (or venues.csv) from annual/data."""
    candidates: list[tuple[int, int, int, Path]] = []
    # (abs_delta, is_future, priority, path)
    for priority, base in enumerate([ANNUAL_DIR, DATA_DIR]):
        for p in base.glob("venues_*.csv"):
            m = re.search(r"venues_(\d{4})\.csv$", p.name)
            if not m:
                continue
            year = int(m.group(1))
            abs_delta = abs(year - schedule_year)
            is_future = 1 if year > schedule_year else 0
            candidates.append((abs_delta, is_future, priority, p))
        # optional generic venues.csv
        generic = base / "venues.csv"
        if generic.exists():
            candidates.append((0, 0, priority, generic))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return candidates[0][3]


def _norm_key(s: object) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\\s-]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\\s+", " ", s).strip()
    return s


_STADIUM_STOPWORDS = {
    "stadium", "field", "complex", "memorial", "park", "center", "centre",
    "coliseum", "arena", "athletics", "athletic", "at", "the", "of",
}


def _norm_key_loose(s: object) -> str:
    base = _norm_key(s)
    tokens = [t for t in base.split() if t and t not in _STADIUM_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def _to_date_yyyy_mm_dd(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.date


def _atomic_csv_write(df: pd.DataFrame, path: Path, retries: int = 5, delay: float = 0.6) -> None:
    """Write via temp file + os.replace to avoid Windows sharing violations."""
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    for i in range(retries):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except PermissionError as e:
            log.warning("PermissionError writing %s: attempt %d/%d – %s", path, i + 1, retries, e)
            time.sleep(delay)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
    alt = path.with_name(f"{path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    log.error("Could not replace %s. Writing fallback: %s", path, alt)
    df.to_csv(alt, index=False)


def _ensure_week_column(df: pd.DataFrame) -> None:
    candidates = ["week", "weekNumber", "week_num", "gameWeek"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        raise KeyError("schedule_df is missing a 'week' (or alias: weekNumber/week_num/gameWeek) column.")
    df["week"] = pd.to_numeric(df[found], errors="coerce").astype("Int64")


def _apply_prev_game_point_diff(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute the point differential from its most recent completed game
    prior to each game in the schedule. Week 1 games will have NaN.
    """
    # Ensure unique columns to avoid merge errors
    if schedule_df.columns.duplicated().any():
        dups = schedule_df.columns[schedule_df.columns.duplicated()].tolist()
        log.warning("Duplicate columns detected in schedule_df: %s", dups)
        schedule_df = schedule_df.loc[:, ~schedule_df.columns.duplicated()]

    if "id" not in schedule_df.columns:
        log.warning("Schedule is missing 'id'; skipping previous point diff computation.")
        return schedule_df

    # Normalize score columns
    if "homePoints" not in schedule_df.columns and "home_points" in schedule_df.columns:
        schedule_df["homePoints"] = schedule_df["home_points"]
    if "awayPoints" not in schedule_df.columns and "away_points" in schedule_df.columns:
        schedule_df["awayPoints"] = schedule_df["away_points"]

    # Pick a datetime column for ordering
    dt_col = "startDateEastern" if "startDateEastern" in schedule_df.columns else "startDate"
    if dt_col not in schedule_df.columns:
        log.warning("Schedule is missing '%s'; skipping previous point diff computation.", dt_col)
        return schedule_df

    schedule_df = schedule_df.copy()
    schedule_df["game_datetime"] = pd.to_datetime(schedule_df[dt_col], errors="coerce", utc=True)

    # Build team-level rows
    base_cols = ["id", "game_datetime", "homeTeam", "awayTeam", "homePoints", "awayPoints"]
    missing = [c for c in base_cols if c not in schedule_df.columns]
    if missing:
        log.warning("Schedule missing columns %s; skipping previous point diff computation.", missing)
        return schedule_df

    home_rows = schedule_df[["id", "game_datetime", "homeTeam", "homePoints", "awayPoints"]].copy()
    home_rows["team"] = home_rows["homeTeam"]
    home_rows["points_for"] = pd.to_numeric(home_rows["homePoints"], errors="coerce")
    home_rows["points_against"] = pd.to_numeric(home_rows["awayPoints"], errors="coerce")

    away_rows = schedule_df[["id", "game_datetime", "awayTeam", "awayPoints", "homePoints"]].copy()
    away_rows["team"] = away_rows["awayTeam"]
    away_rows["points_for"] = pd.to_numeric(away_rows["awayPoints"], errors="coerce")
    away_rows["points_against"] = pd.to_numeric(away_rows["homePoints"], errors="coerce")

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    team_games = team_games.sort_values(["team", "game_datetime", "id"]).copy()

    # For each team, last completed point diff before current game
    def _prev_completed(s: pd.Series) -> pd.Series:
        last_completed = s.where(s.notna()).ffill()
        return last_completed.shift(1)

    team_games["prev_point_diff"] = team_games.groupby("team")["point_diff"].transform(_prev_completed)

    # Map back to schedule rows by id + team without duplicating columns
    home_prev = team_games[["id", "team", "prev_point_diff"]].rename(
        columns={"prev_point_diff": "home_last_point_diff"}
    )
    away_prev = team_games[["id", "team", "prev_point_diff"]].rename(
        columns={"prev_point_diff": "away_last_point_diff"}
    )

    schedule_df = schedule_df.drop(columns=["home_last_point_diff", "away_last_point_diff"], errors="ignore")
    schedule_df = schedule_df.merge(
        home_prev, left_on=["id", "homeTeam"], right_on=["id", "team"], how="left"
    ).drop(columns=["team"])
    schedule_df = schedule_df.merge(
        away_prev, left_on=["id", "awayTeam"], right_on=["id", "team"], how="left"
    ).drop(columns=["team"])
    schedule_df = schedule_df.drop(columns=["game_datetime"], errors="ignore")
    return schedule_df


def _analysis_ready_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a trimmed, analysis-ready view of the weekly schedule."""
    cols = [
        "id",
        "season",
        "week",
        "seasonType",
        "startDateEastern",
        "homeTeam",
        "awayTeam",
        "homePoints",
        "awayPoints",
        "homeTeamRank",
        "awayTeamRank",
        "home_last_point_diff",
        "away_last_point_diff",
        "neutralSite",
        "venue",
        "capacity",
        "conferenceGame",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    return out


def _post_run_report(schedule_df: pd.DataFrame, weekly_out: Path, requested_year: int, used_year: int) -> None:
    """Print a clean, minimal report after saving."""
    n = len(schedule_df)
    if n == 0:
        log.warning("[weekly_update] report: empty schedule")
        return

    def _pct(series: pd.Series) -> str:
        return f"{(series.notna().sum() / n) * 100:.1f}%"

    rank_filled = (
        schedule_df.get("homeTeamRank").notna().sum()
        + schedule_df.get("awayTeamRank").notna().sum()
    ) / (2 * n)
    cap_filled = _pct(schedule_df["capacity"]) if "capacity" in schedule_df.columns else "NA"
    diff_filled = _pct(schedule_df["home_last_point_diff"]) if "home_last_point_diff" in schedule_df.columns else "NA"

    # Basic integrity checks
    dup_id = schedule_df["id"].duplicated().sum() if "id" in schedule_df.columns else 0
    missing_start = schedule_df["startDateEastern"].isna().sum() if "startDateEastern" in schedule_df.columns else 0
    missing_week = schedule_df["week"].isna().sum() if "week" in schedule_df.columns else 0

    date_min = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").min()
    date_max = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").max()

    fallback_note = ""
    if used_year != requested_year:
        fallback_note = f" (fallback from {requested_year})"

    status_line = (
        f"STATUS: SUCCESS | year={used_year}{fallback_note} | rows={n} | "
        f"range={date_min.date() if pd.notnull(date_min) else 'NA'}→"
        f"{date_max.date() if pd.notnull(date_max) else 'NA'} | "
        f"ranks={rank_filled * 100:0.1f}% | cap={cap_filled} | prev_diff={diff_filled}"
    )
    if QUIET:
        print(status_line)
    else:
        log.info(status_line)
    if VERBOSE and (dup_id or missing_start or missing_week):
        log.info("Checks: dup_id=%d, missing_start=%d, missing_week=%d", dup_id, missing_start, missing_week)

    # Optional analysis export
    if VERBOSE and not KEEP_ONLY:
        out_path = weekly_out.with_name(weekly_out.stem + "_analysis.csv")
        try:
            _analysis_ready_view(schedule_df).to_csv(out_path, index=False)
            log.info("Analysis export: %s", out_path.name)
        except Exception as e:
            log.warning("[weekly_update] analysis export failed: %s", e)


def _cleanup_weekly_files(weekly_dir: Path, keep_paths: list[Path]) -> None:
    """Keep only selected files in data/weekly (others are deleted or archived)."""
    if not KEEP_ONLY:
        return
    keep_set = {p.resolve() for p in keep_paths if p}
    for p in weekly_dir.iterdir():
        if not p.is_file():
            continue
        if p.resolve() in keep_set:
            continue
        try:
            p.unlink()
        except Exception:
            pass


# ----------------------------- Load prior snapshot -----------------------------
def _load_previous_weekly(schedule_out: Path) -> Optional[pd.DataFrame]:
    if not schedule_out.exists():
        return None
    try:
        prev = pd.read_csv(schedule_out)
        if "startDateEastern" in prev.columns:
            prev["game_date"] = _to_date_yyyy_mm_dd(prev["startDateEastern"])
        else:
            log.warning("Previous weekly schedule missing startDateEastern; skipping past-rank preservation.")
            return None
        return prev
    except Exception as e:
        log.warning("Could not read previous weekly schedule: %s", e)
        return None


# ----------------------------- Rankings -----------------------------
def _load_latest_rankings_csv(
    weekly_dir: Path,
    preferred_year: Optional[int] = None,
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """Load the most recent rankings CSV; prefer the given year, then prior year when needed."""
    paths = [Path(p) for p in glob.glob(str(weekly_dir / "*rankings*.csv"))]
    if not paths:
        log.warning("No rankings CSVs found in %s (looking for *rankings*.csv).", weekly_dir)
        return None, None

    if preferred_year is not None:
        year_paths = [p for p in paths if str(preferred_year) in p.name]
        if year_paths:
            paths = year_paths
        else:
            prev_year = preferred_year - 1
            prev_paths = [p for p in paths if str(prev_year) in p.name]
            if prev_paths:
                log.info("No rankings found for %s; falling back to %s.", preferred_year, prev_year)
                paths = prev_paths

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df is not None and not df.empty:
                log.info("Using rankings file → %s", p.name)
                return df, p
        except Exception as e:
            log.warning("Failed reading %s: %s", p.name, e)
    log.warning("Could not load any usable rankings CSV in %s.", weekly_dir)
    return None, None


def _load_rankings_history_csv(
    weekly_dir: Path,
    preferred_year: Optional[int] = None,
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """Load historical rankings CSV; prefer given year, then prior year when needed."""
    paths = [Path(p) for p in glob.glob(str(weekly_dir / "rankings_history_*.csv"))]
    if not paths:
        return None, None

    if preferred_year is not None:
        year_paths = [p for p in paths if str(preferred_year) in p.name]
        if year_paths:
            paths = year_paths
        else:
            prev_year = preferred_year - 1
            prev_paths = [p for p in paths if str(prev_year) in p.name]
            if prev_paths:
                log.info("No rankings history for %s; falling back to %s.", preferred_year, prev_year)
                paths = prev_paths

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df is not None and not df.empty:
                log.info("Using rankings history file → %s", p.name)
                return df, p
        except Exception as e:
            log.warning("Failed reading %s: %s", p.name, e)
    return None, None


def _clean_school_name(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\[.*?\]", "", str(s))               # footnotes
    s = re.sub(r"\(\s*\d+(?:[-\s]\d+)?\s*\)", "", s)  # votes/records
    s = re.sub(r"\s*\([^)]*\)", "", s)               # parentheticals
    s = s.replace("—", " ").replace("–", " ")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s


def _build_rank_lookup(
    weekly_dir: Path,
    preferred_year: Optional[int] = None,
) -> tuple[Dict[str, int], Optional[Path]]:
    rankings_raw, rankings_path = _load_latest_rankings_csv(weekly_dir, preferred_year=preferred_year)
    if rankings_raw is None or rankings_raw.empty:
        return {}, rankings_path

    col_map = {
        "RK": "rank", "Rank": "rank", "#": "rank",
        "TEAM": "school", "Team": "school", "School": "school", "team": "school",
    }
    rankings_raw = rankings_raw.rename(columns={k: v for k, v in col_map.items() if k in rankings_raw.columns})
    if not {"rank", "school"}.issubset(rankings_raw.columns):
        log.warning("Rankings CSV missing required columns 'rank' and 'school'; skipping rankings merge.")
        return {}, rankings_path

    rankings_df = rankings_raw[["rank", "school"]].copy()
    rankings_df["rank"] = pd.to_numeric(rankings_df["rank"], errors="coerce")
    rankings_df = rankings_df.dropna(subset=["rank"])
    rankings_df["rank"] = rankings_df["rank"].astype(int)
    rankings_df["school"] = rankings_df["school"].astype(str)

    rankings_df["school_clean"] = rankings_df["school"].map(_clean_school_name)
    rankings_df["school_alias"] = rankings_df["school_clean"].map(_canon)

    return dict(zip(rankings_df["school_alias"].astype(str), rankings_df["rank"].astype(int))), rankings_path


def _build_rank_history_maps(
    history_df: Optional[pd.DataFrame],
) -> tuple[Dict[int, Dict[str, int]], Optional[int]]:
    """Return {week -> {school_alias -> rank}} and latest available week."""
    if history_df is None or history_df.empty:
        return {}, None

    df = history_df.copy()
    if "week" not in df.columns or "school" not in df.columns or "rank" not in df.columns:
        return {}, None

    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["week"])
    df["school_clean"] = df["school"].map(_clean_school_name)
    df["school_alias"] = df["school_clean"].map(_canon)

    if "poll" in df.columns:
        df["poll"] = df["poll"].astype(str)
        df = df[df["poll"].isin(["CFP", "AP"])]

    if df.empty:
        return {}, None

    # Build poll-specific maps
    poll_maps: dict[tuple[str, int], dict[str, int]] = {}
    for poll in ["CFP", "AP"]:
        subset = df[df.get("poll") == poll] if "poll" in df.columns else df
        for wk, g in subset.groupby("week"):
            poll_maps[(poll, int(wk))] = dict(
                zip(g["school_alias"].astype(str), g["rank"].astype(int))
            )

    rank_by_week: dict[int, dict[str, int]] = {}
    for wk in sorted({int(w) for w in df["week"].dropna().unique()}):
        if ("CFP", wk) in poll_maps:
            rank_by_week[wk] = poll_maps[("CFP", wk)]
        elif ("AP", wk) in poll_maps:
            rank_by_week[wk] = poll_maps[("AP", wk)]

    latest_week = max(rank_by_week.keys()) if rank_by_week else None
    return rank_by_week, latest_week


# ----------------------------- Merge steps -----------------------------
def _fetch_schedule(year: int) -> tuple[pd.DataFrame, int]:
    log.info("Fetching schedule…")
    fetcher = ScheduleFetcher(year)
    sched = fetcher.fetch()
    if sched is None or sched.empty:
        raise RuntimeError("Schedule fetch returned no data. Check CFD_API_KEY and network access.")
    year_used = fetcher.year_used or year
    return sched.copy(), int(year_used)


def _ensure_completed_games(year: int) -> None:
    """Create completed_games_<year>.csv if missing (for point diff merge)."""
    path = WEEKLY_DIR / f"completed_games_{year}.csv"
    if path.exists():
        return
    try:
        log.info("Fetching completed games for %s (point diffs)…", year)
        save_completed_fbs_games(year)
    except Exception as e:
        log.warning("Could not fetch completed games for %s: %s", year, e)


def _ensure_venues(year: int) -> None:
    """Create venues_<year>.csv if missing (for alternate venue capacities)."""
    path = ANNUAL_DIR / f"venues_{year}.csv"
    if path.exists():
        return
    try:
        log.info("Fetching venues for %s (alternate capacity lookup)…", year)
        save_venues(year)
    except Exception as e:
        log.warning("Could not fetch venues for %s: %s", year, e)


def _fetch_rankings(year: int) -> None:
    log.info("Fetching rankings (Wikipedia only; CFP→AP, current→prior)…")
    try:
        rf = RankingsFetchers(year=year)
        fetched_df = rf.fetch_current_then_prior_cfp_ap()
        if fetched_df is None or fetched_df.empty:
            log.warning("No fresh rankings parsed; will use the most recent local *rankings*.csv.")
            return
        src = rf.source or "wiki:AP"
        sy = rf.source_year or year
        try:
            top1 = str(fetched_df.loc[fetched_df["rank"].idxmin(), "school"])
        except Exception:
            top1 = "unknown"
        log.info("Rankings fetched from %s (%s); #1 = %s", src, sy, top1)
    except Exception as e:
        log.warning("Rankings fetch failed: %s — proceeding with local *rankings*.csv if available.", e)


def _fetch_rankings_history_api(year: int) -> None:
    log.info("Fetching historical rankings (CFBD /rankings)…")
    try:
        rf = RankingsApiFetcher(year=year, season_type="both")
        df = rf.fetch()
        if df is None or df.empty:
            log.warning("No historical rankings returned; will fall back to latest rankings file.")
            return
        rf.write_artifact(df)
    except Exception as e:
        log.warning("Historical rankings fetch failed: %s — proceeding without history.", e)


def _apply_rankings(
    schedule_df: pd.DataFrame,
    prev_df: Optional[pd.DataFrame],
    rank_lookup: Dict[str, int],
    rank_by_week: Optional[Dict[int, Dict[str, int]]] = None,
    latest_week: Optional[int] = None,
) -> pd.DataFrame:
    today_d = date.today()
    schedule_df["game_date"] = _to_date_yyyy_mm_dd(schedule_df["startDateEastern"])
    is_past = schedule_df["game_date"] < today_d
    is_now_or_future = ~is_past

    # Offseason: if every game is in the past, apply ranks to all rows.
    max_game_date = schedule_df["game_date"].max()
    is_offseason = pd.notna(max_game_date) and max_game_date < today_d
    if is_offseason:
        log.info("All games are in the past; applying rankings to all rows (offseason behavior).")
        is_now_or_future = schedule_df["game_date"].notna()

    for col in ["homeTeamRank", "awayTeamRank"]:
        if col not in schedule_df.columns:
            schedule_df[col] = pd.NA

    # If weekly history is available, use it for past games and latest week for future games.
    if rank_by_week:
        if latest_week is None:
            latest_week = max(rank_by_week.keys()) if rank_by_week else None

        # Past games: use that week's poll
        for wk, idx in schedule_df.loc[is_past].groupby("week").groups.items():
            try:
                wk_int = int(wk)
            except Exception:
                continue
            wk_map = rank_by_week.get(wk_int)
            if not wk_map:
                continue
            schedule_df.loc[idx, "homeTeamRank"] = schedule_df.loc[idx, "homeTeam"].map(wk_map)
            schedule_df.loc[idx, "awayTeamRank"] = schedule_df.loc[idx, "awayTeam"].map(wk_map)

        # Current/future games: use latest available poll week
        if latest_week is not None and latest_week in rank_by_week:
            wk_map = rank_by_week[latest_week]
            schedule_df.loc[is_now_or_future, "homeTeamRank"] = schedule_df.loc[is_now_or_future, "homeTeam"].map(wk_map)
            schedule_df.loc[is_now_or_future, "awayTeamRank"] = schedule_df.loc[is_now_or_future, "awayTeam"].map(wk_map)

        # Fill any remaining past gaps from previous snapshot if available
        required_prev_cols = {"homeTeamRank", "awayTeamRank", "homeTeam", "awayTeam", "game_date"}
        if prev_df is not None and required_prev_cols.issubset(prev_df.columns):
            prev_past = prev_df.loc[prev_df["game_date"] < today_d, [
                "homeTeam", "awayTeam", "game_date", "homeTeamRank", "awayTeamRank"
            ]]
            schedule_df = schedule_df.merge(
                prev_past,
                on=["homeTeam", "awayTeam", "game_date"],
                how="left",
                suffixes=("", "_prev"),
            )
            if "homeTeamRank_prev" in schedule_df.columns:
                schedule_df["homeTeamRank"] = schedule_df["homeTeamRank"].mask(is_past, schedule_df["homeTeamRank_prev"])
            if "awayTeamRank_prev" in schedule_df.columns:
                schedule_df["awayTeamRank"] = schedule_df["awayTeamRank"].mask(is_past, schedule_df["awayTeamRank_prev"])
            schedule_df.drop(columns=["homeTeamRank_prev", "awayTeamRank_prev"], inplace=True, errors="ignore")
        schedule_df["isRankedMatchup"] = schedule_df["homeTeamRank"].notna() & schedule_df["awayTeamRank"].notna()
        return schedule_df

    if rank_lookup:
        # Update for current + future using the latest lookup
        home_new = schedule_df["homeTeam"].map(rank_lookup)
        away_new = schedule_df["awayTeam"].map(rank_lookup)
        schedule_df["homeTeamRank"] = schedule_df["homeTeamRank"].mask(is_now_or_future, home_new)
        schedule_df["awayTeamRank"] = schedule_df["awayTeamRank"].mask(is_now_or_future, away_new)
    else:
        log.warning("Rankings lookup is empty; leaving current/future ranks as-is.")

    schedule_df["isRankedMatchup"] = schedule_df["homeTeamRank"].notna() & schedule_df["awayTeamRank"].notna()
    return schedule_df


def _merge_stadium_capacity(
    schedule_df: pd.DataFrame,
    stadium_candidates: List[Path],
    venues_candidates: List[Path],
) -> tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
    log.info("Merging stadium capacities…")
    stadiums_path = _first_existing_path(stadium_candidates)
    if not stadiums_path or not stadiums_path.exists():
        log.info("No stadiums file found; leaving capacity blank.")
        schedule_df["capacity"] = pd.NA
        return schedule_df, None, None
    log.info("Using stadiums file → %s", stadiums_path.name)

    stad = pd.read_csv(stadiums_path).copy()
    if "stadium" not in stad.columns:
        raise KeyError(f"{stadiums_path.name} is missing required 'stadium' column")
    if "capacity" not in stad.columns:
        raise KeyError(f"{stadiums_path.name} is missing required 'capacity' column")

    venue_col = _pick_venue_column(schedule_df)
    # Normalize school names if present
    if "school" in stad.columns:
        stad["school"] = stad["school"].apply(_canon)
    else:
        stad["school"] = pd.NA

    # 1) Direct venue match (normalized)
    stad["stadium_norm"] = stad["stadium"].map(_norm_key)
    stad_lookup = dict(zip(stad["stadium_norm"], stad["capacity"]))
    schedule_df["capacity"] = schedule_df[venue_col].map(lambda v: stad_lookup.get(_norm_key(v), pd.NA))

    # 1b) Loose match for alternate venue naming
    need = schedule_df["capacity"].isna()
    if need.any():
        stad["stadium_loose"] = stad["stadium"].map(_norm_key_loose)
        counts = stad["stadium_loose"].value_counts()
        unique_loose = counts[counts == 1].index
        loose_lookup = dict(
            zip(
                stad[stad["stadium_loose"].isin(unique_loose)]["stadium_loose"],
                stad[stad["stadium_loose"].isin(unique_loose)]["capacity"],
            )
        )
        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, venue_col].map(
            lambda v: loose_lookup.get(_norm_key_loose(v), pd.NA)
        )

    # 1c) Token-subset match (only when unique)
    need = schedule_df["capacity"].isna()
    if need.any():
        stad_tokens = stad[["stadium_norm", "capacity"]].copy()
        stad_tokens["tokens"] = stad_tokens["stadium_norm"].map(lambda s: set(s.split()))

        def _subset_match(name: object) -> object:
            tokens = set(_norm_key_loose(name).split())
            if not tokens:
                return pd.NA
            candidates = stad_tokens[stad_tokens["tokens"].map(lambda t: tokens.issubset(t) or t.issubset(tokens))]
            if len(candidates) == 1:
                return candidates["capacity"].iloc[0]
            return pd.NA

        schedule_df.loc[need, "capacity"] = schedule_df.loc[need, venue_col].map(_subset_match)

    def _fill_by_school(col_name: str, fill_col: str, mask: pd.Series) -> pd.DataFrame:
        nonlocal schedule_df
        need_mask = schedule_df["capacity"].isna() & mask
        if not need_mask.any():
            return schedule_df
        school_cap = (
            stad.dropna(subset=["school"])
            .groupby("school", dropna=True, as_index=False)["capacity"].max()
            .rename(columns={"capacity": fill_col})
        )
        schedule_df = schedule_df.merge(
            school_cap, left_on=col_name, right_on="school", how="left"
        ).drop(columns=["school"])
        schedule_df.loc[need_mask, "capacity"] = schedule_df.loc[need_mask, fill_col]
        schedule_df = schedule_df.drop(columns=[fill_col])
        return schedule_df

    # 2) Alternate/neutral venues (from /venues)
    venues_path = _first_existing_path(venues_candidates)
    if venues_path and venues_path.exists():
        log.info("Using venues file → %s", venues_path.name)
        venues = pd.read_csv(venues_path)
        name_col = next((c for c in ["name", "venue", "stadium", "venue_name"] if c in venues.columns), None)
        cap_col = next((c for c in ["capacity"] if c in venues.columns), None)
        if name_col and cap_col:
            venues["venue_norm"] = venues[name_col].map(_norm_key)
            venues_lookup = dict(zip(venues["venue_norm"], venues[cap_col]))
            need = schedule_df["capacity"].isna()
            schedule_df.loc[need, "capacity"] = schedule_df.loc[need, venue_col].map(
                lambda v: venues_lookup.get(_norm_key(v), pd.NA)
            )
            # loose match
            need = schedule_df["capacity"].isna()
            if need.any():
                venues["venue_loose"] = venues[name_col].map(_norm_key_loose)
                counts = venues["venue_loose"].value_counts()
                unique_loose = counts[counts == 1].index
                loose_lookup = dict(
                    zip(
                        venues[venues["venue_loose"].isin(unique_loose)]["venue_loose"],
                        venues[venues["venue_loose"].isin(unique_loose)][cap_col],
                    )
                )
                schedule_df.loc[need, "capacity"] = schedule_df.loc[need, venue_col].map(
                    lambda v: loose_lookup.get(_norm_key_loose(v), pd.NA)
                )
        else:
            log.warning("Venues file missing name/capacity columns; skipping venues merge.")
    else:
        venues_path = None

    # 3) Home/away fallback (avoid neutral sites if column exists)
    non_neutral = schedule_df["neutralSite"] == False if "neutralSite" in schedule_df.columns else schedule_df["capacity"].isna()
    schedule_df = _fill_by_school("homeTeam", "cap_by_home", non_neutral)
    schedule_df = _fill_by_school("awayTeam", "cap_by_away", non_neutral)

    return schedule_df, stadiums_path, venues_path


# ----------------------------- Main flow -----------------------------
def main() -> None:
    # Ensure completed games files exist for current/prior year (point diff merge).
    _ensure_completed_games(DEFAULT_YEAR)
    _ensure_completed_games(DEFAULT_YEAR - 1)

    schedule_df, schedule_year = _fetch_schedule(DEFAULT_YEAR)
    if "startDateEastern" not in schedule_df.columns:
        raise KeyError("schedule_df is missing 'startDateEastern'")

    weekly_schedule_out = WEEKLY_DIR / f"full_{schedule_year}_schedule.csv"
    stadiums_path = _find_stadiums_file(schedule_year)
    stadium_candidates = [stadiums_path] if stadiums_path else []
    _ensure_venues(schedule_year)
    venues_path = _find_venues_file(schedule_year)
    venues_candidates = [venues_path] if venues_path else []
    log.info("Using schedule year: %s", schedule_year)
    log.info("WEEKLY_SCHEDULE_OUT: %s", weekly_schedule_out)

    prev_df = _load_previous_weekly(weekly_schedule_out)

    schedule_df["homeTeam"] = schedule_df["homeTeam"].apply(_canon)
    schedule_df["awayTeam"] = schedule_df["awayTeam"].apply(_canon)
    if "seasonType" in schedule_df.columns:
        is_post = schedule_df["seasonType"].astype(str).str.lower().eq("postseason")
        schedule_df.loc[is_post, "week"] = 17
    _ensure_week_column(schedule_df)
    schedule_df = _apply_prev_game_point_diff(schedule_df)

    _fetch_rankings(schedule_year)  # writes artifact if it succeeds
    _fetch_rankings_history_api(schedule_year)  # weekly history if API available
    log.info("Loading rankings from local weekly CSVs…")
    rank_lookup, rankings_path = _build_rank_lookup(WEEKLY_DIR, preferred_year=schedule_year)
    history_df, history_path = _load_rankings_history_csv(WEEKLY_DIR, preferred_year=schedule_year)
    rank_by_week, latest_week = _build_rank_history_maps(history_df)
    schedule_df = _apply_rankings(schedule_df, prev_df, rank_lookup, rank_by_week, latest_week)

    schedule_df, stadiums_path, venues_path = _merge_stadium_capacity(
        schedule_df, stadium_candidates, venues_candidates
    )

    log.info("Saving weekly schedule snapshot…")
    _atomic_csv_write(schedule_df, weekly_schedule_out)
    log.info("Weekly schedule saved: %s", weekly_schedule_out)

    if VERBOSE:
        try:
            n = len(schedule_df)
            date_min = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").min()
            date_max = pd.to_datetime(schedule_df.get("startDateEastern"), errors="coerce").max()
            log.info(
                "[weekly_update] summary: year=%s rows=%d date_range=%s→%s rankings=%s stadiums=%s",
                schedule_year,
                n,
                date_min.date() if pd.notnull(date_min) else "NA",
                date_max.date() if pd.notnull(date_max) else "NA",
                history_path.name if history_path else (rankings_path.name if rankings_path else "none"),
                stadiums_path.name if stadiums_path else "none",
            )
        except Exception as e:
            log.warning("[weekly_update] post-write summary failed: %s", e)

    _post_run_report(schedule_df, weekly_schedule_out, DEFAULT_YEAR, schedule_year)
    _cleanup_weekly_files(WEEKLY_DIR, [weekly_schedule_out, WEEKLY_DIR / f"rankings_{schedule_year}.csv"])
    write_status(
        "weekly_update",
        "success",
        f"Weekly update complete for {schedule_year}",
        {
            "year": int(schedule_year),
            "rows": int(len(schedule_df)),
            "output": str(weekly_schedule_out),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = f"STATUS: FAIL | reason={e}"
        if QUIET:
            print(msg)
        else:
            log.error(msg)
        write_status("weekly_update", "failed", str(e))
        raise
