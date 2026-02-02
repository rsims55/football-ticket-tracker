"""Fetch FBS schedule data from CFBD and enrich with last point differentials."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Allow running as a script without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.http import build_session
from utils.logging_utils import get_logger

# --- Environment ---
load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")
API_BASE = os.getenv("CFD_API_BASE", "https://api.collegefootballdata.com").rstrip("/")
# Optional override for multiple base URLs (comma-separated).
API_BASES = [b.strip().rstrip("/") for b in os.getenv("CFD_API_BASES", "").split(",") if b.strip()]
# seasonType: regular | postseason | both (auto-merge). Default to both.
SEASON_TYPE = os.getenv("CFD_SEASON_TYPE", "both").strip().lower()
# classification: fbs/fcs (used for filter and optionally as query param)
CLASSIFICATION = os.getenv("CFD_CLASSIFICATION", "fbs").strip().lower()
# include past games (default: future only)
INCLUDE_PAST = os.getenv("CFD_INCLUDE_PAST", "0") == "1"
LOG = get_logger("schedule_fetcher")

WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)


def _load_last_point_diffs(year: int) -> dict[str, float]:
    """
    Build {team -> last_point_diff} from data/weekly/completed_games_<YEAR>.csv.
    Uses the most recent COMPLETED game as of 'now' (UTC). If file missing or empty,
    returns {} and the caller will simply not add the columns.
    """
    path = os.path.join(WEEKLY_DIR, f"completed_games_{year}.csv")
    if not os.path.exists(path):
        LOG.info("No completed games file found at %s. Skipping last_point_diff merge.", path)
        return {}

    df = pd.read_csv(path)

    # Normalize date column
    if "startDate" in df.columns:
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    else:
        df["startDate"] = pd.NaT

    required = {"homeTeam", "awayTeam", "homePointDiff", "awayPointDiff", "startDate"}
    missing = required - set(df.columns)
    if missing:
        LOG.warning("completed_games CSV missing columns %s; skipping merge.", sorted(missing))
        return {}

    # Only games up to now (no leakage)
    now_utc = datetime.now(timezone.utc)
    df = df[df["startDate"].notna() & (df["startDate"] <= now_utc)].copy()
    if df.empty:
        return {}

    # Long form (signed from the team's perspective)
    long_home = df.rename(
        columns={"homeTeam": "team", "awayTeam": "opponent", "homePointDiff": "pointDiff"}
    )[["team", "opponent", "startDate", "pointDiff"]].copy()
    long_away = df.rename(
        columns={"awayTeam": "team", "homeTeam": "opponent", "awayPointDiff": "pointDiff"}
    )[["team", "opponent", "startDate", "pointDiff"]].copy()
    long_df = pd.concat([long_home, long_away], ignore_index=True)

    if long_df.empty:
        return {}

    # Pick the most recent game per team (stable + pandas-version-proof)
    latest = (
        long_df.sort_values(["team", "startDate"])
               .drop_duplicates(subset=["team"], keep="last")
               [["team", "pointDiff"]]
               .copy()
    )

    return dict(zip(latest["team"], latest["pointDiff"]))


class ScheduleFetcher:
    def __init__(self, year=None):
        self.year = year or datetime.now().year
        self.year_used: int | None = None
        self.schedule = None
        self.session = build_session()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common alternative column names to the canonical ones used downstream."""
        if df.empty:
            return df

        rename_map: dict[str, str] = {}

        # Date/time fields
        date_alts = ["start_date", "startDateUtc", "startDateUTC", "start_date_utc", "startDateTime"]
        for alt in date_alts:
            if "startDate" not in df.columns and alt in df.columns:
                rename_map[alt] = "startDate"

        # Team fields
        if "homeTeam" not in df.columns:
            for alt in ["home_team", "home"]:
                if alt in df.columns:
                    rename_map[alt] = "homeTeam"
                    break
        if "awayTeam" not in df.columns:
            for alt in ["away_team", "away"]:
                if alt in df.columns:
                    rename_map[alt] = "awayTeam"
                    break

        # Classification/division fields
        if "homeClassification" not in df.columns:
            for alt in ["home_classification", "homeDivision", "home_division"]:
                if alt in df.columns:
                    rename_map[alt] = "homeClassification"
                    break
        if "awayClassification" not in df.columns:
            for alt in ["away_classification", "awayDivision", "away_division"]:
                if alt in df.columns:
                    rename_map[alt] = "awayClassification"
                    break

        # Kickoff TBD field
        if "startTimeTBD" not in df.columns:
            for alt in ["start_time_tbd", "start_time_tba", "startTimeTba"]:
                if alt in df.columns:
                    rename_map[alt] = "startTimeTBD"
                    break

        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _base_candidates(self) -> list[str]:
        if API_BASES:
            return API_BASES
        return [API_BASE]

    def _param_variants(self, year: int, season_type: str | None, week: int | None) -> Iterable[dict[str, object]]:
        """Generate param combinations in a stable order to avoid ambiguous failures."""
        base: dict[str, object] = {"year": year}
        if week is not None:
            base["week"] = week
        if season_type:
            base["seasonType"] = season_type
        if CLASSIFICATION:
            base["classification"] = CLASSIFICATION

        # Primary (CFBD v1/v2 REST): year + seasonType + classification
        yield dict(base)

        # Alternate: use division instead of classification (some APIs use 'division').
        if CLASSIFICATION:
            alt = dict(base)
            alt.pop("classification", None)
            alt["division"] = CLASSIFICATION
            yield alt

    def _fetch_games(self, year: int, season_type: str | None, week: int | None) -> pd.DataFrame | None:
        if not API_KEY:
            LOG.error("CFD_API_KEY is not set. Set it in your environment or .env before fetching schedules.")
            return None

        headers = {"Authorization": f"Bearer {API_KEY}"}

        for base in self._base_candidates():
            url = f"{base}/games"
            for params in self._param_variants(year, season_type, week):
                response = self.session.get(url, headers=headers, params=params, timeout=30)
                if response.status_code != 200:
                    body = response.text.strip()
                    LOG.error(
                        "Failed to fetch schedule: %s (base=%s params=%s url=%s)",
                        response.status_code,
                        base,
                        params,
                        response.url,
                    )
                    if body:
                        LOG.error("Response body: %s", body[:500])
                    continue

                try:
                    data = response.json()
                except Exception as e:
                    body = response.text.strip()
                    LOG.error(
                        "Non-JSON response from schedule API: %s (base=%s params=%s url=%s)",
                        e,
                        base,
                        params,
                        response.url,
                    )
                    if body:
                        LOG.error("Response body: %s", body[:500])
                    continue
                if isinstance(data, dict):
                    LOG.warning("Schedule API response: %s", str(data)[:500])
                    continue

                df = pd.DataFrame(data)
                if df.empty:
                    LOG.warning(
                        "Empty schedule response for year=%s seasonType=%s (base=%s params=%s url=%s)",
                        year,
                        season_type,
                        base,
                        params,
                        response.url,
                    )
                    continue

                df = self._normalize_columns(df)
                if "startDate" not in df.columns:
                    LOG.warning(
                        "No startDate field found in schedule response. Columns: %s (base=%s params=%s url=%s)",
                        list(df.columns),
                        base,
                        params,
                        response.url,
                    )
                    continue
                return df

        return None

    def _dedupe_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """De-duplicate games without touching list/dict columns."""
        if df.empty:
            return df
        for key in ("id", "gameId"):
            if key in df.columns:
                return df.drop_duplicates(subset=[key])

        key_cols = [c for c in ("season", "week", "seasonType", "homeTeam", "awayTeam", "startDate") if c in df.columns]
        if key_cols:
            return df.drop_duplicates(subset=key_cols)

        # Fallback: drop duplicates on hashable columns only.
        hashable_cols: list[str] = []
        for c in df.columns:
            sample = df[c].dropna().head(10)
            if sample.empty:
                hashable_cols.append(c)
                continue
            if any(isinstance(v, (list, dict, set)) for v in sample):
                continue
            hashable_cols.append(c)
        return df.drop_duplicates(subset=hashable_cols) if hashable_cols else df

    def fetch(self, week=None):
        if not API_KEY:
            LOG.error("CFD_API_KEY is not set. Set it in your environment or .env before fetching schedules.")
            return None

        def _fetch_for_year(year: int) -> pd.DataFrame | None:
            season_type = SEASON_TYPE
            dfs: list[pd.DataFrame] = []

            if season_type in ("both", "all"):
                for st in ("regular", "postseason"):
                    df_part = self._fetch_games(year, st, week)
                    if df_part is not None and not df_part.empty:
                        dfs.append(df_part)
            else:
                df_part = self._fetch_games(year, season_type, week)
                if df_part is not None and not df_part.empty:
                    dfs.append(df_part)
                else:
                    # Optional fallback: if regular is empty, try postseason.
                    if season_type == "regular":
                        post = self._fetch_games(year, "postseason", week)
                        if post is not None and not post.empty:
                            dfs.append(post)

            if not dfs:
                return None
            merged = pd.concat(dfs, ignore_index=True)
            return self._dedupe_games(merged)

        df = _fetch_for_year(self.year)
        year_used = self.year
        if df is None or df.empty:
            prev_year = self.year - 1
            LOG.warning("No schedule data for %s; falling back to %s.", self.year, prev_year)
            df = _fetch_for_year(prev_year)
            year_used = prev_year if df is not None and not df.empty else self.year

        if df is None or df.empty:
            return None

        self.year_used = year_used
        if "season" not in df.columns:
            df["season"] = year_used

        # --- your existing logic (unchanged) ---
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
        if not INCLUDE_PAST:
            future_mask = df["startDate"] > pd.Timestamp.now(tz="UTC")
            df_future = df[future_mask]
            if df_future.empty:
                LOG.warning(
                    "All games are in the past for year %s; keeping past games. "
                    "Set CFD_INCLUDE_PAST=1 to suppress this behavior.",
                    year_used,
                )
            else:
                df = df_future
        if "homeClassification" in df.columns and CLASSIFICATION:
            df = df[df["homeClassification"].str.lower() == CLASSIFICATION]

        df = df[df["startDate"].notnull()]
        # Keep full Eastern timestamp (not just date) to preserve kickoff time precision.
        df["startDateEastern"] = df["startDate"].dt.tz_convert("US/Eastern")

        # Guard if column is missing in some rows/season responses
        if "startTimeTBD" not in df.columns:
            df["startTimeTBD"] = False

        # Force postseason games to week 17 for downstream consistency.
        if "seasonType" in df.columns:
            is_post = df["seasonType"].astype(str).str.lower().eq("postseason")
            df.loc[is_post, "week"] = 17

        df["dayOfWeek"] = pd.to_datetime(df["startDate"], errors='coerce').dt.tz_convert("US/Eastern").dt.day_name()
        df["kickoffTimeStr"] = np.where(
            df["startTimeTBD"],
            "",
            pd.to_datetime(df["startDate"], errors='coerce').dt.tz_convert("US/Eastern").dt.strftime("%-I:%M %p")
        )
        # --- end of your existing logic ---

        # --- minimal addition: merge last point differentials ---
        last_diff_map = _load_last_point_diffs(year_used)
        if last_diff_map:
            df["home_last_point_diff"] = df["homeTeam"].map(last_diff_map)
            df["away_last_point_diff"] = df["awayTeam"].map(last_diff_map)
        else:
            # Keep columns consistent even if missing (filled with NaN)
            df["home_last_point_diff"] = np.nan
            df["away_last_point_diff"] = np.nan

        self.schedule = df
        return df

    def save(self, filename=None):
        if self.schedule is not None:
            if filename is None:
                filename = f"data/full_{self.year}_schedule.csv"
            self.schedule.to_csv(filename, index=False)
            LOG.info("Schedule saved to %s", filename)
        else:
            LOG.warning("No schedule to save")


# 🧪 Test run
if __name__ == "__main__":
    sf = ScheduleFetcher()
    df = sf.fetch()
    if df is not None:
        LOG.info(
            "\n%s",
            df[["homeTeam", "awayTeam", "startDateEastern", "dayOfWeek",
                "home_last_point_diff", "away_last_point_diff"]]
            .head()
            .to_string(index=False),
        )
        sf.save()
