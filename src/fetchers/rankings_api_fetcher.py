"""Fetch historical weekly rankings via CFBD /rankings API."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv

# Allow running as a script without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.http import build_session
from utils.logging_utils import get_logger

load_dotenv()

API_KEY = os.getenv("CFD_API_KEY")
API_BASE = os.getenv("CFD_API_BASE", "https://api.collegefootballdata.com").rstrip("/")
API_BASES = [b.strip().rstrip("/") for b in os.getenv("CFD_API_BASES", "").split(",") if b.strip()]
LOG = get_logger("rankings_api_fetcher")

WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)


def _poll_label(poll_name: str | None) -> str:
    if not poll_name:
        return "Unknown"
    low = poll_name.lower()
    if "playoff" in low or "cfp" in low:
        return "CFP"
    if "ap" in low:
        return "AP"
    if "coaches" in low:
        return "Coaches"
    return poll_name.strip()


class RankingsApiFetcher:
    def __init__(self, year: Optional[int] = None, season_type: str = "both"):
        self.year = year or datetime.now().year
        self.season_type = season_type
        self.session = build_session()

    def _base_candidates(self) -> list[str]:
        return API_BASES if API_BASES else [API_BASE]

    def _fetch_raw(self) -> Optional[list[dict[str, Any]]]:
        if not API_KEY:
            LOG.error("CFD_API_KEY is not set; cannot fetch rankings history.")
            return None

        headers = {"Authorization": f"Bearer {API_KEY}"}
        params: dict[str, Any] = {"year": self.year}
        if self.season_type:
            params["seasonType"] = self.season_type

        for base in self._base_candidates():
            url = f"{base}/rankings"
            resp = self.session.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                LOG.error("Rankings API failed: %s (base=%s params=%s)", resp.status_code, base, params)
                if resp.text:
                    LOG.error("Response body: %s", resp.text[:500])
                continue
            try:
                data = resp.json()
            except Exception as e:
                LOG.error("Non-JSON response from rankings API: %s", e)
                if resp.text:
                    LOG.error("Response body: %s", resp.text[:500])
                continue
            if not isinstance(data, list):
                LOG.warning("Unexpected rankings payload type: %s", type(data))
                return None
            return data
        return None

    def _flatten(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for wk in data:
            season = wk.get("season", self.year)
            season_type = wk.get("seasonType") or wk.get("season_type")
            week = wk.get("week") or wk.get("weekNumber")
            polls = wk.get("polls") or []
            for poll in polls:
                poll_name = poll.get("poll") or poll.get("pollName") or poll.get("name")
                label = _poll_label(poll_name)
                ranks = poll.get("ranks") or poll.get("rankings") or []
                for r in ranks:
                    rank = r.get("rank")
                    school = r.get("school") or r.get("team")
                    if rank is None or not school:
                        continue
                    rows.append(
                        {
                            "season": season,
                            "season_type": season_type,
                            "week": week,
                            "poll": label,
                            "rank": rank,
                            "school": school,
                            "conference": r.get("conference"),
                            "first_place_votes": r.get("firstPlaceVotes") or r.get("first_place_votes"),
                            "points": r.get("points"),
                        }
                    )
        return pd.DataFrame(rows)

    def fetch(self) -> Optional[pd.DataFrame]:
        data = self._fetch_raw()
        if not data:
            return None
        df = self._flatten(data)
        if df.empty:
            return None
        return df

    def write_artifact(self, df: pd.DataFrame) -> str:
        out = os.path.join(WEEKLY_DIR, f"rankings_history_{self.year}.csv")
        df.to_csv(out, index=False, encoding="utf-8")
        LOG.info("[rankings_api] wrote artifact → %s (%d rows)", out, len(df))
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical rankings via CFBD /rankings API.")
    parser.add_argument("--year", type=int, default=None, help="Season year (defaults to current year)")
    parser.add_argument("--season-type", type=str, default="both", help="seasonType query param (default: both)")
    args = parser.parse_args()

    fetcher = RankingsApiFetcher(year=args.year, season_type=args.season_type)
    df = fetcher.fetch()
    if df is None or df.empty:
        LOG.error("No rankings data returned.")
        raise SystemExit(1)
    fetcher.write_artifact(df)
