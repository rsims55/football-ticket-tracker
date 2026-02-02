"""Fetch completed FBS games and point differentials from CFBD."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

LOG = get_logger("results_fetcher")

OUT_DIR = os.path.join("data", "weekly")
os.makedirs(OUT_DIR, exist_ok=True)


def _base_candidates() -> list[str]:
    return API_BASES if API_BASES else [API_BASE]


def fetch_completed_fbs_games(year: int) -> pd.DataFrame:
    """
    Pull ALL games for the season (regular + postseason),
    keep only rows with both scores present (i.e., completed),
    then filter to FBS-vs-FBS using home/awayClassification.
    Compute signed point differentials.
    """
    if not API_KEY:
        raise RuntimeError("CFD_API_KEY is not set; cannot fetch completed games.")

    headers = {"Authorization": f"Bearer {API_KEY}"}
    params: dict[str, Any] = {"year": year, "seasonType": "both"}

    data: list[dict[str, Any]] | None = None
    session = build_session()
    for base in _base_candidates():
        url = f"{base}/games"
        r = session.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            LOG.error("Games API failed: %s (base=%s params=%s)", r.status_code, base, params)
            if r.text:
                LOG.error("Response body: %s", r.text[:500])
            continue
        try:
            data = r.json()
        except Exception as e:
            LOG.error("Non-JSON response from games API: %s", e)
            if r.text:
                LOG.error("Response body: %s", r.text[:500])
            continue
        break

    if data is None:
        raise RuntimeError("Failed to fetch completed games from CFBD.")

    df = pd.DataFrame(data)

    # Normalize timestamp
    if "startDate" in df.columns:
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    elif "start_date" in df.columns:
        df["startDate"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    else:
        df["startDate"] = pd.NaT

    # Standardize score columns
    if "home_points" in df.columns and "homePoints" not in df.columns:
        df["homePoints"] = df["home_points"]
    if "away_points" in df.columns and "awayPoints" not in df.columns:
        df["awayPoints"] = df["away_points"]

    # Only completed games
    if "homePoints" not in df.columns or "awayPoints" not in df.columns:
        LOG.warning("homePoints/awayPoints missing in games API response; returning empty dataframe.")
        return df.head(0)
    mask_done = df["homePoints"].notna() & df["awayPoints"].notna()
    df = df.loc[mask_done].copy()

    # ---- FBS filter (reliable) ----
    # CFBD includes homeClassification/awayClassification in /games.
    # Keep rows where BOTH are 'fbs' (case-insensitive).
    for col in ("homeClassification", "awayClassification"):
        if col not in df.columns:
            raise RuntimeError(
                f"Expected '{col}' in CFBD /games response but it was missing. "
                "The API schema may have changed—inspect a sample row to choose a new filter."
            )

    hc = df["homeClassification"].astype(str).str.lower()
    ac = df["awayClassification"].astype(str).str.lower()
    fbs_mask = (hc == "fbs") & (ac == "fbs")
    before = len(df)
    df = df.loc[fbs_mask].copy()
    after = len(df)
    LOG.info("Filtered to FBS-vs-FBS: %s/%s completed games remain", f"{after:,}", f"{before:,}")

    # Compute signed differentials
    df["homePointDiff"] = df["homePoints"] - df["awayPoints"]
    df["awayPointDiff"] = -df["homePointDiff"]

    # Put important columns up front
    preferred_cols = [
        "id", "season", "seasonType", "week", "startDate",
        "homeTeam", "awayTeam", "homePoints", "awayPoints",
        "homePointDiff", "awayPointDiff",
        "homeClassification", "awayClassification",
        "neutralSite", "venue",
        "homeConference", "awayConference",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    return df[cols]


def save_completed_fbs_games(year: int) -> str:
    df = fetch_completed_fbs_games(year)
    out_path = os.path.join(OUT_DIR, f"completed_games_{year}.csv")
    df.to_csv(out_path, index=False)
    LOG.info("Saved %s completed games with differentials → %s", f"{len(df):,}", out_path)
    return out_path


if __name__ == "__main__":
    year = int(os.getenv("YEAR", datetime.now(timezone.utc).year))
    save_completed_fbs_games(year)
