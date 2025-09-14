# src/fetchers/results_fetcher.py
from __future__ import annotations

import os
from datetime import datetime, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")

OUT_DIR = os.path.join("data", "weekly")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_completed_fbs_games(year: int) -> pd.DataFrame:
    """
    Pull ALL games for the season (regular + postseason),
    keep only rows with both scores present (i.e., completed),
    then filter to FBS-vs-FBS using home/awayClassification.
    Compute signed point differentials.
    """
    url = f"https://api.collegefootballdata.com/games?year={year}&seasonType=both"
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json())

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
    mask_done = df.get("homePoints").notna() & df.get("awayPoints").notna()
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
    print(f"ℹ️ Filtered to FBS-vs-FBS: {after:,}/{before:,} completed games remain")

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

    if os.path.exists(out_path):
        print(f"♻️ Overwriting existing file: {out_path}")
    else:
        print(f"💾 Creating new file: {out_path}")

    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df):,} completed FBS-vs-FBS games with differentials → {out_path}")
    return out_path

def save_completed_fbs_games(year: int) -> str:
    df = fetch_completed_fbs_games(year)
    out_path = os.path.join(OUT_DIR, f"completed_games_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df):,} completed FBS-vs-FBS games with differentials → {out_path}")
    return out_path


if __name__ == "__main__":
    year = int(os.getenv("YEAR", datetime.now(timezone.utc).year))
    save_completed_fbs_games(year)
