# src/fetchers/schedule_fetcher.py
import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load API key
load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")

WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)


# --- replace ONLY this function in src/fetchers/schedule_fetcher.py ---
def _load_last_point_diffs(year: int) -> dict[str, float]:
    """
    Build {team -> last_point_diff} from data/weekly/completed_games_<YEAR>.csv.
    Uses the most recent COMPLETED game as of 'now' (UTC). If file missing or empty,
    returns {} and the caller will simply not add the columns.
    """
    path = os.path.join(WEEKLY_DIR, f"completed_games_{year}.csv")
    if not os.path.exists(path):
        print(f"ℹ️ No completed games file found at {path}. Skipping last_point_diff merge.")
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
        print(f"⚠️ completed_games CSV missing columns {sorted(missing)}; skipping merge.")
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

    """
    Build {team -> last_point_diff} from data/weekly/completed_games_<YEAR>.csv.
    Uses the most recent COMPLETED game as of 'now' (UTC). If file missing or empty,
    returns {} and the caller will simply not add the columns.
    """
    path = os.path.join(WEEKLY_DIR, f"completed_games_{year}.csv")
    if not os.path.exists(path):
        print(f"ℹ️ No completed games file found at {path}. Skipping last_point_diff merge.")
        return {}

    df = pd.read_csv(path)

    # Normalize date column
    if "startDate" in df.columns:
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    else:
        df["startDate"] = pd.NaT

    # Ensure expected columns are present
    required = {"homeTeam", "awayTeam", "homePointDiff", "awayPointDiff"}
    missing = required - set(df.columns)
    if missing:
        print(f"⚠️ completed_games CSV missing columns {sorted(missing)}; skipping merge.")
        return {}

    # Only consider games that occurred up to now (no leakage)
    now_utc = datetime.now(timezone.utc)
    df = df[df["startDate"].notna() & (df["startDate"] <= now_utc)].copy()
    if df.empty:
        return {}

    # Long form with one row per team per game (signed from that team's perspective)
    long_home = df.rename(
        columns={"homeTeam": "team", "awayTeam": "opponent", "homePointDiff": "pointDiff"}
    )[["team", "opponent", "startDate", "pointDiff"]]
    long_away = df.rename(
        columns={"awayTeam": "team", "homeTeam": "opponent", "awayPointDiff": "pointDiff"}
    )[["team", "opponent", "startDate", "pointDiff"]]
    long_df = pd.concat([long_home, long_away], ignore_index=True)

    # Take the most recent game per team
    long_df = long_df.sort_values(["team", "startDate"])
    idx = long_df.groupby("team", as_index=False)["startDate"].idxmax()
    latest = long_df.loc[idx, ["team", "pointDiff"]].copy()

    return dict(zip(latest["team"], latest["pointDiff"]))


class ScheduleFetcher:
    def __init__(self, year=None):
        self.year = year or datetime.now().year
        self.schedule = None

    def fetch(self, week=None):
        url = f"https://api.collegefootballdata.com/games?year={self.year}"
        if week:
            url += f"&week={week}"
        headers = {"Authorization": f"Bearer {API_KEY}"}

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch schedule: {response.status_code}")
            return None

        data = response.json()
        df = pd.DataFrame(data)

        if "startDate" not in df.columns:
            print("⚠️ No startDate field found")
            return None

        # --- your existing logic (unchanged) ---
        df["startDate"] = pd.to_datetime(df["startDate"], errors='coerce')
        df = df[(df["startDate"] > pd.Timestamp.now(tz="UTC")) & (df["homeClassification"] == "fbs")]

        df = df[df["startDate"].notnull()]
        df["startDateEastern"] = df["startDate"].dt.tz_convert("US/Eastern")
        df["startDateEastern"] = df["startDateEastern"].dt.date

        # Guard if column is missing in some rows/season responses
        if "startTimeTBD" not in df.columns:
            df["startTimeTBD"] = False

        df["dayOfWeek"] = pd.to_datetime(df["startDate"], errors='coerce').dt.tz_convert("US/Eastern").dt.day_name()
        df["kickoffTimeStr"] = np.where(
            df["startTimeTBD"],
            "",
            pd.to_datetime(df["startDate"], errors='coerce').dt.tz_convert("US/Eastern").dt.strftime("%-I:%M %p")
        )
        # --- end of your existing logic ---

        # --- minimal addition: merge last point differentials ---
        last_diff_map = _load_last_point_diffs(self.year)
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
            print(f"✅ Schedule saved to {filename}")
        else:
            print("⚠️ No schedule to save")


# 🧪 Test run
if __name__ == "__main__":
    sf = ScheduleFetcher()
    df = sf.fetch()
    if df is not None:
        print(df[["homeTeam", "awayTeam", "startDateEastern", "dayOfWeek", "home_last_point_diff", "away_last_point_diff"]].head())
        sf.save()
