import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")

class RankingsFetcher:
    def __init__(self, year=None):
        self.year = year or datetime.now().year
        self.api_key = API_KEY
        self.file_path = None

    def get_latest_week(self, year):
        url = f"https://api.collegefootballdata.com/games?year={year}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            games = response.json()
            weeks = set(game.get("week") for game in games if game.get("week") is not None)
            if weeks:
                return max(weeks)
        return None

    def fetch_rankings(self, year, week):
        url = f"https://api.collegefootballdata.com/rankings?year={year}&seasonType=regular&week={week}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                polls = data[0].get("polls", [])
                preferred = ["Playoff Committee Rankings", "AP Top 25", "Coaches Poll"]
                normalized = {p["poll"].strip().lower(): p for p in polls}

                for name in preferred:
                    key = name.strip().lower()
                    if key in normalized:
                        ranks = normalized[key].get("ranks", [])
                        df = pd.DataFrame(ranks)
                        df = self._clean_rankings_df(df)
                        filename = f"data/rankings_week{week}.csv"
                        df.to_csv(filename, index=False)
                        self.file_path = filename
                        return df
        return None

    def scrape_wikipedia_rankings(self, year=None, poll="AP Top 25"):
        import re
        year = year or self.year
        url = f"https://en.wikipedia.org/wiki/{year - 1}_NCAA_Division_I_FBS_football_rankings"

        try:
            tables = pd.read_html(url)
        except Exception:
            return None

        try:
            ap_table = tables[2]  # AP poll is typically table index 2
            first_col = ap_table.columns[0]
            last_col = ap_table.columns[-2] if "Unnamed" in ap_table.columns[-1] else ap_table.columns[-1]

            df = ap_table[[first_col, last_col]].copy()
            df.columns = ["rank", "school"]
            df = self._clean_rankings_df(df)

            filename = f"data/wiki_{poll.lower().replace(' ', '_')}_{year - 1}.csv"
            df.to_csv(filename, index=False)
            return df
        except Exception:
            return None

    def fetch_and_load(self):
        latest_week = self.get_latest_week(self.year)
        if latest_week:
            df = self.fetch_rankings(self.year, latest_week)
            if df is not None:
                return df

        wiki_df = self.scrape_wikipedia_rankings(self.year, poll="AP Top 25")
        if wiki_df is not None:
            return wiki_df

        fallback_week = self.get_latest_week(self.year - 1)
        if fallback_week:
            df = self.fetch_rankings(self.year - 1, fallback_week)
            if df is not None:
                return df

        return None

    def _clean_rankings_df(self, df):
        df = df.rename(columns={"RK": "rank", "Rank": "rank", "TEAM": "school", "School": "school"})
        df = df[["rank", "school"]]
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"])
        df["rank"] = df["rank"].astype(int)
        df["school"] = df["school"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
        return df
