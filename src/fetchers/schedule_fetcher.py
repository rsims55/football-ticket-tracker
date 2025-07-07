import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")

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

        df["startDate"] = pd.to_datetime(df["startDate"])
        df = df[(df["startDate"] > pd.Timestamp.now(tz="UTC")) & (df["homeClassification"] == "fbs")]

        df["startDateEastern"] = df["startDate"].dt.tz_convert("US/Eastern")
        df["dayOfWeek"] = df["startDateEastern"].dt.day_name()
        df["kickoffTimeStr"] = np.where(
            df["startTimeTBD"],
            "",
            df["startDateEastern"].dt.strftime("%-I:%M %p")
        )

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
        print(df[["homeTeam", "awayTeam", "startDateEastern", "dayOfWeek"]].head())
        sf.save()
