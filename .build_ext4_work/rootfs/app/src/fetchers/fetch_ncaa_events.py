import os
import time
import requests
import pandas as pd
import re
import pytz
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("SEATGEEK_CLIENT_ID")
CLIENT_SECRET = os.getenv("SEATGEEK_CLIENT_SECRET")

class NCAAEventsFetcher:
    BASE_URL = "https://api.seatgeek.com/2/events"
    MAX_RETRIES = 5
    RETRY_BACKOFF_FACTOR = 2
    RATE_LIMIT_DELAY = 0.25

    def __init__(self, year):
        self.year = year
        self.auth = self._build_auth()

    def _build_auth(self):
        return (CLIENT_ID, CLIENT_SECRET) if CLIENT_ID and CLIENT_SECRET else None

    def fetch_all_events(self):
        per_page, page, all_events = 50, 1, []

        while True:
            params = {
                "taxonomies.name": "ncaa_football",
                "datetime_utc.gte": f"{self.year}-01-01T00:00:00",
                "datetime_utc.lte": f"{self.year}-12-31T23:59:59",
                "per_page": per_page,
                "page": page
            }
            if not self.auth:
                params["client_id"] = CLIENT_ID

            success, retries = False, 0
            while not success and retries < self.MAX_RETRIES:
                response = requests.get(self.BASE_URL, params=params, auth=self.auth)
                if response.status_code == 429:
                    wait_time = self.RETRY_BACKOFF_FACTOR ** retries
                    print(f"⚠️ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                elif response.status_code != 200:
                    print(f"❌ Failed page {page}: HTTP {response.status_code}")
                    return all_events
                else:
                    success = True

            if not success:
                print(f"❌ Max retries reached for page {page}. Stopping.")
                break

            events = response.json().get("events", [])
            if not events:
                print(f"ℹ️ No more events at page {page}. Stopping.")
                break

            all_events.extend(events)
            print(f"ℹ️ Page {page} fetched ({len(events)} events).")

            if len(events) < per_page:
                break

            page += 1
            time.sleep(self.RATE_LIMIT_DELAY)

        return all_events

    @staticmethod
    def parse_event_title(title):
        title = title.replace("Football", "").strip()

        patterns = [
            r"^[^:]+:\s*(.*?) vs (.*?)$",
            r"^[^:]+:\s*(.*?) at (.*?)$",
            r"^(.*?) at (.*?)$",
            r"^(.*?) vs (.*?)$"
        ]

        for pattern in patterns:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                teams = match.groups()
                if 'vs' in pattern:
                    return teams[0].strip(), teams[1].strip()
                else:
                    return teams[1].strip(), teams[0].strip()

        return None, None

    def build_events_df(self, events):
        eastern = pytz.timezone('US/Eastern')
        records = []

        for e in events:
            event_id = e.get("id")
            title = e.get("title", "")
            date_utc = pd.to_datetime(e.get("datetime_utc"), utc=True)
            date_est = date_utc.tz_convert(eastern)
            venue_id = e.get("venue", {}).get("id")
            venue_name = e.get("venue", {}).get("name")

            home_team, away_team = self.parse_event_title(title)

            records.append({
                "event_id": event_id,
                "date": date_utc,
                "date_est": date_est.strftime("%Y-%m-%d %H:%M:%S%z"),
                "date_est_only": date_est.strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "title": title
            })

        return pd.DataFrame(records)


# 🔬 Example usage:
if __name__ == "__main__":
    year = 2025
    fetcher = NCAAEventsFetcher(year)
    events = fetcher.fetch_all_events()
    df = fetcher.build_events_df(events)

    os.makedirs("data", exist_ok=True)
    filename = f"data/ncaa_football_events_{year}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} events to {filename}")
