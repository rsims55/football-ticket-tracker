import os
import requests
import random
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
CLIENT_ID = os.getenv("SEATGEEK_CLIENT_ID")
CLIENT_SECRET = os.getenv("SEATGEEK_CLIENT_SECRET")

class TicketPricer:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        self.auth = self._build_auth()

    def _build_auth(self):
        if CLIENT_ID and CLIENT_SECRET:
            # HTTP Basic Auth header with ID and secret
            return (CLIENT_ID, CLIENT_SECRET)
        return None

    def get_summary(self, team, game_date):
        if self.use_mock or not CLIENT_ID:
            print("⚠️ Using mock data (client ID missing or mock mode enabled).")
            return self._mock_data(team)

        params = {
            "q": team,
            "datetime_utc.gte": game_date.strftime("%Y-%m-%dT00:00:00"),
            "datetime_utc.lte": game_date.strftime("%Y-%m-%dT23:59:59"),
        }
        if not self.auth:
            params["client_id"] = CLIENT_ID

        try:
            response = requests.get(
                "https://api.seatgeek.com/2/events",
                params=params,
                auth=self.auth
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ API request failed: {e}")
            return self._mock_data(team)

        events = response.json().get("events", [])
        if not events:
            print("⚠️ No events found, falling back to mock data.")
            return self._mock_data(team)

        event = events[0]
        stats = event.get("stats", {})

        return {
            "event_title": event.get("title", ""),
            "lowest_price": stats.get("lowest_price"),
            "average_price": stats.get("average_price"),
            "listing_count": stats.get("listing_count"),
            "group_price_2": None,
            "group_price_4": None,
            "lowest_price_upper": None,
            "lowest_price_lower": None
        }

    def _mock_data(self, team):
        return {
            "event_title": f"{team} Game",
            "lowest_price": random.randint(20, 150),
            "average_price": random.randint(50, 250),
            "listing_count": random.randint(100, 1000),
            "group_price_2": random.randint(50, 200),
            "group_price_4": random.randint(100, 400),
            "lowest_price_upper": random.randint(20, 100),
            "lowest_price_lower": random.randint(80, 200),
        }

if __name__ == "__main__":
    tp = TicketPricer(use_mock=False)
    print(tp.get_summary("Ohio State", datetime(2025, 11, 29)))
