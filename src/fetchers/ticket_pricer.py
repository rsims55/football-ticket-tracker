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
            return (CLIENT_ID, CLIENT_SECRET)
        return None

    def get_summary(self, home_team, game_date, away_team=None):
        if self.use_mock or not CLIENT_ID:
            print("⚠️ Using mock data (client ID missing or mock mode enabled).")
            return self._mock_data(home_team)

        # 🧠 Build more precise search query
        query = f"{home_team} {away_team}".strip()

        params = {
            "q": query,
            "datetime_utc.gte": game_date.strftime("%Y-%m-%dT00:00:00"),
            "datetime_utc.lte": game_date.strftime("%Y-%m-%dT23:59:59"),
            "taxonomies.name": "ncaa_football",
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
            return self._mock_data(home_team)

        events = response.json().get("events", [])
        if not events:
            print("⚠️ No matching events found.")
            return self._mock_data(home_team)

        event = events[0]
        stats = event.get("stats", {})

        # Show message if stats are all None
        if not any(stats.values()):
            print("ℹ️  Event found but no pricing data is available yet.")

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

# 🔬 Run test cases if script is executed directly
if __name__ == "__main__":
    USE_MOCK = False  # Use real SeatGeek API

    # Each tuple: (home_team, date, away_team)
    games_to_test = [
        ("Michigan", datetime(2025, 10, 4), "Wisconsin"),
        ("Texas", datetime(2025, 9, 27), "South Alabama"),
        ("Clemson", datetime(2025, 11, 1), "Duke"),
        ("Alabama", datetime(2025, 9, 13), "Wisconsin"),
        ("Georgia", datetime(2025, 11, 22), "Old Dominion"),
    ]

    random.shuffle(games_to_test)
    tp = TicketPricer(use_mock=USE_MOCK)

    for home, date, away in games_to_test:
        print(f"\n🎟️  Fetching prices for {away} at {home} on {date.strftime('%Y-%m-%d')}")
        summary = tp.get_summary(home, date, away)
        for key, val in summary.items():
            print(f"   {key}: {val}")
