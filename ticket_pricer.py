import os
import requests
import random
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
SEATGEEK_CLIENT_ID = os.getenv("SEATGEEK_CLIENT_ID")

class TicketPricer:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock

    def get_summary(self, team, game_date):
        """
        Return ticket price summary for a given team and game date.
        Uses real API if available, otherwise returns mock data.
        """
        if self.use_mock or not SEATGEEK_CLIENT_ID:
            return self._mock_data(team)

        url = "https://api.seatgeek.com/2/events"
        params = {
            "q": team,
            "datetime_utc.gte": game_date.strftime("%Y-%m-%dT00:00:00"),
            "datetime_utc.lte": game_date.strftime("%Y-%m-%dT23:59:59"),
            "client_id": SEATGEEK_CLIENT_ID
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return self._mock_data(team)

        events = response.json().get("events", [])
        if not events:
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

# 🔍 Test run
if __name__ == "__main__":
    tp = TicketPricer(use_mock=True)
    summary = tp.get_summary("Ohio State", datetime(2025, 11, 29))
