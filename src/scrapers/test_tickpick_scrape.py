# src/scrapers/test_tickpick_scrape.py
import os
from src.scrapers.tickpick_pricer import TickPickPricer

def main():
    # Only scrape these 3 teams
    urls = [
        "https://www.tickpick.com/ncaa-football/clemson-tigers-football-tickets/",
        "https://www.tickpick.com/ncaa-football/duke-blue-devils-football-tickets/",
        "https://www.tickpick.com/ncaa-football/boston-college-eagles-football-tickets/",
    ]

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    pricer = TickPickPricer(
        team_urls=urls,
        output_dir=output_dir,
        polite_delay_range=(5, 8),  # shorter delays since it's just 3 teams
        retries=3,
        timeout=25,
        use_cache=False,
        verbose=True,
    )
    pricer.run()


if __name__ == "__main__":
    main()
