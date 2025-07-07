import pandas as pd
from collections import defaultdict
import re

class RivalryScraper:
    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/List_of_NCAA_college_football_rivalry_games"
        self.rivalries = {}

    def scrape(self):
        tables = pd.read_html(self.url)
        rivalry_map = defaultdict(set)

        for table in tables:
            colnames = [str(col).lower() for col in table.columns]
            if sum("team" in col for col in colnames) >= 2:
                team_cols = [col for col in table.columns if "Team" in str(col)]
                if len(team_cols) >= 2:
                    t1, t2 = team_cols[0], team_cols[1]
                    for _, row in table.iterrows():
                        team1 = row[t1]
                        team2 = row[t2]
                        if isinstance(team1, str) and isinstance(team2, str):
                            team1 = re.sub(r"\[.*?\]", "", team1).strip()
                            team2 = re.sub(r"\[.*?\]", "", team2).strip()
                            rivalry_map[team1].add(team2)
                            rivalry_map[team2].add(team1)

        self.rivalries = {k: sorted(list(v)) for k, v in rivalry_map.items()}
        return self.rivalries

    def save(self, filename="data/rivalries.csv"):
        if not self.rivalries:
            print("⚠️ No rivalries to save")
            return

        rows = []
        for team, rivals in self.rivalries.items():
            for rival in rivals:
                if team < rival:  # Avoid duplicate mirror rows
                    rows.append({"team": team, "rival": rival})

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} rivalry pairs to {filename}")

# 🔍 Test run
if __name__ == "__main__":
    scraper = RivalryScraper()
    rivalries = scraper.scrape()
    print(f"Found rivalries for {len(rivalries)} teams")
    scraper.save()
