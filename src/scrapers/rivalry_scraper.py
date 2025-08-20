# src/fetchers/rivalry_scraper.py
import os
import re
import json
import pandas as pd
from collections import defaultdict

ALIAS_JSON = os.path.join("data", "permanent", "team_aliases.json")

class RivalryScraper:
    def __init__(self, alias_path: str = ALIAS_JSON):
        self.url = "https://en.wikipedia.org/wiki/List_of_NCAA_college_football_rivalry_games"
        self.alias_path = alias_path
        self.alias_map = self._load_alias_map(alias_path)
        self.rivalries = {}        # dict[str, list[str]]
        self.rivalry_pairs = set() # set[tuple[str, str]]

    # ---------- alias helpers ----------
    def _load_alias_map(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # exact-key map (same behavior as weekly_update)
            return {str(k): str(v) for k, v in data.items()}
        # if file missing, just return empty; we'll keep originals
        return {}

    def _alias(self, name: str) -> str:
        # exact match first (consistent with weekly_update)
        return self.alias_map.get(name, name)

    # ---------- cleaning ----------
    @staticmethod
    def _clean_team(s: str) -> str:
        # remove footnotes in brackets and parentheses, normalize spaces
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ---------- scrape ----------
    def scrape(self):
        tables = pd.read_html(self.url)
        pairs = set()

        for table in tables:
            colnames_lower = [str(col).lower() for col in table.columns]
            # need at least two "team" columns in the header
            if sum("team" in c for c in colnames_lower) >= 2:
                # pick the first two columns that contain "Team" (original case scan)
                team_cols = [col for col in table.columns if "team" in str(col).lower()]
                if len(team_cols) >= 2:
                    t1, t2 = team_cols[0], team_cols[1]
                    for _, row in table.iterrows():
                        team1 = row[t1]
                        team2 = row[t2]
                        if isinstance(team1, str) and isinstance(team2, str):
                            a = self._clean_team(team1)
                            b = self._clean_team(team2)
                            if not a or not b:
                                continue
                            # apply alias map (keep original if no alias)
                            a_alias = self._alias(a)
                            b_alias = self._alias(b)
                            if a_alias and b_alias and a_alias != b_alias:
                                pairs.add(tuple(sorted((a_alias, b_alias))))

        # build dict -> list mapping from pairs
        rivalry_map = defaultdict(set)
        for a, b in pairs:
            rivalry_map[a].add(b)
            rivalry_map[b].add(a)

        self.rivalry_pairs = pairs
        self.rivalries = {k: sorted(v) for k, v in rivalry_map.items()}
        return self.rivalries

    # ---------- save ----------
    def save(self, filename: str = "data/rivalries.csv"):
        if not self.rivalry_pairs and not self.rivalries:
            print("⚠️ No rivalries to save")
            return

        # write unique, de-duplicated pairs
        rows = [{"team": a, "rival": b} for (a, b) in sorted(self.rivalry_pairs)]
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} rivalry pairs to {filename}")


# 🔍 Test run
if __name__ == "__main__":
    scraper = RivalryScraper()
    rivalries = scraper.scrape()
    print(f"Found rivalries for {len(rivalries)} teams")
    scraper.save()
