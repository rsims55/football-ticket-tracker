"""Scrape NCAA college football rivalry pairs from Wikipedia."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path
import sys
from typing import Dict, Set, Tuple

import pandas as pd

# Allow running as a script without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.http import build_session
from utils.logging_utils import get_logger


ALIAS_JSON = os.path.join("data", "permanent", "team_aliases.json")


class RivalryScraper:
    def __init__(self, alias_path: str = ALIAS_JSON, timeout: int = 30, session=None):
        self.url = "https://en.wikipedia.org/wiki/List_of_NCAA_college_football_rivalry_games"
        self.alias_path = alias_path
        self.alias_map = self._load_alias_map(alias_path)
        self.rivalries = {}        # dict[str, list[str]]
        self.rivalry_pairs = set() # set[tuple[str, str]]
        self.timeout = int(timeout)
        self.session = session or build_session()
        self.log = get_logger(self.__class__.__name__)

    # ---------- alias helpers ----------
    def _load_alias_map(self, path: str) -> Dict[str, str]:
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
    def scrape(self) -> Dict[str, list]:
        # Wikipedia blocks default UAs; shared session provides headers + retry/backoff.
        resp = self.session.get(self.url, timeout=self.timeout)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        pairs: Set[Tuple[str, str]] = set()

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
    def save(self, filename: str = "data/rivalries.csv") -> None:
        if not self.rivalry_pairs and not self.rivalries:
            self.log.warning("No rivalries to save")
            return

        # write unique, de-duplicated pairs
        rows = [{"team": a, "rival": b} for (a, b) in sorted(self.rivalry_pairs)]
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        self.log.info("Saved %d rivalry pairs to %s", len(df), filename)


# 🔍 Test run
if __name__ == "__main__":
    log = get_logger("rivalry_scraper")
    scraper = RivalryScraper()
    rivalries = scraper.scrape()
    log.info("Found rivalries for %d teams", len(rivalries))
    scraper.save()
