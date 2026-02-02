"""Scrape FBS stadium data from Wikipedia, with alias cleanup."""
from __future__ import annotations

import json
import os
import re
from io import StringIO
from pathlib import Path
import sys
from typing import Dict

import pandas as pd

# Allow running as a script without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.http import build_session
from utils.logging_utils import get_logger


class StadiumScraper:
    """Pulls the NCAA Division I FBS stadium list and normalizes names."""

    URL = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_FBS_football_stadiums"

    def __init__(
        self,
        alias_path: str = "data/permanent/team_aliases.json",
        timeout: int = 30,
        session=None,
    ):
        self.df: pd.DataFrame | None = None
        self.alias_path = alias_path
        self.log = get_logger(self.__class__.__name__)
        self.alias_map = self._load_aliases()
        self.timeout = int(timeout)
        self.session = session or build_session()

    def _load_aliases(self) -> Dict[str, str]:
        if os.path.exists(self.alias_path):
            try:
                with open(self.alias_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Expecting {"Raw Scraped Name": "Alias"} or {"alias": ["raw1", "raw2"]}
                alias_map = {}
                if isinstance(data, dict):
                    # normalize both styles
                    for k, v in data.items():
                        if isinstance(v, str):
                            alias_map[k.strip()] = v.strip()
                        elif isinstance(v, list):
                            for raw in v:
                                alias_map[raw.strip()] = k.strip()
                return alias_map
            except Exception as e:
                self.log.warning("Could not load alias file %s: %s", self.alias_path, e)
        return {}

    def scrape(self) -> pd.DataFrame:
        # Wikipedia blocks default UAs; shared session provides headers + retry/backoff.
        resp = self.session.get(self.URL, timeout=self.timeout)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), header=0)
        df = tables[0]

        # Auto-map column names
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "team" in col_lower:
                column_map[col] = "school"
            elif "stadium" in col_lower:
                column_map[col] = "stadium"
            elif "capacity" in col_lower:
                column_map[col] = "capacity"
            elif "city" in col_lower:
                column_map[col] = "city"
            elif "state" in col_lower:
                column_map[col] = "state"

        df = df.rename(columns=column_map)
        df = df[["school", "stadium", "capacity", "city", "state"]]

        # Clean school names
        df["school"] = (
            df["school"]
            .astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
        )

        # Apply alias mapping
        if self.alias_map:
            df["school"] = df["school"].apply(lambda x: self.alias_map.get(x, x))

        # Clean capacity
        df["capacity"] = (
            df["capacity"]
            .astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.replace(",", "", regex=True)
            .apply(lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else None)
        )

        self.df = df
        return df

    def save(self, filename: str = "data/stadiums_full.csv") -> None:
        if self.df is not None:
            self.df.to_csv(filename, index=False)
            self.log.info("Stadium info saved to %s", filename)
        else:
            self.log.warning("No data to save")

# 🔍 Test run
if __name__ == "__main__":
    log = get_logger("stadium_scraper")
    scraper = StadiumScraper()
    df = scraper.scrape()
    log.info("\n%s", df.head().to_string(index=False))
    scraper.save()
