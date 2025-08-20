import pandas as pd
import re
import os
import json

class StadiumScraper:
    def __init__(self, alias_path="data/permanent/team_aliases.json"):
        self.url = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_FBS_football_stadiums"
        self.df = None
        self.alias_path = alias_path
        self.alias_map = self._load_aliases()

    def _load_aliases(self):
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
                print(f"⚠️ Could not load alias file {self.alias_path}: {e}")
        return {}

    def scrape(self):
        tables = pd.read_html(self.url, header=0)
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

    def save(self, filename="data/stadiums_full.csv"):
        if self.df is not None:
            self.df.to_csv(filename, index=False)
            print(f"✅ Stadium info saved to {filename}")
        else:
            print("⚠️ No data to save")

# 🔍 Test run
if __name__ == "__main__":
    scraper = StadiumScraper()
    df = scraper.scrape()
    print(df.head())
    scraper.save()
