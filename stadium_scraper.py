import pandas as pd
import re

class StadiumScraper:
    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_FBS_football_stadiums"
        self.df = None

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

        # Clean data
        df["school"] = df["school"].str.replace(r"\[.*?\]", "", regex=True).str.strip()
        df["capacity"] = df["capacity"] \
            .astype(str) \
            .str.replace(r"\[.*?\]", "", regex=True) \
            .str.replace(",", "", regex=True) \
            .apply(lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else None)

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
