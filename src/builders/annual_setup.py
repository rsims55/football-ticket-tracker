#!/usr/bin/env python3
"""
Annual setup script:
- Scrape stadium data
- Scrape rivalry pairs

Run:
  python src/builders/annual_setup.py --year 2025
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# --- Robust project path handling (run from anywhere) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))   # .../src
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))      # project root
for p in (SRC_DIR, PROJ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Project imports ---
from scrapers.stadium_scraper import StadiumScraper
from scrapers.rivalry_scraper import RivalryScraper


# --- Helpers ---
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df):,} rows to {path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Annual data setup for stadiums and rivalries")
    ap.add_argument("--year", type=int, default=datetime.now().year, help="Season year (default: current year)")
    ap.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(PROJ_DIR, "data", "annual"),
        help="Output directory (default: data/annual/)"
    )
    return ap


def scrape_stadiums(year: int, outdir: str) -> pd.DataFrame:
    print("🏟️ Scraping stadium data…")
    df = StadiumScraper().scrape()
    # Expect columns like: ['school','stadium','capacity','city','state', ...]
    if "school" in df.columns:
        df = df.sort_values(df.columns.tolist()).drop_duplicates(subset=["school"], keep="first")
    path = os.path.join(outdir, f"stadiums_{year}.csv")
    safe_to_csv(df, path)
    return df


def scrape_rivalries(year: int, outdir: str) -> pd.DataFrame:
    print("🔥 Scraping rivalry data…")
    rivals_map = RivalryScraper().scrape()  # dict[str, list[str]]

    # Deduplicate unordered pairs using a set of frozensets
    pair_set = set()
    for team, rivals in rivals_map.items():
        for r in rivals or []:
            if not isinstance(r, str):
                continue
            pair_set.add(frozenset((team, r)))

    # Back to two-column DataFrame (alphabetize within each pair)
    rows = []
    for pair in pair_set:
        a, b = sorted(list(pair))
        rows.append({"team": a, "rival": b})

    df = pd.DataFrame(rows).sort_values(["team", "rival"]).reset_index(drop=True)
    path = os.path.join(outdir, f"rivalries_{year}.csv")
    safe_to_csv(df, path)
    return df


def main():
    ap = build_argparser()
    args = ap.parse_args()

    year = args.year
    outdir = args.outdir

    print(f"🗓️  Running annual setup for {year} (output dir: {outdir})")

    try:
        scrape_stadiums(year, outdir)
    except Exception as e:
        print(f"❌ Stadium scraping failed: {e}")

    try:
        scrape_rivalries(year, outdir)
    except Exception as e:
        print(f"❌ Rivalry scraping failed: {e}")

    print("🎉 Annual setup complete.")


if __name__ == "__main__":
    main()
