# src/builders/annual_setup.py
#!/usr/bin/env python3
"""
Annual setup script:
- Scrape stadium data
- Scrape rivalry pairs

Run:
  python src/builders/annual_setup.py --year 2025
  # optional override (kept inside repo unless REPO_ALLOW_NON_REPO_OUT=1):
  python src/builders/annual_setup.py --outdir C:\path\to\repo\data\annual
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd

# --- Robust project path handling (run from anywhere) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR     = CURRENT_DIR.parent                 # .../src
PROJ_DIR    = SRC_DIR.parent                     # repo root
for p in (SRC_DIR, PROJ_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# --- Project imports ---
from scrapers.stadium_scraper import StadiumScraper
from scrapers.rivalry_scraper import RivalryScraper

# --- Repo-lock flags (same behavior as other scripts) ---
REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

def _under_repo(p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(PROJ_DIR.resolve())
    except AttributeError:
        return str(p.resolve()).startswith(str(PROJ_DIR.resolve()))

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df):,} rows to {path}")

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Annual data setup for stadiums and rivalries")
    ap.add_argument("--year", type=int, default=datetime.now().year, help="Season year (default: current year)")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(PROJ_DIR / "data" / "annual"),
        help="Output directory (default: <repo>/data/annual/)"
    )
    return ap

def _resolve_outdir(cli_outdir: str) -> Path:
    """
    Resolve the final output directory with repo-locking:
      - If REPO_DATA_LOCK=1 (default): force <repo>/data/annual
      - Else respect --outdir, but if it escapes the repo and ALLOW_ESCAPE=0, force repo path
    """
    default = PROJ_DIR / "data" / "annual"
    if REPO_DATA_LOCK:
        outdir = default
        # If user passed something else, warn that we're ignoring it
        if cli_outdir and Path(cli_outdir).resolve() != outdir.resolve():
            print("⚠️  REPO_DATA_LOCK=1 → ignoring --outdir; writing under repo data/annual")
    else:
        outdir = Path(cli_outdir).expanduser()
        if not _under_repo(outdir) and not ALLOW_ESCAPE:
            print(f"🚫 --outdir resolves outside repo: {outdir}")
            print("    Forcing repo path; set REPO_ALLOW_NON_REPO_OUT=1 to permit.")
            outdir = default
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def scrape_stadiums(year: int, outdir: Path) -> pd.DataFrame:
    print("🏟️ Scraping stadium data…")
    df = StadiumScraper().scrape()
    # Expect columns like: ['school','stadium','capacity','city','state', ...]
    if "school" in df.columns:
        df = df.sort_values(df.columns.tolist()).drop_duplicates(subset=["school"], keep="first")
    path = outdir / f"stadiums_{year}.csv"
    safe_to_csv(df, path)
    return df

def scrape_rivalries(year: int, outdir: Path) -> pd.DataFrame:
    print("🔥 Scraping rivalry data…")
    rivals_map: Dict[str, list] = RivalryScraper().scrape()  # dict[str, list[str]]
    pair_set = set()
    for team, rivals in (rivals_map or {}).items():
        for r in rivals or []:
            if isinstance(r, str):
                pair_set.add(frozenset((team, r)))

    rows = []
    for pair in pair_set:
        a, b = sorted(list(pair))
        rows.append({"team": a, "rival": b})

    df = pd.DataFrame(rows).sort_values(["team", "rival"]).reset_index(drop=True)
    path = outdir / f"rivalries_{year}.csv"
    safe_to_csv(df, path)
    return df

def main():
    ap = build_argparser()
    args = ap.parse_args()

    year = int(args.year)
    outdir = _resolve_outdir(args.outdir)

    print("[annual_setup] Paths resolved:")
    print(f"  PROJ_DIR: {PROJ_DIR}")
    print(f"  OUTDIR:   {outdir} (REPO_DATA_LOCK={'1' if REPO_DATA_LOCK else '0'}, ALLOW_ESCAPE={'1' if ALLOW_ESCAPE else '0'})")

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
