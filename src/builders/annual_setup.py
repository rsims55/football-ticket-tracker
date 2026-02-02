#!/usr/bin/env python3
# src/builders/annual_setup.py
"""
Annual setup:
  - Scrape stadium data
  - Scrape rivalry pairs

Run:
  python src/builders/annual_setup.py --year 2026
  # Optional override (kept inside repo unless REPO_ALLOW_NON_REPO_OUT=1):
  python src/builders/annual_setup.py --outdir C:\\path\\to\\repo\\data\\annual
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# --- Robust project path handling (run from anywhere) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent  # .../src
PROJ_DIR = SRC_DIR.parent     # repo root
for p in (SRC_DIR, PROJ_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# --- Project imports ---
from scrapers.stadium_scraper import StadiumScraper
from scrapers.rivalry_scraper import RivalryScraper
from utils.logging_utils import get_logger

# --- Repo-lock flags (same behavior as other scripts) ---
REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

log = get_logger("annual_setup")


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
    log.info("Saved %d rows to %s", len(df), path)

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
            log.warning("REPO_DATA_LOCK=1 → ignoring --outdir; writing under repo data/annual")
    else:
        outdir = Path(cli_outdir).expanduser()
        if not _under_repo(outdir) and not ALLOW_ESCAPE:
            log.warning("--outdir resolves outside repo: %s", outdir)
            log.warning("Forcing repo path; set REPO_ALLOW_NON_REPO_OUT=1 to permit.")
            outdir = default
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _archive_old_csvs(outdir: Path) -> None:
    """Move existing CSVs in outdir into data/annual/archive with a timestamp prefix."""
    archive_dir = outdir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in outdir.glob("*.csv"):
        # Skip files already in archive (defensive) and avoid moving new outputs later.
        if archive_dir in p.parents:
            continue
        target = archive_dir / f"{stamp}__{p.name}"
        try:
            shutil.move(str(p), str(target))
            log.info("Archived %s -> %s", p.name, target)
        except Exception as e:
            log.warning("Could not archive %s: %s", p.name, e)

def scrape_stadiums(year: int, outdir: Path) -> pd.DataFrame:
    """Pull the FBS stadium list and persist to data/annual."""
    log.info("Scraping stadium data…")
    df = StadiumScraper().scrape()
    # Expect columns like: ['school','stadium','capacity','city','state', ...]
    if "school" in df.columns:
        df = df.sort_values(df.columns.tolist()).drop_duplicates(subset=["school"], keep="first")
    path = outdir / f"stadiums_{year}.csv"
    safe_to_csv(df, path)
    return df

def scrape_rivalries(year: int, outdir: Path) -> pd.DataFrame:
    """Pull rivalry pairs and persist to data/annual."""
    log.info("Scraping rivalry data…")
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

    log.info("[annual_setup] Paths resolved:")
    log.info("  PROJ_DIR: %s", PROJ_DIR)
    log.info("  OUTDIR:   %s (REPO_DATA_LOCK=%s, ALLOW_ESCAPE=%s)", outdir, "1" if REPO_DATA_LOCK else "0", "1" if ALLOW_ESCAPE else "0")

    _archive_old_csvs(outdir)

    try:
        scrape_stadiums(year, outdir)
    except Exception as e:
        log.error("Stadium scraping failed: %s", e)

    try:
        scrape_rivalries(year, outdir)
    except Exception as e:
        log.error("Rivalry scraping failed: %s", e)

    log.info("Annual setup complete.")

if __name__ == "__main__":
    main()
