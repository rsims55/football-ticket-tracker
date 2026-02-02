"""Fetch venue capacities from CFBD /venues for neutral/alternate sites."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Allow running as a script without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.http import build_session
from utils.logging_utils import get_logger

load_dotenv()

API_KEY = os.getenv("CFD_API_KEY")
API_BASE = os.getenv("CFD_API_BASE", "https://api.collegefootballdata.com").rstrip("/")
API_BASES = [b.strip().rstrip("/") for b in os.getenv("CFD_API_BASES", "").split(",") if b.strip()]

LOG = get_logger("venues_fetcher")

OUT_DIR = os.path.join("data", "annual")
os.makedirs(OUT_DIR, exist_ok=True)


def _base_candidates() -> list[str]:
    return API_BASES if API_BASES else [API_BASE]


def fetch_venues() -> Optional[pd.DataFrame]:
    if not API_KEY:
        raise RuntimeError("CFD_API_KEY is not set; cannot fetch venues.")

    session = build_session()
    headers = {"Authorization": f"Bearer {API_KEY}"}

    data: list[dict[str, Any]] | None = None
    for base in _base_candidates():
        url = f"{base}/venues"
        resp = session.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            LOG.error("Venues API failed: %s (base=%s)", resp.status_code, base)
            if resp.text:
                LOG.error("Response body: %s", resp.text[:500])
            continue
        try:
            data = resp.json()
        except Exception as e:
            LOG.error("Non-JSON response from venues API: %s", e)
            if resp.text:
                LOG.error("Response body: %s", resp.text[:500])
            continue
        break

    if data is None:
        return None
    df = pd.DataFrame(data)
    if df.empty:
        return None
    return df


def save_venues(year: Optional[int] = None) -> str:
    df = fetch_venues()
    if df is None or df.empty:
        raise RuntimeError("No venues returned from CFBD.")
    year = year or datetime.now().year
    out_path = os.path.join(OUT_DIR, f"venues_{year}.csv")
    df.to_csv(out_path, index=False)
    LOG.info("Saved %s venues → %s", f"{len(df):,}", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch venues via CFBD /venues.")
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    save_venues(args.year)
