"""
Kickoff snapshot: scrape ticket prices for games kicking off within a rolling
window around "now" and append the results to price_snapshots_{YEAR}.csv.

Intended to be run every 30 min during game-day windows (via the daemon cron).
The window is [-WINDOW_BEFORE_MIN, +WINDOW_AFTER_MIN] around each game's kickoff.

This gives us a "last price before game" data point, which anchors the price
trajectory and improves model training quality.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from zoneinfo import ZoneInfo

# Ensure src/ is on sys.path so sibling packages are importable
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.status import write_status

try:
    from scrapers.tickpick_pricer import TickPickPricer
except Exception as e:
    raise ImportError(f"Could not import TickPickPricer: {e!r}")

# ---------------------------
# Config
# ---------------------------
TIMEZONE = ZoneInfo("America/New_York")
YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))

# How far before/after kickoff to capture the "kickoff price"
WINDOW_BEFORE_MIN = int(os.getenv("KICKOFF_WINDOW_BEFORE_MIN", "30"))
WINDOW_AFTER_MIN = int(os.getenv("KICKOFF_WINDOW_AFTER_MIN", "15"))

# Paths
_here = Path(__file__).resolve()
_root = next(
    (p for p in [_here] + list(_here.parents) if (p / ".git").exists() or (p / "pyproject.toml").exists()),
    _here.parent.parent.parent,
)
PROJ_DIR = _root
DAILY_DIR = PROJ_DIR / "data" / "daily"
SNAPSHOT_PATH = DAILY_DIR / f"price_snapshots_{YEAR}.csv"
WEEKLY_SCHEDULE_PATH = PROJ_DIR / "data" / "weekly" / f"full_{YEAR}_schedule.csv"

# Polite scraping settings (same env vars as daily_snapshot for consistency)
TP_TIMEOUT = int(os.getenv("TP_TIMEOUT", "25"))
TP_RETRIES = int(os.getenv("TP_RETRIES", "3"))
TP_POLITE_LOW = int(os.getenv("TP_POLITE_LOW", "10"))
TP_POLITE_HIGH = int(os.getenv("TP_POLITE_HIGH", "18"))
TP_BACKOFF_BASE = float(os.getenv("TP_BACKOFF_BASE", "3.0"))
TP_BACKOFF_MULT = float(os.getenv("TP_BACKOFF_MULT", "2.2"))
TP_BACKOFF_MAX = float(os.getenv("TP_BACKOFF_MAX", "120"))
TP_BACKOFF_JITTER = float(os.getenv("TP_BACKOFF_JITTER", "1.0"))


def _out(msg: str) -> None:
    ts = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{ts}] {msg}", flush=True)


def _find_near_kickoff_games(now_et: datetime) -> pd.DataFrame:
    """Return rows from the schedule whose kickoff falls within the scrape window."""
    if not WEEKLY_SCHEDULE_PATH.exists():
        _out(f"⚠️  Schedule not found: {WEEKLY_SCHEDULE_PATH}")
        return pd.DataFrame()

    try:
        sched = pd.read_csv(WEEKLY_SCHEDULE_PATH, low_memory=False)
    except Exception as e:
        _out(f"❌ Could not read schedule: {e}")
        return pd.DataFrame()

    if "startDateEastern" not in sched.columns:
        _out("⚠️  Schedule missing 'startDateEastern' column")
        return pd.DataFrame()

    kickoff_et = pd.to_datetime(sched["startDateEastern"], errors="coerce", utc=True)
    kickoff_et = kickoff_et.dt.tz_convert("America/New_York")

    window_start = now_et - timedelta(minutes=WINDOW_BEFORE_MIN)
    window_end = now_et + timedelta(minutes=WINDOW_AFTER_MIN)

    mask = kickoff_et.notna() & (kickoff_et >= window_start) & (kickoff_et <= window_end)
    near = sched[mask].copy()
    near["_kickoff_et"] = kickoff_et[mask].values
    return near


def _build_pricer() -> TickPickPricer:
    """Create a pricer instance configured for single-event lookups."""
    return TickPickPricer(
        team_urls=[],           # not used for get_summary_by_event_id
        output_dir=str(DAILY_DIR),
        polite_delay_range=(TP_POLITE_LOW, TP_POLITE_HIGH),
        retries=TP_RETRIES,
        timeout=TP_TIMEOUT,
        verbose=True,
        backoff_base_s=TP_BACKOFF_BASE,
        backoff_mult=TP_BACKOFF_MULT,
        backoff_max_s=TP_BACKOFF_MAX,
        backoff_jitter_s=TP_BACKOFF_JITTER,
    )


def _price_row(
    game: pd.Series,
    price_data: Dict[str, Any],
    now_et: datetime,
) -> Dict[str, Any]:
    """Map a schedule row + price lookup result into snapshot schema."""
    date_str = now_et.strftime("%Y-%m-%d")
    time_str = now_et.strftime("%H:%M:%S")

    event_id = str(game.get("id", "")).strip()
    home = str(game.get("homeTeam", "")).strip()
    away = str(game.get("awayTeam", "")).strip()
    title = f"{away} vs. {home}" if home and away else ""

    # Parse game date/time for the snapshot columns
    kickoff_raw = game.get("startDateEastern")
    try:
        ko = pd.to_datetime(kickoff_raw, utc=True).tz_convert("America/New_York")
        game_date = ko.strftime("%Y-%m-%d")
        game_time = ko.strftime("%H:%M")
    except Exception:
        game_date = None
        game_time = None

    return {
        "team_slug": None,
        "team_name": f"{home} Football" if home else None,
        "team_url": None,
        "event_id": event_id,
        "title": title,
        "home_team_guess": home or None,
        "away_team_guess": away or None,
        "date_local": game_date,
        "time_local": game_time,
        "offer_url": price_data.get("source_url"),
        "lowest_price": price_data.get("lowest_price"),
        "highest_price": None,
        "average_price": price_data.get("average_price"),
        "listing_count": price_data.get("listing_count"),
        "date_collected": date_str,
        "time_collected": time_str,
        "is_kickoff_snapshot": True,
        # Pass through enrichment columns that are already in the schedule
        "homeTeam": home or None,
        "awayTeam": away or None,
        "homeConference": game.get("homeConference") or None,
        "awayConference": game.get("awayConference") or None,
        "week": game.get("week") or None,
        "neutralSite": game.get("neutralSite") or None,
        "conferenceGame": game.get("conferenceGame") or None,
        "capacity": game.get("capacity") or None,
        "homeTeamRank": game.get("homeTeamRank") or None,
        "awayTeamRank": game.get("awayTeamRank") or None,
        "isRankedMatchup": game.get("isRankedMatchup") or None,
        "startDateEastern": kickoff_raw,
    }


def _backup_file(path: Path, keep: int = 7) -> None:
    if not path.exists():
        return
    backup_dir = DAILY_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.{ts}.bak"
    try:
        import shutil
        shutil.copy2(str(path), str(backup_path))
        # Prune oldest backups
        baks = sorted(backup_dir.glob(f"{path.name}.*.bak"))
        for old in baks[:-keep]:
            old.unlink(missing_ok=True)
    except Exception as e:
        _out(f"⚠️  Backup failed: {e}")


def _append_to_snapshot(new_rows: List[Dict[str, Any]]) -> None:
    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows)
    for c in ("lowest_price", "highest_price", "average_price"):
        if c in new_df.columns:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    if "listing_count" in new_df.columns:
        new_df["listing_count"] = pd.to_numeric(new_df["listing_count"], errors="coerce")

    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    key_cols = ["event_id", "date_collected", "time_collected"]

    if SNAPSHOT_PATH.exists():
        _backup_file(SNAPSHOT_PATH, keep=7)
        try:
            existing = pd.read_csv(SNAPSHOT_PATH, low_memory=False)
        except Exception as e:
            _out(f"⚠️  Could not read existing snapshot CSV: {e}; writing new rows only")
            existing = pd.DataFrame()

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in key_cols if c in combined.columns],
            keep="last",
        )
        tmp = str(SNAPSHOT_PATH) + ".__tmp__"
        combined.to_csv(tmp, index=False)
        os.replace(tmp, str(SNAPSHOT_PATH))
        _out(f"✅ Appended {len(new_df)} kickoff rows → {SNAPSHOT_PATH} (total: {len(combined)})")
    else:
        tmp = str(SNAPSHOT_PATH) + ".__tmp__"
        new_df.to_csv(tmp, index=False)
        os.replace(tmp, str(SNAPSHOT_PATH))
        _out(f"✅ Created snapshot with {len(new_df)} kickoff rows → {SNAPSHOT_PATH}")


def run() -> None:
    now_et = datetime.now(TIMEZONE)
    _out(f"🏈 Kickoff snapshot check — window: now ±{WINDOW_BEFORE_MIN}/{WINDOW_AFTER_MIN} min")

    near = _find_near_kickoff_games(now_et)
    if near.empty:
        _out("  ℹ️  No games kicking off in current window — nothing to scrape")
        write_status("kickoff_snapshot", "skipped", "No games in kickoff window")
        return

    _out(f"  🎯 {len(near)} game(s) in window:")
    for _, g in near.iterrows():
        _out(f"     • {g.get('awayTeam','?')} @ {g.get('homeTeam','?')} — kickoff {g.get('_kickoff_et','?')}")

    pricer = _build_pricer()
    new_rows: List[Dict[str, Any]] = []
    ok, fail = 0, 0

    for _, game in near.iterrows():
        event_id = game.get("id")
        if not event_id or pd.isna(event_id):
            _out(f"  ⚠️  Skipping game with missing event_id: {game.get('homeTeam')}")
            continue

        _out(f"  🔍 Fetching event {event_id}: {game.get('awayTeam')} @ {game.get('homeTeam')}")
        try:
            price_data = pricer.get_summary_by_event_id(int(event_id))
        except Exception as e:
            _out(f"  ❌ Fetch failed for event {event_id}: {e}")
            fail += 1
            continue

        if "_error" in price_data:
            _out(f"  ⚠️  No price data for event {event_id}: {price_data['_error']}")
            fail += 1
            continue

        row = _price_row(game, price_data, now_et)
        new_rows.append(row)
        _out(f"     💰 lowest=${price_data.get('lowest_price')}, avg=${price_data.get('average_price')}, listings={price_data.get('listing_count')}")
        ok += 1

    _out(f"  📦 Scraped {ok} ok / {fail} failed")

    if new_rows:
        _append_to_snapshot(new_rows)
        write_status(
            "kickoff_snapshot", "success",
            f"{ok} kickoff prices captured",
            {"scraped_ok": ok, "scraped_fail": fail, "rows_added": len(new_rows)},
        )
    else:
        write_status("kickoff_snapshot", "failed", f"0 rows captured ({fail} errors)")


if __name__ == "__main__":
    run()
