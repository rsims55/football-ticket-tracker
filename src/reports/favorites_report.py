#!/usr/bin/env python3
"""Send a daily email summary for favorited games.

Runs after each daily snapshot. Skips silently if no favorites exist
or if email credentials are missing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

FAVORITES_PATH  = ROOT / "data" / "permanent" / "favorites.json"
SNAPSHOT_PATH   = ROOT / "data" / "daily" / "price_snapshots.csv"
SENT_DATE_PATH  = ROOT / "data" / "daily" / ".favorites_report_last_sent"
SEASON_YEAR     = int(os.getenv("SEASON_YEAR", str(datetime.now().year)))


def _load_favorites() -> list[dict]:
    if not FAVORITES_PATH.exists():
        return []
    try:
        with open(FAVORITES_PATH, encoding="utf-8") as f:
            return json.load(f).get("favorites", [])
    except Exception:
        return []


def _load_snapshots() -> pd.DataFrame | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return pd.read_csv(SNAPSHOT_PATH, low_memory=False)
    except Exception:
        return None


def _norm_event_id(value) -> str:
    """Canonical event_id string.

    Snapshots store event_id as a float ("7647963.0") while favorites.json
    stores the plain int form ("7647963"); normalize both to the int form so
    they compare equal.
    """
    num = pd.to_numeric(value, errors="coerce")
    if pd.notna(num):
        return str(int(num))
    return str(value).strip()


def _norm_event_id_series(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    return num.astype("Int64").astype(str).where(num.notna(), s.astype(str).str.strip())


def _predicted_floor(row_dict: dict) -> float:
    """Get predicted floor price for a game row using the CatBoost model."""
    try:
        from gui.predict_trajectory import predict_for_times
        result = predict_for_times(row_dict, [pd.Timestamp.now()])
        if result:
            return float(result[0])
    except Exception:
        pass
    return float("nan")


def build_report(favorites: list[dict], snaps: pd.DataFrame) -> str:
    lines = [
        "# ⭐ Daily Favorites Price Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    now = datetime.now()

    def _sort_key(f):
        try:
            return pd.to_datetime(f.get("startDateEastern", "")).timestamp()
        except Exception:
            return float("inf")

    upcoming = [
        f for f in favorites
        if pd.to_datetime(f.get("startDateEastern", pd.NaT), errors="coerce") >= now
    ]
    if not upcoming:
        return "# ⭐ Daily Favorites Price Report\n\n_No upcoming favorited games._\n"

    # Index snapshots by event once rather than rescanning the full file per favorite.
    if "event_id" in snaps.columns:
        snaps_by_event = {
            eid: grp.reset_index(drop=True)
            for eid, grp in snaps.groupby(_norm_event_id_series(snaps["event_id"]))
        }
    else:
        snaps_by_event = {}

    for fav in sorted(upcoming, key=_sort_key):
        event_id = _norm_event_id(fav.get("event_id", ""))
        home = fav.get("homeTeam", "?")
        away = fav.get("awayTeam", "?")
        game_date = fav.get("startDateEastern", "")
        try:
            game_date_fmt = pd.to_datetime(game_date).strftime("%B %d, %Y")
        except Exception:
            game_date_fmt = game_date

        lines.append(f"## ⭐ {home} vs {away}")
        lines.append(f"**Game Date:** {game_date_fmt}")
        lines.append("")

        ev = snaps_by_event.get(event_id, pd.DataFrame()).copy()
        if ev.empty:
            lines.append("_No price data collected yet._")
            lines.append("")
            continue

        ev["lowest_price"] = pd.to_numeric(ev["lowest_price"], errors="coerce")
        ev["date_collected"] = pd.to_datetime(ev["date_collected"], errors="coerce")

        # Today's price (most recent snapshot)
        latest = ev.sort_values("date_collected").iloc[-1]
        today_price = float(latest["lowest_price"]) if pd.notna(latest["lowest_price"]) else float("nan")

        # All-time lowest observed
        obs_min = float(ev["lowest_price"].min()) if ev["lowest_price"].notna().any() else float("nan")
        obs_min_row = ev.loc[ev["lowest_price"].idxmin()] if ev["lowest_price"].notna().any() else None
        obs_min_date = ""
        if obs_min_row is not None and pd.notna(obs_min_row.get("date_collected")):
            obs_min_date = f" (on {pd.to_datetime(obs_min_row['date_collected']).strftime('%Y-%m-%d')})"

        # Predicted floor
        row_dict = latest.to_dict()
        predicted = _predicted_floor(row_dict)

        def _fmt(v):
            return f"${v:,.2f}" if np.isfinite(v) else "—"

        lines.append(f"- **Current price:** {_fmt(today_price)}")
        lines.append(f"- **Lowest observed:** {_fmt(obs_min)}{obs_min_date}")
        lines.append(f"- **Predicted floor:** {_fmt(predicted)}")

        if np.isfinite(today_price) and np.isfinite(predicted) and today_price <= predicted:
            lines.append("")
            lines.append("> ⚠️ Current price is already at or below the predicted floor — may be a good time to buy.")

        lines.append("")

    return "\n".join(lines)


def _already_sent_today() -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return SENT_DATE_PATH.exists() and SENT_DATE_PATH.read_text(encoding="utf-8").strip() == today
    except Exception:
        return False


def _mark_sent_today() -> None:
    try:
        SENT_DATE_PATH.write_text(datetime.now().strftime("%Y-%m-%d"), encoding="utf-8")
    except Exception:
        pass


def send_favorites_report(favorites: list[dict] | None = None,
                          snaps: pd.DataFrame | None = None) -> None:
    if _already_sent_today():
        print("[favorites_report] Already sent today — skipping.")
        return

    favorites = _load_favorites() if favorites is None else favorites
    if not favorites:
        print("[favorites_report] No favorites — skipping.")
        return

    if snaps is None:
        snaps = _load_snapshots()
    if snaps is None:
        print("[favorites_report] Snapshot file missing — skipping.")
        return

    md_text = build_report(favorites, snaps)

    gmail = os.getenv("GMAIL_ADDRESS")
    to_email = os.getenv("TO_EMAIL")
    app_pw = os.getenv("GMAIL_APP_PASSWORD")
    if not all([gmail, to_email, app_pw]):
        print("[favorites_report] Email credentials missing — skipping send.")
        print(md_text)
        return

    try:
        from reports.send_email import send_markdown_report
        subject = f"⭐ Favorites Price Report — {datetime.now().strftime('%Y-%m-%d')}"
        send_markdown_report(md_text, subject)
        _mark_sent_today()
        print(f"[favorites_report] Sent to {to_email}")
    except Exception as e:
        print(f"[favorites_report] Send failed: {e}")


def main() -> None:
    """Load shared inputs once, then: (1) targeted price texts every run,
    (2) the daily favorites summary (guarded to once per day)."""
    favorites = _load_favorites()
    snaps = _load_snapshots()

    # 1. Targeted NEW LOW / PRICE UP alerts for games <= 7 days out.
    #    Runs on every snapshot and keeps its own dedupe state, so it must not sit
    #    behind the once-a-day guard in send_favorites_report().
    try:
        from reports.favorites_price_alerts import run_alerts
        run_alerts(favorites=favorites, snaps=snaps)
    except Exception as e:
        print(f"[favorites_report] price alerts failed: {e}")

    # 2. Daily summary e-mail.
    send_favorites_report(favorites=favorites, snaps=snaps)


if __name__ == "__main__":
    main()
