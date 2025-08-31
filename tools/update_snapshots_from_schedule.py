#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix previously-unmatched price snapshot rows by rematching them to the full schedule
(using the latest team aliases), and overwrite those rows back into price_snapshots.csv.

- Reads:
    --snap       path/to/price_snapshots.csv  (full file; will be updated in place)
    --unmatched  path/to/unmatched_price_rows.csv  (only the ones you want fixed)
    --schedule   path/to/full_2025_schedule.csv   (read-only)
    --aliases    path/to/team_aliases.json        (read-only)

- Strategy:
    1) Normalize team names using aliases (snapshot & schedule).
    2) Build game_date for both.
    3) Match unmatched rows to schedule in this order:
        a) (home_canon, away_canon, game_date)
        b) (away_canon, home_canon, game_date)  # swapped
        c) From title parse "X vs. Y" and retry (a) then (b)
        d) Single-team date match: (home_canon, *) or (*, away_canon) on game_date
    4) If a match is found, copy schedule fields into the snapshot row and
       standardize home/away team names to canonical.
    5) Write updated snapshots back to --snap.
    6) Any still-unmatched are written back to --unmatched.

- Schedule file is NEVER modified.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, Tuple, Optional

import pandas as pd


# -------------------- helpers

def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum())


def load_aliases(path: Path) -> Dict[str, str]:
    """
    JSON: { "Canonical Team": ["Alias1","Alias2",...], ... }
    Returns alias->canonical mapping (normalized keys).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for canon, aliases in data.items():
        out[_norm(canon)] = canon
        for a in aliases or []:
            out[_norm(a)] = canon
    return out


def canon_team(name: str, alias_to_canon: Dict[str, str]) -> str:
    n = _norm(name)
    return alias_to_canon.get(n, name)


def coerce_date_from_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    out = pd.Series([pd.NaT] * len(df))
    for c in cols:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=False)
            out = out.fillna(dt)
    return pd.to_datetime(out, errors="coerce", utc=False).dt.date


def parse_title_for_teams(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to pull "Home vs. Away" team names out of the snapshot title.
    Returns (home, away) or (None, None) if not parseable.
    """
    if not isinstance(title, str):
        return (None, None)
    t = title.replace(" Vs. ", " vs. ").replace(" VS. ", " vs. ").replace(" VS ", " vs. ").replace(" Vs ", " vs. ")
    if " vs. " in t:
        parts = t.split(" vs. ")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return (None, None)


def mdy_str(d) -> str:
    """
    format date like '8/30/2025' (no leading zeros)
    """
    if pd.isna(d):
        return ""
    return f"{d.month}/{d.day}/{d.year}"


def hhmm_str(dt_like) -> str:
    """
    Produce 'HH:MM' from various schedule time sources.
    """
    if pd.isna(dt_like) or dt_like is None:
        return ""
    # try datetime-like
    try:
        dt = pd.to_datetime(dt_like, errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%H:%M")
    except Exception:
        pass
    # try '15:00' strings
    s = str(dt_like)
    if ":" in s and len(s.split(":")[0]) <= 2:
        hh, mm = s.split(":")[0:2]
        if hh.isdigit() and mm.isdigit():
            return f"{int(hh):02d}:{int(mm):02d}"
    return ""


# snapshot <- schedule fields to copy
FIELD_MAP = (
    ("week", "week"),
    ("homeConference", "homeConference"),
    ("awayConference", "awayConference"),
    ("neutralSite", "neutralSite"),
    ("conferenceGame", "conferenceGame"),
    ("isRankedMatchup", "isRankedMatchup"),
    ("homeTeamRank", "homeTeamRank"),
    ("awayTeamRank", "awayTeamRank"),
    ("stadium", "venue"),
    ("capacity", "capacity"),
    ("attendance", "attendance"),
    ("notes", "notes"),
)


def bool_to_str(b) -> str:
    # snapshots use 'TRUE'/'FALSE' strings
    if isinstance(b, str):
        # keep as-is if already 'TRUE'/'FALSE'
        up = b.strip().upper()
        if up in ("TRUE", "FALSE"):
            return up
    return "TRUE" if bool(b) else "FALSE"


# -------------------- preparation

def prep_schedule(s: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    for c in ("homeTeam", "awayTeam"):
        if c not in s.columns:
            raise ValueError(f"Schedule missing column: {c}")
        s[c] = s[c].astype("string")

    s["home_canon"] = s["homeTeam"].map(lambda x: canon_team(x, alias_map))
    s["away_canon"] = s["awayTeam"].map(lambda x: canon_team(x, alias_map))

    # create game_date
    s["game_date"] = coerce_date_from_cols(s, ["game_date", "startDateEastern", "startDate"])

    # best-effort time column for snapshots' time_local
    # prefer kickoffTimeStr, then startDateEastern's time portion if present
    s["kick_time"] = ""
    if "kickoffTimeStr" in s.columns:
        s["kick_time"] = s["kickoffTimeStr"].apply(hhmm_str)
    if "startDateEastern" in s.columns:
        t2 = s["startDateEastern"].apply(hhmm_str)
        s["kick_time"] = s["kick_time"].mask(s["kick_time"] == "", t2)

    # tidy for join
    s = s.drop_duplicates(subset=["home_canon", "away_canon", "game_date"], keep="first").reset_index(drop=True)
    return s


def prep_unmatched(u: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    # Ensure columns exist
    for c in ("homeTeam", "awayTeam", "title", "date_local"):
        if c not in u.columns:
            u[c] = ""

    u["homeTeam"] = u["homeTeam"].astype("string")
    u["awayTeam"] = u["awayTeam"].astype("string")
    u["title"] = u["title"].astype("string")

    # if home/away missing, try to parse from title
    parsed = u["title"].apply(parse_title_for_teams)
    u["home_filled"] = [h or "" for h, _ in parsed]
    u["away_filled"] = [a or "" for _, a in parsed]
    u["homeTeam"] = u["homeTeam"].mask(u["homeTeam"].eq(""), u["home_filled"])
    u["awayTeam"] = u["awayTeam"].mask(u["awayTeam"].eq(""), u["away_filled"])

    # canonical names
    u["home_canon"] = u["homeTeam"].map(lambda x: canon_team(x, alias_map))
    u["away_canon"] = u["awayTeam"].map(lambda x: canon_team(x, alias_map))
    u["game_date"] = pd.to_datetime(u["date_local"], errors="coerce").dt.date
    return u


# -------------------- matching logic

def match_row(row: pd.Series, sched: pd.DataFrame) -> Optional[pd.Series]:
    """Return the schedule row that matches, or None."""
    h, a, d = row["home_canon"], row["away_canon"], row["game_date"]

    def pick(df) -> Optional[pd.Series]:
        if df.empty:
            return None
        if len(df) == 1:
            return df.iloc[0]
        # If multiple, try to use away/home hints from parsed title, otherwise first
        return df.iloc[0]

    # a) strict
    m = sched[(sched.home_canon == h) & (sched.away_canon == a) & (sched.game_date == d)]
    r = pick(m)
    if r is not None:
        return r

    # b) swapped
    m = sched[(sched.home_canon == a) & (sched.away_canon == h) & (sched.game_date == d)]
    r = pick(m)
    if r is not None:
        return r

    # c) if home/away were blank originally but title parsed differently, we already replaced above.
    #    try single-team date matches:
    m = sched[(sched.home_canon == h) & (sched.game_date == d)]
    r = pick(m)
    if r is not None:
        return r

    m = sched[(sched.away_canon == a) & (sched.game_date == d)]
    r = pick(m)
    if r is not None:
        return r

    # last-ditch: either side equals either team on date
    m = sched[((sched.home_canon == h) | (sched.away_canon == h) |
               (sched.home_canon == a) | (sched.away_canon == a)) &
              (sched.game_date == d)]
    r = pick(m)
    return r


def apply_schedule_fields(snap_row: pd.Series, sch_row: pd.Series) -> pd.Series:
    # overwrite snapshot's home/away to canonical from schedule (ensures consistency)
    snap_row["homeTeam"] = sch_row["home_canon"]
    snap_row["awayTeam"] = sch_row["away_canon"]

    # set date/time from schedule when available
    if not pd.isna(sch_row.get("game_date")):
        snap_row["date_local"] = mdy_str(sch_row["game_date"])
    if "kick_time" in sch_row and sch_row["kick_time"]:
        snap_row["time_local"] = sch_row["kick_time"]

    # copy mapped fields
    for tgt, src in FIELD_MAP:
        if src in sch_row.index:
            val = sch_row[src]
            if tgt in ("neutralSite", "conferenceGame", "isRankedMatchup"):
                val = bool_to_str(val)
            snap_row[tgt] = val

    return snap_row


# -------------------- main

def main():
    ap = argparse.ArgumentParser(description="Fix previously-unmatched snapshot rows using schedule + aliases.")
    ap.add_argument("--snap", required=True, help="Path to price_snapshots.csv (will be updated in place).")
    ap.add_argument("--unmatched", required=True, help="Path to unmatched_price_rows.csv (input + will be overwritten).")
    ap.add_argument("--schedule", required=True, help="Path to full_2025_schedule.csv (read-only).")
    ap.add_argument("--aliases", required=True, help="Path to team_aliases.json (read-only).")
    ap.add_argument("--backup", action="store_true", help="Create a timestamped backup of snapshots before writing.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs.")
    args = ap.parse_args()

    snap_path = Path(args.snap)
    um_path = Path(args.unmatched)
    sch_path = Path(args.schedule)
    al_path = Path(args.aliases)

    if args.verbose:
        print(f"Reading snapshots: {snap_path}")
        print(f"Reading unmatched: {um_path}")
        print(f"Reading schedule : {sch_path}")
        print(f"Reading aliases  : {al_path}")

    # load
    alias_map = load_aliases(al_path)
    snapshots = pd.read_csv(snap_path, dtype="unicode", keep_default_na=False, na_values=[""])
    unmatched = pd.read_csv(um_path, dtype="unicode", keep_default_na=False, na_values=[""])
    schedule = pd.read_csv(sch_path, dtype="unicode", keep_default_na=False, na_values=[""])

    schedule = prep_schedule(schedule, alias_map)
    unmatched = prep_unmatched(unmatched, alias_map)

    # choose an identity key to locate rows in snapshots to overwrite
    # Prefer event_id; fallback to (team_slug, offer_url, date_local, title)
    key_cols = []
    if "event_id" in snapshots.columns and "event_id" in unmatched.columns:
        key_cols = ["event_id"]
    else:
        for c in ("team_slug", "offer_url", "date_local", "title"):
            if c not in snapshots.columns or c not in unmatched.columns:
                raise ValueError(f"Missing key column '{c}' in snapshots or unmatched; cannot locate rows to update.")
        key_cols = ["team_slug", "offer_url", "date_local", "title"]

    if args.verbose:
        print(f"Using key columns to locate rows: {key_cols}")

    # index snapshots by key for fast overwrite
    snapshots_index = snapshots.set_index(key_cols, drop=False)

    fixed_rows = []
    still_unmatched = []

    for _, um_row in unmatched.iterrows():
        sch_row = match_row(um_row, schedule)
        if sch_row is None:
            still_unmatched.append(um_row)
            continue

        # apply schedule fields to a *copy* of the unmatched snapshot row
        fixed = apply_schedule_fields(um_row.copy(), sch_row)
        fixed_rows.append(fixed)

    fixed_df = pd.DataFrame(fixed_rows)
    remain_df = pd.DataFrame(still_unmatched)

    # overwrite fixed rows back into snapshots
    if len(fixed_df):
        # ensure the fixed_df has all snapshot columns (fill any missing with empty)
        for c in snapshots.columns:
            if c not in fixed_df.columns:
                fixed_df[c] = ""

        fixed_df = fixed_df[snapshots.columns]  # same column order
        fixed_df_indexed = fixed_df.set_index(key_cols, drop=False)

        # perform overwrite
        common_keys = fixed_df_indexed.index.intersection(snapshots_index.index)
        if args.verbose:
            print(f"Fixed matches to write: {len(common_keys)}")

        snapshots.loc[snapshots_index.loc[common_keys].index] = fixed_df_indexed.loc[common_keys].values

    # write snapshots (with optional backup)
    if args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = snap_path.with_suffix(snap_path.suffix + f".bak.{ts}")
        snapshots.to_csv(bak, index=False)
        if args.verbose:
            print(f"Backup written: {bak}")

    snapshots.to_csv(snap_path, index=False)
    if args.verbose:
        print(f"Wrote updated snapshots: {snap_path}")

    # write the remaining unmatched (overwrite the same file)
    remain_df.to_csv(um_path, index=False)
    fixed_count = len(fixed_df)
    remain_count = len(remain_df)
    if args.verbose:
        print(f"Fixed from unmatched: {fixed_count}")
        print(f"Still unmatched: {remain_count}")


if __name__ == "__main__":
    main()
