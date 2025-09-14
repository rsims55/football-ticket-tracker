# src/tools/backfill_point_diffs.py

import os
import sys
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np

WEEKLY_DIR = os.path.join("data", "weekly")
ALIASES_PATHS = [
    os.path.join("data", "permanent", "team_aliases.json"),
    os.path.join("data", "permanent", "team_aliases_v2.json"),
    os.path.join("data", "permanent", "aliases.json"),
]

# ---------- Alias + normalization ----------
def _simple_key(s: str) -> str:
    """
    Very conservative normalizer to improve matching when no alias map exists.
    - lowercase
    - strip
    - remove common suffixes like "football"
    - collapse whitespace
    - remove parentheticals
    Does NOT drop mascots/team names (avoids "tigers" -> ""), but helps with obvious cruft.
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    x = s.strip()

    # Drop parentheticals: "Miami (FL)" -> "Miami FL"
    x = x.replace("(", " ").replace(")", " ")
    # Remove obvious cruft
    for token in [" football", " men's", " women’s", " women'", " men'", " men’s", " women"]:
        x = x.replace(token, "")
    # Normalize separators
    x = x.replace("&", "and").replace("’", "'")
    # Lower + collapse spaces
    x = " ".join(x.lower().split())
    return x


def _load_alias_map() -> dict[str, str]:
    """
    Load an alias map if available. Supports two shapes:
    1) { "canonical": ["alias1","alias2", ...], ... }
    2) { "alias1": "canonical", "alias2":"canonical", ... }
    Returns a dict mapping *normalized key* -> canonical (original casing as in file).
    """
    for p in ALIASES_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                alias_to_canonical: dict[str, str] = {}
                # detect shape
                if raw and isinstance(next(iter(raw.values())), list):
                    # shape 1: canonical -> [aliases]
                    for canonical, aliases in raw.items():
                        # include canonical as its own alias too
                        alias_to_canonical[_simple_key(canonical)] = canonical
                        for a in aliases:
                            alias_to_canonical[_simple_key(a)] = canonical
                else:
                    # shape 2: alias -> canonical
                    for a, canonical in raw.items():
                        alias_to_canonical[_simple_key(a)] = canonical
                        # also include canonical->canonical
                        alias_to_canonical.setdefault(_simple_key(canonical), canonical)
                print(f"🔖 Loaded team aliases from {p} ({len(alias_to_canonical)} entries)")
                return alias_to_canonical
            except Exception as e:
                print(f"⚠️ Failed to read alias file {p}: {e}")

    print("ℹ️ No alias file found; using simple normalization only.")
    return {}


def _canonicalize(series: pd.Series, alias_map: dict[str, str], new_col_name: str) -> pd.Series:
    """
    Map raw team text -> canonical using alias_map; if not found, fall back to normalized string.
    Preserve non-null types; return readable canonical values.
    """
    keys = series.astype(str).fillna("").map(_simple_key)
    if alias_map:
        mapped = keys.map(lambda k: alias_map.get(k) or series.loc[keys.index[keys == k][0]])
        # If still unmapped, fall back to a title-cased normalized string
        fallback = keys.map(lambda k: alias_map.get(k) or " ".join(k.split()).title())
        # Prefer mapped (which returns original casing), else fallback
        out = pd.Series(np.where(keys.map(lambda k: k in alias_map), mapped, fallback), index=series.index)
    else:
        # No alias map: just a tidy title-case of normalized key
        out = keys.map(lambda k: " ".join(k.split()).title())
    out = out.rename(new_col_name)
    return out

# ---------- Games loader ----------
def _load_completed_games_long() -> pd.DataFrame:
    """
    Load all completed_games_*.csv and return a long-form dataframe:
      columns: team, opponent, startDate (UTC, datetime64[ns, UTC]), pointDiff
    """
    paths = sorted(glob.glob(os.path.join(WEEKLY_DIR, "completed_games_*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No completed games files found in {WEEKLY_DIR}. "
            "Run src/fetchers/results_fetcher.py first."
        )

    frames = []
    for p in paths:
        df = pd.read_csv(p)

        # Normalize/ensure required cols
        for col_old, col_new in (("home_points", "homePoints"), ("away_points", "awayPoints")):
            if col_old in df.columns and col_new not in df.columns:
                df[col_new] = df[col_old]
        if "startDate" not in df.columns:
            df["startDate"] = pd.NaT
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)

        req = {"homeTeam", "awayTeam", "homePoints", "awayPoints", "startDate"}
        missing = req - set(df.columns)
        if missing:
            print(f"⚠️ Skipping {os.path.basename(p)}; missing columns: {sorted(missing)}")
            continue

        # Completed games only (both points present)
        done = df["homePoints"].notna() & df["awayPoints"].notna()
        df = df.loc[done].copy()
        if df.empty:
            continue

        df["homePointDiff"] = df["homePoints"] - df["awayPoints"]
        df["awayPointDiff"] = -df["homePointDiff"]

        long_home = df.rename(
            columns={"homeTeam": "team", "awayTeam": "opponent", "homePointDiff": "pointDiff"}
        )[["team", "opponent", "startDate", "pointDiff"]].copy()
        long_away = df.rename(
            columns={"awayTeam": "team", "homeTeam": "opponent", "awayPointDiff": "pointDiff"}
        )[["team", "opponent", "startDate", "pointDiff"]].copy()
        long_df = pd.concat([long_home, long_away], ignore_index=True)
        frames.append(long_df)

    if not frames:
        raise RuntimeError("No usable completed games rows found.")

    long_all = pd.concat(frames, ignore_index=True)
    # Normalize whitespace/strings
    for c in ("team", "opponent"):
        long_all[c] = long_all[c].astype(str).str.strip()
    # Ensure timezone-aware UTC dtype + stable sort
    long_all["startDate"] = pd.to_datetime(long_all["startDate"], utc=True)
    long_all = long_all.sort_values(["team", "startDate"], kind="mergesort").reset_index(drop=True)
    return long_all

# ---------- Snapshot ts ----------
def _parse_snapshot_ts(df_snap: pd.DataFrame) -> pd.Series:
    """
    Build a timezone-aware UTC timestamp for each snapshot row from date_collected & time_collected.
    Assumes date/time are local Eastern. If time_collected is missing, assume 12:00 PM ET.
    """
    time_fallback = "12:00:00"
    date_str = df_snap["date_collected"].astype(str).str.strip()
    time_raw = df_snap.get("time_collected", pd.Series([""] * len(df_snap), index=df_snap.index))
    time_str = time_raw.astype(str).str.strip()
    time_str = np.where((time_str == "") | (pd.Series(time_str).isna()), time_fallback, time_str)

    def _combine(dt, tm):
        s = f"{dt} {tm}"
        try:
            return pd.Timestamp(s, tz="America/New_York")
        except Exception:
            try:
                ts = pd.to_datetime(s, errors="coerce")
                if pd.isna(ts):
                    return pd.NaT
                if ts.tzinfo is not None:
                    return ts.tz_convert("America/New_York")
                return ts.tz_localize("America/New_York")
            except Exception:
                return pd.NaT

    ts_local = pd.Series([_combine(d, t) for d, t in zip(date_str, time_str)], index=df_snap.index)
    return ts_local.dt.tz_convert("UTC")

# ---------- Per-team asof ----------
def _merge_asof_per_team(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.Series:
    """
    left_df: columns ['team_key','snapshot_ts_utc']
    right_df: columns ['team_key','game_ts_utc','pointDiff']
    Returns Series of pointDiff aligned to left_df.index.
    """
    out_parts = []
    for key, lgrp in left_df.groupby("team_key", sort=False):
        rgrp = right_df[right_df["team_key"] == key]
        if rgrp.empty:
            out_parts.append(pd.Series(np.nan, index=lgrp.index))
            continue

        lgood = lgrp["snapshot_ts_utc"].notna()
        lbad_idx = lgrp.index[~lgood]

        l_sorted = lgrp.loc[lgood].sort_values("snapshot_ts_utc", kind="mergesort")
        r_sorted = rgrp.sort_values("game_ts_utc", kind="mergesort")

        merged = pd.merge_asof(
            left=l_sorted,
            right=r_sorted,
            left_on="snapshot_ts_utc",
            right_on="game_ts_utc",
            direction="backward",
            allow_exact_matches=True,
        )

        ser = pd.Series(merged["pointDiff"].values, index=l_sorted.index)
        if len(lbad_idx):
            ser = pd.concat([ser, pd.Series(np.nan, index=lbad_idx)])
        ser = ser.reindex(lgrp.index)
        out_parts.append(ser)

    return pd.concat(out_parts).reindex(left_df.index)

# ---------- Main ----------
def backfill_point_diffs(
    snapshots_csv: str,
    output_csv: str | None = None,
    overwrite: bool = True,
    backup: bool = True,
) -> str:
    """
    For each snapshot row, attach the most recent point differential for both teams
    as of the snapshot collection timestamp (no leakage).
    Adds columns:
      - home_last_point_diff_at_snapshot
      - away_last_point_diff_at_snapshot
    """
    if not os.path.exists(snapshots_csv):
        raise FileNotFoundError(f"Snapshots file not found: {snapshots_csv}")

    print(f"📥 Loading snapshots: {snapshots_csv}")
    snaps = pd.read_csv(snapshots_csv)

    needed_cols = {"homeTeam", "awayTeam", "date_collected"}
    missing = needed_cols - set(snaps.columns)
    if missing:
        raise RuntimeError(f"Snapshots CSV missing required columns: {sorted(missing)}")

    # Normalize team strings
    snaps["homeTeam"] = snaps["homeTeam"].astype(str).str.strip()
    snaps["awayTeam"] = snaps["awayTeam"].astype(str).str.strip()

    # Parse snapshot timestamp (ET -> UTC)
    snaps["snapshot_ts_utc"] = _parse_snapshot_ts(snaps)
    if snaps["snapshot_ts_utc"].isna().any():
        bad = snaps["snapshot_ts_utc"].isna().sum()
        print(f"⚠️ {bad} rows have unparseable snapshot timestamps; they will get NaN diffs.")

    # Load completed results (team, opponent, startDate, pointDiff)
    print("📚 Loading completed games timelines…")
    long_all = _load_completed_games_long()

    # ===== Canonicalize team names on BOTH sides =====
    alias_map = _load_alias_map()

    snaps["homeTeam_key"] = _canonicalize(snaps["homeTeam"], alias_map, "homeTeam_key")
    snaps["awayTeam_key"] = _canonicalize(snaps["awayTeam"], alias_map, "awayTeam_key")

    long_all["team_key"] = _canonicalize(long_all["team"], alias_map, "team_key")

    # ---- Diagnostics: overlap BEFORE merge ----
    print("\n🔎 Diagnostics")
    print("snapshots rows:", len(snaps))
    print("completed rows:", len(long_all))
    print("snapshot_ts_utc NaT %:", round(float(snaps['snapshot_ts_utc'].isna().mean()) * 100, 2))
    print("startDate NaT %    :", round(float(long_all['startDate'].isna().mean()) * 100, 2))

    snap_teams_raw = pd.unique(pd.concat([snaps['homeTeam'], snaps['awayTeam']], ignore_index=True))
    long_teams_raw = long_all['team'].unique()
    print("snap unique teams (raw):", len(snap_teams_raw))
    print("games unique teams (raw):", len(long_teams_raw))

    snap_keys = pd.unique(pd.concat([snaps['homeTeam_key'], snaps['awayTeam_key']], ignore_index=True))
    long_keys = long_all['team_key'].unique()
    common = set(snap_keys).intersection(set(long_keys))
    print("snap unique team_keys  :", len(snap_keys))
    print("games unique team_keys :", len(long_keys))
    print("team_key overlap       :", len(common))
    if len(common) == 0:
        only_snap = list(set(snap_keys) - set(long_keys))[:5]
        only_long = list(set(long_keys) - set(snap_keys))[:5]
        print("example only-in-snap key:", only_snap)
        print("example only-in-long key:", only_long)

    # --- Build right-hand long results using canonical key ---
    right_long = (
        long_all.rename(columns={"startDate": "game_ts_utc"})[["team_key", "game_ts_utc", "pointDiff"]].copy()
    )

    # --- Home join (per-team key) ---
    left_home = snaps.loc[:, ["homeTeam_key", "snapshot_ts_utc"]].rename(columns={"homeTeam_key": "team_key"}).copy()
    snaps["home_last_point_diff_at_snapshot"] = _merge_asof_per_team(left_home, right_long).values

    # --- Away join (per-team key) ---
    left_away = snaps.loc[:, ["awayTeam_key", "snapshot_ts_utc"]].rename(columns={"awayTeam_key": "team_key"}).copy()
    snaps["away_last_point_diff_at_snapshot"] = _merge_asof_per_team(left_away, right_long).values

    # Decide output path
    if output_csv is None:
        output_csv = snapshots_csv

    # Backup original BEFORE overwrite if same path
    if backup and overwrite and output_csv == snapshots_csv and os.path.exists(snapshots_csv):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = snapshots_csv.replace(".csv", f".bak_{ts}.csv")
        snaps.to_csv(backup_path, index=False)
        print(f"🧯 Backup written → {backup_path}")

    # Save
    if not overwrite and os.path.exists(output_csv):
        raise FileExistsError(f"File exists and overwrite=False: {output_csv}")

    snaps.to_csv(output_csv, index=False)
    print(f"✅ Backfilled diffs saved → {output_csv}")
    return output_csv


if __name__ == "__main__":
    default_snapshots = os.path.join("data", "daily", "price_snapshots.csv")
    in_path = sys.argv[1] if len(sys.argv) > 1 else default_snapshots
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    backfill_point_diffs(in_path, out_path, overwrite=True, backup=True)
