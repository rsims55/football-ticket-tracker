#!/usr/bin/env python3
from __future__ import annotations

import os, re, sys, json, uuid, argparse, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


# ---------- Config / defaults ----------
PROJ_DIR = Path(__file__).resolve().parents[1]  # repo root if script in scripts/
DATA_DIR = PROJ_DIR / "data"
PERMANENT_DIR = DATA_DIR / "permanent"
WEEKLY_DIR = DATA_DIR / "weekly"
DAILY_DIR = DATA_DIR / "daily"

ALIASES_PATH = PERMANENT_DIR / "team_aliases.json"
SNAPSHOTS_PATH = DAILY_DIR / "price_snapshots.csv"

YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))


# ---------- Helpers ----------
def atomic_csv_write(df: pd.DataFrame, path: Path, retries: int = 5, delay: float = 0.6) -> None:
    """Atomic-ish write with temp+replace; Windows-share tolerant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    for i in range(retries):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except PermissionError as e:
            print(f"⚠️  PermissionError writing {path}: attempt {i+1}/{retries} – {e}")
            time.sleep(delay)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
    # Fallback if still locked
    alt = path.with_name(f"{path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    print(f"❌ Could not replace {path}. Writing fallback: {alt}")
    df.to_csv(alt, index=False)


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup-{ts}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.replace(backup)
        print(f"💾 Backup written: {backup}")
    return backup


def to_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.fillna(default).astype(int)
    return x


def load_alias_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Alias map not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # normalize keys/values as strings
    return {str(k): str(v) for k, v in m.items()}


def canon(name: str, alias_map: Dict[str, str]) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    name = str(name).strip()
    return alias_map.get(name, name)


def latest_rankings_csv(weekly_dir: Path, year: int) -> Optional[Path]:
    candidates = list(weekly_dir.glob(f"wiki_rankings_{year}_*.csv"))
    if not candidates:
        return None
    # newest by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_rank_lookup(rankings_csv: Path, alias_map: Dict[str, str]) -> Dict[tuple, int]:
    """
    Returns {(week:int, canonical_team:str): rank:int}
    Accepts rankings CSV produced by RankingsFetcher (columns include: week, rank, school).
    """
    df = pd.read_csv(rankings_csv)
    if "rank" not in df.columns:
        raise KeyError(f"{rankings_csv.name} missing 'rank' column")
    # Some older files used 'school', newer code may rename to 'rank_school'. Handle both.
    school_col = "school" if "school" in df.columns else ("rank_school" if "rank_school" in df.columns else None)
    if not school_col:
        raise KeyError(f"{rankings_csv.name} missing 'school' (or 'rank_school') column")

    # Ensure week exists, default to 0 (preseason)
    if "week" not in df.columns:
        df["week"] = 0
    df["week"] = to_int_series(df["week"], default=0)

    df["team_alias"] = df[school_col].map(lambda s: canon(s, alias_map))
    df = df[["week", "team_alias", "rank"]].dropna(subset=["team_alias"])
    # Keep only top-25 ints
    df["rank"] = to_int_series(df["rank"], default=0)
    df = df[(df["rank"] >= 1) & (df["rank"] <= 25)]

    lookup = {(int(w), str(t)): int(r) for w, t, r in df.itertuples(index=False, name=None)}
    return lookup


def apply_ranks_to_snapshots(
    snapshots_csv: Path,
    out_csv: Optional[Path],
    alias_map: Dict[str, str],
    rank_lookup: Dict[tuple, int],
) -> pd.DataFrame:
    """
    Updates homeTeamRank/awayTeamRank/isRankedMatchup in price_snapshots.csv using rank_lookup.
    Uses week-offset rule: ranking_week = max(week - 1, 0).
    Writes back to snapshots_csv (or out_csv if provided), after backing up the original.
    """
    if not snapshots_csv.exists():
        raise FileNotFoundError(f"Snapshots file not found at {snapshots_csv}")

    df = pd.read_csv(snapshots_csv)

    # Validate required columns
    required_cols = {"week", "homeTeam", "awayTeam"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"{snapshots_csv.name} is missing required columns: {sorted(missing)}")

    # Make sure rank columns exist
    for col in ["homeTeamRank", "awayTeamRank", "isRankedMatchup"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Canonicalize for lookup (but DO NOT overwrite the original team columns)
    df["home_alias"] = df["homeTeam"].map(lambda s: canon(s, alias_map))
    df["away_alias"] = df["awayTeam"].map(lambda s: canon(s, alias_map))

    # Rankings week = max(week - 1, 0)
    df["week"] = to_int_series(df["week"], default=1)  # treat missing as 1 => rankings_week 0
    df["rankings_week"] = (df["week"] - 1).clip(lower=0)

    # Vectorized lookup by building keys and mapping
    def map_rank(side_alias_col: str) -> pd.Series:
        keys = list(zip(df["rankings_week"].astype(int), df[side_alias_col].astype(str)))
        return pd.Series([rank_lookup.get(k) for k in keys], index=df.index, dtype="float")

    df["homeTeamRank_new"] = map_rank("home_alias")
    df["awayTeamRank_new"] = map_rank("away_alias")

    # Assign for all rows (one-time patch; overwrite existing values)
    df["homeTeamRank"] = df["homeTeamRank_new"]
    df["awayTeamRank"] = df["awayTeamRank_new"]
    df.drop(columns=["homeTeamRank_new", "awayTeamRank_new"], inplace=True)

    # Update isRankedMatchup
    df["isRankedMatchup"] = df["homeTeamRank"].notna() & df["awayTeamRank"].notna()

    # Clean temp columns
    df.drop(columns=["home_alias", "away_alias", "rankings_week"], inplace=True)

    # Write out
    if out_csv is None or out_csv.resolve() == snapshots_csv.resolve():
        backup_file(snapshots_csv)
        atomic_csv_write(df, snapshots_csv)
        print(f"✅ Snapshots updated in-place: {snapshots_csv}")
    else:
        atomic_csv_write(df, out_csv)
        print(f"✅ Snapshots written to: {out_csv}")
    return df


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="One-time patch: update price_snapshots.csv with latest rankings using team aliases and week offset (rankings_week = max(week-1, 0))."
    )
    p.add_argument("--season-year", type=int, default=YEAR, help="Season year (default: env SEASON_YEAR or current year)")
    p.add_argument("--rankings", type=str, default="", help="Path to rankings CSV (defaults to newest wiki_rankings_<YEAR>_*.csv)")
    p.add_argument("--snapshots", type=str, default=str(SNAPSHOTS_PATH), help="Path to price_snapshots.csv")
    p.add_argument("--aliases", type=str, default=str(ALIASES_PATH), help="Path to team_aliases.json")
    p.add_argument("--out", type=str, default="", help="Optional different output path (default: overwrite snapshots after backup)")
    args = p.parse_args()

    season_year = int(args.season_year)
    snapshots_csv = Path(args.snapshots)
    aliases_json = Path(args.aliases)
    out_csv = Path(args.out).resolve() if args.out else None

    alias_map = load_alias_map(aliases_json)

    if args.rankings:
        rankings_csv = Path(args.rankings)
        if not rankings_csv.exists():
            raise FileNotFoundError(f"Rankings CSV not found: {rankings_csv}")
    else:
        rankings_csv = latest_rankings_csv(WEEKLY_DIR, season_year)
        if rankings_csv is None:
            raise FileNotFoundError(f"No rankings CSV found in {WEEKLY_DIR} for {season_year} (pattern wiki_rankings_{season_year}_*.csv).")

    print(f"📘 Aliases:   {aliases_json}")
    print(f"📊 Rankings:  {rankings_csv}")
    print(f"📄 Snapshots: {snapshots_csv}")
    if out_csv:
        print(f"➡️  Output:    {out_csv} (no in-place overwrite)")
    else:
        print("✍️  Output:    in-place (backup created)")

    rank_lookup = build_rank_lookup(rankings_csv, alias_map)
    print(f"🔎 Rank entries loaded: {len(rank_lookup)}")

    df = apply_ranks_to_snapshots(snapshots_csv, out_csv, alias_map, rank_lookup)
    # Quick summary
    updated_rows = int(df["homeTeamRank"].notna().sum() + df["awayTeamRank"].notna().sum())
    print(f"✅ Done. Non-null rank cells set (home+away combined): {updated_rows}")


if __name__ == "__main__":
    sys.exit(main())
