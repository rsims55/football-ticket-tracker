#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fetchers.rankings_api_fetcher import RankingsApiFetcher  # noqa: E402


def load_alias_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def build_alias_maps(alias_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw = load_alias_map(alias_path)
    # normalize keys
    alias = {str(k).strip().lower(): str(v).strip() for k, v in raw.items()}
    rev = {v.lower(): k for k, v in alias.items()}
    return alias, rev


def canon_team(name: str | None, alias: dict[str, str], rev: dict[str, str]) -> str:
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    if key in alias:
        return key
    if key in rev:
        return rev[key]
    return key


def load_rankings_history(path: Path, year: int, fetch: bool) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if not fetch:
        raise FileNotFoundError(f"Rankings history not found: {path}")
    fetcher = RankingsApiFetcher(year=year, season_type="both")
    df = fetcher.fetch()
    if df is None or df.empty:
        raise RuntimeError("No rankings data returned from API.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def select_poll_per_week(rankings: pd.DataFrame, preference: list[str]) -> pd.DataFrame:
    pref_map = {p: i for i, p in enumerate(preference)}
    df = rankings.copy()
    df["poll"] = df["poll"].fillna("Unknown")
    df["poll_rank"] = df["poll"].map(pref_map).fillna(len(preference) + 1)
    # best poll per week
    best = df.groupby("week")["poll_rank"].min().reset_index().rename(columns={"poll_rank": "best_poll"})
    df = df.merge(best, on="week", how="left")
    df = df[df["poll_rank"] == df["best_poll"]]
    return df


def backfill_rankings_by_week(
    snapshots_csv: Path,
    rankings_csv: Path,
    alias_path: Path,
    preference: list[str],
) -> pd.DataFrame:
    alias, rev = build_alias_maps(alias_path)
    df = pd.read_csv(snapshots_csv, low_memory=False)

    if "week" not in df.columns:
        raise KeyError("Snapshots file missing 'week' column. Add it before running.")
    df["rank_week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    # rankings history
    ranks = load_rankings_history(rankings_csv, year=int(df["week"].max()) if df["week"].notna().any() else 0, fetch=False)
    ranks = select_poll_per_week(ranks, preference)
    ranks["team_key"] = ranks["school"].map(lambda s: canon_team(s, alias, rev))
    ranks = ranks[["week", "team_key", "rank"]].dropna(subset=["team_key", "rank"])
    ranks["rank_week"] = pd.to_numeric(ranks["week"], errors="coerce").fillna(0).astype(int)

    # map teams
    df["home_key"] = df.get("homeTeam").map(lambda s: canon_team(s, alias, rev))
    df["away_key"] = df.get("awayTeam").map(lambda s: canon_team(s, alias, rev))

    # merge for home/away
    home_ranks = ranks.rename(columns={"team_key": "home_key", "rank": "homeTeamRank_new"}).drop(columns=["week"])
    away_ranks = ranks.rename(columns={"team_key": "away_key", "rank": "awayTeamRank_new"}).drop(columns=["week"])

    df = df.merge(
        home_ranks,
        left_on=["rank_week", "home_key"],
        right_on=["rank_week", "home_key"],
        how="left",
    )
    df = df.merge(
        away_ranks,
        left_on=["rank_week", "away_key"],
        right_on=["rank_week", "away_key"],
        how="left",
    )

    df["homeTeamRank"] = df["homeTeamRank_new"]
    df["awayTeamRank"] = df["awayTeamRank_new"]
    df["isRankedMatchup"] = df["homeTeamRank"].notna() & df["awayTeamRank"].notna()

    df.drop(
        columns=[
            "rank_week",
            "home_key",
            "away_key",
            "homeTeamRank_new",
            "awayTeamRank_new",
        ],
        errors="ignore",
        inplace=True,
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill snapshots with rankings aligned to game week."
    )
    p.add_argument("--snapshots", required=True, help="Path to price_snapshots CSV")
    p.add_argument("--rankings", default="", help="Path to rankings_history_<year>.csv")
    p.add_argument("--aliases", default="data/permanent/team_aliases.json", help="Path to team_aliases.json")
    p.add_argument("--preference", default="CFP,AP,Coaches", help="Comma-separated poll preference")
    p.add_argument("--fetch", action="store_true", help="Fetch rankings history if missing")
    p.add_argument("--out", default="", help="Optional output path (default: overwrite snapshots)")
    args = p.parse_args()

    snapshots = Path(args.snapshots)
    rankings = Path(args.rankings) if args.rankings else Path("data/weekly") / "rankings_history_2025.csv"
    aliases = Path(args.aliases)

    df = backfill_rankings_by_week(
        snapshots,
        rankings,
        aliases,
        [p.strip() for p in args.preference.split(",") if p.strip()],
    )

    out = Path(args.out) if args.out else snapshots
    df.to_csv(out, index=False)
    print(f"✅ Rankings backfilled using game week → {out}")


if __name__ == "__main__":
    main()
