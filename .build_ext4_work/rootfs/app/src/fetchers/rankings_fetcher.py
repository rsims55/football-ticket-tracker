# src/fetchers/rankings_fetcher.py
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple, Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFD_API_KEY")

# Ensure weekly output dir exists
WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)


class RankingsFetcher:
    """
    Preference order:
      1) Wikipedia (current year): latest populated weekly Team/School column (AP preferred)
      2) Wikipedia (prior year): last populated weekly Team/School column
      3) CFBD API (current year, latest week) – Playoff -> AP -> Coaches
      4) CFBD API (prior year, latest week)

    Returns DataFrame with ['rank', 'school'] and writes a CSV in data/weekly/.
    """

    def __init__(self, year: Optional[int] = None):
        self.year = year or datetime.now().year
        self.api_key = API_KEY
        self.file_path: Optional[str] = None

    # -----------------------------
    # Public entry
    # -----------------------------
    def fetch_and_load(self) -> Optional[pd.DataFrame]:
        # 1) Wikipedia current year
        df = self._fetch_wiki_latest(self.year)
        if df is not None:
            return df

        # 2) Wikipedia prior year
        df = self._fetch_wiki_latest(self.year - 1)
        if df is not None:
            return df

        # 3) CFBD API current year
        wk = self.get_latest_week(self.year)
        if wk:
            df = self._fetch_api_rankings(self.year, wk)
            if df is not None:
                return df

        # 4) CFBD API prior year
        wk_prev = self.get_latest_week(self.year - 1)
        if wk_prev:
            df = self._fetch_api_rankings(self.year - 1, wk_prev)
            if df is not None:
                return df

        return None

    # -----------------------------
    # Wikipedia
    # -----------------------------
    def _fetch_wiki_latest(self, year: int) -> Optional[pd.DataFrame]:
        """
        Load https://en.wikipedia.org/wiki/{year}_NCAA_Division_I_FBS_football_rankings,
        find AP-like (preferred) or Coaches weekly table, pick rightmost populated
        Team/School column, return ['rank','school'] or None.
        """
        url = f"https://en.wikipedia.org/wiki/{year}_NCAA_Division_I_FBS_football_rankings"
        try:
            tables = pd.read_html(url)  # type: ignore[arg-type]
        except Exception:
            return None

        candidates: List[pd.DataFrame] = []
        for t in tables:
            df = t.copy()

            # Try to find a viable (rank, team) pair; store attrs so we can sort by width/AP-likeness later
            pair = self._find_rank_and_latest_team_col(df)
            if pair is None:
                continue
            rank_col, team_col, team_col_pos = pair
            df.attrs["__rank_col__"] = rank_col
            df.attrs["__team_col__"] = team_col
            df.attrs["__team_col_pos__"] = team_col_pos
            candidates.append(df)

        if not candidates:
            return None

        # Prefer wider tables – AP is usually wider
        candidates.sort(key=lambda d: len(d.columns), reverse=True)

        # Prefer AP-like tables first (heuristic)
        ap_like = [d for d in candidates if self._looks_like_ap(d)]
        ap_ids = {id(x) for x in ap_like}
        others = [d for d in candidates if id(d) not in ap_ids]
        ordered = ap_like + others

        for df in ordered:
            rank_col = df.attrs["__rank_col__"]
            team_col = df.attrs["__team_col__"]
            out = df[[rank_col, team_col]].copy()
            out.columns = ["rank", "school"]
            out = self._clean_rankings_df(out)
            if out is not None and not out.empty:
                fname = os.path.join(WEEKLY_DIR, f"wiki_rankings_{year}.csv")
                try:
                    out.to_csv(fname, index=False)
                    self.file_path = fname
                except Exception:
                    pass
                return out

        return None

    def _looks_like_ap(self, df: pd.DataFrame) -> bool:
        # Heuristic: AP weekly tables often have many "Week" columns and "Preseason"
        cols = [self._col_to_str(c).lower() for c in df.columns]
        has_pre = any("preseason" in c for c in cols)
        weekish = sum(1 for c in cols if "week" in c) >= 3
        return has_pre and weekish

    # -------- core detection for wiki tables --------
    def _find_rank_and_latest_team_col(self, df: pd.DataFrame) -> Optional[Tuple[Any, Any, int]]:
        """
        Return (rank_col, team_col, team_col_pos) if we can identify a proper pair.
        Strategy:
          * Flatten columns to simple strings for scanning, but keep original labels.
          * Identify all columns whose header includes 'Team' or 'School' (case-insensitive),
            and which contain many team-like strings (>=15 rows).
          * Choose the rightmost such column (latest week).
          * Pair it with the nearest rank-ish column (header contains 'Rank' or '#'),
            preferably to its immediate left; otherwise first column.
        """
        # Build flat names and mapping to original labels
        flat_cols = [self._col_to_str(c) for c in df.columns]
        col_map_flat_to_orig = {fc: orig for fc, orig in zip(flat_cols, df.columns)}

        # Potential team columns (by header)
        header_team_cands = [
            i for i, name in enumerate(flat_cols)
            if re.search(r"\b(team|school)\b", name, re.I)
        ]

        # If no obvious "team/school" in header, fall back to content-based scan
        content_team_cands = []
        if not header_team_cands:
            for i, c in enumerate(df.columns):
                s = df[c].astype(str).str.strip()
                team_count = s.apply(self._is_teamlike).sum()
                if team_count >= 15:
                    content_team_cands.append(i)

        team_cands = header_team_cands or content_team_cands
        if not team_cands:
            return None

        # Rightmost viable team column that isn't clearly metadata (points/prev/record/votes/etc.)
        bad_hdr = re.compile(r"(points|prev|previous|record|votes|trend|movement|conf|conference|note|rv)", re.I)
        best_idx: Optional[int] = None
        for i in reversed(team_cands):
            name = flat_cols[i]
            if bad_hdr.search(name):
                continue
            # Content check: avoid numeric columns posing as team
            s = df[df.columns[i]].astype(str).str.strip()
            team_count = s.apply(self._is_teamlike).sum()
            if team_count >= 15:
                best_idx = i
                break

        if best_idx is None:
            return None

        team_col_orig = df.columns[best_idx]

        # Find a rank-ish column to pair with (prefer a rank column to the left; else first column)
        rank_idx = None
        rank_hdr = re.compile(r"(rank|^#\s*$)", re.I)
        for j in range(best_idx - 1, -1, -1):
            if rank_hdr.search(flat_cols[j]):
                rank_idx = j
                break
        if rank_idx is None:
            # fall back to first column if it looks like rankish numbers 1..25
            if len(df.columns) > 0:
                rank_idx = 0

        rank_col_orig = df.columns[rank_idx] if rank_idx is not None else None
        if rank_col_orig is None:
            return None

        return (rank_col_orig, team_col_orig, best_idx)

    @staticmethod
    def _col_to_str(c: Any) -> str:
        """Turn a possibly-MultiIndex column label into a readable string."""
        if isinstance(c, tuple):
            parts = [str(x) for x in c if str(x) != "nan"]
            return " | ".join(parts).strip()
        return str(c)

    @staticmethod
    def _is_teamlike(s: str) -> bool:
        if not s:
            return False
        sl = str(s).strip().lower()
        if sl in {"nan", "—", "-", "nr"}:
            return False
        # Reject pure numbers
        if sl.isdigit():
            return False
        # Reject obvious records/points patterns
        if re.match(r"^\(?\d{1,2}-\d{1,2}\)?$", sl):
            return False
        if re.match(r"^\d{1,4}(\.\d+)?$", sl):  # points, etc.
            return False
        # Looks like text (team name)
        return True

    # -----------------------------
    # CFBD API (fallback)
    # -----------------------------
    def get_latest_week(self, year: int) -> Optional[int]:
        url = f"https://api.collegefootballdata.com/games?year={year}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                games = resp.json()
                weeks = {g.get("week") for g in games if g.get("week") is not None}
                return max(weeks) if weeks else None
        except Exception:
            return None
        return None

    def _fetch_api_rankings(self, year: int, week: int) -> Optional[pd.DataFrame]:
        url = f"https://api.collegefootballdata.com/rankings?year={year}&seasonType=regular&week={week}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.get(url, headers=headers, timeout=20)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        polls = data[0].get("polls", [])
        preferred = ["Playoff Committee Rankings", "AP Top 25", "Coaches Poll"]
        normalized = {str(p.get("poll", "")).strip().lower(): p for p in polls}

        for name in preferred:
            key = name.strip().lower()
            if key in normalized:
                ranks = normalized[key].get("ranks", [])
                df = pd.DataFrame(ranks)
                df = self._clean_rankings_df(df)
                if df is not None and not df.empty:
                    fname = os.path.join(WEEKLY_DIR, f"rankings_week{week}_{year}.csv")
                    try:
                        df.to_csv(fname, index=False)
                        self.file_path = fname
                    except Exception:
                        pass
                    return df
        return None

    # -----------------------------
    # Utilities
    # -----------------------------
    def _clean_rankings_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None

        # Normalize column names
        df = df.rename(columns={
            "RK": "rank", "Rank": "rank", "#": "rank",
            "TEAM": "school", "School": "school", "Team": "school", "team": "school",
        })

        # If we still don't have 'rank'/'school', try basic fallback
        if "rank" not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: "rank"})
        if "school" not in df.columns:
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[1]: "school"})

        df = df[["rank", "school"]].copy()

        # Coerce rank to int
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"])
        df["rank"] = df["rank"].astype(int)

        # Clean school text
        def clean_school(s: str) -> str:
            s = re.sub(r"\[.*?\]", "", str(s))            # footnotes
            s = re.sub(r"\(.*?\)", "", s)                 # parentheses
            s = s.replace("—", " ").replace("–", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        df["school"] = df["school"].astype(str).map(clean_school)
        df = df[df["school"].str.len() > 0]

        # Keep top 25 typical length (works even if table had more rows)
        df = df.sort_values("rank").head(25).reset_index(drop=True)
        return df
