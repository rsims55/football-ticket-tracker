# src/fetchers/rankings_fetcher.py
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

load_dotenv()

WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
DEBUG = os.getenv("RANKINGS_DEBUG", "0") == "1"


class RankingsFetchers:
    """
    Wikipedia-only strict priority:
      1) Current CFP (#CFP_rankings / variants)
      2) Current AP  (#AP_Poll / variants)
      3) Prior   CFP
      4) Prior   AP

    Week selection (per request):
      - For each (year, poll), try numbered weeks descending: 16 → 1
      - If (year == current year) AND poll == AP, also try Preseason (treated as week 0)
        if none of the numbered weeks are present.
      - Return the first parseable top-25 table.

    Exposes:
      - fetch_current_then_prior_cfp_ap() -> DataFrame | None
      - self.source -> 'wiki:CFP' | 'wiki:AP'
      - self.source_year -> int
    """

    SECTION_IDS: Dict[str, List[str]] = {
        "CFP": ["CFP_rankings", "College_Football_Playoff_rankings", "CFP_rankings_2"],
        "AP":  ["AP_Poll", "AP_poll", "AP_Top_25"],
    }

    POLL_NAME = {"CFP": "CFP", "AP": "AP"}

    def __init__(self, year: Optional[int] = None):
        self.year = year or datetime.now().year
        self.source: Optional[str] = None
        self.source_year: Optional[int] = None

    # -------- Public API --------
    def fetch_current_then_prior_cfp_ap(self) -> Optional[pd.DataFrame]:
        """
        Try in order:
          (year, CFP) → (year, AP) → (year-1, CFP) → (year-1, AP)
        For each (year, poll), try Week 16 down to Week 1 (numbered weeks).
        If (year == current year) and poll == AP, also try Preseason (week 0) afterward.
        Returns the first parsed ranking table as a normalized DataFrame.
        """
        order = [
            (self.year, "CFP"),
            (self.year, "AP"),
            (self.year - 1, "CFP"),
            (self.year - 1, "AP"),
        ]
        if DEBUG:
            print("📈 Fetching latest rankings (Wikipedia only; CFP→AP, current→prior)…")
            print("🔎 Week preference: 16 → 1, then (current-year AP only) Preseason")

        for y, label in order:
            soup = self._get_soup(y)
            if soup is None:
                continue

            # 1) Try numbered weeks 16..1
            found_df: Optional[pd.DataFrame] = None
            found_week: Optional[int] = None
            for week in range(16, 0, -1):
                df = self._fetch_year_label_for_week(soup, y, label, target_week=week)
                if self._ok(df):
                    found_df, found_week = df.copy(), week
                    break

            # 2) For CURRENT YEAR + AP only: if none found, try Preseason (week 0)
            if found_df is None and (y == self.year) and (label == "AP"):
                df = self._fetch_year_label_for_week(soup, y, label, target_week=0)
                if self._ok(df):
                    found_df, found_week = df.copy(), 0

            if self._ok(found_df):
                # stamp source + season (use the year we actually fetched)
                self.source, self.source_year = f"wiki:{label}", y
                out_df = found_df.copy()
                out_df["season"] = y
                # ensure week column reflects the chosen week
                if "week" not in out_df.columns or out_df["week"].isna().all():
                    out_df["week"] = int(found_week if found_week is not None else -1)

                # Write artifact
                self._write_artifact(out_df, y)

                # Print confirmation exactly as requested
                week_str = "Preseason" if (found_week == 0) else f"Week {found_week}"
                print(f"✅ Selected {label} {y} {week_str}")

                if DEBUG:
                    print(f"✅ Selected {label} {y} {week_str} with {len(out_df)} rows")

                return out_df

            if DEBUG:
                print(f"↩️  No usable week found for {label} {y}")

        return None

    # -------- Internals --------
    def _fetch_year_label_for_week(
        self, soup: BeautifulSoup, year: int, label: str, target_week: int
    ) -> Optional[pd.DataFrame]:
        """
        Attempt to parse the specified (year, label, target_week).
        1) Section-targeted scan (walk forward until next <h2>) filtering to the target week
        2) Page-wide fallback constrained to the target week
        """
        # 1) Section-targeted
        for sid in self.SECTION_IDS.get(label, []):
            df = self._parse_section_scan_wikitables(
                soup, sid, prefer_label=label, target_week=target_week
            )
            if self._ok(df):
                if DEBUG:
                    print(f"[rankings] {year}/{label} week={target_week} via id='{sid}' → {len(df)} rows")
                df = df.copy()
                df["poll"] = self.POLL_NAME.get(label, label)
                df["week"] = int(target_week)
                return df

        # 2) Page-wide fallback
        df = self._page_wide_best_table(soup, prefer_label=label, target_week=target_week)
        if self._ok(df):
            if DEBUG:
                print(f"[rankings] {year}/{label} week={target_week} via page-wide fallback → {len(df)} rows")
            df = df.copy()
            df["poll"] = self.POLL_NAME.get(label, label)
            df["week"] = int(target_week)
            return df

        return None

    def _get_soup(self, year: int) -> Optional[BeautifulSoup]:
        url = f"https://en.wikipedia.org/wiki/{year}_NCAA_Division_I_FBS_football_rankings"
        try:
            # Optional local override for testing saved HTML (e.g., data/weekly/wiki_2025.html)
            local = os.getenv("RANKINGS_HTML_OVERRIDE", "").strip()
            if local and os.path.exists(local):
                if DEBUG:
                    print(f"[rankings] using local HTML override: {local}")
                with open(local, "r", encoding="utf-8") as f:
                    html = f.read()
            else:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
                r.raise_for_status()
                html = r.text

            # Only save HTML when explicitly requested
            SAVE_HTML = os.getenv("RANKINGS_SAVE_HTML", "0") == "1"
            if DEBUG and not local and SAVE_HTML:
                out = os.path.join(WEEKLY_DIR, f"wiki_{year}.html")
                try:
                    with open(out, "w", encoding="utf-8") as f:
                        f.write(html)
                    print(f"[rankings] saved HTML → {out}")
                except Exception as e:
                    print(f"[rankings] could not write HTML: {e}")

            return BeautifulSoup(html, "lxml")
        except Exception as e:
            if DEBUG:
                print(f"[rankings] fetch failed for {year}: {e}")
            return None

    # ---- Section scan: robust forward walk until next <h2> ----
    def _parse_section_scan_wikitables(
        self, soup: BeautifulSoup, section_id: str, prefer_label: str, target_week: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Robust scan: starting at the <h2>/<h3> for the given section_id, iterate forward
        through .next_elements until the next <h2>. Parse any <table> encountered
        (direct or nested) without requiring class='wikitable'.

        If target_week is provided:
          - For AP, only parse if the table has a column matching that week (or Preseason=0)
          - For CFP (or non-matrix AP), only accept if nearest heading resolves to that week
        """
        anchor = soup.find(id=section_id)
        if anchor is None:
            return None

        heading = anchor if anchor.name in ("h2", "h3") else anchor.find_parent(["h2", "h3"]) or anchor
        if heading is None:
            return None

        tried = 0
        if DEBUG:
            print(f"[rankings] DEBUG: forward-scan from <{heading.name} id='{section_id}'> (prefer='{prefer_label}', week={target_week})")

        for node in heading.next_elements:
            if isinstance(node, Tag):
                if node is not heading and node.name == "h2":
                    break
                if node.name == "table":
                    tried += 1
                    if prefer_label == "AP":
                        # Require AP table to support the requested week column
                        col_idx, _wk = self._ap_col_index_for_week(node, target_week)
                        if col_idx < 0:
                            continue
                        # Parse with forced AP column/week
                        week_num, poll_date = self._meta_for_table(node, prefer_label)
                        df = self._parse_rank_table(
                            node,
                            week_num=(target_week if target_week is not None else _wk),
                            poll_date=poll_date,
                            prefer_label=prefer_label,
                            ap_forced_col=col_idx,
                            ap_forced_week=(target_week if target_week is not None else _wk),
                        )
                        if self._ok(df):
                            return df
                    else:
                        # CFP or other: only accept if nearest heading says the same week
                        wk_from_head, poll_date = self._meta_for_table(node, prefer_label)
                        if target_week is not None and wk_from_head is not None and wk_from_head != target_week:
                            continue
                        df = self._parse_rank_table(
                            node,
                            week_num=(target_week if target_week is not None else wk_from_head),
                            poll_date=poll_date,
                            prefer_label=prefer_label,
                        )
                        if self._ok(df):
                            return df

        if DEBUG and tried:
            print(f"[rankings] section '{section_id}' scanned {tried} tables → none parsed for week {target_week}")
        return None

    # ---- Label-aware page-wide fallback (filtered to a target week) ----
    def _page_wide_best_table(
        self, soup: BeautifulSoup, prefer_label: str, target_week: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Walk all tables. For AP: only consider tables that have the exact target week column.
        For CFP/others: only accept if nearest heading week matches the target week (when provided).
        """
        for tbl in soup.find_all("table"):
            # quick probe to see if it's plausibly a rankings table
            probe, n = self._try_table_quick(tbl)
            if probe is None:
                continue

            head_text = self._nearest_heading_text(tbl).lower()
            is_cfp = ("cfp" in head_text) or ("playoff" in head_text)
            ap_col_idx, _apw = self._ap_week1_or_pre_col_index(tbl)
            is_ap = (" ap" in f" {head_text} ") or (ap_col_idx >= 1)

            if prefer_label == "CFP" and not is_cfp:
                continue
            if prefer_label == "AP" and not is_ap:
                continue

            if prefer_label == "AP":
                # Require exact target week column (including Preseason when target_week == 0)
                col_idx, wk = self._ap_col_index_for_week(tbl, target_week)
                if col_idx < 0:
                    continue
                week_num = target_week if target_week is not None else wk
                _, poll_date = self._meta_for_table(tbl, prefer_label)
                return self._parse_rank_table(
                    tbl,
                    week_num=week_num,
                    poll_date=poll_date,
                    prefer_label=prefer_label,
                    ap_forced_col=col_idx,
                    ap_forced_week=week_num,
                )
            else:
                wk_from_head, poll_date = self._meta_for_table(tbl, prefer_label)
                if target_week is not None and wk_from_head is not None and wk_from_head != target_week:
                    continue
                return self._parse_rank_table(
                    tbl,
                    week_num=(target_week if target_week is not None else wk_from_head),
                    poll_date=poll_date,
                    prefer_label=prefer_label,
                )

        return None

    # ---- Heading helpers ----
    def _nearest_heading_text(self, node: Tag) -> str:
        cur = node
        while cur and cur.previous_sibling:
            cur = cur.previous_sibling
            if isinstance(cur, Tag) and cur.name in ("h2", "h3"):
                return cur.get_text(" ", strip=True)
        body = node.find_parent("body") if node else None
        h1 = body.find("h1") if body else None
        return h1.get_text(" ", strip=True) if h1 else ""

    def _meta_for_table(self, tbl: Tag, prefer_label: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Derive (week_num, poll_date) from the nearest heading, when possible.
        Rules:
        - If header has 'Preseason' → week = 0
        - If header has 'Week N (Final)' → week = max(N-1, 0)
        - Else if header has 'Week N' → week = N
        - poll_date parsed if present in parentheses like '(September 2, 2025)'
        """
        txt = self._nearest_heading_text(tbl)
        week_num: Optional[int] = None
        poll_date: Optional[str] = None

        if txt:
            tl = txt.lower()

            # Preseason
            if re.search(r"\bpreseason\b", tl, re.IGNORECASE):
                week_num = 0
            else:
                # Look for "Week N"
                m_week = re.search(r"\bweek\s*(\d+)\b", tl, re.IGNORECASE)
                if m_week:
                    n = int(m_week.group(1))

                    # If "(Final)" appears anywhere in the heading, walk back one week (floor at 0)
                    is_final = re.search(r"\(.*final.*\)", tl, re.IGNORECASE) is not None
                    week_num = max(n - 1, 0) if is_final else n

            # Optional poll date like "(September 2, 2025)"
            m_date = re.search(
                r"\((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\)",
                txt
            )
            if m_date:
                poll_date = m_date.group(0).strip("()")

        return week_num, poll_date

    # ---- AP matrix helpers (supports arbitrary week selection) ----
    def _ap_find_header_row(self, table: Tag) -> Optional[Tag]:
        thead = table.find("thead")
        if thead:
            hr = thead.find("tr")
            if hr:
                return hr
        tbody = table.find("tbody") or table
        for r in tbody.find_all("tr"):
            ths = r.find_all("th")
            if len(ths) >= 3:  # rank + multiple week headers
                return r
        return None

    def _ap_week1_or_pre_col_index(self, table: Tag) -> Tuple[int, Optional[int]]:
        """
        Legacy helper: Return (column_index, week_num) for AP matrix:
          - Prefer 'Week 1' → (idx, 1)
          - Else 'Preseason' → (idx, 0)
          - Else (-1, None)
        """
        hr = self._ap_find_header_row(table)
        if not hr:
            return -1, None
        headers = hr.find_all(["th", "td"])
        pre_idx = -1
        wk1_idx = -1
        for idx, h in enumerate(headers):
            t = h.get_text(" ", strip=True).lower()
            if "preseason" in t:
                pre_idx = idx
            if re.search(r"\bweek\s*1\b", t):
                wk1_idx = idx
        if wk1_idx >= 0:
            return wk1_idx, 1
        if pre_idx >= 0:
            return pre_idx, 0
        return -1, None

    def _ap_col_index_for_week(self, table: Tag, target_week: Optional[int]) -> Tuple[int, Optional[int]]:
        """
        Find the AP matrix column for the requested week.
        - target_week == 0 maps to 'Preseason'
        - target_week >= 1 maps to 'Week N'
        Returns (col_index, resolved_week) or (-1, None) if not present.
        """
        if target_week is None:
            return -1, None
        hr = self._ap_find_header_row(table)
        if not hr:
            return -1, None
        headers = hr.find_all(["th", "td"])
        for idx, h in enumerate(headers):
            t = h.get_text(" ", strip=True).lower()
            if target_week == 0 and "preseason" in t:
                return idx, 0
            if target_week >= 1 and re.search(rf"\bweek\s*{target_week}\b", t):
                return idx, target_week
        return -1, None

    # ---- Table parsers ----
    def _parse_rank_table(
        self,
        table: Tag,
        week_num: Optional[int],
        poll_date: Optional[str],
        prefer_label: str,
        ap_forced_col: Optional[int] = None,
        ap_forced_week: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Parse rows where the first cell is a numeric rank (accept '1' or '1.')
        and the chosen column contains the team (may include '(##)' FP votes).

        If prefer_label == 'AP' and ap_forced_col is provided, use that column.
        Else for AP, fall back to Week1/Preseason heuristic (legacy).
        """
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")
        out: List[dict] = []

        # Decide which AP column to use (matrix)
        ap_col, ap_week = (None, None)
        if prefer_label == "AP":
            if ap_forced_col is not None and ap_forced_col >= 0:
                ap_col, ap_week = ap_forced_col, ap_forced_week
            else:
                cached = getattr(table, "_ap_wk1orpre_colweek", None)
                if cached is None:
                    ap_col, ap_week = self._ap_week1_or_pre_col_index(table)
                    setattr(table, "_ap_wk1orpre_colweek", (ap_col, ap_week))
                else:
                    ap_col, ap_week = cached

        # If heading didn't provide week info, use the detected AP week
        if prefer_label == "AP" and week_num is None and ap_week is not None:
            week_num = ap_week

        for tr in rows:
            cells = tr.find_all(["th", "td"])  # allow nested
            if not cells:
                continue

            rtxt_raw = cells[0].get_text(" ", strip=True)
            m = re.search(r"(\d+)", rtxt_raw)  # accept '1.' too
            if not m:
                continue
            rtxt = int(m.group(1))

            # pick the team cell
            team_cell = None
            if ap_col is not None and ap_col >= 1 and ap_col < len(cells):
                team_cell = cells[ap_col]
            else:
                for c in cells[1:]:
                    t = c.get_text(" ", strip=True)
                    if t:
                        team_cell = c
                        break
            if team_cell is None:
                continue

            raw_team = team_cell.get_text(" ", strip=True)
            fpv = self._extract_first_place_votes(raw_team)
            team = self._clean_team(raw_team)
            if not team:
                continue

            out.append({"rank": rtxt, "school": team, "first_place_votes": fpv})
            if len(out) >= 25:
                break

        if not out:
            return None

        df = pd.DataFrame(out)
        df = self._clean_df(df)
        if df is None:
            return None

        # metadata (poll stamped later; season stamped in fetcher)
        if week_num is not None:
            df["week"] = int(week_num)
        if poll_date:
            try:
                df["poll_date"] = pd.to_datetime(poll_date).dt.date.astype(str)
            except Exception:
                df["poll_date"] = poll_date
        return df

    def _extract_first_place_votes(self, s: str) -> int:
        m = re.search(r"\((\d{1,3})\)\s*$", s)
        # Ignore state abbreviations like (FL), (UT), etc.
        if m and not re.search(r"\([A-Z]{2}\)", s):
            try:
                return int(m.group(1))
            except Exception:
                return 0
        return 0

    def _try_table_quick(self, table: Tag) -> Tuple[Optional[pd.DataFrame], int]:
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")
        out: List[dict] = []

        ap_col, _ = self._ap_week1_or_pre_col_index(table)

        for tr in rows:
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue

            rtxt_raw = cells[0].get_text(" ", strip=True)
            m = re.search(r"(\d+)", rtxt_raw)
            if not m:
                continue

            team_cell = None
            if ap_col >= 1 and ap_col < len(cells):
                team_cell = cells[ap_col]
            else:
                for c in cells[1:]:
                    t = c.get_text(" ", strip=True)
                    if t:
                        team_cell = c
                        break
            if team_cell is None:
                continue

            team = self._clean_team(team_cell.get_text(" ", strip=True))
            if not team:
                continue

            out.append({"rank": int(m.group(1)), "school": team})
            if len(out) >= 25:
                break

        if len(out) >= 15:
            return self._clean_df(pd.DataFrame(out)), len(out)
        return None, 0

    # ---- Cleaning ----
    def _clean_team(self, s: str) -> str:
        s = re.sub(r"\[.*?\]", "", s)  # remove footnotes like [1]
        # remove (25), (10-2), and also (14 2) formatting
        s = re.sub(r"\(\s*\d+(?:[-\s]\d+)?\s*\)", "", s)
        s = s.replace("—", " ").replace("–", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _clean_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df = df.rename(
            columns={
                "RK": "rank",
                "Rank": "rank",
                "#": "rank",
                "TEAM": "school",
                "Team": "school",
                "School": "school",
                "team": "school",
            }
        )
        if "rank" not in df.columns:
            df = df.rename(columns={df.columns[0]: "rank"})
        if "school" not in df.columns and len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "school"})
        if "first_place_votes" not in df.columns:
            df["first_place_votes"] = 0

        df = df[["rank", "school", "first_place_votes"]].copy()
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"])
        df["rank"] = df["rank"].astype(int)
        df["school"] = df["school"].astype(str)
        df = df[df["school"].str.len() > 0]
        return df.sort_values("rank").head(25).reset_index(drop=True)

    def _ok(self, df: Optional[pd.DataFrame]) -> bool:
        return df is not None and not df.empty and {"rank", "school"}.issubset(df.columns)

    def _write_artifact(self, df: pd.DataFrame, year: int) -> None:
        poll = (self.source or "").split(":")[-1] if self.source else "AP"
        out = os.path.join(WEEKLY_DIR, f"wiki_rankings_{year}_{poll}.csv")
        try:
            df = df.copy()
            df["poll"] = poll  # stamp final poll name
            # Ensure columns; 'week' is numeric (0 for Preseason if present)
            cols = ["season", "poll", "week", "poll_date", "rank", "school", "first_place_votes"]
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            df.to_csv(out, index=False, encoding="utf-8")
            if DEBUG:
                print(f"[rankings] wrote artifact → {out} ({len(df)} rows)")
        except Exception:
            if DEBUG:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Fetch latest CFP/AP rankings from Wikipedia.")
    parser.add_argument("--year", type=int, default=None, help="Season year (defaults to current year)")
    args = parser.parse_args()

    try:
        fetcher = RankingsFetchers(year=args.year)
        df = fetcher.fetch_current_then_prior_cfp_ap()
        if df is None or df.empty:
            print("❌ No rankings found (no table parsed).")
            sys.exit(1)

        print("✅ Parsed rankings:")
        try:
            print(df.head(10).to_string(index=False))
        except Exception:
            print(df.head(10))

        # Print where it wrote (avoid Series truthiness bug)
        year_used = fetcher.source_year or (args.year or datetime.now().year)
        poll_val = "AP"
        if "poll" in df.columns and len(df) > 0:
            try:
                poll_val = str(df["poll"].iloc[0]) if pd.notna(df["poll"].iloc[0]) else "AP"
            except Exception:
                pass
        outpath = os.path.join(WEEKLY_DIR, f"wiki_rankings_{year_used}_{poll_val}.csv")
        if os.path.exists(outpath):
            print(f"💾 Wrote CSV → {outpath}")
        else:
            print("⚠️ CSV not detected after parse; check write permissions to data/weekly/")
    except Exception as e:
        print(f"💣 Runner crashed: {e}")
        sys.exit(2)
