# src/fetchers/rankings_fetcher.py
from __future__ import annotations

import os, re
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

load_dotenv()

WEEKLY_DIR = os.path.join("data", "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
DEBUG = os.getenv("RANKINGS_DEBUG", "0") == "1"


class RankingsFetcher:
    """
    Wikipedia-only strict priority:
      1) Current CFP (#CFP_rankings / variants)
      2) Current AP  (#AP_Poll / variants)
      3) Prior   CFP
      4) Prior   AP

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

        for y, label in order:
            df = self._fetch_year_label(y, label)
            if self._ok(df):
                # stamp source + season (use the year we actually fetched)
                self.source, self.source_year = f"wiki:{label}", y
                df = df.copy()
                df["season"] = y
                self._write_artifact(df, y)
                return df
        return None

    # -------- Internals --------
    def _fetch_year_label(self, year: int, label: str) -> Optional[pd.DataFrame]:
        soup = self._get_soup(year)
        if soup is None:
            return None

        # 1) Section-targeted: scan ALL wikitables inside the section until the NEXT <h2>
        for sid in self.SECTION_IDS.get(label, []):
            df = self._parse_section_scan_wikitables(soup, sid, prefer_label=label)
            if self._ok(df):
                if DEBUG:
                    print(f"[rankings] {year}/{label} via id='{sid}' → {len(df)} rows")
                df = df.copy()
                df["poll"] = self.POLL_NAME.get(label, label)
                return df

        # 2) Label-aware page-wide fallback
        df = self._page_wide_best_table(soup, prefer_label=label)
        if self._ok(df):
            if DEBUG:
                print(f"[rankings] {year}/{label} via page-wide fallback → {len(df)} rows")
            df = df.copy()
            df["poll"] = self.POLL_NAME.get(label, label)
            return df

        if DEBUG:
            print(f"[rankings] {year}/{label} → not found")
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

            if DEBUG and not local:
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

    # ---- Section scan: walk siblings until next <h2>; DO NOT stop at <h3> ----
    def _parse_section_scan_wikitables(
        self, soup: BeautifulSoup, section_id: str, prefer_label: str
    ) -> Optional[pd.DataFrame]:
        anchor = soup.find(id=section_id)
        if anchor is None:
            return None

        heading = anchor if anchor.name in ("h2", "h3") else anchor.find_parent(["h2", "h3"]) or anchor

        node = heading.next_sibling
        tried = 0
        while node is not None:
            if isinstance(node, Tag) and node.name == "h2":
                break
            if isinstance(node, Tag):
                # direct table
                if node.name == "table" and "wikitable" in (node.get("class") or []):
                    tried += 1
                    week_num, poll_date = self._meta_for_table(node, prefer_label)
                    df = self._parse_rank_table(node, week_num, poll_date, prefer_label)
                    if self._ok(df):
                        return df
                # nested tables inside wrappers
                for tbl in node.find_all("table", class_="wikitable"):
                    tried += 1
                    week_num, poll_date = self._meta_for_table(tbl, prefer_label)
                    df = self._parse_rank_table(tbl, week_num, poll_date, prefer_label)
                    if self._ok(df):
                        return df
            node = node.next_sibling

        if DEBUG and tried:
            print(f"[rankings] section '{section_id}' scanned {tried} wikitables → none parsed")
        return None

    # ---- Label-aware page-wide fallback (prevents AP being mis-tagged as CFP) ----
    def _page_wide_best_table(self, soup: BeautifulSoup, prefer_label: str) -> Optional[pd.DataFrame]:
        for tbl in soup.find_all("table", class_="wikitable"):
            probe, n = self._try_table_quick(tbl)
            if probe is None:
                continue

            head_text = self._nearest_heading_text(tbl).lower()
            is_cfp = ("cfp" in head_text) or ("playoff" in head_text)
            ap_col_idx = self._ap_preseason_col_index(tbl)
            is_ap = (" ap" in f" {head_text} ") or (ap_col_idx >= 1)

            if prefer_label == "CFP" and not is_cfp:
                continue
            if prefer_label == "AP" and not is_ap:
                continue

            week_num, poll_date = self._meta_for_table(tbl, prefer_label)
            return self._parse_rank_table(tbl, week_num, poll_date, prefer_label)

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
        - Preseason → week 0
        - 'Week N'  → week N
        """
        txt = self._nearest_heading_text(tbl)
        week_num: Optional[int] = None
        poll_date: Optional[str] = None

        if txt:
            if re.search(r"preseason", txt, re.IGNORECASE):
                week_num = 0
            else:
                m_week = re.search(r"week\s*(\d+)", txt, re.IGNORECASE)
                if m_week:
                    week_num = int(m_week.group(1))

            m_date = re.search(
                r"\((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\)",
                txt
            )
            if m_date:
                poll_date = m_date.group(0).strip("()")

        return week_num, poll_date

    # ---- AP matrix helper (choose 'Preseason' column explicitly) ----
    def _ap_preseason_col_index(self, table: Tag) -> int:
        """
        For AP matrix tables (header has 'Preseason', 'Week 1', ...),
        return the absolute column index for the 'Preseason' column.
        If not found, return -1.
        """
        thead = table.find("thead")
        header_row = None
        if thead:
            header_row = thead.find("tr")
        if not header_row:
            tbody = table.find("tbody") or table
            rows = tbody.find_all("tr")
            if rows:
                for r in rows:
                    ths = r.find_all("th")
                    if len(ths) >= 3:  # rank + many week headers
                        header_row = r
                        break
        if not header_row:
            return -1

        headers = header_row.find_all(["th", "td"])
        for idx, h in enumerate(headers):
            t = h.get_text(" ", strip=True).lower()
            if "preseason" in t:
                return idx
        return -1

    # ---- Table parsers ----
    def _parse_rank_table(
        self, table: Tag, week_num: Optional[int], poll_date: Optional[str], prefer_label: str
    ) -> Optional[pd.DataFrame]:
        """
        Parse rows where the first cell is a numeric rank (accept '1' or '1.')
        and the chosen column contains the team (may include '(##)' FP votes).
        """
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")
        out: List[dict] = []

        # Detect AP matrix header once per table
        ap_col = None
        if prefer_label == "AP":
            ap_col = getattr(table, "_ap_pre_col", None)
            if ap_col is None:
                ap_col = self._ap_preseason_col_index(table)
                setattr(table, "_ap_pre_col", ap_col)

        # If heading didn't carry 'Preseason', but this is an AP matrix and we're using the
        # 'Preseason' column, force week = 0 as requested.
        if prefer_label == "AP" and week_num is None and ap_col is not None and ap_col >= 1:
            week_num = 0

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
        m = re.search(r"\((\d{1,3})\)", s)
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

        ap_col = self._ap_preseason_col_index(table)

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
        s = re.sub(r"\[.*?\]", "", s)                  # remove footnotes like [1]
        s = re.sub(r"\(\s*\d+(?:-\d+)?\s*\)", "", s)   # remove (25) or (10-2) — keep (FL) etc.
        s = s.replace("—", " ").replace("–", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _clean_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "RK": "rank", "Rank": "rank", "#": "rank",
            "TEAM": "school", "Team": "school", "School": "school", "team": "school",
        })
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
                import traceback; traceback.print_exc()
