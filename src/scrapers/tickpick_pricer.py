# src/scrapers/tickpick_pricer.py
__all__ = ["TickPickPricer"]

import os
import re
import json
import csv
import time
import random
from typing import List, Dict, Any, Optional, Iterable, Tuple
from datetime import datetime

import requests
from bs4 import BeautifulSoup

try:
    import cloudscraper  # optional; used automatically if present
except Exception:
    cloudscraper = None


# ---------- Regex helpers ----------
EVENT_ID_RE = re.compile(r"/(\d{6,})/?$")  # last numeric segment in the URL
# e.g. ...-8-30-25-7pm/<id>/   OR sometimes '3am' placeholders
DATE_FROM_URL_RE = re.compile(
    r"-([1-9]|1[0-2])-([1-9]|[12]\d|3[01])-(\d{2})-([1-9]|1[0-2])(am|pm)",
    re.IGNORECASE,
)

# FAQ snippets on team pages (average price + total tickets available)
AVG_PRICE_RE = re.compile(
    r"average ticket price is \$\s*([0-9][0-9,]*(?:\.\d{2})?)", re.IGNORECASE
)
TICKETS_AVAIL_RE = re.compile(
    r"There are\s*([0-9][0-9,]*)\s*tickets still available", re.IGNORECASE
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7",
    "Cache-Control": "no-cache",
}

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


class TickPickPricer:
    """
    Scrape event offer info from TickPick team pages by parsing JSON-LD blocks.
    - Input: list of team page URLs (ncaa-football/<team>-football-tickets/)
    - Output: de-duplicated events w/ event_id, date_local, time_local, low/high prices, deep link
    Also enriches the 'next' event on each page with average price and inventory counts parsed from team page FAQ.

    Added: per-event helpers get_summary_by_event_id() for daily snapshots.
    """

    def __init__(
        self,
        team_urls: Iterable[str],
        output_dir: str = "data",
        polite_delay_range: tuple = (12, 18),
        retries: int = 3,
        timeout: int = 25,
        use_cache: bool = False,
        cache_dir: str = "data/_cache_tickpick_html",
        verbose: bool = True,
    ):
        self.team_urls = list(team_urls)
        self.output_dir = output_dir
        self.polite_delay_range = polite_delay_range
        self.retries = retries
        self.timeout = timeout
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # session: try cloudscraper first (bypasses some Cloudflare checks), else requests
        if cloudscraper:
            self.session = cloudscraper.create_scraper()
        else:
            self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # -------------------- Public entry --------------------

    def run(self) -> Dict[str, str]:
        """
        Crawl all team pages, parse offers, de-duplicate, export CSV+JSON.
        Returns {"csv": <path>, "json": <path>}.
        """
        all_rows: List[Dict[str, Any]] = []

        for idx, url in enumerate(self.team_urls, 1):
            self.current_team_url = url
            try:
                html = self._get_html(url)
            except Exception as e:
                print(f"[{idx}/{len(self.team_urls)}] ❌ fetch failed: {url} :: {e}")
                self._polite_sleep()
                continue

            if self._looks_like_cloudflare(html):
                print(f"[{idx}/{len(self.team_urls)}] ⛔ Cloudflare block for {url}")
                self._polite_sleep()
                continue

            # Parse page
            blocks = self._extract_jsonld(html)
            team_name = self._infer_team_name(html, blocks, url)

            if self.verbose:
                print(f"[{idx}/{len(self.team_urls)}] {team_name}")

            rows = self._extract_offers(blocks)
            # Enrich the soonest event on this page with FAQ metrics
            metrics = self._extract_faq_metrics_from_html(html)
            if rows and (metrics.get("avg_price") is not None or metrics.get("tickets_available") is not None):
                def row_key(r):
                    d = r.get("date_local") or "9999-12-31"
                    t = r.get("time_local") or "23:59"
                    return (d, t)
                target = min([r for r in rows if r.get("date_local")], key=row_key, default=rows[0])
                target["avg_price_from_page"] = metrics.get("avg_price")
                target["tickets_available_from_page"] = metrics.get("tickets_available")

            if self.verbose:
                print(f"  → found {len(rows)} offer rows on this page")

            all_rows.extend(rows)
            self._polite_sleep()

        cleaned = self._dedupe(all_rows)
        paths = self._export(cleaned)
        if self.verbose:
            print(f"✅ Exported {len(cleaned)} unique events")
            print(f"CSV:  {paths['csv']}")
            print(f"JSON: {paths['json']}")
        return paths

    # -------------------- HTTP helpers --------------------

    def _polite_sleep(self) -> None:
        low, high = self.polite_delay_range
        delay = random.randint(low, high)
        if self.verbose:
            print(f"  … sleeping {delay}s")
        time.sleep(delay)

    def _cache_path(self, url: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", url.strip("/"))
        return os.path.join(self.cache_dir, f"{safe}.html")

    def _get_html(self, url: str) -> str:
        if self.use_cache:
            cp = self._cache_path(url)
            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    return f.read()

        delay = 0.0
        last_err = None
        for _ in range(self.retries):
            try:
                if delay:
                    time.sleep(delay)
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200 and resp.text:
                    html = resp.text
                    if self.use_cache:
                        with open(self._cache_path(url), "w", encoding="utf-8") as f:
                            f.write(html)
                    return html
                # retry on 403/429 or 5xx
                if resp.status_code in (403, 429) or 500 <= resp.status_code < 600:
                    delay = max(0.75, (delay or 0.6) * 1.8)
                    continue
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                last_err = e
                delay = max(0.75, (delay or 0.6) * 1.8)
                continue
        raise RuntimeError(f"GET failed for {url}: {last_err}")

    @staticmethod
    def _looks_like_cloudflare(html: str) -> bool:
        if not html:
            return True
        lowered = html.lower()
        return ("just a moment" in lowered and "cloudflare" in lowered) or (
            "verifying you are human" in lowered
        )

    # -------------------- Parse page --------------------

    @staticmethod
    def _extract_jsonld(html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[Dict[str, Any]] = []
        for tag in soup.find_all("script", {"type": "application/ld+json"}):
            txt = tag.string or tag.get_text("", strip=False)
            if not txt:
                continue
            try:
                data = json.loads(txt)
                if isinstance(data, dict):
                    out.append(data)
                elif isinstance(data, list):
                    out.extend([d for d in data if isinstance(d, dict)])
            except Exception:
                continue
        return out

    @staticmethod
    def _event_id_from_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        m = EVENT_ID_RE.search(url)
        return m.group(1) if m else None

    @staticmethod
    def _parse_date_from_url(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (date_local 'YYYY-MM-DD', time_local 'HH:MM') if present in URL.
        """
        if not url:
            return None, None
        m = DATE_FROM_URL_RE.search(url)
        if not m:
            return None, None
        month = int(m.group(1))
        day = int(m.group(2))
        yy = int(m.group(3))
        hour = int(m.group(4))
        ampm = m.group(5).lower()

        # convert hour to 24h
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

        year = 2000 + yy  # 20xx
        return f"{year:04d}-{month:02d}-{day:02d}", f"{hour:02d}:00"

    @staticmethod
    def _split_title(title: str) -> Tuple[Optional[str], Optional[str]]:
        # Simple heuristic: "Home vs. Away"
        if not title:
            return None, None
        parts = [p.strip() for p in title.split(" vs. ")]
        home, away = (parts + [None, None])[:2]
        return home, away

    @staticmethod
    def _titleize_from_url(url: str) -> str:
        tail = url.rstrip("/").split("/")[-1]
        tail = tail.replace("-tickets", "").replace("-", " ")
        return tail.title()

    def _infer_team_name(self, html: str, blocks: List[Dict[str, Any]], url: str) -> str:
        """
        Try JSON-LD first (SportsTeam names), then <title>/<h1>, then fallback to URL.
        """
        # 1) JSON-LD: look for SportsTeam name containing 'Football' first
        team_names = []
        for b in blocks:
            if b.get("@type") == "SportsTeam":
                nm = b.get("name")
                if isinstance(nm, str):
                    team_names.append(nm)
        if team_names:
            # Prefer names that explicitly say "Football"
            football = [n for n in team_names if "football" in n.lower()]
            if football:
                return football[0]
            return team_names[0]

        # 2) Page title / H1
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            t = soup.title.string.strip()
            # e.g., "Clemson Tigers Football Tickets | TickPick"
            if " | " in t:
                t = t.split(" | ", 1)[0]
            return t
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        # 3) Fallback: derive from URL tail
        return self._titleize_from_url(url)

    def _extract_offers(self, jsonld_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for block in jsonld_blocks:
            # Offers typically on an Event-like block; sometimes alongside SportsTeam
            offers = block.get("offers")
            if isinstance(offers, dict) or isinstance(offers, list):
                title = block.get("name") or block.get("headline") or ""
                link = None
                low, high = None, None

                def _consume_offer(offer: Dict[str, Any]):
                    nonlocal link, low, high
                    # prefer direct link from the offer, else from the enclosing block
                    link = offer.get("url") or block.get("url") or link
                    low_raw = offer.get("lowPrice")
                    high_raw = offer.get("highPrice")
                    try:
                        low = float(low_raw) if low_raw is not None else low
                    except Exception:
                        pass
                    try:
                        high = float(high_raw) if high_raw is not None else high
                    except Exception:
                        pass

                if isinstance(offers, dict):
                    _consume_offer(offers)
                else:
                    for off in offers:
                        if isinstance(off, dict):
                            _consume_offer(off)

                event_id = self._event_id_from_url(link)
                date_local, time_local = self._parse_date_from_url(link)
                home, away = self._split_title(title)

                rows.append(
                    {
                        "event_id": event_id,
                        "title": title,
                        "home_team_guess": home,
                        "away_team_guess": away,
                        "date_local": date_local,
                        "time_local": time_local,
                        "low_price": low,
                        "high_price": high,
                        "offer_url": link,
                        "source_team_url": getattr(self, "current_team_url", None),
                    }
                )
        return rows

    @staticmethod
    def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        De-duplicate by event_id if present, else by offer_url. Keep the row
        with the **lowest** low_price as the canonical record.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = r.get("event_id") or r.get("offer_url")
            if not key:
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = r
                continue
            a = prev.get("low_price")
            b = r.get("low_price")
            if (b is not None and a is None) or (b is not None and a is not None and b < a):
                best[key] = r
        return list(best.values())

    def _extract_faq_metrics_from_html(self, html: str) -> Dict[str, Optional[float]]:
        """
        Parse the team page FAQ section to find:
          - average ticket price (float)
          - tickets available (int)
        Returns {"avg_price": float|None, "tickets_available": int|None}
        """
        avg_price = None
        tickets_avail = None

        m_avg = AVG_PRICE_RE.search(html or "")
        if m_avg:
            try:
                avg_price = float(m_avg.group(1).replace(",", ""))
            except Exception:
                avg_price = None

        m_avail = TICKETS_AVAIL_RE.search(html or "")
        if m_avail:
            try:
                tickets_avail = int(m_avail.group(1).replace(",", ""))
            except Exception:
                tickets_avail = None

        return {"avg_price": avg_price, "tickets_available": tickets_avail}

    # -------------------- Per-event helpers for daily snapshot --------------------

    def get_summary_by_event_id(self, event_id: int) -> Dict[str, Optional[float]]:
        """
        Quick fetch of a single event by its ID.
        Returns dict with: lowest_price, average_price, listing_count, source_url (best-effort).
        """
        url = f"https://www.tickpick.com/ncaa-football/{int(event_id)}/"
        try:
            html = self._get_html(url)
        except Exception as e:
            return {"_error": str(e)}

        blocks = self._extract_jsonld(html)
        prices: Dict[str, Optional[float]] = {}

        def _consume_offer(offer: Dict[str, Any], fallback_url: str):
            # fill keys if present; do not overwrite with None
            lp = _safe_float(offer.get("lowPrice"))
            if lp is not None and prices.get("lowest_price") is None:
                prices["lowest_price"] = lp
            avg = _safe_float(offer.get("price"))
            if avg is not None and prices.get("average_price") is None:
                prices["average_price"] = avg
            inv = offer.get("inventoryLevel")
            inv_i = _safe_int(inv) if inv is not None else None
            if inv_i is not None and prices.get("listing_count") is None:
                prices["listing_count"] = inv_i
            if prices.get("source_url") is None:
                prices["source_url"] = offer.get("url") or fallback_url

        for b in blocks:
            if b.get("@type") == "Event":
                offers = b.get("offers")
                if isinstance(offers, dict):
                    _consume_offer(offers, b.get("url") or url)
                    break
                elif isinstance(offers, list):
                    for off in offers:
                        if isinstance(off, dict):
                            _consume_offer(off, b.get("url") or url)
                    break

        # Overlay FAQ metrics if present
        faq = self._extract_faq_metrics_from_html(html)
        if faq.get("avg_price") is not None:
            prices["average_price"] = faq["avg_price"]
        if faq.get("tickets_available") is not None:
            prices["listing_count"] = faq["tickets_available"]

        # Provide at least a source_url
        if prices and prices.get("source_url") is None:
            prices["source_url"] = url

        return prices or {"_error": "no price data"}

    def get_summary(self, event_id=None, home_team=None, away_team=None, start_dt=None) -> Dict[str, Optional[float]]:
        """
        Generic signature used as a fallback.
        """
        if event_id:
            return self.get_summary_by_event_id(int(event_id))
        return {}

    # -------------------- Export batch --------------------

    def _export(self, rows: List[Dict[str, Any]]) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"tickpick_prices_{ts}.csv")
        json_path = os.path.join(self.output_dir, f"tickpick_prices_{ts}.json")

        fields = [
            "event_id",
            "title",
            "home_team_guess",
            "away_team_guess",
            "date_local",
            "time_local",
            "low_price",
            "high_price",
            "offer_url",
            "source_team_url",
            "avg_price_from_page",
            "tickets_available_from_page",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        return {"csv": csv_path, "json": json_path}
