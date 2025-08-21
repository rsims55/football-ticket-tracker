# src/scrapers/tickpick_pricer.py
__all__ = ["TickPickPricer"]

import os
import re
import json
import csv
import time
import random
from typing import List, Dict, Any, Optional, Iterable, Tuple, Union
from datetime import datetime
from collections import deque
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import cloudscraper  # optional; used automatically if present
except Exception:
    cloudscraper = None


# ---------- Regex helpers ----------
EVENT_ID_RE = re.compile(r"/(\d{6,})/?$")  # last numeric segment in the URL
DATE_FROM_URL_RE = re.compile(
    r"-([1-9]|1[0-2])-([1-9]|[12]\d|3[01])-(\d{2})-([1-9]|1[0-2])(am|pm)",
    re.IGNORECASE,
)
AVG_PRICE_RE = re.compile(
    r"average ticket price is \$\s*([0-9][0-9,]*(?:\.\d{2})?)", re.IGNORECASE
)
TICKETS_AVAIL_RE = re.compile(
    r"There are\s*([0-9][0-9,]*)\s*tickets still available", re.IGNORECASE
)

DEFAULT_HEADERS_BASE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7",
    "Cache-Control": "no-cache",
}
UA_POOL = [
    # A small, realistic pool (rotate per request)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:124.0) Gecko/20100101 Firefox/124.0",
]

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

def _read_proxies_from_source(proxies: Optional[Union[List[str], str]]) -> List[str]:
    if proxies is None:
        env = os.getenv("TICKPICK_PROXIES", "").strip()
        if not env:
            return []
        return [p.strip() for p in env.split(",") if p.strip()]
    if isinstance(proxies, list):
        return [p.strip() for p in proxies if str(p).strip()]
    path = str(proxies).strip()
    if os.path.isfile(path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out
    return [path]


class _ProxyRotator:
    """Rotate proxies after N successful requests; health‑check on switch."""
    def __init__(self, session: requests.Session, proxies: List[str], rotate_every: int = 50, timeout: int = 12, verbose: bool = True):
        self.session = session
        self.proxies_raw = proxies[:]
        self.rotate_every = max(1, int(rotate_every))
        self.timeout = timeout
        self.verbose = verbose
        self._index = -1
        self._requests_since_switch = 0
        self._current: Optional[Dict[str, str]] = None
        if self.proxies_raw:
            self._switch_proxy(initial=True)

    @property
    def active(self) -> Optional[Dict[str, str]]:
        return self._current

    def mark_request(self, success: bool) -> None:
        if not self._current:
            return
        if success:
            self._requests_since_switch += 1
            if self._requests_since_switch >= self.rotate_every:
                self._switch_proxy()
        else:
            self._switch_proxy()

    def _switch_proxy(self, initial: bool = False) -> None:
        if not self.proxies_raw:
            self._current = None
        tries = 0
        while self.proxies_raw and tries < len(self.proxies_raw):
            self._index = (self._index + 1) % len(self.proxies_raw)
            cand = self.proxies_raw[self._index].strip()
            proxy_dict = {"http": cand, "https": cand}
            if self._health_check(proxy_dict):
                self._current = proxy_dict
                self._requests_since_switch = 0
                if self.verbose:
                    print(f"{'🔌' if initial else '🔁'} Proxy set to [{self._index+1}/{len(self.proxies_raw)}]: {cand}")
                return
            else:
                if self.verbose:
                    print(f"⚠️ Proxy failed health check, skipping: {cand}")
            tries += 1
        if self.verbose:
            print("🚫 No working proxies available; continuing without proxy.")
        self._current = None

    def _health_check(self, proxy_dict: Dict[str, str]) -> bool:
        try:
            r = self.session.get("https://icanhazip.com", proxies=proxy_dict, timeout=self.timeout)
            return r.status_code == 200 and r.text.strip() != ""
        except Exception:
            return False


class _RateLimiter:
    """Min interval + rolling per-hour cap."""
    def __init__(self, min_interval_s: float = 6.0, max_per_hour: int = 120, verbose: bool = True):
        self.min_interval = float(min_interval_s)
        self.max_per_hour = int(max_per_hour)
        self.verbose = verbose
        self._last_ts = 0.0
        self._hits = deque()  # timestamps of requests in last hour

    def wait(self):
        now = time.time()
        # enforce min interval
        delta = now - self._last_ts
        if delta < self.min_interval:
            sleep_s = self.min_interval - delta + random.uniform(0.05, 0.25)  # tiny jitter
            if self.verbose:
                print(f"  ⏳ rate limit sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
        # enforce per-hour cap (rolling)
        cutoff = now - 3600.0
        while self._hits and self._hits[0] < cutoff:
            self._hits.popleft()
        if len(self._hits) >= self.max_per_hour:
            # sleep until the oldest hit ages out
            sleep_s = self._hits[0] + 3600.0 - now + 0.5
            if self.verbose:
                print(f"  ⏳ hourly cap reached; sleeping {sleep_s:.1f}s")
            time.sleep(max(0, sleep_s))
        # mark
        self._last_ts = time.time()
        self._hits.append(self._last_ts)


class TickPickPricer:
    """
    Scrape event offer info from TickPick team pages by parsing JSON-LD blocks.
    Polite mode: rate limiting, Retry-After handling, cache TTL, robots.txt (optional), UA rotation,
    Cloudflare stop rule, and proxy rotation.
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
        # proxy rotation
        proxies: Optional[Union[List[str], str]] = None,
        rotate_every: int = 50,
        # polite extras
        min_interval_s: float = 6.0,
        max_per_hour: int = 120,
        retry_after_cap_s: int = 120,
        cache_ttl_s: int = 24 * 3600,  # 1 day
        respect_robots: bool = True,
        cloudflare_stop_after: int = 5,
        cloudflare_window_s: int = 15 * 60,
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

        # session
        if cloudscraper:
            self.session = cloudscraper.create_scraper()
        else:
            self.session = requests.Session()

        # UA baseline (we’ll rotate per request)
        self.session.headers.update(DEFAULT_HEADERS_BASE)

        # proxy rotation
        proxy_list = _read_proxies_from_source(proxies)
        self._rotator = _ProxyRotator(
            session=self.session,
            proxies=proxy_list,
            rotate_every=rotate_every,
            timeout=min(12, self.timeout),
            verbose=self.verbose,
        )

        # rate limiter
        self._limiter = _RateLimiter(min_interval_s=min_interval_s, max_per_hour=max_per_hour, verbose=self.verbose)

        # polite config
        self.retry_after_cap_s = int(retry_after_cap_s)
        self.cache_ttl_s = int(cache_ttl_s)
        self.respect_robots = bool(respect_robots)
        self._robots_cache: Dict[str, Dict[str, Any]] = {}
        self._cf_events = deque()  # timestamps of cloudflare hits
        self.cloudflare_stop_after = int(cloudflare_stop_after)
        self.cloudflare_window_s = int(cloudflare_window_s)

    # -------------------- Public entry --------------------

    def run(self) -> Dict[str, str]:
        all_rows: List[Dict[str, Any]] = []

        for idx, url in enumerate(self.team_urls, 1):
            self.current_team_url = url

            if self.respect_robots and not self._robots_allowed(url):
                if self.verbose:
                    print(f"[{idx}/{len(self.team_urls)}] 🤖 robots.txt disallows {url} — skipping")
                continue

            try:
                html = self._get_html(url)
                self._rotator.mark_request(success=True)
            except Exception as e:
                self._rotator.mark_request(success=False)
                print(f"[{idx}/{len(self.team_urls)}] ❌ fetch failed: {url} :: {e}")
                self._polite_sleep()
                continue

            if self._looks_like_cloudflare(html):
                self._record_cloudflare_and_maybe_stop()
                print(f"[{idx}/{len(self.team_urls)}] ⛔ Cloudflare block for {url}")
                self._rotator.mark_request(success=False)
                self._polite_sleep()
                continue

            blocks = self._extract_jsonld(html)
            team_name = self._infer_team_name(html, blocks, url)
            if self.verbose:
                print(f"[{idx}/{len(self.team_urls)}] {team_name}")

            rows = self._extract_offers(blocks)
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
            print(f"CSV:  {paths['csv']}\nJSON: {paths['json']}")
        return paths

    # -------------------- Politeness helpers --------------------

    def _record_cloudflare_and_maybe_stop(self):
        now = time.time()
        cutoff = now - self.cloudflare_window_s
        while self._cf_events and self._cf_events[0] < cutoff:
            self._cf_events.popleft()
        self._cf_events.append(now)
        if len(self._cf_events) >= self.cloudflare_stop_after:
            raise RuntimeError("Too many Cloudflare blocks in a short window; aborting run politely.")

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or "/"
        # cache robots per host for 1 day
        entry = self._robots_cache.get(base)
        now = time.time()
        if not entry or now - entry["ts"] > 24 * 3600:
            try:
                r = self.session.get(f"{base}/robots.txt", timeout=min(10, self.timeout), proxies=self._rotator.active, headers=self._headers_for_request())
                text = r.text if r.status_code == 200 else ""
            except Exception:
                text = ""
            rules = self._parse_robots(text)
            self._robots_cache[base] = {"ts": now, "rules": rules}
        rules = self._robots_cache[base]["rules"]
        return self._robots_is_allowed(rules, path)

    @staticmethod
    def _parse_robots(text: str) -> Dict[str, List[str]]:
        # extremely small parser: collect Disallow lines under User-agent: *
        lines = [ln.strip() for ln in (text or "").splitlines()]
        ua_star = False
        disallows: List[str] = []
        for ln in lines:
            if not ln or ln.startswith("#"):
                continue
            if ln.lower().startswith("user-agent:"):
                ua = ln.split(":", 1)[1].strip()
                ua_star = (ua == "*" or ua == '"*"')
            elif ua_star and ln.lower().startswith("disallow:"):
                path = ln.split(":", 1)[1].strip()
                if path:
                    disallows.append(path)
        return {"*": disallows}

    @staticmethod
    def _robots_is_allowed(rules: Dict[str, List[str]], path: str) -> bool:
        dis = rules.get("*", [])
        for rule in dis:
            if path.startswith(rule):
                return False
        return True

    def _headers_for_request(self) -> Dict[str, str]:
        # rotate UA and add small header jitter
        headers = dict(DEFAULT_HEADERS_BASE)
        headers["User-Agent"] = random.choice(UA_POOL)
        if random.random() < 0.2:
            headers["Accept-Language"] = random.choice(["en-US,en;q=0.9", "en-US,en;q=0.7", "en-GB,en;q=0.8"])
        return headers

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

    def _fresh_cache_html(self, url: str) -> Optional[str]:
        if not self.use_cache:
            return None
        cp = self._cache_path(url)
        if os.path.exists(cp):
            age = time.time() - os.path.getmtime(cp)
            if age <= self.cache_ttl_s:
                with open(cp, "r", encoding="utf-8") as f:
                    return f.read()
        return None

    def _save_cache_html(self, url: str, html: str) -> None:
        if not self.use_cache:
            return
        cp = self._cache_path(url)
        with open(cp, "w", encoding="utf-8") as f:
            f.write(html)

    def _get_html(self, url: str) -> str:
        # cache TTL
        cached = self._fresh_cache_html(url)
        if cached is not None:
            if self.verbose:
                print("  💾 cache hit (fresh)")
            return cached

        delay = 0.0
        last_err = None
        consecutive_failures = 0

        for attempt in range(1, self.retries + 1):
            # global rate limit
            self._limiter.wait()

            try:
                if delay:
                    time.sleep(delay)

                resp = self.session.get(
                    url,
                    timeout=self.timeout,
                    proxies=self._rotator.active,
                    headers=self._headers_for_request(),
                )

                # Honor Retry-After (for 429/503 typically)
                if resp.status_code in (429, 503):
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        sleep_s = self._parse_retry_after(ra)
                        if self.verbose:
                            print(f"  🔁 Retry-After={sleep_s}s (status {resp.status_code})")
                        time.sleep(min(self.retry_after_cap_s, sleep_s))
                    # mark failure to encourage rotation next loop
                    self._rotator.mark_request(success=False)
                    delay = max(0.75, (delay or 0.6) * 1.8)
                    consecutive_failures += 1
                    continue

                if resp.status_code == 200 and resp.text:
                    html = resp.text
                    self._save_cache_html(url, html)
                    return html

                if resp.status_code in (403, 404) or 500 <= resp.status_code < 600:
                    # Treat as retryable (except 404 often permanent, but be gentle)
                    if self.verbose:
                        print(f"  ⚠️ HTTP {resp.status_code}; backoff & rotate")
                    self._rotator.mark_request(success=False)
                    delay = max(0.75, (delay or 0.6) * 1.8)
                    consecutive_failures += 1
                    continue

                # other statuses -> raise
                resp.raise_for_status()
                return resp.text

            except Exception as e:
                last_err = e
                self._rotator.mark_request(success=False)
                delay = max(0.75, (delay or 0.6) * 1.8)
                consecutive_failures += 1
                continue

            finally:
                # stop if we’re clearly hitting protection repeatedly
                if consecutive_failures >= 3:
                    # brief cool-off to be nice
                    cool = min(90, 10 * consecutive_failures)
                    if self.verbose:
                        print(f"  🧊 cooling off {cool}s after repeated failures")
                    time.sleep(cool)

        raise RuntimeError(f"GET failed for {url}: {last_err}")

    @staticmethod
    def _parse_retry_after(value: str) -> int:
        # Either seconds or HTTP-date
        value = value.strip()
        if value.isdigit():
            return int(value)
        try:
            dt = parsedate_to_datetime(value)
            return max(0, int((dt - datetime.utcnow()).total_seconds()))
        except Exception:
            return 30

    @staticmethod
    def _looks_like_cloudflare(html: str) -> bool:
        if not html:
            return True
        lowered = html.lower()
        return ("just a moment" in lowered and "cloudflare" in lowered) or ("verifying you are human" in lowered)

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
        if not url:
            return None, None
        m = DATE_FROM_URL_RE.search(url)
        if not m:
            return None, None
        month = int(m.group(1)); day = int(m.group(2)); yy = int(m.group(3)); hour = int(m.group(4))
        ampm = m.group(5).lower()
        if ampm == "pm" and hour != 12: hour += 12
        if ampm == "am" and hour == 12: hour = 0
        year = 2000 + yy
        return f"{year:04d}-{month:02d}-{day:02d}", f"{hour:02d}:00"

    @staticmethod
    def _split_title(title: str) -> Tuple[Optional[str], Optional[str]]:
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
        team_names = []
        for b in blocks:
            if b.get("@type") == "SportsTeam":
                nm = b.get("name")
                if isinstance(nm, str):
                    team_names.append(nm)
        if team_names:
            football = [n for n in team_names if "football" in n.lower()]
            return football[0] if football else team_names[0]
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            t = soup.title.string.strip()
            if " | " in t:
                t = t.split(" | ", 1)[0]
            return t
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        return self._titleize_from_url(url)

    def _extract_offers(self, jsonld_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for block in jsonld_blocks:
            offers = block.get("offers")
            if isinstance(offers, dict) or isinstance(offers, list):
                title = block.get("name") or block.get("headline") or ""
                link = None; low, high = None, None
                def _consume_offer(offer: Dict[str, Any]):
                    nonlocal link, low, high
                    link = offer.get("url") or block.get("url") or link
                    low_raw = offer.get("lowPrice"); high_raw = offer.get("highPrice")
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
        best: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = r.get("event_id") or r.get("offer_url")
            if not key:
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = r; continue
            a = prev.get("low_price"); b = r.get("low_price")
            if (b is not None and a is None) or (b is not None and a is not None and b < a):
                best[key] = r
        return list(best.values())

    def _extract_faq_metrics_from_html(self, html: str) -> Dict[str, Optional[float]]:
        avg_price = None; tickets_avail = None
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

    # -------------------- Per-event helpers --------------------

    def get_summary_by_event_id(self, event_id: int) -> Dict[str, Optional[float]]:
        url = f"https://www.tickpick.com/ncaa-football/{int(event_id)}/"
        try:
            resp_html = self._get_html(url)
            self._rotator.mark_request(success=True)
        except Exception as e:
            self._rotator.mark_request(success=False)
            return {"_error": str(e)}
        blocks = self._extract_jsonld(resp_html)
        prices: Dict[str, Optional[float]] = {}
        def _consume_offer(offer: Dict[str, Any], fallback_url: str):
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
                    _consume_offer(offers, b.get("url") or url); break
                elif isinstance(offers, list):
                    for off in offers:
                        if isinstance(off, dict):
                            _consume_offer(off, b.get("url") or url)
                    break
        faq = self._extract_faq_metrics_from_html(resp_html)
        if faq.get("avg_price") is not None:
            prices["average_price"] = faq["avg_price"]
        if faq.get("tickets_available") is not None:
            prices["listing_count"] = faq["tickets_available"]
        if prices and prices.get("source_url") is None:
            prices["source_url"] = url
        return prices or {"_error": "no price data"}

    def get_summary(self, event_id=None, home_team=None, away_team=None, start_dt=None) -> Dict[str, Optional[float]]:
        if event_id:
            return self.get_summary_by_event_id(int(event_id))
        return {}

    # -------------------- Export --------------------

    def _export(self, rows: List[Dict[str, Any]]) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"tickpick_prices_{ts}.csv")
        json_path = os.path.join(self.output_dir, f"tickpick_prices_{ts}.json")
        fields = [
            "event_id","title","home_team_guess","away_team_guess","date_local","time_local",
            "low_price","high_price","offer_url","source_team_url","avg_price_from_page","tickets_available_from_page",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return {"csv": csv_path, "json": json_path}
