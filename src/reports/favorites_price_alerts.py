#!/usr/bin/env python3
"""Targeted price alerts for favorited games that are 7 days out or less.

Runs after every price snapshot (invoked from ``favorites_report.main``). Two alerts,
one text per triggering game:

  * NEW LOW  — the latest observed ``lowest_price`` is below every prior observation
               for that game (i.e. a new all-history low).
  * PRICE UP — the latest observed price is >= ALERT_RISE_PCT above the lowest price
               seen in the trailing ALERT_RISE_WINDOW_DAYS.

Dedupe state lives in ``data/daily/.favorites_alert_state.json`` so the same low / rise
is never texted twice. A NEW LOW re-arms the PRICE UP alert; the price falling back to
the recent low also re-arms it.

Channels (set in the ``.env`` file):
  ALERT_SMS_TO    comma-separated carrier email-to-SMS gateway addresses, e.g.
                  "8035551234@txt.att.net". Primary channel.
  ALERT_EMAIL_TO  optional extra plain-email recipient(s), comma-separated.
  Falls back to TO_EMAIL if neither is set.

Tunables (env, all optional):
  ALERT_MAX_DAYS=7             only games with 0 <= days_until_game <= this
  ALERT_RISE_PCT=0.10          rise threshold, as a fraction above the trailing low
  ALERT_RISE_WINDOW_DAYS=7     trailing window (days) for the rise baseline low
  ALERT_MIN_HISTORY=2          minimum non-null price observations before alerting
  FAVORITES_ALERTS_DISABLED=1  kill switch
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

from reports.favorites_report import (  # noqa: E402
    _load_favorites,
    _load_snapshots,
    _norm_event_id,
    _norm_event_id_series,
)

STATE_PATH = ROOT / "data" / "daily" / ".favorites_alert_state.json"

_EPS = 1e-9


# --------------------------------------------------------------------------- env

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


MAX_DAYS = _env_float("ALERT_MAX_DAYS", 7.0)
RISE_PCT = _env_float("ALERT_RISE_PCT", 0.10)
RISE_WINDOW_DAYS = _env_float("ALERT_RISE_WINDOW_DAYS", 7.0)
MIN_HISTORY = _env_int("ALERT_MIN_HISTORY", 2)


def _recipients() -> list[str]:
    raw = []
    for name in ("ALERT_SMS_TO", "ALERT_EMAIL_TO"):
        val = os.getenv(name, "") or ""
        raw += [a.strip() for a in val.split(",") if a.strip()]
    if not raw:
        fallback = os.getenv("TO_EMAIL", "") or ""
        raw = [a.strip() for a in fallback.split(",") if a.strip()]
    # de-dup, preserve order
    seen, out = set(), []
    for a in raw:
        if a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out


# ------------------------------------------------------------------------- state

def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".__tmp__")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        print(f"[favorites_price_alerts] could not persist state: {e}")


# -------------------------------------------------------------------- snapshot io

def _collected_dt(df: pd.DataFrame) -> pd.Series:
    """Best-effort snapshot timestamp from date_collected + time_collected."""
    if "date_collected" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    date_s = df["date_collected"].astype(str).str.strip()
    if "time_collected" in df.columns:
        combined = date_s + " " + df["time_collected"].astype(str).str.strip()
    else:
        combined = date_s
    try:
        dt = pd.to_datetime(combined, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        dt = pd.to_datetime(combined, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(date_s, errors="coerce")
    return dt


def _index_by_event(snaps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "event_id" not in snaps.columns:
        return {}
    df = snaps.copy()
    df["_price"] = pd.to_numeric(df.get("lowest_price"), errors="coerce")
    df["_dt"] = _collected_dt(df)
    df = df[df["_price"].notna() & df["_dt"].notna()]
    out: dict[str, pd.DataFrame] = {}
    for eid, grp in df.groupby(_norm_event_id_series(df["event_id"])):
        out[eid] = grp.sort_values("_dt").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------- days out

def _days_until_game(hist: pd.DataFrame, fav: dict) -> float:
    """Days from now until kickoff. Prefer the snapshot column, then startDateEastern
    (snapshot or favorite), else NaN."""
    if "days_until_game" in hist.columns:
        v = pd.to_numeric(hist["days_until_game"].iloc[-1], errors="coerce")
        if pd.notna(v):
            return float(v)
    for src in (
        hist["startDateEastern"].iloc[-1] if "startDateEastern" in hist.columns else None,
        fav.get("startDateEastern"),
    ):
        dt = pd.to_datetime(src, errors="coerce")
        if pd.notna(dt):
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.tz_localize(None)
            return (dt - pd.Timestamp.now()).total_seconds() / 86400.0
    return float("nan")


# --------------------------------------------------------------------- evaluation

def evaluate(favorites: list[dict], snaps: pd.DataFrame, state: dict) -> list[dict]:
    """Pure-ish: mutates ``state`` in place, returns the list of alerts to send.

    Alert dict: {eid, kind ('low'|'rise'), home, away, price, ref, days, pct}
    """
    by_event = _index_by_event(snaps)
    alerts: list[dict] = []

    for fav in favorites:
        eid = _norm_event_id(fav.get("event_id", ""))
        if not eid or eid not in by_event:
            continue
        hist = by_event[eid]
        if len(hist) < max(MIN_HISTORY, 2):
            continue

        days = _days_until_game(hist, fav)
        if pd.isna(days) or days < 0 or days > MAX_DAYS:
            continue

        prices = hist["_price"].to_numpy(dtype=float)
        dts = hist["_dt"]
        latest_price = float(prices[-1])
        latest_dt = dts.iloc[-1]
        prior_price = prices[:-1]
        prior_min = float(np.min(prior_price))

        # trailing-window low, excluding the latest observation
        w0 = latest_dt - pd.Timedelta(days=RISE_WINDOW_DAYS)
        in_window = prior_price[(dts.iloc[:-1] >= w0).to_numpy()]
        window_low = float(np.min(in_window)) if in_window.size else prior_min

        st = state.get(eid, {})
        low_alerted = st.get("low_alerted")
        rise_baseline = st.get("rise_baseline")

        home = fav.get("homeTeam", "?")
        away = fav.get("awayTeam", "?")

        fired_low = (
            latest_price < prior_min - _EPS
            and (low_alerted is None or latest_price < float(low_alerted) - _EPS)
        )
        if fired_low:
            alerts.append({
                "eid": eid, "kind": "low", "home": home, "away": away,
                "price": latest_price, "ref": prior_min, "days": days, "pct": None,
            })
            st["low_alerted"] = latest_price
            st["rise_baseline"] = None          # new low re-arms the rise alert
        elif (
            np.isfinite(window_low) and window_low > 0
            and latest_price >= window_low * (1.0 + RISE_PCT) - _EPS
        ):
            if rise_baseline is None or window_low < float(rise_baseline) - _EPS:
                pct = latest_price / window_low - 1.0
                alerts.append({
                    "eid": eid, "kind": "rise", "home": home, "away": away,
                    "price": latest_price, "ref": window_low, "days": days, "pct": pct,
                })
                st["rise_baseline"] = window_low
        elif rise_baseline is not None and latest_price <= window_low * (1.0 + _EPS):
            st["rise_baseline"] = None          # back at the recent low -> re-arm

        st["updated"] = datetime.now().isoformat(timespec="seconds")
        st["last_price"] = latest_price
        state[eid] = st

    return alerts


# ------------------------------------------------------------------------ sending

def _money(v: float) -> str:
    return f"${v:,.0f}" if abs(v - round(v)) < 0.01 else f"${v:,.2f}"


def _format(alert: dict) -> str:
    home, away = alert["home"], alert["away"]
    days = alert["days"]
    day_txt = "today" if days < 1 else f"{int(round(days))}d out"
    if alert["kind"] == "low":
        return (
            f"NEW LOW - {home} vs {away}\n"
            f"{_money(alert['price'])} (prev low {_money(alert['ref'])}) - {day_txt}"
        )
    return (
        f"PRICE UP {alert['pct']:.0%} - {home} vs {away}\n"
        f"{_money(alert['price'])} vs {_money(alert['ref'])} "
        f"{int(RISE_WINDOW_DAYS)}-day low - {day_txt}"
    )


def _send(alerts: list[dict]) -> bool:
    """Returns True if every alert was sent (or there were none)."""
    if not alerts:
        return True
    recips = _recipients()
    if not recips:
        print("[favorites_price_alerts] no recipients configured — printing instead:")
        for a in alerts:
            print("  " + _format(a).replace("\n", " | "))
        return False

    try:
        from reports.send_email import send_plain
    except Exception as e:
        print(f"[favorites_price_alerts] send_plain unavailable: {e}")
        return False

    ok = True
    for a in alerts:
        body = _format(a)
        try:
            send_plain(recips, "CFB ticket alert", body)
            print(f"[favorites_price_alerts] sent {a['kind']} alert for {a['eid']}")
        except Exception as e:
            ok = False
            print(f"[favorites_price_alerts] send failed for {a['eid']}: {e}")
    return ok


# --------------------------------------------------------------------- entrypoint

def run_alerts(favorites: list[dict] | None = None,
               snaps: pd.DataFrame | None = None) -> list[dict]:
    if os.getenv("FAVORITES_ALERTS_DISABLED", "").strip() in ("1", "true", "True"):
        print("[favorites_price_alerts] disabled via FAVORITES_ALERTS_DISABLED.")
        return []

    favorites = _load_favorites() if favorites is None else favorites
    if not favorites:
        print("[favorites_price_alerts] no favorites — skipping.")
        return []

    snaps = _load_snapshots() if snaps is None else snaps
    if snaps is None or snaps.empty:
        print("[favorites_price_alerts] no snapshot data — skipping.")
        return []

    first_run = not STATE_PATH.exists()
    state = _load_state()
    alerts = evaluate(favorites, snaps, state)

    if first_run:
        # Seed the dedupe state from whatever prices happen to look like right now,
        # without texting. Only genuine *changes* after this point alert. Opt in to
        # sending this first batch with ALERT_PRIME_SEND=1.
        if alerts and os.getenv("ALERT_PRIME_SEND", "").strip() not in ("1", "true", "True"):
            print(f"[favorites_price_alerts] first run — priming state, "
                  f"suppressing {len(alerts)} initial alert(s):")
            for a in alerts:
                print("    " + _format(a).replace("\n", " | "))
            alerts = []
        _save_state(state)
        if not alerts:
            print("[favorites_price_alerts] state primed.")
            return []

    if _send(alerts):
        _save_state(state)          # only persist if delivery succeeded, so a failed
    else:                          # send is retried on the next snapshot
        print("[favorites_price_alerts] delivery incomplete — state not advanced.")

    if not alerts:
        print("[favorites_price_alerts] nothing to alert.")
    return alerts


if __name__ == "__main__":
    run_alerts()
