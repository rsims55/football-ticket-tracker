"""
Full-season backtest: score the CURRENT model against every completed game.

evaluate_predictions.py only scores whatever happens to be sitting in
predicted_prices_optimal.csv, which predict_price.py prunes to "upcoming + last
24h" — so it can never look back further than a day. That's a cache limitation,
not a data limitation: price_snapshots.csv keeps every snapshot for every event
for the life of the season, so we can reconstruct what the current model would
have predicted for ANY completed game and check it against the real outcome.

Method (mirrors predict_price.py exactly, just without the "upcoming only" filter):
  - For each completed game (kickoff in the past), take its most recent
    pre-kickoff snapshot as the anchor row (same anchor predict_price.py uses
    for a live game the night before kickoff).
  - Run the current model's trajectory scan (gui/predict_trajectory.py) on that
    anchor to get a predicted floor price + predicted floor time.
  - Compare to the REAL observed floor price/time, computed straight from the
    full snapshot history for that event (independent of any cached file).

Output: data/predicted/backtest_<YEAR>.csv (one row per scored game) + a
printed summary. Safe to re-run any time; it never writes to the live
predicted_prices_optimal.csv or evaluation_metrics.csv used by the daily job.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from modeling.predict_price import (  # noqa: E402
    _load_snapshots,
    _observed_min_per_event,
    _latest_row_per_event,
    _infer_kickoff,
)
from gui.predict_trajectory import predict_trajectory_catboost  # noqa: E402
from modeling.online_price_model import (  # noqa: E402
    BucketedRLSState, predict_trajectory_online,
    load_enriched_snapshots as _load_enriched_snapshots,
    robust_latest_row_per_event as _robust_latest_row_per_event,
)

YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
PROJ_DIR = Path(__file__).resolve().parents[2]
ONLINE_STATE_PATH = PROJ_DIR / "models" / "online_price_state.npz"

SCAN_STEP_HOURS = int(os.getenv("PREDICT_STEP_HOURS", "6"))
SCAN_MAX_HOURS = int(os.getenv("PREDICT_MAX_HOURS", "2500"))
# Only score games whose snapshot collection has clearly wrapped up.
MIN_HOURS_PAST_KICKOFF = float(os.getenv("BACKTEST_MIN_HOURS_PAST", "6"))
# Anchor-selection outlier guard: a snapshot whose price is more than this many
# times above/below the SAME event's own trailing price_roll3_mean is treated as a
# bad scrape rather than a real price (see module docstring / online model's known
# anchor-fragility issue). Game-relative on purpose -- a thin-resale-market game
# that's consistently $300 compares fine against its own history and is never
# touched; only a single wild spike/crash relative to that game gets skipped.
ANCHOR_OUTLIER_RATIO = float(os.getenv("BACKTEST_OUTLIER_RATIO", "4.0"))

# _load_enriched_snapshots / _robust_latest_row_per_event now live in
# online_price_model.py (as load_enriched_snapshots / robust_latest_row_per_event)
# since predict_price.py needs the exact same loader for live predictions -- see
# that module's docstring for why skipping it silently breaks the model.


def _score_games(anchors: pd.DataFrame, obs: pd.DataFrame, predict_fn, label: str) -> pd.DataFrame:
    now = pd.Timestamp.now()
    rows = []
    skipped = 0

    for _, row in anchors.iterrows():
        row_dict = row.to_dict()
        kickoff = _infer_kickoff(row_dict)
        if pd.isna(kickoff) or (now - kickoff).total_seconds() / 3600.0 < MIN_HOURS_PAST_KICKOFF:
            skipped += 1
            continue
        if pd.isna(row.get("lowest_price")):
            skipped += 1
            continue

        eid = row.get("event_id")
        if eid not in obs.index:
            skipped += 1
            continue
        actual_price = obs.loc[eid, "observed_lowest_price_num"]
        actual_dt = pd.to_datetime(obs.loc[eid, "observed_lowest_dt"], errors="coerce")
        if pd.isna(actual_price):
            skipped += 1
            continue

        try:
            summary = predict_fn(row_dict)
        except Exception as e:
            skipped += 1
            print(f"[backtest:{label}]   SKIP {row.get('homeTeam','?')} vs {row.get('awayTeam','?')}: {e}")
            continue

        pred_price = summary.get("predicted_min_price", np.nan)
        pred_time = summary.get("predicted_min_time", pd.NaT)
        if not np.isfinite(pred_price):
            skipped += 1
            continue

        abs_error = abs(pred_price - actual_price)
        rows.append({
            "event_id": eid,
            "homeTeam": row.get("homeTeam"),
            "awayTeam": row.get("awayTeam"),
            "startDateEastern": kickoff,
            "week": row.get("week"),
            "anchor_snapshot_dt": row.get("collected_dt"),
            "anchor_price": row.get("lowest_price"),
            "predicted_lowest_price": round(float(pred_price), 2),
            "predicted_lowest_dt": pred_time,
            "actual_lowest_price": actual_price,
            "actual_lowest_dt": actual_dt,
            "abs_error": abs_error,
            "percent_error": abs_error / actual_price if actual_price > 0 else np.nan,
            "timing_signed_error_hours": (
                (pred_time - actual_dt).total_seconds() / 3600.0
                if pd.notna(pred_time) and pd.notna(actual_dt) else np.nan
            ),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["timing_abs_error_hours"] = out["timing_signed_error_hours"].abs()
        out = out.sort_values("startDateEastern")
        out_path = PROJ_DIR / "data" / "predicted" / f"backtest_{label}_{YEAR}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"[backtest:{label}] Scored {len(out)} completed games, skipped {skipped}. Wrote {out_path}")
    else:
        print(f"[backtest:{label}] Scored 0 completed games, skipped {skipped}.")
    return out


def run_backtest_catboost() -> pd.DataFrame:
    print(f"[backtest:catboost] Loading {YEAR} snapshots …")
    df = _load_snapshots()
    print(f"[backtest:catboost]   {len(df):,} rows, {df['event_id'].nunique():,} events")
    obs = _observed_min_per_event(df).set_index("event_id")
    anchors = _latest_row_per_event(df)

    def predict_fn(row_dict):
        _, summary = predict_trajectory_catboost(row_dict, hours_back=SCAN_MAX_HOURS, step_hours=SCAN_STEP_HOURS)
        return summary

    return _score_games(anchors, obs, predict_fn, "catboost")


RANGE_Z = float(os.getenv("BACKTEST_RANGE_Z", "1.0"))


def _add_range_coverage(out: pd.DataFrame, anchors: pd.DataFrame, state, z: float) -> pd.DataFrame:
    """For each scored game, also get the bucketed model's best-bucket price/timing
    RANGE (predict_price_range_online) and check whether the REAL floor price and
    REAL floor time actually fell inside it -- the direct validation of the
    range feature, not just the point estimate."""
    from modeling.online_price_model import predict_price_range_online
    anchors_by_id = anchors.set_index("event_id")
    lows, highs, win_starts, win_ends, labels, price_in, time_in = ([] for _ in range(7))

    for _, r in out.iterrows():
        eid = r["event_id"]
        rng = {}
        if eid in anchors_by_id.index:
            row_dict = anchors_by_id.loc[eid].to_dict()
            row_dict["event_id"] = eid
            try:
                rng = predict_price_range_online(row_dict, state, z=z)
            except Exception:
                rng = {}
        if not rng:
            lows.append(np.nan); highs.append(np.nan)
            win_starts.append(pd.NaT); win_ends.append(pd.NaT)
            labels.append(None); price_in.append(np.nan); time_in.append(np.nan)
            continue
        best = rng["best"]
        lo, hi, ws, we = best["price_low"], best["price_high"], best["window_start"], best["window_end"]
        lows.append(lo); highs.append(hi); win_starts.append(ws); win_ends.append(we); labels.append(best["bucket"])
        price_in.append(bool(lo <= r["actual_lowest_price"] <= hi))
        actual_dt = r["actual_lowest_dt"]
        time_in.append(bool(pd.notna(actual_dt) and ws <= actual_dt <= we) if pd.notna(actual_dt) else np.nan)

    out = out.copy()
    out["range_price_low"], out["range_price_high"] = lows, highs
    out["range_window_start"], out["range_window_end"] = win_starts, win_ends
    out["range_bucket"] = labels
    out["price_in_range"], out["time_in_range"] = price_in, time_in
    return out


def run_backtest_online() -> pd.DataFrame:
    if not BucketedRLSState.exists(ONLINE_STATE_PATH):
        print(f"[backtest:online] No state at {ONLINE_STATE_PATH} — run online_price_update.py first.")
        return pd.DataFrame()
    state = BucketedRLSState.load(ONLINE_STATE_PATH)
    print(f"[backtest:online] Loading {YEAR} snapshots (enriched, season-restricted) …")
    df = _load_enriched_snapshots(restrict_to_season=YEAR)
    print(f"[backtest:online]   {len(df):,} rows, {df['event_id'].nunique():,} events")
    obs = _observed_min_per_event(df).set_index("event_id")
    anchors = _robust_latest_row_per_event(df, outlier_ratio=ANCHOR_OUTLIER_RATIO)

    def predict_fn(row_dict):
        _, summary = predict_trajectory_online(row_dict, state, hours_back=SCAN_MAX_HOURS, step_hours=SCAN_STEP_HOURS)
        return summary

    out = _score_games(anchors, obs, predict_fn, "online")
    if not out.empty:
        out = _add_range_coverage(out, anchors, state, RANGE_Z)
        bucket_counts = out["range_bucket"].value_counts()
        print(f"[backtest:online] best-bucket distribution across games:\n{bucket_counts.to_string()}")
    return out


def summarize(out: pd.DataFrame, label: str) -> dict:
    if out.empty:
        print(f"[backtest:{label}] Nothing to summarize.")
        return {}

    mae = out["abs_error"].mean()
    rmse = np.sqrt((out["abs_error"] ** 2).mean())
    within5 = (out["percent_error"] <= 0.05).mean()
    t = out["timing_abs_error_hours"].dropna()
    bias = out["timing_signed_error_hours"].dropna().mean()
    within24 = (t <= 24).sum() if len(t) else 0
    within6 = (t <= 6).sum() if len(t) else 0

    print(f"\n=== Full-season backtest [{label}] — {len(out)} completed games ===")
    print(f"Price MAE:           ${mae:,.2f}")
    print(f"Price RMSE:          ${rmse:,.2f}")
    print(f"Price within 5%:     {within5:.1%}")
    if len(t):
        print(f"Timing MAE:          {t.mean():.1f} h  (median {t.median():.1f} h)")
        print(f"Timing within 24h:   {within24}/{len(t)}")
        print(f"Timing within 6h:    {within6}/{len(t)}")
        direction = "later" if bias > 0 else "earlier"
        print(f"Bias: predicted floor avg {abs(bias):.1f} h {direction} than the real floor")

    if "price_in_range" in out.columns:
        pir = out["price_in_range"].dropna()
        tir = out["time_in_range"].dropna()
        if len(pir):
            print(f"Real price inside predicted range:  {pir.sum()}/{len(pir)} ({pir.mean():.1%})")
        if len(tir):
            print(f"Real floor time inside predicted window: {tir.sum()}/{len(tir)} ({tir.mean():.1%})")
        n_buckets_used = out["range_bucket"].nunique()
        print(f"Distinct best-buckets picked across {len(out)} games: {n_buckets_used} "
              f"(1 would mean every game still converges to the same window)")

    print("\nWorst 10 misses by price error:")
    cols = ["homeTeam", "awayTeam", "startDateEastern", "predicted_lowest_price",
            "actual_lowest_price", "abs_error", "percent_error"]
    print(out.sort_values("abs_error", ascending=False)[cols].head(10).to_string(index=False))

    return {
        "label": label, "n": len(out), "price_mae": mae, "price_rmse": rmse,
        "within_5pct": within5, "timing_mae_h": t.mean() if len(t) else np.nan,
        "timing_median_h": t.median() if len(t) else np.nan,
        "within_24h": within24, "within_6h": within6, "n_timing": len(t), "bias_h": bias,
    }


if __name__ == "__main__":
    which = os.getenv("BACKTEST_MODEL", "both")  # catboost | online | both
    summaries = []
    if which in ("catboost", "both"):
        summaries.append(summarize(run_backtest_catboost(), "catboost"))
    if which in ("online", "both"):
        summaries.append(summarize(run_backtest_online(), "online"))

    summaries = [s for s in summaries if s]
    if len(summaries) == 2:
        print("\n=== Side-by-side comparison ===")
        a, b = summaries
        for k in ["n", "price_mae", "price_rmse", "within_5pct", "timing_mae_h",
                  "timing_median_h", "within_24h", "within_6h", "bias_h"]:
            print(f"  {k:<16} {a['label']:>10}: {a[k]!s:<12}  {b['label']:>10}: {b[k]!s}")
