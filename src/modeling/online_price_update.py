"""
Cold-start or incrementally update the Stage 2 online price model (bucketed --
see online_price_model.py's module docstring for the architecture).

  - No state file yet: build the team/conference vocabulary from ALL historical
    data, do one closed-form ridge-regression fit PER TIME BUCKET (the RLS warm
    start), save state with a watermark = the newest snapshot timestamp used.
  - State exists: load it, pull only snapshots newer than the watermark, and apply
    true incremental RLS updates row by row in chronological order -- each row
    updates only ITS OWN bucket's model.

This is meant to run right after each snapshot batch (currently wired into
job_daily_snapshot in daemon.py, NOT job_near_term_snapshot -- see the comment
there for why). See [[pricing-redesign]] in project memory.

State: models/online_price_state.npz + .json (see online_price_model.BucketedRLSState).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from modeling.online_price_model import (  # noqa: E402
    BucketedRLSState, Vocab, ridge_batch_fit_bucketed, update_bucketed,
    load_enriched_snapshots, TARGET, BUCKET_LABELS,
)

PROJ_DIR = Path(__file__).resolve().parents[2]
STATE_PATH = PROJ_DIR / "models" / "online_price_state.npz"
RIDGE = float(os.getenv("ONLINE_RIDGE", "10.0"))
FORGETTING_FACTOR = float(os.getenv("ONLINE_LAMBDA", "0.999"))


def _load_and_prep() -> pd.DataFrame:
    print("[online_price_update] Loading all seasons of snapshots …")
    df = load_enriched_snapshots()   # no season restriction: train on everything available
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    import numpy as np
    df["_y"] = np.log1p(df[TARGET].clip(lower=0))
    df = df.dropna(subset=["_y", "hours_until_game", "_snapshot_ts"])
    return df.sort_values("_snapshot_ts")


def cold_start(df: pd.DataFrame) -> BucketedRLSState:
    print(f"[online_price_update] Cold start: fitting on {len(df):,} rows …")
    vocab = Vocab.build(df)
    print(f"[online_price_update]   vocab: {len(vocab.teams)} teams, {len(vocab.conferences)} conferences")
    t0 = time.time()
    state = ridge_batch_fit_bucketed(df, "_y", vocab, ridge=RIDGE)
    print(f"[online_price_update]   bucketed ridge fit done in {time.time()-t0:.1f}s")
    for lbl in BUCKET_LABELS:
        st = state.buckets[lbl]
        print(f"[online_price_update]     {lbl:>10}: {st.n_updates:>7,} rows, sigma2={st.sigma2:.4f}")
    state.watermark = df["_snapshot_ts"].max().isoformat()
    state.n_updates = len(df)
    return state


def incremental_update(state: BucketedRLSState, df: pd.DataFrame) -> BucketedRLSState:
    watermark = pd.to_datetime(state.watermark) if state.watermark else pd.Timestamp.min
    new_rows = df[df["_snapshot_ts"] > watermark].sort_values("_snapshot_ts")
    if new_rows.empty:
        print("[online_price_update] No new snapshots since last update — nothing to do.")
        return state

    print(f"[online_price_update] Applying {len(new_rows):,} incremental RLS updates …")
    state = update_bucketed(state, new_rows, y_col="_y", lam=FORGETTING_FACTOR)
    state.watermark = new_rows["_snapshot_ts"].max().isoformat()
    return state


def run() -> BucketedRLSState:
    df = _load_and_prep()
    if BucketedRLSState.exists(STATE_PATH):
        state = BucketedRLSState.load(STATE_PATH)
        print(f"[online_price_update] Loaded state: {state.n_updates:,} rows seen, "
              f"watermark={state.watermark}")
        state = incremental_update(state, df)
    else:
        state = cold_start(df)
    state.save(STATE_PATH)
    print(f"[online_price_update] Saved state to {STATE_PATH} "
          f"(total rows seen: {state.n_updates:,}, watermark={state.watermark})")
    return state


if __name__ == "__main__":
    run()
