# run_pipeline_now.py
from __future__ import annotations
import os, sys
from pathlib import Path

try:
    from src.cfb_tix.daemon import detect_paths, do_sync   # source layout
except Exception:
    from cfb_tix.daemon import detect_paths, do_sync       # installed layout

# --- Resolve repo + import daemon jobs ---
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
from cfb_tix import daemon as d

# Enable sync for this one call; scope and message set explicitly
os.environ["CFB_TIX_ENABLE_SYNC"] = "1"                  # push-only path is enabled
os.environ.setdefault("CFB_TIX_COMMIT_SCOPE", "data,models")
os.environ["CFB_TIX_COMMIT_MESSAGE"] = "manual snapshot" # exact message you wanted
os.environ["CFB_TIX_USE_RELEASE_SYNC"] = "0"             # never download/overwrite

paths = detect_paths()
do_sync(paths, "manual")

# Detect standard paths (daemon knows the repo)
p = d.detect_paths()

# Warn if sync isn’t disabled (we usually batch locally, then push once)
lock = ROOT / ".cfb_tix.NOSYNC"
if not lock.exists() and os.getenv("CFB_TIX_DISABLE_SYNC", "0") != "1":
    print("⚠️  Sync is ENABLED (no .cfb_tix.NOSYNC found). "
          "If you’re batching local runs, create the lock first.")

print("\n==> weekly_update")
d.job_weekly_update(p)

print("\n==> daily_snapshot")
d.job_daily_snapshot(p)

print("\n==> train_model")
d.job_train_model(p)

print("\n==> predict_price")
d.job_predict_price(p)

# --- Post-run summary ---
try:
    import pandas as pd
    snap = ROOT / "data" / "daily" / "price_snapshots.csv"
    pred = ROOT / "data" / "predicted" / "predicted_prices_optimal.csv"
    rows_snap = pd.read_csv(snap).shape[0] if snap.exists() else 0
    rows_pred = pd.read_csv(pred).shape[0] if pred.exists() else 0
    print("\n✅ Pipeline complete.")
    print(f"   SNAPSHOTS: {rows_snap:>5} rows  → {snap}")
    print(f"   PREDICTED: {rows_pred:>5} rows  → {pred}")
except Exception as e:
    print(f"\nℹ️  Summary skipped: {e}")
