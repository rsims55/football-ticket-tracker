#!/usr/bin/env python3
"""
Run a 3-way CatBoost sweep (no-prune, 99%, 95%), collect metrics, and pick best.
"""
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "src" / "modeling" / "train_catboost_min.py"
REPORT_DIR = ROOT / "reports"

RUNS = [
    {"name": "no_prune", "env": {"PRUNE_FEATURES": "0"}},
    {"name": "prune_99", "env": {"PRUNE_FEATURES": "1", "PRUNE_IMPORTANCE_CUTOFF": "0.99"}},
    {"name": "prune_95", "env": {"PRUNE_FEATURES": "1", "PRUNE_IMPORTANCE_CUTOFF": "0.95"}},
]

# Ranking: prefer higher price_within_5pct, then lower price_mae
PRIMARY_METRIC = "price_within_5pct"
SECONDARY_METRIC = "price_mae"


def run_one(name: str, overrides: dict) -> pd.DataFrame:
    report_path = REPORT_DIR / f"catboost_report_{name}.csv"
    feats_path = REPORT_DIR / f"catboost_features_{name}.csv"
    env = os.environ.copy()
    env.update(overrides)
    # Avoid pruning feature list based on previous run's feature importances
    env["PRUNE_FEATURES"] = overrides.get("PRUNE_FEATURES", "0")
    env["REPO_DATA_LOCK"] = "0"
    env["MODEL_REPORT_PATH"] = str(report_path)
    env["MODEL_FEATURES_PATH"] = str(feats_path)

    print(f"\n=== Running {name} ===", flush=True)
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        env=env,
        cwd=str(ROOT),
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    df = pd.read_csv(report_path)
    if df.empty:
        raise RuntimeError(f"Empty report for {name}")
    df["run_name"] = name
    return df


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in RUNS:
        rows.append(run_one(run["name"], run["env"]).iloc[0])

    summary = pd.DataFrame(rows)
    cols = [
        "run_name",
        "rows_total_used",
        "price_mae",
        "price_rmse",
        "price_within_5pct",
        "gap_pct_mae",
        "gap_pct_within_0p05",
        "time_mae_hours",
        "time_within_24h",
    ]
    cols = [c for c in cols if c in summary.columns]
    summary = summary[cols]

    print("\n=== Sweep Summary ===")
    print(summary.to_string(index=False))

    if PRIMARY_METRIC in summary.columns:
        summary = summary.sort_values(
            by=[PRIMARY_METRIC, SECONDARY_METRIC],
            ascending=[False, True],
        )
        best = summary.iloc[0]
        print("\n=== Best Run ===")
        print(best.to_string())

    out_path = REPORT_DIR / "catboost_sweep_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved sweep summary → {out_path}")


if __name__ == "__main__":
    main()
