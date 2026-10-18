"""
Stage 1 of the two-stage pricing pipeline: WHAT drives ticket price, and HOW MUCH.

Unlike train_catboost_min.py (which predicts the *future* price gap so the app can
tell you when to buy), this trains CatBoost to explain the *observed* price at each
snapshot directly: price = f(team/game/time/recent-price features). Because the
target is raw dollars, SHAP attributions come out in dollars too -- "being a rivalry
game adds $8.40 on average" rather than an abstract importance score.

Also switches on _add_price_history_features(), which train_catboost_min.py defines
but never calls -- recent price trend (price_prev, price_delta_pct, rolling mean) is
a real candidate driver and Stage 2 needs it too.

Outputs:
  reports/price_drivers_<year>.csv        one row per feature: mean |impact|, mean
                                           signed impact (for boolean features)
  reports/price_drivers_<year>_teams.csv  average $ contribution by team (home/away)
  models/price_driver_catboost.cbm        this model (separate from catboost_price_min.cbm)

This is read-only with respect to the production trajectory model / GUI -- it does
not touch catboost_price_min.cbm, catboost_report_<year>.csv, or catboost_features.csv.
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

from catboost import CatBoostRegressor, Pool  # noqa: E402

import modeling.train_catboost_min as tcm  # noqa: E402

YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
PROJ_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJ_DIR / "models" / "price_driver_catboost.cbm"
DRIVERS_PATH = PROJ_DIR / "reports" / f"price_drivers_{YEAR}.csv"
TEAM_DRIVERS_PATH = PROJ_DIR / "reports" / f"price_drivers_{YEAR}_teams.csv"

# Same numeric/categorical schema as the production model, with two exclusions
# found via VIF/collinearity checks (see MEMORY.md 2026-09-14 entry):
#   - price_prev/price_delta_pct/price_roll3_mean: ~perfectly autocorrelated with
#     the target (consecutive snapshots barely move), so CatBoost leans on them
#     almost exclusively and gives team/rank/capacity/rivalry ~zero SHAP credit --
#     fine for Stage 2 forecasting, wrong for "what structurally drives price".
#   - home/away_wins/losses_at_snapshot: VIF 4.7-7.8, collinear with `week` (more
#     games played -> more accumulated record) while contributing ~$0.04-0.21 each
#     individually -- dropping them took week's VIF from 34 to 2.15 at near-zero
#     information cost. snapshots_so_far (a scrape-cadence artifact, not an economic
#     driver) was worse still: VIF 119, and drove hours_until_game's VIF to 138.
_EXCLUDED_FOR_COLLINEARITY = {
    "home_wins_at_snapshot", "home_losses_at_snapshot",
    "away_wins_at_snapshot", "away_losses_at_snapshot",
}
NUMERIC_FEATURES = [f for f in tcm.NUMERIC_FEATURES if f not in _EXCLUDED_FOR_COLLINEARITY] + [
    "time_since_last_snapshot_hours",
]
CATEGORICAL_FEATURES = list(tcm.CATEGORICAL_FEATURES)
BOOLEAN_FEATURES = {"neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"}
TARGET = "lowest_price"


def _load_dataset() -> pd.DataFrame:
    print("[price_driver_report] Loading all seasons of snapshots …")
    df = tcm._load_all_snapshots()
    df = df[~tcm._is_postseason_row(df)].copy()
    df = tcm._ensure_event_id(df)
    df = tcm._coerce_booleans(df)
    df = tcm._build_targets(df)                # hours_until_game, kickoff_hour/dayofweek
    # Drop post-kickoff scrapes before building price-history features -- 2025 data
    # has some, and a post-game resale price isn't a real "ticket price" observation.
    df = df[df["hours_until_game"] > 0].copy()
    df = tcm._add_price_history_features(df)    # price_prev, price_delta_pct, rolling mean, …
    df, missing_cols = tcm._add_missing_indicators(df, NUMERIC_FEATURES)
    return df, missing_cols


def _prep_features(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for c in numeric_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("NA")
    # Only price observations that happened before kickoff (matches production
    # convention -- a post-kickoff price isn't a "ticket buying" price).
    df = df.dropna(subset=[TARGET, "hours_until_game"])
    df = df[df["hours_until_game"] > 0]
    return df


def train_driver_model() -> tuple[CatBoostRegressor, pd.DataFrame, list[str]]:
    df, missing_cols = _load_dataset()
    numeric_features = NUMERIC_FEATURES + missing_cols
    df = _prep_features(df, numeric_features)

    # _kickoff_ts is needed by _group_time_split but isn't a model feature.
    keep = ["event_id", "_kickoff_ts"] + numeric_features + CATEGORICAL_FEATURES + [TARGET]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    print(f"[price_driver_report] {len(df):,} priced, pre-kickoff snapshot rows "
          f"across {df['event_id'].nunique():,} events.")

    split = tcm._group_time_split(df, test_frac=0.15)
    train_df, test_df = df.iloc[split.train_idx], df.iloc[split.test_idx]

    # Indices must be positions within feature_cols (what's actually handed to Pool),
    # not within train_df.columns -- those differ once event_id/TARGET/_kickoff_ts are dropped.
    feature_cols = [c for c in train_df.columns if c not in ("event_id", "_kickoff_ts", TARGET)]
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]

    train_pool = Pool(train_df[feature_cols], train_df[TARGET], cat_features=cat_idx)
    test_pool = Pool(test_df[feature_cols], test_df[TARGET], cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2000,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        verbose=False,
    )
    print("[price_driver_report] Training …")
    model.fit(train_pool, eval_set=test_pool)

    pred = model.predict(test_pool)
    mae = float(np.mean(np.abs(pred - test_df[TARGET].to_numpy())))
    print(f"[price_driver_report] Held-out price MAE: ${mae:,.2f} "
          f"(explaining the observed price itself, not a future floor)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    return model, df[feature_cols], feature_cols


def compute_driver_report(model: CatBoostRegressor, X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]
    pool = Pool(X, cat_features=cat_idx)

    print("[price_driver_report] Computing SHAP values …")
    shap = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = shap[:, :-1]          # last column is the expected-value bias term
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols, index=X.index)

    rows = []
    for col in feature_cols:
        mean_abs = float(shap_df[col].abs().mean())
        entry = {"feature": col, "mean_abs_impact_usd": mean_abs}
        if col in BOOLEAN_FEATURES and col in X.columns:
            is_true = X[col].astype(str).isin(["True", "1", "true"])
            if is_true.any() and (~is_true).any():
                entry["avg_impact_when_true_usd"] = float(shap_df.loc[is_true, col].mean())
                entry["avg_impact_when_false_usd"] = float(shap_df.loc[~is_true, col].mean())
        rows.append(entry)

    report = pd.DataFrame(rows).sort_values("mean_abs_impact_usd", ascending=False).reset_index(drop=True)
    DRIVERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(DRIVERS_PATH, index=False)

    # Per-team average $ contribution (home and away identity, separately).
    team_rows = []
    for col, label in [("homeTeam", "home"), ("awayTeam", "away")]:
        if col not in X.columns:
            continue
        per_team = shap_df[col].groupby(X[col]).mean().sort_values(ascending=False)
        for team, val in per_team.items():
            team_rows.append({"team": team, "role": label, "avg_impact_usd": float(val)})
    team_report = pd.DataFrame(team_rows).sort_values("avg_impact_usd", ascending=False)
    team_report.to_csv(TEAM_DRIVERS_PATH, index=False)

    return report, team_report


def print_summary(report: pd.DataFrame, team_report: pd.DataFrame, top_n: int = 15) -> None:
    print("\n=== WHAT drives price, and HOW MUCH (mean |impact| in $, SHAP) ===")
    for _, r in report.head(top_n).iterrows():
        line = f"  {r['feature']:<38} ${r['mean_abs_impact_usd']:>7.2f}"
        if pd.notna(r.get("avg_impact_when_true_usd")):
            line += (f"   (true: {r['avg_impact_when_true_usd']:+.2f}, "
                      f"false: {r['avg_impact_when_false_usd']:+.2f})")
        print(line)

    if not team_report.empty:
        print("\nBiggest home-team price premiums:")
        home = team_report[team_report["role"] == "home"].head(8)
        for _, r in home.iterrows():
            print(f"  {r['team']:<32} {r['avg_impact_usd']:+7.2f}")
        print("\nBiggest home-team price discounts:")
        for _, r in team_report[team_report["role"] == "home"].tail(8).iterrows():
            print(f"  {r['team']:<32} {r['avg_impact_usd']:+7.2f}")

    print(f"\nFull tables: {DRIVERS_PATH}\n              {TEAM_DRIVERS_PATH}")


if __name__ == "__main__":
    model, X, feature_cols = train_driver_model()
    report, team_report = compute_driver_report(model, X, feature_cols)
    print_summary(report, team_report)
