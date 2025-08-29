#!/usr/bin/env python3
"""
Generate a weekly model report:
- Best predictors (feature importances)
- Accuracy over the past 7 days
- Table of predicted vs actual with errors
- Suggestions / next variables to explore
- Saves markdown report and a CSV excerpt for the past week
- Optionally emails the report if reports/send_email.py is available

Run:
  python src/reports/generate_weekly_report.py
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Paths / Config
# -----------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/ticket_price_model.pkl")

# You moved these under data/predicted/
EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", "data/predicted/evaluation_metrics.csv")
MERGED_OUTPUT = os.getenv("MERGED_OUTPUT", "data/predicted/merged_eval_results.csv")

REPORT_DIR = os.getenv("REPORT_DIR", "reports")
WEEKLY_DIR = os.path.join(REPORT_DIR, "weekly")
WEEK_WINDOW_DAYS = int(os.getenv("WEEK_WINDOW_DAYS", "7"))

# Optional email recipient (you already keep this private in your env or code)
REPORT_RECIPIENT = os.getenv("WEEKLY_REPORT_EMAIL", "")  # e.g., "randisims55@gmail.com"


# -----------------------
# Helpers
# -----------------------
def _load_eval_df() -> pd.DataFrame:
    """
    Prefer the merged evaluation file if it exists, else fall back to the raw log.
    Expected columns include at least:
      startDateEastern, homeTeam, awayTeam, predicted_lowest_price, actual_lowest_price,
      abs_error, percent_error
    """
    path = MERGED_OUTPUT if os.path.exists(MERGED_OUTPUT) else EVAL_LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Normalize date
    if "startDateEastern" in df.columns:
        df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce").dt.date
    else:
        # If missing, try a common alt name
        for alt in ("start_date", "game_date", "date_est_only"):
            if alt in df.columns:
                df["startDateEastern"] = pd.to_datetime(df[alt], errors="coerce").dt.date
                break

    # Coerce numeric columns if they exist
    for col in ("predicted_lowest_price", "actual_lowest_price", "abs_error", "percent_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_recent_evaluations(window_days: int = WEEK_WINDOW_DAYS) -> pd.DataFrame:
    df = _load_eval_df()
    if df.empty or "startDateEastern" not in df.columns:
        return pd.DataFrame()

    cutoff = datetime.now().date() - timedelta(days=window_days)
    recent = df[df["startDateEastern"] >= cutoff].copy()

    # Sort for a nicer table (largest misses first)
    if "abs_error" in recent.columns:
        recent.sort_values(by="abs_error", ascending=False, inplace=True)

    return recent


def get_feature_importance(top_k: int = 20) -> tuple[str, list[str]]:
    """
    Returns (markdown_text, weak_features_list).
    Handles:
      - Pipeline(ColumnTransformer(...), RandomForestRegressor)
      - Bare RandomForestRegressor with feature_names_in_
    Produces:
      - Expanded top-K transformed features
      - Aggregated importances by original input column (sum over one-hot levels)
    """
    if not os.path.exists(MODEL_PATH):
        return "❌ Model file not found.", []

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return f"❌ Failed to load model: {e}", []

    # --- unwrap pipeline if present ---
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = None
    estimator = model
    feature_names_expanded = None

    if isinstance(model, Pipeline):
        # try to locate final estimator and a preprocessor
        estimator = getattr(model, "steps", [(-1, model)])[-1][1]
        for _, step in model.steps:
            if hasattr(step, "get_feature_names_out"):
                preprocessor = step
                break
            if isinstance(step, ColumnTransformer):
                preprocessor = step

    # we need a tree-based estimator with feature_importances_
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return "❌ Model does not expose feature_importances_.", []

    importances = np.asarray(importances)

    # --- expanded feature names from preprocessor if available ---
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names_expanded = preprocessor.get_feature_names_out()
        except Exception:
            feature_names_expanded = None

    if feature_names_expanded is None:
        # fall back to feature_names_in_ if lengths match, else generic names
        if hasattr(estimator, "feature_names_in_") and len(estimator.feature_names_in_) == len(importances):
            feature_names_expanded = estimator.feature_names_in_
        else:
            feature_names_expanded = np.array([f"feature_{i}" for i in range(len(importances))])

    # Length guard
    n = min(len(feature_names_expanded), len(importances))
    feature_names_expanded = np.asarray(feature_names_expanded[:n], dtype=str)
    importances = importances[:n]

    # --- build expanded (top-K) markdown ---
    order = np.argsort(importances)[::-1]
    top_idx = order[:top_k]
    lines_expanded = [
        _humanize_feature(feature_names_expanded[i], importances[i])
        for i in top_idx
    ]


    # --- aggregate to original columns (sum one-hot levels) ---
    # Heuristic mapping: ColumnTransformer usually yields names like "onehot__col_value"
    # or "remainder__col". We map back to `col` by:
    #   1) strip "<prefix>__" (anything before the first "__")
    #   2) for one-hot, split from the RIGHT on "_" once: "col_value" -> base "col"
    # This preserves columns that already include "_" in their names.
    base_map = {}
    for name, imp in zip(feature_names_expanded, importances):
        # 1) remove transformer prefix
        base = name.split("__", 1)[-1]
        # 2) collapse one-hot suffix to base column
        if "_" in base:
            # rsplit once; if it's not one-hot, this still leaves base intact in most cases
            base = base.rsplit("_", 1)[0]
        base_map[base] = base_map.get(base, 0.0) + float(imp)

    # Sort aggregated importances
    agg_items = sorted(base_map.items(), key=lambda x: x[1], reverse=True)
    lines_agg = [f"- {k}: {v:.4f}" for k, v in agg_items[:top_k]]

    # Identify weak/orphan features (importance < 0.01 after aggregation)
    weak_features = [k for k, v in agg_items if v < 0.01]

    # Compose markdown section
    md = []
    md.append("### Top Transformed Features (expanded)")
    md.extend(lines_expanded if lines_expanded else ["(none)"])
    md.append("\n### Aggregated by Original Column")
    md.extend(lines_agg if lines_agg else ["(none)"])
    if weak_features:
        md.append("\n**Possibly unrelated (near-zero importance):** " + ", ".join(weak_features[:20]))

    return "\n".join(md), weak_features


def _safe_rmse(df: pd.DataFrame) -> float:
    """
    Compute RMSE from predicted vs actual when available; otherwise approximate
    from abs_error if that's all we have (less ideal).
    """
    if {"predicted_lowest_price", "actual_lowest_price"}.issubset(df.columns):
        diff2 = (df["predicted_lowest_price"] - df["actual_lowest_price"]) ** 2
        return float(np.sqrt(np.nanmean(diff2)))
    if "abs_error" in df.columns:
        return float(np.sqrt(np.nanmean((df["abs_error"]) ** 2)))
    return float("nan")


def _format_currency(x) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return ""
    
def _humanize_feature(name: str, importance: float) -> str:
    # Strip transformers like "num__" or "cat__"
    if "__" in name:
        prefix, base = name.split("__", 1)
    else:
        prefix, base = "", name

    if prefix == "num":
        return f"- {base.replace('_',' ')} was important, contributing {importance:.1%} to predictions."
    elif prefix == "cat":
        # split conference features nicely
        if "Conference_" in base:
            col, val = base.split("_", 1)
            return f"- Teams from the {val} {col.replace('Conference','conference').lower()} mattered, contributing {importance:.1%}."
        else:
            return f"- {base.replace('_',' ')} category influenced predictions (~{importance:.1%})."
    else:
        return f"- {base.replace('_',' ')} influenced predictions (~{importance:.1%})."


def build_report() -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(WEEKLY_DIR, exist_ok=True)

    report_md_path = os.path.join(WEEKLY_DIR, f"weekly_report_{today_str}.md")
    recent_csv_path = os.path.join(WEEKLY_DIR, f"weekly_eval_rows_{today_str}.csv")

    report = [f"# 📈 Weekly Ticket Price Model Report\n**Date:** {today_str}\n"]

    # -----------------------
    # Section 1: Feature Importance
    # -----------------------
    report.append("## 🔍 Best Predictors of Ticket Price\n")
    fi_text, weak_features = get_feature_importance()
    report.append(fi_text + "\n")

    # -----------------------
    # Section 2: Accuracy (Past Week)
    # -----------------------
    df = get_recent_evaluations(WEEK_WINDOW_DAYS)
    if df.empty:
        report.append("## 📊 Model Accuracy (Past 7 Days)\nNo games to evaluate in the past week.\n")
    else:
        # Persist the slice for easier inspection later
        cols_to_save = [
            c for c in [
                "startDateEastern", "homeTeam", "awayTeam",
                "predicted_lowest_price", "actual_lowest_price",
                "abs_error", "percent_error", "weekNumber",
                "dayOfWeek", "kickoffHour"
            ] if c in df.columns
        ]
        df[cols_to_save].to_csv(recent_csv_path, index=False)

        report.append("## 📊 Model Accuracy (Past 7 Days)\n")
        report.append(f"- Games evaluated: **{len(df)}**")

        mae = float(df["abs_error"].mean()) if "abs_error" in df.columns else float("nan")
        rmse = _safe_rmse(df)
        over_5 = int((df["percent_error"] > 0.05).sum()) if "percent_error" in df.columns else 0

        if not np.isnan(mae):
            report.append(f"- MAE: **{_format_currency(mae)}**")
        if not np.isnan(rmse):
            report.append(f"- RMSE: **{_format_currency(rmse)}**")
        if "percent_error" in df.columns:
            report.append(f"- Games > 5% error: **{over_5} / {len(df)}**\n")

        # -----------------------
        # Section 3: Table of predictions
        # -----------------------
        report.append("## 🎯 Predicted vs Actual Prices\n")
        report.append("| Game | Date (ET) | Predicted | Actual | Abs Error | % Error |")
        report.append("|------|-----------|-----------|--------|-----------|---------|")

        for _, row in df.iterrows():
            game = f"{row.get('homeTeam','')} vs {row.get('awayTeam','')}"
            date_str = row.get("startDateEastern", "")
            p = row.get("predicted_lowest_price", float("nan"))
            a = row.get("actual_lowest_price", float("nan"))
            ae = row.get("abs_error", float("nan"))
            pe = row.get("percent_error", float("nan"))
            pe_pct = f"{pe * 100:.1f}%" if pd.notna(pe) else ""
            report.append(
                f"| {game} | {date_str} | {_format_currency(p)} | {_format_currency(a)} | {_format_currency(ae)} | {pe_pct} |"
            )

        # -----------------------
        # Section 4: Heuristic suggestions
        # -----------------------
        report.append("\n## 💡 Suggestions")
        # If many misses > 5%, prompt deeper diagnostics
        if "percent_error" in df.columns and len(df) > 0 and (over_5 / len(df)) > 0.40:
            report.append("- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.")
        # Missing rankings?
        if any(col in df.columns for col in ["homeTeamRank", "awayTeamRank"]):
            if df.get("homeTeamRank") is not None and df["homeTeamRank"].isna().any():
                report.append("- Some home rankings are missing; verify postseason/final AP pulls.")
            if df.get("awayTeamRank") is not None and df["awayTeamRank"].isna().any():
                report.append("- Some away rankings are missing; verify postseason/final AP pulls.")
        # Always-on ideas
        report.append("- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).")
        report.append("- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.")
        report.append("- Check stadium capacity normalization (capacity vs. sold % if/when available).\n")

        # Flag near-zero-importance features
        if weak_features:
            report.append(
                "- Near-zero importance this week (may be unrelated): "
                + ", ".join(sorted(set(weak_features))[:20])
            )

    # -----------------------
    # Write report file
    # -----------------------
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✅ Weekly report saved to {report_md_path}")
    if not df.empty and os.path.exists(recent_csv_path):
        print(f"🗂  Weekly eval rows saved to {recent_csv_path}")

    # -----------------------
    # Optional email hook
    # -----------------------
    # If you have reports/send_email.py with a send_markdown_report(path, to) function,
    # this will email the markdown. Otherwise it's a no-op.
    try:
        if REPORT_RECIPIENT:
            from reports.send_email import send_markdown_report  # your existing helper
            send_markdown_report(report_md_path, REPORT_RECIPIENT)
            print(f"📧 Report emailed to {REPORT_RECIPIENT}")
    except Exception as e:
        print(f"⚠️  Skipping email send (not configured or failed): {e}")

    return report_md_path


if __name__ == "__main__":
    build_report()
