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


def get_feature_importance() -> str:
    """
    Returns a markdown list of feature importances.
    - Uses model.feature_names_in_ if available (sklearn >=1.0+)
    - Otherwise falls back to your known training feature list
    """
    if not os.path.exists(MODEL_PATH):
        return "❌ Model file not found."

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return f"❌ Failed to load model: {e}"

    # Fallback static feature list that mirrors your training setup
    fallback_features = [
        "days_until_game", "capacity", "neutralSite", "conferenceGame",
        "isRivalry", "isRankedMatchup", "homeTeamRank", "awayTeamRank",
        # If you added these recently, they’ll render if present in the model:
        "weekNumber", "dayOfWeek", "kickoffHour"
    ]

    # Pull importances safely
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "❌ Model does not expose feature_importances_ (not a tree-based model?)."

    # Feature names if present; else fallback
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = fallback_features[: len(importances)]

    # Prefix or trim to align lengths
    n = min(len(features), len(importances))
    features = features[:n]
    importances = np.asarray(importances[:n])

    order = np.argsort(importances)[::-1]
    lines = [f"- {features[i]} (importance: {importances[i]:.3f})" for i in order]
    return "\n".join(lines)


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
    report.append(get_feature_importance() + "\n")

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
