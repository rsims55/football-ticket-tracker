import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/ticket_price_model.pkl"
EVAL_LOG_PATH = "data/evaluation_metrics.csv"
REPORT_DIR = "reports"

def get_feature_importance():
    if not os.path.exists(MODEL_PATH):
        return "❌ Model file not found."

    model = joblib.load(MODEL_PATH)
    features = [
        "days_until_game", "capacity", "neutralSite", "conferenceGame",
        "isRivalry", "isRankedMatchup", "homeTeamRank", "awayTeamRank"
    ]
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    return "\n".join(
        [f"- {features[i]} (importance: {importances[i]:.3f})" for i in sorted_idx]
    )

def get_recent_evaluations():
    if not os.path.exists(EVAL_LOG_PATH):
        return pd.DataFrame()

    df = pd.read_csv(EVAL_LOG_PATH)
    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"]).dt.date
    one_week_ago = datetime.now().date() - timedelta(days=7)
    return df[df["startDateEastern"] >= one_week_ago]

def build_report():
    today_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"weekly_report_{today_str}.md")

    report = [f"# 📈 Weekly Ticket Price Model Report\n**Date:** {today_str}\n"]

    # Section 1: Feature Importance
    report.append("## 🔍 Best Predictors of Ticket Price\n")
    report.append(get_feature_importance() + "\n")

    # Section 2: Evaluation Metrics
    df = get_recent_evaluations()
    if df.empty:
        report.append("## 📊 Model Accuracy\nNo games to evaluate in the past week.\n")
    else:
        report.append("## 📊 Model Accuracy\n")
        report.append(f"- Number of games evaluated: {len(df)}")
        mae = df["abs_error"].mean()
        rmse = (df["abs_error"] ** 2).mean() ** 0.5
        over_5 = (df["percent_error"] > 0.05).sum()
        report.append(f"- MAE: ${mae:.2f}")
        report.append(f"- RMSE: ${rmse:.2f}")
        report.append(f"- Games > 5% error: {over_5} of {len(df)}\n")

        # Section 3: Table of predictions
        report.append("## 🎯 Predicted vs Actual Prices\n")
        report.append("| Game | Predicted | Actual | Abs Error | % Error |")
        report.append("|------|-----------|--------|-----------|---------|")
        for _, row in df.iterrows():
            game = f"{row['homeTeam']} vs {row['awayTeam']}"
            p = row["predicted_lowest_price"]
            a = row["actual_lowest_price"]
            ae = row["abs_error"]
            pe = row["percent_error"] * 100
            report.append(f"| {game} | ${p:.2f} | ${a:.2f} | ${ae:.2f} | {pe:.1f}% |")

        # Section 4: Heuristic suggestions
        report.append("\n## 💡 Suggestions")
        if over_5 / len(df) > 0.4:
            report.append("- Consider revisiting variables for highly ranked vs unranked matchups.")
        if df["homeTeamRank"].isna().any() or df["awayTeamRank"].isna().any():
            report.append("- Some rankings were missing; double check post-bowl data availability.")
        report.append("- Variables to explore: team momentum, previous week performance, weather.")

    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"✅ Weekly report saved to {report_path}")
    return report_path

if __name__ == "__main__":
    build_report()
