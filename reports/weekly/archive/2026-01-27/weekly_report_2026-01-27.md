# 📈 Weekly Ticket Price Model Report
**Date:** 2026-01-27

## 🔍 Best Predictors of Ticket Price

### Top Features
- homeTeam contributed ~25.7% of total importance.
- week contributed ~19.4% of total importance.
- capacity contributed ~10.5% of total importance.
- kickoff_hour contributed ~9.0% of total importance.
- hours_until_game contributed ~8.2% of total importance.
- homeConference contributed ~8.1% of total importance.
- home_last_point_diff_at_snapshot contributed ~6.3% of total importance.
- homeTeamRank_missing contributed ~5.2% of total importance.
- away_last_point_diff_at_snapshot contributed ~3.8% of total importance.
- awayConference contributed ~2.4% of total importance.
- homeTeamRank contributed ~1.4% of total importance.
- week_missing contributed ~0.0% of total importance.

**Possibly unrelated (near-zero importance):** week_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, away_last_point_diff_at_snapshot_missing, hours_until_game_missing, capacity_missing

## 📊 Model Accuracy (Latest CatBoost Training)

- Rows evaluated: **109372**
- gap_pct MAE: **0.2128**
- gap_pct RMSE: **0.2523**
- gap_pct within 0.05: **0.1375**
- Price MAE: **$21.45**
- Price RMSE: **$69.10**
- Price within 5%: **0.0941**

### Price MAE by Bucket
- $0–20: **$6.97**
- $20–50: **$11.17**
- $50–100: **$13.88**
- $100–200: **$34.24**
- $200–inf: **$161.47**

### ⏱️ Timing Accuracy
- MAE (hours): **385.48 h**
- Within 24h: **0.1083**

## 🧾 Notes
- This report summarizes the latest CatBoost model training metrics and feature importances.
- Detailed predicted-vs-actual tables are not included here (requires evaluation runs against held-out weeks).