# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-10

## 🧠 Latest CatBoost Training Summary

- Rows evaluated: **107118**
- gap_pct MAE: **0.1941**
- gap_pct within 0.05: **0.1476**
- Price MAE: **$10.24**
- Price RMSE: **$17.57**
- Price within 5%: **0.0969**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-08 16:02:03_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-10 03:02:34_) — Snapshot appended (2920 new rows). Total now: 9390
- **model_train**: **success** (_2026-03-06 21:29:55_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-09 08:15:08_) — Weekly report generated: reports\weekly\2026-03-09\weekly_report_2026-03-09.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-10 03:02:34**
- Predictions last updated: **2026-02-01 19:21:39**
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 24.0397 (~24.0%)
- hours_until_game: 11.8798 (~11.9%)
- awayTeam: 11.2961 (~11.3%)
- capacity: 10.5614 (~10.6%)
- week: 9.4194 (~9.4%)
- kickoff_hour: 9.0927 (~9.1%)
- away_last_point_diff_at_snapshot: 8.5624 (~8.6%)
- home_last_point_diff_at_snapshot: 8.4914 (~8.5%)
- homeConference: 6.0889 (~6.1%)
- homeTeamRank: 0.5683 (~0.6%)
- season_year: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
