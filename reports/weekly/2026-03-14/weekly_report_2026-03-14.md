# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-14

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-14 06:47:20**
- Features in model: **16**
- Rows evaluated: **107118**
- gap_pct MAE: **0.1945**
- gap_pct within 0.05: **0.1511**
- Price MAE: **$10.73**
- Price RMSE: **$19.46**
- Price within 5%: **0.0964**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-13 21:38:42_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-14 03:36:12_) — Snapshot appended (3010 new rows). Total now: 21987
- **model_train**: **success** (_2026-03-14 06:47:20_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-13 22:55:33_) — Weekly report generated: reports\weekly\2026-03-13\weekly_report_2026-03-13.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-14 03:36:12**
- Model last trained: **2026-03-14 06:47:20**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 19.6026 (~19.6%)
- hours_until_game: 11.3054 (~11.3%)
- capacity: 10.4762 (~10.5%)
- home_last_point_diff_at_snapshot: 8.4818 (~8.5%)
- awayTeam: 8.3966 (~8.4%)
- week: 8.0184 (~8.0%)
- away_last_point_diff_at_snapshot: 6.6832 (~6.7%)
- kickoff_hour: 6.4135 (~6.4%)
- awayConference: 5.6973 (~5.7%)
- home_losses_at_snapshot: 4.8782 (~4.9%)
- homeConference: 3.5008 (~3.5%)
- away_losses_at_snapshot: 3.0708 (~3.1%)
- kickoff_dayofweek: 2.2094 (~2.2%)
- away_wins_at_snapshot: 0.7329 (~0.7%)
- homeTeamRank: 0.5331 (~0.5%)
- season_year: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
