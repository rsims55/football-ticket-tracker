# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-18

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-18 06:48:42**
- Features in model: **14**
- Rows evaluated: **304710**
- gap_pct MAE: **0.2051**
- gap_pct within 0.05: **0.0873**
- Price MAE: **$17.96**
- Price RMSE: **$26.62**
- Price within 5%: **0.0616**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-05-13 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-18 05:58:49_) — Snapshot appended (3192 new rows). Total now: 218357
- **model_train**: **success** (_2026-05-18 06:48:42_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-11 16:23:22_) — Weekly report generated: reports\weekly\2026-05-11\weekly_report_2026-05-11.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-18 05:58:49**
- Model last trained: **2026-05-18 06:48:42**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 20.8041 (~20.8%)
- homeTeam: 19.0767 (~19.1%)
- awayTeam: 13.8183 (~13.8%)
- capacity: 9.4898 (~9.5%)
- week: 9.4520 (~9.5%)
- home_last_point_diff_at_snapshot: 7.1435 (~7.1%)
- home_losses_at_snapshot: 4.1986 (~4.2%)
- away_last_point_diff_at_snapshot: 3.6271 (~3.6%)
- home_wins_at_snapshot: 3.1082 (~3.1%)
- kickoff_hour: 2.8656 (~2.9%)
- homeConference: 2.7568 (~2.8%)
- kickoff_dayofweek: 2.1571 (~2.2%)
- season_year: 0.8024 (~0.8%)
- homeTeamRank: 0.6998 (~0.7%)


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
