# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-11

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-11 16:23:11**
- Features in model: **17**
- Rows evaluated: **283593**
- gap_pct MAE: **0.2051**
- gap_pct within 0.05: **0.0841**
- Price MAE: **$18.24**
- Price RMSE: **$26.38**
- Price within 5%: **0.0610**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-05-11 13:44:20_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-11 16:18:51_) — Snapshot appended (3158 new rows). Total now: 196641
- **model_train**: **success** (_2026-05-11 16:23:11_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-11 08:15:06_) — Weekly report generated: reports\weekly\2026-05-11\weekly_report_2026-05-11.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-11 16:18:51**
- Model last trained: **2026-05-11 16:23:11**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 21.6206 (~21.6%)
- homeTeam: 18.9040 (~18.9%)
- awayTeam: 13.3065 (~13.3%)
- week: 8.8218 (~8.8%)
- capacity: 7.1325 (~7.1%)
- home_last_point_diff_at_snapshot: 5.8199 (~5.8%)
- home_losses_at_snapshot: 4.0975 (~4.1%)
- kickoff_hour: 4.0869 (~4.1%)
- kickoff_dayofweek: 2.3487 (~2.3%)
- homeConference: 2.2318 (~2.2%)
- away_wins_at_snapshot: 2.0233 (~2.0%)
- awayConference: 1.9823 (~2.0%)
- season_year: 1.8869 (~1.9%)
- homeTeamRank: 1.6667 (~1.7%)
- home_wins_at_snapshot: 1.5212 (~1.5%)
- away_losses_at_snapshot: 1.4147 (~1.4%)
- away_last_point_diff_at_snapshot: 1.1347 (~1.1%)


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
