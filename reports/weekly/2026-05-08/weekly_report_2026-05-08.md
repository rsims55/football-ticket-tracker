# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-08

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-08 06:48:23**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1938**
- gap_pct within 0.05: **0.1396**
- Price MAE: **$11.39**
- Price RMSE: **$23.61**
- Price within 5%: **0.0789**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-05-06 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-08 07:22:19_) — Snapshot appended (3192 new rows). Total now: 188404
- **model_train**: **success** (_2026-05-08 06:48:23_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-07 08:15:09_) — Weekly report generated: reports\weekly\2026-05-07\weekly_report_2026-05-07.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-08 07:22:19**
- Model last trained: **2026-05-08 06:48:23**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.0746 (~18.1%)
- capacity: 12.2391 (~12.2%)
- hours_until_game: 10.9907 (~11.0%)
- awayTeam: 8.2871 (~8.3%)
- kickoff_hour: 8.2164 (~8.2%)
- home_last_point_diff_at_snapshot: 7.6921 (~7.7%)
- week: 7.2846 (~7.3%)
- away_last_point_diff_at_snapshot: 6.9092 (~6.9%)
- homeConference: 5.2565 (~5.3%)
- home_losses_at_snapshot: 4.7603 (~4.8%)
- away_wins_at_snapshot: 2.9099 (~2.9%)
- kickoff_dayofweek: 2.1308 (~2.1%)
- home_wins_at_snapshot: 1.8846 (~1.9%)
- isRivalry: 0.8151 (~0.8%)
- away_losses_at_snapshot_missing: 0.8135 (~0.8%)
- homeTeamRank_missing: 0.6044 (~0.6%)
- homeTeamRank: 0.5254 (~0.5%)
- neutralSite: 0.1801 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1668 (~0.2%)
- away_wins_at_snapshot_missing: 0.1376 (~0.1%)
- home_wins_at_snapshot_missing: 0.1211 (~0.1%)
- season_year: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, capacity_missing, season_year_missing, hours_until_game_missing, isRankedMatchup, home_losses_at_snapshot_missing, week_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
