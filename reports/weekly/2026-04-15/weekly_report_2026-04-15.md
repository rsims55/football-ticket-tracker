# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-15

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-15 06:48:36**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1937**
- gap_pct within 0.05: **0.1485**
- Price MAE: **$11.07**
- Price RMSE: **$22.12**
- Price within 5%: **0.0868**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-15 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-15 01:31:32_) — Snapshot appended (3184 new rows). Total now: 107894
- **model_train**: **success** (_2026-04-15 06:48:36_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-14 08:15:10_) — Weekly report generated: reports\weekly\2026-04-14\weekly_report_2026-04-14.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-15 01:31:32**
- Model last trained: **2026-04-15 06:48:36**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.8601 (~22.9%)
- capacity: 10.4272 (~10.4%)
- hours_until_game: 10.0924 (~10.1%)
- awayTeam: 9.0846 (~9.1%)
- home_last_point_diff_at_snapshot: 8.2441 (~8.2%)
- kickoff_hour: 8.0308 (~8.0%)
- week: 7.0568 (~7.1%)
- awayConference: 6.1179 (~6.1%)
- away_last_point_diff_at_snapshot: 4.9201 (~4.9%)
- homeConference: 4.0043 (~4.0%)
- home_losses_at_snapshot: 3.0750 (~3.1%)
- away_wins_at_snapshot: 2.2255 (~2.2%)
- away_losses_at_snapshot: 1.5644 (~1.6%)
- homeTeamRank: 0.6756 (~0.7%)
- away_losses_at_snapshot_missing: 0.4296 (~0.4%)
- home_wins_at_snapshot_missing: 0.2538 (~0.3%)
- home_losses_at_snapshot_missing: 0.2511 (~0.3%)
- away_last_point_diff_at_snapshot_missing: 0.2510 (~0.3%)
- neutralSite: 0.2113 (~0.2%)
- home_last_point_diff_at_snapshot_missing: 0.1122 (~0.1%)
- away_wins_at_snapshot_missing: 0.1121 (~0.1%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, isRivalry, season_year_missing, hours_until_game_missing, week_missing, capacity_missing, neutralSite_missing, kickoff_hour_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
