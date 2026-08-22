# 📈 Weekly Ticket Price Model Report
**Date:** 2026-08-10

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-08-10 06:53:46**
- Features in model: **31**
- Rows evaluated: **541053**
- gap_pct MAE: **0.1873**
- gap_pct within 0.05: **0.1266**
- Price MAE: **$14.18**
- Price RMSE: **$23.44**
- Price within 5%: **0.0771**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-08-05 05:30:14_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-08-10 03:54:15_) — Snapshot appended (3160 new rows). Total now: 457013
- **model_train**: **success** (_2026-08-10 06:53:46_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-08-03 08:15:07_) — Weekly report generated: reports\weekly\2026-08-03\weekly_report_2026-08-03.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **In-season**
- Snapshots last updated: **2026-08-10 03:54:15**
- Model last trained: **2026-08-10 06:53:46**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 34.5841 (~34.6%)
- homeTeam: 21.0130 (~21.0%)
- awayTeam: 15.2370 (~15.2%)
- week: 8.1126 (~8.1%)
- capacity: 4.4108 (~4.4%)
- home_last_point_diff_at_snapshot: 3.4730 (~3.5%)
- kickoff_hour: 2.6323 (~2.6%)
- homeConference: 2.5975 (~2.6%)
- season_year: 2.4255 (~2.4%)
- home_losses_at_snapshot: 2.1228 (~2.1%)
- away_wins_at_snapshot: 1.1621 (~1.2%)
- home_wins_at_snapshot: 0.9497 (~0.9%)
- away_losses_at_snapshot: 0.6894 (~0.7%)
- homeTeamRank: 0.4761 (~0.5%)
- home_wins_at_snapshot_missing: 0.1141 (~0.1%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- away_losses_at_snapshot_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** hours_until_game_missing, isRankedMatchup, isRivalry, neutralSite, kickoff_hour_missing, season_year_missing, week_missing, capacity_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, home_last_point_diff_at_snapshot_missing, away_losses_at_snapshot_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
