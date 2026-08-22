# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-13

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-13 06:49:27**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1977**
- gap_pct within 0.05: **0.1331**
- Price MAE: **$11.22**
- Price RMSE: **$22.61**
- Price within 5%: **0.0717**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-08 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-13 05:59:07_) — Snapshot appended (3184 new rows). Total now: 100902
- **model_train**: **success** (_2026-04-13 06:49:27_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-12 08:15:14_) — Weekly report generated: reports\weekly\2026-04-12\weekly_report_2026-04-12.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-13 05:59:07**
- Model last trained: **2026-04-13 06:49:27**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 23.1758 (~23.2%)
- awayTeam: 12.3748 (~12.4%)
- capacity: 10.0423 (~10.0%)
- kickoff_hour: 7.9806 (~8.0%)
- awayConference: 7.8065 (~7.8%)
- hours_until_game: 7.6062 (~7.6%)
- home_last_point_diff_at_snapshot: 7.5190 (~7.5%)
- week: 6.0177 (~6.0%)
- homeConference: 5.7728 (~5.8%)
- away_last_point_diff_at_snapshot: 4.1742 (~4.2%)
- home_losses_at_snapshot: 2.3602 (~2.4%)
- away_wins_at_snapshot: 1.4727 (~1.5%)
- away_losses_at_snapshot: 1.1111 (~1.1%)
- homeTeamRank: 0.8067 (~0.8%)
- home_last_point_diff_at_snapshot_missing: 0.3575 (~0.4%)
- away_losses_at_snapshot_missing: 0.3426 (~0.3%)
- neutralSite: 0.3330 (~0.3%)
- away_last_point_diff_at_snapshot_missing: 0.2097 (~0.2%)
- home_losses_at_snapshot_missing: 0.1892 (~0.2%)
- home_wins_at_snapshot_missing: 0.1605 (~0.2%)
- homeTeamRank_missing: 0.0944 (~0.1%)
- away_wins_at_snapshot_missing: 0.0926 (~0.1%)
- season_year: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRivalry, isRankedMatchup, hours_until_game_missing, week_missing, capacity_missing, season_year_missing, kickoff_hour_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
