# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-13

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-13 22:55:18**
- Features in model: **35**
- Rows evaluated: **107118**
- gap_pct MAE: **0.2032**
- gap_pct within 0.05: **0.1405**
- Price MAE: **$11.47**
- Price RMSE: **$21.80**
- Price within 5%: **0.0810**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-13 21:38:42_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-13 22:52:54_) — Snapshot appended (3010 new rows). Total now: 21210
- **model_train**: **success** (_2026-03-13 22:55:19_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-13 08:15:07_) — Weekly report generated: reports\weekly\2026-03-13\weekly_report_2026-03-13.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-13 22:52:54**
- Model last trained: **2026-03-13 22:55:18**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 20.7576 (~20.8%)
- awayTeam: 12.6044 (~12.6%)
- capacity: 11.1922 (~11.2%)
- hours_until_game: 8.8766 (~8.9%)
- kickoff_hour: 7.8894 (~7.9%)
- home_last_point_diff_at_snapshot: 7.5208 (~7.5%)
- week: 6.2598 (~6.3%)
- homeConference: 6.0951 (~6.1%)
- away_last_point_diff_at_snapshot: 5.6206 (~5.6%)
- home_losses_at_snapshot: 3.6564 (~3.7%)
- kickoff_dayofweek: 2.1530 (~2.2%)
- away_wins_at_snapshot: 2.0786 (~2.1%)
- away_losses_at_snapshot: 1.7417 (~1.7%)
- home_wins_at_snapshot: 1.4479 (~1.4%)
- homeTeamRank: 1.0342 (~1.0%)
- neutralSite: 0.2684 (~0.3%)
- away_losses_at_snapshot_missing: 0.2350 (~0.2%)
- isRivalry: 0.1486 (~0.1%)
- home_last_point_diff_at_snapshot_missing: 0.1325 (~0.1%)
- away_wins_at_snapshot_missing: 0.1249 (~0.1%)
- homeTeamRank_missing: 0.1047 (~0.1%)
- away_last_point_diff_at_snapshot_missing: 0.0577 (~0.1%)
- season_year: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, capacity_missing, season_year_missing, hours_until_game_missing, isRankedMatchup, home_wins_at_snapshot_missing, home_losses_at_snapshot_missing, kickoff_hour_missing, week_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
