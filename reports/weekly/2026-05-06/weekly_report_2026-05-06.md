# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-06

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-06 06:48:58**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1976**
- gap_pct within 0.05: **0.1495**
- Price MAE: **$11.03**
- Price RMSE: **$22.20**
- Price within 5%: **0.0854**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-05-06 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-06 06:31:48_) — Snapshot appended (3192 new rows). Total now: 181383
- **model_train**: **success** (_2026-05-06 06:48:58_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-05 08:15:07_) — Weekly report generated: reports\weekly\2026-05-05\weekly_report_2026-05-05.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-06 06:31:48**
- Model last trained: **2026-05-06 06:48:58**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 20.7797 (~20.8%)
- awayTeam: 14.7455 (~14.7%)
- capacity: 11.6585 (~11.7%)
- kickoff_hour: 8.4644 (~8.5%)
- home_last_point_diff_at_snapshot: 7.2679 (~7.3%)
- hours_until_game: 7.1642 (~7.2%)
- homeConference: 6.5360 (~6.5%)
- week: 5.5467 (~5.5%)
- away_last_point_diff_at_snapshot: 4.7582 (~4.8%)
- home_losses_at_snapshot: 3.0143 (~3.0%)
- kickoff_dayofweek: 2.1915 (~2.2%)
- away_wins_at_snapshot: 2.0665 (~2.1%)
- home_wins_at_snapshot: 1.7026 (~1.7%)
- homeTeamRank: 1.2775 (~1.3%)
- away_losses_at_snapshot_missing: 0.7526 (~0.8%)
- isRivalry: 0.4801 (~0.5%)
- homeTeamRank_missing: 0.4422 (~0.4%)
- home_last_point_diff_at_snapshot_missing: 0.3571 (~0.4%)
- neutralSite: 0.3374 (~0.3%)
- away_wins_at_snapshot_missing: 0.2252 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1572 (~0.2%)
- home_wins_at_snapshot_missing: 0.0713 (~0.1%)
- home_losses_at_snapshot_missing: 0.0033 (~0.0%)
- season_year: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, hours_until_game_missing, isRankedMatchup, capacity_missing, season_year_missing, kickoff_hour_missing, week_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
