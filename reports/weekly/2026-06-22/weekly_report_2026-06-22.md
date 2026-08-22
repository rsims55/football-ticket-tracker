# 📈 Weekly Ticket Price Model Report
**Date:** 2026-06-22

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-06-22 06:50:14**
- Features in model: **34**
- Rows evaluated: **391211**
- gap_pct MAE: **0.1953**
- gap_pct within 0.05: **0.1131**
- Price MAE: **$15.51**
- Price RMSE: **$23.42**
- Price within 5%: **0.0826**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-06-17 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-06-22 02:47:15_) — Snapshot appended (3176 new rows). Total now: 306185
- **model_train**: **success** (_2026-06-22 06:50:14_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-06-15 08:15:07_) — Weekly report generated: reports\weekly\2026-06-15\weekly_report_2026-06-15.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-06-22 02:47:15**
- Model last trained: **2026-06-22 06:50:14**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 29.1743 (~29.2%)
- homeTeam: 18.7023 (~18.7%)
- awayTeam: 13.8668 (~13.9%)
- week: 7.9129 (~7.9%)
- home_last_point_diff_at_snapshot: 6.9896 (~7.0%)
- capacity: 6.2123 (~6.2%)
- home_losses_at_snapshot: 2.5573 (~2.6%)
- away_last_point_diff_at_snapshot: 2.2017 (~2.2%)
- kickoff_hour: 2.0809 (~2.1%)
- awayConference: 1.8609 (~1.9%)
- homeTeamRank: 1.4394 (~1.4%)
- kickoff_dayofweek: 1.3622 (~1.4%)
- season_year: 1.3529 (~1.4%)
- home_wins_at_snapshot: 1.2427 (~1.2%)
- homeConference: 1.0861 (~1.1%)
- home_last_point_diff_at_snapshot_missing: 0.7823 (~0.8%)
- homeTeamRank_missing: 0.5227 (~0.5%)
- home_losses_at_snapshot_missing: 0.2377 (~0.2%)
- away_wins_at_snapshot: 0.1644 (~0.2%)
- home_wins_at_snapshot_missing: 0.1456 (~0.1%)
- away_losses_at_snapshot_missing: 0.1049 (~0.1%)
- season_year_missing: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year_missing, neutralSite, isRivalry, hours_until_game_missing, isRankedMatchup, capacity_missing, away_wins_at_snapshot_missing, kickoff_hour_missing, week_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
