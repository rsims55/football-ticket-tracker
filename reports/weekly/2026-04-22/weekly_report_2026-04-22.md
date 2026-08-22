# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-22

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-22 06:48:05**
- Features in model: **35**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1938**
- gap_pct within 0.05: **0.1460**
- Price MAE: **$11.17**
- Price RMSE: **$22.54**
- Price within 5%: **0.0789**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-22 04:37:59_) — Snapshot appended (3184 new rows). Total now: 132366
- **model_train**: **success** (_2026-04-22 06:48:05_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-21 08:15:11_) — Weekly report generated: reports\weekly\2026-04-21\weekly_report_2026-04-21.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-22 04:37:59**
- Model last trained: **2026-04-22 06:48:05**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 19.5619 (~19.6%)
- hours_until_game: 11.5766 (~11.6%)
- home_last_point_diff_at_snapshot: 8.9041 (~8.9%)
- capacity: 8.8085 (~8.8%)
- kickoff_hour: 8.1437 (~8.1%)
- week: 7.9695 (~8.0%)
- away_last_point_diff_at_snapshot: 7.7778 (~7.8%)
- awayTeam: 7.0875 (~7.1%)
- awayConference: 5.6508 (~5.7%)
- home_losses_at_snapshot: 5.4959 (~5.5%)
- away_losses_at_snapshot: 2.7009 (~2.7%)
- homeConference: 2.5998 (~2.6%)
- home_wins_at_snapshot: 2.0587 (~2.1%)
- away_wins_at_snapshot: 1.1132 (~1.1%)
- homeTeamRank: 0.2332 (~0.2%)
- neutralSite: 0.1466 (~0.1%)
- away_last_point_diff_at_snapshot_missing: 0.0877 (~0.1%)
- home_wins_at_snapshot_missing: 0.0835 (~0.1%)
- season_year: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_losses_at_snapshot_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, week_missing, capacity_missing, season_year_missing, hours_until_game_missing, isRankedMatchup, isRivalry, home_last_point_diff_at_snapshot_missing, kickoff_hour_missing, neutralSite_missing, away_losses_at_snapshot_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
