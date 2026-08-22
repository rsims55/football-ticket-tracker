# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-07

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-07 06:48:05**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1961**
- gap_pct within 0.05: **0.1474**
- Price MAE: **$11.18**
- Price RMSE: **$22.06**
- Price within 5%: **0.0858**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-01 05:30:13_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-07 01:46:57_) — Snapshot appended (3184 new rows). Total now: 80521
- **model_train**: **success** (_2026-04-07 06:48:05_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-06 08:15:08_) — Weekly report generated: reports\weekly\2026-04-06\weekly_report_2026-04-06.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-07 01:46:57**
- Model last trained: **2026-04-07 06:48:05**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 21.5162 (~21.5%)
- hours_until_game: 11.5183 (~11.5%)
- capacity: 10.6057 (~10.6%)
- kickoff_hour: 8.6134 (~8.6%)
- home_last_point_diff_at_snapshot: 8.1287 (~8.1%)
- awayTeam: 7.5904 (~7.6%)
- week: 7.5828 (~7.6%)
- away_last_point_diff_at_snapshot: 6.0301 (~6.0%)
- awayConference: 5.4763 (~5.5%)
- home_losses_at_snapshot: 3.6074 (~3.6%)
- homeConference: 2.9722 (~3.0%)
- away_wins_at_snapshot: 2.2845 (~2.3%)
- away_losses_at_snapshot: 2.0533 (~2.1%)
- away_losses_at_snapshot_missing: 0.6030 (~0.6%)
- homeTeamRank: 0.5301 (~0.5%)
- away_last_point_diff_at_snapshot_missing: 0.2801 (~0.3%)
- home_wins_at_snapshot_missing: 0.2730 (~0.3%)
- home_losses_at_snapshot_missing: 0.2305 (~0.2%)
- neutralSite: 0.1042 (~0.1%)
- season_year: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, week_missing, capacity_missing, isRankedMatchup, isRivalry, season_year_missing, hours_until_game_missing, neutralSite_missing, away_wins_at_snapshot_missing, home_last_point_diff_at_snapshot_missing, kickoff_hour_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
