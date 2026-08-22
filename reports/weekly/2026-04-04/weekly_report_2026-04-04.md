# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-04

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-04 06:48:45**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1965**
- gap_pct within 0.05: **0.1554**
- Price MAE: **$11.46**
- Price RMSE: **$24.39**
- Price within 5%: **0.0941**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-01 05:30:13_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-04 01:56:20_) — Snapshot appended (3184 new rows). Total now: 70033
- **model_train**: **success** (_2026-04-04 06:48:45_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-03 08:15:08_) — Weekly report generated: reports\weekly\2026-04-03\weekly_report_2026-04-03.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-04 01:56:20**
- Model last trained: **2026-04-04 06:48:45**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.9721 (~23.0%)
- awayTeam: 10.9516 (~11.0%)
- capacity: 9.7660 (~9.8%)
- hours_until_game: 8.7515 (~8.8%)
- kickoff_hour: 8.0759 (~8.1%)
- home_last_point_diff_at_snapshot: 7.4271 (~7.4%)
- awayConference: 6.9614 (~7.0%)
- week: 6.0843 (~6.1%)
- homeConference: 5.0790 (~5.1%)
- away_last_point_diff_at_snapshot: 4.9532 (~5.0%)
- home_losses_at_snapshot: 2.7004 (~2.7%)
- away_wins_at_snapshot: 1.9148 (~1.9%)
- away_losses_at_snapshot: 1.7365 (~1.7%)
- homeTeamRank: 0.9786 (~1.0%)
- away_losses_at_snapshot_missing: 0.4562 (~0.5%)
- neutralSite: 0.3750 (~0.4%)
- home_losses_at_snapshot_missing: 0.2295 (~0.2%)
- home_wins_at_snapshot_missing: 0.2088 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1972 (~0.2%)
- home_last_point_diff_at_snapshot_missing: 0.1126 (~0.1%)
- away_wins_at_snapshot_missing: 0.0481 (~0.0%)
- homeTeamRank_missing: 0.0201 (~0.0%)
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
