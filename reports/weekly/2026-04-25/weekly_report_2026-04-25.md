# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-25

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-25 06:47:49**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1925**
- gap_pct within 0.05: **0.1391**
- Price MAE: **$11.17**
- Price RMSE: **$22.92**
- Price within 5%: **0.0785**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-25 04:07:27_) — Snapshot appended (3169 new rows). Total now: 142881
- **model_train**: **success** (_2026-04-25 06:47:49_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-24 08:15:09_) — Weekly report generated: reports\weekly\2026-04-24\weekly_report_2026-04-24.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-25 04:07:27**
- Model last trained: **2026-04-25 06:47:49**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.0526 (~18.1%)
- hours_until_game: 12.5520 (~12.6%)
- capacity: 11.9040 (~11.9%)
- week: 8.1976 (~8.2%)
- kickoff_hour: 8.0359 (~8.0%)
- home_last_point_diff_at_snapshot: 7.5740 (~7.6%)
- awayTeam: 7.1674 (~7.2%)
- away_last_point_diff_at_snapshot: 6.3996 (~6.4%)
- home_losses_at_snapshot: 5.4266 (~5.4%)
- homeConference: 4.1648 (~4.2%)
- away_wins_at_snapshot: 3.0278 (~3.0%)
- kickoff_dayofweek: 2.4476 (~2.4%)
- home_wins_at_snapshot: 1.8826 (~1.9%)
- isRivalry: 0.9857 (~1.0%)
- away_losses_at_snapshot_missing: 0.8153 (~0.8%)
- homeTeamRank_missing: 0.6461 (~0.6%)
- homeTeamRank: 0.2945 (~0.3%)
- away_last_point_diff_at_snapshot_missing: 0.2018 (~0.2%)
- home_wins_at_snapshot_missing: 0.1464 (~0.1%)
- neutralSite: 0.0779 (~0.1%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, capacity_missing, season_year_missing, hours_until_game_missing, week_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
