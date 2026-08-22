# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-01

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-01 06:48:09**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1935**
- gap_pct within 0.05: **0.1419**
- Price MAE: **$11.22**
- Price RMSE: **$22.98**
- Price within 5%: **0.0849**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-29 05:30:18_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-01 05:02:21_) — Snapshot appended (3192 new rows). Total now: 163914
- **model_train**: **success** (_2026-05-01 06:48:09_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-30 08:15:07_) — Weekly report generated: reports\weekly\2026-04-30\weekly_report_2026-04-30.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-01 05:02:21**
- Model last trained: **2026-05-01 06:48:09**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 17.8925 (~17.9%)
- capacity: 12.2214 (~12.2%)
- hours_until_game: 11.0252 (~11.0%)
- kickoff_hour: 8.4963 (~8.5%)
- awayTeam: 7.9667 (~8.0%)
- home_last_point_diff_at_snapshot: 7.8793 (~7.9%)
- week: 7.5660 (~7.6%)
- away_last_point_diff_at_snapshot: 6.7058 (~6.7%)
- homeConference: 5.0156 (~5.0%)
- home_losses_at_snapshot: 4.9236 (~4.9%)
- away_wins_at_snapshot: 2.9803 (~3.0%)
- kickoff_dayofweek: 2.1561 (~2.2%)
- home_wins_at_snapshot: 2.0221 (~2.0%)
- isRivalry: 0.8230 (~0.8%)
- away_losses_at_snapshot_missing: 0.8226 (~0.8%)
- homeTeamRank_missing: 0.5395 (~0.5%)
- homeTeamRank: 0.4651 (~0.5%)
- away_last_point_diff_at_snapshot_missing: 0.1685 (~0.2%)
- home_wins_at_snapshot_missing: 0.1222 (~0.1%)
- neutralSite: 0.1182 (~0.1%)
- away_wins_at_snapshot_missing: 0.0900 (~0.1%)
- season_year: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, capacity_missing, season_year_missing, hours_until_game_missing, isRankedMatchup, home_losses_at_snapshot_missing, week_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
