# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-02

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-02 06:48:09**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1937**
- gap_pct within 0.05: **0.1355**
- Price MAE: **$11.35**
- Price RMSE: **$23.44**
- Price within 5%: **0.0780**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-29 05:30:18_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-02 03:38:10_) — Snapshot appended (3181 new rows). Total now: 167426
- **model_train**: **success** (_2026-05-02 06:48:09_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-01 08:15:06_) — Weekly report generated: reports\weekly\2026-05-01\weekly_report_2026-05-01.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-02 03:38:10**
- Model last trained: **2026-05-02 06:48:09**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.1433 (~18.1%)
- capacity: 12.1957 (~12.2%)
- hours_until_game: 10.9266 (~10.9%)
- awayTeam: 8.2483 (~8.2%)
- kickoff_hour: 8.1934 (~8.2%)
- home_last_point_diff_at_snapshot: 7.6435 (~7.6%)
- week: 7.3470 (~7.3%)
- away_last_point_diff_at_snapshot: 6.9306 (~6.9%)
- homeConference: 5.2488 (~5.2%)
- home_losses_at_snapshot: 4.8378 (~4.8%)
- away_wins_at_snapshot: 2.9188 (~2.9%)
- kickoff_dayofweek: 2.1015 (~2.1%)
- home_wins_at_snapshot: 1.8903 (~1.9%)
- isRivalry: 0.8176 (~0.8%)
- away_losses_at_snapshot_missing: 0.8160 (~0.8%)
- homeTeamRank_missing: 0.6062 (~0.6%)
- homeTeamRank: 0.5270 (~0.5%)
- neutralSite: 0.1807 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1674 (~0.2%)
- away_wins_at_snapshot_missing: 0.1380 (~0.1%)
- home_wins_at_snapshot_missing: 0.1214 (~0.1%)
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
