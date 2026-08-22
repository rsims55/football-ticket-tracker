# 📈 Weekly Ticket Price Model Report
**Date:** 2026-08-11

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-08-11 18:43:06**
- Features in model: **33**
- Rows evaluated: **545416**
- gap_pct MAE: **0.1967**
- gap_pct within 0.05: **0.1214**
- Price MAE: **$15.15**
- Price RMSE: **$26.04**
- Price within 5%: **0.0792**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-08-11 16:37:51_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-08-11 18:33:05_) — Snapshot appended (3184 new rows). Total now: 461396
- **model_train**: **success** (_2026-08-11 18:43:06_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-08-10 08:15:08_) — Weekly report generated: reports\weekly\2026-08-10\weekly_report_2026-08-10.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **In-season**
- Snapshots last updated: **2026-08-11 18:33:05**
- Model last trained: **2026-08-11 18:43:06**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 33.6730 (~33.7%)
- homeTeam: 20.8220 (~20.8%)
- awayTeam: 14.7323 (~14.7%)
- week: 7.9481 (~7.9%)
- home_last_point_diff_at_snapshot: 4.9994 (~5.0%)
- capacity: 4.6166 (~4.6%)
- home_losses_at_snapshot: 3.0056 (~3.0%)
- kickoff_hour: 2.9943 (~3.0%)
- away_last_point_diff_at_snapshot: 1.6164 (~1.6%)
- awayConference: 1.4120 (~1.4%)
- homeConference: 1.1138 (~1.1%)
- season_year: 0.9268 (~0.9%)
- away_losses_at_snapshot: 0.9007 (~0.9%)
- away_wins_at_snapshot: 0.7742 (~0.8%)
- home_wins_at_snapshot: 0.2120 (~0.2%)
- homeTeamRank: 0.1684 (~0.2%)
- away_wins_at_snapshot_missing: 0.0844 (~0.1%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- away_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year_missing, hours_until_game_missing, isRankedMatchup, isRivalry, neutralSite, capacity_missing, kickoff_hour_missing, week_missing, away_losses_at_snapshot_missing, home_losses_at_snapshot_missing, home_wins_at_snapshot_missing, home_last_point_diff_at_snapshot_missing, kickoff_dayofweek_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
