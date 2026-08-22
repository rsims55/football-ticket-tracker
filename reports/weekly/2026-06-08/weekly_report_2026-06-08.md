# 📈 Weekly Ticket Price Model Report
**Date:** 2026-06-08

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-06-08 06:50:07**
- Features in model: **33**
- Rows evaluated: **342765**
- gap_pct MAE: **0.2144**
- gap_pct within 0.05: **0.1084**
- Price MAE: **$18.64**
- Price RMSE: **$29.64**
- Price within 5%: **0.0787**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-06-04 09:12:43_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-06-08 02:39:01_) — Snapshot appended (3179 new rows). Total now: 257121
- **model_train**: **success** (_2026-06-08 06:50:07_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-06-04 10:52:11_) — Weekly report generated: reports\weekly\2026-06-04\weekly_report_2026-06-04.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-06-08 02:39:01**
- Model last trained: **2026-06-08 06:50:07**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 20.8530 (~20.9%)
- awayTeam: 17.7358 (~17.7%)
- hours_until_game: 12.1272 (~12.1%)
- capacity: 9.4904 (~9.5%)
- week: 9.1081 (~9.1%)
- home_last_point_diff_at_snapshot: 6.5346 (~6.5%)
- kickoff_hour: 5.9134 (~5.9%)
- homeConference: 5.6373 (~5.6%)
- home_losses_at_snapshot: 2.2845 (~2.3%)
- homeTeamRank: 2.1940 (~2.2%)
- kickoff_dayofweek: 1.7733 (~1.8%)
- away_losses_at_snapshot: 1.7150 (~1.7%)
- away_wins_at_snapshot: 1.4363 (~1.4%)
- season_year: 1.3981 (~1.4%)
- neutralSite: 0.4702 (~0.5%)
- isRivalry: 0.4126 (~0.4%)
- away_losses_at_snapshot_missing: 0.2793 (~0.3%)
- away_last_point_diff_at_snapshot_missing: 0.2718 (~0.3%)
- home_losses_at_snapshot_missing: 0.1607 (~0.2%)
- away_wins_at_snapshot_missing: 0.1437 (~0.1%)
- home_last_point_diff_at_snapshot_missing: 0.0325 (~0.0%)
- home_wins_at_snapshot_missing: 0.0283 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year_missing, hours_until_game_missing, isRankedMatchup, capacity_missing, week_missing, kickoff_hour_missing, neutralSite_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
