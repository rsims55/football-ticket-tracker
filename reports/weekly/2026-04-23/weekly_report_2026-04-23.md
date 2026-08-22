# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-23

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-23 06:48:17**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1863**
- gap_pct within 0.05: **0.1571**
- Price MAE: **$10.84**
- Price RMSE: **$21.78**
- Price within 5%: **0.0866**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-23 02:35:11_) — Snapshot appended (3180 new rows). Total now: 135870
- **model_train**: **success** (_2026-04-23 06:48:17_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-22 08:15:10_) — Weekly report generated: reports\weekly\2026-04-22\weekly_report_2026-04-22.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-23 02:35:11**
- Model last trained: **2026-04-23 06:48:17**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 20.1078 (~20.1%)
- hours_until_game: 10.4325 (~10.4%)
- capacity: 10.2272 (~10.2%)
- awayTeam: 9.7440 (~9.7%)
- week: 8.5084 (~8.5%)
- kickoff_hour: 8.2809 (~8.3%)
- home_last_point_diff_at_snapshot: 8.1592 (~8.2%)
- away_last_point_diff_at_snapshot: 6.6448 (~6.6%)
- homeConference: 5.3533 (~5.4%)
- home_losses_at_snapshot: 4.3154 (~4.3%)
- away_losses_at_snapshot: 2.9375 (~2.9%)
- home_wins_at_snapshot: 2.0728 (~2.1%)
- away_wins_at_snapshot: 1.8328 (~1.8%)
- homeTeamRank: 0.4269 (~0.4%)
- neutralSite: 0.1741 (~0.2%)
- home_wins_at_snapshot_missing: 0.1457 (~0.1%)
- isRivalry: 0.1422 (~0.1%)
- away_wins_at_snapshot_missing: 0.1368 (~0.1%)
- away_last_point_diff_at_snapshot_missing: 0.1095 (~0.1%)
- away_losses_at_snapshot_missing: 0.0949 (~0.1%)
- home_last_point_diff_at_snapshot_missing: 0.0814 (~0.1%)
- home_losses_at_snapshot_missing: 0.0718 (~0.1%)
- season_year: 0.0000 (~0.0%)
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

**Possibly unrelated (near-zero importance):** season_year, season_year_missing, hours_until_game_missing, isRankedMatchup, capacity_missing, week_missing, kickoff_hour_missing, neutralSite_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
