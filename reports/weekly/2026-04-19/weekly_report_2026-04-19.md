# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-19

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-19 06:47:56**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1954**
- gap_pct within 0.05: **0.1485**
- Price MAE: **$11.17**
- Price RMSE: **$22.30**
- Price within 5%: **0.0882**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-15 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-19 07:22:08_) — Snapshot appended (3184 new rows). Total now: 121878
- **model_train**: **success** (_2026-04-19 06:47:56_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-18 08:15:13_) — Weekly report generated: reports\weekly\2026-04-18\weekly_report_2026-04-18.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-19 07:22:08**
- Model last trained: **2026-04-19 06:47:56**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 21.4369 (~21.4%)
- hours_until_game: 12.1860 (~12.2%)
- capacity: 11.0133 (~11.0%)
- home_last_point_diff_at_snapshot: 8.3896 (~8.4%)
- week: 8.2867 (~8.3%)
- kickoff_hour: 8.1816 (~8.2%)
- awayTeam: 8.0614 (~8.1%)
- away_last_point_diff_at_snapshot: 5.3571 (~5.4%)
- awayConference: 4.8839 (~4.9%)
- home_losses_at_snapshot: 3.8876 (~3.9%)
- homeConference: 2.6278 (~2.6%)
- away_wins_at_snapshot: 2.1292 (~2.1%)
- away_losses_at_snapshot: 1.5079 (~1.5%)
- homeTeamRank: 0.5851 (~0.6%)
- away_losses_at_snapshot_missing: 0.5540 (~0.6%)
- away_last_point_diff_at_snapshot_missing: 0.3091 (~0.3%)
- home_wins_at_snapshot_missing: 0.3013 (~0.3%)
- home_losses_at_snapshot_missing: 0.2544 (~0.3%)
- neutralSite: 0.0471 (~0.0%)
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
