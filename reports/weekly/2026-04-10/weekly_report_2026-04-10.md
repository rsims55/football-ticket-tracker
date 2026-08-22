# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-10

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-10 06:48:39**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1955**
- gap_pct within 0.05: **0.1462**
- Price MAE: **$11.21**
- Price RMSE: **$22.30**
- Price within 5%: **0.0933**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-08 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-10 07:43:54_) — Snapshot appended (3184 new rows). Total now: 91883
- **model_train**: **success** (_2026-04-10 06:48:39_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-09 08:15:08_) — Weekly report generated: reports\weekly\2026-04-09\weekly_report_2026-04-09.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-10 07:43:54**
- Model last trained: **2026-04-10 06:48:39**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 21.5127 (~21.5%)
- hours_until_game: 11.5706 (~11.6%)
- capacity: 10.7197 (~10.7%)
- kickoff_hour: 8.4924 (~8.5%)
- home_last_point_diff_at_snapshot: 8.2143 (~8.2%)
- week: 7.7148 (~7.7%)
- awayTeam: 7.6952 (~7.7%)
- away_last_point_diff_at_snapshot: 5.9771 (~6.0%)
- awayConference: 5.3202 (~5.3%)
- home_losses_at_snapshot: 3.6893 (~3.7%)
- homeConference: 2.7929 (~2.8%)
- away_wins_at_snapshot: 2.3364 (~2.3%)
- away_losses_at_snapshot: 2.0639 (~2.1%)
- homeTeamRank: 0.5422 (~0.5%)
- away_losses_at_snapshot_missing: 0.5133 (~0.5%)
- away_last_point_diff_at_snapshot_missing: 0.2864 (~0.3%)
- home_wins_at_snapshot_missing: 0.2792 (~0.3%)
- home_losses_at_snapshot_missing: 0.2357 (~0.2%)
- neutralSite: 0.0436 (~0.0%)
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
