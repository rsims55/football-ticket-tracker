# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-21

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-21 06:47:57**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1955**
- gap_pct within 0.05: **0.1350**
- Price MAE: **$11.13**
- Price RMSE: **$21.72**
- Price within 5%: **0.0861**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-15 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-21 02:50:26_) — Snapshot appended (3172 new rows). Total now: 128870
- **model_train**: **success** (_2026-04-21 06:47:57_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-20 08:15:13_) — Weekly report generated: reports\weekly\2026-04-20\weekly_report_2026-04-20.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-21 02:50:26**
- Model last trained: **2026-04-21 06:47:57**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 19.7890 (~19.8%)
- awayTeam: 12.6448 (~12.6%)
- capacity: 12.2361 (~12.2%)
- hours_until_game: 11.2728 (~11.3%)
- home_last_point_diff_at_snapshot: 8.8540 (~8.9%)
- kickoff_hour: 7.7363 (~7.7%)
- week: 6.7220 (~6.7%)
- away_last_point_diff_at_snapshot: 5.7723 (~5.8%)
- home_losses_at_snapshot: 4.4396 (~4.4%)
- homeConference: 3.2757 (~3.3%)
- kickoff_dayofweek: 2.4642 (~2.5%)
- away_losses_at_snapshot: 1.7492 (~1.7%)
- away_wins_at_snapshot: 1.0333 (~1.0%)
- homeTeamRank: 0.7413 (~0.7%)
- away_losses_at_snapshot_missing: 0.5407 (~0.5%)
- isRivalry: 0.3055 (~0.3%)
- home_losses_at_snapshot_missing: 0.2734 (~0.3%)
- neutralSite: 0.1190 (~0.1%)
- home_last_point_diff_at_snapshot_missing: 0.0311 (~0.0%)
- season_year: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- away_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, hours_until_game_missing, isRankedMatchup, capacity_missing, season_year_missing, away_last_point_diff_at_snapshot_missing, week_missing, away_wins_at_snapshot_missing, home_wins_at_snapshot_missing, kickoff_hour_missing, neutralSite_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
