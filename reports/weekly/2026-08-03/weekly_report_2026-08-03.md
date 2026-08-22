# 📈 Weekly Ticket Price Model Report
**Date:** 2026-08-03

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-08-03 06:51:35**
- Features in model: **34**
- Rows evaluated: **521848**
- gap_pct MAE: **0.2008**
- gap_pct within 0.05: **0.1117**
- Price MAE: **$15.17**
- Price RMSE: **$25.46**
- Price within 5%: **0.0716**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-07-29 05:30:13_) — Weekly update complete for 2026
- **daily_snapshot**: **failed** (_2026-08-03 06:22:24_) — No snapshot CSV produced.
- **model_train**: **success** (_2026-08-03 06:51:35_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-07-27 08:15:07_) — Weekly report generated: reports\weekly\2026-07-27\weekly_report_2026-07-27.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **In-season**
- Snapshots last updated: **2026-07-30 07:03:29**
- Model last trained: **2026-08-03 06:51:35**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 31.9549 (~32.0%)
- homeTeam: 21.1235 (~21.1%)
- awayTeam: 14.5926 (~14.6%)
- week: 7.0494 (~7.0%)
- home_last_point_diff_at_snapshot: 4.1425 (~4.1%)
- capacity: 3.8255 (~3.8%)
- kickoff_hour: 3.2184 (~3.2%)
- homeConference: 2.6818 (~2.7%)
- away_last_point_diff_at_snapshot: 2.4574 (~2.5%)
- awayConference: 2.4344 (~2.4%)
- home_losses_at_snapshot: 1.7281 (~1.7%)
- season_year: 1.4673 (~1.5%)
- home_wins_at_snapshot: 1.1057 (~1.1%)
- kickoff_dayofweek: 0.9414 (~0.9%)
- homeTeamRank: 0.4846 (~0.5%)
- away_wins_at_snapshot_missing: 0.2700 (~0.3%)
- away_wins_at_snapshot: 0.1785 (~0.2%)
- away_losses_at_snapshot_missing: 0.1330 (~0.1%)
- home_wins_at_snapshot_missing: 0.1246 (~0.1%)
- home_losses_at_snapshot_missing: 0.0864 (~0.1%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** capacity_missing, season_year_missing, neutralSite, isRivalry, hours_until_game_missing, isRankedMatchup, neutralSite_missing, week_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
