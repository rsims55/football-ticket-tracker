# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-26

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-26 06:48:46**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1940**
- gap_pct within 0.05: **0.1495**
- Price MAE: **$11.31**
- Price RMSE: **$23.73**
- Price within 5%: **0.0878**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-26 02:08:16_) — Snapshot appended (3180 new rows). Total now: 146392
- **model_train**: **success** (_2026-04-26 06:48:47_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-25 08:15:11_) — Weekly report generated: reports\weekly\2026-04-25\weekly_report_2026-04-25.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-26 02:08:16**
- Model last trained: **2026-04-26 06:48:46**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.8586 (~18.9%)
- capacity: 11.5833 (~11.6%)
- hours_until_game: 10.4733 (~10.5%)
- awayTeam: 9.4306 (~9.4%)
- kickoff_hour: 8.2462 (~8.2%)
- home_last_point_diff_at_snapshot: 8.0489 (~8.0%)
- week: 7.0061 (~7.0%)
- away_last_point_diff_at_snapshot: 5.8917 (~5.9%)
- homeConference: 5.0687 (~5.1%)
- home_losses_at_snapshot: 4.6262 (~4.6%)
- away_wins_at_snapshot: 2.8245 (~2.8%)
- kickoff_dayofweek: 2.3461 (~2.3%)
- home_wins_at_snapshot: 2.0299 (~2.0%)
- homeTeamRank: 0.7659 (~0.8%)
- isRivalry: 0.7591 (~0.8%)
- away_losses_at_snapshot_missing: 0.7339 (~0.7%)
- homeTeamRank_missing: 0.4976 (~0.5%)
- away_wins_at_snapshot_missing: 0.2632 (~0.3%)
- neutralSite: 0.2225 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1554 (~0.2%)
- home_wins_at_snapshot_missing: 0.1364 (~0.1%)
- home_last_point_diff_at_snapshot_missing: 0.0321 (~0.0%)
- season_year: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, season_year_missing, hours_until_game_missing, isRankedMatchup, capacity_missing, home_losses_at_snapshot_missing, kickoff_hour_missing, week_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
