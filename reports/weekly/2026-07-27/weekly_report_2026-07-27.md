# 📈 Weekly Ticket Price Model Report
**Date:** 2026-07-27

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-07-27 06:52:11**
- Features in model: **33**
- Rows evaluated: **511396**
- gap_pct MAE: **0.1788**
- gap_pct within 0.05: **0.1340**
- Price MAE: **$13.40**
- Price RMSE: **$23.08**
- Price within 5%: **0.0871**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-07-22 05:30:12_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-07-27 05:20:34_) — Snapshot appended (3185 new rows). Total now: 427219
- **model_train**: **success** (_2026-07-27 06:52:11_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-07-20 08:15:09_) — Weekly report generated: reports\weekly\2026-07-20\weekly_report_2026-07-20.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-07-27 05:20:34**
- Model last trained: **2026-07-27 06:52:11**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 33.3216 (~33.3%)
- homeTeam: 20.7932 (~20.8%)
- awayTeam: 14.1115 (~14.1%)
- week: 7.2306 (~7.2%)
- capacity: 5.5977 (~5.6%)
- home_last_point_diff_at_snapshot: 3.8150 (~3.8%)
- away_last_point_diff_at_snapshot: 2.2928 (~2.3%)
- kickoff_hour: 2.0420 (~2.0%)
- home_losses_at_snapshot: 1.8554 (~1.9%)
- homeConference: 1.8295 (~1.8%)
- awayConference: 1.6254 (~1.6%)
- season_year: 1.5813 (~1.6%)
- kickoff_dayofweek: 1.4032 (~1.4%)
- home_wins_at_snapshot: 1.0773 (~1.1%)
- home_losses_at_snapshot_missing: 0.5649 (~0.6%)
- homeTeamRank: 0.5052 (~0.5%)
- away_losses_at_snapshot_missing: 0.1723 (~0.2%)
- home_wins_at_snapshot_missing: 0.1054 (~0.1%)
- neutralSite: 0.0758 (~0.1%)
- capacity_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** capacity_missing, isRankedMatchup, isRivalry, season_year_missing, hours_until_game_missing, week_missing, neutralSite_missing, away_wins_at_snapshot_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
