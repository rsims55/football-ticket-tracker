# 📈 Weekly Ticket Price Model Report
**Date:** 2026-07-20

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-07-20 06:53:21**
- Features in model: **33**
- Rows evaluated: **487900**
- gap_pct MAE: **0.2042**
- gap_pct within 0.05: **0.1063**
- Price MAE: **$15.75**
- Price RMSE: **$25.75**
- Price within 5%: **0.0719**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-07-15 05:30:14_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-07-20 06:36:55_) — Snapshot appended (3174 new rows). Total now: 403599
- **model_train**: **success** (_2026-07-20 06:53:21_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-07-13 08:15:08_) — Weekly report generated: reports\weekly\2026-07-13\weekly_report_2026-07-13.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-07-20 06:36:55**
- Model last trained: **2026-07-20 06:53:21**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 31.6256 (~31.6%)
- homeTeam: 20.4342 (~20.4%)
- awayTeam: 13.8227 (~13.8%)
- week: 8.5087 (~8.5%)
- capacity: 5.0289 (~5.0%)
- home_last_point_diff_at_snapshot: 5.0122 (~5.0%)
- awayConference: 3.0518 (~3.1%)
- away_last_point_diff_at_snapshot: 2.2247 (~2.2%)
- kickoff_hour: 2.2166 (~2.2%)
- home_losses_at_snapshot: 1.7727 (~1.8%)
- homeConference: 1.5655 (~1.6%)
- home_wins_at_snapshot: 1.3069 (~1.3%)
- kickoff_dayofweek: 1.2416 (~1.2%)
- season_year: 0.8484 (~0.8%)
- homeTeamRank: 0.6037 (~0.6%)
- isRivalry: 0.3245 (~0.3%)
- home_wins_at_snapshot_missing: 0.2096 (~0.2%)
- away_losses_at_snapshot_missing: 0.2017 (~0.2%)
- capacity_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** capacity_missing, hours_until_game_missing, isRankedMatchup, neutralSite, season_year_missing, kickoff_hour_missing, week_missing, neutralSite_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, home_last_point_diff_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
