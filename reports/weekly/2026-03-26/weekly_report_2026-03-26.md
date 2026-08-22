# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-26

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-26 06:47:01**
- Features in model: **34**
- Rows evaluated: **107118**
- gap_pct MAE: **0.1971**
- gap_pct within 0.05: **0.1492**
- Price MAE: **$10.81**
- Price RMSE: **$19.95**
- Price within 5%: **0.0865**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-25 13:12:21_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-26 04:38:30_) — Snapshot appended (3183 new rows). Total now: 38580
- **model_train**: **success** (_2026-03-26 06:47:01_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-25 14:32:14_) — Weekly report generated: reports\weekly\2026-03-25\weekly_report_2026-03-25.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-26 04:38:30**
- Model last trained: **2026-03-26 06:47:01**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 21.5706 (~21.6%)
- hours_until_game: 11.3596 (~11.4%)
- capacity: 10.7060 (~10.7%)
- kickoff_hour: 8.6538 (~8.7%)
- home_last_point_diff_at_snapshot: 8.1998 (~8.2%)
- awayTeam: 7.5474 (~7.5%)
- week: 7.2895 (~7.3%)
- away_last_point_diff_at_snapshot: 6.0338 (~6.0%)
- awayConference: 5.6391 (~5.6%)
- home_losses_at_snapshot: 3.4856 (~3.5%)
- homeConference: 3.1087 (~3.1%)
- away_wins_at_snapshot: 2.2074 (~2.2%)
- away_losses_at_snapshot: 2.0788 (~2.1%)
- away_losses_at_snapshot_missing: 0.5826 (~0.6%)
- homeTeamRank: 0.5758 (~0.6%)
- away_last_point_diff_at_snapshot_missing: 0.2706 (~0.3%)
- home_wins_at_snapshot_missing: 0.2637 (~0.3%)
- home_losses_at_snapshot_missing: 0.2227 (~0.2%)
- neutralSite: 0.1243 (~0.1%)
- isRivalry: 0.0801 (~0.1%)
- season_year: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
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

**Possibly unrelated (near-zero importance):** season_year, week_missing, isRankedMatchup, capacity_missing, season_year_missing, hours_until_game_missing, neutralSite_missing, away_wins_at_snapshot_missing, home_last_point_diff_at_snapshot_missing, kickoff_hour_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
