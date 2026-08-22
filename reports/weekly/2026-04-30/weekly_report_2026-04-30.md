# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-30

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-30 06:48:06**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1934**
- gap_pct within 0.05: **0.1450**
- Price MAE: **$11.21**
- Price RMSE: **$22.93**
- Price within 5%: **0.0808**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-29 05:30:18_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-30 07:43:04_) — Snapshot appended (3192 new rows). Total now: 161280
- **model_train**: **success** (_2026-04-30 06:48:06_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-29 08:15:09_) — Weekly report generated: reports\weekly\2026-04-29\weekly_report_2026-04-29.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-30 07:43:04**
- Model last trained: **2026-04-30 06:48:06**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 17.9293 (~17.9%)
- capacity: 12.1741 (~12.2%)
- hours_until_game: 11.1960 (~11.2%)
- kickoff_hour: 8.4515 (~8.5%)
- home_last_point_diff_at_snapshot: 7.7478 (~7.7%)
- awayTeam: 7.7413 (~7.7%)
- week: 7.6338 (~7.6%)
- away_last_point_diff_at_snapshot: 6.7692 (~6.8%)
- home_losses_at_snapshot: 4.9999 (~5.0%)
- homeConference: 4.9364 (~4.9%)
- away_wins_at_snapshot: 3.0265 (~3.0%)
- kickoff_dayofweek: 2.1895 (~2.2%)
- home_wins_at_snapshot: 2.0534 (~2.1%)
- isRivalry: 0.8358 (~0.8%)
- away_losses_at_snapshot_missing: 0.8353 (~0.8%)
- homeTeamRank_missing: 0.5478 (~0.5%)
- homeTeamRank: 0.4556 (~0.5%)
- away_last_point_diff_at_snapshot_missing: 0.1711 (~0.2%)
- home_wins_at_snapshot_missing: 0.1241 (~0.1%)
- away_wins_at_snapshot_missing: 0.0913 (~0.1%)
- neutralSite: 0.0902 (~0.1%)
- season_year: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, capacity_missing, season_year_missing, hours_until_game_missing, isRankedMatchup, home_losses_at_snapshot_missing, week_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, isRankedMatchup_missing, isRivalry_missing, neutralSite_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
