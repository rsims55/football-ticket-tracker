# 📈 Weekly Ticket Price Model Report
**Date:** 2026-06-29

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-06-29 06:50:31**
- Features in model: **32**
- Rows evaluated: **414889**
- gap_pct MAE: **0.2332**
- gap_pct within 0.05: **0.0758**
- Price MAE: **$18.50**
- Price RMSE: **$27.30**
- Price within 5%: **0.0437**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-06-24 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-06-29 01:53:10_) — Snapshot appended (3188 new rows). Total now: 330136
- **model_train**: **success** (_2026-06-29 06:50:31_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-06-22 08:15:09_) — Weekly report generated: reports\weekly\2026-06-22\weekly_report_2026-06-22.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-06-29 01:53:10**
- Model last trained: **2026-06-29 06:50:31**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 21.1950 (~21.2%)
- homeTeam: 17.4785 (~17.5%)
- awayTeam: 13.7319 (~13.7%)
- week: 8.8776 (~8.9%)
- capacity: 8.5611 (~8.6%)
- homeConference: 6.4662 (~6.5%)
- home_last_point_diff_at_snapshot: 6.1565 (~6.2%)
- kickoff_hour: 3.6956 (~3.7%)
- home_losses_at_snapshot: 2.9470 (~2.9%)
- away_wins_at_snapshot: 2.5019 (~2.5%)
- home_wins_at_snapshot: 2.4483 (~2.4%)
- kickoff_dayofweek: 2.0412 (~2.0%)
- season_year: 2.0410 (~2.0%)
- homeTeamRank: 1.1817 (~1.2%)
- away_losses_at_snapshot_missing: 0.3378 (~0.3%)
- neutralSite: 0.1424 (~0.1%)
- home_losses_at_snapshot_missing: 0.1100 (~0.1%)
- away_wins_at_snapshot_missing: 0.0628 (~0.1%)
- home_wins_at_snapshot_missing: 0.0236 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** isRankedMatchup, season_year_missing, hours_until_game_missing, isRivalry, home_last_point_diff_at_snapshot_missing, capacity_missing, week_missing, kickoff_hour_missing, neutralSite_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing, kickoff_dayofweek_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
