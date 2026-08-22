# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-28

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-28 06:48:54**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1927**
- gap_pct within 0.05: **0.1450**
- Price MAE: **$11.19**
- Price RMSE: **$22.97**
- Price within 5%: **0.0859**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-28 05:39:06_) — Snapshot appended (3192 new rows). Total now: 153414
- **model_train**: **success** (_2026-04-28 06:48:54_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-27 08:15:07_) — Weekly report generated: reports\weekly\2026-04-27\weekly_report_2026-04-27.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-28 05:39:06**
- Model last trained: **2026-04-28 06:48:54**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.0840 (~18.1%)
- capacity: 12.0297 (~12.0%)
- hours_until_game: 11.7566 (~11.8%)
- kickoff_hour: 8.2949 (~8.3%)
- home_last_point_diff_at_snapshot: 7.8688 (~7.9%)
- week: 7.8219 (~7.8%)
- awayTeam: 7.1186 (~7.1%)
- away_last_point_diff_at_snapshot: 6.8300 (~6.8%)
- home_losses_at_snapshot: 5.0580 (~5.1%)
- homeConference: 4.5935 (~4.6%)
- away_wins_at_snapshot: 2.9909 (~3.0%)
- kickoff_dayofweek: 2.3137 (~2.3%)
- home_wins_at_snapshot: 2.1014 (~2.1%)
- isRivalry: 0.9089 (~0.9%)
- away_losses_at_snapshot_missing: 0.7518 (~0.8%)
- homeTeamRank_missing: 0.5958 (~0.6%)
- homeTeamRank: 0.3893 (~0.4%)
- away_last_point_diff_at_snapshot_missing: 0.1860 (~0.2%)
- home_wins_at_snapshot_missing: 0.1350 (~0.1%)
- away_wins_at_snapshot_missing: 0.0993 (~0.1%)
- neutralSite: 0.0718 (~0.1%)
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
