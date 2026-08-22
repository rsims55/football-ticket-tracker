# 📈 Weekly Ticket Price Model Report
**Date:** 2026-06-04

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-06-04 10:51:59**
- Features in model: **35**
- Rows evaluated: **328920**
- gap_pct MAE: **0.2418**
- gap_pct within 0.05: **0.0721**
- Price MAE: **$20.81**
- Price RMSE: **$30.72**
- Price within 5%: **0.0470**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-06-04 09:12:43_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-06-04 10:46:10_) — Snapshot appended (3190 new rows). Total now: 243093
- **model_train**: **success** (_2026-06-04 10:51:59_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-25 08:15:11_) — Weekly report generated: reports\weekly\2026-05-25\weekly_report_2026-05-25.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-06-04 10:46:10**
- Model last trained: **2026-06-04 10:51:59**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 18.6270 (~18.6%)
- hours_until_game: 16.2927 (~16.3%)
- awayTeam: 11.2060 (~11.2%)
- week: 9.3515 (~9.4%)
- capacity: 8.9261 (~8.9%)
- home_last_point_diff_at_snapshot: 8.0110 (~8.0%)
- awayConference: 4.6352 (~4.6%)
- homeConference: 4.3521 (~4.4%)
- kickoff_hour: 4.1699 (~4.2%)
- away_last_point_diff_at_snapshot: 3.4763 (~3.5%)
- home_losses_at_snapshot: 2.5256 (~2.5%)
- away_wins_at_snapshot: 2.1175 (~2.1%)
- home_wins_at_snapshot: 1.8337 (~1.8%)
- homeTeamRank: 1.5979 (~1.6%)
- away_losses_at_snapshot: 1.3170 (~1.3%)
- season_year: 0.7276 (~0.7%)
- homeTeamRank_missing: 0.3443 (~0.3%)
- neutralSite: 0.1945 (~0.2%)
- away_losses_at_snapshot_missing: 0.1575 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1363 (~0.1%)
- capacity_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** capacity_missing, week_missing, hours_until_game_missing, isRankedMatchup, isRivalry, season_year_missing, home_last_point_diff_at_snapshot_missing, neutralSite_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, home_wins_at_snapshot_missing, kickoff_hour_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
