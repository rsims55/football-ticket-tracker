# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-24

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-24 06:48:53**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1937**
- gap_pct within 0.05: **0.1456**
- Price MAE: **$11.01**
- Price RMSE: **$23.07**
- Price within 5%: **0.0795**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-22 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-24 03:42:22_) — Snapshot appended (3170 new rows). Total now: 139369
- **model_train**: **success** (_2026-04-24 06:48:53_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-23 08:15:11_) — Weekly report generated: reports\weekly\2026-04-23\weekly_report_2026-04-23.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-24 03:42:22**
- Model last trained: **2026-04-24 06:48:53**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 19.9113 (~19.9%)
- capacity: 10.9584 (~11.0%)
- awayTeam: 10.2500 (~10.2%)
- hours_until_game: 9.0630 (~9.1%)
- home_last_point_diff_at_snapshot: 8.2671 (~8.3%)
- kickoff_hour: 8.2042 (~8.2%)
- week: 7.2875 (~7.3%)
- awayConference: 5.6543 (~5.7%)
- homeConference: 4.6556 (~4.7%)
- away_last_point_diff_at_snapshot: 4.5289 (~4.5%)
- home_losses_at_snapshot: 3.9871 (~4.0%)
- away_wins_at_snapshot: 2.1632 (~2.2%)
- home_wins_at_snapshot: 2.1027 (~2.1%)
- homeTeamRank: 0.8798 (~0.9%)
- away_losses_at_snapshot_missing: 0.6477 (~0.6%)
- away_wins_at_snapshot_missing: 0.3655 (~0.4%)
- neutralSite: 0.3641 (~0.4%)
- home_last_point_diff_at_snapshot_missing: 0.2464 (~0.2%)
- home_wins_at_snapshot_missing: 0.2280 (~0.2%)
- isRivalry: 0.1359 (~0.1%)
- homeTeamRank_missing: 0.0365 (~0.0%)
- away_last_point_diff_at_snapshot_missing: 0.0325 (~0.0%)
- home_losses_at_snapshot_missing: 0.0302 (~0.0%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, hours_until_game_missing, week_missing, capacity_missing, season_year_missing, kickoff_hour_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
