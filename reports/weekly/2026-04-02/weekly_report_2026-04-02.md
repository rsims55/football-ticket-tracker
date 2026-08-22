# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-02

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-02 06:49:09**
- Features in model: **34**
- Rows evaluated: **107118**
- gap_pct MAE: **0.2063**
- gap_pct within 0.05: **0.1361**
- Price MAE: **$11.81**
- Price RMSE: **$22.98**
- Price within 5%: **0.0746**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-01 05:30:13_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-02 03:17:36_) — Snapshot appended (3175 new rows). Total now: 63039
- **model_train**: **success** (_2026-04-02 06:49:09_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-01 08:15:08_) — Weekly report generated: reports\weekly\2026-04-01\weekly_report_2026-04-01.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-02 03:17:36**
- Model last trained: **2026-04-02 06:49:09**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.2405 (~22.2%)
- awayTeam: 12.9577 (~13.0%)
- capacity: 9.8182 (~9.8%)
- kickoff_hour: 8.1042 (~8.1%)
- home_last_point_diff_at_snapshot: 7.7151 (~7.7%)
- awayConference: 7.6968 (~7.7%)
- hours_until_game: 7.4127 (~7.4%)
- homeConference: 5.9866 (~6.0%)
- week: 5.5118 (~5.5%)
- away_last_point_diff_at_snapshot: 4.2567 (~4.3%)
- home_losses_at_snapshot: 2.4520 (~2.5%)
- away_wins_at_snapshot: 1.5132 (~1.5%)
- away_losses_at_snapshot: 1.5006 (~1.5%)
- homeTeamRank: 1.1668 (~1.2%)
- away_losses_at_snapshot_missing: 0.3918 (~0.4%)
- neutralSite: 0.3463 (~0.3%)
- home_last_point_diff_at_snapshot_missing: 0.2569 (~0.3%)
- away_last_point_diff_at_snapshot_missing: 0.1912 (~0.2%)
- home_wins_at_snapshot_missing: 0.1679 (~0.2%)
- home_losses_at_snapshot_missing: 0.1606 (~0.2%)
- homeTeamRank_missing: 0.1526 (~0.2%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, isRivalry, season_year_missing, hours_until_game_missing, week_missing, capacity_missing, kickoff_hour_missing, away_wins_at_snapshot_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
