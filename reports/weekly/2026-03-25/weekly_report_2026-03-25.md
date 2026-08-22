# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-25

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-25 14:32:04**
- Features in model: **34**
- Rows evaluated: **107118**
- gap_pct MAE: **0.2009**
- gap_pct within 0.05: **0.1387**
- Price MAE: **$11.37**
- Price RMSE: **$21.78**
- Price within 5%: **0.0800**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-25 13:12:21_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-25 14:29:29_) — Snapshot appended (3183 new rows). Total now: 35961
- **model_train**: **success** (_2026-03-25 14:32:04_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-25 08:15:08_) — Weekly report generated: reports\weekly\2026-03-25\weekly_report_2026-03-25.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-25 14:29:29**
- Model last trained: **2026-03-25 14:32:04**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.3172 (~22.3%)
- awayTeam: 11.6117 (~11.6%)
- capacity: 10.0186 (~10.0%)
- hours_until_game: 9.0326 (~9.0%)
- kickoff_hour: 8.2312 (~8.2%)
- home_last_point_diff_at_snapshot: 7.5976 (~7.6%)
- awayConference: 6.6468 (~6.6%)
- week: 6.3654 (~6.4%)
- away_last_point_diff_at_snapshot: 4.9680 (~5.0%)
- homeConference: 4.6239 (~4.6%)
- home_losses_at_snapshot: 2.7935 (~2.8%)
- away_wins_at_snapshot: 1.6941 (~1.7%)
- away_losses_at_snapshot: 1.5920 (~1.6%)
- homeTeamRank: 0.8279 (~0.8%)
- away_losses_at_snapshot_missing: 0.4247 (~0.4%)
- neutralSite: 0.4154 (~0.4%)
- away_last_point_diff_at_snapshot_missing: 0.2174 (~0.2%)
- home_wins_at_snapshot_missing: 0.2150 (~0.2%)
- home_losses_at_snapshot_missing: 0.2079 (~0.2%)
- home_last_point_diff_at_snapshot_missing: 0.0916 (~0.1%)
- isRivalry: 0.0874 (~0.1%)
- homeTeamRank_missing: 0.0203 (~0.0%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
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

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, season_year_missing, hours_until_game_missing, week_missing, capacity_missing, kickoff_hour_missing, away_wins_at_snapshot_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
