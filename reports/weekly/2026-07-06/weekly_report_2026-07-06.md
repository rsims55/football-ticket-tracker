# 📈 Weekly Ticket Price Model Report
**Date:** 2026-07-06

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-07-06 06:50:52**
- Features in model: **31**
- Rows evaluated: **439148**
- gap_pct MAE: **0.1782**
- gap_pct within 0.05: **0.1398**
- Price MAE: **$13.82**
- Price RMSE: **$21.21**
- Price within 5%: **0.0963**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-07-01 05:31:09_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-07-06 02:34:33_) — Snapshot appended (3144 new rows). Total now: 354604
- **model_train**: **success** (_2026-07-06 06:50:52_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-06-29 08:15:07_) — Weekly report generated: reports\weekly\2026-06-29\weekly_report_2026-06-29.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-07-06 02:34:33**
- Model last trained: **2026-07-06 06:50:52**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 30.1849 (~30.2%)
- homeTeam: 17.6965 (~17.7%)
- awayTeam: 13.0957 (~13.1%)
- week: 8.9719 (~9.0%)
- home_last_point_diff_at_snapshot: 6.0730 (~6.1%)
- capacity: 5.9943 (~6.0%)
- kickoff_hour: 3.6685 (~3.7%)
- home_losses_at_snapshot: 3.5838 (~3.6%)
- homeConference: 3.0397 (~3.0%)
- away_last_point_diff_at_snapshot: 2.2655 (~2.3%)
- homeTeamRank: 1.7845 (~1.8%)
- awayConference: 1.5826 (~1.6%)
- season_year: 0.8584 (~0.9%)
- home_losses_at_snapshot_missing: 0.4418 (~0.4%)
- homeTeamRank_missing: 0.3004 (~0.3%)
- home_wins_at_snapshot_missing: 0.2430 (~0.2%)
- away_losses_at_snapshot_missing: 0.2155 (~0.2%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** capacity_missing, season_year_missing, neutralSite, isRivalry, hours_until_game_missing, isRankedMatchup, neutralSite_missing, away_wins_at_snapshot_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, week_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
