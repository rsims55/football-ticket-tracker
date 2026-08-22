# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-05

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-05 06:49:33**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1968**
- gap_pct within 0.05: **0.1418**
- Price MAE: **$11.60**
- Price RMSE: **$24.16**
- Price within 5%: **0.0779**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-01 05:30:13_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-05 07:47:29_) — Snapshot appended (3184 new rows). Total now: 74403
- **model_train**: **success** (_2026-04-05 06:49:33_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-04 08:15:16_) — Weekly report generated: reports\weekly\2026-04-04\weekly_report_2026-04-04.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-05 07:47:29**
- Model last trained: **2026-04-05 06:49:33**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.5620 (~22.6%)
- awayTeam: 12.1443 (~12.1%)
- capacity: 9.6877 (~9.7%)
- hours_until_game: 8.0848 (~8.1%)
- kickoff_hour: 8.0632 (~8.1%)
- home_last_point_diff_at_snapshot: 7.7569 (~7.8%)
- awayConference: 7.2460 (~7.2%)
- homeConference: 6.0190 (~6.0%)
- week: 5.6182 (~5.6%)
- away_last_point_diff_at_snapshot: 4.4315 (~4.4%)
- home_losses_at_snapshot: 2.4454 (~2.4%)
- away_wins_at_snapshot: 1.6909 (~1.7%)
- away_losses_at_snapshot: 1.6026 (~1.6%)
- homeTeamRank: 0.9932 (~1.0%)
- away_losses_at_snapshot_missing: 0.4588 (~0.5%)
- neutralSite: 0.2562 (~0.3%)
- home_last_point_diff_at_snapshot_missing: 0.2549 (~0.3%)
- home_losses_at_snapshot_missing: 0.2486 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1953 (~0.2%)
- home_wins_at_snapshot_missing: 0.1838 (~0.2%)
- isRivalry: 0.0567 (~0.1%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, capacity_missing, season_year_missing, hours_until_game_missing, week_missing, neutralSite_missing, kickoff_hour_missing, away_wins_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
