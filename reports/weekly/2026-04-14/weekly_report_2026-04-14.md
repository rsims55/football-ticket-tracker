# 📈 Weekly Ticket Price Model Report
**Date:** 2026-04-14

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-04-14 06:48:16**
- Features in model: **34**
- Rows evaluated: **107127**
- gap_pct MAE: **0.1948**
- gap_pct within 0.05: **0.1469**
- Price MAE: **$11.37**
- Price RMSE: **$22.84**
- Price within 5%: **0.0784**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-04-08 05:30:07_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-04-14 05:35:44_) — Snapshot appended (3184 new rows). Total now: 104398
- **model_train**: **success** (_2026-04-14 06:48:16_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-04-13 08:15:08_) — Weekly report generated: reports\weekly\2026-04-13\weekly_report_2026-04-13.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-04-14 05:35:44**
- Model last trained: **2026-04-14 06:48:16**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.3028 (~22.3%)
- hours_until_game: 11.0353 (~11.0%)
- capacity: 10.5575 (~10.6%)
- kickoff_hour: 8.4043 (~8.4%)
- home_last_point_diff_at_snapshot: 8.3823 (~8.4%)
- awayTeam: 7.7175 (~7.7%)
- week: 7.5354 (~7.5%)
- awayConference: 5.6196 (~5.6%)
- away_last_point_diff_at_snapshot: 5.4290 (~5.4%)
- home_losses_at_snapshot: 3.4658 (~3.5%)
- homeConference: 3.4568 (~3.5%)
- away_wins_at_snapshot: 2.1500 (~2.1%)
- away_losses_at_snapshot: 1.7468 (~1.7%)
- homeTeamRank: 0.6717 (~0.7%)
- away_losses_at_snapshot_missing: 0.4724 (~0.5%)
- away_last_point_diff_at_snapshot_missing: 0.2823 (~0.3%)
- home_losses_at_snapshot_missing: 0.2762 (~0.3%)
- home_wins_at_snapshot_missing: 0.2569 (~0.3%)
- neutralSite: 0.1560 (~0.2%)
- home_last_point_diff_at_snapshot_missing: 0.0815 (~0.1%)
- season_year: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
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

**Possibly unrelated (near-zero importance):** season_year, capacity_missing, isRankedMatchup, isRivalry, season_year_missing, hours_until_game_missing, week_missing, neutralSite_missing, kickoff_hour_missing, away_wins_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
