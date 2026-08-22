# 📈 Weekly Ticket Price Model Report
**Date:** 2026-07-13

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-07-13 06:52:00**
- Features in model: **33**
- Rows evaluated: **463520**
- gap_pct MAE: **0.1913**
- gap_pct within 0.05: **0.1047**
- Price MAE: **$14.74**
- Price RMSE: **$23.06**
- Price within 5%: **0.0706**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-07-08 05:30:09_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-07-13 02:11:04_) — Snapshot appended (3186 new rows). Total now: 379094
- **model_train**: **success** (_2026-07-13 06:52:00_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-07-06 08:15:08_) — Weekly report generated: reports\weekly\2026-07-06\weekly_report_2026-07-06.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-07-13 02:11:04**
- Model last trained: **2026-07-13 06:52:00**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 27.3619 (~27.4%)
- homeTeam: 19.4774 (~19.5%)
- awayTeam: 13.6292 (~13.6%)
- week: 6.0455 (~6.0%)
- capacity: 5.7323 (~5.7%)
- home_last_point_diff_at_snapshot: 5.7012 (~5.7%)
- awayConference: 4.0663 (~4.1%)
- kickoff_hour: 3.8513 (~3.9%)
- home_losses_at_snapshot: 2.6300 (~2.6%)
- homeConference: 2.3697 (~2.4%)
- away_last_point_diff_at_snapshot: 2.3209 (~2.3%)
- home_wins_at_snapshot: 2.0952 (~2.1%)
- kickoff_dayofweek: 1.6820 (~1.7%)
- season_year: 1.1521 (~1.2%)
- homeTeamRank: 0.6954 (~0.7%)
- away_losses_at_snapshot_missing: 0.4132 (~0.4%)
- homeTeamRank_missing: 0.3110 (~0.3%)
- home_losses_at_snapshot_missing: 0.3087 (~0.3%)
- neutralSite: 0.0837 (~0.1%)
- home_wins_at_snapshot_missing: 0.0731 (~0.1%)
- capacity_missing: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
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

**Possibly unrelated (near-zero importance):** capacity_missing, isRivalry, season_year_missing, hours_until_game_missing, isRankedMatchup, neutralSite_missing, away_wins_at_snapshot_missing, kickoff_hour_missing, home_last_point_diff_at_snapshot_missing, week_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
