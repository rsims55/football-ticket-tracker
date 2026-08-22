# 📈 Weekly Ticket Price Model Report
**Date:** 2026-06-15

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-06-15 06:50:50**
- Features in model: **36**
- Rows evaluated: **366969**
- gap_pct MAE: **0.2029**
- gap_pct within 0.05: **0.1025**
- Price MAE: **$16.10**
- Price RMSE: **$23.90**
- Price within 5%: **0.0768**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-06-10 05:30:08_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-06-15 07:50:18_) — Snapshot appended (3188 new rows). Total now: 282533
- **model_train**: **success** (_2026-06-15 06:50:50_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-06-08 08:15:08_) — Weekly report generated: reports\weekly\2026-06-08\weekly_report_2026-06-08.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-06-15 07:50:18**
- Model last trained: **2026-06-15 06:50:50**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- week: 16.6870 (~16.7%)
- hours_until_game: 14.2594 (~14.3%)
- capacity: 13.3328 (~13.3%)
- homeConference: 10.0243 (~10.0%)
- homeTeam: 8.5703 (~8.6%)
- home_last_point_diff_at_snapshot: 7.6921 (~7.7%)
- kickoff_hour: 6.8511 (~6.9%)
- awayTeam: 4.7927 (~4.8%)
- homeTeamRank: 4.4680 (~4.5%)
- away_last_point_diff_at_snapshot_missing: 4.0700 (~4.1%)
- awayConference: 2.6238 (~2.6%)
- away_wins_at_snapshot: 1.8979 (~1.9%)
- home_wins_at_snapshot: 1.8512 (~1.9%)
- home_losses_at_snapshot: 1.7967 (~1.8%)
- away_losses_at_snapshot_missing: 0.6981 (~0.7%)
- away_last_point_diff_at_snapshot: 0.3846 (~0.4%)
- season_year: 0.0000 (~0.0%)
- isRivalry: 0.0000 (~0.0%)
- neutralSite: 0.0000 (~0.0%)
- away_losses_at_snapshot: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- kickoff_dayofweek: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- home_last_point_diff_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- away_wins_at_snapshot_missing: 0.0000 (~0.0%)
- home_losses_at_snapshot_missing: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRivalry, neutralSite, away_losses_at_snapshot, week_missing, capacity_missing, season_year_missing, hours_until_game_missing, kickoff_dayofweek, isRankedMatchup, home_last_point_diff_at_snapshot_missing, kickoff_hour_missing, neutralSite_missing, away_wins_at_snapshot_missing, home_losses_at_snapshot_missing, home_wins_at_snapshot_missing, kickoff_dayofweek_missing, homeTeamRank_missing, isRankedMatchup_missing, isRivalry_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
