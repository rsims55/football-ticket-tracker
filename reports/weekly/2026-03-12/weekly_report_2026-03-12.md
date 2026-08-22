# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-12

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-12 12:06:30**
- Features in model: **36**
- Rows evaluated: **107118**
- gap_pct MAE: **0.1967**
- gap_pct within 0.05: **0.1470**
- Price MAE: **$10.98**
- Price RMSE: **$20.88**
- Price within 5%: **0.0915**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-12 10:50:38_) — Weekly update complete for 2026
- **daily_snapshot**: **failed** (_2026-03-12 12:04:14_) — Unhandled exception: 'date_local'
- **model_train**: **success** (_2026-03-12 12:06:30_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-12 11:33:24_) — Weekly report generated: reports\weekly\2026-03-12\weekly_report_2026-03-12.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-12 11:31:19**
- Model last trained: **2026-03-12 12:06:30**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 22.6101 (~22.6%)
- capacity: 9.5441 (~9.5%)
- awayTeam: 9.0329 (~9.0%)
- hours_until_game: 8.5409 (~8.5%)
- home_last_point_diff_at_snapshot: 8.1178 (~8.1%)
- kickoff_hour: 6.5598 (~6.6%)
- awayConference: 6.4128 (~6.4%)
- homeConference: 6.3057 (~6.3%)
- week: 6.1313 (~6.1%)
- away_last_point_diff_at_snapshot: 4.6953 (~4.7%)
- home_losses_at_snapshot: 3.9029 (~3.9%)
- kickoff_dayofweek: 1.7947 (~1.8%)
- away_wins_at_snapshot: 1.4427 (~1.4%)
- homeTeamRank: 1.3557 (~1.4%)
- home_wins_at_snapshot: 1.3191 (~1.3%)
- away_losses_at_snapshot: 0.8734 (~0.9%)
- neutralSite: 0.2986 (~0.3%)
- away_wins_at_snapshot_missing: 0.2747 (~0.3%)
- away_losses_at_snapshot_missing: 0.1763 (~0.2%)
- away_last_point_diff_at_snapshot_missing: 0.1694 (~0.2%)
- isRivalry: 0.1664 (~0.2%)
- home_last_point_diff_at_snapshot_missing: 0.1264 (~0.1%)
- home_losses_at_snapshot_missing: 0.0841 (~0.1%)
- homeTeamRank_missing: 0.0648 (~0.1%)
- season_year: 0.0000 (~0.0%)
- isRankedMatchup: 0.0000 (~0.0%)
- season_year_missing: 0.0000 (~0.0%)
- hours_until_game_missing: 0.0000 (~0.0%)
- week_missing: 0.0000 (~0.0%)
- capacity_missing: 0.0000 (~0.0%)
- kickoff_hour_missing: 0.0000 (~0.0%)
- home_wins_at_snapshot_missing: 0.0000 (~0.0%)
- neutralSite_missing: 0.0000 (~0.0%)
- kickoff_dayofweek_missing: 0.0000 (~0.0%)
- isRivalry_missing: 0.0000 (~0.0%)
- isRankedMatchup_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year, isRankedMatchup, season_year_missing, hours_until_game_missing, week_missing, capacity_missing, kickoff_hour_missing, home_wins_at_snapshot_missing, neutralSite_missing, kickoff_dayofweek_missing, isRivalry_missing, isRankedMatchup_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
