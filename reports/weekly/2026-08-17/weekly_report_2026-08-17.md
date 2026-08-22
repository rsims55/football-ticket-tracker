# 📈 Weekly Ticket Price Model Report
**Date:** 2026-08-17

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-08-17 17:20:26**
- Features in model: **19**
- Rows evaluated: **566452**
- gap_pct MAE: **0.1911**
- gap_pct within 0.05: **0.1252**
- Price MAE: **$14.24**
- Price RMSE: **$24.26**
- Price within 5%: **0.0856**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-08-17 15:35:23_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-08-17 17:11:48_) — Snapshot appended (3215 new rows). Total now: 482528
- **model_train**: **success** (_2026-08-17 17:20:26_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-08-17 08:15:08_) — Weekly report generated: reports\weekly\2026-08-17\weekly_report_2026-08-17.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed
- **predict_price**: **success** (_2026-08-17 17:20:52_) — 887 predictions generated for 2026

## 🗓️ Season State & Data Freshness

- Season state: **In-season**
- Snapshots last updated: **2026-08-17 17:11:48**
- Model last trained: **2026-08-17 17:20:26**
- Predictions last updated: **2026-08-17 17:20:52**
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 29.8969 (~29.9%)
- homeTeam: 18.3242 (~18.3%)
- awayTeam: 14.2447 (~14.2%)
- week: 7.5699 (~7.6%)
- capacity: 6.0256 (~6.0%)
- home_last_point_diff_at_snapshot: 4.4437 (~4.4%)
- awayConference: 3.8356 (~3.8%)
- kickoff_hour: 3.3921 (~3.4%)
- homeConference: 2.7418 (~2.7%)
- away_last_point_diff_at_snapshot: 2.1732 (~2.2%)
- home_losses_at_snapshot: 2.1283 (~2.1%)
- season_year: 1.6985 (~1.7%)
- home_wins_at_snapshot: 1.3775 (~1.4%)
- homeTeamRank: 1.1121 (~1.1%)
- kickoff_dayofweek: 0.8271 (~0.8%)
- away_last_point_diff_at_snapshot_missing: 0.2088 (~0.2%)
- awayTeamRank: 0.0000 (~0.0%)
- awayTeamRank_missing: 0.0000 (~0.0%)
- homeTeamRank_missing: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** awayTeamRank, awayTeamRank_missing, homeTeamRank_missing


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
