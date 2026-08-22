# 📈 Weekly Ticket Price Model Report
**Date:** 2026-03-24

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-03-24 17:53:02**
- Features in model: **16**
- Rows evaluated: **107118**
- gap_pct MAE: **0.2010**
- gap_pct within 0.05: **0.1348**
- Price MAE: **$11.07**
- Price RMSE: **$20.02**
- Price within 5%: **0.0790**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-03-24 16:36:13_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-03-24 17:50:35_) — Snapshot appended (3183 new rows). Total now: 30724
- **model_train**: **success** (_2026-03-24 17:53:02_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-03-24 17:31:49_) — Weekly report generated: reports\weekly\2026-03-24\weekly_report_2026-03-24.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-03-24 17:50:35**
- Model last trained: **2026-03-24 17:53:02**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- homeTeam: 20.8759 (~20.9%)
- hours_until_game: 11.4363 (~11.4%)
- capacity: 10.4360 (~10.4%)
- week: 8.0337 (~8.0%)
- awayTeam: 7.7267 (~7.7%)
- away_last_point_diff_at_snapshot: 7.5307 (~7.5%)
- home_last_point_diff_at_snapshot: 7.5209 (~7.5%)
- kickoff_hour: 7.3860 (~7.4%)
- home_losses_at_snapshot: 4.8430 (~4.8%)
- homeConference: 4.5052 (~4.5%)
- away_wins_at_snapshot: 2.8451 (~2.8%)
- awayConference: 2.5743 (~2.6%)
- kickoff_dayofweek: 2.3575 (~2.4%)
- home_wins_at_snapshot: 1.4070 (~1.4%)
- homeTeamRank: 0.5217 (~0.5%)
- season_year: 0.0000 (~0.0%)

**Possibly unrelated (near-zero importance):** season_year


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
