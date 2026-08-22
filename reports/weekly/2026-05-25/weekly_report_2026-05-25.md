# 📈 Weekly Ticket Price Model Report
**Date:** 2026-05-25

## 🧠 Latest CatBoost Training Summary

- Model last trained: **2026-05-25 06:49:03**
- Features in model: **12**
- Rows evaluated: **327990**
- gap_pct MAE: **0.1989**
- gap_pct within 0.05: **0.0998**
- Price MAE: **$16.86**
- Price RMSE: **$25.30**
- Price within 5%: **0.0788**
- Timing MAE: **255.9 h**  •  Median |Δ|: **121.5 h**
- Within 6h: **145/798**  •  Within 24h: **235/798**
- Bias: predictions avg **58.3 h later** than actual low
## ✅ Pipeline Status (Latest)

- **weekly_update**: **success** (_2026-05-20 05:30:09_) — Weekly update complete for 2026
- **daily_snapshot**: **success** (_2026-05-25 06:40:11_) — Snapshot appended (711 new rows). Total now: 242150
- **model_train**: **success** (_2026-05-25 06:49:03_) — CatBoost training complete for 2026
- **weekly_report**: **success** (_2026-05-18 08:15:07_) — Weekly report generated: reports\weekly\2026-05-18\weekly_report_2026-05-18.md
- **health_check**: **success** (_2026-03-07 19:21:04_) — Health check passed

## 🗓️ Season State & Data Freshness

- Season state: **Offseason**
- Snapshots last updated: **2026-05-25 06:40:11**
- Model last trained: **2026-05-25 06:49:03**
- Predictions last updated: **2026-02-01 19:21:39** _(stale — model retrained since last prediction run; regenerate predictions)_
- Postseason games are **excluded** from model + GUI (for now).

## 🔍 Best Predictors of Ticket Price

- hours_until_game: 20.6167 (~20.6%)
- homeTeam: 17.9672 (~18.0%)
- week: 10.8867 (~10.9%)
- awayTeam: 10.4007 (~10.4%)
- home_last_point_diff_at_snapshot: 9.5948 (~9.6%)
- capacity: 9.5704 (~9.6%)
- homeConference: 5.8657 (~5.9%)
- kickoff_hour: 5.2855 (~5.3%)
- home_losses_at_snapshot: 4.9809 (~5.0%)
- homeTeamRank: 2.1582 (~2.2%)
- kickoff_dayofweek: 1.4368 (~1.4%)
- season_year: 1.2365 (~1.2%)


## 📊 Model Accuracy (Past 7 Days)
No games to evaluate in the past week.
