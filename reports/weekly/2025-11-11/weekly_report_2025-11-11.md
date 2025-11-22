# 📈 Weekly Ticket Price Model Report
**Date:** 2025-11-11

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **13**
- MAE (price): **$16.60**
- RMSE (price): **$19.40**
- Games > 5% price error: **12 / 13**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **239.29 h**  •  Median |Δ|: **134.00 h**
- Within 6h: **3/13**  •  Within 12h: **4/13**  •  Within 24h: **4/13**
- Bias: predictions are on average **96.83 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Notre Dame Fighting Irish vs Navy Midshipmen | 2025-11-08 | $81.03 | $41.00 | $40.03 | 97.6% | 2025-11-08 12:00 | 2025-11-08 12:00 | 0.00 |
| Utah State Aggies vs Nevada Wolf Pack | 2025-11-08 | $37.03 | $12.00 | $25.03 | 208.6% | 2025-11-08 11:00 | 2025-11-08 12:00 | 1.00 |
| Mississippi State Bulldogs vs Georgia Bulldog | 2025-11-08 | $48.99 | $24.00 | $24.99 | 104.1% | 2025-11-01 22:00 | 2025-11-07 12:00 | 134.00 |
| Hawaii Rainbow Warriors vs San Diego State Aztec | 2025-11-08 | $52.89 | $29.00 | $23.89 | 82.4% | 2025-10-09 20:00 | 2025-10-13 18:00 | 94.00 |
| Liberty Flames vs Missouri State Bear | 2025-11-08 | $35.92 | $17.00 | $18.92 | 111.3% | 2025-10-14 23:00 | 2025-11-08 12:00 | 589.00 |
| Mississippi Rebels vs The Citadel Bulldog | 2025-11-08 | $24.14 | $6.00 | $18.14 | 302.3% | 2025-11-08 03:00 | 2025-11-08 00:00 | 3.00 |
| Army West Point Black Knights vs Temple Owl | 2025-11-08 | $48.66 | $32.00 | $16.66 | 52.1% | 2025-11-04 18:00 | 2025-10-19 06:00 | 396.00 |
| Middle Tennessee Blue Raiders vs Florida International Panther | 2025-11-08 | $17.50 | $3.00 | $14.50 | 483.3% | 2025-10-10 07:00 | 2025-11-08 12:00 | 701.00 |
| UTEP Miners vs Jacksonville State Gamecock | 2025-11-08 | $18.27 | $6.00 | $12.27 | 204.5% | 2025-11-08 00:00 | 2025-11-08 06:00 | 6.00 |
| New Mexico State Aggies vs Kennesaw State Owl | 2025-11-08 | $25.18 | $17.00 | $8.18 | 48.1% | 2025-11-01 23:00 | 2025-11-05 18:58 | 91.98 |
| UConn Huskies vs Duke Blue Devil | 2025-11-08 | $16.38 | $10.00 | $6.38 | 63.8% | 2025-10-11 23:00 | 2025-10-29 12:00 | 421.00 |
| Oregon State Beavers vs Sam Houston State Bearkat | 2025-11-08 | $12.03 | $6.00 | $6.03 | 100.5% | 2025-10-25 11:00 | 2025-10-03 12:00 | 527.00 |
| Delaware Blue Hens vs Louisiana Tech Bulldog | 2025-11-08 | $19.79 | $19.00 | $0.79 | 4.2% | 2025-10-11 18:00 | 2025-10-17 20:45 | 146.75 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 23% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.