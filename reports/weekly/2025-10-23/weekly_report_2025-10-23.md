# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-23

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **2**
- MAE (price): **$10.14**
- RMSE (price): **$10.32**
- Games > 5% price error: **2 / 2**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **347.50 h**  •  Median |Δ|: **347.50 h**
- Within 6h: **0/2**  •  Within 12h: **0/2**  •  Within 24h: **0/2**
- Bias: predictions are on average **318.50 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Delaware Blue Hens vs Middle Tennessee Blue Raider | 2025-10-22 | $17.08 | $5.00 | $12.08 | 241.6% | 2025-09-25 00:00 | 2025-10-22 18:00 | 666.00 |
| New Mexico State Aggies vs Missouri State Bear | 2025-10-22 | $25.20 | $17.00 | $8.20 | 48.2% | 2025-10-15 23:00 | 2025-10-14 18:00 | 29.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 50% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.