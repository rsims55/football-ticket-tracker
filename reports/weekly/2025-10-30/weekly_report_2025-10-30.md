# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-30

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **2**
- MAE (price): **$10.04**
- RMSE (price): **$10.95**
- Games > 5% price error: **2 / 2**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **368.00 h**  •  Median |Δ|: **368.00 h**
- Within 6h: **0/2**  •  Within 12h: **0/2**  •  Within 24h: **0/2**
- Bias: predictions are on average **330.00 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Middle Tennessee Blue Raiders vs Jacksonville State Gamecock | 2025-10-29 | $17.40 | $3.00 | $14.40 | 480.0% | 2025-09-30 10:00 | 2025-10-29 12:00 | 698.00 |
| Missouri State Bears vs Florida International Panther | 2025-10-29 | $23.68 | $18.00 | $5.68 | 31.6% | 2025-10-28 14:00 | 2025-10-27 00:00 | 38.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 50% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.