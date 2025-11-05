# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-24

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **1**
- MAE (price): **$3.00**
- RMSE (price): **$3.00**
- Games > 5% price error: **1 / 1**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **306.00 h**  •  Median |Δ|: **306.00 h**
- Within 6h: **0/1**  •  Within 12h: **0/1**  •  Within 24h: **0/1**
- Bias: predictions are on average **306.00 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Georgia State Panthers vs South Alabama Jaguar | 2025-10-23 | $10.00 | $7.00 | $3.00 | 42.9% | 2025-10-23 18:00 | 2025-10-11 00:00 | 306.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 100% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.