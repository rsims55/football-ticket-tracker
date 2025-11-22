# 📈 Weekly Ticket Price Model Report
**Date:** 2025-11-06

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **2**
- MAE (price): **$8.16**
- RMSE (price): **$8.22**
- Games > 5% price error: **2 / 2**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **614.00 h**  •  Median |Δ|: **614.00 h**
- Within 6h: **0/2**  •  Within 12h: **0/2**  •  Within 24h: **0/2**
- Bias: predictions are on average **614.00 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| South Florida Bulls vs UTSA Roadrunner | 2025-11-06 | $19.07 | $10.00 | $9.07 | 90.7% | 2025-10-09 11:00 | 2025-08-28 18:00 | 1001.00 |
| Appalachian State Mountaineers vs Georgia Southern Eagle | 2025-11-06 | $45.26 | $38.00 | $7.26 | 19.1% | 2025-11-05 23:00 | 2025-10-27 12:00 | 227.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 100% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.