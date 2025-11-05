# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-29

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **2**
- MAE (price): **$13.30**
- RMSE (price): **$13.33**
- Games > 5% price error: **2 / 2**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **17.00 h**  •  Median |Δ|: **17.00 h**
- Within 6h: **1/2**  •  Within 12h: **1/2**  •  Within 24h: **1/2**
- Bias: predictions are on average **12.00 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Texas State Bobcats vs James Madison Duke | 2025-10-28 | $16.25 | $2.00 | $14.25 | 712.5% | 2025-10-28 17:00 | 2025-10-27 12:00 | 29.00 |
| Kennesaw State Owls vs UTEP Miner | 2025-10-28 | $18.35 | $6.00 | $12.35 | 205.8% | 2025-10-27 13:00 | 2025-10-27 18:00 | 5.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 50% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.