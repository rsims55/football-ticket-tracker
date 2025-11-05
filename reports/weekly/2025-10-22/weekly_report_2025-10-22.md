# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-22

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **2**
- MAE (price): **$9.05**
- RMSE (price): **$9.20**
- Games > 5% price error: **2 / 2**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **696.50 h**  •  Median |Δ|: **696.50 h**
- Within 6h: **0/2**  •  Within 12h: **0/2**  •  Within 24h: **0/2**
- Bias: predictions are on average **696.50 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Louisiana Tech Bulldogs vs Western Kentucky Hilltopper | 2025-10-21 | $17.73 | $7.00 | $10.73 | 153.3% | 2025-09-21 20:00 | 2025-10-21 12:00 | 712.00 |
| Florida International Panthers vs Kennesaw State Owl | 2025-10-21 | $13.37 | $6.00 | $7.37 | 122.8% | 2025-09-22 03:00 | 2025-10-20 12:00 | 681.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 0% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.