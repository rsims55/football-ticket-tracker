# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-09

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **3**
- MAE (price): **$6.81**
- RMSE (price): **$7.22**
- Games > 5% price error: **3 / 3**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **287.00 h**  •  Median |Δ|: **203.00 h**
- Within 6h: **1/3**  •  Within 12h: **1/3**  •  Within 24h: **1/3**
- Bias: predictions are on average **151.67 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Middle Tennessee Blue Raiders vs Missouri State Bear | 2025-10-08 | $15.21 | $6.00 | $9.21 | 153.5% | 2025-09-09 11:00 | 2025-10-06 18:00 | 655.00 |
| UTEP Miners vs Liberty Flame | 2025-10-08 | $17.70 | $10.00 | $7.70 | 77.0% | 2025-10-08 09:00 | 2025-10-08 12:00 | 3.00 |
| Georgia Southern Eagles vs Southern Miss Golden Eagle | 2025-10-09 | $17.51 | $14.00 | $3.51 | 25.1% | 2025-10-05 23:00 | 2025-09-27 12:00 | 203.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 33% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.