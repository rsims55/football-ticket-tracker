# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-31

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **3**
- MAE (price): **$7.77**
- RMSE (price): **$10.20**
- Games > 5% price error: **2 / 3**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **698.97 h**  •  Median |Δ|: **252.00 h**
- Within 6h: **0/3**  •  Within 12h: **0/3**  •  Within 24h: **0/3**
- Bias: predictions are on average **384.30 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| UTSA Roadrunners vs Tulane Green Wave | 2025-10-30 | $22.47 | $6.00 | $16.47 | 274.5% | 2025-10-20 14:00 | 2025-10-29 18:00 | 220.00 |
| Rice Owls vs Memphis Tiger | 2025-10-31 | $22.60 | $29.00 | $6.40 | 22.1% | 2025-10-03 00:00 | 2025-10-13 12:00 | 252.00 |
| Coastal Carolina Chanticleers vs Marshall Thundering Herd | 2025-10-30 | $18.45 | $18.00 | $0.45 | 2.5% | 2025-10-30 18:00 | 2025-08-24 01:06 | 1624.90 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 33% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.