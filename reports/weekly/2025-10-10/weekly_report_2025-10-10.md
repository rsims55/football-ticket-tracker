# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-10

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **3**
- MAE (price): **$7.30**
- RMSE (price): **$8.53**
- Games > 5% price error: **3 / 3**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **300.67 h**  •  Median |Δ|: **233.00 h**
- Within 6h: **0/3**  •  Within 12h: **0/3**  •  Within 24h: **1/3**
- Bias: predictions are on average **300.67 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Kennesaw State Owls vs Louisiana Tech Bulldog | 2025-10-09 | $18.49 | $5.00 | $13.49 | 269.8% | 2025-10-08 19:00 | 2025-10-09 18:00 | 23.00 |
| Sam Houston State Bearkats vs Jacksonville State Gamecock | 2025-10-09 | $17.84 | $13.00 | $4.84 | 37.2% | 2025-09-14 13:00 | 2025-09-24 06:00 | 233.00 |
| Tulane Green Wave vs East Carolina Pirate | 2025-10-09 | $5.57 | $2.00 | $3.57 | 178.5% | 2025-09-09 20:00 | 2025-10-06 18:00 | 646.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 0% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.