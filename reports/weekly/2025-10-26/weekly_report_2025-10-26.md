# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-26

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **4**
- MAE (price): **$38.16**
- RMSE (price): **$58.58**
- Games > 5% price error: **4 / 4**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **252.25 h**  •  Median |Δ|: **200.50 h**
- Within 6h: **1/4**  •  Within 12h: **1/4**  •  Within 24h: **1/4**
- Bias: predictions are on average **188.75 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Wyoming Cowboys vs Colorado State Ram | 2025-10-25 | $168.96 | $54.00 | $114.96 | 212.9% | 2025-10-12 13:00 | 2025-10-24 00:00 | 275.00 |
| Navy Midshipmen vs Florida Atlantic Owl | 2025-10-25 | $42.13 | $26.00 | $16.13 | 62.0% | 2025-10-25 00:00 | 2025-10-19 18:00 | 126.00 |
| Washington State Cougars vs Toledo Rocket | 2025-10-25 | $37.86 | $24.00 | $13.86 | 57.8% | 2025-10-25 01:00 | 2025-10-25 00:00 | 1.00 |
| Purdue Boilermakers vs Rutgers Scarlet Knight | 2025-10-25 | $10.68 | $3.00 | $7.68 | 256.0% | 2025-09-26 11:00 | 2025-10-21 18:00 | 607.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 50% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.