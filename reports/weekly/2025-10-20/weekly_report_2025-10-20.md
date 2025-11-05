# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-20

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **4**
- MAE (price): **$66.08**
- RMSE (price): **$83.87**
- Games > 5% price error: **4 / 4**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **552.62 h**  •  Median |Δ|: **629.88 h**
- Within 6h: **0/4**  •  Within 12h: **0/4**  •  Within 24h: **1/4**
- Bias: predictions are on average **84.13 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Notre Dame Fighting Irish vs USC Trojan | 2025-10-18 | $190.10 | $45.00 | $145.10 | 322.4% | 2025-10-18 13:00 | 2025-09-09 12:00 | 937.00 |
| Stanford Cardinal vs Florida State Seminole | 2025-10-18 | $95.91 | $25.00 | $70.91 | 283.6% | 2025-10-17 07:00 | 2025-10-17 20:45 | 13.75 |
| Virginia Cavaliers vs Washington State Cougar | 2025-10-18 | $66.26 | $21.00 | $45.26 | 215.5% | 2025-09-19 09:00 | 2025-10-17 20:45 | 683.75 |
| Air Force Falcons vs Wyoming Cowboy | 2025-10-18 | $21.04 | $18.00 | $3.04 | 16.9% | 2025-09-24 00:00 | 2025-10-18 00:00 | 576.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 25% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.