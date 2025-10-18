# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-13

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **10**
- MAE (price): **$18.16**
- RMSE (price): **$24.07**
- Games > 5% price error: **10 / 10**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **274.60 h**  •  Median |Δ|: **164.00 h**
- Within 6h: **1/10**  •  Within 12h: **1/10**  •  Within 24h: **3/10**
- Bias: predictions are on average **44.60 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Texas Longhorns vs Oklahoma Sooner | 2025-10-11 | $296.55 | $235.00 | $61.55 | 26.2% | 2025-09-28 02:00 | 2025-10-10 18:00 | 304.00 |
| Notre Dame Fighting Irish vs North Carolina State Wolfpack | 2025-10-11 | $50.72 | $26.00 | $24.72 | 95.1% | 2025-10-03 20:00 | 2025-10-11 12:00 | 184.00 |
| Michigan State Spartans vs UCLA Bruin | 2025-10-11 | $69.04 | $47.00 | $22.04 | 46.9% | 2025-10-11 09:00 | 2025-10-09 18:00 | 39.00 |
| Mississippi Rebels vs Washington State Cougar | 2025-10-11 | $34.84 | $18.00 | $16.84 | 93.6% | 2025-10-11 08:00 | 2025-10-10 18:00 | 14.00 |
| Army West Point Black Knights vs Charlotte 49er | 2025-10-11 | $49.42 | $66.00 | $16.58 | 25.1% | 2025-10-04 22:00 | 2025-08-30 18:00 | 844.00 |
| Oregon State Beavers vs Wake Forest Demon Deacon | 2025-10-11 | $16.27 | $5.00 | $11.27 | 225.4% | 2025-09-12 23:00 | 2025-10-10 12:00 | 661.00 |
| Nevada Wolf Pack vs San Diego State Aztec | 2025-10-11 | $20.97 | $11.00 | $9.97 | 90.6% | 2025-09-20 23:00 | 2025-08-29 18:00 | 533.00 |
| UNLV Rebels vs Air Force Falcon | 2025-10-11 | $26.39 | $18.00 | $8.39 | 46.6% | 2025-09-11 12:00 | 2025-09-05 12:00 | 144.00 |
| Wyoming Cowboys vs San Jose State Spartan | 2025-10-11 | $30.93 | $25.00 | $5.93 | 23.7% | 2025-10-11 16:00 | 2025-10-10 18:00 | 22.00 |
| Hawaii Rainbow Warriors vs Utah State Aggie | 2025-10-11 | $30.34 | $26.00 | $4.34 | 16.7% | 2025-10-11 11:00 | 2025-10-11 12:00 | 1.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 60% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.