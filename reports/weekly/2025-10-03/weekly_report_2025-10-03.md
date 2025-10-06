# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-03

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **13**
- MAE (price): **$22.67**
- RMSE (price): **$43.16**
- Games > 5% price error: **10 / 13**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **288.82 h**  •  Median |Δ|: **309.00 h**
- Within 6h: **4/13**  •  Within 12h: **5/13**  •  Within 24h: **5/13**
- Bias: predictions are on average **184.11 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Georgia Bulldogs vs Alabama Crimson Tide | 2025-09-27 | $468.51 | $337.00 | $131.51 | 39.0% | 2025-09-12 00:00 | 2025-09-26 12:00 | 348.00 |
| Arkansas Razorbacks vs Notre Dame Fighting Irish | 2025-09-27 | $117.89 | $49.00 | $68.89 | 140.6% | 2025-08-30 15:00 | 2025-09-25 12:00 | 621.00 |
| Penn State Nittany Lions vs Oregon Duck | 2025-09-27 | $410.63 | $370.00 | $40.63 | 11.0% | 2025-09-01 10:00 | 2025-09-24 06:00 | 548.00 |
| Navy Midshipmen vs Rice Owl | 2025-09-27 | $23.97 | $10.00 | $13.97 | 139.7% | 2025-09-02 21:00 | 2025-09-27 12:00 | 591.00 |
| Missouri State Bears vs Western Kentucky Hilltopper | 2025-09-27 | $25.46 | $12.00 | $13.46 | 112.2% | 2025-09-26 14:00 | 2025-09-26 18:00 | 4.00 |
| Kennesaw State Owls vs Middle Tennessee Blue Raider | 2025-09-27 | $20.76 | $12.00 | $8.76 | 73.0% | 2025-09-26 19:00 | 2025-09-26 12:00 | 7.00 |
| UTEP Miners vs Louisiana Tech Bulldog | 2025-09-27 | $16.58 | $11.00 | $5.58 | 50.7% | 2025-09-27 10:00 | 2025-09-25 18:00 | 40.00 |
| Miami (OH) RedHawks vs Lindenwood Lion | 2025-09-27 | $22.92 | $18.00 | $4.92 | 27.3% | 2025-09-27 09:00 | 2025-09-27 06:00 | 3.00 |
| Colorado Buffaloes vs BYU Cougar | 2025-09-27 | $42.18 | $46.00 | $3.82 | 8.3% | 2025-09-14 15:00 | 2025-09-27 12:00 | 309.00 |
| Western Michigan Broncos vs Rhode Island Ram | 2025-09-27 | $16.06 | $14.00 | $2.06 | 14.7% | 2025-08-28 19:00 | 2025-09-25 00:00 | 653.00 |
| Colorado State Rams vs Washington State Cougar | 2025-09-27 | $23.06 | $24.00 | $0.94 | 3.9% | 2025-09-27 14:00 | 2025-09-01 12:00 | 626.00 |
| New Mexico State Aggies vs Sam Houston State Bearkat | 2025-10-02 | $23.93 | $24.00 | $0.07 | 0.3% | 2025-09-13 12:00 | 2025-09-13 10:23 | 1.61 |
| Vanderbilt Commodores vs Utah State Aggie | 2025-09-27 | $8.04 | $8.00 | $0.04 | 0.5% | 2025-09-05 15:00 | 2025-09-05 12:00 | 3.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 46% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.