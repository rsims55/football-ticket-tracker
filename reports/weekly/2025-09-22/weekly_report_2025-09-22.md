# 📈 Weekly Ticket Price Model Report
**Date:** 2025-09-22

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **27**
- MAE (price): **$15.75**
- RMSE (price): **$27.89**
- Games > 5% price error: **22 / 27**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **194.04 h**  •  Median |Δ|: **59.51 h**
- Within 6h: **8/27**  •  Within 12h: **10/27**  •  Within 24h: **11/27**
- Bias: predictions are on average **169.11 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Washington State Cougars vs Washington Huskie | 2025-09-20 | $152.38 | $64.00 | $88.38 | 138.1% | 2025-09-14 14:00 | 2025-09-19 18:00 | 124.00 |
| Nebraska Cornhuskers vs Michigan Wolverine | 2025-09-20 | $159.46 | $83.00 | $76.46 | 92.1% | 2025-08-31 12:00 | 2025-09-19 18:00 | 462.00 |
| Tennessee Volunteers vs UAB Blazer | 2025-09-20 | $82.58 | $34.00 | $48.58 | 142.9% | 2025-09-20 07:00 | 2025-09-20 06:00 | 1.00 |
| Liberty Flames vs James Madison Duke | 2025-09-20 | $101.56 | $53.00 | $48.56 | 91.6% | 2025-08-27 16:00 | 2025-09-20 12:00 | 572.00 |
| Notre Dame Fighting Irish vs Purdue Boilermaker | 2025-09-20 | $79.47 | $47.00 | $32.47 | 69.1% | 2025-09-20 09:00 | 2025-09-20 12:00 | 3.00 |
| Army West Point Black Knights vs North Texas Mean Green | 2025-09-20 | $34.51 | $14.00 | $20.51 | 146.5% | 2025-09-20 10:00 | 2025-09-20 06:00 | 4.00 |
| San Diego State Aztecs vs California Golden Bear | 2025-09-20 | $67.00 | $49.00 | $18.00 | 36.7% | 2025-08-28 21:00 | 2025-09-18 18:00 | 501.00 |
| Texas Longhorns vs Sam Houston State Bearkat | 2025-09-20 | $96.47 | $81.00 | $15.47 | 19.1% | 2025-08-30 19:00 | 2025-09-18 00:00 | 437.00 |
| Kennesaw State Owls vs Arkansas State Red Wolve | 2025-09-20 | $17.03 | $6.00 | $11.03 | 183.8% | 2025-09-08 00:00 | 2025-09-19 18:00 | 282.00 |
| San Jose State Spartans vs Idaho Vandal | 2025-09-20 | $16.09 | $6.00 | $10.09 | 168.2% | 2025-09-20 11:00 | 2025-09-19 18:00 | 17.00 |
| Central Michigan Chippewas vs Wagner Seahawk | 2025-09-20 | $28.99 | $19.00 | $9.99 | 52.6% | 2025-08-29 11:00 | 2025-09-20 12:00 | 529.00 |
| Jacksonville State Gamecocks vs Murray State Racer | 2025-09-20 | $23.56 | $33.00 | $9.44 | 28.6% | 2025-09-15 23:00 | 2025-09-09 18:00 | 149.00 |
| Florida International Panthers vs Delaware Blue Hen | 2025-09-20 | $17.34 | $10.00 | $7.34 | 73.4% | 2025-08-22 03:00 | 2025-09-20 12:00 | 705.00 |
| Georgia Southern Eagles vs Maine Black Bear | 2025-09-20 | $21.74 | $16.00 | $5.74 | 35.9% | 2025-08-22 03:00 | 2025-09-20 18:00 | 711.00 |
| Hawaii Rainbow Warriors vs Fresno State Bulldog | 2025-09-20 | $38.16 | $33.00 | $5.16 | 15.6% | 2025-09-20 17:00 | 2025-09-15 16:30 | 120.49 |
| Air Force Falcons vs Boise State Bronco | 2025-09-20 | $21.00 | $17.00 | $4.00 | 23.5% | 2025-09-03 18:00 | 2025-09-17 12:00 | 330.00 |
| Middle Tennessee Blue Raiders vs Marshall Thundering Herd | 2025-09-20 | $15.66 | $12.00 | $3.66 | 30.5% | 2025-09-18 16:00 | 2025-09-19 18:00 | 26.00 |
| UTEP Miners vs Louisiana Monroe Warhawk | 2025-09-20 | $16.29 | $13.00 | $3.29 | 25.3% | 2025-09-12 08:00 | 2025-09-18 18:00 | 154.00 |
| Louisiana Tech Bulldogs vs Southern Miss Golden Eagle | 2025-09-20 | $22.00 | $19.00 | $3.00 | 15.8% | 2025-09-20 15:00 | 2025-09-20 06:00 | 9.00 |
| LSU Tigers vs Southeastern Louisiana Lion | 2025-09-20 | $14.25 | $12.00 | $2.25 | 18.8% | 2025-08-29 14:00 | 2025-08-29 12:00 | 2.00 |
| Ohio Bobcats vs Gardner-Webb Runnin' Bulldog | 2025-09-20 | $3.00 | $2.00 | $1.00 | 50.0% | 2025-09-13 05:00 | 2025-09-15 16:30 | 59.51 |
| Vanderbilt Commodores vs Georgia State Panther | 2025-09-20 | $9.59 | $9.00 | $0.59 | 6.6% | 2025-09-10 00:00 | 2025-09-10 00:00 | 0.00 |
| UConn Huskies vs Ball State Cardinal | 2025-09-20 | $8.15 | $8.00 | $0.15 | 1.9% | 2025-08-29 03:00 | 2025-08-29 00:00 | 3.00 |
| South Florida Bulls vs South Carolina State Bulldog | 2025-09-20 | $10.14 | $10.00 | $0.14 | 1.4% | 2025-08-27 23:00 | 2025-08-28 06:00 | 7.00 |
| Mississippi State Bulldogs vs Northern Illinois Huskie | 2025-09-20 | $6.00 | $6.00 | $0.00 | 0.0% | 2025-09-02 14:00 | 2025-09-01 12:00 | 26.00 |
| Western Kentucky Hilltoppers vs Nevada Wolf Pack | 2025-09-20 | $16.00 | $16.00 | $0.00 | 0.0% | 2025-09-20 17:00 | 2025-09-20 12:00 | 5.00 |
| Texas State Bobcats vs Nicholls State Colonel | 2025-09-20 | $16.00 | $16.00 | $0.00 | 0.0% | 2025-09-07 18:00 | 2025-09-07 18:00 | 0.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 37% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.