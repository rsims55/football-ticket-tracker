# 📈 Weekly Ticket Price Model Report
**Date:** 2025-11-01

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **37**
- MAE (price): **$26.90**
- RMSE (price): **$45.21**
- Games > 5% price error: **36 / 37**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **326.52 h**  •  Median |Δ|: **282.00 h**
- Within 6h: **1/37**  •  Within 12h: **1/37**  •  Within 24h: **2/37**
- Bias: predictions are on average **72.50 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Ohio State Buckeyes vs Penn State Nittany Lion | 2025-11-01 | $292.89 | $115.00 | $177.89 | 154.7% | 2025-10-31 00:00 | 2025-10-20 12:00 | 252.00 |
| Tennessee Volunteers vs Oklahoma Sooner | 2025-11-01 | $381.74 | $234.00 | $147.74 | 63.1% | 2025-10-18 03:00 | 2025-10-24 18:00 | 159.00 |
| Boston College Eagles vs Notre Dame Fighting Irish | 2025-11-01 | $203.76 | $128.00 | $75.76 | 59.2% | 2025-10-31 16:00 | 2025-10-24 06:00 | 178.00 |
| Utah Utes vs Cincinnati Bearcat | 2025-11-01 | $104.21 | $52.00 | $52.21 | 100.4% | 2025-10-02 06:00 | 2025-10-23 12:00 | 510.00 |
| Illinois Fighting Illini vs Rutgers Scarlet Knight | 2025-11-01 | $75.16 | $23.00 | $52.16 | 226.8% | 2025-10-03 09:00 | 2025-08-29 18:00 | 831.00 |
| Florida State Seminoles vs Wake Forest Demon Deacon | 2025-11-01 | $52.82 | $8.00 | $44.82 | 560.2% | 2025-10-10 18:00 | 2025-10-19 06:00 | 204.00 |
| Kansas Jayhawks vs Oklahoma State Cowboy | 2025-11-01 | $72.64 | $36.00 | $36.64 | 101.8% | 2025-10-02 07:00 | 2025-10-17 20:45 | 373.75 |
| Kansas State Wildcats vs Texas Tech Red Raider | 2025-11-01 | $79.43 | $46.00 | $33.43 | 72.7% | 2025-10-18 23:00 | 2025-10-15 00:00 | 95.00 |
| Maryland Terrapins vs Indiana Hoosier | 2025-11-01 | $38.90 | $12.00 | $26.90 | 224.2% | 2025-10-07 10:00 | 2025-09-20 18:00 | 400.00 |
| Clemson Tigers vs Duke Blue Devil | 2025-11-01 | $44.71 | $19.00 | $25.71 | 135.3% | 2025-10-25 21:00 | 2025-10-18 12:00 | 177.00 |
| Iowa State Cyclones vs Arizona State Sun Devil | 2025-11-01 | $62.34 | $38.00 | $24.34 | 64.1% | 2025-10-23 07:00 | 2025-10-26 18:00 | 83.00 |
| Mississippi Rebels vs South Carolina Gamecock | 2025-11-01 | $103.13 | $79.00 | $24.13 | 30.5% | 2025-10-08 18:00 | 2025-10-23 00:00 | 342.00 |
| North Carolina State Wolfpack vs Georgia Tech Yellow Jacket | 2025-11-01 | $74.32 | $54.00 | $20.32 | 37.6% | 2025-10-21 01:00 | 2025-10-20 12:00 | 13.00 |
| Minnesota Golden Gophers vs Michigan State Spartan | 2025-11-01 | $52.04 | $32.00 | $20.04 | 62.6% | 2025-10-31 12:00 | 2025-10-19 18:00 | 282.00 |
| Boise State Broncos vs Fresno State Bulldog | 2025-11-01 | $64.33 | $45.00 | $19.33 | 43.0% | 2025-10-04 15:00 | 2025-08-24 01:06 | 997.90 |
| Syracuse Orange vs North Carolina Tar Heel | 2025-10-31 | $20.60 | $2.00 | $18.60 | 930.0% | 2025-10-01 20:00 | 2025-10-20 18:00 | 454.00 |
| Michigan Wolverines vs Purdue Boilermaker | 2025-11-01 | $70.57 | $52.00 | $18.57 | 35.7% | 2025-10-30 15:00 | 2025-10-20 12:00 | 243.00 |
| Virginia Tech Hokies vs Louisville Cardinal | 2025-11-01 | $26.54 | $10.00 | $16.54 | 165.4% | 2025-10-20 18:00 | 2025-10-20 12:00 | 6.00 |
| Nebraska Cornhuskers vs USC Trojan | 2025-11-01 | $76.73 | $92.00 | $15.27 | 16.6% | 2025-10-25 21:00 | 2025-10-20 12:00 | 129.00 |
| Arkansas Razorbacks vs Mississippi State Bulldog | 2025-11-01 | $28.74 | $14.00 | $14.74 | 105.3% | 2025-10-29 11:00 | 2025-10-18 12:00 | 263.00 |
| South Alabama Jaguars vs Louisiana Lafayette Ragin Cajun | 2025-11-01 | $34.66 | $20.00 | $14.66 | 73.3% | 2025-10-04 12:00 | 2025-10-23 12:00 | 456.00 |
| Southern Methodist (SMU) Mustangs vs Miami Hurricane | 2025-11-01 | $113.35 | $99.00 | $14.35 | 14.5% | 2025-10-31 16:00 | 2025-10-03 12:00 | 676.00 |
| Baylor Bears vs UCF Knight | 2025-11-01 | $35.46 | $22.00 | $13.46 | 61.2% | 2025-10-24 10:00 | 2025-10-07 06:00 | 412.00 |
| Georgia Bulldogs vs Florida Gator | 2025-11-01 | $133.48 | $121.00 | $12.48 | 10.3% | 2025-10-31 12:00 | 2025-10-20 12:00 | 264.00 |
| Houston Cougars vs West Virginia Mountaineer | 2025-11-01 | $28.07 | $16.00 | $12.07 | 75.4% | 2025-10-06 10:00 | 2025-10-27 12:00 | 506.00 |
| Colorado Buffaloes vs Arizona Wildcat | 2025-11-01 | $79.53 | $69.00 | $10.53 | 15.3% | 2025-10-18 23:00 | 2025-10-15 00:00 | 95.00 |
| California Golden Bears vs Virginia Cavalier | 2025-11-01 | $32.52 | $24.00 | $8.52 | 35.5% | 2025-10-21 05:00 | 2025-10-15 00:00 | 149.00 |
| Temple Owls vs East Carolina Pirate | 2025-11-01 | $11.07 | $3.00 | $8.07 | 269.0% | 2025-10-03 00:00 | 2025-10-17 20:45 | 356.75 |
| Bowling Green State Falcons vs Buffalo Bull | 2025-11-01 | $5.26 | $13.00 | $7.74 | 59.5% | 2025-10-16 12:00 | 2025-10-17 20:45 | 32.75 |
| Louisiana Tech Bulldogs vs Sam Houston State Bearkat | 2025-10-31 | $20.44 | $14.00 | $6.44 | 46.0% | 2025-10-01 21:00 | 2025-10-31 18:00 | 717.00 |
| Western Michigan Broncos vs Central Michigan Chippewa | 2025-11-01 | $23.98 | $18.00 | $5.98 | 33.2% | 2025-10-19 10:00 | 2025-10-03 12:00 | 382.00 |
| Auburn Tigers vs Kentucky Wildcat | 2025-11-01 | $47.29 | $42.00 | $5.29 | 12.6% | 2025-10-31 23:00 | 2025-10-18 18:00 | 317.00 |
| Louisiana Monroe Warhawks vs Old Dominion Monarch | 2025-11-01 | $21.06 | $17.00 | $4.06 | 23.9% | 2025-10-15 05:00 | 2025-10-06 08:03 | 212.94 |
| Stanford Cardinal vs Pittsburgh Panther | 2025-11-01 | $6.81 | $4.00 | $2.81 | 70.2% | 2025-10-25 20:00 | 2025-10-13 12:00 | 296.00 |
| UNLV Rebels vs New Mexico Lobo | 2025-11-01 | $30.54 | $29.00 | $1.54 | 5.3% | 2025-10-28 18:00 | 2025-10-20 12:00 | 198.00 |
| Troy Trojans vs Arkansas State Red Wolve | 2025-11-01 | $14.96 | $16.00 | $1.04 | 6.5% | 2025-10-03 11:00 | 2025-10-24 12:00 | 505.00 |
| North Texas Mean Green vs Navy Midshipmen | 2025-11-01 | $41.06 | $42.00 | $0.94 | 2.2% | 2025-10-31 15:00 | 2025-10-10 06:00 | 513.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 65% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.