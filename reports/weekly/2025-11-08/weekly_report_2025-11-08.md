# 📈 Weekly Ticket Price Model Report
**Date:** 2025-11-08

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **32**
- MAE (price): **$19.17**
- RMSE (price): **$26.87**
- Games > 5% price error: **31 / 32**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **486.50 h**  •  Median |Δ|: **330.50 h**
- Within 6h: **3/32**  •  Within 12h: **3/32**  •  Within 24h: **4/32**
- Bias: predictions are on average **298.56 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Penn State Nittany Lions vs Indiana Hoosier | 2025-11-08 | $93.09 | $23.00 | $70.09 | 304.7% | 2025-10-15 15:00 | 2025-10-31 18:00 | 387.00 |
| Alabama Crimson Tide vs LSU Tiger | 2025-11-08 | $235.44 | $168.00 | $67.44 | 40.1% | 2025-11-06 07:00 | 2025-10-09 18:00 | 661.00 |
| Clemson Tigers vs Florida State Seminole | 2025-11-08 | $118.32 | $66.00 | $52.32 | 79.3% | 2025-11-07 12:00 | 2025-10-25 18:00 | 306.00 |
| Missouri Tigers vs Texas A&M Aggie | 2025-11-08 | $165.49 | $116.00 | $49.49 | 42.7% | 2025-11-01 22:00 | 2025-08-31 12:00 | 1498.00 |
| Kentucky Wildcats vs Florida Gator | 2025-11-08 | $78.22 | $39.00 | $39.22 | 100.6% | 2025-10-14 11:00 | 2025-11-01 00:00 | 421.00 |
| North Carolina Tar Heels vs Stanford Cardinal | 2025-11-08 | $68.87 | $31.00 | $37.87 | 122.2% | 2025-10-17 00:00 | 2025-10-26 12:00 | 228.00 |
| Iowa Hawkeyes vs Oregon Duck | 2025-11-08 | $103.45 | $68.00 | $35.45 | 52.1% | 2025-11-07 18:00 | 2025-10-21 18:00 | 408.00 |
| Virginia Cavaliers vs Wake Forest Demon Deacon | 2025-11-08 | $59.36 | $28.00 | $31.36 | 112.0% | 2025-11-03 09:00 | 2025-09-15 16:30 | 1168.49 |
| Purdue Boilermakers vs Ohio State Buckeye | 2025-11-08 | $81.96 | $55.00 | $26.96 | 49.0% | 2025-11-08 00:00 | 2025-10-25 00:00 | 336.00 |
| West Virginia Mountaineers vs Colorado Buffaloe | 2025-11-08 | $70.34 | $46.00 | $24.34 | 52.9% | 2025-10-30 18:00 | 2025-10-27 12:00 | 78.00 |
| Rutgers Scarlet Knights vs Maryland Terrapin | 2025-11-08 | $36.65 | $19.00 | $17.65 | 92.9% | 2025-11-07 10:00 | 2025-10-27 12:00 | 262.00 |
| UCLA Bruins vs Nebraska Cornhusker | 2025-11-08 | $77.27 | $62.00 | $15.27 | 24.6% | 2025-11-01 13:00 | 2025-10-31 18:00 | 19.00 |
| Memphis Tigers vs Tulane Green Wave | 2025-11-07 | $29.09 | $14.00 | $15.09 | 107.8% | 2025-10-11 11:00 | 2025-10-25 00:00 | 325.00 |
| Wisconsin Badgers vs Washington Huskie | 2025-11-08 | $23.75 | $10.00 | $13.75 | 137.5% | 2025-10-16 10:00 | 2025-10-29 00:00 | 302.00 |
| Rice Owls vs UAB Blazer | 2025-11-08 | $19.83 | $8.00 | $11.83 | 147.9% | 2025-10-11 00:00 | 2025-11-06 12:00 | 636.00 |
| Boston College Eagles vs Southern Methodist (SMU) Mustang | 2025-11-08 | $15.11 | $4.00 | $11.11 | 277.8% | 2025-10-09 18:00 | 2025-10-28 12:00 | 450.00 |
| TCU Horned Frogs vs Iowa State Cyclone | 2025-11-08 | $69.86 | $59.00 | $10.86 | 18.4% | 2025-10-17 00:00 | 2025-10-27 12:00 | 252.00 |
| East Carolina Pirates vs Charlotte 49er | 2025-11-08 | $11.88 | $22.00 | $10.12 | 46.0% | 2025-10-25 07:00 | 2025-09-13 10:23 | 1004.61 |
| Louisiana Lafayette Ragin Cajuns vs Texas State Bobcat | 2025-11-08 | $30.62 | $39.00 | $8.38 | 21.5% | 2025-11-07 06:00 | 2025-10-27 12:00 | 258.00 |
| Miami Hurricanes vs Syracuse Orange | 2025-11-08 | $24.93 | $17.00 | $7.93 | 46.6% | 2025-10-22 18:00 | 2025-10-23 00:00 | 6.00 |
| Florida Atlantic Owls vs Tulsa Golden Hurricane | 2025-11-08 | $11.88 | $4.00 | $7.88 | 197.0% | 2025-10-25 07:00 | 2025-10-25 06:00 | 1.00 |
| Arizona Wildcats vs Kansas Jayhawk | 2025-11-08 | $23.83 | $16.00 | $7.83 | 48.9% | 2025-11-07 06:00 | 2025-10-18 12:00 | 474.00 |
| USC Trojans vs Northwestern Wildcat | 2025-11-07 | $35.51 | $28.00 | $7.51 | 26.8% | 2025-10-23 00:00 | 2025-10-21 12:00 | 36.00 |
| Marshall Thundering Herd vs James Madison Duke | 2025-11-08 | $23.11 | $16.00 | $7.11 | 44.4% | 2025-11-06 10:00 | 2025-10-27 12:00 | 238.00 |
| Vanderbilt Commodores vs Auburn Tiger | 2025-11-08 | $112.04 | $106.00 | $6.04 | 5.7% | 2025-11-07 18:00 | 2025-09-07 12:00 | 1470.00 |
| Coastal Carolina Chanticleers vs Georgia State Panther | 2025-11-08 | $18.68 | $13.00 | $5.68 | 43.7% | 2025-11-07 18:00 | 2025-10-24 18:00 | 336.00 |
| Texas Tech Red Raiders vs BYU Cougar | 2025-11-08 | $79.49 | $75.00 | $4.49 | 6.0% | 2025-10-30 18:00 | 2025-09-01 12:00 | 1422.00 |
| Louisville Cardinals vs California Golden Bear | 2025-11-08 | $6.88 | $3.00 | $3.88 | 129.3% | 2025-10-15 00:00 | 2025-08-30 06:00 | 1098.00 |
| Arkansas State Red Wolves vs Southern Miss Golden Eagle | 2025-11-08 | $15.69 | $13.00 | $2.69 | 20.7% | 2025-10-14 00:00 | 2025-09-01 12:00 | 1020.00 |
| Colorado State Rams vs UNLV Rebel | 2025-11-08 | $24.75 | $23.00 | $1.75 | 7.6% | 2025-11-07 19:00 | 2025-10-28 06:00 | 253.00 |
| San Jose State Spartans vs Air Force Falcon | 2025-11-08 | $17.24 | $16.00 | $1.24 | 7.7% | 2025-11-08 02:00 | 2025-10-30 06:00 | 212.00 |
| Eastern Michigan Eagles vs Bowling Green State Falcon | 2025-11-08 | $22.68 | $22.00 | $0.68 | 3.1% | 2025-10-26 08:00 | 2025-10-26 06:00 | 2.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 72% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.