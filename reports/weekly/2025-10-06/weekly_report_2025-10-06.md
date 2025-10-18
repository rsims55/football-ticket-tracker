# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-06

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **51**
- MAE (price): **$26.87**
- RMSE (price): **$56.76**
- Games > 5% price error: **46 / 51**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **262.58 h**  •  Median |Δ|: **170.00 h**
- Within 6h: **10/51**  •  Within 12h: **12/51**  •  Within 24h: **13/51**
- Bias: predictions are on average **91.19 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Alabama Crimson Tide vs Vanderbilt Commodore | 2025-10-04 | $350.74 | $60.00 | $290.74 | 484.6% | 2025-09-17 00:00 | 2025-10-02 20:24 | 380.41 |
| Florida State Seminoles vs Miami Hurricane | 2025-10-04 | $302.59 | $136.00 | $166.59 | 122.5% | 2025-10-01 21:00 | 2025-08-29 18:00 | 795.00 |
| Florida Gators vs Texas Longhorn | 2025-10-04 | $218.91 | $76.00 | $142.91 | 188.0% | 2025-10-04 02:00 | 2025-10-03 12:00 | 14.00 |
| Georgia Bulldogs vs Kentucky Wildcat | 2025-10-04 | $148.03 | $52.00 | $96.03 | 184.7% | 2025-09-11 00:00 | 2025-10-02 20:24 | 524.41 |
| UCLA Bruins vs Penn State Nittany Lion | 2025-10-04 | $98.53 | $30.00 | $68.53 | 228.4% | 2025-09-12 18:00 | 2025-10-02 20:24 | 482.41 |
| North Carolina State Wolfpack vs Campbell Fighting Camel | 2025-10-04 | $101.21 | $36.00 | $65.21 | 181.1% | 2025-09-10 00:00 | 2025-10-03 12:00 | 564.00 |
| Appalachian State Mountaineers vs Oregon State Beaver | 2025-10-04 | $133.45 | $86.00 | $47.45 | 55.2% | 2025-10-03 23:00 | 2025-10-03 12:00 | 11.00 |
| Texas A&M Aggies vs Mississippi State Bulldog | 2025-10-04 | $117.20 | $78.00 | $39.20 | 50.3% | 2025-09-11 15:00 | 2025-10-03 12:00 | 525.00 |
| Michigan Wolverines vs Wisconsin Badger | 2025-10-04 | $123.65 | $87.00 | $36.65 | 42.1% | 2025-10-02 11:00 | 2025-10-03 12:00 | 25.00 |
| Oklahoma Sooners vs Kent State Golden Flashe | 2025-10-04 | $52.31 | $19.00 | $33.31 | 175.3% | 2025-09-04 16:00 | 2025-09-01 12:00 | 76.00 |
| TCU Horned Frogs vs Colorado Buffaloe | 2025-10-04 | $83.04 | $50.00 | $33.04 | 66.1% | 2025-09-10 11:00 | 2025-10-03 12:00 | 553.00 |
| San Diego State Aztecs vs Colorado State Ram | 2025-10-03 | $14.15 | $43.00 | $28.85 | 67.1% | 2025-09-13 03:00 | 2025-09-01 12:00 | 279.00 |
| Baylor Bears vs Kansas State Wildcat | 2025-10-04 | $35.62 | $7.00 | $28.62 | 408.9% | 2025-09-26 10:00 | 2025-10-03 12:00 | 170.00 |
| Navy Midshipmen vs Air Force Falcon | 2025-10-04 | $83.45 | $55.00 | $28.45 | 51.7% | 2025-09-05 11:00 | 2025-10-03 12:00 | 673.00 |
| Ohio State Buckeyes vs Minnesota Golden Gopher | 2025-10-04 | $75.51 | $49.00 | $26.51 | 54.1% | 2025-09-05 23:00 | 2025-10-03 12:00 | 661.00 |
| Cincinnati Bearcats vs Iowa State Cyclone | 2025-10-04 | $34.36 | $56.00 | $21.64 | 38.6% | 2025-09-06 11:00 | 2025-09-06 12:00 | 1.00 |
| North Carolina Tar Heels vs Clemson Tiger | 2025-10-04 | $51.37 | $72.00 | $20.63 | 28.7% | 2025-10-03 18:00 | 2025-10-03 12:00 | 6.00 |
| Northwestern Wildcats vs Louisiana Monroe Warhawk | 2025-10-04 | $69.89 | $50.00 | $19.89 | 39.8% | 2025-10-04 02:00 | 2025-09-26 06:00 | 188.00 |
| Nebraska Cornhuskers vs Michigan State Spartan | 2025-10-04 | $52.18 | $35.00 | $17.18 | 49.1% | 2025-09-27 21:00 | 2025-10-03 12:00 | 135.00 |
| UAB Blazers vs Army West Point Black Knight | 2025-10-04 | $27.42 | $13.00 | $14.42 | 110.9% | 2025-10-03 06:00 | 2025-10-03 06:00 | 0.00 |
| San Jose State Spartans vs New Mexico Lobo | 2025-10-03 | $17.11 | $5.00 | $12.11 | 242.2% | 2025-10-03 13:00 | 2025-10-03 12:00 | 1.00 |
| California Golden Bears vs Duke Blue Devil | 2025-10-04 | $35.57 | $24.00 | $11.57 | 48.2% | 2025-09-23 09:00 | 2025-09-18 12:00 | 117.00 |
| Northern Illinois Huskies vs Miami (OH) RedHawk | 2025-10-04 | $26.50 | $15.00 | $11.50 | 76.7% | 2025-10-02 20:00 | 2025-10-02 20:24 | 0.41 |
| UMass Minutemen vs Western Michigan Bronco | 2025-10-04 | $27.97 | $17.00 | $10.97 | 64.5% | 2025-09-10 19:00 | 2025-10-03 12:00 | 545.00 |
| Buffalo Bulls vs Eastern Michigan Eagle | 2025-10-04 | $21.42 | $12.00 | $9.42 | 78.5% | 2025-10-03 11:00 | 2025-10-03 12:00 | 1.00 |
| Purdue Boilermakers vs Illinois Fighting Illini | 2025-10-04 | $14.72 | $6.00 | $8.72 | 145.3% | 2025-09-05 16:00 | 2025-08-27 15:51 | 216.14 |
| Delaware Blue Hens vs Western Kentucky Hilltopper | 2025-10-03 | $17.61 | $26.00 | $8.39 | 32.3% | 2025-09-06 00:00 | 2025-10-02 20:24 | 644.41 |
| BYU Cougars vs West Virginia Mountaineer | 2025-10-03 | $63.77 | $72.00 | $8.23 | 11.4% | 2025-09-14 11:00 | 2025-10-03 12:00 | 457.00 |
| Ball State Cardinals vs Ohio Bobcat | 2025-10-04 | $19.48 | $12.00 | $7.48 | 62.3% | 2025-09-09 13:00 | 2025-10-03 00:00 | 563.00 |
| Virginia Tech Hokies vs Wake Forest Demon Deacon | 2025-10-04 | $30.69 | $24.00 | $6.69 | 27.9% | 2025-09-22 23:00 | 2025-09-25 00:00 | 49.00 |
| Old Dominion Monarchs vs Coastal Carolina Chanticleer | 2025-10-04 | $32.00 | $26.00 | $6.00 | 23.1% | 2025-10-04 02:00 | 2025-09-27 12:00 | 158.00 |
| Fresno State Bulldogs vs Nevada Wolf Pack | 2025-10-04 | $20.85 | $15.00 | $5.85 | 39.0% | 2025-09-10 00:00 | 2025-08-28 06:00 | 306.00 |
| Akron Zips vs Central Michigan Chippewa | 2025-10-04 | $20.69 | $15.00 | $5.69 | 37.9% | 2025-09-20 07:00 | 2025-10-03 12:00 | 317.00 |
| Navy Midshipmen vs Air Force Falcon | 2025-10-04 | $83.45 | $89.00 | $5.55 | 6.2% | 2025-09-04 11:00 | 2025-09-27 12:00 | 553.00 |
| Notre Dame Fighting Irish vs Boise State Bronco | 2025-10-04 | $51.33 | $47.00 | $4.33 | 9.2% | 2025-10-04 14:00 | 2025-10-03 12:00 | 26.00 |
| Arizona Wildcats vs Oklahoma State Cowboy | 2025-10-04 | $15.23 | $12.00 | $3.23 | 26.9% | 2025-09-20 20:00 | 2025-09-20 18:00 | 2.00 |
| Wyoming Cowboys vs UNLV Rebel | 2025-10-04 | $33.27 | $36.00 | $2.73 | 7.6% | 2025-10-04 16:00 | 2025-09-05 18:00 | 694.00 |
| Memphis Tigers vs Tulsa Golden Hurricane | 2025-10-04 | $13.39 | $11.00 | $2.39 | 21.7% | 2025-09-13 13:00 | 2025-08-24 01:06 | 491.90 |
| Pittsburgh Panthers vs Boston College Eagle | 2025-10-04 | $6.38 | $4.00 | $2.38 | 59.5% | 2025-09-14 15:00 | 2025-09-17 12:00 | 69.00 |
| Southern Methodist (SMU) Mustangs vs Syracuse Orange | 2025-10-04 | $25.36 | $23.00 | $2.36 | 10.3% | 2025-09-06 15:00 | 2025-09-22 18:00 | 387.00 |
| Houston Cougars vs Texas Tech Red Raider | 2025-10-04 | $53.00 | $51.00 | $2.00 | 3.9% | 2025-09-19 09:00 | 2025-09-14 09:47 | 119.22 |
| UConn Huskies vs Florida International Panther | 2025-10-04 | $11.60 | $10.00 | $1.60 | 16.0% | 2025-09-06 17:00 | 2025-08-28 18:00 | 215.00 |
| Georgia State Panthers vs James Madison Duke | 2025-10-04 | $17.29 | $16.00 | $1.29 | 8.1% | 2025-09-30 23:00 | 2025-09-27 00:00 | 95.00 |
| South Florida Bulls vs Charlotte 49er | 2025-10-03 | $14.15 | $13.00 | $1.15 | 8.8% | 2025-09-13 03:00 | 2025-09-07 00:00 | 147.00 |
| Temple Owls vs UTSA Roadrunner | 2025-10-04 | $5.07 | $4.00 | $1.07 | 26.8% | 2025-09-13 10:00 | 2025-09-16 17:03 | 79.06 |
| Louisville Cardinals vs Virginia Cavalier | 2025-10-04 | $6.02 | $5.00 | $1.02 | 20.4% | 2025-09-09 23:00 | 2025-08-24 01:06 | 405.90 |
| Maryland Terrapins vs Washington Huskie | 2025-10-04 | $12.68 | $12.00 | $0.68 | 5.7% | 2025-09-20 10:00 | 2025-09-20 12:00 | 2.00 |
| UCF Knights vs Kansas Jayhawk | 2025-10-04 | $10.23 | $10.00 | $0.23 | 2.3% | 2025-09-05 18:00 | 2025-09-05 18:00 | 0.00 |
| Troy Trojans vs South Alabama Jaguar | 2025-10-04 | $22.13 | $22.00 | $0.13 | 0.6% | 2025-09-05 11:00 | 2025-09-05 10:43 | 0.27 |
| Rice Owls vs Florida Atlantic Owl | 2025-10-04 | $10.99 | $11.00 | $0.01 | 0.1% | 2025-09-06 00:00 | 2025-10-03 12:00 | 660.00 |
| Arkansas State Red Wolves vs Texas State Bobcat | 2025-10-04 | $13.00 | $13.00 | $0.00 | 0.0% | 2025-09-10 07:00 | 2025-09-10 00:00 | 7.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 45% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.