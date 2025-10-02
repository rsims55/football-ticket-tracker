# 📈 Weekly Ticket Price Model Report
**Date:** 2025-09-20

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **34**
- MAE (price): **$11.72**
- RMSE (price): **$19.96**
- Games > 5% price error: **22 / 34**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **153.05 h**  •  Median |Δ|: **119.50 h**
- Within 6h: **8/34**  •  Within 12h: **8/34**  •  Within 24h: **9/34**
- Bias: predictions are on average **121.29 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Washington State Cougars vs Washington Huskie | 2025-09-20 | $156.63 | $89.00 | $67.63 | 76.0% | 2025-09-18 10:00 | 2025-09-20 00:00 | 38.00 |
| Memphis Tigers vs Arkansas Razorback | 2025-09-20 | $31.21 | $92.00 | $60.79 | 66.1% | 2025-09-13 20:00 | 2025-08-28 06:00 | 398.00 |
| Oklahoma Sooners vs Auburn Tiger | 2025-09-20 | $38.06 | $78.00 | $39.94 | 51.2% | 2025-09-14 01:00 | 2025-08-31 12:00 | 325.00 |
| Colorado Buffaloes vs Wyoming Cowboy | 2025-09-20 | $131.66 | $161.00 | $29.34 | 18.2% | 2025-09-17 06:00 | 2025-09-10 00:00 | 174.00 |
| Rutgers Scarlet Knights vs Iowa Hawkeye | 2025-09-19 | $46.19 | $23.00 | $23.19 | 100.8% | 2025-09-18 09:00 | 2025-09-16 11:39 | 45.35 |
| Indiana Hoosiers vs Illinois Fighting Illini | 2025-09-20 | $155.33 | $134.00 | $21.33 | 15.9% | 2025-09-19 23:00 | 2025-08-24 01:06 | 645.90 |
| Florida State Seminoles vs Kent State Golden Flashe | 2025-09-20 | $82.00 | $63.00 | $19.00 | 30.2% | 2025-08-21 09:00 | 2025-08-29 18:00 | 201.00 |
| Oklahoma State Cowboys vs Tulsa Golden Hurricane | 2025-09-19 | $57.03 | $40.00 | $17.03 | 42.6% | 2025-09-19 00:00 | 2025-09-18 18:00 | 6.00 |
| Utah Utes vs Texas Tech Red Raider | 2025-09-20 | $102.00 | $85.00 | $17.00 | 20.0% | 2025-09-06 09:00 | 2025-08-29 12:00 | 189.00 |
| Baylor Bears vs Arizona State Sun Devil | 2025-09-20 | $35.44 | $51.00 | $15.56 | 30.5% | 2025-09-12 10:00 | 2025-09-07 18:00 | 112.00 |
| Clemson Tigers vs Syracuse Orange | 2025-09-20 | $46.11 | $60.00 | $13.89 | 23.2% | 2025-09-13 15:00 | 2025-09-10 00:00 | 87.00 |
| Kansas Jayhawks vs West Virginia Mountaineer | 2025-09-20 | $27.70 | $37.00 | $9.30 | 25.1% | 2025-09-19 09:00 | 2025-09-08 12:00 | 261.00 |
| Miami (OH) RedHawks vs UNLV Rebel | 2025-09-20 | $28.16 | $37.00 | $8.84 | 23.9% | 2025-09-19 00:00 | 2025-09-08 12:00 | 252.00 |
| Missouri Tigers vs South Carolina Gamecock | 2025-09-20 | $85.23 | $94.00 | $8.77 | 9.3% | 2025-08-31 21:00 | 2025-09-01 00:00 | 3.00 |
| Colorado State Rams vs UTSA Roadrunner | 2025-09-20 | $24.05 | $32.00 | $7.95 | 24.8% | 2025-09-19 11:00 | 2025-08-31 06:00 | 461.00 |
| Virginia Tech Hokies vs Wofford Terrier | 2025-09-20 | $19.34 | $25.00 | $5.66 | 22.6% | 2025-09-12 11:00 | 2025-09-08 12:00 | 95.00 |
| South Alabama Jaguars vs Coastal Carolina Chanticleer | 2025-09-20 | $22.26 | $27.00 | $4.74 | 17.6% | 2025-09-19 23:00 | 2025-09-08 12:00 | 275.00 |
| Miami Hurricanes vs Florida Gator | 2025-09-20 | $165.54 | $170.00 | $4.46 | 2.6% | 2025-09-20 02:00 | 2025-09-13 11:27 | 158.54 |
| Akron Zips vs Duquesne Duke | 2025-09-20 | $25.60 | $30.00 | $4.40 | 14.7% | 2025-09-18 23:00 | 2025-09-08 12:00 | 251.00 |
| USC Trojans vs Michigan State Spartan | 2025-09-20 | $32.92 | $29.00 | $3.92 | 13.5% | 2025-09-05 00:00 | 2025-09-05 12:00 | 12.00 |
| Utah State Aggies vs McNeese State Cowboy | 2025-09-20 | $27.81 | $24.00 | $3.81 | 15.9% | 2025-08-26 09:00 | 2025-09-05 18:00 | 249.00 |
| Eastern Michigan Eagles vs Louisiana Lafayette Ragin Cajun | 2025-09-20 | $22.41 | $26.00 | $3.59 | 13.8% | 2025-09-12 11:00 | 2025-09-08 12:00 | 95.00 |
| TCU Horned Frogs vs Southern Methodist (SMU) Mustang | 2025-09-20 | $86.35 | $83.00 | $3.35 | 4.0% | 2025-09-19 10:00 | 2025-09-06 18:00 | 304.00 |
| Buffalo Bulls vs Troy Trojan | 2025-09-20 | $25.74 | $27.00 | $1.26 | 4.7% | 2025-09-16 23:00 | 2025-09-08 12:00 | 203.00 |
| East Carolina Pirates vs BYU Cougar | 2025-09-20 | $66.22 | $65.00 | $1.22 | 1.9% | 2025-09-06 23:00 | 2025-09-08 12:00 | 37.00 |
| Oregon Ducks vs Oregon State Beaver | 2025-09-20 | $99.70 | $99.00 | $0.70 | 0.7% | 2025-09-08 20:00 | 2025-09-08 18:00 | 2.00 |
| UCF Knights vs North Carolina Tar Heel | 2025-09-20 | $36.53 | $37.00 | $0.47 | 1.3% | 2025-09-12 11:00 | 2025-09-06 18:00 | 137.00 |
| Mississippi Rebels vs Tulane Green Wave | 2025-09-20 | $49.47 | $49.00 | $0.47 | 1.0% | 2025-09-13 19:00 | 2025-09-08 12:00 | 127.00 |
| Louisville Cardinals vs Bowling Green State Falcon | 2025-09-20 | $5.34 | $5.00 | $0.34 | 6.8% | 2025-08-30 06:00 | 2025-08-30 06:00 | 0.00 |
| Duke Blue Devils vs North Carolina State Wolfpack | 2025-09-20 | $29.26 | $29.00 | $0.26 | 0.9% | 2025-08-30 06:00 | 2025-08-30 06:00 | 0.00 |
| Western Michigan Broncos vs Toledo Rocket | 2025-09-20 | $14.17 | $14.00 | $0.17 | 1.2% | 2025-09-06 18:00 | 2025-09-06 18:00 | 0.00 |
| Virginia Cavaliers vs Stanford Cardinal | 2025-09-20 | $14.84 | $15.00 | $0.16 | 1.1% | 2025-09-09 00:00 | 2025-09-06 12:00 | 60.00 |
| Wisconsin Badgers vs Maryland Terrapin | 2025-09-20 | $16.07 | $16.00 | $0.07 | 0.4% | 2025-08-29 00:00 | 2025-08-29 00:00 | 0.00 |
| Georgia Tech Yellow Jackets vs Temple Owl | 2025-09-20 | $12.00 | $12.00 | $0.00 | 0.0% | 2025-08-31 00:00 | 2025-08-31 00:00 | 0.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 68% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.