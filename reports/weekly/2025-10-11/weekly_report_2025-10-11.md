# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-11

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **40**
- MAE (price): **$16.07**
- RMSE (price): **$27.15**
- Games > 5% price error: **36 / 40**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **368.78 h**  •  Median |Δ|: **258.50 h**
- Within 6h: **3/40**  •  Within 12h: **5/40**  •  Within 24h: **5/40**
- Bias: predictions are on average **208.42 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| LSU Tigers vs South Carolina Gamecock | 2025-10-11 | $136.87 | $28.00 | $108.87 | 388.8% | 2025-10-06 23:00 | 2025-10-09 18:00 | 67.00 |
| Illinois Fighting Illini vs Ohio State Buckeye | 2025-10-11 | $190.13 | $121.00 | $69.13 | 57.1% | 2025-10-10 23:00 | 2025-09-27 12:00 | 323.00 |
| Colorado Buffaloes vs Iowa State Cyclone | 2025-10-11 | $43.59 | $100.00 | $56.41 | 56.4% | 2025-09-27 23:00 | 2025-09-19 12:00 | 203.00 |
| Tennessee Volunteers vs Arkansas Razorback | 2025-10-11 | $153.17 | $200.00 | $46.83 | 23.4% | 2025-09-20 22:00 | 2025-09-01 20:44 | 457.25 |
| Utah Utes vs Arizona State Sun Devil | 2025-10-11 | $52.81 | $93.00 | $40.19 | 43.2% | 2025-09-27 23:00 | 2025-08-31 00:00 | 671.00 |
| Missouri Tigers vs Alabama Crimson Tide | 2025-10-11 | $192.50 | $226.00 | $33.50 | 14.8% | 2025-10-10 21:00 | 2025-08-24 01:06 | 1147.90 |
| James Madison Dukes vs Louisiana Lafayette Ragin Cajun | 2025-10-11 | $83.84 | $57.00 | $26.84 | 47.1% | 2025-09-11 07:00 | 2025-09-16 17:03 | 130.06 |
| Auburn Tigers vs Georgia Bulldog | 2025-10-11 | $246.50 | $222.00 | $24.50 | 11.0% | 2025-10-10 13:00 | 2025-10-06 12:00 | 97.00 |
| Boston College Eagles vs Clemson Tiger | 2025-10-11 | $54.12 | $30.00 | $24.12 | 80.4% | 2025-09-16 18:00 | 2025-09-18 18:00 | 48.00 |
| Kansas State Wildcats vs TCU Horned Frog | 2025-10-11 | $45.49 | $69.00 | $23.51 | 34.1% | 2025-09-27 23:00 | 2025-09-27 12:00 | 11.00 |
| USC Trojans vs Michigan Wolverine | 2025-10-11 | $68.59 | $92.00 | $23.41 | 25.4% | 2025-09-11 03:00 | 2025-09-17 18:00 | 159.00 |
| Florida State Seminoles vs Pittsburgh Panther | 2025-10-11 | $53.63 | $33.00 | $20.63 | 62.5% | 2025-09-19 18:00 | 2025-08-27 15:51 | 554.14 |
| Boise State Broncos vs New Mexico Lobo | 2025-10-11 | $156.98 | $138.00 | $18.98 | 13.8% | 2025-09-14 10:00 | 2025-09-24 18:00 | 248.00 |
| Oregon Ducks vs Indiana Hoosier | 2025-10-11 | $85.89 | $70.00 | $15.89 | 22.7% | 2025-09-11 04:00 | 2025-08-29 00:00 | 316.00 |
| Arizona Wildcats vs BYU Cougar | 2025-10-11 | $63.12 | $50.00 | $13.12 | 26.2% | 2025-09-28 08:00 | 2025-09-22 00:00 | 152.00 |
| Penn State Nittany Lions vs Northwestern Wildcat | 2025-10-11 | $46.44 | $55.00 | $8.56 | 15.6% | 2025-10-04 21:00 | 2025-09-03 18:00 | 747.00 |
| UTSA Roadrunners vs Rice Owl | 2025-10-11 | $21.54 | $14.00 | $7.54 | 53.9% | 2025-10-02 11:00 | 2025-09-27 12:00 | 119.00 |
| Colorado State Rams vs Fresno State Bulldog | 2025-10-10 | $11.65 | $19.00 | $7.35 | 38.7% | 2025-09-10 23:00 | 2025-10-10 18:00 | 715.00 |
| Temple Owls vs Navy Midshipmen | 2025-10-11 | $21.86 | $15.00 | $6.86 | 45.7% | 2025-09-11 21:00 | 2025-09-27 00:00 | 363.00 |
| Texas Tech Red Raiders vs Kansas Jayhawk | 2025-10-11 | $53.26 | $47.00 | $6.26 | 13.3% | 2025-10-02 07:00 | 2025-09-01 12:00 | 739.00 |
| Wisconsin Badgers vs Iowa Hawkeye | 2025-10-11 | $76.99 | $71.00 | $5.99 | 8.4% | 2025-10-10 23:00 | 2025-08-24 01:06 | 1149.90 |
| Minnesota Golden Gophers vs Purdue Boilermaker | 2025-10-11 | $45.30 | $40.00 | $5.30 | 13.2% | 2025-10-10 15:00 | 2025-09-01 12:00 | 939.00 |
| Bowling Green State Falcons vs Toledo Rocket | 2025-10-11 | $39.05 | $34.00 | $5.05 | 14.9% | 2025-10-11 02:00 | 2025-09-01 12:00 | 950.00 |
| Marshall Thundering Herd vs Old Dominion Monarch | 2025-10-11 | $16.86 | $12.00 | $4.86 | 40.5% | 2025-10-09 12:00 | 2025-09-27 12:00 | 288.00 |
| Florida Atlantic Owls vs UAB Blazer | 2025-10-11 | $5.34 | $10.00 | $4.66 | 46.6% | 2025-09-11 13:00 | 2025-09-26 18:00 | 365.00 |
| Washington Huskies vs Rutgers Scarlet Knight | 2025-10-10 | $11.65 | $7.00 | $4.65 | 66.4% | 2025-09-10 23:00 | 2025-10-09 18:00 | 691.00 |
| Georgia State Panthers vs Appalachian State Mountaineer | 2025-10-11 | $17.51 | $13.00 | $4.51 | 34.7% | 2025-10-07 23:00 | 2025-09-26 18:00 | 269.00 |
| Cincinnati Bearcats vs UCF Knight | 2025-10-11 | $33.05 | $29.00 | $4.05 | 14.0% | 2025-09-19 10:00 | 2025-09-27 06:00 | 188.00 |
| Eastern Michigan Eagles vs Northern Illinois Huskie | 2025-10-11 | $18.29 | $22.00 | $3.71 | 16.9% | 2025-09-27 10:00 | 2025-09-25 18:00 | 40.00 |
| Maryland Terrapins vs Nebraska Cornhusker | 2025-10-11 | $32.69 | $29.00 | $3.69 | 12.7% | 2025-09-16 10:00 | 2025-09-18 18:00 | 56.00 |
| Oklahoma State Cowboys vs Houston Cougar | 2025-10-11 | $40.93 | $38.00 | $2.93 | 7.7% | 2025-10-02 07:00 | 2025-09-25 12:00 | 163.00 |
| Texas State Bobcats vs Troy Trojan | 2025-10-11 | $19.57 | $17.00 | $2.57 | 15.1% | 2025-10-02 02:00 | 2025-09-07 18:00 | 584.00 |
| Georgia Tech Yellow Jackets vs Virginia Tech Hokie | 2025-10-11 | $23.44 | $21.00 | $2.44 | 11.6% | 2025-09-11 19:00 | 2025-08-24 01:06 | 449.90 |
| Coastal Carolina Chanticleers vs Louisiana Monroe Warhawk | 2025-10-11 | $17.33 | $16.00 | $1.33 | 8.3% | 2025-10-10 18:00 | 2025-08-24 01:06 | 1144.90 |
| Texas A&M Aggies vs Florida Gator | 2025-10-11 | $145.29 | $144.00 | $1.29 | 0.9% | 2025-09-20 22:00 | 2025-09-27 12:00 | 158.00 |
| Kent State Golden Flashes vs UMass Minutemen | 2025-10-11 | $12.94 | $12.00 | $0.94 | 7.8% | 2025-09-15 11:00 | 2025-09-15 16:30 | 5.51 |
| Western Michigan Broncos vs Ball State Cardinal | 2025-10-11 | $18.29 | $19.00 | $0.71 | 3.7% | 2025-09-27 10:00 | 2025-09-27 12:00 | 2.00 |
| Southern Methodist (SMU) Mustangs vs Stanford Cardinal | 2025-10-11 | $10.66 | $10.00 | $0.66 | 6.6% | 2025-09-11 03:00 | 2025-09-10 00:00 | 27.00 |
| Akron Zips vs Miami (OH) RedHawk | 2025-10-11 | $19.60 | $19.00 | $0.60 | 3.2% | 2025-09-27 14:00 | 2025-09-27 12:00 | 2.00 |
| North Texas Mean Green vs South Florida Bull | 2025-10-10 | $22.50 | $22.00 | $0.50 | 2.3% | 2025-09-16 00:00 | 2025-09-16 11:39 | 11.65 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 62% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.