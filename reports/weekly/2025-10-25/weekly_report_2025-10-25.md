# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-25

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **44**
- MAE (price): **$28.75**
- RMSE (price): **$48.81**
- Games > 5% price error: **42 / 44**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **336.46 h**  •  Median |Δ|: **176.50 h**
- Within 6h: **3/44**  •  Within 12h: **3/44**  •  Within 24h: **4/44**
- Bias: predictions are on average **236.94 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| LSU Tigers vs Texas A&M Aggie | 2025-10-25 | $300.10 | $96.00 | $204.10 | 212.6% | 2025-10-11 03:00 | 2025-10-13 12:00 | 57.00 |
| Vanderbilt Commodores vs Missouri Tiger | 2025-10-25 | $166.80 | $46.00 | $120.80 | 262.6% | 2025-10-24 23:00 | 2025-10-18 00:00 | 167.00 |
| South Carolina Gamecocks vs Alabama Crimson Tide | 2025-10-25 | $107.11 | $218.00 | $110.89 | 50.9% | 2025-10-24 20:00 | 2025-10-11 12:00 | 320.00 |
| Miami Hurricanes vs Stanford Cardinal | 2025-10-25 | $87.54 | $9.00 | $78.54 | 872.7% | 2025-09-27 00:00 | 2025-08-30 06:00 | 666.00 |
| Oklahoma Sooners vs Mississippi Rebel | 2025-10-25 | $145.05 | $73.00 | $72.05 | 98.7% | 2025-09-25 03:00 | 2025-09-01 12:00 | 567.00 |
| Michigan State Spartans vs Michigan Wolverine | 2025-10-25 | $158.85 | $88.00 | $70.85 | 80.5% | 2025-10-24 10:00 | 2025-10-20 12:00 | 94.00 |
| Kentucky Wildcats vs Tennessee Volunteer | 2025-10-25 | $142.88 | $78.00 | $64.88 | 83.2% | 2025-10-04 22:00 | 2025-10-10 18:00 | 140.00 |
| Washington Huskies vs Illinois Fighting Illini | 2025-10-25 | $70.56 | $18.00 | $52.56 | 292.0% | 2025-10-24 15:00 | 2025-10-20 12:00 | 99.00 |
| Iowa Hawkeyes vs Minnesota Golden Gopher | 2025-10-25 | $118.81 | $69.00 | $49.81 | 72.2% | 2025-10-17 13:00 | 2025-10-12 18:00 | 115.00 |
| Texas Tech Red Raiders vs Oklahoma State Cowboy | 2025-10-25 | $88.86 | $138.00 | $49.14 | 35.6% | 2025-10-16 13:00 | 2025-08-31 12:00 | 1105.00 |
| Utah Utes vs Colorado Buffaloe | 2025-10-25 | $161.62 | $210.00 | $48.38 | 23.0% | 2025-10-11 23:00 | 2025-09-14 20:35 | 650.40 |
| North Carolina Tar Heels vs Virginia Cavalier | 2025-10-25 | $79.90 | $49.00 | $30.90 | 63.1% | 2025-10-07 00:00 | 2025-10-13 06:00 | 150.00 |
| Kansas Jayhawks vs Kansas State Wildcat | 2025-10-25 | $163.71 | $140.00 | $23.71 | 16.9% | 2025-10-24 23:00 | 2025-10-10 12:00 | 347.00 |
| Northern Illinois Huskies vs Ball State Cardinal | 2025-10-25 | $35.07 | $14.00 | $21.07 | 150.5% | 2025-10-25 02:00 | 2025-10-18 18:00 | 152.00 |
| Charlotte 49ers vs North Texas Mean Green | 2025-10-24 | $22.89 | $2.00 | $20.89 | 1044.5% | 2025-10-10 18:00 | 2025-10-18 12:00 | 186.00 |
| West Virginia Mountaineers vs TCU Horned Frog | 2025-10-25 | $66.79 | $46.00 | $20.79 | 45.2% | 2025-10-16 09:00 | 2025-10-13 12:00 | 69.00 |
| Virginia Tech Hokies vs California Golden Bear | 2025-10-24 | $22.89 | $3.00 | $19.89 | 663.0% | 2025-10-10 18:00 | 2025-10-21 12:00 | 258.00 |
| Mississippi State Bulldogs vs Texas Longhorn | 2025-10-25 | $57.51 | $38.00 | $19.51 | 51.3% | 2025-10-05 18:00 | 2025-10-20 12:00 | 354.00 |
| Eastern Michigan Eagles vs Ohio Bobcat | 2025-10-25 | $30.86 | $12.00 | $18.86 | 157.2% | 2025-10-21 10:00 | 2025-10-18 00:00 | 82.00 |
| Nevada Wolf Pack vs Boise State Bronco | 2025-10-24 | $21.02 | $4.00 | $17.02 | 425.5% | 2025-10-03 11:00 | 2025-10-09 06:00 | 139.00 |
| Rice Owls vs UConn Huskie | 2025-10-25 | $21.48 | $6.00 | $15.48 | 258.0% | 2025-09-27 00:00 | 2025-10-02 20:24 | 140.41 |
| Georgia Tech Yellow Jackets vs Syracuse Orange | 2025-10-25 | $39.23 | $26.00 | $13.23 | 50.9% | 2025-09-26 05:00 | 2025-09-05 18:00 | 491.00 |
| Nebraska Cornhuskers vs Northwestern Wildcat | 2025-10-25 | $42.81 | $30.00 | $12.81 | 42.7% | 2025-10-18 21:00 | 2025-10-19 12:00 | 15.00 |
| Indiana Hoosiers vs UCLA Bruin | 2025-10-25 | $80.19 | $68.00 | $12.19 | 17.9% | 2025-09-27 10:00 | 2025-09-14 12:46 | 309.22 |
| Arizona State Sun Devils vs Houston Cougar | 2025-10-25 | $43.47 | $32.00 | $11.47 | 35.8% | 2025-10-11 23:00 | 2025-08-24 01:06 | 1173.90 |
| Old Dominion Monarchs vs Appalachian State Mountaineer | 2025-10-25 | $33.12 | $23.00 | $10.12 | 44.0% | 2025-10-24 23:00 | 2025-10-03 12:00 | 515.00 |
| Oregon Ducks vs Wisconsin Badger | 2025-10-25 | $104.84 | $95.00 | $9.84 | 10.4% | 2025-10-24 11:00 | 2025-10-20 12:00 | 95.00 |
| Cincinnati Bearcats vs Baylor Bear | 2025-10-25 | $46.15 | $37.00 | $9.15 | 24.7% | 2025-09-27 21:00 | 2025-09-25 12:00 | 57.00 |
| Tulsa Golden Hurricane vs Temple Owl | 2025-10-25 | $19.11 | $10.00 | $9.11 | 91.1% | 2025-10-11 10:00 | 2025-10-11 12:00 | 2.00 |
| Miami (OH) RedHawks vs Western Michigan Bronco | 2025-10-25 | $30.86 | $37.00 | $6.14 | 16.6% | 2025-10-21 10:00 | 2025-08-28 06:00 | 1300.00 |
| Kent State Golden Flashes vs Bowling Green State Falcon | 2025-10-25 | $17.06 | $11.00 | $6.06 | 55.1% | 2025-10-11 05:00 | 2025-10-11 06:00 | 1.00 |
| Central Michigan Chippewas vs UMass Minutemen | 2025-10-25 | $28.95 | $24.00 | $4.95 | 20.6% | 2025-10-16 08:00 | 2025-10-13 12:00 | 68.00 |
| Iowa State Cyclones vs BYU Cougar | 2025-10-25 | $64.31 | $69.00 | $4.69 | 6.8% | 2025-09-25 03:00 | 2025-10-10 18:00 | 375.00 |
| Memphis Tigers vs South Florida Bull | 2025-10-25 | $15.11 | $11.00 | $4.11 | 37.4% | 2025-10-05 05:00 | 2025-08-28 18:00 | 899.00 |
| Southern Miss Golden Eagles vs Louisiana Monroe Warhawk | 2025-10-25 | $6.00 | $2.00 | $4.00 | 200.0% | 2025-10-18 13:00 | 2025-10-13 12:00 | 121.00 |
| Arkansas Razorbacks vs Auburn Tiger | 2025-10-25 | $27.37 | $31.00 | $3.63 | 11.7% | 2025-10-24 00:00 | 2025-09-01 20:44 | 1251.25 |
| Louisville Cardinals vs Boston College Eagle | 2025-10-25 | $7.46 | $4.00 | $3.46 | 86.5% | 2025-09-30 15:00 | 2025-10-12 00:00 | 273.00 |
| Arkansas State Red Wolves vs Georgia Southern Eagle | 2025-10-25 | $15.26 | $18.00 | $2.74 | 15.2% | 2025-09-30 00:00 | 2025-09-27 12:00 | 60.00 |
| Fresno State Bulldogs vs San Diego State Aztec | 2025-10-25 | $21.44 | $19.00 | $2.44 | 12.8% | 2025-10-01 00:00 | 2025-09-10 00:00 | 504.00 |
| Troy Trojans vs Louisiana Lafayette Ragin Cajun | 2025-10-25 | $14.91 | $17.00 | $2.09 | 12.3% | 2025-09-26 11:00 | 2025-09-26 06:00 | 5.00 |
| Wake Forest Demon Deacons vs Southern Methodist (SMU) Mustang | 2025-10-25 | $19.12 | $18.00 | $1.12 | 6.2% | 2025-09-25 03:00 | 2025-09-10 00:00 | 363.00 |
| Buffalo Bulls vs Akron Zip | 2025-10-25 | $24.04 | $25.00 | $0.96 | 3.8% | 2025-10-24 09:00 | 2025-10-13 12:00 | 261.00 |
| Pittsburgh Panthers vs North Carolina State Wolfpack | 2025-10-25 | $6.48 | $6.00 | $0.48 | 8.0% | 2025-10-05 15:00 | 2025-10-09 18:00 | 99.00 |
| New Mexico Lobos vs Utah State Aggie | 2025-10-25 | $20.25 | $20.00 | $0.25 | 1.2% | 2025-10-01 00:00 | 2025-09-01 12:00 | 708.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 68% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.