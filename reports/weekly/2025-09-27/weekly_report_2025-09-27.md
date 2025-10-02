# 📈 Weekly Ticket Price Model Report
**Date:** 2025-09-27

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **40**
- MAE (price): **$6.44**
- RMSE (price): **$11.19**
- Games > 5% price error: **21 / 40**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **193.93 h**  •  Median |Δ|: **80.86 h**
- Within 6h: **13/40**  •  Within 12h: **14/40**  •  Within 24h: **15/40**
- Bias: predictions are on average **107.28 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Virginia Cavaliers vs Florida State Seminole | 2025-09-26 | $71.43 | $35.00 | $36.43 | 104.1% | 2025-09-07 13:00 | 2025-09-26 18:00 | 461.00 |
| South Carolina Gamecocks vs Kentucky Wildcat | 2025-09-27 | $108.57 | $75.00 | $33.57 | 44.8% | 2025-09-01 12:00 | 2025-09-01 12:00 | 0.00 |
| Buffalo Bulls vs Uconn Huskie | 2025-09-27 | $50.92 | $25.00 | $25.92 | 103.7% | 2025-09-07 18:00 | 2025-09-07 18:00 | 0.00 |
| Missouri Tigers vs UMass Minutemen | 2025-09-27 | $59.43 | $84.00 | $24.57 | 29.2% | 2025-09-26 03:00 | 2025-09-15 08:03 | 258.95 |
| West Virginia Mountaineers vs Utah Ute | 2025-09-27 | $60.09 | $74.00 | $13.91 | 18.8% | 2025-09-18 11:00 | 2025-09-13 10:23 | 120.61 |
| Northwestern Wildcats vs UCLA Bruin | 2025-09-27 | $98.53 | $112.00 | $13.47 | 12.0% | 2025-09-25 00:00 | 2025-09-01 12:00 | 564.00 |
| Texas A&M Aggies vs Auburn Tiger | 2025-09-27 | $139.29 | $126.00 | $13.29 | 10.5% | 2025-09-26 23:00 | 2025-09-01 12:00 | 611.00 |
| Illinois Fighting Illini vs USC Trojan | 2025-09-27 | $78.12 | $65.00 | $13.12 | 20.2% | 2025-08-28 16:00 | 2025-08-24 01:06 | 110.90 |
| Oklahoma State Cowboys vs Baylor Bear | 2025-09-27 | $59.98 | $73.00 | $13.02 | 17.8% | 2025-09-18 07:00 | 2025-08-29 00:00 | 487.00 |
| Kansas State Wildcats vs UCF Knight | 2025-09-27 | $45.49 | $58.00 | $12.51 | 21.6% | 2025-09-13 23:00 | 2025-09-14 09:47 | 10.79 |
| Oregon State Beavers vs Houston Cougar | 2025-09-26 | $13.45 | $6.00 | $7.45 | 124.2% | 2025-09-12 11:00 | 2025-09-26 12:00 | 337.00 |
| Northern Illinois Huskies vs San Diego State Aztec | 2025-09-27 | $27.89 | $34.00 | $6.11 | 18.0% | 2025-09-27 00:00 | 2025-08-24 01:06 | 814.90 |
| Arizona State Sun Devils vs TCU Horned Frog | 2025-09-26 | $24.92 | $19.00 | $5.92 | 31.2% | 2025-08-27 18:00 | 2025-08-27 15:51 | 2.14 |
| Boise State Broncos vs Appalachian State Mountaineer | 2025-09-27 | $46.36 | $51.00 | $4.64 | 9.1% | 2025-09-20 00:00 | 2025-09-10 00:00 | 240.00 |
| Iowa State Cyclones vs Arizona Wildcat | 2025-09-27 | $44.30 | $48.00 | $3.70 | 7.7% | 2025-09-26 00:00 | 2025-09-15 16:30 | 247.49 |
| Pittsburgh Panthers vs Louisville Cardinal | 2025-09-27 | $6.40 | $3.00 | $3.40 | 113.3% | 2025-08-28 06:00 | 2025-08-24 01:06 | 100.90 |
| New Mexico Lobos vs New Mexico State Aggie | 2025-09-27 | $31.65 | $35.00 | $3.35 | 9.6% | 2025-09-26 00:00 | 2025-08-24 01:06 | 790.90 |
| Tulsa Golden Hurricane vs Tulane Green Wave | 2025-09-27 | $7.25 | $4.00 | $3.25 | 81.2% | 2025-09-02 13:00 | 2025-09-15 08:03 | 307.05 |
| Washington Huskies vs Ohio State Buckeye | 2025-09-27 | $142.21 | $145.00 | $2.79 | 1.9% | 2025-09-07 17:00 | 2025-09-13 10:23 | 137.39 |
| Minnesota Golden Gophers vs Rutgers Scarlet Knight | 2025-09-27 | $42.49 | $45.00 | $2.51 | 5.6% | 2025-09-27 00:00 | 2025-08-30 12:00 | 660.00 |
| Mississippi Rebels vs LSU Tiger | 2025-09-27 | $243.33 | $241.00 | $2.33 | 1.0% | 2025-08-28 09:00 | 2025-09-14 09:47 | 408.79 |
| Louisiana Lafayette Ragin Cajuns vs Marshall Thundering Herd | 2025-09-27 | $20.07 | $22.00 | $1.93 | 8.8% | 2025-09-25 12:00 | 2025-09-06 12:00 | 456.00 |
| Iowa Hawkeyes vs Indiana Hoosier | 2025-09-27 | $78.70 | $77.00 | $1.70 | 2.2% | 2025-09-26 15:00 | 2025-09-14 09:47 | 293.21 |
| James Madison Dukes vs Georgia Southern Eagle | 2025-09-27 | $122.55 | $121.00 | $1.55 | 1.3% | 2025-09-06 16:00 | 2025-09-06 18:00 | 2.00 |
| Mississippi State Bulldogs vs Tennessee Volunteer | 2025-09-27 | $50.25 | $49.00 | $1.25 | 2.6% | 2025-09-06 12:00 | 2025-09-06 12:00 | 0.00 |
| Syracuse Orange vs Duke Blue Devil | 2025-09-27 | $35.05 | $34.00 | $1.05 | 3.1% | 2025-08-28 04:00 | 2025-08-27 15:51 | 12.14 |
| Southern Miss Golden Eagles vs Jacksonville State Gamecock | 2025-09-27 | $5.01 | $4.00 | $1.01 | 25.2% | 2025-09-12 23:00 | 2025-09-15 16:30 | 65.51 |
| North Carolina State Wolfpack vs Virginia Tech Hokie | 2025-09-27 | $86.98 | $86.00 | $0.98 | 1.1% | 2025-09-18 10:00 | 2025-09-14 09:47 | 96.22 |
| Kansas Jayhawks vs Cincinnati Bearcat | 2025-09-27 | $83.69 | $83.00 | $0.69 | 0.8% | 2025-08-28 07:00 | 2025-08-28 06:00 | 1.00 |
| Boston College Eagles vs California Golden Bear | 2025-09-27 | $15.48 | $15.00 | $0.48 | 3.2% | 2025-08-28 18:00 | 2025-08-28 18:00 | 0.00 |
| Wake Forest Demon Deacons vs Georgia Tech Yellow Jacket | 2025-09-27 | $23.55 | $24.00 | $0.45 | 1.9% | 2025-09-14 17:00 | 2025-09-14 20:35 | 3.60 |
| Central Michigan Chippewas vs Eastern Michigan Eagle | 2025-09-27 | $28.56 | $29.00 | $0.44 | 1.5% | 2025-09-16 22:00 | 2025-09-15 08:03 | 37.95 |
| Toledo Rockets vs Akron Zip | 2025-09-27 | $20.26 | $20.00 | $0.26 | 1.3% | 2025-09-10 00:00 | 2025-09-10 00:00 | 0.00 |
| Florida Atlantic Owls vs Memphis Tiger | 2025-09-27 | $7.25 | $7.00 | $0.25 | 3.6% | 2025-09-02 13:00 | 2025-09-01 12:00 | 25.00 |
| North Texas Mean Green vs South Alabama Jaguar | 2025-09-27 | $22.13 | $22.00 | $0.13 | 0.6% | 2025-08-29 12:00 | 2025-08-29 12:00 | 0.00 |
| Ohio Bobcats vs Bowling Green State Falcon | 2025-09-27 | $3.04 | $3.00 | $0.04 | 1.3% | 2025-09-11 06:00 | 2025-09-10 00:00 | 30.00 |
| Old Dominion Monarchs vs Liberty Flame | 2025-09-27 | $20.96 | $21.00 | $0.04 | 0.2% | 2025-08-30 00:00 | 2025-08-30 00:00 | 0.00 |
| Air Force Falcons vs Hawaii Rainbow Warrior | 2025-09-27 | $17.01 | $17.00 | $0.01 | 0.1% | 2025-08-29 15:00 | 2025-08-29 12:00 | 3.00 |
| Stanford Cardinal vs San Jose State Spartan | 2025-09-27 | $5.99 | $6.00 | $0.01 | 0.2% | 2025-09-17 16:00 | 2025-09-15 08:03 | 55.95 |
| Louisiana Monroe Warhawks vs Arkansas State Red Wolve | 2025-09-27 | $21.00 | $21.00 | $0.00 | 0.0% | 2025-09-10 05:00 | 2025-09-10 00:00 | 5.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 60% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.