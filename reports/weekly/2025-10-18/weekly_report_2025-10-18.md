# 📈 Weekly Ticket Price Model Report
**Date:** 2025-10-18

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **45**
- MAE (price): **$20.09**
- RMSE (price): **$32.41**
- Games > 5% price error: **38 / 45**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **328.45 h**  •  Median |Δ|: **258.00 h**
- Within 6h: **2/45**  •  Within 12h: **3/45**  •  Within 24h: **4/45**
- Bias: predictions are on average **207.03 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| North Texas Mean Green vs UTSA Roadrunner | 2025-10-18 | $135.66 | $17.00 | $118.66 | 698.0% | 2025-09-20 22:00 | 2025-09-09 06:00 | 280.00 |
| South Carolina Gamecocks vs Oklahoma Sooner | 2025-10-18 | $153.67 | $68.00 | $85.67 | 126.0% | 2025-09-18 20:00 | 2025-10-10 12:00 | 520.00 |
| Kentucky Wildcats vs Texas Longhorn | 2025-10-18 | $144.39 | $63.00 | $81.39 | 129.2% | 2025-09-27 22:00 | 2025-10-12 12:00 | 350.00 |
| Charlotte 49ers vs Temple Owl | 2025-10-18 | $60.77 | $2.00 | $58.77 | 2938.5% | 2025-09-19 10:00 | 2025-10-11 00:00 | 518.00 |
| Wisconsin Badgers vs Ohio State Buckeye | 2025-10-18 | $121.37 | $68.00 | $53.37 | 78.5% | 2025-09-28 21:00 | 2025-10-06 08:03 | 179.06 |
| Arkansas Razorbacks vs Texas A&M Aggie | 2025-10-18 | $127.21 | $77.00 | $50.21 | 65.2% | 2025-10-18 02:00 | 2025-10-06 12:00 | 278.00 |
| Georgia Bulldogs vs Mississippi Rebel | 2025-10-18 | $230.59 | $188.00 | $42.59 | 22.7% | 2025-09-18 04:00 | 2025-08-30 12:00 | 448.00 |
| Alabama Crimson Tide vs Tennessee Volunteer | 2025-10-18 | $236.92 | $198.00 | $38.92 | 19.7% | 2025-09-20 02:00 | 2025-09-25 12:00 | 130.00 |
| Iowa Hawkeyes vs Penn State Nittany Lion | 2025-10-18 | $99.36 | $64.00 | $35.36 | 55.2% | 2025-09-28 18:00 | 2025-10-03 12:00 | 114.00 |
| Appalachian State Mountaineers vs Coastal Carolina Chanticleer | 2025-10-18 | $60.77 | $96.00 | $35.23 | 36.7% | 2025-09-19 10:00 | 2025-08-30 12:00 | 478.00 |
| James Madison Dukes vs Old Dominion Monarch | 2025-10-18 | $105.06 | $77.00 | $28.06 | 36.4% | 2025-09-27 18:00 | 2025-09-26 18:00 | 24.00 |
| BYU Cougars vs Utah Ute | 2025-10-18 | $243.46 | $217.00 | $26.46 | 12.2% | 2025-10-17 23:00 | 2025-08-30 00:00 | 1175.00 |
| Arizona State Sun Devils vs Texas Tech Red Raider | 2025-10-18 | $59.06 | $34.00 | $25.06 | 73.7% | 2025-10-05 08:00 | 2025-08-24 01:06 | 1014.90 |
| Rutgers Scarlet Knights vs Oregon Duck | 2025-10-18 | $64.04 | $41.00 | $23.04 | 56.2% | 2025-09-28 21:00 | 2025-09-25 12:00 | 81.00 |
| Bowling Green State Falcons vs Central Michigan Chippewa | 2025-10-18 | $4.16 | $21.00 | $16.84 | 80.2% | 2025-10-02 09:00 | 2025-09-01 12:00 | 741.00 |
| Colorado State Rams vs Hawaii Rainbow Warrior | 2025-10-18 | $21.10 | $35.00 | $13.90 | 39.7% | 2025-10-16 08:00 | 2025-10-06 12:00 | 236.00 |
| Florida Gators vs Mississippi State Bulldog | 2025-10-18 | $69.17 | $56.00 | $13.17 | 23.5% | 2025-10-18 02:00 | 2025-09-24 12:00 | 566.00 |
| Clemson Tigers vs Southern Methodist (SMU) Mustang | 2025-10-18 | $43.94 | $31.00 | $12.94 | 41.7% | 2025-10-11 21:00 | 2025-10-06 12:00 | 129.00 |
| Oklahoma State Cowboys vs Cincinnati Bearcat | 2025-10-18 | $81.56 | $94.00 | $12.44 | 13.2% | 2025-10-09 13:00 | 2025-10-06 12:00 | 73.00 |
| Tulane Green Wave vs Army West Point Black Knight | 2025-10-18 | $20.74 | $10.00 | $10.74 | 107.4% | 2025-10-18 02:00 | 2025-10-06 12:00 | 278.00 |
| Marshall Thundering Herd vs Texas State Bobcat | 2025-10-18 | $21.10 | $11.00 | $10.10 | 91.8% | 2025-10-16 08:00 | 2025-10-06 12:00 | 236.00 |
| Ball State Cardinals vs Akron Zip | 2025-10-18 | $29.80 | $20.00 | $9.80 | 49.0% | 2025-10-13 16:00 | 2025-10-06 12:00 | 172.00 |
| UCF Knights vs West Virginia Mountaineer | 2025-10-18 | $37.12 | $28.00 | $9.12 | 32.6% | 2025-09-19 10:00 | 2025-10-02 20:24 | 322.41 |
| TCU Horned Frogs vs Baylor Bear | 2025-10-18 | $82.50 | $74.00 | $8.50 | 11.5% | 2025-09-26 00:00 | 2025-10-06 12:00 | 252.00 |
| Syracuse Orange vs Pittsburgh Panther | 2025-10-18 | $14.50 | $6.00 | $8.50 | 141.7% | 2025-09-18 04:00 | 2025-09-18 06:00 | 2.00 |
| Michigan Wolverines vs Washington Huskie | 2025-10-18 | $98.29 | $90.00 | $8.29 | 9.2% | 2025-10-16 15:00 | 2025-10-03 00:00 | 327.00 |
| UCLA Bruins vs Maryland Terrapin | 2025-10-18 | $24.73 | $17.00 | $7.73 | 45.5% | 2025-10-02 23:00 | 2025-10-03 00:00 | 1.00 |
| Louisiana Lafayette Ragin Cajuns vs Southern Miss Golden Eagle | 2025-10-18 | $29.88 | $37.00 | $7.12 | 19.2% | 2025-10-17 06:00 | 2025-10-06 12:00 | 258.00 |
| UAB Blazers vs Memphis Tiger | 2025-10-18 | $22.04 | $15.00 | $7.04 | 46.9% | 2025-10-17 06:00 | 2025-10-06 12:00 | 258.00 |
| Indiana Hoosiers vs Michigan State Spartan | 2025-10-18 | $82.98 | $76.00 | $6.98 | 9.2% | 2025-09-18 13:00 | 2025-09-15 16:30 | 68.49 |
| Auburn Tigers vs Missouri Tiger | 2025-10-18 | $51.83 | $58.00 | $6.17 | 10.6% | 2025-10-18 02:00 | 2025-10-06 12:00 | 278.00 |
| Boise State Broncos vs UNLV Rebel | 2025-10-18 | $74.37 | $69.00 | $5.37 | 7.8% | 2025-09-20 13:00 | 2025-09-15 16:30 | 116.49 |
| Boston College Eagles vs UConn Huskie | 2025-10-18 | $11.04 | $6.00 | $5.04 | 84.0% | 2025-10-10 06:00 | 2025-08-30 06:00 | 984.00 |
| Northwestern Wildcats vs Purdue Boilermaker | 2025-10-18 | $110.17 | $115.00 | $4.83 | 4.2% | 2025-10-16 00:00 | 2025-09-24 06:00 | 522.00 |
| Miami (OH) RedHawks vs Eastern Michigan Eagle | 2025-10-18 | $26.51 | $30.00 | $3.49 | 11.6% | 2025-10-04 10:00 | 2025-10-06 08:03 | 46.06 |
| UMass Minutemen vs Buffalo Bull | 2025-10-18 | $34.81 | $32.00 | $2.81 | 8.8% | 2025-10-18 02:00 | 2025-10-06 12:00 | 278.00 |
| Louisiana Monroe Warhawks vs Troy Trojan | 2025-10-18 | $21.09 | $19.00 | $2.09 | 11.0% | 2025-10-01 05:00 | 2025-10-08 18:00 | 181.00 |
| Duke Blue Devils vs Georgia Tech Yellow Jacket | 2025-10-18 | $21.86 | $20.00 | $1.86 | 9.3% | 2025-09-18 09:00 | 2025-09-22 18:00 | 105.00 |
| South Florida Bulls vs Florida Atlantic Owl | 2025-10-18 | $18.63 | $17.00 | $1.63 | 9.6% | 2025-09-27 00:00 | 2025-08-28 18:00 | 702.00 |
| Toledo Rockets vs Kent State Golden Flashe | 2025-10-18 | $23.90 | $25.00 | $1.10 | 4.4% | 2025-10-16 23:00 | 2025-10-06 12:00 | 251.00 |
| Georgia Southern Eagles vs Georgia State Panther | 2025-10-18 | $21.90 | $23.00 | $1.10 | 4.8% | 2025-10-14 23:00 | 2025-10-03 06:00 | 281.00 |
| Vanderbilt Commodores vs LSU Tiger | 2025-10-18 | $151.96 | $151.00 | $0.96 | 0.6% | 2025-10-10 11:00 | 2025-09-10 00:00 | 731.00 |
| New Mexico Lobos vs Nevada Wolf Pack | 2025-10-18 | $21.28 | $22.00 | $0.72 | 3.3% | 2025-09-24 00:00 | 2025-09-14 20:35 | 219.40 |
| Houston Cougars vs Arizona Wildcat | 2025-10-18 | $20.55 | $20.00 | $0.55 | 2.8% | 2025-10-01 06:00 | 2025-09-07 18:00 | 564.00 |
| Ohio Bobcats vs Northern Illinois Huskie | 2025-10-18 | $4.16 | $4.00 | $0.16 | 4.0% | 2025-10-02 09:00 | 2025-10-02 20:24 | 11.41 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 69% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.