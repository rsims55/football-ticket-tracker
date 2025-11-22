# 📈 Weekly Ticket Price Model Report
**Date:** 2025-11-22

## 🔍 Best Predictors of Ticket Price

❌ Model does not expose feature_importances_.

### ⚠️ Advanced diagnostics skipped
Reason: Could not recover original feature names from the model/pipeline.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **103**
- MAE (price): **$19.42**
- RMSE (price): **$27.06**
- Games > 5% price error: **100 / 103**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **518.84 h**  •  Median |Δ|: **280.00 h**
- Within 6h: **4/103**  •  Within 12h: **5/103**  •  Within 24h: **6/103**
- Bias: predictions are on average **332.73 h later than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Missouri Tigers vs Mississippi State Bulldog | 2025-11-15 | $140.80 | $40.00 | $100.80 | 252.0% | 2025-11-14 23:00 | 2025-11-12 00:00 | 71.00 |
| James Madison Dukes vs Washington State Cougar | 2025-11-22 | $104.39 | $33.00 | $71.39 | 216.3% | 2025-11-21 15:00 | 2025-10-26 18:00 | 621.00 |
| Florida Gators vs Tennessee Volunteer | 2025-11-22 | $149.40 | $80.00 | $69.40 | 86.8% | 2025-11-21 23:00 | 2025-11-09 12:00 | 299.00 |
| LSU Tigers vs Arkansas Razorback | 2025-11-15 | $74.15 | $5.00 | $69.15 | 1383.0% | 2025-11-14 12:00 | 2025-11-12 00:00 | 60.00 |
| Alabama Crimson Tide vs Oklahoma Sooner | 2025-11-15 | $178.85 | $120.00 | $58.85 | 49.0% | 2025-10-16 06:00 | 2025-11-12 00:00 | 642.00 |
| North Carolina Tar Heels vs Duke Blue Devil | 2025-11-22 | $102.99 | $47.00 | $55.99 | 119.1% | 2025-11-05 00:00 | 2025-11-11 06:00 | 150.00 |
| Virginia Tech Hokies vs Miami Hurricane | 2025-11-22 | $61.50 | $7.00 | $54.50 | 778.6% | 2025-10-24 03:00 | 2025-11-11 18:00 | 447.00 |
| Wyoming Cowboys vs Nevada Wolf Pack | 2025-11-22 | $90.01 | $36.00 | $54.01 | 150.0% | 2025-11-22 02:00 | 2025-08-28 06:00 | 2060.00 |
| Texas A&M Aggies vs South Carolina Gamecock | 2025-11-15 | $73.46 | $125.00 | $51.54 | 41.2% | 2025-10-23 00:00 | 2025-09-27 12:00 | 612.00 |
| Cincinnati Bearcats vs BYU Cougar | 2025-11-22 | $96.86 | $46.00 | $50.86 | 110.6% | 2025-11-21 18:00 | 2025-09-01 12:00 | 1950.00 |
| Northwestern Wildcats vs Minnesota Golden Gopher | 2025-11-22 | $82.88 | $34.00 | $48.88 | 143.8% | 2025-11-20 00:00 | 2025-11-11 12:00 | 204.00 |
| Ohio State Buckeyes vs Rutgers Scarlet Knight | 2025-11-22 | $77.57 | $30.00 | $47.57 | 158.6% | 2025-10-24 23:00 | 2025-10-29 12:00 | 109.00 |
| Florida State Seminoles vs Virginia Tech Hokie | 2025-11-15 | $57.62 | $13.00 | $44.62 | 343.2% | 2025-10-24 18:00 | 2025-10-31 12:00 | 162.00 |
| San Diego State Aztecs vs Boise State Bronco | 2025-11-15 | $67.46 | $29.00 | $38.46 | 132.6% | 2025-11-15 00:00 | 2025-11-12 00:00 | 72.00 |
| Penn State Nittany Lions vs Nebraska Cornhusker | 2025-11-22 | $43.94 | $8.00 | $35.94 | 449.2% | 2025-11-15 21:00 | 2025-11-08 00:00 | 189.00 |
| Michigan State Spartans vs Penn State Nittany Lion | 2025-11-15 | $61.02 | $26.00 | $35.02 | 134.7% | 2025-11-15 00:00 | 2025-11-12 00:00 | 72.00 |
| Oklahoma State Cowboys vs Kansas State Wildcat | 2025-11-15 | $68.39 | $34.00 | $34.39 | 101.1% | 2025-11-06 18:00 | 2025-11-11 18:00 | 120.00 |
| Texas State Bobcats vs Louisiana Monroe Warhawk | 2025-11-22 | $44.25 | $10.00 | $34.25 | 342.5% | 2025-11-22 02:00 | 2025-11-11 06:00 | 260.00 |
| Tennessee Volunteers vs New Mexico State Aggie | 2025-11-15 | $81.15 | $47.00 | $34.15 | 72.7% | 2025-11-15 11:00 | 2025-11-11 18:00 | 89.00 |
| Iowa State Cyclones vs Kansas Jayhawk | 2025-11-22 | $57.81 | $25.00 | $32.81 | 131.2% | 2025-11-20 23:00 | 2025-11-09 18:00 | 269.00 |
| Texas Tech Red Raiders vs UCF Knight | 2025-11-15 | $68.70 | $36.00 | $32.70 | 90.8% | 2025-11-06 18:00 | 2025-08-31 12:00 | 1614.00 |
| Ohio State Buckeyes vs UCLA Bruin | 2025-11-15 | $78.83 | $47.00 | $31.83 | 67.7% | 2025-10-17 23:00 | 2025-11-09 18:00 | 547.00 |
| Colorado Buffaloes vs Arizona State Sun Devil | 2025-11-22 | $77.16 | $47.00 | $30.16 | 64.2% | 2025-11-08 23:00 | 2025-11-12 00:00 | 73.00 |
| Texas Longhorns vs Arkansas Razorback | 2025-11-22 | $148.12 | $118.00 | $30.12 | 25.5% | 2025-11-21 12:00 | 2025-10-26 12:00 | 624.00 |
| Oregon Ducks vs USC Trojan | 2025-11-22 | $74.11 | $104.00 | $29.89 | 28.7% | 2025-10-23 05:00 | 2025-09-01 12:00 | 1241.00 |
| Ball State Cardinals vs Eastern Michigan Eagle | 2025-11-15 | $29.79 | $3.00 | $26.79 | 893.0% | 2025-11-10 16:00 | 2025-10-11 12:00 | 724.00 |
| Georgia Bulldogs vs Texas Longhorn | 2025-11-15 | $165.26 | $139.00 | $26.26 | 18.9% | 2025-10-16 04:00 | 2025-08-28 06:00 | 1174.00 |
| Maryland Terrapins vs Michigan Wolverine | 2025-11-22 | $43.23 | $69.00 | $25.77 | 37.3% | 2025-10-25 09:00 | 2025-09-13 10:23 | 1006.61 |
| Northwestern Wildcats vs Michigan Wolverine | 2025-11-15 | $155.24 | $130.00 | $25.24 | 19.4% | 2025-11-08 18:00 | 2025-11-08 06:00 | 12.00 |
| Georgia Tech Yellow Jackets vs Pittsburgh Panther | 2025-11-22 | $34.16 | $9.00 | $25.16 | 279.6% | 2025-10-28 03:00 | 2025-09-10 00:00 | 1155.00 |
| Stanford Cardinal vs California Golden Bear | 2025-11-22 | $92.03 | $67.00 | $25.03 | 37.4% | 2025-10-24 00:00 | 2025-08-21 18:26 | 1517.57 |
| Arizona State Sun Devils vs West Virginia Mountaineer | 2025-11-15 | $43.19 | $19.00 | $24.19 | 127.3% | 2025-11-01 23:00 | 2025-08-27 15:51 | 1591.14 |
| Baylor Bears vs Utah Ute | 2025-11-15 | $39.58 | $16.00 | $23.58 | 147.4% | 2025-11-07 10:00 | 2025-11-06 12:00 | 22.00 |
| Illinois Fighting Illini vs Maryland Terrapin | 2025-11-15 | $25.52 | $5.00 | $20.52 | 410.4% | 2025-10-16 09:00 | 2025-11-08 18:00 | 561.00 |
| UCF Knights vs Oklahoma State Cowboy | 2025-11-22 | $24.94 | $5.00 | $19.94 | 398.8% | 2025-10-24 10:00 | 2025-11-09 00:00 | 374.00 |
| Arizona Wildcats vs Baylor Bear | 2025-11-22 | $23.83 | $4.00 | $19.83 | 495.7% | 2025-11-21 06:00 | 2025-11-11 12:00 | 234.00 |
| Mississippi Rebels vs Florida Gator | 2025-11-15 | $109.23 | $90.00 | $19.23 | 21.4% | 2025-10-23 13:00 | 2025-10-25 00:00 | 35.00 |
| LSU Tigers vs Western Kentucky Hilltopper | 2025-11-22 | $21.53 | $3.00 | $18.53 | 617.7% | 2025-11-04 01:00 | 2025-11-05 18:58 | 41.98 |
| Iowa Hawkeyes vs Michigan State Spartan | 2025-11-22 | $41.47 | $23.00 | $18.47 | 80.3% | 2025-11-21 18:00 | 2025-11-02 00:00 | 474.00 |
| Cincinnati Bearcats vs Arizona Wildcat | 2025-11-15 | $42.23 | $24.00 | $18.23 | 76.0% | 2025-10-18 21:00 | 2025-11-11 18:00 | 573.00 |
| San Diego State Aztecs vs San Jose State Spartan | 2025-11-22 | $40.08 | $22.00 | $18.08 | 82.2% | 2025-10-23 05:00 | 2025-09-27 12:00 | 617.00 |
| Miami Hurricanes vs North Carolina State Wolfpack | 2025-11-15 | $20.87 | $3.00 | $17.87 | 595.7% | 2025-10-18 04:00 | 2025-11-11 18:00 | 590.00 |
| Tulsa Golden Hurricane vs Oregon State Beaver | 2025-11-15 | $19.64 | $3.00 | $16.64 | 554.7% | 2025-11-14 18:00 | 2025-11-12 00:00 | 66.00 |
| Boise State Broncos vs Colorado State Ram | 2025-11-22 | $46.42 | $30.00 | $16.42 | 54.7% | 2025-11-21 18:00 | 2025-10-26 00:00 | 642.00 |
| Charlotte 49ers vs UTSA Roadrunner | 2025-11-15 | $17.94 | $2.00 | $15.94 | 797.0% | 2025-10-30 04:00 | 2025-10-30 06:00 | 2.00 |
| UAB Blazers vs North Texas Mean Green | 2025-11-15 | $24.99 | $10.00 | $14.99 | 149.9% | 2025-11-14 06:00 | 2025-11-12 00:00 | 54.00 |
| UTSA Roadrunners vs East Carolina Pirate | 2025-11-22 | $22.01 | $8.00 | $14.01 | 175.1% | 2025-11-13 14:00 | 2025-10-27 12:00 | 410.00 |
| Utah Utes vs Kansas State Wildcat | 2025-11-22 | $68.91 | $55.00 | $13.91 | 25.3% | 2025-11-08 23:00 | 2025-11-11 18:00 | 67.00 |
| Appalachian State Mountaineers vs Marshall Thundering Herd | 2025-11-22 | $44.25 | $57.00 | $12.75 | 22.4% | 2025-11-22 02:00 | 2025-11-06 12:00 | 374.00 |
| South Alabama Jaguars vs Southern Miss Golden Eagle | 2025-11-22 | $35.47 | $23.00 | $12.47 | 54.2% | 2025-10-25 07:00 | 2025-09-26 00:00 | 703.00 |
| Clemson Tigers vs Furman Paladin | 2025-11-22 | $19.34 | $7.00 | $12.34 | 176.3% | 2025-11-11 15:00 | 2025-11-08 06:00 | 81.00 |
| Wisconsin Badgers vs Illinois Fighting Illini | 2025-11-22 | $19.89 | $8.00 | $11.89 | 148.6% | 2025-10-30 10:00 | 2025-11-05 18:58 | 152.98 |
| Kentucky Wildcats vs Tennessee Tech Golden Eagle | 2025-11-15 | $20.61 | $9.00 | $11.61 | 129.0% | 2025-11-15 00:00 | 2025-11-10 18:00 | 102.00 |
| East Carolina Pirates vs Memphis Tiger | 2025-11-15 | $24.99 | $14.00 | $10.99 | 78.5% | 2025-10-21 05:00 | 2025-10-13 18:00 | 179.00 |
| Texas A&M Aggies vs Samford Bulldog | 2025-11-22 | $55.96 | $45.00 | $10.96 | 24.4% | 2025-11-22 00:00 | 2025-09-10 00:00 | 1752.00 |
| UNLV Rebels vs Utah State Aggie | 2025-11-15 | $30.67 | $20.00 | $10.67 | 53.4% | 2025-11-12 18:00 | 2025-09-07 12:00 | 1590.00 |
| Oklahoma Sooners vs Missouri Tiger | 2025-11-22 | $42.65 | $32.00 | $10.65 | 33.3% | 2025-11-16 01:00 | 2025-10-30 06:00 | 403.00 |
| Buffalo Bulls vs Miami (OH) RedHawk | 2025-11-19 | $27.45 | $17.00 | $10.45 | 61.5% | 2025-11-19 08:00 | 2025-11-12 00:00 | 176.00 |
| Vanderbilt Commodores vs Kentucky Wildcat | 2025-11-22 | $46.42 | $36.00 | $10.42 | 28.9% | 2025-11-21 18:00 | 2025-09-13 10:23 | 1663.61 |
| Georgia Southern Eagles vs Old Dominion Monarch | 2025-11-22 | $22.27 | $12.00 | $10.27 | 85.6% | 2025-11-21 18:00 | 2025-11-12 00:00 | 234.00 |
| Missouri State Bears vs UTEP Miner | 2025-11-15 | $24.14 | $14.00 | $10.14 | 72.4% | 2025-11-14 14:00 | 2025-11-11 12:00 | 74.00 |
| James Madison Dukes vs Appalachian State Mountaineer | 2025-11-15 | $80.87 | $71.00 | $9.87 | 13.9% | 2025-10-16 08:00 | 2025-10-03 12:00 | 308.00 |
| Northern Illinois Huskies vs Western Michigan Bronco | 2025-11-18 | $25.87 | $16.00 | $9.87 | 61.7% | 2025-11-05 10:00 | 2025-11-10 12:00 | 122.00 |
| Boston College Eagles vs Georgia Tech Yellow Jacket | 2025-11-15 | $12.73 | $3.00 | $9.73 | 324.3% | 2025-10-16 18:00 | 2025-11-10 18:00 | 600.00 |
| Washington State Cougars vs Louisiana Tech Bulldog | 2025-11-15 | $21.60 | $12.00 | $9.60 | 80.0% | 2025-10-17 11:00 | 2025-11-11 12:00 | 601.00 |
| Florida Atlantic Owls vs UConn Huskie | 2025-11-22 | $15.41 | $6.00 | $9.41 | 156.8% | 2025-11-21 04:00 | 2025-10-23 06:00 | 694.00 |
| UAB Blazers vs South Florida Bull | 2025-11-22 | $26.74 | $18.00 | $8.74 | 48.6% | 2025-11-13 12:00 | 2025-11-06 18:00 | 162.00 |
| Houston Cougars vs TCU Horned Frog | 2025-11-22 | $31.71 | $23.00 | $8.71 | 37.9% | 2025-10-27 00:00 | 2025-10-31 06:00 | 102.00 |
| Tulane Green Wave vs Florida Atlantic Owl | 2025-11-15 | $47.69 | $56.00 | $8.31 | 14.8% | 2025-11-14 04:00 | 2025-11-07 18:00 | 154.00 |
| USC Trojans vs Iowa Hawkeye | 2025-11-15 | $39.28 | $31.00 | $8.28 | 26.7% | 2025-10-31 12:00 | 2025-09-06 12:00 | 1320.00 |
| UNLV Rebels vs Hawaii Rainbow Warrior | 2025-11-21 | $41.82 | $34.00 | $7.82 | 23.0% | 2025-11-18 23:00 | 2025-08-29 18:00 | 1949.00 |
| Toledo Rockets vs Ball State Cardinal | 2025-11-22 | $28.63 | $21.00 | $7.63 | 36.3% | 2025-11-05 10:00 | 2025-11-10 18:00 | 128.00 |
| Indiana Hoosiers vs Wisconsin Badger | 2025-11-15 | $52.50 | $60.00 | $7.50 | 12.5% | 2025-11-05 23:00 | 2025-08-30 06:00 | 1625.00 |
| Nevada Wolf Pack vs San Jose State Spartan | 2025-11-15 | $18.33 | $11.00 | $7.33 | 66.6% | 2025-10-25 06:00 | 2025-10-27 00:00 | 42.00 |
| Southern Methodist (SMU) Mustangs vs Louisville Cardinal | 2025-11-22 | $19.23 | $13.00 | $6.23 | 47.9% | 2025-10-23 06:00 | 2025-10-31 00:00 | 186.00 |
| Louisiana Monroe Warhawks vs South Alabama Jaguar | 2025-11-15 | $21.06 | $15.00 | $6.06 | 40.4% | 2025-10-29 05:00 | 2025-11-10 12:00 | 295.00 |
| Duke Blue Devils vs Virginia Cavalier | 2025-11-15 | $20.87 | $15.00 | $5.87 | 39.1% | 2025-10-16 15:00 | 2025-11-07 18:00 | 531.00 |
| BYU Cougars vs TCU Horned Frog | 2025-11-15 | $59.16 | $65.00 | $5.84 | 9.0% | 2025-11-13 23:00 | 2025-09-01 12:00 | 1763.00 |
| Georgia State Panthers vs Marshall Thundering Herd | 2025-11-15 | $16.80 | $11.00 | $5.80 | 52.7% | 2025-11-14 16:00 | 2025-11-10 12:00 | 100.00 |
| Sam Houston State Bearkats vs Delaware Blue Hen | 2025-11-15 | $19.65 | $14.00 | $5.65 | 40.4% | 2025-10-18 17:00 | 2025-10-18 18:00 | 1.00 |
| Pittsburgh Panthers vs Notre Dame Fighting Irish | 2025-11-15 | $80.86 | $86.00 | $5.14 | 6.0% | 2025-11-01 15:00 | 2025-10-21 12:00 | 267.00 |
| Arkansas State Red Wolves vs Louisiana Lafayette Ragin Cajun | 2025-11-20 | $15.98 | $11.00 | $4.98 | 45.3% | 2025-10-27 00:00 | 2025-11-01 18:00 | 138.00 |
| Georgia Southern Eagles vs Coastal Carolina Chanticleer | 2025-11-15 | $16.80 | $12.00 | $4.80 | 40.0% | 2025-11-14 16:00 | 2025-11-11 12:00 | 76.00 |
| UConn Huskies vs Air Force Falcon | 2025-11-15 | $18.78 | $14.00 | $4.78 | 34.1% | 2025-11-15 00:00 | 2025-08-24 01:06 | 1990.90 |
| North Carolina State Wolfpack vs Florida State Seminole | 2025-11-21 | $72.50 | $68.00 | $4.50 | 6.6% | 2025-10-31 18:00 | 2025-08-30 12:00 | 1494.00 |
| Florida International Panthers vs Liberty Flame | 2025-11-15 | $13.73 | $10.00 | $3.73 | 37.3% | 2025-10-17 03:00 | 2025-11-05 18:58 | 471.98 |
| Southern Miss Golden Eagles vs Texas State Bobcat | 2025-11-15 | $7.60 | $4.00 | $3.60 | 90.0% | 2025-11-08 23:00 | 2025-10-27 06:00 | 305.00 |
| Fresno State Bulldogs vs Wyoming Cowboy | 2025-11-15 | $20.59 | $17.00 | $3.59 | 21.1% | 2025-10-22 00:00 | 2025-09-23 00:00 | 696.00 |
| Wake Forest Demon Deacons vs North Carolina Tar Heel | 2025-11-15 | $28.26 | $31.00 | $2.74 | 8.8% | 2025-11-11 03:00 | 2025-08-24 01:06 | 1897.90 |
| Ohio Bobcats vs UMass Minutemen | 2025-11-18 | $4.73 | $2.00 | $2.73 | 136.5% | 2025-11-03 06:00 | 2025-11-02 00:00 | 30.00 |
| Kent State Golden Flashes vs Central Michigan Chippewa | 2025-11-19 | $13.70 | $11.00 | $2.70 | 24.5% | 2025-10-28 03:00 | 2025-11-11 18:00 | 351.00 |
| South Carolina Gamecocks vs Coastal Carolina Chanticleer | 2025-11-22 | $34.65 | $32.00 | $2.65 | 8.3% | 2025-10-25 23:00 | 2025-11-09 06:00 | 343.00 |
| Rice Owls vs North Texas Mean Green | 2025-11-22 | $20.36 | $23.00 | $2.64 | 11.5% | 2025-10-25 00:00 | 2025-11-11 18:00 | 426.00 |
| Navy Midshipmen vs South Florida Bull | 2025-11-15 | $40.45 | $38.00 | $2.45 | 6.4% | 2025-11-15 00:00 | 2025-11-11 18:00 | 78.00 |
| New Mexico Lobos vs Colorado State Ram | 2025-11-15 | $20.59 | $23.00 | $2.41 | 10.5% | 2025-10-22 00:00 | 2025-10-27 18:00 | 138.00 |
| Western Kentucky Hilltoppers vs Middle Tennessee Blue Raider | 2025-11-15 | $11.30 | $9.00 | $2.30 | 25.6% | 2025-11-03 19:00 | 2025-11-02 00:00 | 43.00 |
| Washington Huskies vs Purdue Boilermaker | 2025-11-15 | $12.04 | $10.00 | $2.04 | 20.4% | 2025-10-16 20:00 | 2025-10-28 12:00 | 280.00 |
| Temple Owls vs Tulane Green Wave | 2025-11-22 | $6.00 | $4.00 | $2.00 | 50.0% | 2025-11-18 09:00 | 2025-09-27 12:00 | 1245.00 |
| Bowling Green State Falcons vs Akron Zip | 2025-11-18 | $4.73 | $6.00 | $1.27 | 21.2% | 2025-11-03 06:00 | 2025-11-10 18:00 | 180.00 |
| Wake Forest Demon Deacons vs Delaware Blue Hen | 2025-11-22 | $6.27 | $5.00 | $1.27 | 25.4% | 2025-11-01 06:00 | 2025-10-22 18:00 | 228.00 |
| Jacksonville State Gamecocks vs Kennesaw State Owl | 2025-11-15 | $30.93 | $32.00 | $1.07 | 3.3% | 2025-11-11 02:00 | 2025-11-11 00:00 | 2.00 |
| UCLA Bruins vs Washington Huskie | 2025-11-22 | $25.55 | $25.00 | $0.55 | 2.2% | 2025-11-11 00:00 | 2025-11-07 06:00 | 90.00 |
| Troy Trojans vs Georgia State Panther | 2025-11-22 | $15.52 | $16.00 | $0.48 | 3.0% | 2025-10-24 10:00 | 2025-10-24 12:00 | 2.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 65% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.