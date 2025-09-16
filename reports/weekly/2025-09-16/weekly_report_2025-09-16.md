# 📈 Weekly Ticket Price Model Report
**Date:** 2025-09-16

## 🔍 Best Predictors of Ticket Price

### Top Transformed Features (expanded)
- days until game was important, contributing 35.1% to predictions.
- isRankedMatchup was important, contributing 20.3% to predictions.
- capacity was important, contributing 14.2% to predictions.
- week was important, contributing 8.0% to predictions.
- awayTeamRank was important, contributing 6.3% to predictions.
- isRivalry was important, contributing 2.3% to predictions.
- homeTeamRank was important, contributing 2.1% to predictions.
- collectionSlot 12:00 category influenced predictions (~2.0%).
- Teams from the SEC awayconference mattered, contributing 1.2%.
- neutralSite was important, contributing 1.1% to predictions.
- Teams from the Big Ten awayconference mattered, contributing 1.0%.
- collectionSlot 00:00 category influenced predictions (~1.0%).
- collectionSlot 18:00 category influenced predictions (~0.6%).
- collectionSlot 06:00 category influenced predictions (~0.6%).
- Teams from the Big 12 awayconference mattered, contributing 0.6%.
- Teams from the Big 12 homeconference mattered, contributing 0.5%.
- conferenceGame was important, contributing 0.4% to predictions.
- Teams from the SEC homeconference mattered, contributing 0.4%.
- Teams from the ACC awayconference mattered, contributing 0.4%.
- Teams from the Mountain West awayconference mattered, contributing 0.3%.

### Aggregated by Original Column
- days_until: 0.3505
- isRankedMatchup: 0.2028
- capacity: 0.1419
- week: 0.0804
- awayTeamRank: 0.0635
- collectionSlot: 0.0428
- awayConference: 0.0407
- isRivalry: 0.0229
- homeTeamRank: 0.0212
- homeConference: 0.0181
- neutralSite: 0.0107
- conferenceGame: 0.0044

**Possibly unrelated (near-zero importance):** conferenceGame

### ⚠️ Advanced diagnostics skipped
Reason: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.

Detailed tracebacks of the workers should have been printed to stderr in the executor process if faulthandler was not disabled.

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **72**
- MAE (price): **$10.32**
- RMSE (price): **$26.06**
- Games > 5% price error: **54 / 72**

### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)
- MAE (hours): **82.04 h**  •  Median |Δ|: **22.39 h**
- Within 6h: **22/72**  •  Within 12h: **22/72**  •  Within 24h: **46/72**
- Bias: predictions are on average **59.01 h earlier than** actual lows

## 🎯 Predicted vs Actual Prices & Timing

| Game | Date (ET) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt (ET) | Actual Low (ET) | Abs Δ (h) |
|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|
| Notre Dame Fighting Irish vs Texas A&M Aggie | 2025-09-13 | $312.79 | $130.00 | $182.79 | 140.6% | 2025-09-11 12:00 | 2025-09-13 10:23 | 46.39 |
| Northwestern Wildcats vs Oregon Duck | 2025-09-13 | $169.15 | $103.00 | $66.15 | 64.2% | 2025-09-12 00:00 | 2025-09-13 10:23 | 34.39 |
| LSU Tigers vs Florida Gator | 2025-09-13 | $115.88 | $68.00 | $47.88 | 70.4% | 2025-09-09 00:00 | 2025-09-13 10:23 | 106.39 |
| Mississippi Rebels vs Arkansas Razorback | 2025-09-13 | $13.81 | $61.00 | $47.19 | 77.4% | 2025-09-07 00:00 | 2025-09-13 10:23 | 154.39 |
| South Carolina Gamecocks vs Vanderbilt Commodore | 2025-09-13 | $66.88 | $101.00 | $34.12 | 33.8% | 2025-09-12 00:00 | 2025-09-13 10:23 | 34.39 |
| Alabama Crimson Tide vs Wisconsin Badger | 2025-09-13 | $37.89 | $6.00 | $31.89 | 531.5% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| West Virginia Mountaineers vs Pittsburgh Panther | 2025-09-13 | $159.60 | $129.00 | $30.60 | 23.7% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Ohio State Buckeyes vs Ohio Bobcat | 2025-09-13 | $100.44 | $74.00 | $26.44 | 35.7% | 2025-08-30 00:00 | 2025-09-06 06:00 | 174.00 |
| Wyoming Cowboys vs Utah Ute | 2025-09-13 | $67.27 | $45.00 | $22.27 | 49.5% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Auburn Tigers vs South Alabama Jaguar | 2025-09-13 | $48.43 | $30.00 | $18.43 | 61.4% | 2025-08-16 00:00 | 2025-09-13 10:23 | 682.39 |
| Texas Longhorns vs UTEP Miner | 2025-09-13 | $55.76 | $39.00 | $16.76 | 43.0% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Missouri Tigers vs Louisiana Lafayette Ragin Cajun | 2025-09-13 | $35.13 | $19.00 | $16.13 | 84.9% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Coastal Carolina Chanticleers vs East Carolina Pirate | 2025-09-13 | $53.43 | $40.00 | $13.43 | 33.6% | 2025-09-12 00:00 | 2025-09-13 10:23 | 34.39 |
| Tennessee Volunteers vs Georgia Bulldog | 2025-09-13 | $306.41 | $293.00 | $13.41 | 4.6% | 2025-08-28 06:00 | 2025-08-28 06:00 | 0.00 |
| Nebraska Cornhuskers vs Houston Christian Huskie | 2025-09-13 | $19.08 | $7.00 | $12.08 | 172.6% | 2025-08-28 18:00 | 2025-09-13 10:23 | 376.39 |
| Michigan Wolverines vs Central Michigan Chippewa | 2025-09-13 | $75.51 | $66.00 | $9.51 | 14.4% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Michigan State Spartans vs Youngstown State Penguin | 2025-09-13 | $43.13 | $35.00 | $8.13 | 23.2% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Temple Owls vs Oklahoma Sooner | 2025-09-13 | $21.90 | $30.00 | $8.10 | 27.0% | 2025-09-06 18:00 | 2025-09-13 10:23 | 160.39 |
| Missouri State Bears vs Southern Methodist (SMU) Mustang | 2025-09-13 | $29.00 | $21.00 | $8.00 | 38.1% | 2025-09-06 06:00 | 2025-09-06 00:00 | 6.00 |
| Illinois Fighting Illini vs Western Michigan Bronco | 2025-09-13 | $23.91 | $16.00 | $7.91 | 49.4% | 2025-08-27 18:00 | 2025-09-13 10:23 | 400.39 |
| Arkansas State Red Wolves vs Iowa State Cyclone | 2025-09-13 | $34.71 | $27.00 | $7.71 | 28.6% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Ohio State Buckeyes vs Ohio Bobcat | 2025-09-13 | $100.44 | $93.00 | $7.44 | 8.0% | 2025-08-30 00:00 | 2025-09-08 18:00 | 234.00 |
| Texas Tech Red Raiders vs Oregon State Beaver | 2025-09-13 | $26.94 | $20.00 | $6.94 | 34.7% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Iowa Hawkeyes vs UMass Minutemen | 2025-09-13 | $19.30 | $13.00 | $6.30 | 48.5% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Purdue Boilermakers vs USC Trojan | 2025-09-13 | $23.19 | $17.00 | $6.19 | 36.4% | 2025-09-12 18:00 | 2025-08-27 14:45 | 387.24 |
| Arizona State Sun Devils vs Texas State Bobcat | 2025-09-13 | $10.54 | $5.00 | $5.54 | 110.8% | 2025-09-10 00:00 | 2025-08-28 18:00 | 294.00 |
| TCU Horned Frogs vs Abilene Christian Wildcat | 2025-09-13 | $33.14 | $28.00 | $5.14 | 18.4% | 2025-09-12 06:00 | 2025-09-13 10:23 | 28.39 |
| Hawaii Rainbow Warriors vs Portland State Viking | 2025-09-13 | $28.50 | $24.00 | $4.50 | 18.8% | 2025-08-25 18:00 | 2025-08-24 01:06 | 40.90 |
| North Carolina Tar Heels vs Richmond Spider | 2025-09-13 | $38.48 | $34.00 | $4.48 | 13.2% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| UAB Blazers vs Akron Zip | 2025-09-13 | $17.34 | $13.00 | $4.34 | 33.4% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Kennesaw State Owls vs Merrimack Warrior | 2025-09-13 | $16.79 | $21.00 | $4.21 | 20.0% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Cincinnati Bearcats vs Northwestern State Demon | 2025-09-13 | $13.20 | $9.00 | $4.20 | 46.7% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Florida International Panthers vs Florida Atlantic Owl | 2025-09-13 | $20.16 | $16.00 | $4.16 | 26.0% | 2025-08-14 06:00 | 2025-09-13 10:23 | 724.39 |
| Baylor Bears vs Samford Bulldog | 2025-09-13 | $15.02 | $11.00 | $4.02 | 36.5% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Bowling Green State Falcons vs Liberty Flame | 2025-09-13 | $24.03 | $28.00 | $3.97 | 14.2% | 2025-08-14 12:00 | 2025-08-27 14:44 | 314.75 |
| Mississippi State Bulldogs vs Alcorn State Brave | 2025-09-13 | $9.80 | $6.00 | $3.80 | 63.3% | 2025-09-07 00:00 | 2025-09-06 18:00 | 6.00 |
| Georgia Tech Yellow Jackets vs Clemson Tiger | 2025-09-13 | $70.67 | $67.00 | $3.67 | 5.5% | 2025-08-21 12:00 | 2025-08-21 16:43 | 4.72 |
| Rice Owls vs Prairie View A&M Panther | 2025-09-13 | $16.62 | $13.00 | $3.62 | 27.8% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Indiana Hoosiers vs Indiana State Sycamore | 2025-09-12 | $24.45 | $21.00 | $3.45 | 16.4% | 2025-09-07 06:00 | 2025-09-07 06:00 | 0.00 |
| California Golden Bears vs Minnesota Golden Gopher | 2025-09-13 | $23.16 | $20.00 | $3.16 | 15.8% | 2025-09-12 00:00 | 2025-09-13 10:23 | 34.39 |
| Utah State Aggies vs Air Force Falcon | 2025-09-13 | $30.25 | $28.00 | $2.25 | 8.0% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| North Texas Mean Green vs Washington State Cougar | 2025-09-13 | $19.20 | $17.00 | $2.20 | 12.9% | 2025-09-12 18:00 | 2025-09-13 10:23 | 16.39 |
| Tulsa Golden Hurricane vs Navy Midshipmen | 2025-09-13 | $16.11 | $14.00 | $2.11 | 15.1% | 2025-08-31 12:00 | 2025-09-13 10:23 | 310.39 |
| Troy Trojans vs Memphis Tiger | 2025-09-13 | $12.07 | $10.00 | $2.07 | 20.7% | 2025-09-09 00:00 | 2025-09-13 10:23 | 106.39 |
| Marshall Thundering Herd vs Eastern Kentucky Colonel | 2025-09-13 | $14.05 | $12.00 | $2.05 | 17.1% | 2025-09-09 18:00 | 2025-09-09 12:00 | 6.00 |
| Arizona Wildcats vs Kansas State Wildcat | 2025-09-12 | $10.96 | $9.00 | $1.96 | 21.8% | 2025-09-06 00:00 | 2025-09-06 00:00 | 0.00 |
| UTSA Roadrunners vs Incarnate Word Cardinal | 2025-09-13 | $16.95 | $15.00 | $1.95 | 13.0% | 2025-09-08 18:00 | 2025-09-08 18:00 | 0.00 |
| Louisiana Tech Bulldogs vs New Mexico State Aggie | 2025-09-13 | $17.57 | $16.00 | $1.57 | 9.8% | 2025-09-12 12:00 | 2025-09-13 10:23 | 22.39 |
| Georgia State Panthers vs Murray State Racer | 2025-09-13 | $7.16 | $6.00 | $1.16 | 19.3% | 2025-09-06 12:00 | 2025-09-06 12:00 | 0.00 |
| Ball State Cardinals vs New Hampshire Wildcat | 2025-09-13 | $16.06 | $15.00 | $1.06 | 7.1% | 2025-09-05 18:00 | 2025-09-05 18:00 | 0.00 |
| Penn State Nittany Lions vs Villanova Wildcat | 2025-09-13 | $34.92 | $34.00 | $0.92 | 2.7% | 2025-08-23 06:00 | 2025-09-10 00:00 | 426.00 |
| Stanford Cardinal vs Boston College Eagle | 2025-09-13 | $3.76 | $3.00 | $0.76 | 25.3% | 2025-08-25 06:00 | 2025-08-24 01:06 | 28.90 |
| Maryland Terrapins vs Towson Tiger | 2025-09-13 | $2.37 | $3.00 | $0.63 | 21.0% | 2025-09-03 12:00 | 2025-09-05 10:43 | 46.73 |
| Georgia Southern Eagles vs Jacksonville State Gamecock | 2025-09-13 | $28.38 | $29.00 | $0.62 | 2.1% | 2025-08-16 06:00 | 2025-08-24 01:06 | 187.10 |
| Kentucky Wildcats vs Eastern Michigan Eagle | 2025-09-13 | $17.59 | $17.00 | $0.59 | 3.5% | 2025-09-10 18:00 | 2025-09-10 00:00 | 18.00 |
| Virginia Tech Hokies vs Old Dominion Monarch | 2025-09-13 | $8.49 | $8.00 | $0.49 | 6.1% | 2025-09-06 06:00 | 2025-09-06 06:00 | 0.00 |
| Maryland Terrapins vs Towson Tiger | 2025-09-13 | $2.37 | $2.00 | $0.37 | 18.5% | 2025-09-03 12:00 | 2025-09-05 10:43 | 46.73 |
| Charlotte 49ers vs Monmouth Hawk | 2025-09-13 | $5.36 | $5.00 | $0.36 | 7.2% | 2025-09-07 18:00 | 2025-09-08 18:00 | 24.00 |
| Fresno State Bulldogs vs Southern Jaguar | 2025-09-13 | $14.29 | $14.00 | $0.29 | 2.1% | 2025-08-30 00:00 | 2025-08-30 00:00 | 0.00 |
| Nevada Wolf Pack vs Middle Tennessee Blue Raider | 2025-09-13 | $11.25 | $11.00 | $0.25 | 2.3% | 2025-09-09 06:00 | 2025-09-09 06:00 | 0.00 |
| Kent State Golden Flashes vs Buffalo Bull | 2025-09-13 | $14.24 | $14.00 | $0.24 | 1.7% | 2025-09-08 12:00 | 2025-09-08 12:00 | 0.00 |
| UCLA Bruins vs New Mexico Lobo | 2025-09-12 | $6.22 | $6.00 | $0.22 | 3.7% | 2025-09-10 18:00 | 2025-09-10 00:00 | 18.00 |
| Houston Cougars vs Colorado Buffaloe | 2025-09-12 | $53.22 | $53.00 | $0.22 | 0.4% | 2025-08-28 06:00 | 2025-08-28 06:00 | 0.00 |
| Toledo Rockets vs Morgan State Bear | 2025-09-13 | $18.20 | $18.00 | $0.20 | 1.1% | 2025-09-05 12:00 | 2025-09-05 12:00 | 0.00 |
| Virginia Cavaliers vs William & Mary Tribe | 2025-09-13 | $5.12 | $5.00 | $0.12 | 2.4% | 2025-09-08 18:00 | 2025-09-08 18:00 | 0.00 |
| Rutgers Scarlet Knights vs Norfolk State Spartan | 2025-09-13 | $15.95 | $16.00 | $0.05 | 0.3% | 2025-09-10 18:00 | 2025-09-10 00:00 | 18.00 |
| Tulane Green Wave vs Duke Blue Devil | 2025-09-13 | $13.05 | $13.00 | $0.05 | 0.4% | 2025-08-29 06:00 | 2025-08-29 06:00 | 0.00 |
| Miami Hurricanes vs South Florida Bull | 2025-09-13 | $9.02 | $9.00 | $0.02 | 0.2% | 2025-08-27 00:00 | 2025-08-27 14:44 | 14.75 |
| Syracuse Orange vs Colgate Raider | 2025-09-12 | $2.01 | $2.00 | $0.01 | 0.5% | 2025-08-31 12:00 | 2025-08-31 12:00 | 0.00 |
| Wake Forest Demon Deacons vs North Carolina State Wolfpack | 2025-09-11 | $19.00 | $19.00 | $0.00 | 0.0% | 2025-09-01 18:00 | 2025-09-01 12:00 | 6.00 |
| Delaware Blue Hens vs UConn Huskie | 2025-09-13 | $15.00 | $15.00 | $0.00 | 0.0% | 2025-08-31 06:00 | 2025-08-31 06:00 | 0.00 |
| Southern Miss Golden Eagles vs Appalachian State Mountaineer | 2025-09-13 | $6.00 | $6.00 | $0.00 | 0.0% | 2025-09-06 12:00 | 2025-09-06 12:00 | 0.00 |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).
- Timing: 15% of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.
- Near-zero importance this week (may be unrelated): conferenceGame