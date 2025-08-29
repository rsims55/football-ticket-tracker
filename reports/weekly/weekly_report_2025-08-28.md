# 📈 Weekly Ticket Price Model Report
**Date:** 2025-08-28

## 🔍 Best Predictors of Ticket Price

### Top Transformed Features (expanded)
- capacity was important, contributing 31.1% to predictions.
- Teams from the SEC awayconference mattered, contributing 12.0%.
- awayTeamRank was important, contributing 11.2% to predictions.
- days until game was important, contributing 10.9% to predictions.
- week was important, contributing 9.0% to predictions.
- homeTeamRank was important, contributing 6.3% to predictions.
- isRivalry was important, contributing 4.7% to predictions.
- Teams from the Big Ten awayconference mattered, contributing 4.2%.
- neutralSite was important, contributing 1.7% to predictions.
- Teams from the Big 12 awayconference mattered, contributing 1.6%.
- conferenceGame was important, contributing 1.1% to predictions.
- Teams from the SEC homeconference mattered, contributing 0.9%.
- Teams from the ACC awayconference mattered, contributing 0.6%.
- Teams from the Big 12 homeconference mattered, contributing 0.6%.
- Teams from the ACC homeconference mattered, contributing 0.5%.
- isRankedMatchup was important, contributing 0.5% to predictions.
- Teams from the FBS Independents awayconference mattered, contributing 0.4%.
- Teams from the American Athletic awayconference mattered, contributing 0.2%.
- Teams from the Sun Belt homeconference mattered, contributing 0.2%.
- Teams from the Big Ten homeconference mattered, contributing 0.2%.

### Aggregated by Original Column
- capacity: 0.3112
- awayConference: 0.1996
- awayTeamRank: 0.1118
- days_until: 0.1090
- week: 0.0904
- homeTeamRank: 0.0627
- isRivalry: 0.0474
- homeConference: 0.0318
- neutralSite: 0.0170
- conferenceGame: 0.0107
- isRankedMatchup: 0.0049
- collectionSlot: 0.0036

**Possibly unrelated (near-zero importance):** isRankedMatchup, collectionSlot

## 📊 Model Accuracy (Past 7 Days)

- Games evaluated: **18**
- MAE: **$10.43**
- RMSE: **$26.69**
- Games > 5% error: **12 / 18**

## 🎯 Predicted vs Actual Prices

| Game | Date (ET) | Predicted | Actual | Abs Error | % Error |
|------|-----------|-----------|--------|-----------|---------|
|  vs  | 2025-08-28 | $20.57 | $129.00 | $108.43 | 84.1% |
|  vs  | 2025-08-28 | $79.26 | $106.00 | $26.74 | 25.2% |
|  vs  | 2025-08-28 | $25.11 | $13.00 | $12.11 | 93.2% |
|  vs  | 2025-08-28 | $23.58 | $15.00 | $8.58 | 57.2% |
|  vs  | 2025-08-28 | $14.21 | $20.00 | $5.79 | 28.9% |
|  vs  | 2025-08-28 | $17.44 | $12.00 | $5.44 | 45.3% |
|  vs  | 2025-08-28 | $21.31 | $17.00 | $4.31 | 25.4% |
|  vs  | 2025-08-28 | $28.77 | $25.00 | $3.77 | 15.1% |
|  vs  | 2025-08-28 | $21.31 | $25.00 | $3.69 | 14.8% |
|  vs  | 2025-08-28 | $28.77 | $26.00 | $2.77 | 10.7% |
|  vs  | 2025-08-28 | $56.91 | $55.00 | $1.91 | 3.5% |
|  vs  | 2025-08-28 | $5.64 | $4.00 | $1.64 | 41.0% |
|  vs  | 2025-08-28 | $21.38 | $20.00 | $1.38 | 6.9% |
|  vs  | 2025-08-28 | $16.21 | $17.00 | $0.79 | 4.6% |
|  vs  | 2025-08-28 | $6.24 | $6.00 | $0.24 | 4.0% |
|  vs  | 2025-08-28 | $13.16 | $13.00 | $0.16 | 1.2% |
|  vs  | 2025-08-28 | $10.96 | $11.00 | $0.04 | 0.4% |
|  vs  | 2025-08-28 | $13.02 | $13.00 | $0.02 | 0.2% |

## 💡 Suggestions
- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.
- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).
- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.
- Check stadium capacity normalization (capacity vs. sold % if/when available).

- Near-zero importance this week (may be unrelated): collectionSlot, isRankedMatchup