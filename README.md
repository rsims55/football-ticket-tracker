# 🏈 College Football Ticket Price Forecasting

This project forecasts college football ticket prices and identifies the **optimal time and date to purchase tickets** for any given game. It combines scraped game data, rankings, stadium info, rivalries, and mock pricing snapshots, and will evolve into a fully supervised machine learning pipeline.

---

## 🔧 Features

- ✅ **Enriched dataset builder**: pulls and merges game schedule, team rankings, venue size, rivalries, and pricing data
- ⏱ **Scheduled price snapshots**: collects pricing data 4× daily
- 📊 **Ranking fetcher**: fetches AP/Playoff rankings (with fallback to Wikipedia)
- 🧠 **ML model**: predicts lowest ticket price using RandomForest
- 🔍 **Optimal purchase time estimator**: simulates price predictions at all (date, time) combinations pre-game
- 📈 **Feature importance visualization**
- 🔁 **Postseason retraining plan**: build supervised models on true pricing outcomes

---

## 📁 Project Structure

```
cfb-ticket-model/
├── data/                        # Enriched data + snapshots
├── models/                      # Trained model + plots
├── build_dataset.py            # Build enriched schedule
├── price_logger.py             # Collect ticket prices 4× daily
├── train_price_model.py        # Train ML model
├── optimal_purchase_finder.py  # Predict optimal day/time to buy
├── feature_importance_plot.py  # Visualize feature importance
├── rankings_fetcher.py         # Ranking puller (CFD + Wikipedia)
├── schedule_fetcher.py         # Game schedule puller
├── stadium_scraper.py          # Venue capacity info
├── rivalry_scraper.py          # Known rivalries
├── ticket_pricer.py            # Live/mock pricing logic
├── .env                        # CFD API key and environment settings
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

## ⚙️ Setup

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/cfb-ticket-model.git
cd cfb-ticket-model
```

2. Create `.env` file:

```
CFD_API_KEY=your_key_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the full pipeline:

```bash
python build_dataset.py
python price_logger.py
python train_price_model.py
python optimal_purchase_finder.py
```

---

## 📆 Cron Schedule (Recommended)

```cron
# Rankings - Mon/Wed at 7am
0 7 * * 1,3 /usr/bin/python3 /path/to/rankings_fetcher.py

# Prices - 4× daily
0 6,12,18,0 * * * /usr/bin/python3 /path/to/price_logger.py
```

---

## 🧠 Future Plans

- Use actual pricing outcomes after each game to retrain the model
- Expand predictions to group pricing and listing counts
- Build a dashboard to visualize price forecasts and buy recommendations

---

## 🤝 Contributing

If you're passionate about sports analytics or modeling dynamic prices, feel free to submit PRs or ideas!

---

## 📄 License

@RandiSims2025
