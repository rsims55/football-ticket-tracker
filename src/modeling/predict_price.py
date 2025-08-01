import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

MODEL_PATH = "models/ticket_price_model.pkl"
SCHEDULE_PATH = "data/enriched_schedule_2025.csv"
OUTPUT_PATH = "data/predicted_prices_optimal.csv"
COLLECTION_TIMES = ["06:00", "12:00", "18:00", "00:00"]
MAX_DAYS_OUT = 30

FEATURES = [
    "days_until_game", "capacity", "neutralSite", "conferenceGame",
    "isRivalry", "isRankedMatchup", "homeTeamRank", "awayTeamRank"
]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_price_model.py first.")
    return joblib.load(MODEL_PATH)

def simulate_predictions(df, model):
    results = []
    for _, row in df.iterrows():
        best_price = float('inf')
        best_date = None
        best_time = None

        game_date = pd.to_datetime(row["startDateEastern"]).date()

        for delta in range(1, MAX_DAYS_OUT + 1):
            sim_date = game_date - timedelta(days=delta)
            days_until_game = delta

            for time_str in COLLECTION_TIMES:
                sim_time = datetime.strptime(time_str, "%H:%M").time()

                input_row = pd.DataFrame([{f: row.get(f, -1) for f in FEATURES}])
                input_row["days_until_game"] = days_until_game
                input_row = input_row.fillna(-1)

                price_pred = model.predict(input_row)[0]

                if price_pred < best_price:
                    best_price = price_pred
                    best_date = sim_date
                    best_time = sim_time

        results.append({
            "homeTeam": row.get("homeTeam"),
            "awayTeam": row.get("awayTeam"),
            "startDateEastern": row.get("startDateEastern"),
            "week": row.get("week"),
            "predicted_lowest_price": round(best_price, 2),
            "optimal_purchase_date": best_date,
            "optimal_purchase_time": best_time.strftime("%H:%M") if best_time else None
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    if not os.path.exists(SCHEDULE_PATH):
        raise FileNotFoundError("Enriched schedule not found. Run build_dataset.py first.")

    schedule_df = pd.read_csv(SCHEDULE_PATH)
    model = load_model()
    result_df = simulate_predictions(schedule_df, model)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Optimal purchase predictions saved to {OUTPUT_PATH}")
