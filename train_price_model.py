import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

SNAPSHOT_PATH = "data/price_snapshots.csv"
MODEL_PATH = "models/ticket_price_model.pkl"

def train_model():
    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError("Snapshot data not found. Run the price logger first.")

    df = pd.read_csv(SNAPSHOT_PATH)

    # Drop rows with missing target
    df = df.dropna(subset=["lowest_price"])

    # Select features and target
    features = [
        "days_until_game", "capacity", "neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup",
        "homeTeamRank", "awayTeamRank"
    ]
    df = df[features + ["lowest_price"]].copy()

    # Fill missing values and encode
    df = df.fillna(-1)

    X = df.drop(columns=["lowest_price"])
    y = df["lowest_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. MSE on test set: {mse:.2f}")

    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
