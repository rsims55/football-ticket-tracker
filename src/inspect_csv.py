import pandas as pd
df = pd.read_csv("data/predicted_prices_optimal.csv")
print(df.columns)
print(df[["homeTeam", "awayTeam", "startDateEastern"]].head())
