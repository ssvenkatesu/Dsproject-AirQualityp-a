"""
evaluate_model.py
Load saved model and evaluate it on dataset.
"""

import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = "aq_model.pkl"
DATA_PATH = "../data/city_day.csv"
TARGET_COLUMN = "PM2.5"

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# drop rows with missing target
df = df.dropna(subset=[TARGET_COLUMN])
df = df.fillna(df.mean(numeric_only=True))

X = df[features].values
y = df[TARGET_COLUMN].values

print("Evaluating model...")
preds = model.predict(X)
mse = mean_squared_error(y, preds)
r2 = r2_score(y, preds)
print(f"✅ Evaluation complete: MSE={mse:.4f}, R2={r2:.4f}")
