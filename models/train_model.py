"""
train_model.py
Train a RandomForestRegressor on air quality data and save the trained model.
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIG ===
DATA_PATH = "../data/city_day.csv"   # Change if using different dataset
MODEL_PATH = "aq_model.pkl"
INFO_PATH = "model_info.json"
IMPORTANCE_PATH = "feature_importance.csv"

TARGET_COLUMN = "PM2.5"  # Change this to your target (check column names!)

# === 1. LOAD DATA ===
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# === 2. CLEANING ===
print("Cleaning data...")
df = df.dropna(subset=[TARGET_COLUMN])
df = df.fillna(df.mean(numeric_only=True))  # replace missing numeric with mean

# === 3. FEATURE SELECTION ===
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric if c != TARGET_COLUMN]
print(f"Selected {len(features)} features for training.")

X = df[features].values
y = df[TARGET_COLUMN].values

# === 4. SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. TRAIN MODEL ===
print("Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# === 6. EVALUATE ===
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"✅ Model trained! MSE={mse:.4f}, R2={r2:.4f}")

# === 7. SAVE MODEL ===
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "features": features}, f)
print(f"Model saved to {MODEL_PATH}")

# === 8. SAVE METADATA ===
info = {
    "model_name": "RandomForestRegressor",
    "dataset": os.path.basename(DATA_PATH),
    "target_column": TARGET_COLUMN,
    "features": features,
    "n_estimators": 200,
    "metrics": {"mse": mse, "r2": r2},
    "created_on": datetime.now().isoformat()
}
with open(INFO_PATH, "w") as f:
    json.dump(info, f, indent=4)
print(f"Model info saved to {INFO_PATH}")

# === 9. FEATURE IMPORTANCE ===
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)
importance.to_csv(IMPORTANCE_PATH, index=False)
print(f"Feature importances saved to {IMPORTANCE_PATH}")

print("\n🎉 Training complete! You can now use this model for prediction.")
