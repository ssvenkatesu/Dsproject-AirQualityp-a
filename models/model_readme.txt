# Air Quality Prediction Model

This folder contains trained models and metadata files.

Main file:
- aq_model.pkl — Trained RandomForestRegressor
  (includes 'model' object and 'features' list)

Usage example:
--------------------
import pickle
with open("models/aq_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]

# Make a prediction
sample_input = [[55.0, 20.0, 8.0, 1.5, 35.0, 120.0]]
prediction = model.predict(sample_input)
print("Predicted PM2.5:", prediction[0])

--------------------
To retrain:
Use the Streamlit app → 'Train Model (quick)'
