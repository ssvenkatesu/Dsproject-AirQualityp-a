"""
Streamlit App — Urban Air Quality Analysis & Prediction
Developed by Siva Sri Venkatesu S S
"""

# --- Path Fix so "src" works even when running from /app ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports ---
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# --- Local Modules ---
from src import data_processing as dp
from src import eda as eda
from src import model as model_mod

# --- Streamlit Config ---
st.set_page_config(page_title="Urban Air Quality — DSProject", layout="wide")
st.title("🌆 Urban Air Quality — Analysis & Prediction")

# --- Folder Setup ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Sidebar Navigation ---
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploratory Analysis", "Train Model", "Predict"],
    captions=["View datasets", "Generate plots", "Train ML model", "Predict air quality"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Siva Sri Venkatesu S S 👨‍💻")

# --- Load All Datasets ---
@st.cache_data(show_spinner=False)
def load_datasets():
    return dp.load_all_datasets(str(DATA_DIR))

dfs = load_datasets()
ds_names = [k for k, v in dfs.items() if isinstance(v, pd.DataFrame)]

# --- Auto-detect target ---
inferred = dp.infer_target_column(dfs)
default_target_text = f"{inferred[1]} (from {inferred[0]})" if inferred else "None detected"
st.sidebar.write("🎯 Detected Target:", default_target_text)

# -------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------------
if menu == "Overview":
    st.header("📘 Available Datasets")
    for name, df in dfs.items():
        st.subheader(f"📄 {name}")
        if isinstance(df, pd.DataFrame):
            st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            st.dataframe(df.head(5))
        else:
            st.warning("⚠️ Failed to load dataset.")
    st.info("Use **Exploratory Analysis** for plots or **Train Model** to build a predictor.")

# -------------------------------------------------------------------
# PAGE 2: EDA
# -------------------------------------------------------------------
elif menu == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis (EDA)")

    ds_choice = st.selectbox("Choose dataset", ["-- select --"] + ds_names)
    if ds_choice and ds_choice != "-- select --":
        df = dfs[ds_choice].copy()
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        # Optional date parsing
        if st.button("🧭 Parse Dates & Add Features"):
            df = dp.parse_dates(df)
            st.success("Datetime features added successfully!")
            st.dataframe(df.head(5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns for plotting.")
        else:
            pollutant = st.selectbox("Select numeric column to analyze", numeric_cols)
            city_cols = [c for c in df.columns if 'city' in c.lower() or 'name' in c.lower()]
            city_col = st.selectbox("Optional city column", [None] + city_cols)
            city_val = None
            if city_col:
                cities = df[city_col].dropna().unique().tolist()
                if cities:
                    city_val = st.selectbox("Select a city", [None] + cities)
            resample = st.selectbox("Resample Frequency", [None, "D", "W", "M"])

            ts_path = str(OUTPUTS_DIR / f"ts_{ds_choice}_{pollutant}.png")
            hist_path = str(OUTPUTS_DIR / f"hist_{ds_choice}_{pollutant}.png")
            corr_path = str(OUTPUTS_DIR / f"corr_{ds_choice}.png")

            with st.spinner("Generating plots..."):
                dp.parse_dates(df)
                p1 = eda.plot_time_series(df, pollutant, ts_path, city_col=city_col, city=city_val, resample=resample)
                p2 = eda.plot_histogram(df, pollutant, hist_path)
                p3 = eda.plot_correlation_heatmap(df, corr_path)

            st.subheader("📊 Generated Plots")
            if p1: st.image(p1, caption="Time Series Plot")
            if p2: st.image(p2, caption="Histogram")
            if p3: st.image(p3, caption="Correlation Heatmap")

            if city_col:
                p4 = eda.plot_pollutant_by_city(df, pollutant, city_col, OUTPUTS_DIR / f"city_avg_{pollutant}.png")
                if p4: st.image(p4, caption="Top Cities by Mean Pollutant")

# -------------------------------------------------------------------
# PAGE 3: TRAIN MODEL
# -------------------------------------------------------------------
elif menu == "Train Model":
    st.header("⚙️ Train Machine Learning Model")

    ds_choice = st.selectbox("Choose dataset for training", ["-- select --"] + ds_names)
    if ds_choice and ds_choice != "-- select --":
        df = dfs[ds_choice].copy()
        df = dp.parse_dates(df)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for model training.")
        else:
            suggested = dp.infer_target_column({ds_choice: df})
            default_target = suggested[1] if suggested else numeric_cols[0]
            target = st.selectbox("🎯 Target Column", numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
            default_feats = [c for c in numeric_cols if c != target]
            features = st.multiselect("🧩 Feature Columns", numeric_cols, default=default_feats[:8])

            if st.button("🚀 Train Model"):
                with st.spinner("Training model..."):
                    try:
                        metrics = model_mod.train_and_save(df, target, str(MODELS_DIR / "aq_model.pkl"), features=features)
                        st.success(f"✅ Model trained successfully!")
                        st.write(f"**MSE:** {metrics['mse']:.3f}")
                        st.write(f"**R²:** {metrics['r2']:.3f}")
                        st.write(f"**Train rows:** {metrics['n_train']} | **Test rows:** {metrics['n_test']}")

                        # Load model for importance visualization
                        with open(MODELS_DIR / "aq_model.pkl", "rb") as f:
                            model_data = pickle.load(f)
                        model = model_data["model"]
                        feats = model_data["features"]

                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                            fig, ax = plt.subplots(figsize=(8, 4))
                            imp_df = pd.DataFrame({"Feature": feats, "Importance": importances}).sort_values(by="Importance", ascending=False)
                            ax.barh(imp_df["Feature"], imp_df["Importance"])
                            ax.invert_yaxis()
                            ax.set_xlabel("Importance")
                            ax.set_title("Feature Importance (RandomForest)")
                            st.pyplot(fig)

                            # Save files
                            imp_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
                            with open(MODELS_DIR / "model_info.json", "w") as f:
                                json.dump({
                                    "model_name": "RandomForestRegressor",
                                    "created_on": datetime.now().isoformat(),
                                    "target_column": target,
                                    "features": feats,
                                    "metrics": metrics
                                }, f, indent=4)
                            st.info("📁 Metadata & feature importance saved in /models folder.")
                        else:
                            st.info("Model type has no feature importance attribute.")

                        # Download button
                        with open(MODELS_DIR / "aq_model.pkl", "rb") as f:
                            st.download_button(
                                label="⬇️ Download Trained Model",
                                data=f,
                                file_name="aq_model.pkl",
                                mime="application/octet-stream"
                            )

                    except Exception as e:
                        st.error("❌ Training failed: " + str(e))
                        st.code(traceback.format_exc())

# -------------------------------------------------------------------
# PAGE 4: PREDICT (Enhanced with sliders + category)
# -------------------------------------------------------------------
elif menu == "Predict":
    st.header("🔮 Predict Air Quality")

    model_path = MODELS_DIR / "aq_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Model' tab.")
    else:
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            features = model_data["features"]

            st.write(f"✅ Model loaded successfully with **{len(features)}** features.")
            st.subheader("🎚️ Adjust feature values for prediction:")

            user_inputs = {}
            cols = st.columns(2)  # display sliders in two columns
            for i, feat in enumerate(features):
                with cols[i % 2]:
                    # create a slider with dynamic range
                    user_inputs[feat] = st.slider(
                        f"{feat}",
                        min_value=0.0,
                        max_value=500.0,
                        value=50.0,
                        step=1.0
                    )

            if st.button("🔍 Predict"):
                pred_value = model.predict([[user_inputs[f] for f in features]])[0]

                # --- Categorize the air quality ---
                def categorize_air_quality(value):
                    if value <= 50:
                        return "🟢 Good"
                    elif 50 < value <= 100:
                        return "🟡 Moderate"
                    elif 100 < value <= 200:
                        return "🟠 Poor"
                    elif 200 < value <= 300:
                        return "🔴 Very Poor"
                    else:
                        return "⚫ Hazardous"

                category = categorize_air_quality(pred_value)

                st.success(f"🌫️ Predicted Air Quality Value: **{pred_value:.2f}**")
                st.markdown(f"### 💡 Air Quality Category: {category}")

                st.info("""
                **Air Quality Index (AQI) Reference:**
                - 🟢 0–50 → Good  
                - 🟡 51–100 → Moderate  
                - 🟠 101–200 → Poor  
                - 🔴 201–300 → Very Poor  
                - ⚫ 301+ → Hazardous
                """)

        except Exception as e:
            st.error("Failed to load model: " + str(e))
            st.code(traceback.format_exc())

# -------------------------------------------------------------------
# END
# -------------------------------------------------------------------

else:
    st.info("Use the sidebar to navigate between sections.")
