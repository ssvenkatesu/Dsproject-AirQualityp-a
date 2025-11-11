"""
model.py
Training, evaluation and persistence utilities for pollutant prediction.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional


def train_and_save(df: pd.DataFrame,
                   target_col: str,
                   model_out_path: str,
                   features: Optional[List[str]] = None,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   n_estimators: int = 200) -> Dict:
    """
    Train a RandomForestRegressor on df to predict target_col.
    Saves model + feature list to model_out_path (pickle).
    Returns metrics dict.
    """
    if df is None:
        raise ValueError("No DataFrame provided")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in df columns")

    # select numeric features if none provided
    if features is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric if c != target_col]

    if len(features) == 0:
        raise ValueError("No features available to train on.")

    df2 = df[features + [target_col]].dropna()
    if df2.shape[0] < 30:
        raise ValueError(f"Not enough rows to train a reliable model (after dropna got {df2.shape[0]} rows)")

    X = df2[features].values
    y = df2[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    out = {
        "model": model,
        "features": features,
        "metrics": {"mse": mse, "r2": r2, "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])}
    }

    # persist
    out_path = Path(model_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    return out["metrics"]


def load_model(model_path: str) -> Dict:
    """
    Load the pickled model artifact (dict with keys 'model','features').
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(model_path)
    with open(p, "rb") as f:
        return pickle.load(f)


def predict_from_model(loaded: Dict, feature_values: Dict[str, float]) -> float:
    """
    Given loaded model dict and feature_values mapping feature->value, return prediction.
    """
    model = loaded['model']
    features = loaded['features']
    X = np.array([[float(feature_values.get(f, 0.0)) for f in features]])
    return float(model.predict(X)[0])


def evaluate_model_on_df(loaded: Dict, df: pd.DataFrame, target_col: str) -> Dict:
    """
    Evaluate a loaded model on a DataFrame (requires df to have required features + target).
    Returns mse and r2.
    """
    features = loaded['features']
    model = loaded['model']
    df2 = df[features + [target_col]].dropna()
    if df2.shape[0] == 0:
        raise ValueError("No rows to evaluate after dropna")
    X = df2[features].values
    y = df2[target_col].values
    preds = model.predict(X)
    mse = float(mean_squared_error(y, preds))
    r2 = float(r2_score(y, preds))
    return {"mse": mse, "r2": r2, "n": int(df2.shape[0])}
