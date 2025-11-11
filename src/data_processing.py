"""
data_processing.py
Helpers to load, clean and prepare air quality datasets for analysis & modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_all_datasets(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSVs found in data_dir into a dict keyed by stem (filename without ext).
    """
    data_dir = Path(data_dir)
    dfs = {}
    for p in data_dir.glob("*.csv"):
        try:
            dfs[p.stem] = pd.read_csv(p)
        except Exception as e:
            print(f"[load_all_datasets] failed to read {p}: {e}")
            dfs[p.stem] = None
    return dfs


def guess_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic to pick a datetime column name if present.
    """
    if df is None:
        return None
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("datetime","date","time","timestamp"))]
    # prefer exact names
    for pref in ("datetime","date","time","timestamp"):
        if pref in [c.lower() for c in df.columns]:
            return next(c for c in df.columns if c.lower() == pref)
    return candidates[0] if candidates else None


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified 'date_parsed' column if any date-like column exists, and add time features.
    """
    df = df.copy()
    dt_col = guess_datetime_column(df)
    if dt_col:
        try:
            df['date_parsed'] = pd.to_datetime(df[dt_col], errors='coerce')
        except Exception:
            # try more permissive
            df['date_parsed'] = pd.to_datetime(df[dt_col].astype(str), errors='coerce', utc=False)
    else:
        # try common column combos
        if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df[['year','month','day']])
            except Exception:
                pass

    # add derived features if date_parsed exists
    if 'date_parsed' in df.columns:
        df['year'] = df['date_parsed'].dt.year
        df['month'] = df['date_parsed'].dt.month
        df['day'] = df['date_parsed'].dt.day
        df['weekday'] = df['date_parsed'].dt.weekday
        # if time-of-day exists create hour
        try:
            df['hour'] = df['date_parsed'].dt.hour
        except Exception:
            df['hour'] = np.nan
    return df


def infer_target_column(dfs: Dict[str, pd.DataFrame], prefer: Tuple[str, ...] = ('city_hour','city_day','station_hour','station_day')) -> Optional[Tuple[str, str]]:
    """
    Return (dataset_name, target_col) inferred from available datasets.
    Priority:
      - explicit PM2.5-like columns (pm25, pm_2_5, PM2.5)
      - preference by dataset ordering
      - otherwise first numeric column
    """
    pm_candidates = {"pm25","pm_2_5","pm2_5","pm2.5","pm25_0","pm_2.5","PM2.5","PM25"}
    for ds in prefer:
        df = dfs.get(ds)
        if isinstance(df, pd.DataFrame):
            cols = set(df.columns)
            for c in cols:
                if c in pm_candidates:
                    return ds, c
    # fallback: search all datasets for PM variants
    for name, df in dfs.items():
        if isinstance(df, pd.DataFrame):
            cols = set(df.columns)
            inter = cols & pm_candidates
            if inter:
                return name, sorted(list(inter))[0]
    # final fallback: first numeric column in preferred dataset order
    for ds in prefer:
        df = dfs.get(ds)
        if isinstance(df, pd.DataFrame):
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric:
                return ds, numeric[0]
    # any dataset
    for name, df in dfs.items():
        if isinstance(df, pd.DataFrame):
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric:
                return name, numeric[0]
    return None


def basic_clean(df: pd.DataFrame, drop_threshold: float = 0.7) -> pd.DataFrame:
    """
    Basic cleaning steps:
    - drop cols with > drop_threshold fraction missing
    - drop rows that are all NaN or have no numeric data
    - strip whitespace from string column names
    """
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    # drop columns with high missing
    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > drop_threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # drop rows with no numeric values
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        df = df.dropna(axis=0, how='all', subset=df.select_dtypes(include=[np.number]).columns.tolist())
    else:
        df = df.dropna(how='all')
    return df


def prepare_model_df(df: pd.DataFrame, target_col: str, max_features: Optional[int] = 20) -> pd.DataFrame:
    """
    Prepare a modeling DataFrame:
    - parse dates (if present)
    - run basic_clean
    - select numeric columns, remove the target from features
    - (optional) limit to max_features numeric features
    """
    if df is None:
        raise ValueError("Input df is None")
    df = df.copy()
    df = parse_dates(df)
    df = basic_clean(df)
    # select numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric:
        # if target is not numeric try converting
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception:
            pass
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in df")
    if target_col in numeric:
        numeric.remove(target_col)
    features = numeric[:max_features] if max_features else numeric
    result = df[features + [target_col]].copy()
    return result


def add_lag_features(df: pd.DataFrame, col: str, lags: Tuple[int, ...] = (1,24,168)) -> pd.DataFrame:
    """
    Add lag features for a column assuming df has a datetime index or 'date_parsed'.
    lags in units of rows (for hourly data typical lags are 1,24,168).
    Returns new DataFrame with added columns: <col>_lag{lag}
    """
    df = df.copy()
    if 'date_parsed' in df.columns:
        df = df.sort_values('date_parsed')
        df = df.set_index('date_parsed')
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.reset_index(drop=False) if 'date_parsed' in df.columns else df
    return df
