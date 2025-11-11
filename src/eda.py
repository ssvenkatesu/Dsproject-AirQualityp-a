"""
eda.py
Exploratory Data Analysis helpers that produce matplotlib plots saved to outputs/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # seaborn used only for nicer heatmap layout; optional

# NOTE: If seaborn isn't desired remove the import and adjust heatmap code accordingly.


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_time_series(df: pd.DataFrame, col: str, out_path: str, city_col: str = None, city: str = None, resample: str = None):
    """
    Save a time series line plot for 'col'. If 'date_parsed' exists uses it as x-axis.
    Optional filter by city_col == city.
    resample can be 'D','W','M' etc (pandas offset alias).
    Returns path as string or None on failure.
    """
    try:
        df = df.copy()
        if city_col and city:
            if city_col in df.columns:
                df = df[df[city_col] == city]
        if 'date_parsed' in df.columns:
            df = df.sort_values('date_parsed')
            ts = df.set_index('date_parsed')
            if resample:
                ts = ts.resample(resample).mean()
            plt.figure(figsize=(10, 4))
            plt.plot(ts.index, ts[col], marker=None, linewidth=1)
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.title(f"{col} over time" + (f" — {city}" if city else ""))
            plt.tight_layout()
        else:
            plt.figure(figsize=(8, 3))
            plt.plot(df.index, df[col].values, marker='o', linestyle='-', markersize=2)
            plt.title(f"{col} over rows")
            plt.tight_layout()
        out = Path(out_path)
        ensure_parent(out)
        plt.savefig(out)
        plt.close()
        return str(out)
    except Exception as e:
        print("[plot_time_series] error:", e)
        try:
            plt.close()
        finally:
            return None


def plot_histogram(df: pd.DataFrame, col: str, out_path: str, bins: int = 40):
    try:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        out = Path(out_path)
        ensure_parent(out)
        plt.savefig(out)
        plt.close()
        return str(out)
    except Exception as e:
        print("[plot_histogram] error:", e)
        try:
            plt.close()
        finally:
            return None


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str, numeric_only: bool = True):
    try:
        if numeric_only:
            df = df.select_dtypes(include=[np.number])
        corr = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.3)
        plt.title("Correlation matrix")
        plt.tight_layout()
        out = Path(out_path)
        ensure_parent(out)
        plt.savefig(out)
        plt.close()
        return str(out)
    except Exception as e:
        print("[plot_correlation_heatmap] error:", e)
        try:
            plt.close()
        finally:
            return None


def plot_pollutant_by_city(df: pd.DataFrame, pollutant_col: str, city_col: str, out_path: str, top_n: int = 10):
    """
    Bar plot of average pollutant per city, top_n cities shown.
    """
    try:
        grp = df.groupby(city_col)[pollutant_col].mean().dropna().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(10, 5))
        grp.plot(kind='bar')
        plt.ylabel(f"Mean {pollutant_col}")
        plt.title(f"Top {top_n} cities by mean {pollutant_col}")
        plt.tight_layout()
        out = Path(out_path)
        ensure_parent(out)
        plt.savefig(out)
        plt.close()
        return str(out)
    except Exception as e:
        print("[plot_pollutant_by_city] error:", e)
        try:
            plt.close()
        finally:
            return None
