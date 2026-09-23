import numpy as np
import pandas as pd


def create_release_features(df):
    # quarter/decade from the month/year already extracted in SQL
    df = df.copy()
    df["release_quarter"] = np.ceil(df["release_month"] / 3).astype(int)
    df["release_decade"] = (df["release_year"] // 10 * 10).astype(int)
    return df


def create_log_features(df):
    # log1p transform of the heavily skewed financial columns
    df = df.copy()
    df["log_budget"] = np.log1p(df["budget"])
    df["log_runtime"] = np.log1p(df["runtime"])
    df["log_revenue"] = np.log1p(df["revenue"])
    return df


def create_frequency_map(series):
    # map each category to its share of rows (computed on train only)
    counts = series.value_counts()
    return (counts / len(series)).to_dict()


def apply_frequency_map(series, freq_map):
    # unseen categories fall back to 0.0
    return series.map(freq_map).fillna(0.0).astype(float)


def log1p_safe(value):
    # helper for back-transforming historical revenue/budget features
    return np.log1p(value)


def create_historical_log_features(df):
    # log1p versions of the raw historical revenue/budget columns (NaN-safe)
    df = df.copy()
    for col in [
        "director_previous_avg_revenue",
        "director_previous_avg_budget",
        "director_previous_max_revenue",
        "production_company_previous_avg_revenue",
        "cast_previous_avg_revenue",
        "lead_actor_previous_avg_revenue",
        "franchise_previous_avg_revenue",
    ]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    return df