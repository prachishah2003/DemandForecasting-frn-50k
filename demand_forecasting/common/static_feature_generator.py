# demand_forecasting/common/static_feature_generator.py
# =============================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


# ---------------------------------------------------------------
# Helper: ensure item_id
# ---------------------------------------------------------------
def ensure_item_id(df: pd.DataFrame) -> pd.DataFrame:
    if "item_id" not in df.columns:
        df = df.copy()
        df["item_id"] = (
            df.get("city_id", "").astype(str)
            + "_"
            + df.get("store_id", "").astype(str)
            + "_"
            + df.get("product_id", "").astype(str)
        )
    return df


# ---------------------------------------------------------------
# Compute demand statistics per item
# ---------------------------------------------------------------
def compute_series_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_item_id(df)
    stats = (
        df.groupby("item_id")["demand"]
        .agg(
            avg_demand="mean",
            std_demand="std",
            min_demand="min",
            max_demand="max",
            count="count",
        )
        .fillna(0)
    )
    stats["cv"] = stats["std_demand"] / (stats["avg_demand"] + 1e-8)

    zero_frac = (
        df.assign(is_zero=df["demand"] <= 0)
        .groupby("item_id")["is_zero"]
        .mean()
        .rename("zero_fraction")
    )
    stats = stats.join(zero_frac, how="left").fillna(0)
    return stats


# ---------------------------------------------------------------
# Price tier classification
# ---------------------------------------------------------------
def classify_price_tier(df: pd.DataFrame) -> pd.Series:
    df = ensure_item_id(df)
    if "price" not in df.columns:
        return pd.Series("unknown", index=df["item_id"].unique(), name="price_tier")

    avg_price = df.groupby("item_id")["price"].mean()

    q1, q2 = avg_price.quantile([0.33, 0.66])

    def bucket(v):
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    return avg_price.apply(bucket).rename("price_tier")


# ---------------------------------------------------------------
# Volatility classification
# ---------------------------------------------------------------
def classify_volatility(stats: pd.DataFrame) -> pd.Series:
    cv = stats["cv"].replace([np.inf, -np.inf], 0).fillna(0)
    q1, q2 = cv.quantile([0.33, 0.66])

    def bucket(v):
        if v <= q1:
            return "low_vol"
        if v <= q2:
            return "mid_vol"
        return "high_vol"

    return cv.apply(bucket).rename("volatility_class")


# ---------------------------------------------------------------
# Intermittency flag
# ---------------------------------------------------------------
def detect_intermittent(stats: pd.DataFrame) -> pd.Series:
    return (stats["zero_fraction"] > 0.4).astype(int).rename("intermittent_flag")


# ---------------------------------------------------------------
# Product metadata static features (category, brand, etc.)
# ---------------------------------------------------------------
def derive_product_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_item_id(df)
    cols = [c for c in ["category", "subcategory", "brand", "perishability"] if c in df.columns]

    if not cols:
        return pd.DataFrame({"item_id": df["item_id"].unique()})

    meta = df.groupby("item_id")[cols].first().reset_index()
    return meta


# ---------------------------------------------------------------
# KMeans-based clustering
# ---------------------------------------------------------------
def cluster_time_series(stats: pd.DataFrame, n_clusters: int = 10) -> pd.Series:
    X = stats[["avg_demand", "cv", "zero_fraction"]].replace([np.inf, -np.inf], 0).fillna(0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return pd.Series(labels, index=stats.index, name="cluster_id")


# ---------------------------------------------------------------
# Main static feature generator
# ---------------------------------------------------------------
def generate_static_features(df: pd.DataFrame, n_clusters: int = 12) -> pd.DataFrame:
    df = ensure_item_id(df.copy())

    stats = compute_series_statistics(df)
    price_tier = classify_price_tier(df)
    vol_class = classify_volatility(stats)
    intermittent = detect_intermittent(stats)
    clusters = cluster_time_series(stats, n_clusters=n_clusters)
    product_meta = derive_product_static_features(df)

    static_df = (
        stats.join(price_tier, how="left")
        .join(vol_class, how="left")
        .join(intermittent, how="left")
        .join(clusters, how="left")
        .reset_index()
    )

    static_df = static_df.merge(product_meta, on="item_id", how="left")

    return static_df


# ---------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------
def save_static_features(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_static_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
