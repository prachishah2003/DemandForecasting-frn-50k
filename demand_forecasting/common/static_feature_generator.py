# demand_forecasting/common/static_feature_generator.py
# =============================================================
"""
Static Feature Generator (Patched for FreshRetailNet-50K)

FRN schema uses:
    product_id
    dt
    sale_amount
    city_id
    store_id
(and may include: discount, holiday_flag, activity_flag, etc.)

This module:
- Builds composite item_id = city_store_product
- Uses sale_amount instead of demand
- Computes rich per-item statistics
- Aggregates sequence columns (hours_sale, hours_stock_status) into SCALAR features
- Handles absence of optional metadata fields gracefully
- Produces static features for Stage-1 models
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


# ---------------------------------------------------------------
# Helper: ensure item_id
# For FRN: item_id = city_id_store_id_product_id
# ---------------------------------------------------------------
def ensure_item_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # FRN uses product_id, city_id, store_id → combine them
    if "item_id" not in df.columns:
        df["item_id"] = (
            df.get("city_id", 0).astype(str)
            + "_"
            + df.get("store_id", 0).astype(str)
            + "_"
            + df.get("product_id", 0).astype(str)
        )

    df["item_id"] = df["item_id"].astype(str)
    return df


# ---------------------------------------------------------------
# Compute demand (sale_amount) statistics per item
# ---------------------------------------------------------------
def compute_series_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_item_id(df)
    if "sale_amount" not in df.columns:
        raise ValueError("FRN dataset missing 'sale_amount' column.")

    stats = (
        df.groupby("item_id")["sale_amount"]
        .agg(
            avg_demand="mean",
            std_demand="std",
            min_demand="min",
            max_demand="max",
            count="count",
        )
        .fillna(0)
    )

    # Coefficient of Variation
    stats["cv"] = stats["std_demand"] / (stats["avg_demand"] + 1e-8)

    # Zero-demand fraction
    zero_frac = (
        df.assign(is_zero=df["sale_amount"] <= 0)
        .groupby("item_id")["is_zero"]
        .mean()
        .rename("zero_fraction")
    )

    stats = stats.join(zero_frac, how="left").fillna(0)
    return stats


# ---------------------------------------------------------------
# Aggregate sequence columns into SCALAR per-item features
# (avoids ndarray cells → fixes AutoGluon 1.4 "unhashable ndarray" issues)
# ---------------------------------------------------------------
def compute_sequence_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_item_id(df)
    agg_parts = []

    # ---- hours_sale ----
    if "hours_sale" in df.columns:
        def agg_hours_sale(series: pd.Series) -> pd.Series:
            arrays = []
            for v in series:
                if v is None:
                    continue
                arr = np.asarray(v, dtype=float).flatten()
                if arr.size > 0:
                    arrays.append(arr)
            if not arrays:
                return pd.Series({
                    "avg_hourly_sale": np.nan,
                    "max_hourly_sale": np.nan,
                    "active_hour_fraction": np.nan,
                })
            all_hours = np.concatenate(arrays)
            return pd.Series({
                "avg_hourly_sale": float(np.mean(all_hours)),
                "max_hourly_sale": float(np.max(all_hours)),
                "active_hour_fraction": float(np.mean(all_hours > 0)),
            })

        hs = df.groupby("item_id")["hours_sale"].apply(agg_hours_sale)
        agg_parts.append(hs.to_frame() if isinstance(hs, pd.Series) else hs)

    # ---- hours_stock_status ----
    if "hours_stock_status" in df.columns:
        def agg_hours_stock(series: pd.Series) -> pd.Series:
            arrays = []
            for v in series:
                if v is None:
                    continue
                arr = np.asarray(v, dtype=float).flatten()
                if arr.size > 0:
                    arrays.append(arr)
            if not arrays:
                return pd.Series({
                    "stockout_hour_fraction": np.nan,
                })
            all_hours = np.concatenate(arrays)
            return pd.Series({
                "stockout_hour_fraction": float(np.mean(all_hours > 0.5)),
            })

        hs_stock = df.groupby("item_id")["hours_stock_status"].apply(agg_hours_stock)
        agg_parts.append(hs_stock.to_frame() if isinstance(hs_stock, pd.Series) else hs_stock)

    # No sequence fields present
    if not agg_parts:
        return pd.DataFrame({"item_id": df["item_id"].unique()})

    # Merge all aggregated parts safely
    seq_stats = agg_parts[0]
    for part in agg_parts[1:]:
        part = part.to_frame() if isinstance(part, pd.Series) else part
        seq_stats = seq_stats.join(part, how="outer")

    return seq_stats.reset_index()
# ---------------------------------------------------------------
# Price tier classification — FRN does NOT include price field
# So we fallback to "unknown" for all items
# ---------------------------------------------------------------
def classify_price_tier(df: pd.DataFrame) -> pd.Series:
    df = ensure_item_id(df)

    if "price" not in df.columns:
        # FRN has no price → return "unknown"
        return pd.Series(
            "unknown",
            index=df["item_id"].unique(),
            name="price_tier"
        )

    # If price exists, compute quantile buckets
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
# Product metadata static features (FRN does NOT have category/brand)
# We return only item_id unless source data includes optional fields
# ---------------------------------------------------------------
def derive_product_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_item_id(df)

    # FRN dataset does not include these fields but keep compatibility
    meta_cols = [
        c
        for c in ["category", "subcategory", "brand", "perishability"]
        if c in df.columns
    ]

    if not meta_cols:
        return pd.DataFrame({"item_id": df["item_id"].unique()})

    return df.groupby("item_id")[meta_cols].first().reset_index()


# ---------------------------------------------------------------
# KMeans clustering of time-series behavior
# ---------------------------------------------------------------
def cluster_time_series(stats: pd.DataFrame, n_clusters: int = 12) -> pd.Series:
    X = stats[["avg_demand", "cv", "zero_fraction"]].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # Reduce clusters for very small datasets
    n_clusters = max(1, min(n_clusters, len(X)))

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
    seq_stats = compute_sequence_aggregates(df)

    static_df = (
        stats.join(price_tier, how="left")
        .join(vol_class, how="left")
        .join(intermittent, how="left")
        .join(clusters, how="left")
        .reset_index()
    )

    # Join product metadata + sequence-based features
    static_df = static_df.merge(product_meta, on="item_id", how="left")

    if not seq_stats.empty:
        static_df = static_df.merge(seq_stats, on="item_id", how="left")

    return static_df


# ---------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------
def save_static_features(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_static_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
