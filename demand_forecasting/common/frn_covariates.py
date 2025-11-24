# demand_forecasting/common/frn_covariates.py
# =============================================================

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utility: ensure item_id exists
# ---------------------------------------------------------------------
def _ensure_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is an 'item_id' column for grouping."""
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


# ---------------------------------------------------------------------
# Temporal Features
# ---------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar/time-based features from 'timestamp'.

    Requires:
        - 'timestamp' column (datetime-like)
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["dow"] = ts.dt.weekday                 # 0=Mon...6=Sun
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["dom"] = ts.dt.day
    df["month"] = ts.dt.month
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["year"] = ts.dt.year
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    return df


# ---------------------------------------------------------------------
# Demand Lags
# ---------------------------------------------------------------------
def add_demand_lags(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 7, 14),
) -> pd.DataFrame:
    """
    Add lagged demand features per series: demand_lag<lag>.

    Requires:
        - 'item_id' or (city_id, store_id, product_id)
        - 'timestamp'
        - 'demand'
    """
    df = _ensure_item_id(df.copy())
    df = df.sort_values(["item_id", "timestamp"])

    if "demand" not in df.columns:
        raise ValueError("Expected column 'demand' to create lag features.")

    for lag in lags:
        df[f"demand_lag{lag}"] = df.groupby("item_id")["demand"].shift(lag)

    return df


# ---------------------------------------------------------------------
# Rolling Windows / CV
# ---------------------------------------------------------------------
def add_demand_rolling(
    df: pd.DataFrame,
    windows: Iterable[int] = (7, 28),
) -> pd.DataFrame:
    """
    Add rolling mean windows and CV for demand.

    Adds:
        - demand_roll<w>
        - demand_cv14 (fixed 14-day coefficient of variation)
    """
    df = _ensure_item_id(df.copy())
    df = df.sort_values(["item_id", "timestamp"])

    if "demand" not in df.columns:
        raise ValueError("Expected column 'demand' to create rolling features.")

    for w in windows:
        df[f"demand_roll{w}"] = df.groupby("item_id")["demand"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )

    # 14-day CV
    w = 14
    grp = df.groupby("item_id")["demand"]
    mean14 = grp.transform(lambda s: s.rolling(w, min_periods=3).mean())
    std14 = grp.transform(lambda s: s.rolling(w, min_periods=3).std())
    df["demand_cv14"] = std14 / (mean14 + 1e-8)

    return df


# ---------------------------------------------------------------------
# Price & Promotion Features
# ---------------------------------------------------------------------
def add_price_and_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price, discount and promotion-related covariates.

    Optional columns:
        - price
        - discount
        - promo_flag (0/1)
    """
    df = _ensure_item_id(df.copy())
    df = df.sort_values(["item_id", "timestamp"])

    has_price = "price" in df.columns
    has_discount = "discount" in df.columns
    has_promo = "promo_flag" in df.columns

    # Price dynamics
    if has_price:
        df["price_pct_change_1d"] = (
            df.groupby("item_id")["price"]
            .pct_change()
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        df["price_roll7"] = df.groupby("item_id")["price"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )

    # Discount lag
    if has_discount:
        df["discount_lag1"] = (
            df.groupby("item_id")["discount"].shift(1).fillna(0.0)
        )

    # Promo intensity
    if has_discount and has_promo:
        df["promo_intensity"] = df["discount"].fillna(0) * df["promo_flag"]
    elif has_discount:
        df["promo_intensity"] = df["discount"].fillna(0)
    elif has_promo:
        df["promo_intensity"] = df["promo_flag"].astype(float)
    else:
        df["promo_intensity"] = 0.0

    # Promo recency and frequency
    if has_promo:
        promo = df.groupby("item_id")["promo_flag"]
        # helper cumulative index that resets on promo==1
        csum = promo.cumsum()
        block_id = csum.where(df["promo_flag"] == 1).ffill()
        df["days_since_last_promo"] = (
            df.groupby("item_id")[block_id.name]
            .cumcount()
            .fillna(999)
            .astype(int)
        )
        df["promo_count_last30"] = promo.transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
    else:
        df["days_since_last_promo"] = 999
        df["promo_count_last30"] = 0

    return df


# ---------------------------------------------------------------------
# Weather Features
# ---------------------------------------------------------------------
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather covariates & simple lags if present.

    Optional columns:
        - avg_temperature
        - avg_humidity
        - precpt
    """
    df = _ensure_item_id(df.copy())
    df = df.sort_values(["item_id", "timestamp"])

    for col in ["avg_temperature", "avg_humidity", "precpt"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("item_id")[col].shift(1)
            df[f"{col}_roll7"] = df.groupby("item_id")[col].transform(
                lambda s: s.rolling(7, min_periods=1).mean()
            )

    # Simple extreme temperature flags
    if "avg_temperature" in df.columns:
        t = df["avg_temperature"]
        df["extreme_heat_flag"] = (t >= t.quantile(0.95)).astype(int)
        df["extreme_cold_flag"] = (t <= t.quantile(0.05)).astype(int)

    return df


# ---------------------------------------------------------------------
# Stock / Availability Features
# ---------------------------------------------------------------------
def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stock / out-of-stock-related covariates if present.

    Optional columns:
        - oos_flag (1 if out-of-stock)
        - stock_on_hand
    """
    df = _ensure_item_id(df.copy())
    df = df.sort_values(["item_id", "timestamp"])

    if "oos_flag" in df.columns:
        df["oos_last7"] = df.groupby("item_id")["oos_flag"].transform(
            lambda s: s.rolling(7, min_periods=1).sum()
        )
        df["in_stock_flag"] = (1 - df["oos_flag"]).astype(int)
    else:
        df["oos_last7"] = 0
        df["in_stock_flag"] = 1

    if "stock_on_hand" in df.columns:
        df["stock_roll7"] = df.groupby("item_id")["stock_on_hand"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )

    return df


# ---------------------------------------------------------------------
# Holiday Features
# ---------------------------------------------------------------------
def add_holiday_features(
    df: pd.DataFrame,
    holiday_dates: Optional[Iterable[pd.Timestamp]],
) -> pd.DataFrame:
    """
    Add holiday indicators & proximity.

    Args:
        holiday_dates: iterable of pd.Timestamp or None.
    """
    df = df.copy()
    if holiday_dates is None:
        df["is_holiday"] = 0
        df["days_to_next_holiday"] = 30
        return df

    holidays_sorted = np.sort(pd.to_datetime(list(holiday_dates)).values)

    df["is_holiday"] = df["timestamp"].isin(holidays_sorted).astype(int)

    idx = holidays_sorted.searchsorted(
        df["timestamp"].values.astype("datetime64[ns]")
    )
    next_h = np.where(
        idx >= len(holidays_sorted),
        np.datetime64("NaT"),
        holidays_sorted[idx],
    )
    dt = (
        next_h - df["timestamp"].values.astype("datetime64[ns]")
    ) / np.timedelta64(1, "D")
    df["days_to_next_holiday"] = np.where(
        np.isfinite(dt), dt, 30
    ).astype(int)

    return df


# ---------------------------------------------------------------------
# Imputation / Recovery Features
# ---------------------------------------------------------------------
def add_imputation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add uncertainty & flag features from latent demand recovery.

    Optional columns:
        - recovered_std
        - recovered_flag (1 if imputed)
    """
    df = df.copy()

    if "recovered_std" in df.columns:
        median_std = df["recovered_std"].median()
        df["imputation_std"] = df["recovered_std"].fillna(median_std)
    else:
        df["imputation_std"] = 0.0

    if "recovered_flag" in df.columns:
        df["imputation_flag"] = df["recovered_flag"].astype(int)
    else:
        df["imputation_flag"] = 0

    return df


# ---------------------------------------------------------------------
# Master entrypoint
# ---------------------------------------------------------------------
def add_all_covariates(
    df: pd.DataFrame,
    holiday_dates: Optional[Iterable[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Main entrypoint: add all covariates to the dataframe.

    Steps:
      1. Ensure item_id exists
      2. Temporal features
      3. Demand lags
      4. Demand rolling stats
      5. Price & promo
      6. Weather
      7. Stock
      8. Holiday
      9. Imputation features
      10. Fill remaining NaNs in covariates with 0
    """
    df = _ensure_item_id(df.copy())

    # Temporal
    df = add_temporal_features(df)

    # Demand
    df = add_demand_lags(df, lags=(1, 7, 14))
    df = add_demand_rolling(df, windows=(7, 28))

    # Price & promo
    df = add_price_and_promo_features(df)

    # Weather
    df = add_weather_features(df)

    # Stock
    df = add_stock_features(df)

    # Holiday
    df = add_holiday_features(df, holiday_dates=holiday_dates)

    # Imputation / recovery
    df = add_imputation_features(df)

    # Final cleaning: fill remaining NaNs in covariates
    covariate_cols = [
        c for c in df.columns
        if c not in ("demand",)  # keep demand untouched
    ]
    df[covariate_cols] = df[covariate_cols].fillna(0)

    return df
