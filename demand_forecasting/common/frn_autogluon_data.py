# demand_forecasting/common/frn_autogluon_data.py
# =============================================================

from __future__ import annotations

from typing import Optional, List
import pandas as pd
from datasets import load_dataset
from autogluon.timeseries import TimeSeriesDataFrame

from demand_forecasting.common.frn_covariates import add_all_covariates


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _build_series_id(df: pd.DataFrame) -> pd.Series:
    """Create unique series key: city_store_product."""
    return (
        df.get("city_id", "").astype(str)
        + "_"
        + df.get("store_id", "").astype(str)
        + "_"
        + df.get("product_id", "").astype(str)
    )


# ---------------------------------------------------------------------
# Load FreshRetailNet from HuggingFace
# ---------------------------------------------------------------------
def load_freshretailnet_raw(split: str = "train") -> pd.DataFrame:
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    return dataset[split].to_pandas()


# ---------------------------------------------------------------------
# Merge recovered or censored demand
# ---------------------------------------------------------------------
def maybe_merge_recovered_demand(
    df: pd.DataFrame,
    demand_type: str,
    recovered_path: Optional[str] = None,
    recovered_column: str = "recovered_sale_amount",
) -> pd.DataFrame:
    """
    Merge censored vs recovered demand based on config.

    If demand_type = 'censored':
        demand = sale_amount

    If demand_type = 'recovered':
        Use recovered_sale_amount, and optionally recovered_std + recovered_flag.
    """
    df = df.copy()

    # ------------------------------------------
    # Case 1: Censored
    # ------------------------------------------
    if demand_type == "censored":
        df["demand"] = df["sale_amount"]
        return df

    # ------------------------------------------
    # Case 2: Recovered
    # ------------------------------------------
    if demand_type == "recovered":
        if recovered_path is None:
            raise ValueError("Recovered path must be provided for demand_type='recovered'.")

        # Load recovery output
        rec = (
            pd.read_parquet(recovered_path)
            if recovered_path.endswith(".parquet")
            else pd.read_csv(recovered_path)
        )

        key_cols = ["city_id", "store_id", "product_id", "dt"]
        for k in key_cols:
            if k not in df.columns or k not in rec.columns:
                raise ValueError(f"Missing key {k} in extracted or raw data.")

        merge_cols = key_cols + [recovered_column]
        # include uncertainty cols if present
        for extra in ["recovered_std", "recovered_flag"]:
            if extra in rec.columns:
                merge_cols.append(extra)

        df = df.merge(rec[merge_cols], on=key_cols, how="left", validate="one_to_one")

        # Fallback to original sale_amount for rows without recovery
        df["demand"] = df[recovered_column].fillna(df["sale_amount"])
        return df

    raise ValueError(f"Unknown demand_type={demand_type}. Expected 'censored' or 'recovered'.")


# ---------------------------------------------------------------------
# Convert to AutoGluon TimeSeriesDataFrame
# ---------------------------------------------------------------------
def make_timeseries_dataframe(
    df: pd.DataFrame,
    prediction_length: int,
    holiday_dates: Optional[List[pd.Timestamp]] = None,
) -> TimeSeriesDataFrame:
    """
    Transform cleaned + merged FRN dataframe into AutoGluon TimeSeriesDataFrame.

    Steps:
        - Build item_id and timestamp
        - Ensure demand column exists
        - Apply covariates via add_all_covariates(...)
        - Drop raw-only columns (dt, sale_amount)
    """
    df = df.copy()

    # Build unique item_id
    df["item_id"] = _build_series_id(df)
    df["timestamp"] = pd.to_datetime(df["dt"])

    # Ensure target exists
    if "demand" not in df.columns:
        df["demand"] = df.get("sale_amount", 0)

    # Add covariates
    df = add_all_covariates(df, holiday_dates=holiday_dates)

    # Remove raw-only columns
    drop_cols = {"dt", "sale_amount"}
    keep_cols = [c for c in df.columns if c not in drop_cols]
    df = df[keep_cols]

    # Convert to AutoGluon format
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
        target_column="demand",
    )
    return ts_df


# ---------------------------------------------------------------------
# High-level loader used in training scripts
# ---------------------------------------------------------------------
def load_frn_for_forecasting(
    split: str = "train",
    demand_type: str = "censored",
    recovered_path: Optional[str] = None,
    prediction_length: int = 7,
    holiday_dates: Optional[List[pd.Timestamp]] = None,
) -> TimeSeriesDataFrame:
    """
    Master loader used by all training scripts.

    Loads:
        HF → merge censored/recovered → apply all covariates → AG TimeSeriesDataFrame
    """
    df = load_freshretailnet_raw(split)

    df = maybe_merge_recovered_demand(
        df=df,
        demand_type=demand_type,
        recovered_path=recovered_path,
    )

    ts_df = make_timeseries_dataframe(
        df=df,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    return ts_df
