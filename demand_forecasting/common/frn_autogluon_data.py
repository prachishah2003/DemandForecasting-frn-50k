"""
FRN → AutoGluon TimeSeriesDataFrame converter

Handles:
 - sale_amount as target
 - sequence columns hours_sale + hours_stock_status
 - out-of-stock hours count
 - holidays
"""

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


# ---------- Helper to ensure item_id exists ----------
def ensure_item_id(df):
    if "item_id" not in df.columns:
        df["item_id"] = (
            df.get("city_id", 0).astype(str)
            + "_"
            + df.get("store_id", 0).astype(str)
            + "_"
            + df.get("product_id", 0).astype(str)
        )
    return df


# ---------- Sequence dtype cleaning ----------
def normalize_sequence_columns(df):
    for col in ["hours_sale", "hours_stock_status"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: list(x) if hasattr(x, "__iter__") else []
            )
    return df


# ---------- Main loader ----------
def load_freshretailnet_raw(split: str):
    """
    Uses HuggingFace dataset locally converted to DataFrame.
    """
    from datasets import load_dataset

    hf = load_dataset("Dingdong-Inc/FreshRetailNet-50K")[split]
    return hf.to_pandas()


def load_frn_for_forecasting(
    split: str = "train",
    demand_type: str = "censored",
    recovered_path: str = None,
    prediction_length: int = 7,
    holiday_dates=None,
):
    df = load_freshretailnet_raw(split)
    df = ensure_item_id(df)
    df = normalize_sequence_columns(df)

    df.rename(columns={"sale_amount": "target"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["dt"])

    # Holiday flag applied if config enabled
    if holiday_dates is not None:
        df["holiday_flag"] = df["dt"].isin(holiday_dates).astype(int)

    # Required columns for AG 1.4
    required = ["item_id", "timestamp", "target"]
    df = df[required + [c for c in df.columns if c not in required]]

    # Convert to sorted AG format:
    df.sort_values(["item_id", "timestamp"], inplace=True)

    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
        target_column="target",
    )
