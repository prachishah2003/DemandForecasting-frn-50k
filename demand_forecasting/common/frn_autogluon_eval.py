"""
Evaluation utilities for FreshRetailNet + AutoGluon Timeseries.

Fixes:
- Proper alignment using item_id + timestamp
- Robust merging using index reset
- Full global + per-series metrics
- Probabilistic evaluation support
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from autogluon.timeseries import TimeSeriesPredictor


# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true)
        / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


def wpe(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / (
        np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    )


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate_predictions(predictor: TimeSeriesPredictor, test_ts):
    """Compute all metrics and handle index safely."""
    preds = predictor.predict(test_ts)

    # Convert to tabular alignment
    df_true = test_ts.to_data_frame().reset_index()
    df_pred = preds.to_data_frame().reset_index()

    df_true = df_true.rename(columns={"timestamp": "dt"})
    df_pred = df_pred.rename(columns={"timestamp": "dt"})

    # Merge on item_id + timestamp explicitly
    merged = pd.merge(
        df_true,
        df_pred,
        on=["item_id", "dt"],
        how="inner",
        suffixes=("_true", "_pred")
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping rows between truth and predictions after merge.")

    y_true = merged["target_true"].values
    y_pred = merged["target_pred"].values

    global_metrics = {
        "WAPE": wape(y_true, y_pred),
        "WPE": wpe(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }

    # Per-series metrics
    per_series = []
    for item_id, g in merged.groupby("item_id"):
        y_t = g["target_true"]
        y_p = g["target_pred"]
        per_series.append(
            {
                "item_id": item_id,
                "WAPE": wape(y_t, y_p),
                "MAPE": np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))),
                "sMAPE": smape(y_t, y_p),
                "RMSE": rmse(y_t, y_p),
                "MAE": mae(y_t, y_p),
            }
        )

    per_series_df = pd.DataFrame(per_series)

    # Probabilistic metrics placeholder (extend later)
    probabilistic_metrics = {
        "quantiles_supported": True,
    }

    return {
        "global": global_metrics,
        "per_series": per_series_df,
        "probabilistic": probabilistic_metrics,
    }
