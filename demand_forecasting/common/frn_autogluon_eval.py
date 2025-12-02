"""
Evaluation utilities for FreshRetailNet + AutoGluon Timeseries.

✔ Stable alignment using item_id + timestamp
✔ Works with AutoGluon 1.4 predictor.predict()
✔ Supports global + per-series metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------- Metrics ----------------
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


# ---------------- Evaluation ----------------
def evaluate_predictions(pred_df: pd.DataFrame, test_ts):
    """
    pred_df -> output of predictor.predict(train_ts)
    test_ts -> TimeSeriesDataFrame (truth)
    """

    df_true = test_ts.to_data_frame().reset_index()

    # Predictor output DF
    df_pred = pred_df.reset_index()
    df_pred = df_pred.rename(columns={"timestamp": "dt"})

    # Align truth DF
    df_true = df_true.reset_index()
    df_true = df_true.rename(columns={"timestamp": "dt"})

    # AutoGluon renames predicted target column to "mean"
    pred_col = "mean" if "mean" in df_pred.columns else df_pred.columns[-1]
    true_col = "target" if "target" in df_true.columns else df_true.columns[-1]

    merged = pd.merge(
        df_true[["item_id", "dt", true_col]],
        df_pred[["item_id", "dt", pred_col]],
        on=["item_id", "dt"],
        how="inner"
    )

    if merged.empty:
        raise RuntimeError("⚠ No overlapping timestamps between prediction and truth!")

    y_true = merged[true_col].values
    y_pred = merged[pred_col].values

    # Global metrics
    global_metrics = {
        "WAPE": wape(y_true, y_pred),
        "WPE": wpe(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }

    # Per-series metrics
    per_series = []
    for item, g in merged.groupby("item_id"):
        yt = g[true_col].values
        yp = g[pred_col].values
        per_series.append({
            "item_id": item,
            "WAPE": wape(yt, yp),
            "sMAPE": smape(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
        })

    probabilistic_metrics = {"note": "DeepAR default mean metric only"}

    return {
        "global": global_metrics,
        "per_series": pd.DataFrame(per_series),
        "probabilistic": probabilistic_metrics,
    }
