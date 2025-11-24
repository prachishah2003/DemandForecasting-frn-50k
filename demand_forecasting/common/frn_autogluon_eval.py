# demand_forecasting/common/frn_autogluon_eval.py
# =============================================================
"""
Evaluation utilities for FRN forecasting experiments.

Functions:
  - wape, wpe, mape, smape, rmse, mae: basic metrics
  - quantile_loss (pinball), coverage: probabilistic metrics
  - evaluate_predictions(predictor, test_ts_df): main entrypoint that
    returns global metrics, per-series metrics (DataFrame), and
    probabilistic metrics (if available).

Notes:
  - Expects an AutoGluon TimeSeriesPredictor object and a
    TimeSeriesDataFrame (or object with .to_data_frame()) for test data.
  - Robust to whether the predictor returns point forecasts only or
    probabilistic forecasts with quantiles (e.g., '0.1','0.5','0.9').
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


# -------------------------
# Deterministic metrics
# -------------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(y_pred - y_true) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# -------------------------
# Probabilistic metrics
# -------------------------
def quantile_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """
    Pinball loss (quantile loss) for quantile q.
    """
    y = np.array(y_true)
    f = np.array(y_pred_q)
    diff = y - f
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.array(y_true)
    return float(np.mean((y >= lower) & (y <= upper)))


# -------------------------
# Helpers for conversion
# -------------------------
def _truth_to_dataframe(test_ts: TimeSeriesDataFrame) -> pd.DataFrame:
    """
    Convert AutoGluon TimeSeriesDataFrame (or similar) to a pandas DataFrame
    with columns: item_id, timestamp, target
    """
    # TimeSeriesDataFrame has .to_data_frame()
    if hasattr(test_ts, "to_data_frame"):
        df = test_ts.to_data_frame().reset_index()
    else:
        # assume it's already a DataFrame
        df = pd.DataFrame(test_ts).reset_index()
    # unify column name for target: could be 'target' or predictor.target outside
    # We'll keep column named 'target' if present, else try to infer (last numeric column)
    if "target" not in df.columns:
        # pick numeric column not in ['item_id','timestamp']
        num_cols = [c for c in df.columns if c not in ("item_id", "timestamp")]
        if len(num_cols) == 0:
            raise ValueError("Could not find target column in test dataframe.")
        df = df.rename(columns={num_cols[-1]: "target"})
    return df[["item_id", "timestamp", "target"]]


def _predictions_to_dataframe(predictions) -> pd.DataFrame:
    """
    Convert AutoGluon prediction object to a pandas DataFrame with:
      - item_id, timestamp, mean (point forecast)
      - and optionally quantile columns (strings like '0.1','0.5','0.9')
    """
    # Predictions from predictor.predict(test_df) can be:
    # - an AutoGluon TimeSeriesDataFrame-like object with .mean and .quantile columns
    # - a pandas.DataFrame-like object with columns
    # We'll try a few access patterns.
    try:
        # If predictions has .mean attribute (AutoGluon vX)
        if hasattr(predictions, "mean"):
            mean_df = predictions.mean.to_frame(name="mean").reset_index()
        else:
            # maybe predictions is already DataFrame-like with 'mean' column
            mean_df = pd.DataFrame(predictions)[["item_id", "timestamp", "mean"]].reset_index(drop=True)
    except Exception:
        # fallback: try converting to DataFrame and pick a sensible column
        pred_df_try = pd.DataFrame(predictions).reset_index()
        if "mean" in pred_df_try.columns:
            mean_df = pred_df_try[["item_id", "timestamp", "mean"]]
        else:
            # try to locate the first numeric column besides item_id/timestamp
            candidate_cols = [c for c in pred_df_try.columns if c not in ("item_id", "timestamp")]
            if len(candidate_cols) == 0:
                raise ValueError("No usable prediction column found in predictions.")
            mean_df = pred_df_try[["item_id", "timestamp", candidate_cols[0]]].rename(columns={candidate_cols[0]: "mean"})

    # Now try to extract quantiles if available
    # Several predictor implementations expose quantiles as attributes or as numeric string columns
    quantile_cols = {}
    try:
        # try .quantiles attribute (AutoGluon style)
        if hasattr(predictions, "quantiles"):
            # predictions.quantiles may itself be a mapping or DataFrame-like
            q_df = predictions.quantiles
            # If q_df is dict-like with numeric keys
            if isinstance(q_df, dict):
                for q, dfq in q_df.items():
                    # dfq likely a TimeSeriesDataFrame-like; convert to DataFrame
                    try:
                        quantile_cols[str(q)] = pd.DataFrame(dfq).reset_index()[["item_id", "timestamp", dfq.columns[-1]]].rename(columns={dfq.columns[-1]: str(q)})
                    except Exception:
                        # fallback: attempt to_frame
                        quantile_cols[str(q)] = q_df[q].to_frame(name=str(q)).reset_index()
            else:
                # q_df is DataFrame-like with columns containing quantile labels
                qdf = pd.DataFrame(q_df).reset_index()
                for col in qdf.columns:
                    if col not in ("item_id", "timestamp"):
                        quantile_cols[str(col)] = qdf[["item_id", "timestamp", col]].rename(columns={col: str(col)})
    except Exception:
        # swallow and try next approach
        quantile_cols = {}

    # Also check if the mean_df includes numeric-string columns e.g. '0.1', '0.5'
    df_all = mean_df.copy()
    try:
        pred_df_full = pd.DataFrame(predictions).reset_index()
        # detect quantile-looking columns
        for col in pred_df_full.columns:
            if col not in ("item_id", "timestamp", "mean") and isinstance(col, (str,)) and col.replace('.', '', 1).isdigit():
                quantile_cols[col] = pred_df_full[["item_id", "timestamp", col]].rename(columns={col: col})
        # merge mean + other columns if not already present
        if "mean" not in pred_df_full.columns:
            # merge mean_df with pred_df_full on item_id,timestamp
            df_all = mean_df.merge(pred_df_full, on=["item_id", "timestamp"], how="left")
        else:
            df_all = pred_df_full
    except Exception:
        pass

    # At this point df_all has mean column (ensured). quantile_cols is mapping of label->dataframe
    # Make final preds_df containing mean and quantile columns (wide format)
    preds_df = df_all[["item_id", "timestamp", "mean"]].copy()
    for qlabel, qdf in quantile_cols.items():
        # align qdf to preds_df by merging on keys
        qdf2 = qdf.rename(columns={qlabel: f"q_{qlabel}"})
        preds_df = preds_df.merge(qdf2[["item_id", "timestamp", f"q_{qlabel}"]], on=["item_id", "timestamp"], how="left")

    return preds_df


# -------------------------
# Evaluation entrypoint
# -------------------------
def evaluate_predictions(predictor: TimeSeriesPredictor, test_ts: TimeSeriesDataFrame) -> Dict[str, Any]:
    """
    Evaluate a trained AutoGluon TimeSeriesPredictor on test data.

    Returns a dictionary:
      {
        "global": {metric_name: value, ...},
        "per_series": pd.DataFrame (item_id + metrics cols),
        "probabilistic": {
            "quantile_losses": { 'q0.1': val, ... },
            "coverage": { 'p10_p90': val, ... }
        }
      }
    """
    # --- convert truth to df: item_id, timestamp, target ---
    truth_df = _truth_to_dataframe(test_ts)

    # --- get predictions from predictor ---
    preds = predictor.predict(test_ts)

    # convert preds into DataFrame with mean and optional q_* columns
    preds_df = _predictions_to_dataframe(preds)

    # Ensure timestamp types match for merge
    truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"])
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])

    # Merge on item_id + timestamp
    merged = truth_df.merge(preds_df, on=["item_id", "timestamp"], how="inner", validate="one_to_one")
    if merged.shape[0] == 0:
        raise RuntimeError("No overlapping rows between truth and predictions after merge.")

    y_true = merged["target"].values
    y_pred = merged["mean"].values

    # --- Global deterministic metrics ---
    global_metrics = {
        "WAPE": wape(y_true, y_pred),
        "WPE": wpe(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }

    # --- Per-series metrics ---
    per_series_records = []
    for item_id, group in merged.groupby("item_id"):
        yt = group["target"].values
        yp = group["mean"].values
        per_series_records.append(
            {
                "item_id": item_id,
                "WAPE": wape(yt, yp),
                "WPE": wpe(yt, yp),
                "MAPE": mape(yt, yp),
                "sMAPE": smape(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAE": mae(yt, yp),
            }
        )
    per_series_df = pd.DataFrame(per_series_records)

    # --- Probabilistic metrics (if quantiles exist) ---
    probabilistic: Dict[str, Any] = {"quantile_losses": {}, "coverage": {}}
    # detect quantile columns of form q_<label>
    qcols = [c for c in merged.columns if c.startswith("q_")]
    if len(qcols) > 0:
        # infer quantile labels
        quantiles = []
        for c in qcols:
            label = c[len("q_") :]
            try:
                q = float(label)
                quantiles.append((q, c))
            except Exception:
                continue
        # compute pinball loss per quantile
        for q, col in quantiles:
            qname = f"q{q:.2f}"
            probabilistic["quantile_losses"][qname] = quantile_loss(merged["target"].values, merged[col].values, q=q)
        # compute coverage for common intervals if present (example: 0.1 & 0.9)
        quantile_map = {float(c[len("q_"):]): c for c in qcols}
        # try 10-90, 5-95 if available
        for lower_q, upper_q in [(0.1, 0.9), (0.05, 0.95), (0.25, 0.75)]:
            if (lower_q in quantile_map) and (upper_q in quantile_map):
                lower_col = quantile_map[lower_q]
                upper_col = quantile_map[upper_q]
                cov = interval_coverage(merged["target"].values, merged[lower_col].values, merged[upper_col].values)
                probabilistic["coverage"][f"{int(lower_q*100)}-{int(upper_q*100)}_coverage"] = cov

    # also attempt to detect quantiles provided as numeric string columns (e.g., '0.1','0.5','0.9')
    numeric_qcols = [c for c in merged.columns if isinstance(c, str) and c.replace(".", "", 1).isdigit()]
    if len(numeric_qcols) > 0:
        # compute pinball loss for each
        for c in numeric_qcols:
            try:
                q = float(c)
            except Exception:
                continue
            probabilistic["quantile_losses"][f"q{q:.2f}"] = quantile_loss(merged["target"].values, merged[c].values, q=q)

        # coverage detection for numeric style
        numeric_map = {float(c): c for c in numeric_qcols}
        for lower_q, upper_q in [(0.1, 0.9), (0.05, 0.95), (0.25, 0.75)]:
            if (lower_q in numeric_map) and (upper_q in numeric_map):
                lower_col = numeric_map[lower_q]
                upper_col = numeric_map[upper_q]
                cov = interval_coverage(merged["target"].values, merged[lower_col].values, merged[upper_col].values)
                probabilistic["coverage"][f"{int(lower_q*100)}-{int(upper_q*100)}_coverage"] = cov

    # If no probabilistic info found, probabilistic dict stays empty-ish
    return {"global": global_metrics, "per_series": per_series_df, "probabilistic": probabilistic}
