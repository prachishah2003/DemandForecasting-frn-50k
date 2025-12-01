# =============================================================
# Train DeepAR Stage-1 on FRN (top 5000 item_ids)
# - Uses parquet: demand_forecasting/data/train/frn_train.parquet
# - Expands hours_sale / hours_stock_status into 24 scalar features
# - Merges static features from data/static_features_stage1.parquet
# - Uses AutoGluon TimeSeriesPredictor (DeepAR)
# =============================================================

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from demand_forecasting.common.utils import seed_everything
from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.static_feature_generator import load_static_features
from demand_forecasting.common.frn_autogluon_eval import evaluate_predictions


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="demand_forecasting/config/default.yaml",
        help="Path to YAML config",
    )
    return p.parse_args()


# -------------------------------------------------------------
# Utility: expand hourly array columns → 24 scalar columns each
# -------------------------------------------------------------
def expand_array_column(df: pd.DataFrame, col: str, prefix: str, size: int = 24) -> pd.DataFrame:
    if col not in df.columns:
        return df

    def to_vec(x):
        if x is None:
            return np.zeros(size, dtype=float)
        arr = np.array(x, dtype=float).flatten()
        if arr.size < size:
            # pad with zeros
            pad = np.zeros(size - arr.size, dtype=float)
            arr = np.concatenate([arr, pad])
        return arr[:size]

    mat = np.vstack(df[col].apply(to_vec).values)
    cols = [f"{prefix}{i}" for i in range(size)]
    expanded = pd.DataFrame(mat, columns=cols, index=df.index)

    df = df.drop(columns=[col])
    df = pd.concat([df, expanded], axis=1)
    return df


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ----------------- seeding & paths -----------------
    seed = getattr(cfg.experiment, "seed", 123)
    seed_everything(seed)

    output_dir = Path(cfg.experiment.output_dir) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq
    max_items = getattr(cfg.dataset, "max_items", 5000)

    train_path = Path("demand_forecasting/data/train/frn_train.parquet")
    static_path = Path(cfg.static_features.path)

    # ----------------- load reduced parquet -----------------
    print(f"📥 Loading reduced train parquet: {train_path}")
    df = pd.read_parquet(train_path)
    print("Dataset:", df.shape)

    # Ensure dt is datetime and item_id exists
    if "item_id" not in df.columns:
        df["item_id"] = (
            df["city_id"].astype(str)
            + "_"
            + df["store_id"].astype(str)
            + "_"
            + df["product_id"].astype(str)
        )

    df["dt"] = pd.to_datetime(df["dt"])

    # ----------------- subset to top N item series -----------------
    unique_items = df["item_id"].value_counts().index[:max_items]
    df = df[df["item_id"].isin(unique_items)].copy()
    print(f"Using only {len(unique_items)} item series")

    # ----------------- expand hourly arrays -----------------
    df = expand_array_column(df, "hours_sale", "hour_sale_", size=24)
    df = expand_array_column(df, "hours_stock_status", "hour_stock_", size=24)
    print(f"🧮 Expanded hourly columns → shape: {df.shape}")

    # ----------------- load / align static features -----------------
    if cfg.static_features.enable and static_path.exists():
        print("📄 Loading static features...")
        static_df = load_static_features(static_path)
        # only keep features for item_ids we actually train on
        static_df = static_df[static_df["item_id"].isin(unique_items)].copy()
        # merge per-item static features into each row
        df = df.merge(static_df, on="item_id", how="left")
        print("Static features merged. df shape:", df.shape)
    else:
        print("⚠️ Static features disabled or file not found, continuing without merge.")

    # ----------------- prepare TimeSeriesDataFrame -----------------
    # AutoGluon expects 'target' as column name
    df = df.rename(columns={"sale_amount": "target"})
    if "target" not in df.columns:
        raise ValueError("Column 'sale_amount' (renamed to 'target') not found in dataframe.")

    # Sort for TS construction
    df_ts = df.sort_values(["item_id", "dt"])
    ts = TimeSeriesDataFrame.from_data_frame(
        df_ts,
        id_column="item_id",
        timestamp_column="dt",
    )

    # Split into train / test
    train_ts, test_ts = ts.train_test_split(prediction_length)
    print(f"Train series: {len(train_ts.item_ids)} | Test series: {len(test_ts.item_ids)}")

    # ----------------- train DeepAR via AutoGluon -----------------
    print("🚀 Training DeepAR...")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        path=str(output_dir),
        eval_metric="WQL",
        target="target",
        verbosity=2,
    )

    # Map YAML params directly into DeepAR hyperparameters
    deepar_params = {
        "cell_type": cfg.models.deepar.params.cell_type,
        "hidden_size": cfg.models.deepar.params.hidden_size,
        "num_layers": cfg.models.deepar.params.num_layers,
        "dropout_rate": cfg.models.deepar.params.dropout_rate,
        # IMPORTANT: no 'lr' / 'learning_rate' here to avoid conflicts.
    }

    hyperparameters = {
        "DeepAR": deepar_params,
    }

    predictor.fit(
        train_data=train_ts,
        hyperparameters=hyperparameters,
    )

    # ----------------- evaluation -----------------
    print("📊 Evaluating...")
    results = evaluate_predictions(predictor, test_ts)

    print("Global metrics:")
    print(results["global"])

    # Save outputs
    per_series_path = output_dir / "per_series.csv"
    global_path = output_dir / "global_metrics.json"
    prob_path = output_dir / "prob_metrics.json"

    results["per_series"].to_csv(per_series_path, index=False)
    pd.Series(results["global"]).to_json(global_path)
    pd.Series(results["probabilistic"]).to_json(prob_path)

    print("Saved:")
    print(" -", per_series_path)
    print(" -", global_path)
    print(" -", prob_path)
    print("[DeepAR] Done!")


if __name__ == "__main__":
    main()
