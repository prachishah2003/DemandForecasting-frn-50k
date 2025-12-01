# =============================================================
# Train DeepAR Stage-1 on FRN (top 5000 item_ids)
# - Uses parquet: demand_forecasting/data/train/frn_train.parquet
# - Expands hours_sale / hours_stock_status into 24 scalar features
# - Loads static features separately into static_features_df
# - AutoGluon TimeSeriesPredictor (DeepAR)
# - Evaluation via frn_autogluon_eval.evaluate_predictions()
# =============================================================

import argparse
from pathlib import Path
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
    p.add_argument(
        "--max_items",
        type=int,
        help="Override maximum item series to train",
    )
    return p.parse_args()


# -------------------------------------------------------------
# Expand hourly arrays into 24 scalar columns
# -------------------------------------------------------------
def expand_array_column(df, col, prefix, size=24):
    if col not in df.columns:
        return df

    def fix(arr):
        if arr is None or isinstance(arr, float):
            return np.zeros(size)
        arr = np.array(arr).flatten()
        if len(arr) < size:
            arr = np.concatenate([arr, np.zeros(size - len(arr))])
        return arr[:size]

    M = np.vstack(df[col].apply(fix).values)
    new_cols = [f"{prefix}{i}" for i in range(size)]
    df = df.drop(columns=[col])
    return pd.concat([df, pd.DataFrame(M, columns=new_cols, index=df.index)], axis=1)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.experiment.seed)

    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq
    max_items = args.max_items or cfg.dataset.max_items

    output_dir = Path(cfg.experiment.output_dir) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("demand_forecasting/data/train/frn_train.parquet")
    static_path = Path(cfg.static_features.path)

    print(f"📥 Loading reduced FRN parquet: {train_path}")
    df = pd.read_parquet(train_path)
    print("Dataset shape:", df.shape)

    # Ensure item_id exists
    if "item_id" not in df.columns:
        df["item_id"] = (
            df["city_id"].astype(str)
            + "_"
            + df["store_id"].astype(str)
            + "_"
            + df["product_id"].astype(str)
        )
    df["dt"] = pd.to_datetime(df["dt"])

    # Keep only top-N item series
    unique_items = df["item_id"].value_counts().index[:max_items]
    df = df[df["item_id"].isin(unique_items)].copy()
    print(f"Using {len(unique_items)} item series")

    # Expand hourly arrays
    df = expand_array_column(df, "hours_sale", "h_sale_")
    df = expand_array_column(df, "hours_stock_status", "h_stock_")

    # Extract static features separately (AutoGluon rule)
    if cfg.static_features.enable and static_path.exists():
        print("📌 Loading static features...")
        static_df = load_static_features(static_path)
        static_df = static_df[static_df["item_id"].isin(unique_items)].copy()
        static_df = static_df.set_index("item_id")
        print("Static DF:", static_df.shape)
    else:
        print("⚠ STATIC DISABLED — using no static_features_df")
        static_df = None

    # Prepare TS dataframe (target only)
    df = df.rename(columns={"sale_amount": "target"})
    df = df.sort_values(["item_id", "dt"])
    ts = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="dt",
        static_features_df=static_df,
    )

    # Train/Test split aligned by prediction_length
    train_ts = ts.slice_by_timestep(slice(None, -prediction_length))
    test_ts = ts.slice_by_timestep(slice(-prediction_length, None))
    print(f"TS Train: {train_ts.shape}, TS Test: {test_ts.shape}")

    print("🚀 Training DeepAR...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        target="target",
        eval_metric="WQL",
        path=str(output_dir),
        verbosity=2,
    )

    hp = {
        "DeepAR": {
            "cell_type": cfg.models.deepar.params.cell_type,
            "hidden_size": cfg.models.deepar.params.hidden_size,
            "num_layers": cfg.models.deepar.params.num_layers,
            "dropout_rate": cfg.models.deepar.params.dropout_rate,
        }
    }

    predictor.fit(train_ts, hyperparameters=hp)

    print("📊 Evaluating with frn_autogluon_eval...")
    results = evaluate_predictions(predictor, test_ts)

    # Save results
    (output_dir / "metrics").mkdir(exist_ok=True)

    results["per_series"].to_csv(output_dir / "metrics/per_series.csv", index=False)
    pd.Series(results["global"]).to_json(output_dir / "metrics/global.json")
    pd.Series(results["probabilistic"]).to_json(output_dir / "metrics/probabilistic.json")

    print("\n🎯 Global metrics:")
    print(results["global"])
    print("\n📁 Saved results inside:", output_dir / "metrics")
    print("\n[✔] DeepAR training complete!")


if __name__ == "__main__":
    main()
