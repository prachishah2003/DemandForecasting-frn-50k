# =============================================================
# Train DeepAR Stage-1 on FRN (top N item_ids)
# - Uses parquet: demand_forecasting/data/train/frn_train.parquet
# - Expands hours_sale / hours_stock_status into 24 scalar features
# - Loads static features into static_features_df (not merged into df)
# - Uses AutoGluon TimeSeriesPredictor (DeepAR)
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
def expand_array_column(df: pd.DataFrame, col: str, prefix: str, size: int = 24) -> pd.DataFrame:
    """Expand an array-like column into size scalar columns."""
    if col not in df.columns:
        return df

    def fix(arr):
        # Handle None, NaN, scalars
        if arr is None or (isinstance(arr, float) and np.isnan(arr)):
            return np.zeros(size, dtype=float)
        arr = np.array(arr, dtype=float).flatten()
        if arr.size < size:
            arr = np.concatenate([arr, np.zeros(size - arr.size, dtype=float)])
        return arr[:size]

    mat = np.vstack(df[col].apply(fix).values)
    new_cols = [f"{prefix}{i}" for i in range(size)]
    expanded = pd.DataFrame(mat, columns=new_cols, index=df.index)

    df = df.drop(columns=[col])
    df = pd.concat([df, expanded], axis=1)
    return df


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Seeding
    seed = getattr(cfg.experiment, "seed", 123)
    seed_everything(seed)

    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq
    max_items = args.max_items or getattr(cfg.dataset, "max_items", 5000)

    output_dir = Path(cfg.experiment.output_dir) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("demand_forecasting/data/train/frn_train.parquet")
    static_path = Path(cfg.static_features.path)

    # ----------------- Load reduced parquet -----------------
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

    # Ensure datetime
    df["dt"] = pd.to_datetime(df["dt"])

    # ----------------- Keep only top-N item series -----------------
    unique_items = df["item_id"].value_counts().index[:max_items]
    df = df[df["item_id"].isin(unique_items)].copy()
    print(f"Using {len(unique_items)} item series")

    # ----------------- Expand hourly arrays -----------------
    df = expand_array_column(df, "hours_sale", "h_sale_", size=24)
    df = expand_array_column(df, "hours_stock_status", "h_stock_", size=24)
    print("After expanding hourly cols:", df.shape)

    # ----------------- Static features as separate DataFrame -----------------
       # Convert static_df to proper static feature format
    if cfg.static_features.enable and static_path.exists():
        print("📌 Loading static features...")
        static_df = load_static_features(static_path)

    # Keep only item_ids used in this training subset
        static_df = static_df[static_df["item_id"].isin(unique_items)].copy()

    # Drop hourly features accidentally inside static (if any)
        bad_cols = [c for c in static_df.columns if "hour" in c]
        if bad_cols:
            print(f"⚠ Removing hourly columns from static: {bad_cols}")
            static_df = static_df.drop(columns=bad_cols)

    # Ensure one row per item_id
        static_df = static_df.groupby("item_id").first().reset_index()

        print("Static DF after cleanup:", static_df.shape)
    else:
        print("⚠ Static features disabled or missing")
        static_df = None

    # ----------------- Build TimeSeriesDataFrame -----------------
    # Rename target
    df = df.rename(columns={"sale_amount": "target"})
    if "target" not in df.columns:
        raise ValueError("Column 'sale_amount' (renamed to 'target') not found in dataframe.")

    # Sort for TS
    df = df.sort_values(["item_id", "dt"])

    ts = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="dt",
        static_features_df=static_df,
    )

    # ----------------- Train / Test split -----------------
    #train_ts, test_ts = ts.train_test_split(prediction_length=prediction_length)
    #print(f"TS Train shape: {train_ts.shape}, TS Test shape: {test_ts.shape}")
    train_ts = ts.slice_by_timestep(None, -prediction_length)      # up to -prediction_length
    test_ts = ts.slice_by_timestep(-prediction_length, None)       # last prediction_length steps

    print(f"Train shape: {train_ts.shape} | Test shape: {test_ts.shape}")
    print(f"Train series: {len(train_ts.item_ids)} | Test series: {len(test_ts.item_ids)}")

    # ----------------- Train DeepAR via AutoGluon -----------------
    print("🚀 Training DeepAR...")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        target="target",
        eval_metric="WQL",
        path=str(output_dir),
        verbosity=2,
    )

    deepar_cfg = cfg.models.deepar.params

    # Build hyperparameters dict, being careful with lr / learning_rate clash
    deepar_params = {}

    if hasattr(deepar_cfg, "cell_type"):
        deepar_params["cell_type"] = deepar_cfg.cell_type
    if hasattr(deepar_cfg, "hidden_size"):
        deepar_params["hidden_size"] = deepar_cfg.hidden_size
    elif hasattr(deepar_cfg, "num_cells"):
        deepar_params["hidden_size"] = deepar_cfg.num_cells

    if hasattr(deepar_cfg, "dropout_rate"):
        deepar_params["dropout_rate"] = deepar_cfg.dropout_rate
    if hasattr(deepar_cfg, "num_layers"):
        deepar_params["num_layers"] = deepar_cfg.num_layers

    # ⚠ Avoid specifying both 'lr' and 'learning_rate' – AG / GluonTS treat them as aliases.
    # If you really want to set it, do it via 'lr' only, or just rely on defaults.
    if hasattr(deepar_cfg, "lr"):
        deepar_params["lr"] = deepar_cfg.lr
    elif hasattr(deepar_cfg, "learning_rate"):
        deepar_params["lr"] = deepar_cfg.learning_rate

    hyperparameters = {
        "DeepAR": deepar_params,
    }

    predictor.fit(
        train_data=train_ts,
        hyperparameters=hyperparameters,
    )

    # ----------------- Evaluation -----------------
    print("📊 Evaluating with frn_autogluon_eval...")
    # IMPORTANT: tell the eval helper that test_ts is already the test split
    results = evaluate_predictions(
        predictor,
        train_ts,
        test_ts,
        assume_already_aligned=True,  # <- you add this flag in frn_autogluon_eval
    )
    # Save results
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    results["per_series"].to_csv(metrics_dir / "per_series.csv", index=False)
    pd.Series(results["global"]).to_json(metrics_dir / "global.json")
    pd.Series(results["probabilistic"]).to_json(metrics_dir / "probabilistic.json")

    print("\n🎯 Global metrics:")
    print(results["global"])
    print("\n📁 Saved results inside:", metrics_dir)
    print("\n[✔] DeepAR training complete!")


if __name__ == "__main__":
    main()
