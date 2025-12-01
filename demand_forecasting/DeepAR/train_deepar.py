# demand_forecasting/DeepAR/train_deepar.py
# =============================================================
"""
Train DeepAR on Reduced FreshRetailNet (5000 series)
- Uses reduced parquet dataset already saved in Colab
- Static features automatically filtered to matching item_ids
- AutoGluon 1.4 API
"""

import argparse
from pathlib import Path
import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor
from demand_forecasting.common.utils import seed_everything
from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.static_feature_generator import (
    generate_static_features,
    save_static_features,
    load_static_features,
)
from demand_forecasting.common.holiday_utils import load_india_holidays
from demand_forecasting.common.frn_autogluon_data import df_to_time_series
from demand_forecasting.common.frn_autogluon_eval import evaluate_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="demand_forecasting/config/default.yaml")
    p.add_argument("--generate_static", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    output_dir = Path(cfg.experiment.output_dir) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq

    # --------------------------------------------------------
    # Load reduced dataset parquet
    # --------------------------------------------------------
    train_path = "demand_forecasting/data/train/frn_train.parquet"
    print(f"[DeepAR] Loading reduced dataset: {train_path}")
    df_train = pd.read_parquet(train_path)

    print("[DeepAR] Train dataframe:", df_train.shape)
    unique_items = df_train["item_id"].unique()
    print(f"[DeepAR] Unique item_ids: {len(unique_items)}")

    # --------------------------------------------------------
    # Load India holiday dates (optional)
    # --------------------------------------------------------
    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(
            start_year=cfg.holidays.start_year,
            end_year=cfg.holidays.end_year,
        )

    # --------------------------------------------------------
    # Static features
    # --------------------------------------------------------
    static_path = cfg.static_features.path
    if args.generate_static or not Path(static_path).exists():
        print("[DeepAR] Generating static features...")
        static_df = generate_static_features(df_train)
        save_static_features(static_df, static_path)
    else:
        print("[DeepAR] Loading static features from disk...")
        static_df = load_static_features(static_path)

    # Keep only items in df_train
    static_df = static_df[static_df["item_id"].isin(unique_items)]
    print("[DeepAR] Static features:", static_df.shape)

    # --------------------------------------------------------
    # Convert to TimeSeriesDataFrame
    # --------------------------------------------------------
    print("[DeepAR] Converting to TimeSeriesDataFrame...")
    ts_full = df_to_time_series(
        df_train,
        prediction_length=prediction_length,
        freq=freq
    )

    train, test = ts_full.train_test_split(prediction_length)
    print(f"Train items: {len(train.item_ids)}  |  Test items: {len(test.item_ids)}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    print("[DeepAR] Training model...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric="WQL",
        path=str(output_dir),
    )

    hyperparams = {"DeepAR": dict(cfg.models.deepar.params)}

    predictor.fit(
        train_data=train,
        hyperparameters=hyperparams,
        static_features=static_df
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    print("[DeepAR] Evaluating...")
    results = evaluate_predictions(predictor, test)
    print(results["global"])

    results["per_series"].to_csv(output_dir / "per_series.csv", index=False)
    pd.Series(results["global"]).to_json(output_dir / "global_metrics.json")
    pd.Series(results["probabilistic"]).to_json(output_dir / "prob.json")

    print("\n======== TRAINING COMPLETE =========")


if __name__ == "__main__":
    main()
