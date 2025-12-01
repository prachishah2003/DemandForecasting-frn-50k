# demand_forecasting/DeepAR/train_deepar.py
# =============================================================
"""
Train DeepAR on FreshRetailNet Selected Subset (5000 items)
- Uses AutoGluon 1.4 API
- Static features auto-filtered to keep only used item_ids
"""

import argparse
from pathlib import Path
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from demand_forecasting.common.utils import seed_everything
from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.frn_autogluon_data import (
    load_frn_for_forecasting,
)
from demand_forecasting.common.static_feature_generator import (
    generate_static_features,
    save_static_features,
    load_static_features,
)
from demand_forecasting.common.holiday_utils import load_india_holidays
from demand_forecasting.common.frn_autogluon_eval import evaluate_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="demand_forecasting/config/default.yaml",
    )
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
    max_items = cfg.dataset.max_items or 5000

    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(
            start_year=cfg.holidays.start_year,
            end_year=cfg.holidays.end_year,
        )

    # --------------------------------------------------------
    # Load RAW FRN (train split), apply subset filtering FIRST
    # --------------------------------------------------------
    print(f"[DeepAR] Loading FRN raw dataset...")
    df_train = load_frn_for_forecasting(
        split="train",
        demand_type=cfg.dataset.demand_type,
        recovered_path=cfg.dataset.recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
        return_df=True,
    ).reset_index()

    unique_items = df_train["item_id"].unique()[:max_items]
    df_train = df_train[df_train["item_id"].isin(unique_items)]
    print(f"[DeepAR] Using only first {len(unique_items)} item series")

    # --------------------------------------------------------
    # Static features
    # --------------------------------------------------------
    static_path = cfg.static_features.path

    if args.generate_static or not Path(static_path).exists():
        print("[DeepAR] Generating static features...")
        static_df = generate_static_features(df_train)
        save_static_features(static_df, static_path)
        print(f"Saved static features: {static_path}")
    else:
        print("[DeepAR] Loading static features from disk...")
        static_df = load_static_features(static_path)
        static_df = static_df[static_df["item_id"].isin(unique_items)]

    # --------------------------------------------------------
    # Convert to TimeSeriesDataFrame
    # --------------------------------------------------------
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=cfg.dataset.demand_type,
        recovered_path=cfg.dataset.recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
        item_ids=unique_items.tolist(),
    )

    train, test = ts.train_test_split(prediction_length)
    print(f"Train items: {len(train.item_ids)} | Test items: {len(test.item_ids)}")

    # --------------------------------------------------------
    print("[DeepAR] Training model...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        path=str(output_dir),
        eval_metric="WQL",
    )

    hyperparameters = {
        "DeepAR": cfg.models.deepar.params.to_dict(),
    }

    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        static_features=static_df,
    )

    # --------------------------------------------------------
    print("[DeepAR] Evaluating...")
    results = evaluate_predictions(predictor, test)
    print(results["global"])

    results["per_series"].to_csv(output_dir / "per_series.csv", index=False)
    pd.Series(results["global"]).to_json(output_dir / "global_metrics.json")
    pd.Series(results["probabilistic"]).to_json(output_dir / "prob.json")

    print("[DeepAR] Done!")


if __name__ == "__main__":
    main()
