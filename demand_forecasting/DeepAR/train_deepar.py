# demand_forecasting/DeepAR/train_deepar.py
# =============================================================
"""
Train DeepAR on FreshRetailNet-50K with:
  - Stage-1 feature set (static item-level features)
  - Support for censored/recovered demand
  - India holiday flags
  - AutoGluon 1.4 compatibility
  - Full evaluation metrics
"""

import argparse
from pathlib import Path
import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor

from demand_forecasting.common.utils import seed_everything
from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.frn_autogluon_data import load_frn_for_forecasting
from demand_forecasting.common.static_feature_generator import (
    generate_static_features,
    save_static_features,
    load_static_features,
)
from demand_forecasting.common.holiday_utils import load_india_holidays
from demand_forecasting.common.frn_autogluon_eval import evaluate_predictions


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="demand_forecasting/config/default.yaml",
        help="Path to YAML config file."
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output dir (default: cfg.experiment.output_dir/DeepAR)"
    )
    p.add_argument(
        "--generate_static",
        action="store_true",
        help="Recompute static features instead of loading from disk."
    )
    p.add_argument(
        "--demand_type",
        choices=["censored", "recovered"],
        default=None,
        help="Override config dataset.demand_type"
    )
    p.add_argument(
        "--recovered_path",
        default=None,
        help="Override config dataset.recovered_path"
    )
    return p.parse_args()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    args = parse_args()

    # ------------------ Load YAML config ------------------
    cfg = load_config(args.config)

    seed_everything(cfg.experiment.seed)

    # Output directory logic
    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Runtime dataset overrides
    demand_type = args.demand_type or cfg.dataset.demand_type
    recovered_path = args.recovered_path or cfg.dataset.recovered_path
    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq

    # ------------------ Holidays --------------------------
    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(
            start_year=cfg.holidays.start_year,
            end_year=cfg.holidays.end_year,
        )

    # ------------------ Static Features -------------------
    static_df = None
    static_cfg = cfg.static_features

    if static_cfg.enable:
        if args.generate_static or not Path(static_cfg.path).exists():
            print("[DeepAR] Generating static features from FRN...")
            # Only load training raw data for static feature gen
            ts_train = load_frn_for_forecasting(
                split="train",
                demand_type=demand_type,
                recovered_path=recovered_path,
                prediction_length=prediction_length,
                holiday_dates=holiday_dates,
                return_raw_df=True,  # <---- IMPORTANT PATCH
            )
            raw_df = ts_train  # already returned raw
            static_df = generate_static_features(raw_df, n_clusters=static_cfg.clusters)
            save_static_features(static_df, static_cfg.path)
        else:
            print("[DeepAR] Loading static features...")
            static_df = load_static_features(static_cfg.path)

    # ------------------ Load Time Series Data --------------
    print("[DeepAR] Loading FRN dataset...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    train, test = ts.train_test_split(prediction_length)

    # ------------------ DeepAR Hyperparameters -------------
    deepar_cfg = cfg.models.deepar.params
    hyperparameters = {
        "DeepAR": {
            "learning_rate": float(deepar_cfg.learning_rate),
            "dropout_rate": float(deepar_cfg.dropout_rate),
            "num_layers": int(deepar_cfg.num_layers),
            "num_cells": int(deepar_cfg.num_cells),
            "cell_type": deepar_cfg.cell_type,
            "use_gpu": True,
        }
    }

    # ------------------ Train Model -----------------------
    print("[DeepAR] Initializing TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        path=str(output_dir),
        eval_metric="MAPE",
    )

    print("[DeepAR] Training...")
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        static_features=static_df,
    )

    # ------------------ Evaluate ---------------------------
    print("[DeepAR] Evaluating...")
    results = evaluate_predictions(predictor, test)

    # Print global metrics
    print("\n========== DeepAR Global Metrics ==========")
    print(results["global"])

    # Save results
    (output_dir / "results").mkdir(exist_ok=True)

    per_series_out = output_dir / "results" / "deepar_per_series.csv"
    results["per_series"].to_csv(per_series_out, index=False)

    prob_out = output_dir / "results" / "deepar_probabilistic.json"
    pd.Series(results["probabilistic"]).to_json(prob_out)

    global_out = output_dir / "results" / "deepar_global_metrics.json"
    pd.Series(results["global"]).to_json(global_out)

    print(f"[DeepAR] Results saved to {output_dir/'results'}")
    print("==========================================================")


if __name__ == "__main__":
    main()
