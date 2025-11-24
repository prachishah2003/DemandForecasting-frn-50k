# demand_forecasting/ETS/train_ets.py
# =============================================================
"""
Train ETS using YAML config (GPU-aware version).
"""

import argparse
from pathlib import Path
import json
import torch     # <-- GPU check added

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from demand_forecasting.common.utils import seed_everything
from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.frn_autogluon_data import (
    load_frn_for_forecasting,
    load_freshretailnet_raw,
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
    p.add_argument("--config", default="demand_forecasting/config/default.yaml")
    p.add_argument("--output_dir", default=None, help="Override config experiment.output_dir")
    p.add_argument("--generate_static", action="store_true")
    p.add_argument("--demand_type", choices=["censored", "recovered"], default=None)
    p.add_argument("--recovered_path", default=None)
    return p.parse_args()


def main():
    # ---------------------------------------------
    # GPU CHECK
    # ---------------------------------------------
    print("\n================ GPU CHECK ================")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
    else:
        print("⚠ WARNING: GPU NOT AVAILABLE — ETS will run on CPU")
    print("===========================================\n")

    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out) / "ETS"
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_type = args.demand_type or cfg.dataset.demand_type
    recovered_path = args.recovered_path or cfg.dataset.recovered_path
    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq

    # holidays
    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(cfg.holidays.start_year, cfg.holidays.end_year)

    # static features
    if cfg.static_features.enable:
        if args.generate_static:
            print("[ETS] Generating static features...")
            raw_df = load_freshretailnet_raw("train")
            static_df = generate_static_features(raw_df, n_clusters=cfg.static_features.clusters)
            save_static_features(static_df, cfg.static_features.path)
        else:
            print("[ETS] Loading static features...")
            static_df = load_static_features(cfg.static_features.path)
    else:
        static_df = None

    # load data
    print("[ETS] Loading FRN dataset...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    train, test = ts.train_test_split(prediction_length)

    # hyperparams from YAML (fallback sensible defaults)
    ets_cfg = cfg.models.ets.params
    hyperparameters = {
        "ETS": {
            "seasonal_period": int(getattr(ets_cfg, "seasonal_period", 7)),
            "trend": getattr(ets_cfg, "trend", "add"),
            "seasonal": getattr(ets_cfg, "seasonal", "add"),
            "damped_trend": bool(getattr(ets_cfg, "damped_trend", True)),
        }
    }

    # ---------------------------------------------
    # GPU-AWARE PREDICTOR INITIALIZATION
    # ---------------------------------------------
    print("[ETS] Initializing Predictor...")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric=cfg.experiment.get("eval_metric", "WQL") if hasattr(cfg.experiment, "eval_metric") else "WQL",
        path=str(output_dir),
        # This does NOT accelerate ETS but ensures future models use GPU
        enable_ctx="gpu" if torch.cuda.is_available() else "cpu",
    )

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------
    print("[ETS] Training ETS...")
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        static_features=static_df,
    )

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------
    print("[ETS] Evaluating...")
    results = evaluate_predictions(predictor, test)
    print(results["global"])

    # save outputs
    results["per_series"].to_csv(output_dir / "ets_per_series.csv", index=False)
    pd.Series(results["probabilistic"]).to_json(output_dir / "ets_probabilistic.json")
    pd.Series(results["global"]).to_json(output_dir / "ets_global_metrics.json")

    print(f"[ETS] Saved results to {output_dir}")
    print("🎉 ETS completed (GPU-aware mode)\n")


if __name__ == "__main__":
    main()
