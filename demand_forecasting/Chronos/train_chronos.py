# demand_forecasting/Chronos/train_chronos.py
# =============================================================
"""
Train Chronos (Chronos-Bolt) on FreshRetailNet with:
  - YAML config (config/default.yaml)
  - censored or recovered demand
  - full covariates (handled in frn_autogluon_data)
  - static features
  - India holiday calendar
  - extended evaluation (WAPE, WPE, MAPE, sMAPE, RMSE, MAE, quantiles)
"""

import argparse
from pathlib import Path

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


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="demand_forecasting/config/default.yaml",
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for output directory. If not set, uses experiment.output_dir/Chronos",
    )
    p.add_argument(
        "--generate_static",
        action="store_true",
        help="Regenerate static features instead of loading from disk.",
    )
    p.add_argument(
        "--demand_type",
        choices=["censored", "recovered"],
        default=None,
        help="Override dataset.demand_type from config if provided.",
    )
    p.add_argument(
        "--recovered_path",
        default=None,
        help="Override dataset.recovered_path from config if provided.",
    )
    return p.parse_args()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    args = parse_args()

    # ------------------ Load YAML config ------------------
    cfg = load_config(args.config)

    # Seed
    seed_everything(cfg.experiment.seed)

    # Output directory: CLI override > config.experiment.output_dir/Chronos
    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out) / "Chronos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset settings (with CLI overrides)
    demand_type = args.demand_type or cfg.dataset.demand_type
    recovered_path = args.recovered_path or cfg.dataset.recovered_path
    prediction_length = cfg.dataset.prediction_length
    freq = cfg.dataset.freq

    # ------------------ Holidays --------------------------
    holiday_dates = None
    if cfg.holidays.enable:
        # currently only India is supported in our helper
        holiday_dates = load_india_holidays(
            start_year=cfg.holidays.start_year,
            end_year=cfg.holidays.end_year,
        )

    # ------------------ Static features -------------------
    if cfg.static_features.enable:
        if args.generate_static:
            print("[Chronos] Generating static features from raw FRN...")
            raw_df = load_freshretailnet_raw("train")
            static_df = generate_static_features(
                raw_df,
                n_clusters=cfg.static_features.clusters,
            )
            save_static_features(static_df, cfg.static_features.path)
        else:
            print("[Chronos] Loading static features from disk...")
            static_df = load_static_features(cfg.static_features.path)
    else:
        static_df = None

    # ------------------ Load FRN data ---------------------
    print("[Chronos] Loading FRN dataset...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    train, test = ts.train_test_split(prediction_length)

    # ------------------ Chronos hyperparameters -----------
    chronos_cfg = cfg.models.chronos
    ft_cfg = chronos_cfg.fine_tune

    hyperparameters = {
        "Chronos": {
            "model_path": f"bolt_{chronos_cfg.size}",
            "fine_tune": bool(ft_cfg.enable),
            "fine_tune_steps": int(ft_cfg.steps),
            "fine_tune_lr": float(ft_cfg.lr),
        }
    }

    # ------------------ Train predictor -------------------
    print("[Chronos] Initializing AutoGluon TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric="WQL",  # kept consistent with your FRN config
        path=str(output_dir),
    )

    print("[Chronos] Fitting Chronos model...")
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        static_features=static_df,
    )

    # ------------------ Evaluate --------------------------
    print("[Chronos] Evaluating on holdout test set...")
    results = evaluate_predictions(predictor, test)

    print("\n==================== Chronos Results ====================")
    print(results["global"])

    # Save per-series metrics
    per_series_path = output_dir / "chronos_per_series.csv"
    results["per_series"].to_csv(per_series_path, index=False)
    print(f"[Chronos] Saved per-series metrics to: {per_series_path}")

    # Save probabilistic metrics (if any)
    prob_path = output_dir / "chronos_probabilistic.json"
    pd.Series(results["probabilistic"]).to_json(prob_path)
    print(f"[Chronos] Saved probabilistic metrics to: {prob_path}")

    # Save global metrics
    global_path = output_dir / "chronos_global_metrics.json"
    pd.Series(results["global"]).to_json(global_path)
    print(f"[Chronos] Saved global metrics to: {global_path}")

    print("==========================================================")


if __name__ == "__main__":
    main()
