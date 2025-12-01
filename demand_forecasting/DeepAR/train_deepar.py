# demand_forecasting/DeepAR/train_deepar.py
# =============================================================
"""
Train DeepAR on FreshRetailNet-50K with Stage-1 features.

Features in this script:
- YAML config (config/default.yaml)
- censored or recovered demand (via demand_type)
- India holiday calendar (optional)
- Static features generated from:
    * sale_amount time series stats
    * hours_sale (sequence → scalar aggregates)
    * hours_stock_status (sequence → scalar aggregates)
- AutoGluon 1.4-compatible DeepAR hyperparameters
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
        help="Override experiment.output_dir (DeepAR subfolder is added automatically).",
    )
    p.add_argument(
        "--generate_static",
        action="store_true",
        help="Regenerate static features from raw FRN and overwrite the parquet file.",
    )
    p.add_argument(
        "--demand_type",
        choices=["censored", "recovered"],
        default=None,
        help="Override dataset.demand_type from config.",
    )
    p.add_argument(
        "--recovered_path",
        default=None,
        help="Override dataset.recovered_path from config when using recovered demand.",
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

    # Output directory: CLI override > config default
    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    demand_type = args.demand_type or cfg.dataset.demand_type
    recovered_path = args.recovered_path or cfg.dataset.recovered_path
    prediction_length = int(cfg.dataset.prediction_length)
    freq = cfg.dataset.freq

    # Optional: limit number of series for Colab / memory
    # (non-breaking: if not present in YAML, getattr fallback is None)
    max_series = getattr(cfg.dataset, "max_series_stage1", None)

    # ------------------ Holidays --------------------------
    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(
            start_year=int(cfg.holidays.start_year),
            end_year=int(cfg.holidays.end_year),
        )

    # ------------------ Static Features -------------------
    if cfg.static_features.enable:
        if args.generate_static:
            print("[DeepAR] Generating static features from raw FRN...")
            # This reads the original FRN dataset with hours_sale, hours_stock_status, etc.
            raw_df = load_freshretailnet_raw(split="train")
            static_df = generate_static_features(
                raw_df, n_clusters=int(cfg.static_features.clusters)
            )
            save_static_features(static_df, cfg.static_features.path)
        else:
            print("[DeepAR] Loading static features from disk...")
            static_df = load_static_features(cfg.static_features.path)
    else:
        print("[DeepAR] Static features disabled via config.")
        static_df = None

    # ------------------ Load FRN TimeSeries ------------------
    print("[DeepAR] Loading FRN dataset as TimeSeriesDataFrame...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )

    # Optional: subselect first N series for memory reasons (only if configured)
    if max_series is not None and max_series > 0:
        print(f"[DeepAR] Restricting to first {max_series} item_ids for Stage-1.")
        all_ids = ts.index.get_level_values("item_id").unique()
        keep_ids = all_ids[:max_series]
        ts = ts.loc[keep_ids]

    # Train / test split
    train, test = ts.train_test_split(prediction_length)

    # ------------------ Hyperparameters -------------------
    deepar_cfg = cfg.models.deepar.params

    # Map YAML fields → AutoGluon DeepAR hyperparameters (AutoGluon 1.4)
    # Avoid conflicting aliases like (epochs vs max_epochs) or (learning_rate vs lr).
    lr = float(
        getattr(
            deepar_cfg,
            "lr",
            getattr(deepar_cfg, "learning_rate", 1e-3),
        )
    )
    max_epochs = int(getattr(deepar_cfg, "max_epochs", 10))
    num_layers = int(getattr(deepar_cfg, "num_layers", 2))
    hidden_size = int(
        getattr(
            deepar_cfg,
            "hidden_size",
            getattr(deepar_cfg, "num_cells", 40),
        )
    )
    dropout_rate = float(getattr(deepar_cfg, "dropout_rate", 0.1))

    hyperparameters = {
        "DeepAR": {
            "lr": lr,
            "max_epochs": max_epochs,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "use_gpu": True,
        }
    }

    # ------------------ Train model -----------------------
    eval_metric = getattr(cfg.experiment, "eval_metric", "MAPE")

    print("[DeepAR] Initializing TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric=eval_metric,
        path=str(output_dir),
    )

    # NOTE: AutoGluon 1.4's TimeSeriesPredictor.fit()
    # DOES NOT accept 'static_features' directly, so we don't pass it here.
    # Static features are generated & saved for possible later use / other models.
    print("[DeepAR] Training DeepAR (Stage-1)...")
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
    )

    # ------------------ Evaluate ---------------------------
    print("[DeepAR] Evaluating on holdout set...")
    results = evaluate_predictions(predictor, test)

    # 'results' is expected to be a dict with:
    #   - results["global"]          : global scalar metrics
    #   - results["per_series"]      : DataFrame per item_id
    #   - results["probabilistic"]   : quantile-based metrics
    print("\n==================== DeepAR Results ====================")
    global_metrics = results.get("global", {})
    print(global_metrics)

    # Save per-series metrics
    per_series = results.get("per_series", None)
    if per_series is not None:
        per_series_path = output_dir / "deepar_per_series.csv"
        per_series.to_csv(per_series_path, index=False)
        print(f"[DeepAR] Saved per-series metrics to: {per_series_path}")

    # Save probabilistic metrics
    probabilistic = results.get("probabilistic", None)
    if probabilistic is not None:
        prob_path = output_dir / "deepar_probabilistic.json"
        pd.Series(probabilistic).to_json(prob_path)
        print(f"[DeepAR] Saved probabilistic metrics to: {prob_path}")

    # Save global metrics
    if global_metrics:
        global_path = output_dir / "deepar_global_metrics.json"
        pd.Series(global_metrics).to_json(global_path)
        print(f"[DeepAR] Saved global metrics to: {global_path}")

    print("==========================================================")


if __name__ == "__main__":
    main()
