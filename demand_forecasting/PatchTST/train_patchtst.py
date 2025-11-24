# demand_forecasting/PatchTST/train_patchtst.py
# =============================================================
"""
Train PatchTST on FreshRetailNet with:
  - YAML config (config/default.yaml)
  - censored or recovered demand
  - full covariates (auto-applied)
  - static features
  - India holiday calendar
  - extended evaluation framework
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
        help="Path to YAML config file."
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory (otherwise config.experiment.output_dir/PatchTST)"
    )
    p.add_argument(
        "--generate_static",
        action="store_true",
        help="Regenerate static features instead of loading"
    )
    p.add_argument(
        "--demand_type",
        choices=["censored", "recovered"],
        default=None,
        help="Override dataset.demand_type from config"
    )
    p.add_argument(
        "--recovered_path",
        default=None,
        help="Override dataset.recovered_path from config"
    )
    return p.parse_args()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    args = parse_args()

    # Load YAML config as namespace
    cfg = load_config(args.config)

    seed_everything(cfg.experiment.seed)

    # Output directory
    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out) / "PatchTST"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configs
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
    if cfg.static_features.enable:
        if args.generate_static:
            print("[PatchTST] Generating static features from raw FRN...")
            raw_df = load_freshretailnet_raw("train")
            static_df = generate_static_features(
                raw_df,
                n_clusters=cfg.static_features.clusters,
            )
            save_static_features(static_df, cfg.static_features.path)
        else:
            print("[PatchTST] Loading static features from disk...")
            static_df = load_static_features(cfg.static_features.path)
    else:
        static_df = None

    # ------------------ Load FRN dataset ------------------
    print("[PatchTST] Loading FRN dataset...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    train, test = ts.train_test_split(prediction_length)

    # ------------------ Hyperparameters -------------------
    pt_cfg = cfg.models.patchtst.params

    hyperparameters = {
        "PatchTST": {
            "learning_rate": float(pt_cfg.learning_rate),
            "d_model": int(pt_cfg.d_model),
            "n_heads": int(pt_cfg.n_heads),
            "patch_len": int(pt_cfg.patch_len),
            "stride": int(pt_cfg.stride),
            "dropout": float(pt_cfg.dropout),
            "batch_size": int(pt_cfg.batch_size),
        }
    }

    # ------------------ Train PatchTST --------------------
    print("[PatchTST] Initializing Predictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric="WQL",
        path=str(output_dir),
    )

    print("[PatchTST] Training PatchTST model...")
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        static_features=static_df,
    )

    # ------------------ Evaluate --------------------------
    print("[PatchTST] Evaluating model...")
    results = evaluate_predictions(predictor, test)

    print("\n==================== PatchTST RESULTS ====================")
    print(results["global"])

    # Save per-series metrics
    per_series_path = output_dir / "patchtst_per_series.csv"
    results["per_series"].to_csv(per_series_path, index=False)
    print(f"[PatchTST] Saved per-series metrics to {per_series_path}")

    # Save probabilistic metrics
    prob_path = output_dir / "patchtst_probabilistic.json"
    pd.Series(results["probabilistic"]).to_json(prob_path)
    print(f"[PatchTST] Saved probabilistic metrics to {prob_path}")

    # Save global metrics
    global_path = output_dir / "patchtst_global_metrics.json"
    pd.Series(results["global"]).to_json(global_path)
    print(f"[PatchTST] Saved global metrics to {global_path}")

    print("==========================================================")


if __name__ == "__main__":
    main()
