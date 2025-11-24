# demand_forecasting/experiments/run_all_stage1.py
# =============================================================
"""
Run all Stage-1 training jobs as specified in config/default.yaml.
This script will iterate over enabled models in cfg.models and run them sequentially.
It uses the same training pattern (single AutoGluon predictor per model) and saves metrics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor

from demand_forecasting.common.config_loader import load_config
from demand_forecasting.common.utils import seed_everything
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
    p.add_argument("--output_dir", default=None, help="Override base output dir")
    p.add_argument("--generate_static", action="store_true")
    p.add_argument("--demand_type", choices=["censored", "recovered"], default=None)
    p.add_argument("--recovered_path", default=None)
    return p.parse_args()


def build_hyperparams_for_model(cfg, model_name):
    """Map YAML model configs to AutoGluon hyperparameter dicts."""
    model_name_lower = model_name.lower()
    model_block = getattr(cfg.models, model_name_lower, None)
    if model_block is None:
        return None

    if model_name_lower == "chronos":
        chronos = model_block
        ft = chronos.fine_tune
        return {
            "Chronos": {
                "model_path": f"bolt_{chronos.size}",
                "fine_tune": bool(ft.enable),
                "fine_tune_steps": int(ft.steps),
                "fine_tune_lr": float(ft.lr),
            }
        }

    # If generic params container exists, convert it
    params = getattr(model_block, "params", None)
    if params is None:
        return {}

    # Convert attr namespace to dict of scalars for hyperparameters
    hp = {}
    # map model key names (AutoGluon expects the model-name key)
    hp_key = model_name
    for k, v in vars(params).items():
        hp[k] = v

    return {model_name: hp}


def parse_enabled_models(cfg):
    """Return list of (model_name, hyperparams) for enabled models in config."""
    enabled = []
    for model_name in vars(cfg.models).keys():
        model_block = getattr(cfg.models, model_name)
        # check enable flag
        if getattr(model_block, "enable", False):
            hyperparams = build_hyperparams_for_model(cfg, model_name.capitalize() if model_name.islower() else model_name)
            enabled.append((model_name.capitalize(), hyperparams))
    return enabled


def train_and_eval_single(model_name, hyperparameters, train, test, static_df, cfg, output_dir):
    print(f"[run_all_stage1] Training {model_name} ...")
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor(
        prediction_length=cfg.dataset.prediction_length,
        freq=cfg.dataset.freq,
        eval_metric=cfg.experiment.get("eval_metric", "WQL") if hasattr(cfg.experiment, "eval_metric") else "WQL",
        path=str(model_dir),
    )

    predictor.fit(train_data=train, hyperparameters=hyperparameters, static_features=static_df)

    results = evaluate_predictions(predictor, test)
    results["per_series"].to_csv(model_dir / f"{model_name.lower()}_per_series.csv", index=False)
    pd.Series(results["probabilistic"]).to_json(model_dir / f"{model_name.lower()}_probabilistic.json")
    pd.Series(results["global"]).to_json(model_dir / f"{model_name.lower()}_global_metrics.json")

    print(f"[run_all_stage1] Finished {model_name}: {results['global']}")
    return results["global"]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    base_out = args.output_dir or str(cfg.experiment.output_dir)
    output_dir = Path(base_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_type = args.demand_type or cfg.dataset.demand_type
    recovered_path = args.recovered_path or cfg.dataset.recovered_path
    prediction_length = cfg.dataset.prediction_length

    # holidays
    holiday_dates = None
    if cfg.holidays.enable:
        holiday_dates = load_india_holidays(cfg.holidays.start_year, cfg.holidays.end_year)

    # static features (global, reused by all models)
    if cfg.static_features.enable:
        if args.generate_static:
            print("[run_all_stage1] Generating static features...")
            raw_df = load_freshretailnet_raw("train")
            static_df = generate_static_features(raw_df, n_clusters=cfg.static_features.clusters)
            save_static_features(static_df, cfg.static_features.path)
        else:
            print("[run_all_stage1] Loading static features...")
            static_df = load_static_features(cfg.static_features.path)
    else:
        static_df = None

    # Load data once
    print("[run_all_stage1] Loading FRN dataset...")
    ts = load_frn_for_forecasting(
        split="train",
        demand_type=demand_type,
        recovered_path=recovered_path,
        prediction_length=prediction_length,
        holiday_dates=holiday_dates,
    )
    train, test = ts.train_test_split(prediction_length)

    # Determine enabled models from config
    enabled_models = []
    for mname in vars(cfg.models).keys():
        mblock = getattr(cfg.models, mname)
        if getattr(mblock, "enable", False):
            # model display name: capitalize (Chronos, PatchTST etc.)
            display_name = mname.capitalize()
            hyperparams = build_hyperparams_for_model(cfg, display_name)
            enabled_models.append((display_name, hyperparams))

    summary = {}
    for model_name, hyperparams in enabled_models:
        global_metrics = train_and_eval_single(model_name, hyperparams, train, test, static_df, cfg, output_dir)
        summary[model_name] = global_metrics

    # Save consolidated summary
    with open(output_dir / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[run_all_stage1] Summary saved to", output_dir / "stage1_summary.json")


if __name__ == "__main__":
    main()
