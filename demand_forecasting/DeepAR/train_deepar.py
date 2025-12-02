# =============================================================
# Train DeepAR Stage-1 on FRN (static features enabled correctly)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="demand_forecasting/config/default.yaml")
    return p.parse_args()


# Expand 24-length vectors → 24 columns
def expand_array_column(df, col, prefix, size=24):
    if col not in df.columns:
        return df
    def fix(x):
        arr = np.array(x).flatten() if x is not None else np.zeros(size)
        if len(arr) < size:
            arr = np.concatenate([arr, np.zeros(size - len(arr))])
        return arr[:size]

    mat = np.vstack(df[col].apply(fix))
    new_cols = [f"{prefix}{i}" for i in range(size)]
    df = df.drop(columns=[col])
    return pd.concat([df, pd.DataFrame(mat, columns=new_cols, index=df.index)], axis=1)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.experiment.seed)
    max_items = cfg.dataset.max_items
    pred_len = cfg.dataset.prediction_length
    freq = cfg.dataset.freq

    output_dir = Path(cfg.experiment.output_dir) / "DeepAR"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("demand_forecasting/data/train/frn_train.parquet")
    static_path = Path(cfg.static_features.path)

    print("📥 Loading training parquet...")
    df = pd.read_parquet(train_path)

    if "item_id" not in df.columns:
        df["item_id"] = (
            df["city_id"].astype(str) + "_" +
            df["store_id"].astype(str) + "_" +
            df["product_id"].astype(str)
        )

    df["dt"] = pd.to_datetime(df["dt"])

    # limit items
    keep_items = df["item_id"].value_counts().index[:max_items]
    df = df[df["item_id"].isin(keep_items)]
    print("Training on", len(keep_items), "item_ids")

    # expand sales + stock hour vectors
    df = expand_array_column(df, "hours_sale", "h_sale_")
    df = expand_array_column(df, "hours_stock_status", "h_stock_")

    # Convert target
    df = df.rename(columns={"sale_amount": "target"})

    print("📌 Loading Static Features...")
    static_df = load_static_features(static_path)
    static_df = static_df[static_df["item_id"].isin(keep_items)].copy()

    # drop vector features not allowed in static_fw
    drop_cols = ["hours_sale", "hours_stock_status"]
    static_df = static_df.drop(columns=[c for c in drop_cols if c in static_df.columns])

    # one static row per item
    static_df = static_df.drop_duplicates("item_id").set_index("item_id")
    print("Static shape (unique):", static_df.shape)

    # build TimeSeriesDataFrame
    df = df.sort_values(["item_id", "dt"])
    ts = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="dt",
    )

    ts.static_features = static_df  # << true static injection

    # split train/test
    train_ts = ts.slice_by_timestep(slice(None, -pred_len))
    test_ts = ts.slice_by_timestep(slice(-pred_len, None))

    print("TS Train:", train_ts.shape, "TS Test:", test_ts.shape)

    print("🚀 Fitting DeepAR...")
    predictor = TimeSeriesPredictor(
        prediction_length=pred_len,
        freq=freq,
        path=str(output_dir),
        target="target",
        eval_metric="WQL",
        verbosity=2,
    )

    hp = {
        "DeepAR": {
            "cell_type":     cfg.models.deepar.params.cell_type,
            "hidden_size":   cfg.models.deepar.params.hidden_size,
            "num_layers":    cfg.models.deepar.params.num_layers,
            "dropout_rate":  cfg.models.deepar.params.dropout_rate,
        }
    }

    predictor.fit(train_ts, hyperparameters=hp)

    print("📊 Running evaluation...")
    results = evaluate_predictions(predictor, test_ts)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    results["per_series"].to_csv(metrics_dir / "per_series.csv", index=False)
    pd.Series(results["global"]).to_json(metrics_dir / "global.json")
    pd.Series(results["probabilistic"]).to_json(metrics_dir / "probabilistic.json")

    print("\n🎯 Global metrics:", results["global"])
    print("✔ DONE — metrics saved to", metrics_dir)


if __name__ == "__main__":
    main()
