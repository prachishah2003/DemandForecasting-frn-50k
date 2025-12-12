# patched app.py (overwrite your existing file in latent_demand_recovery/exp)
import os, subprocess, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torch import nn
import pypots
from data import load_data
from model import load_model
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
from datasets import load_dataset
import argparse
from pathlib import Path

def set_seed(seed_value=1024):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    return


def _imputation(CONFIG):
    # load data (this is your existing function which returns train_set, ts_origin, valid_idx, params etc)
    data = load_data(CONFIG)
    (
        train_set,
        ts_origin,
        valid_idx,
    ) = (
        data['train_set'],
        data['ts_origin'],
        data['valid_idx'],
    )
    # update specific dataset params config
    CONFIG.update(data['params'])
    model = load_model(CONFIG)
    model.fit(train_set)
    results = model.predict(train_set)
    if len(results['imputation'].shape) == 4:
        imputation = results['imputation'].mean(axis=1)[:, :, : CONFIG['OT']]
    else:
        imputation = results['imputation'][:, :, : CONFIG['OT']]
    imputation = np.where(imputation > 0, imputation, 0)
    model_name = CONFIG['model']
    missing_rate = CONFIG['missing_rate']
    if not os.path.exists('./demand'):
        os.makedirs('./demand', exist_ok=True)
    np.save(f'./demand/{model_name}_imputation_{missing_rate}.npy', imputation)
    if CONFIG['missing_rate'] > 0:
        evaluation_mnar(train_set['X'], imputation)
    return imputation


def _load_frn_dataframe_prefer_local():
    """
    Prefer a local reduced parquet if it exists; otherwise load from HF.
    This keeps all downstream shapes consistent with the dataset you're actually experimenting with.
    """
    local_path = Path("demand_forecasting/data/train/frn_train.parquet")
    if local_path.exists():
        df = pd.read_parquet(local_path)
        print(f"[DATA] Loaded reduced parquet from {local_path} shape={df.shape}")
    else:
        print("[DATA] Local reduced parquet not found — loading full dataset from HuggingFace (this may be large).")
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        df = dataset['train'].to_pandas()
        print(f"[DATA] Loaded HF dataset shape={df.shape}")
    # keep original ordering expectation
    df = df.sort_values(by=['store_id', 'product_id', 'dt']).reset_index(drop=True)
    return df


def _demand_recovery(imputation):
    # Load FRN dataframe (prefer local reduced parquet)
    data = _load_frn_dataframe_prefer_local()
    horizon = 90

    # hours_sale raw array (list-of-lists -> ndarray)
    hours_sale = np.array(data['hours_sale'].tolist())

    # Compute expected target shape from the data itself rather than hard-coded factors.
    # Original code used reshape(series_num*3, 30, 24) — but that relies on a fixed multiplier.
    # We'll compute hours_sale_origin by inferring from hours_sale length and known horizon/windowing.
    # hours_sale shape: (n_rows, 24)
    n_rows = hours_sale.shape[0]

    # We expect data to be organized in blocks of 'horizon' days per series.
    # Determine number of series rows (rows that form a full horizon)
    # If n_rows isn't divisible by horizon, floor it to nearest full group to be safe.
    n_full_horizon_groups = n_rows // horizon
    if n_full_horizon_groups == 0:
        raise RuntimeError(f"Not enough rows ({n_rows}) to form even one horizon ({horizon})")

    # The original code used reshape(series_num * 3, 30, 24).
    # To be robust, we'll attempt to infer `rows_per_series`:
    # - If the hours_sale total rows divides evenly into (30*24) blocks, prefer that
    # - Otherwise, attempt to emulate original approach by deriving rows_per_series = n_full_horizon_groups * factor guess
    # Simpler approach: compute hours_sale_origin by reshaping into (-1, 30, 24) if possible.
    if n_rows % (30 * 24) == 0:
        # can reshape into (num_series, 30, 24)
        try:
            hours_sale_origin = hours_sale.reshape(-1, 30, 24)
        except Exception:
            hours_sale_origin = hours_sale.reshape(n_rows // 30, 30, 24)
    else:
        # fallback: try reshape into (-1, 30, 24) on first floor(n_rows / 30)*30 rows
        use_rows = (n_rows // 30) * 30
        hours_sale_origin = hours_sale[:use_rows].reshape(-1, 30, 24)
        print(f"[WARN] hours_sale rows ({n_rows}) not divisible by 30*24. Using first {use_rows} rows for reshape.")

    # Now prepare the imputation assignment target slice (6:22)
    # imputation is expected to be shaped (n_imputed_rows, 30, 16)
    imputed = np.asarray(imputation).reshape(-1, 30, 16)
    target_rows = hours_sale_origin.shape[0]

    # Robust assignment: direct, repeat/truncate, or trim
    if imputed.shape[0] == target_rows:
        hours_sale_origin[..., 6:22] = imputed
    elif imputed.shape[0] < target_rows:
        factor = int(np.ceil(target_rows / imputed.shape[0]))
        imputed_large = np.repeat(imputed, factor, axis=0)[:target_rows]
        hours_sale_origin[..., 6:22] = imputed_large
        print(f"[WARN] imputed rows ({imputed.shape[0]}) < target rows ({target_rows}), repeated x{factor} then truncated.")
    else:
        # imputed larger than expected — trim
        hours_sale_origin[..., 6:22] = imputed[:target_rows]
        print(f"[WARN] imputed rows ({imputed.shape[0]}) > target rows ({target_rows}), trimmed to match target.")

    # Sum across hours to produce sale_amount_pred — same logic as before
    sale_amount_pred = hours_sale_origin.sum(axis=-1).reshape(-1, horizon)
    data[f'sale_amount_pred'] = sale_amount_pred.reshape(-1)
    if not os.path.exists('./demand'):
        os.makedirs('./demand', exist_ok=True)
    data.to_parquet(f'./demand/demand.parquet')
    print("[INFO] Demand parquet written to ./demand/demand.parquet")
    return data


def demand_recovery(CONFIG):
    imputation = _imputation(CONFIG)
    demand_df = None
    # If missing_rate == 0 we assume this is a real recovery run (not MNAR eval)
    if CONFIG['missing_rate'] == 0:
        demand_df = _demand_recovery(imputation)
        evaluation_decoupling(demand_df)
    return demand_df


def evaluation_mnar(X, imputation):
    # Prefer loading the same DF to keep shapes aligned
    data = _load_frn_dataframe_prefer_local()
    horizon = 90
    n_rows = data.shape[0]

    hours_sale = np.array(data['hours_sale'].tolist())
    hours_stock_status = np.array(data['hours_stock_status'].tolist())

    # Build hours_sale_origin similar to _demand_recovery logic
    if n_rows % (30 * 24) == 0:
        hours_sale_origin = hours_sale.reshape(-1, 30, 24)
    else:
        use_rows = (n_rows // 30) * 30
        hours_sale_origin = hours_sale[:use_rows].reshape(-1, 30, 24)
        print(f"[WARN] hours_sale rows ({n_rows}) not divisible by 30*24. Using first {use_rows} rows for reshape.")

    # robust imputed shape
    imputed = np.asarray(imputation).reshape(-1, 30, 16)
    target_rows = hours_sale_origin.shape[0]

    if imputed.shape[0] == target_rows:
        hours_sale_impute = hours_sale_origin.copy()
        hours_sale_impute[..., 6:22] = imputed
    elif imputed.shape[0] < target_rows:
        factor = int(np.ceil(target_rows / imputed.shape[0]))
        imputed_large = np.repeat(imputed, factor, axis=0)[:target_rows]
        hours_sale_impute = hours_sale_origin.copy()
        hours_sale_impute[..., 6:22] = imputed_large
        print(f"[WARN] (mnar) imputed rows ({imputed.shape[0]}) < target rows ({target_rows}), repeated x{factor} then truncated.")
    else:
        hours_sale_impute = hours_sale_origin.copy()
        hours_sale_impute[..., 6:22] = imputed[:target_rows]
        print(f"[WARN] (mnar) imputed rows ({imputed.shape[0]}) > target rows ({target_rows}), trimmed to match target.")

    # compute evaluation stats (as you had them)
    sale_amount_pred = hours_sale_impute.sum(axis=-1).reshape(-1, 90).reshape(-1)
    stock_hour_X = np.isnan(X[..., 0].reshape(-1, 30, 16)).sum(axis=-1).reshape(-1, 90).reshape(-1)
    stock_hour_origin = hours_stock_status[:, 6:22].sum(axis=1)
    valid_idx = (stock_hour_X > 0) & (stock_hour_origin == 0)
    sale_amount = data['sale_amount'].values
    print('wape', (np.abs(sale_amount_pred - sale_amount) * valid_idx).sum() / (sale_amount * valid_idx).sum())
    print('wpe', ((sale_amount_pred - sale_amount) * valid_idx).sum() / (sale_amount * valid_idx).sum())


def evaluation_decoupling(data):
    df = data[['city_id', 'store_id', 'product_id', 'dt', 'holiday_flag', 'discount', 'sale_amount', 'sale_amount_pred','stock_hour6_22_cnt']].copy()
    mu = df.query('stock_hour6_22_cnt==0').groupby(['store_id', 'product_id'])['sale_amount'].mean()
    mu = mu.reset_index().rename(columns={'sale_amount':'mu'})
    corr = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).apply(lambda subdf:subdf[['stock_hour6_22_cnt', 'sale_amount', 'sale_amount_pred']].corr().iloc[:1, 1:])
    stock_nunique = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).agg({'stock_hour6_22_cnt':'nunique'}).reset_index()
    stock_nunique = stock_nunique.rename(columns={'stock_hour6_22_cnt':'nunique'})
    corr = corr.reset_index().merge(mu, on=['store_id', 'product_id']).merge(stock_nunique.query('nunique>3'), on=['store_id', 'product_id', 'holiday_flag'])
    metric = pd.DataFrame({
            'method':['sale_amount', 'sale_amount_pred'],
            'decoupling score':np.nansum(corr[['sale_amount', 'sale_amount_pred']].values * corr[['mu']].values, axis=0)/corr['mu'].sum()
             })
    print(metric)


# default params config
CONFIG = {
    'model': 'DLinear',
    'saving_path': './save',
    'EPOCHS': 5,
    'batch_size': 128,
    'patience': 5,
    'n_layers': 2,
    'd_model': 64,
    'd_ffn': 32,
    'n_heads': 4,
    'd_k': 16,
    'd_v': 16,
    'dropout': 0.,
    'attn_dropout': 0.,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'OT': 1,
    'missing_rate':0.3,
    'n_patches':7,
    'alpha': 1e-2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default='TimesNet',
        help="Demand Recovery Model, default='TimesNet'"
    )
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.,
        help="Missing Rate for Artificial MNAR Evaluation, default missing_rate = 0 for latent demand recovery"
    )
    args = parser.parse_args()
    print(args)
    CONFIG['model'] = args.model
    CONFIG['missing_rate'] = args.missing_rate
    ## set random seed
    set_seed(seed_value=1024)
    ## latent demand recovery
    demand_df = demand_recovery(CONFIG)
