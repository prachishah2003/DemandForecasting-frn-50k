import os
import pandas as pd
import numpy as np
from datasets import load_dataset

def mimic_missing(patch_ts, p=0.5, max_missing_patch=7, min_missing_patch=3):
    patch_len = patch_ts.shape[-1]
    patch_num = patch_ts.shape[1]
    batch_size = patch_ts.shape[0]
    patch_time = np.arange(patch_len)[None,None,:] # patch_time which could broadcast
    ## missing mechanism
    patch_missing_cnt = np.isnan(patch_ts).sum(axis=-1, keepdims=True)
    non_missing_idx = patch_missing_cnt==0
    # continuous-patch missing: conti_idx 
    non_missing_cumsum = np.zeros_like(non_missing_idx.astype(int))
    sum_vec = np.zeros_like(non_missing_idx[:,0].astype(int))
    conti_idx = np.zeros_like(non_missing_idx) # (batch_size, day_len, 1)
    for i in range(patch_num):
        sum_vec = np.where(non_missing_idx[:,i], sum_vec, 0)
        sum_vec = sum_vec + non_missing_idx[:,i].astype(int)
        non_missing_cumsum[:,i] = sum_vec.copy()
        if i>max_missing_patch:
            conti_len = np.random.randint(low=min_missing_patch, high=max_missing_patch+1, size=sum_vec.shape)
            conti_len = np.where((conti_len < sum_vec) & (np.random.rand(*sum_vec.shape)<p/10), conti_len, 0)
            conti_tmp = np.arange(batch_size * max_missing_patch).reshape(batch_size, max_missing_patch, 1)
            conti_tmp = max_missing_patch - (conti_tmp - conti_tmp[:,0:1])
            conti_tmp = (conti_tmp <= conti_len[:,None]) & (conti_tmp > 1)
            conti_idx[:,i-max_missing_patch+1:i+1] = conti_idx[:,i-max_missing_patch+1:i+1] | conti_tmp
            
    # intra-patch missing: intra_idx
    intra_rand_idx = (np.random.rand(*patch_missing_cnt.shape) < p) & non_missing_idx  
    patch_missing_start_time = np.random.randint(low=0, high=int(patch_len*(1-p)), size=patch_missing_cnt.shape)
    patch_missing_end_time = patch_len - patch_missing_start_time
    intra_missing_idx_front = patch_time>=patch_missing_start_time
    intra_missing_idx_backend = patch_time<=patch_missing_end_time
    shape = list(intra_missing_idx_front.shape)
    shape[-1] = 1
    intra_missing_idx = np.where(np.random.rand(*shape)<=0.5, intra_missing_idx_front, intra_missing_idx_backend)
    intra_idx = intra_rand_idx & intra_missing_idx # (batch_size, patch_num, patch_len)


    # valid_idx = np.squeeze(intra_rand_idx | conti_idx)
    sample_idx = intra_idx | conti_idx

    patch_ts_missing = np.where(sample_idx, np.nan, patch_ts)
    valid_idx = ~np.isnan(patch_ts)&np.isnan(patch_ts_missing)
    return patch_ts_missing, valid_idx
        
        
def load_data(CONFIG):
    """
    CONFIG should be a dict-like object containing at least:
      - 'missing_rate' : float in (0,1)
      - optional 'max_items' : int (defaults to 5000)
    """
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    data = dataset['train'].to_pandas()

    # --- reduce to top-N series to save RAM ---
    # create a series identifier — uses store_id + product_id (matches how you sort below)
    if 'item_id' not in data.columns:
        data['item_id'] = data['store_id'].astype(str) + "_" + data['product_id'].astype(str)

    max_items = int(5000)
    top_items = data['item_id'].value_counts().head(max_items).index.tolist()
    data = data[data['item_id'].isin(top_items)].reset_index(drop=True)

    # sort (keeps same behavior as before)
    data = data.sort_values(by=['store_id', 'product_id', 'dt']).reset_index(drop=True)

    # horizon and series count (after reduction)
    horizon = 90
    series_num = data.shape[0] // horizon
    if series_num * horizon != data.shape[0]:
        # warn but continue: if not exact multiple, trim the tail to create full series blocks
        trim = data.shape[0] - series_num * horizon
        if trim > 0:
            data = data.iloc[:-trim].reset_index(drop=True)
            series_num = data.shape[0] // horizon

    # --- convert list-like hourly columns to arrays ---
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_stock_status = np.array(data['hours_stock_status'].tolist())

    # previous code reshaped into (series_num*3, 30, 24) and then took hours 6:22
    # keep same logic but recompute carefully using the available rows
    try:
        hours_sale_origin = hours_sale.reshape(series_num*3, 30, 24)[...,6:22]
        hours_stock_status = hours_stock_status.reshape(series_num*3, 30, 24)[...,6:22]
    except Exception as e:
        # fallback: attempt to infer patching based on array length
        total_rows = hours_sale.shape[0]
        # try to compute group factor (default 3 if divisible)
        factor = 3
        if total_rows % (horizon) == 0:
            factor = total_rows // series_num // (horizon // 30) if series_num>0 else 3
        # ensure reshape is possible; otherwise raise with helpful message
        try:
            hours_sale_origin = hours_sale.reshape(series_num*factor, 30, 24)[...,6:22]
            hours_stock_status = hours_stock_status.reshape(series_num*factor, 30, 24)[...,6:22]
        except Exception:
            raise RuntimeError(f"Reshape failed after reduction. data rows={data.shape[0]}, series_num={series_num}, "
                               f"hours_sale.shape={hours_sale.shape}. You may need to examine the original data layout.") from e

    # Mask out sales where stock_status == 1 (out of stock)
    hours_sale = np.where(hours_stock_status==1, np.nan, hours_sale_origin)

    covariate = data[['discount', 'holiday_flag', 'precpt', 'avg_temperature']].values
    # reshape covariates similarly (rows should match hours_sale rows before reshape)
    covariate = covariate.reshape(series_num*3, 30, 4)
    covariate = covariate / (covariate.max(axis=1, keepdims=True) + 0.1)

    # apply missing mechanism on reduced hours_sale
    hours_sale, valid_idx = mimic_missing(hours_sale, p=CONFIG['missing_rate'], max_missing_patch=7, min_missing_patch=3)

    # generate train dataset (same final shape logic as before)
    data_combined = np.concatenate(
        [hours_sale[...,None], np.broadcast_to(covariate[:,:,None,:], hours_sale.shape + (4,))],
        axis=-1
    )
    data_combined = np.concatenate(
        [data_combined, np.broadcast_to(np.arange(16)[None,None,:,None]/15, hours_sale[...,None].shape)],
        axis=-1
    )

    # reshape to (-1, 30*16, 6)
    data_combined = data_combined.reshape(-1, 30*16, 6)
    valid_idx = valid_idx.reshape(-1, 30*16, 1)
    print("Total valid imputed points:", int(valid_idx.sum()))
    hours_sale_origin = hours_sale_origin.reshape(-1, 30*16, 1)

    train_set = {'X': data_combined}
    output = {
        'train_set': train_set,
        'ts_origin': hours_sale_origin,
        'valid_idx': valid_idx,
        'params': {
            'n_steps': 30*16,
            'n_features': 6,
            'patch_len': 16,
            'OT': 1
        }
    }
    return output
