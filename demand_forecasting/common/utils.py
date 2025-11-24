# demand_forecasting/common/utils.py
# =============================================================

import numpy as np
import pandas as pd
import random
import os


# ---------------------------------------------------------------
# Seeding for reproducibility
# ---------------------------------------------------------------
def seed_everything(seed: int = 42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------
# Safe log
# ---------------------------------------------------------------
def safe_log(x, eps=1e-8):
    """Numerically stable log."""
    return np.log(np.maximum(x, eps))


# ---------------------------------------------------------------
# Ensure Datetime Conversion
# ---------------------------------------------------------------
def to_datetime(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------
# Memory Optimization (optional)
# ---------------------------------------------------------------
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerical columns to save memory."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df
