# demand_forecasting/common/frn_autogluon_config.py
# =============================================================
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FRNForecastConfig:
    """
    Master configuration class controlling all forecasting pipeline behavior.
    Supports:
      - prediction settings
      - evaluation settings
      - holiday integration
      - static features
      - covariate toggles
      - Chronos fine-tuning settings
      - reproducibility & logging
    """

    # -----------------------------
    # Core Forecast Settings
    # -----------------------------
    prediction_length: int = 7
    freq: str = "D"
    target: str = "demand"

    # Loss / metric used internally by models
    # Supported: "WQL", "MASE", "MAPE", "sMAPE", "RMSE"
    eval_metric: str = "WQL"

    # -----------------------------
    # Demand Type: censored OR recovered
    # -----------------------------
    demand_type: str = "censored"
    recovered_path: Optional[str] = None  # only if demand_type == "recovered"

    # -----------------------------
    # Holiday Calendar
    # -----------------------------
    enable_holidays: bool = True
    holiday_start_year: int = 2017
    holiday_end_year: int = 2026

    # -----------------------------
    # Static Features
    # -----------------------------
    enable_static_features: bool = True
    static_features_path: str = "demand_forecasting/static_features/static_features.parquet"
    static_clusters: int = 12

    # -----------------------------
    # Covariates: Enable/Disable
    # -----------------------------
    enable_lag_features: bool = True
    enable_demand_rolling: bool = True
    enable_price_features: bool = True
    enable_promo_features: bool = True
    enable_weather_features: bool = True
    enable_stock_features: bool = True
    enable_holiday_features: bool = True
    enable_imputation_features: bool = True

    # -----------------------------
    # Covariate Detailed Settings
    # -----------------------------
    demand_lags: List[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 28])
    promo_recency_cap: int = 999

    # -----------------------------
    # Stage-2 / Clustering Controls
    # -----------------------------
    enable_item_clustering: bool = True

    # -----------------------------
    # Chronos Fine-Tuning
    # -----------------------------
    chronos_size: str = "small"  # model_path: bolt_small, bolt_base, etc.
    chronos_fine_tune: bool = False
    chronos_fine_tune_steps: int = 1000
    chronos_fine_tune_lr: float = 1e-4

    # -----------------------------
    # Logging, Debug, Reproducibility
    # -----------------------------
    seed: int = 42
    verbosity: int = 2
