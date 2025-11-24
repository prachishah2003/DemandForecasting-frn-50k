# demand_forecasting/common/holiday_utils.py
# =============================================================

import pandas as pd
import holidays


# ---------------------------------------------------------------
# Load Official India Holidays (python-holidays)
# ---------------------------------------------------------------
def load_india_holidays(start_year: int = 2017, end_year: int = 2026):
    """
    Returns a DatetimeIndex of all official India public holidays.
    Uses python-holidays which includes:
        - National holidays
        - Major public holidays
    """
    india = holidays.India(years=range(start_year, end_year + 1))
    dates = sorted(india.keys())
    return pd.to_datetime(dates)
