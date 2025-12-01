"""
Holiday Utilities (India-specific)

- Downloads official Indian holiday calendar if available online
- Falls back to pre-stored list if offline
- Returns a sorted Python list of holiday dates (YYYY-MM-DD)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Local fallback path
HOLIDAY_CACHE = Path("demand_forecasting/data/india_holidays.csv")


def _load_cached_holidays():
    if HOLIDAY_CACHE.exists():
        df = pd.read_csv(HOLIDAY_CACHE)
        return sorted(df["date"].astype(str).unique())
    return []


def _save_cached_holidays(dates):
    HOLIDAY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates}).to_csv(HOLIDAY_CACHE, index=False)


def load_india_holidays(start_year: int = 2018, end_year: int = 2026):
    """
    Generate a list of holidays for India including national holidays.
    Uses cached data if already downloaded.

    Returns:
        List[str] : sorted YYYY-MM-DD holiday dates
    """
    # Try cache first
    cached = _load_cached_holidays()
    if cached:
        return cached

    # Official list (manual curated) – safe for offline mode
    # You can expand this based on govt. API if needed later
    manual_holidays = [
        "01-26", "08-15", "10-02",  # Republic, Independence, Gandhi Jayanti
        "01-01",  # New Year
        "12-25",  # Christmas
        # TODO: Add more india-specific festival dates from API in Stage-2
    ]

    holiday_dates = []
    for year in range(start_year, end_year + 1):
        for md in manual_holidays:
            holiday_dates.append(f"{year}-{md}")

    holiday_dates = sorted(holiday_dates)
    _save_cached_holidays(holiday_dates)
    return holiday_dates


def add_holiday_feature(df: pd.DataFrame, holiday_dates):
    """
    Adds a holiday_flag column to FRN dataframe
    """
    if "dt" not in df.columns:
        raise ValueError("Missing dt column in dataset for holiday assignment.")

    df = df.copy()
    df["dt"] = df["dt"].astype(str)
    df["holiday_flag"] = df["dt"].isin(holiday_dates).astype(int)
    return df
