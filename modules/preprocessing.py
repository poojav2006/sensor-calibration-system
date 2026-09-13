"""
Signal Preprocessing and Conditioning Module.
Includes missing value imputation, outlier detection, digital smoothing filters, and signal synchronization.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class PreprocessingConfig:
    handle_missing: bool = True
    missing_strategy: str = "linear_interpolation"
    remove_outliers: bool = True
    outlier_z_threshold: float = 3.0
    apply_filter: bool = True
    filter_type: str = "Moving Average"  # "Moving Average", "Median Filter", "Both"
    filter_window: int = 5
    synchronize_lag: bool = False
    lag_correction_samples: int = 2


def preprocess_sensor_data(df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Preprocesses raw sensor data:
    1. Detects and handles missing samples
    2. Identifies and cleans transient outliers
    3. Applies smoothing filtration (Moving Average / Median)
    4. Synchronizes signals and updates rate of change
    """
    clean_df = df.copy()
    raw_count = len(clean_df)

    # 1. Missing Value Check & Handling
    missing_count = int(clean_df["low_cost_sensor"].isna().sum())
    if config.handle_missing and missing_count > 0:
        if config.missing_strategy == "linear_interpolation":
            clean_df["low_cost_sensor"] = clean_df["low_cost_sensor"].interpolate(method="linear", limit_direction="both")
        elif config.missing_strategy == "drop":
            clean_df = clean_df.dropna(subset=["low_cost_sensor"]).reset_index(drop=True)

    # 2. Outlier Detection using Local Median Absolute Deviation (MAD)
    outlier_count = 0
    clean_df["is_outlier"] = False
    if config.remove_outliers and len(clean_df) > 10:
        rolling_med = clean_df["low_cost_sensor"].rolling(window=7, center=True, min_periods=3).median()
        diff = np.abs(clean_df["low_cost_sensor"] - rolling_med)
        mad = np.median(diff.dropna()) + 1e-6
        z_robust = 0.6745 * diff / mad
        outlier_mask = (z_robust > config.outlier_z_threshold)
        outlier_count = int(outlier_mask.sum())
        clean_df["is_outlier"] = outlier_mask
        # Replace outlier points with rolling median
        clean_df.loc[outlier_mask, "low_cost_sensor"] = rolling_med[outlier_mask]
        clean_df["low_cost_sensor"] = clean_df["low_cost_sensor"].interpolate(method="linear", limit_direction="both")

    # 3. Digital Filtering (Moving Average / Median)
    sensor_filtered = clean_df["low_cost_sensor"].to_numpy()
    if config.apply_filter and len(clean_df) >= config.filter_window:
        win = max(3, int(config.filter_window))
        if config.filter_type == "Median Filter":
            sensor_filtered = pd.Series(sensor_filtered).rolling(win, center=True, min_periods=1).median().to_numpy()
        elif config.filter_type == "Moving Average":
            sensor_filtered = pd.Series(sensor_filtered).rolling(win, center=True, min_periods=1).mean().to_numpy()
        elif config.filter_type == "Both":
            s_med = pd.Series(sensor_filtered).rolling(win, center=True, min_periods=1).median()
            sensor_filtered = s_med.rolling(win, center=True, min_periods=1).mean().to_numpy()

    clean_df["low_cost_sensor_clean"] = np.round(sensor_filtered, 3)

    # 4. Optional Lag Synchronization
    if config.synchronize_lag and config.lag_correction_samples > 0:
        shift = config.lag_correction_samples
        clean_df["low_cost_sensor_clean"] = clean_df["low_cost_sensor_clean"].shift(-shift).bfill()

    # Recalculate cleaned raw error
    clean_df["clean_error"] = np.round(clean_df["low_cost_sensor_clean"] - clean_df["reference_sensor"], 3)
    clean_df["rate_of_change"] = np.round(np.gradient(clean_df["low_cost_sensor_clean"], 1.0), 4)

    stats = {
        "raw_count": raw_count,
        "valid_count": len(clean_df),
        "missing_count": missing_count,
        "outlier_count": outlier_count,
        "filter_applied": config.apply_filter,
        "filter_type": config.filter_type,
        "filter_window": config.filter_window,
    }

    return clean_df, stats
