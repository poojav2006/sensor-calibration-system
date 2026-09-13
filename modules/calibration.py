"""
Conventional Sensor Calibration Module.
Implements:
1. Offset Calibration: T_corr = T_sensor - mean_error
2. Linear Regression Calibration: T_corr = a * T_sensor + b
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class OffsetCalibrationResult:
    offset: float
    formula: str
    description: str


@dataclass
class LinearCalibrationResult:
    slope: float
    intercept: float
    formula: str
    r2: float
    description: str


class OffsetCalibrator:
    """
    Conventional Offset Calibration.
    Corrected Temperature = T_sensor - mean_error
    """
    def __init__(self):
        self.offset: float = 0.0
        self.is_fitted: bool = False

    @property
    def formula(self) -> str:
        if not self.is_fitted:
            return "T_corrected = T_sensor"
        sign = "-" if self.offset >= 0 else "+"
        return f"T_corrected = T_sensor {sign} {abs(self.offset):.3f} °C"

    def fit(self, sensor_readings: np.ndarray, reference_readings: np.ndarray) -> OffsetCalibrationResult:
        errors = np.asarray(sensor_readings) - np.asarray(reference_readings)
        self.offset = float(np.mean(errors))
        self.is_fitted = True
        return OffsetCalibrationResult(
            offset=round(self.offset, 4),
            formula=self.formula,
            description="Static baseline offset correction calculated by averaging measurement residual bias."
        )

    def predict(self, sensor_readings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("OffsetCalibrator has not been fitted yet.")
        return np.asarray(sensor_readings) - self.offset


class LinearCalibrator:
    """
    Conventional Linear Regression Calibration.
    Corrected Temperature = a * T_sensor + b
    Determined dynamically using Ordinary Least Squares (OLS).
    """
    def __init__(self):
        self.slope: float = 1.0
        self.intercept: float = 0.0
        self.r2: float = 0.0
        self.is_fitted: bool = False

    @property
    def formula(self) -> str:
        if not self.is_fitted:
            return "T_corrected = T_sensor"
        sign = "+" if self.intercept >= 0 else "-"
        return f"T_corrected = {self.slope:.4f} × T_sensor {sign} {abs(self.intercept):.3f} °C"

    def fit(self, sensor_readings: np.ndarray, reference_readings: np.ndarray) -> LinearCalibrationResult:
        x = np.asarray(sensor_readings)
        y = np.asarray(reference_readings)
        poly = np.polyfit(x, y, deg=1)
        self.slope = float(poly[0])
        self.intercept = float(poly[1])
        self.is_fitted = True

        y_pred = self.slope * x + self.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

        return LinearCalibrationResult(
            slope=round(self.slope, 4),
            intercept=round(self.intercept, 4),
            formula=self.formula,
            r2=round(self.r2, 4),
            description="First-order linear fit compensating for both sensitivity gain shift and zero offset."
        )

    def predict(self, sensor_readings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("LinearCalibrator has not been fitted yet.")
        return self.slope * np.asarray(sensor_readings) + self.intercept
