"""
Machine Learning Calibration Module using Random Forest Regressor.
Supports single-variable (sensor temperature) and multi-variable (ambient, humidity, dT/dt) features.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass
class MLConfig:
    feature_mode: str = "Single Sensor"  # "Single Sensor" or "Multi-Variable Environmental"
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 3
    random_state: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_split: bool = True  # Ensures train and test both cover the full thermal envelope


class MLCalibrator:
    """
    Random Forest Regressor calibrator mapping low-cost sensor readings
    (and optional environmental variables) to calibrated reference temperatures.
    """
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.train_split_info: Dict[str, int] = {}
        self.feature_importances: Dict[str, float] = {}

    def prepare_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                       pd.Series, pd.Series, pd.Series,
                                                       pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into Train (70%), Validation (15%), and Test (15%) splits.
        With shuffle_split=True, both splits represent the full calibration temperature range.
        """
        sensor_col = "low_cost_sensor_clean" if "low_cost_sensor_clean" in df.columns else "low_cost_sensor"
        if self.config.feature_mode == "Multi-Variable Environmental":
            self.feature_names = [sensor_col, "ambient_temperature", "relative_humidity", "rate_of_change"]
        else:
            self.feature_names = [sensor_col]

        X = df[self.feature_names]
        y = df["reference_sensor"]

        # 70% Train, 30% Temp (Split into 15% Val, 15% Test)
        X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
            X, y, df,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.random_state,
            shuffle=self.config.shuffle_split
        )

        X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
            X_temp, y_temp, df_temp,
            test_size=0.5,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle_split
        )

        self.train_split_info = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(df),
        }

        return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, any]:
        """Trains the Random Forest model on the training split."""
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        self.feature_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }

        return {
            "status": "TRAINED",
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "train_samples": len(X_train),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts corrected reference temperature for given features."""
        if not self.is_trained or self.model is None:
            raise ValueError("ML model has not been trained yet. Please train the model first.")
        X_input = X[self.feature_names]
        return self.model.predict(X_input)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "train_split_info": self.train_split_info,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MLCalibrator":
        data = joblib.load(filepath)
        instance = cls(config=data["config"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.feature_importances = data["feature_importances"]
        instance.train_split_info = data["train_split_info"]
        instance.is_trained = True
        return instance
