"""
Metrological Uncertainty Estimation and GUM Budget Module.
Calculates Type A statistical residual uncertainty and combines with Type B sources
to evaluate expanded measurement uncertainty with coverage factor k=2.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class UncertaintyBudgetParams:
    sensor_repeatability: float = 0.20        # °C low-cost sensor repeatability (Type A)
    ref_uncertainty: float = 0.05             # °C manufacturer calibration uncertainty of reference (Type B)
    adc_resolution_uncertainty: float = 0.02  # °C quantization uncertainty (Type B, delta/sqrt(12))
    thermal_gradient_uncertainty: float = 0.08# °C spatial temperature inhomogeneity (Type B)
    coverage_factor_k: float = 2.0            # k=2 for ~95% confidence interval


def evaluate_uncertainty(
    y_test_ref: np.ndarray,
    y_test_pred: np.ndarray,
    params: UncertaintyBudgetParams = UncertaintyBudgetParams()
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates model residual uncertainty on the independent test split:
    residual = ML prediction - Reference
    and compiles a GUM-compliant uncertainty budget.
    """
    residuals = np.asarray(y_test_pred) - np.asarray(y_test_ref)

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))
    rmse_res = float(np.sqrt(np.mean(residuals ** 2)))

    # Component 1: ML Model Fit Residual Uncertainty (Type A)
    u_model = std_res

    # Component 2: Low-Cost Sensor Inherent Noise / Repeatability (Type A)
    u_sensor = params.sensor_repeatability

    # Component 3: Calibrated Reference Sensor Standard Uncertainty (Type B)
    u_ref = params.ref_uncertainty

    # Component 4: Digital ADC Quantization Uncertainty (Type B)
    u_adc = params.adc_resolution_uncertainty

    # Component 5: Thermal Gradient & Inhomogeneity (Type B)
    u_env = params.thermal_gradient_uncertainty

    # Combined Standard Uncertainty (Root Sum Square - RSS)
    u_combined = float(np.sqrt(u_model**2 + u_sensor**2 + u_ref**2 + u_adc**2 + u_env**2))

    # Expanded Uncertainty: U = k * u_c
    u_expanded = float(params.coverage_factor_k * u_combined)

    # Uncertainty Budget Table
    budget_data = [
        {
            "Uncertainty Source": "ML Residual Uncertainty (Model fit dispersion)",
            "Type": "Type A",
            "Distribution": "Normal",
            "Standard Uncertainty u_i (°C)": round(u_model, 4),
            "Sensitivity Coeff c_i": 1.0,
            "Contribution u_i² (°C²)": round(u_model ** 2, 6),
        },
        {
            "Uncertainty Source": "Low-Cost Sensor Repeatability / Jitter",
            "Type": "Type A",
            "Distribution": "Normal",
            "Standard Uncertainty u_i (°C)": round(u_sensor, 4),
            "Sensitivity Coeff c_i": 1.0,
            "Contribution u_i² (°C²)": round(u_sensor ** 2, 6),
        },
        {
            "Uncertainty Source": "Reference Sensor Calibration Standard",
            "Type": "Type B",
            "Distribution": "Normal",
            "Standard Uncertainty u_i (°C)": round(u_ref, 4),
            "Sensitivity Coeff c_i": 1.0,
            "Contribution u_i² (°C²)": round(u_ref ** 2, 6),
        },
        {
            "Uncertainty Source": "ADC Quantization & Digital Resolution",
            "Type": "Type B",
            "Distribution": "Rectangular",
            "Standard Uncertainty u_i (°C)": round(u_adc, 4),
            "Sensitivity Coeff c_i": 1.0,
            "Contribution u_i² (°C²)": round(u_adc ** 2, 6),
        },
        {
            "Uncertainty Source": "Spatial Thermal Inhomogeneity / Gradient",
            "Type": "Type B",
            "Distribution": "Rectangular",
            "Standard Uncertainty u_i (°C)": round(u_env, 4),
            "Sensitivity Coeff c_i": 1.0,
            "Contribution u_i² (°C²)": round(u_env ** 2, 6),
        },
    ]

    budget_df = pd.DataFrame(budget_data)

    summary = {
        "mean_residual": round(mean_res, 4),
        "residual_std": round(std_res, 4),
        "residual_rmse": round(rmse_res, 4),
        "u_combined": round(u_combined, 4),
        "coverage_factor_k": params.coverage_factor_k,
        "u_expanded": round(u_expanded, 4),
        "n_residuals": len(residuals),
    }

    return budget_df, summary
