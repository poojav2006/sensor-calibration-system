"""
Realistic sensor and environmental simulator for temperature measurement.
Simulates low-cost sensors (with offset, gain error, nonlinearity, response lag, drift, and jitter)
alongside a calibrated reference sensor.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class SensorParams:
    profile: str = "Heating + Cooling"
    t_min: float = 20.0
    t_max: float = 50.0
    n_samples: int = 500
    sampling_interval: float = 1.0  # seconds
    noise_std: float = 0.35          # °C Gaussian noise standard deviation
    offset_error: float = 1.50       # °C static offset error
    drift_rate: float = 0.001        # °C per sample drift
    gain_error: float = 0.04         # 4% sensitivity/gain scale error
    nonlinear_coeff: float = 0.0012   # Quadratic thermal distortion curvature
    tau_lag: float = 4.0             # Thermal response time constant in seconds
    ref_noise_std: float = 0.03      # °C calibrated reference sensor noise
    ref_offset: float = 0.02         # °C calibrated reference offset
    random_seed: int = 42
    add_outliers: bool = True
    add_missing: bool = True


def generate_experiment_data(params: SensorParams) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Generates synchronized physical, low-cost sensor, and calibrated reference readings.
    Simulates realistic imperfections: static offset, thermal response lag,
    gain error, quadratic nonlinearity, temporal drift, and measurement jitter.
    """
    rng = np.random.default_rng(params.random_seed)
    n = params.n_samples
    t = np.arange(n) * params.sampling_interval

    # 1. True Physical Temperature Profile Simulation
    if params.profile == "Constant":
        t_true = np.full(n, (params.t_min + params.t_max) / 2.0)
    elif params.profile == "Heating":
        # Smooth Sigmoidal heating ramp
        x = np.linspace(-3.5, 3.5, n)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        t_true = params.t_min + (params.t_max - params.t_min) * sigmoid
    elif params.profile == "Cooling":
        # Exponential thermal decay
        decay = np.exp(-t / (0.35 * t[-1] + 1e-5))
        t_true = params.t_min + (params.t_max - params.t_min) * decay
    elif params.profile == "Random environmental variation":
        # Environmental room thermal fluctuations with cyclic diurnal variation
        base = (params.t_min + params.t_max) / 2.0
        amp = (params.t_max - params.t_min) / 2.4
        t_true = (
            base
            + amp * 0.7 * np.sin(2 * np.pi * t / (n * 0.45))
            + amp * 0.3 * np.cos(2 * np.pi * t / (n * 0.18))
            + np.cumsum(rng.normal(0, 0.035, n))
        )
        t_true = np.clip(t_true, params.t_min, params.t_max)
    else:  # Default: "Heating + Cooling"
        half = n // 2
        x_heat = np.linspace(0, np.pi, half)
        heat_phase = params.t_min + (params.t_max - params.t_min) * (1.0 - np.cos(x_heat)) / 2.0
        cool_len = n - half
        cool_phase = params.t_min + (heat_phase[-1] - params.t_min) * np.exp(-np.linspace(0, 2.8, cool_len))
        t_true = np.concatenate([heat_phase, cool_phase])

    # 2. Calibrated Reference Sensor Model
    # High-quality laboratory reference thermometer with minimal noise and documented calibration
    ref_noise = rng.normal(0, params.ref_noise_std, n)
    t_ref = t_true + params.ref_offset + ref_noise

    # 3. Imperfect Low-Cost Sensor Model
    # A) Thermal Response Lag (Discrete 1st-order IIR low-pass filter)
    alpha = params.sampling_interval / (params.tau_lag + params.sampling_interval)
    t_lagged = np.zeros(n)
    t_lagged[0] = t_true[0]
    for i in range(1, n):
        t_lagged[i] = alpha * t_true[i] + (1.0 - alpha) * t_lagged[i - 1]

    # B) Systematic Errors: Offset, Gain Sensitivity, Quadratic Nonlinearity, and Drift
    t_center = (params.t_min + params.t_max) / 2.0
    gain_err = params.gain_error * (t_lagged - t_center)
    nonlin_err = params.nonlinear_coeff * ((t_lagged - t_center) ** 2)
    drift_err = params.drift_rate * np.arange(n)
    raw_noise = rng.normal(0, params.noise_std, n)

    t_sensor = t_lagged + params.offset_error + gain_err + nonlin_err + drift_err + raw_noise

    # C) Add occasional realistic electrical glitches & missing packets (for Preprocessing demo)
    if params.add_outliers and n > 20:
        outlier_indices = rng.choice(range(10, n - 10), size=min(4, max(1, n // 50)), replace=False)
        for idx in outlier_indices:
            spike = rng.choice([-1.0, 1.0]) * rng.uniform(3.5, 5.5)
            t_sensor[idx] += spike

    if params.add_missing and n > 30:
        missing_indices = rng.choice(range(15, n - 15), size=min(3, max(1, n // 80)), replace=False)
        for idx in missing_indices:
            t_sensor[idx] = np.nan

    # 4. Auxiliary Simulated Environmental Sensors (BME280 emulation)
    amb_temp = 23.5 + 1.2 * np.sin(2 * np.pi * t / (n * 0.8)) + rng.normal(0, 0.08, n)
    rel_humidity = 55.0 - 0.3 * (t_true - params.t_min) + rng.normal(0, 0.4, n)
    rel_humidity = np.clip(rel_humidity, 20.0, 85.0)

    # Numerical derivative dT/dt
    # Fill nan temporarily for gradient
    s_temp = pd.Series(t_sensor).interpolate(method="linear", limit_direction="both").to_numpy()
    dt_rate = np.gradient(s_temp, params.sampling_interval)

    # Raw Error calculation: Low-Cost Sensor - Reference Sensor
    raw_error = t_sensor - t_ref

    base_time = pd.Timestamp("2026-09-08 10:00:00")
    timestamps = [base_time + pd.Timedelta(seconds=float(s)) for s in t]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "time_s": np.round(t, 2),
        "true_temperature": np.round(t_true, 3),
        "low_cost_sensor": np.round(t_sensor, 3),
        "reference_sensor": np.round(t_ref, 3),
        "raw_error": np.round(raw_error, 3),
        "ambient_temperature": np.round(amb_temp, 2),
        "relative_humidity": np.round(rel_humidity, 1),
        "rate_of_change": np.round(dt_rate, 4),
    })

    metadata = {
        "profile": params.profile,
        "n_samples": n,
        "t_min": params.t_min,
        "t_max": params.t_max,
        "sampling_interval": params.sampling_interval,
        "offset_error": params.offset_error,
        "noise_std": params.noise_std,
        "drift_rate": params.drift_rate,
        "gain_error": params.gain_error,
        "tau_lag": params.tau_lag,
        "ref_offset": params.ref_offset,
        "ref_noise_std": params.ref_noise_std,
        "random_seed": params.random_seed,
    }

    return df, metadata
