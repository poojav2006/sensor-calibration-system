# ML-Based Self-Calibrating & Uncertainty-Aware Measurement System Using Low-Cost Sensors

An interactive engineering software prototype and measurement simulator developed for undergraduate/graduate engineering project demonstration and viva evaluation.

---

## 📌 Project Overview

Low-cost sensors (such as NTC thermistors, LM35, or commodity digital sensors) are economical and scalable for IoT and industrial automation. However, they exhibit significant physical measurement errors:
- **Static zero-point offset error**
- **Gain / Sensitivity drift**
- **Nonlinear thermal response transfer characteristics**
- **Thermal inertia and response lag ($\tau$)**
- **High-frequency electronic jitter and EMI spikes**
- **Environmental cross-sensitivity (ambient temperature and humidity coupling)**

This project implements an automated **self-calibrating measurement pipeline** where a low-cost temperature sensor is paired with a certified precision reference sensor. By employing **Machine Learning (Random Forest Regression)** and **GUM-compliant Metrological Uncertainty Analysis**, the system learns the multidimensional error surface, corrects nonlinear measurement errors in real time, and reports calibrated measurements with a rigorous **Expanded Uncertainty Interval ($T \pm U$, $k=2$)**.

> [!NOTE]
> **Prototype Status**: The current implementation is a high-fidelity **Software Simulation Prototype**. The software architecture is decoupled and designed so that the simulation data provider can later be replaced with physical ESP32 and DS18B20 hardware.

---

## 🚀 Key Features

1. **Realistic Sensor Simulator**:
   - Models thermal lag ($\tau$), nonlinear curvature, drift, static bias, and Gaussian jitter.
   - Temperature profiles: *Heating + Cooling (default)*, *Constant*, *Heating*, *Cooling*, *Random environmental variation*.
2. **Digital Preprocessing Pipeline**:
   - Missing packet handling (linear interpolation / drop).
   - Robust outlier rejection via Median Absolute Deviation (MAD / Z-Score).
   - Digital filtering: Moving Average and Median filters.
3. **Multi-Method Calibration Engine**:
   - **Method 1**: Conventional Offset Calibration ($T_{corr} = T_{sensor} - \mu_{err}$)
   - **Method 2**: Conventional Linear Regression ($T_{corr} = a \cdot T_{sensor} + b$)
   - **Method 3**: Random Forest Machine Learning Regressor (Single-variable & Multi-variable environmental modes)
4. **Independent Benchmark Comparison**:
   - Evaluates Raw Sensor vs Offset vs Linear vs ML on an independent test dataset (70% Train, 15% Validation, 15% Test).
   - Dynamically computes MAE, RMSE, Max Absolute Error, and $R^2$ Score.
5. **GUM-Compliant Uncertainty Estimation**:
   - Type A statistical residual evaluation combined with Type B instrument uncertainties.
   - Combined Standard Uncertainty ($u_c$) and Expanded Uncertainty ($U = 2 u_c$, $\approx 95\%$ confidence).
6. **1-Click DEMO MODE**:
   - Runs the entire pipeline end-to-end in seconds for rapid faculty demonstration.
7. **Comprehensive Engineering Graphs**:
   - Reference vs Raw Sensor
   - Error Before Calibration
   - Reference vs Raw vs ML Corrected
   - Residual Error Distribution Histogram
   - Predicted vs Reference Scatter Plot with ideal $y = x$ line
8. **Live Sensor Simulation**:
   - Real-time step generation with instantaneous ML calibration and uncertainty intervals.

---

## 📂 Project Structure

```
sensor_calibration_system/
├── app.py                      # Main interactive Streamlit web dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & viva guide
├── modules/
│   ├── __init__.py
│   ├── sensor_simulator.py     # Thermal profiles & sensor physical error model
│   ├── preprocessing.py        # Outlier rejection, missing values & digital filters
│   ├── calibration.py          # Conventional offset & linear regression calibration
│   ├── ml_model.py             # Random Forest regressor & feature pipeline
│   ├── uncertainty.py          # Metrological residual evaluation & GUM budget
│   └── evaluation.py           # MAE, RMSE, Max Error & method benchmark
├── data/
│   └── sample_data.csv         # Realistic sample experiment dataset
└── models/
    └── trained_model.pkl       # Serialized trained Random Forest calibrator
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.10+ (recommended Python 3.12) is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or explicitly: `pip install streamlit pandas numpy scikit-learn plotly matplotlib joblib`)*

---

## 🖥️ Running the Application

Launch the interactive web dashboard with:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## 📖 Step-by-Step Demonstration Workflow

1. **Open the Dashboard**: The landing page displays the Hero KPI cards, final measurement card ($T_{corr} \pm U$), and the live comparison graph.
2. **Click `🚀 1-CLICK DEMO MODE`**: Automatically runs simulation, preprocessing, conventional calibration, ML training, and uncertainty analysis in seconds.
3. **Visit `2. Experiment Setup`**: Customize temperature profile, thermal bounds ($20^\circ\text{C}$ to $50^\circ\text{C}$), noise levels, and sensor drift. Click **START EXPERIMENT**.
4. **Visit `3. Data Acquisition`**: Inspect the synchronized measurement table and download the raw CSV.
5. **Visit `4. Preprocessing`**: Toggle filter types (Moving Average vs Median) and observe the before/after graph.
6. **Visit `5. Conventional Calibration`**: Inspect the dynamically calculated offset bias and linear regression slope ($a$) and intercept ($b$).
7. **Visit `6. ML Model`**: View train/val/test splits, tune Random Forest hyperparameters, and view feature importances.
8. **Visit `7. Uncertainty Analysis`**: Examine the GUM uncertainty budget table, residual histogram, and expanded uncertainty interval.
9. **Visit `8. Performance Comparison`**: View the benchmark table proving ML error reduction over conventional techniques, explore the 5 engineering graphs, and export the Printable Technical Summary Report.
10. **Visit `9. System Architecture` & `10. Hardware Integration`**: Review the transition roadmap to physical ESP32/DS18B20 hardware.
11. **Visit `11. Viva / Explanation`**: Review quick answers to common examiner and teacher questions.

---

## 🔬 Hardware Transition Roadmap

The software architecture is engineered to match real microcontroller telemetry:
- **Microcontroller**: ESP32-WROOM-32 (Dual-core 240 MHz, 12-bit ADC)
- **Low-Cost Transducer**: DS18B20 digital temperature probe (1-Wire protocol)
- **Reference Standard**: Pt100 Class A RTD with MAX31865 amplifier (±0.05 °C accuracy)
- **Auxiliary Environment**: BME280 sensor (I2C)
- **Local Readout**: 0.96-inch OLED display (I2C)
- **Telemetry Link**: Serial UART or WiFi MQTT feeding into this software engine.
