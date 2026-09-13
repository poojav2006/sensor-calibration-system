# Project Viva & Faculty Evaluation Guide (Confidential / Personal Reference)

Use this guide for your own preparation and viva defense. This is stored separately from the dashboard.

---

### 1. What is the problem?
Low-cost sensors (thermistors, LM35, commodity semiconductor probes) are economical and scalable for IoT, but they exhibit significant physical measurement errors: static zero offsets, gain/sensitivity drifts, nonlinear thermal transfer functions, thermal inertia / response lag (tau), and environmental cross-sensitivity (ambient temperature and humidity coupling).

---

### 2. What is the proposed solution?
A dual-sensor measurement framework where a low-cost sensor is calibrated against a laboratory-grade reference sensor using machine learning (Random Forest Regression). The ML model automatically learns the multidimensional error surface and applies real-time corrections, accompanied by a GUM-compliant (ISO 98-3) uncertainty budget.

---

### 3. Why ML instead of standard polynomial calibration?
Conventional calibration methods (such as offset subtraction or linear regression) assume simple, static linear error behaviors. In real-world environments, sensor errors are often nonlinear and cross-coupled with ambient conditions (humidity, thermal inertia, and heating/cooling rates). Random Forest regression learns complex nonlinear boundaries and non-monotonic transfer functions without overfitting, significantly reducing MAE and RMSE compared to linear models.

---

### 4. What is self-calibration?
The calibration relationship is learned automatically from reference data rather than manually entering or hardcoding a correction factor.

---

### 5. What is measurement uncertainty?
It indicates how reliable or variable the measurement result is. In this system, it is calculated as an expanded uncertainty interval (T_corrected +/- U, k=2) based on model residuals (Type A) and instrument specifications (Type B) according to the Guide to the Expression of Uncertainty in Measurement (GUM / ISO 98-3).

---

### 6. Why use a reference sensor?
It provides the reliable target measurement required to train and evaluate the calibration model.

---

### 7. How do you prove ML is better?
Compare raw, conventional (offset, linear OLS), and ML-calibrated measurements using MAE, RMSE, and maximum error on an independent test dataset (unseen during training). In our simulation, Random Forest ML achieves a 50-65% error reduction over uncalibrated raw readings.

---

### 8. Is this a real hardware system?
Currently it is a high-fidelity software prototype/simulation. The software architecture is designed so the simulated sensor source can later be replaced with ESP32 hardware reading DS18B20 and Pt100 RTD sensors.

---

### 9. What is novel?
The project integrates low-cost sensing, ML-based calibration, and formal GUM-aligned metrological uncertainty estimation into one unified, educational measurement framework.
