import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure modules directory is discoverable
APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from modules.sensor_simulator import SensorParams, generate_experiment_data
from modules.preprocessing import PreprocessingConfig, preprocess_sensor_data
from modules.calibration import OffsetCalibrator, LinearCalibrator
from modules.ml_model import MLConfig, MLCalibrator
from modules.uncertainty import UncertaintyBudgetParams, evaluate_uncertainty
from modules.evaluation import compare_calibration_methods, compute_metrics

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="ML Sensor Self-Calibration & Uncertainty System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL ENGINEERING STYLING ---
st.markdown('''
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 4px;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.25);
        margin: 15px 0 25px 0;
    }
    .hero-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
    }
    .hero-main-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }
    .hero-details {
        display: flex;
        justify-content: space-around;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #312e81;
    }
    .hero-detail-item {
        font-size: 0.95rem;
    }
    .badge-simulated {
        background-color: #f59e0b;
        color: #000000;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-left: 8px;
    }
    .badge-online {
        background-color: #10b981;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .badge-active {
        background-color: #3b82f6;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
''', unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
def init_session_state():
    defaults = {
        "raw_data": None,
        "clean_data": None,
        "prep_stats": None,
        "offset_cal": None,
        "linear_cal": None,
        "ml_cal": None,
        "df_train": None,
        "df_val": None,
        "df_test": None,
        "comparison_df": None,
        "comparison_details": None,
        "budget_df": None,
        "uncertainty_summary": None,
        "experiment_meta": None,
        "sensor_params": SensorParams(),
        "prep_config": PreprocessingConfig(),
        "ml_config": MLConfig(),
        "u_params": UncertaintyBudgetParams(),
        "status_sensor": "SIMULATED ONLINE",
        "status_reference": "SIMULATED ONLINE",
        "status_ml": "NOT TRAINED",
        "live_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def run_full_pipeline():
    with st.spinner("Executing end-to-end measurement & calibration pipeline..."):
        # 1. Experiment Simulation
        raw_df, meta = generate_experiment_data(st.session_state.sensor_params)
        st.session_state.raw_data = raw_df
        st.session_state.experiment_meta = meta

        # 2. Preprocessing
        clean_df, p_stats = preprocess_sensor_data(raw_df, st.session_state.prep_config)
        st.session_state.clean_data = clean_df
        st.session_state.prep_stats = p_stats

        # 3. Conventional Calibration
        sensor_col = "low_cost_sensor_clean"
        offset_cal = OffsetCalibrator()
        offset_cal.fit(clean_df[sensor_col], clean_df["reference_sensor"])
        st.session_state.offset_cal = offset_cal

        linear_cal = LinearCalibrator()
        linear_cal.fit(clean_df[sensor_col], clean_df["reference_sensor"])
        st.session_state.linear_cal = linear_cal

        # 4. ML Model Training
        ml_cal = MLCalibrator(st.session_state.ml_config)
        X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = ml_cal.prepare_splits(clean_df)
        ml_cal.train(X_train, y_train)
        st.session_state.ml_cal = ml_cal
        st.session_state.df_train = df_train
        st.session_state.df_val = df_val
        st.session_state.df_test = df_test
        st.session_state.status_ml = "TRAINED (RF)"

        # Save model
        model_file = APP_DIR / "models" / "trained_model.pkl"
        ml_cal.save(str(model_file))

        # 5. Performance Comparison
        comp_df, comp_details = compare_calibration_methods(
            test_df=df_test,
            offset_model=offset_cal,
            linear_model=linear_cal,
            ml_model=ml_cal,
            sensor_col=sensor_col
        )
        st.session_state.comparison_df = comp_df
        st.session_state.comparison_details = comp_details

        # 6. Uncertainty Estimation
        y_test_pred = comp_details["pred_ml"]
        y_test_ref = df_test["reference_sensor"].to_numpy()
        budget_df, u_summary = evaluate_uncertainty(
            y_test_ref=y_test_ref,
            y_test_pred=y_test_pred,
            params=st.session_state.u_params
        )
        st.session_state.budget_df = budget_df
        st.session_state.uncertainty_summary = u_summary

# Cold-start pipeline if empty
if st.session_state.raw_data is None:
    sample_csv = APP_DIR / "data" / "sample_data.csv"
    if sample_csv.exists():
        run_full_pipeline()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🎛️ Navigation")
    pages = [
        "1. Dashboard",
        "2. Experiment Setup",
        "3. Data Acquisition",
        "4. Preprocessing",
        "5. Conventional Calibration",
        "6. ML Model",
        "7. Uncertainty Analysis",
        "8. Performance Comparison",
        "9. System Architecture",
        "10. Hardware Integration"
    ]
    selected_page = st.radio("Select View:", pages, index=0)

    st.markdown("---")
    st.subheader("⚡ Quick Action Bar")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 DEMO MODE", use_container_width=True, help="1-Click Complete Pipeline Execution"):
            run_full_pipeline()
            st.success("Demo pipeline executed!")
            st.rerun()

    with col_btn2:
        if st.button("🔄 Reset / New", use_container_width=True, help="Generate fresh simulated experiment"):
            st.session_state.sensor_params.random_seed = int(np.random.randint(1, 10000))
            run_full_pipeline()
            st.rerun()

    st.markdown("---")
    st.subheader("📡 System Telemetry Status")
    st.caption("Current telemetry reflects simulated sensor states.")

    st.markdown(f'''
    - **Sensor:** <span class="badge-online">{st.session_state.status_sensor}</span>
    - **Reference:** <span class="badge-online">{st.session_state.status_reference}</span>
    - **Data Acquisition:** <span class="badge-active">ACTIVE</span>
    - **Data Processing:** <span class="badge-active">COMPLETE</span>
    - **ML Engine:** <span class="badge-active">{st.session_state.status_ml}</span>
    - **Uncertainty Engine:** <span class="badge-active">ACTIVE</span>
    ''', unsafe_allow_html=True)

    st.info("💡 **Notice**: Current implementation is a software simulation of the proposed measurement system. Hardware integration is the next development stage.")

# =====================================================================
# PAGE 1: DASHBOARD
# =====================================================================
if selected_page == "1. Dashboard":
    st.title("ML-Based Self-Calibrating & Uncertainty-Aware Measurement System")
    st.markdown("#### Low-Cost Temperature Sensor Calibration — Software Prototype <span class='badge-simulated'>SIMULATED</span>", unsafe_allow_html=True)
    st.caption("Interactive measurement prototype demonstrating low-cost sensor calibration using Random Forest ML and GUM-compliant uncertainty estimation.")

    # Action buttons
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1.5, 1.2, 1.2, 1.2])
    with c_btn1:
        if st.button("🚀 1-CLICK DEMO MODE", type="primary", use_container_width=True):
            run_full_pipeline()
            st.rerun()
    with c_btn2:
        if st.button("▶️ RUN EXPERIMENT", use_container_width=True):
            raw_df, meta = generate_experiment_data(st.session_state.sensor_params)
            st.session_state.raw_data = raw_df
            st.session_state.experiment_meta = meta
            st.rerun()
    with c_btn3:
        if st.button("⚙️ RUN CALIBRATION", use_container_width=True):
            if st.session_state.clean_data is not None:
                st.session_state.offset_cal.fit(st.session_state.clean_data["low_cost_sensor_clean"], st.session_state.clean_data["reference_sensor"])
                st.session_state.linear_cal.fit(st.session_state.clean_data["low_cost_sensor_clean"], st.session_state.clean_data["reference_sensor"])
                st.success("Calibrations updated!")
            st.rerun()
    with c_btn4:
        if st.button("🧠 TRAIN ML MODEL", use_container_width=True):
            if st.session_state.clean_data is not None:
                ml_cal = MLCalibrator(st.session_state.ml_config)
                X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = ml_cal.prepare_splits(st.session_state.clean_data)
                ml_cal.train(X_train, y_train)
                st.session_state.ml_cal = ml_cal
                st.session_state.df_test = df_test
                st.session_state.status_ml = "TRAINED (RF)"
                st.success("ML model trained!")
            st.rerun()

    st.markdown("---")

    if st.session_state.comparison_details is not None and st.session_state.df_test is not None:
        last_row = st.session_state.df_test.iloc[-1]
        raw_val = float(last_row["low_cost_sensor_clean"])
        ref_val = float(last_row["reference_sensor"])
        pred_val = float(st.session_state.comparison_details["pred_ml"][-1])
        u_exp = float(st.session_state.uncertainty_summary["u_expanded"])
        ml_err = abs(pred_val - ref_val)
        raw_mae = float(st.session_state.comparison_details["raw_metrics"]["MAE"])
        ml_mae = float(st.session_state.comparison_details["ml_metrics"]["MAE"])
        ml_rmse = float(st.session_state.comparison_details["ml_metrics"]["RMSE"])

        # FINAL MEASUREMENT CARD
        st.markdown(f'''
        <div class="hero-card">
            <div class="hero-title">FINAL MEASUREMENT RESULT (TEST SAMPLE)</div>
            <div class="hero-main-value">{pred_val:.2f} ± {u_exp:.2f} °C</div>
            <div style="font-size: 0.95rem; color: #a5b4fc;">Estimated Expanded Uncertainty (simulation, k=2, ~95% confidence)</div>
            <div class="hero-details">
                <div class="hero-detail-item"><b>Raw Sensor:</b> {raw_val:.2f} °C</div>
                <div class="hero-detail-item"><b>ML Corrected:</b> {pred_val:.2f} °C</div>
                <div class="hero-detail-item"><b>Reference:</b> {ref_val:.2f} °C</div>
                <div class="hero-detail-item"><b>ML Error:</b> {ml_err:.2f} °C</div>
                <div class="hero-detail-item"><b>Uncertainty:</b> ±{u_exp:.2f} °C</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # KPI Metrics
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-title">Raw Temperature</div>
                <div class="metric-card-value">{raw_val:.2f} <span style="font-size:1rem;">°C</span></div>
                <div class="metric-card-sub">Raw Error: {raw_val - ref_val:+.2f} °C</div>
            </div>
            ''', unsafe_allow_html=True)
        with k2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-title">ML Corrected Temp</div>
                <div class="metric-card-value">{pred_val:.2f} <span style="font-size:1rem;">°C</span></div>
                <div class="metric-card-sub">Error: {pred_val - ref_val:+.2f} °C</div>
            </div>
            ''', unsafe_allow_html=True)
        with k3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-title">Reference Temp</div>
                <div class="metric-card-value">{ref_val:.2f} <span style="font-size:1rem;">°C</span></div>
                <div class="metric-card-sub">Reference Measurement</div>
            </div>
            ''', unsafe_allow_html=True)
        with k4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-title">ML Test MAE</div>
                <div class="metric-card-value">{ml_mae:.3f} <span style="font-size:1rem;">°C</span></div>
                <div class="metric-card-sub">Raw MAE: {raw_mae:.3f} °C</div>
            </div>
            ''', unsafe_allow_html=True)
        with k5:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-title">Estimated Uncertainty</div>
                <div class="metric-card-value">±{u_exp:.3f} <span style="font-size:1rem;">°C</span></div>
                <div class="metric-card-sub">Coverage factor k = 2</div>
            </div>
            ''', unsafe_allow_html=True)

        # LIVE COMPARISON GRAPH
        st.subheader("📈 Reference vs Raw vs ML Corrected Temperature")
        test_df_plot = st.session_state.df_test.copy().sort_values("time_s")
        test_df_plot["ML Corrected"] = st.session_state.comparison_details["pred_ml"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df_plot["time_s"], y=test_df_plot["low_cost_sensor_clean"],
            mode="lines", name="Raw Low-Cost Sensor",
            line=dict(color="#f87171", width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=test_df_plot["time_s"], y=test_df_plot["reference_sensor"],
            mode="lines", name="Reference Sensor",
            line=dict(color="#10b981", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=test_df_plot["time_s"], y=test_df_plot["ML Corrected"],
            mode="lines", name="ML Corrected",
            line=dict(color="#38bdf8", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=list(test_df_plot["time_s"]) + list(test_df_plot["time_s"])[::-1],
            y=list(test_df_plot["ML Corrected"] + u_exp) + list(test_df_plot["ML Corrected"] - u_exp)[::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"±{u_exp:.2f} °C Uncertainty (k=2)"
        ))
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Temperature (°C)",
            template="plotly_dark",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Conclusion**: {st.session_state.comparison_details['conclusion']}")

        # --- LIVE SENSOR SIMULATION WIDGET ---
        st.markdown("---")
        st.subheader("🔴 LIVE SENSOR SIMULATION")
        st.caption("Simulates real-time thermal readings streaming into the system with instantaneous ML calibration and uncertainty bounds.")

        sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 3])
        with sim_col1:
            run_live_step = st.button("⚡ GENERATE NEXT LIVE SAMPLE", type="primary", use_container_width=True)
        with sim_col2:
            if st.button("🧹 Clear Live History", use_container_width=True):
                st.session_state.live_history = []
                st.rerun()

        if run_live_step:
            # Generate one live simulated sample
            curr_step = len(st.session_state.live_history) + 1
            t_base = 25.0 + 10.0 * np.sin(curr_step * 0.2)
            true_t = t_base + float(np.random.normal(0, 0.05))
            ref_t = true_t + float(np.random.normal(0, 0.03))
            raw_t = true_t + 1.5 + 0.04 * (true_t - 25.0) + 0.0012 * ((true_t - 25.0)**2) + float(np.random.normal(0, 0.35))
            
            # Predict using trained ML model
            input_df = pd.DataFrame([{
                "low_cost_sensor_clean": raw_t,
                "ambient_temperature": 24.0,
                "relative_humidity": 50.0,
                "rate_of_change": 0.0
            }])
            ml_corr_t = float(st.session_state.ml_cal.predict(input_df)[0])

            st.session_state.live_history.append({
                "Step": curr_step,
                "Raw Temperature (°C)": round(raw_t, 2),
                "ML Corrected (°C)": round(ml_corr_t, 2),
                "Reference (°C)": round(ref_t, 2),
                "Uncertainty (±°C)": round(u_exp, 2),
                "Residual Error (°C)": round(abs(ml_corr_t - ref_t), 2)
            })

        if st.session_state.live_history:
            live_df = pd.DataFrame(st.session_state.live_history)
            latest = live_df.iloc[-1]

            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Live Raw Temperature", f"{latest['Raw Temperature (°C)']:.2f} °C")
            lc2.metric("Live ML Corrected", f"{latest['ML Corrected (°C)']:.2f} °C")
            lc3.metric("Live Reference", f"{latest['Reference (°C)']:.2f} °C")
            lc4.metric("Estimated Uncertainty", f"±{latest['Uncertainty (±°C)']:.2f} °C")

            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(x=live_df["Step"], y=live_df["Raw Temperature (°C)"], mode="lines+markers", name="Raw Sensor", line=dict(color="#f87171", dash="dot")))
            fig_live.add_trace(go.Scatter(x=live_df["Step"], y=live_df["Reference (°C)"], mode="lines+markers", name="Reference", line=dict(color="#10b981", width=2.5)))
            fig_live.add_trace(go.Scatter(x=live_df["Step"], y=live_df["ML Corrected (°C)"], mode="lines+markers", name="ML Corrected", line=dict(color="#38bdf8", width=2.5)))
            fig_live.update_layout(xaxis_title="Sample Step", yaxis_title="Temperature (°C)", template="plotly_dark", height=320)
            st.plotly_chart(fig_live, use_container_width=True)

    else:
        st.info("⚠️ Please click **1-CLICK DEMO MODE** above to run the calibration pipeline.")

# =====================================================================
# PAGE 2: EXPERIMENT SETUP
# =====================================================================
elif selected_page == "2. Experiment Setup":
    st.title("⚙️ Thermal Experiment Simulator Setup")
    st.markdown("Configure simulated sensor error parameters and temperature profile.")

    with st.form("experiment_setup_form"):
        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            profile = st.selectbox(
                "Temperature Profile",
                ["Heating + Cooling", "Heating", "Cooling", "Constant", "Random environmental variation"],
                index=0
            )
            t_min = st.slider("Minimum Temperature (°C)", min_value=15.0, max_value=30.0, value=20.0, step=0.5)
            t_max = st.slider("Maximum Temperature (°C)", min_value=35.0, max_value=70.0, value=50.0, step=0.5)
            n_samples = st.slider("Number of Samples", min_value=100, max_value=1500, value=500, step=50)
            sampling_interval = st.number_input("Sampling Interval (seconds)", min_value=0.1, max_value=10.0, value=1.0, step=0.5)

        with col_prof2:
            st.markdown("##### 🔬 Sensor Imperfections")
            offset_error = st.slider("Sensor Static Offset Error (°C)", min_value=-5.0, max_value=5.0, value=1.5, step=0.1)
            noise_std = st.slider("Sensor Random Noise (std dev, °C)", min_value=0.05, max_value=1.5, value=0.35, step=0.05)
            drift_rate = st.slider("Sensor Drift Rate (°C/1000 samples)", min_value=-5.0, max_value=5.0, value=1.0, step=0.5) / 1000.0
            gain_error = st.slider("Sensor Gain Error (%)", min_value=-15.0, max_value=15.0, value=4.0, step=0.5) / 100.0
            tau_lag = st.slider("Sensor Response Lag (τ, seconds)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)

            st.markdown("##### 🧪 Reference Measurement")
            ref_noise = st.slider("Reference Sensor Noise (°C)", min_value=0.01, max_value=0.10, value=0.03, step=0.01)
            random_seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=42)

        submit_setup = st.form_submit_button("🔥 START EXPERIMENT", type="primary", use_container_width=True)

    if submit_setup:
        st.session_state.sensor_params = SensorParams(
            profile=profile,
            t_min=t_min,
            t_max=t_max,
            n_samples=n_samples,
            sampling_interval=sampling_interval,
            offset_error=offset_error,
            noise_std=noise_std,
            drift_rate=drift_rate,
            gain_error=gain_error,
            tau_lag=tau_lag,
            ref_noise_std=ref_noise,
            random_seed=int(random_seed)
        )
        run_full_pipeline()
        st.success(f"Generated {n_samples} synchronized points under '{profile}' profile!")
        st.rerun()


# =====================================================================
# PAGE 3: DATA ACQUISITION
# =====================================================================
elif selected_page == "3. Data Acquisition":
    st.title("📊 Data Acquisition (DAQ) System")
    st.markdown("Synchronized measurement streams from the low-cost sensor and reference sensor standard.")

    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data

        c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
        c_stat1.metric("Total Acquired Samples", len(df))
        c_stat2.metric("Mean Raw Error", f"{df['raw_error'].mean():.3f} °C")
        c_stat3.metric("Raw Error Std Dev", f"{df['raw_error'].std():.3f} °C")
        c_stat4.metric("Sampling Rate", f"{1.0/st.session_state.sensor_params.sampling_interval:.1f} Hz")

        st.subheader("📋 Synchronized Acquisition Table")
        st.caption("Raw Error = Low-Cost Sensor - Reference Sensor")
        st.dataframe(df, use_container_width=True, height=350)

        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Raw Dataset CSV",
            data=csv_bytes,
            file_name="raw_sensor_experiment_dataset.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.warning("Please run the experiment first.")


# =====================================================================
# PAGE 4: PREPROCESSING
# =====================================================================
elif selected_page == "4. Preprocessing":
    st.title("🧹 Data Preprocessing & Signal Conditioning")
    st.markdown("Rejects outliers, imputes missing packets, and filters high-frequency noise.")

    if st.session_state.raw_data is not None:
        with st.expander("🛠️ Preprocessing Configuration Controls", expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                handle_missing = st.checkbox("Handle Missing Values / NaNs", value=True)
                missing_strat = st.selectbox("Imputation Strategy", ["linear_interpolation", "drop"])
            with col_p2:
                remove_outliers = st.checkbox("Reject Transient Outliers", value=True)
                z_thresh = st.slider("Outlier Z-Threshold (MAD)", min_value=1.5, max_value=5.0, value=3.0, step=0.2)
            with col_p3:
                apply_filter = st.checkbox("Apply Digital Smoothing Filter", value=True)
                filter_type = st.selectbox("Filter Type", ["Moving Average", "Median Filter", "Both"])
                filter_win = st.slider("Filter Window Size", min_value=3, max_value=15, value=5, step=2)

            if st.button("Apply Preprocessing Settings", type="primary"):
                st.session_state.prep_config = PreprocessingConfig(
                    handle_missing=handle_missing,
                    missing_strategy=missing_strat,
                    remove_outliers=remove_outliers,
                    outlier_z_threshold=z_thresh,
                    apply_filter=apply_filter,
                    filter_type=filter_type,
                    filter_window=filter_win
                )
                clean_df, p_stats = preprocess_sensor_data(st.session_state.raw_data, st.session_state.prep_config)
                st.session_state.clean_data = clean_df
                st.session_state.prep_stats = p_stats
                st.success("Preprocessing updated!")
                st.rerun()

        stats = st.session_state.prep_stats or {}
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Raw Samples", stats.get("raw_count", len(st.session_state.raw_data)))
        sc2.metric("Valid Clean Samples", stats.get("valid_count", len(st.session_state.raw_data)))
        sc3.metric("Missing Values Imputed", stats.get("missing_count", 0))
        sc4.metric("Outliers Removed", stats.get("outlier_count", 0))

        st.subheader("📉 Before vs After Signal Conditioning")
        clean_df = st.session_state.clean_data
        fig_prep = go.Figure()
        fig_prep.add_trace(go.Scatter(
            x=clean_df["time_s"], y=st.session_state.raw_data["low_cost_sensor"],
            mode="lines+markers", name="Raw Low-Cost Sensor (Noisy)",
            line=dict(color="#f87171", width=1.5), marker=dict(size=4)
        ))
        fig_prep.add_trace(go.Scatter(
            x=clean_df["time_s"], y=clean_df["low_cost_sensor_clean"],
            mode="lines", name=f"Filtered ({filter_type})",
            line=dict(color="#38bdf8", width=2.5)
        ))
        fig_prep.add_trace(go.Scatter(
            x=clean_df["time_s"], y=clean_df["reference_sensor"],
            mode="lines", name="Reference Sensor",
            line=dict(color="#10b981", width=2, dash="dash")
        ))
        fig_prep.update_layout(
            xaxis_title="Time (s)", yaxis_title="Temperature (°C)",
            template="plotly_dark", height=380
        )
        st.plotly_chart(fig_prep, use_container_width=True)

        csv_clean_bytes = clean_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Preprocessed Dataset CSV",
            data=csv_clean_bytes,
            file_name="preprocessed_sensor_dataset.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please generate an experiment first.")

# =====================================================================
# PAGE 5: CONVENTIONAL CALIBRATION
# =====================================================================
elif selected_page == "5. Conventional Calibration":
    st.title("📏 Conventional Sensor Calibration")
    st.markdown("Evaluates traditional Offset subtraction and Linear regression calibration methods.")

    if st.session_state.offset_cal is not None and st.session_state.linear_cal is not None:
        off = st.session_state.offset_cal
        lin = st.session_state.linear_cal

        c_meth1, c_meth2 = st.columns(2)
        with c_meth1:
            st.subheader("Method 1: Offset Calibration")
            st.markdown(r'''
            Calculates static mean error offset:
            $$T_{corrected} = T_{sensor} - \text{mean\_error}$$
            ''')
            st.info(f"**Calculated Equation**: `{off.formula}`")
            st.metric("Mean Error Offset", f"{off.offset:.4f} °C")

        with c_meth2:
            st.subheader("Method 2: Linear Regression Calibration")
            st.markdown(r'''
            Fits first-order linear transfer function using Ordinary Least Squares:
            $$T_{corrected} = a \cdot T_{sensor} + b$$
            ''')
            st.info(f"**Calculated Equation**: `{lin.formula}`")
            st.metric("Slope (a)", f"{lin.slope:.4f}")
            st.metric("Intercept (b)", f"{lin.intercept:.4f} °C")
            st.metric("R² Score", f"{lin.r2:.4f}")

        st.subheader("📊 Calibration Characteristic Curves")
        cal_x = np.linspace(st.session_state.sensor_params.t_min, st.session_state.sensor_params.t_max, 100)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=st.session_state.clean_data["low_cost_sensor_clean"],
            y=st.session_state.clean_data["reference_sensor"],
            mode="markers", name="Experimental Data Pairs",
            marker=dict(color="#94a3b8", opacity=0.6, size=5)
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal_x, y=cal_x, mode="lines", name="Ideal Unity (y = x)",
            line=dict(color="#ffffff", dash="dash", width=1.5)
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal_x, y=off.predict(cal_x), mode="lines", name="Offset Calibrated",
            line=dict(color="#f59e0b", width=2)
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal_x, y=lin.predict(cal_x), mode="lines", name="Linear Calibrated",
            line=dict(color="#10b981", width=2.5)
        ))
        fig_cal.update_layout(
            xaxis_title="Low-Cost Sensor Temperature (°C)",
            yaxis_title="Reference Sensor Temperature (°C)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.warning("Please run the experiment first.")


# =====================================================================
# PAGE 6: ML MODEL
# =====================================================================
elif selected_page == "6. ML Model":
    st.title("🤖 Machine Learning Calibration (Random Forest)")
    st.markdown("Scikit-learn Random Forest Regressor learns nonlinear sensor transfer curves and environmental coupling.")

    with st.expander("⚙️ Hyperparameters & Input Features", expanded=True):
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            feat_mode = st.radio(
                "Input Features:",
                ["Single Sensor", "Multi-Variable Environmental"],
                index=0 if st.session_state.ml_config.feature_mode == "Single Sensor" else 1,
                help="Single: Low-cost sensor only. Multi-variable: Low-cost sensor + Simulated Ambient Temp + Humidity + Rate of Change."
            )
            n_trees = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        with col_ml2:
            max_depth = st.slider("Max Tree Depth (max_depth)", min_value=2, max_value=20, value=10, step=1)
            min_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=3, step=1)
            ml_seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=42)

        if st.button("🔥 TRAIN ML MODEL", type="primary", use_container_width=True):
            st.session_state.ml_config = MLConfig(
                feature_mode=feat_mode,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_split,
                random_state=int(ml_seed)
            )
            if st.session_state.clean_data is not None:
                ml_cal = MLCalibrator(st.session_state.ml_config)
                X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = ml_cal.prepare_splits(st.session_state.clean_data)
                train_meta = ml_cal.train(X_train, y_train)
                st.session_state.ml_cal = ml_cal
                st.session_state.df_train = df_train
                st.session_state.df_val = df_val
                st.session_state.df_test = df_test
                st.session_state.status_ml = "TRAINED (RF)"

                comp_df, comp_details = compare_calibration_methods(
                    test_df=df_test,
                    offset_model=st.session_state.offset_cal,
                    linear_model=st.session_state.linear_cal,
                    ml_model=ml_cal,
                    sensor_col="low_cost_sensor_clean"
                )
                st.session_state.comparison_df = comp_df
                st.session_state.comparison_details = comp_details

                budget_df, u_summary = evaluate_uncertainty(
                    y_test_ref=df_test["reference_sensor"].to_numpy(),
                    y_test_pred=comp_details["pred_ml"],
                    params=st.session_state.u_params
                )
                st.session_state.budget_df = budget_df
                st.session_state.uncertainty_summary = u_summary
                st.success("Random Forest ML model trained successfully!")
                st.rerun()

    if st.session_state.ml_cal is not None and st.session_state.ml_cal.is_trained:
        ml = st.session_state.ml_cal
        c_split1, c_split2, c_split3, c_split4 = st.columns(4)
        c_split1.metric("Training Samples (70%)", ml.train_split_info.get("train_samples", 0))
        c_split2.metric("Validation Samples (15%)", ml.train_split_info.get("val_samples", 0))
        c_split3.metric("Test Samples (15%)", ml.train_split_info.get("test_samples", 0))
        c_split4.metric("Model Training Status", "TRAINED")

        st.subheader("🌲 Feature Importance")
        feat_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in ml.feature_importances.items()
        ]).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            title="Random Forest Feature Importance",
            template="plotly_dark", color="Importance", color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(height=260)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Please click 'TRAIN ML MODEL' to train the calibrator.")

# =====================================================================
# PAGE 7: UNCERTAINTY ANALYSIS
# =====================================================================
elif selected_page == "7. Uncertainty Analysis":
    st.title("📐 Metrological Uncertainty Analysis")
    st.markdown("Quantifies measurement uncertainty using the GUM (Guide to the Expression of Uncertainty in Measurement) framework.")

    st.info(r'''
    **Uncertainty Estimation Principle**:
    - **Residual**: `residual = ML prediction - Reference`
    - **Type A Uncertainty ($u_A$)**: Statistical dispersion (residual standard deviation $s_{res}$) evaluated from test data.
    - **Type B Uncertainty ($u_B$)**: Reference sensor specification, ADC quantization, and spatial thermal gradients.
    - **Combined Standard Uncertainty ($u_c$)**: Root-Sum-Square (RSS): $u_c = \sqrt{\sum u_i^2}$
    - **Estimated Expanded Uncertainty ($U$)**: $U \approx k \times u_c$ with $k = 2$ (~95% confidence).
    *(Clearly labeled as **Estimated Expanded Uncertainty (simulation)**).*
    ''')

    if st.session_state.uncertainty_summary is not None and st.session_state.budget_df is not None:
        u_sum = st.session_state.uncertainty_summary

        u_col1, u_col2, u_col3, u_col4 = st.columns(4)
        u_col1.metric("Mean Residual", f"{u_sum['mean_residual']:+.3f} °C")
        u_col2.metric("Residual Std Dev", f"{u_sum['residual_std']:.3f} °C")
        u_col3.metric("Combined Uncertainty (uc)", f"{u_sum['u_combined']:.3f} °C")
        u_col4.metric("Expanded Uncertainty (U, k=2)", f"±{u_sum['u_expanded']:.3f} °C")

        st.subheader("📑 Uncertainty Budget Table")
        st.dataframe(st.session_state.budget_df, use_container_width=True)

        u_csv = st.session_state.budget_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Uncertainty Budget CSV",
            data=u_csv,
            file_name="uncertainty_budget.csv",
            mime="text/csv"
        )

        st.subheader("📊 Residual Error Distribution Histogram")
        y_test_pred = st.session_state.comparison_details["pred_ml"]
        y_test_ref = st.session_state.df_test["reference_sensor"].to_numpy()
        residuals = y_test_pred - y_test_ref

        fig_res = px.histogram(
            x=residuals, nbins=25,
            title="Residual Error Distribution (ML Prediction - Reference)",
            labels={"x": "Residual Error (°C)", "y": "Frequency"},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_res.add_vline(x=0.0, line_dash="dash", line_color="#ffffff")
        fig_res.update_layout(height=340)
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.warning("Please train the ML model first to evaluate uncertainty.")


# =====================================================================
# PAGE 8: PERFORMANCE COMPARISON
# =====================================================================
elif selected_page == "8. Performance Comparison":
    st.title("⚖️ Model Performance Comparison & Engineering Graphs")
    st.markdown("Compares Raw Low-Cost Sensor, Offset Calibration, Linear Calibration, and Random Forest ML on the **independent test dataset**.")

    if st.session_state.comparison_df is not None and st.session_state.df_test is not None:
        st.subheader("📊 Calibration Performance Table (Test Data)")
        st.dataframe(st.session_state.comparison_df, use_container_width=True)

        comp_csv = st.session_state.comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Performance Comparison CSV",
            data=comp_csv,
            file_name="calibration_comparison_results.csv",
            mime="text/csv"
        )

        st.info(f"💡 **Conclusion**: {st.session_state.comparison_details['conclusion']}")

        test_df = st.session_state.df_test.copy().sort_values("time_s")
        y_ref = test_df["reference_sensor"].to_numpy()
        y_raw = test_df["low_cost_sensor_clean"].to_numpy()
        y_ml = st.session_state.comparison_details["pred_ml"]

        # 5 GRAPHS
        st.markdown("### 📈 Engineering Graphs")

        # Graph 1 & 2
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### Graph 1: Reference vs Raw Low-Cost Sensor")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=test_df["time_s"], y=y_raw, name="Raw Low-Cost Sensor", line=dict(color="#f87171", dash="dot")))
            fig1.add_trace(go.Scatter(x=test_df["time_s"], y=y_ref, name="Reference Sensor", line=dict(color="#10b981", width=2.5)))
            fig1.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)", template="plotly_dark", height=340)
            st.plotly_chart(fig1, use_container_width=True)

        with g_col2:
            st.markdown("#### Graph 2: Sensor Error Before Calibration")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=test_df["time_s"], y=(y_raw - y_ref), name="Raw Error", line=dict(color="#ef4444", width=2)))
            fig2.add_hline(y=0.0, line_dash="dash", line_color="#ffffff")
            fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Error (°C)", template="plotly_dark", height=340)
            st.plotly_chart(fig2, use_container_width=True)

        # Graph 3
        st.markdown("#### Graph 3: Reference vs Raw vs ML-Corrected Temperature")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=test_df["time_s"], y=y_raw, name="Raw Low-Cost Sensor", line=dict(color="#f87171", dash="dot")))
        fig3.add_trace(go.Scatter(x=test_df["time_s"], y=y_ref, name="Reference Sensor", line=dict(color="#10b981", width=3)))
        fig3.add_trace(go.Scatter(x=test_df["time_s"], y=y_ml, name="ML Corrected", line=dict(color="#38bdf8", width=2.5)))
        fig3.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)", template="plotly_dark", height=380)
        st.plotly_chart(fig3, use_container_width=True)

        # Graph 4 & 5
        g_col3, g_col4 = st.columns(2)
        with g_col3:
            st.markdown("#### Graph 4: Prediction Error Distribution")
            ml_errs = y_ml - y_ref
            fig4 = px.histogram(x=ml_errs, nbins=20, labels={"x": "ML Prediction - Reference (°C)", "y": "Count"}, template="plotly_dark", color_discrete_sequence=["#38bdf8"])
            fig4.add_vline(x=0.0, line_dash="dash", line_color="#ffffff")
            fig4.update_layout(height=340)
            st.plotly_chart(fig4, use_container_width=True)

        with g_col4:
            st.markdown("#### Graph 5: Predicted vs Reference Temperature (y = x Line)")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=y_ref, y=y_ml, mode="markers", name="ML Predicted", marker=dict(color="#38bdf8", size=6, opacity=0.8)))
            min_v, max_v = min(y_ref.min(), y_ml.min()), max(y_ref.max(), y_ml.max())
            fig5.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal y = x", line=dict(color="#f59e0b", dash="dash", width=2)))
            fig5.update_layout(xaxis_title="Reference Temperature (°C)", yaxis_title="Predicted Temperature (°C)", template="plotly_dark", height=340)
            st.plotly_chart(fig5, use_container_width=True)

        # SUMMARY TECHNICAL REPORT
        st.markdown("---")
        st.subheader("📄 Printable Technical Summary Report")
        report_text = f'''================================================================================
ENGINEERING TECHNICAL REPORT: ML-BASED SENSOR CALIBRATION SYSTEM
================================================================================
Timestamp: {pd.Timestamp.now()}
Prototype State: Software Simulation (Hardware Integration Pending)
Temperature Profile: {st.session_state.sensor_params.profile}
Operating Range: {st.session_state.sensor_params.t_min:.1f} °C to {st.session_state.sensor_params.t_max:.1f} °C
Total Experiment Samples: {len(st.session_state.raw_data)}
Test Evaluation Samples: {len(test_df)}

PERFORMANCE COMPARISON (TEST DATASET):
--------------------------------------------------------------------------------
1. Raw Low-Cost Sensor:
   - MAE:  {st.session_state.comparison_details['raw_metrics']['MAE']:.4f} °C
   - RMSE: {st.session_state.comparison_details['raw_metrics']['RMSE']:.4f} °C
   - Max Error: {st.session_state.comparison_details['raw_metrics']['Max Absolute Error']:.4f} °C

2. Conventional Offset Calibration:
   - Formula: {st.session_state.offset_cal.formula}
   - MAE:  {st.session_state.comparison_details['offset_metrics']['MAE']:.4f} °C
   - RMSE: {st.session_state.comparison_details['offset_metrics']['RMSE']:.4f} °C

3. Conventional Linear Regression:
   - Formula: {st.session_state.linear_cal.formula}
   - MAE:  {st.session_state.comparison_details['linear_metrics']['MAE']:.4f} °C
   - RMSE: {st.session_state.comparison_details['linear_metrics']['RMSE']:.4f} °C

4. Random Forest ML Calibration:
   - Features: {st.session_state.ml_cal.feature_names}
   - Trees: {st.session_state.ml_config.n_estimators}, Depth: {st.session_state.ml_config.max_depth}
   - MAE:  {st.session_state.comparison_details['ml_metrics']['MAE']:.4f} °C
   - RMSE: {st.session_state.comparison_details['ml_metrics']['RMSE']:.4f} °C
   - Max Error: {st.session_state.comparison_details['ml_metrics']['Max Absolute Error']:.4f} °C
   - Error Reduction: {st.session_state.comparison_details['mae_improvement_pct']}% MAE reduction over raw sensor

UNCERTAINTY ANALYSIS (GUM / ISO 98-3):
--------------------------------------------------------------------------------
- Combined Standard Uncertainty (uc): {st.session_state.uncertainty_summary['u_combined']:.4f} °C
- Coverage Factor (k): 2.0 (~95% confidence level)
- Estimated Expanded Uncertainty (U): ±{st.session_state.uncertainty_summary['u_expanded']:.4f} °C

CONCLUSION:
{st.session_state.comparison_details['conclusion']}
================================================================================
'''
        st.text_area("Generated Technical Report", report_text, height=250)
        st.download_button(
            label="📥 Download Technical Report TXT",
            data=report_text,
            file_name="calibration_technical_report.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please run the experiment and train the ML model first.")

# =====================================================================
# PAGE 9: SYSTEM ARCHITECTURE
# =====================================================================
elif selected_page == "9. System Architecture":
    st.title("🏗️ System Architecture")
    st.markdown("### Status: <span class='badge-simulated'>Software Simulation — Hardware Integration Pending</span>", unsafe_allow_html=True)

    st.markdown('''
    ```
    Physical Temperature
            │
            ├───> Low-Cost Sensor (DS18B20 / NTC)
            │           │
            └───> Reference Sensor (Calibrated Pt100)
                        │
                        ▼
            ESP32 / Data Acquisition
                        │
                        ▼
                   Data Storage
                        │
                        ▼
                   Preprocessing (MAD Outlier Rejection, Filtering)
                        │
                        ▼
            Conventional Calibration + ML Calibration (Random Forest)
                        │
                        ▼
            Performance Comparison (MAE, RMSE, Max Error)
                        │
                        ▼
            Uncertainty Evaluation (GUM Budget, k=2)
                        │
                        ▼
            Corrected Temperature + Uncertainty (T ± U °C)
                        │
                        ▼
            OLED / PC / Web Dashboard
    ```
    ''')

    st.info('''
    **Architectural Highlights**:
    - **Decoupled Data Source**: The interface between acquisition and ML consumes standard data arrays, allowing the software simulator to be swapped for real serial/USB/WiFi telemetry from an ESP32 micro-controller without altering calibration logic.
    - **Self-Calibration Workflow**: Calibration transfer functions are derived automatically from reference measurement pairs.
    ''')


# =====================================================================
# PAGE 10: HARDWARE INTEGRATION
# =====================================================================
elif selected_page == "10. Hardware Integration":
    st.title("🔌 Hardware Integration Roadmap")
    st.markdown("Roadmap to migrate from the current **Software Simulation Prototype** to the **Physical Hardware Prototype**.")

    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        st.subheader("🖥️ Current Prototype")
        st.markdown(r'''
        - **Status**: Software simulation.
        - **Transducer Physics**: Models thermal response lag ($\tau$), static offset, gain sensitivity error, quadratic curvature, and measurement jitter.
        - **Host Platform**: Laptop / PC running Python, Streamlit, and Scikit-learn.
        - **Purpose**: Academic verification, model training, and viva demonstration.
        ''')

    with col_hw2:
        st.subheader("🛠️ Future Prototype")
        st.markdown(r'''
        - **Target Microcontroller**: **ESP32** (Dual-core 240 MHz, 12-bit ADC, built-in WiFi/BLE).
        - **Low-Cost Sensor**: **DS18B20** digital temperature sensor (1-Wire protocol) or **NTC Thermistor**.
        - **Reference Sensor**: **Class A Pt100 RTD** with MAX31865 precision amplifier (±0.05 °C accuracy).
        - **Auxiliary Sensor**: **BME280** (I2C) for ambient temperature and relative humidity.
        - **Display**: 0.96-inch I2C **OLED display** showing $T_{corr} \pm U$.
        - **Host Link**: USB Serial UART or MQTT over WiFi.
        ''')

    st.markdown("---")
    st.subheader("🔌 Proposed Hardware Circuit Schematic")
    st.code('''
    [ ESP32 Microcontroller ]
         │
         ├── GPIO 4  ──────> DS18B20 1-Wire Data (4.7kΩ pull-up to 3.3V)
         │
         ├── GPIO 21 (SDA) ─> BME280 SDA + 0.96" OLED SDA
         ├── GPIO 22 (SCL) ─> BME280 SCL + 0.96" OLED SCL
         │
         ├── SPI Pins ─────> MAX31865 RTD Amplifier (CS: GPIO 5, MOSI: 23, MISO: 19, SCK: 18)
         │                     └──> 4-Wire Pt100 Reference Probe
         │
         └── USB / UART ───> Host PC Running Python ML Calibration Engine
    ''', language="text")

    st.success('''
    **Future Integration Workflow**:
    1. ESP32 will replace the simulated data source.
    2. DS18B20 and Pt100 readings will stream over serial JSON packets.
    3. The trained Random Forest model will process real sensor readings in real-time.
    ''')


# --- FOOTER ---
st.markdown("---")
st.caption("🔬 **ML-Based Self-Calibrating & Uncertainty-Aware Measurement System** | Software Simulation Prototype")
