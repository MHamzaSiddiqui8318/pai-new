"""
CT-264 Programming for AI - CCA Project
AI-Based Patient Admission Prediction System - Streamlit Web Interface

Streamlit version of 03_admission_app.py for online deployment.
Run locally:   streamlit run streamlit_app.py
Deploy:        Push to GitHub → connect to Streamlit Community Cloud
"""

import os
import json
import random

import streamlit as st
import joblib
import numpy as np
import pandas as pd

from admission_utils import (FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                             clean_patient_flow, calibrate_probability, risk_band)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

CONDITIONS = [("hypertension", "Hypertension"), ("diabetes", "Diabetes"),
              ("heart_failure", "Heart failure"), ("coronary_disease", "Coronary disease"),
              ("arrhythmia", "Arrhythmia"), ("copd_asthma", "COPD / Asthma"),
              ("kidney_disease", "Kidney disease"), ("cancer_history", "Cancer history"),
              ("mental_health_disorder", "Mental health disorder"),
              ("substance_use_disorder", "Substance use disorder")]

VITALS = [("heart_rate", "Heart rate (bpm)", 20, 250),
          ("systolic_bp", "Systolic BP (mmHg)", 50, 300),
          ("diastolic_bp", "Diastolic BP (mmHg)", 20, 200),
          ("respiratory_rate", "Resp. rate (/min)", 5, 60),
          ("oxygen_saturation", "SpO₂ (%)", 50, 100),
          ("temperature_c", "Temperature (°C)", 32.0, 43.0)]

HISTORY = [("prior_ed_visits", "ED visits (last year)"),
           ("prior_admissions", "Admissions (last year)"),
           ("prior_surgeries", "Previous surgeries"),
           ("prior_imaging_tests", "Previous imaging tests"),
           ("prior_ekgs", "Previous ECGs"),
           ("num_comorbidities", "No. of past conditions")]

BAND_COLORS = {"HIGH": "#c0392b", "MODERATE": "#d68910", "LOW": "#1e8449"}
BAND_EMOJI = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}


# ── Load model (cached so it only loads once) ──────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "admission_model.joblib"))
    with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
        info = json.load(f)
    demo = pd.read_csv(os.path.join(MODEL_DIR, "demo_patients.csv"))
    return model, info, demo


def predict_patient(model, info, patient):
    """Run prediction on a single patient dict."""
    row = pd.DataFrame([{c: patient.get(c, np.nan) for c in FEATURES}])
    row, _ = clean_patient_flow(row, info["top_complaints"])
    score = float(model.predict_proba(row[FEATURES])[:, 1][0])
    probability = float(calibrate_probability(score, info["prior_ratio"]))
    band, action = risk_band(score, info["threshold"])
    return {"score": score, "probability": probability, "band": band,
            "action": action, "admit": score >= info["threshold"]}


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="Patient Admission Prediction",
                   page_icon="🏥", layout="wide")

model, info, demo_df = load_model()
cats = info["categories"]
defaults = info["defaults"]

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("# 🏥 AI-Based Patient Admission Prediction System")
st.markdown(f"Emergency department triage decision support &nbsp;|&nbsp; "
            f"**Model:** {info['model_name']} &nbsp;|&nbsp; "
            f"**Test ROC-AUC:** {info['test_metrics']['ROC-AUC']:.3f}")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩺 Predict Admission", "📊 Model Performance", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT ADMISSION
# ══════════════════════════════════════════════════════════════════════
with tab1:

    # --- Load random test patient button ---
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        load_random = st.button("🎲 Load Random Test Patient", use_container_width=True)
    with btn_col2:
        st.caption("Fills the form with a real patient from the unseen test set and shows the true outcome.")

    if load_random:
        rec = demo_df.iloc[random.randrange(len(demo_df))].to_dict()
        st.session_state["demo_patient"] = rec
        st.session_state["actual_admitted"] = int(rec.get("actual_admitted", 0))
    else:
        if "demo_patient" not in st.session_state:
            st.session_state["demo_patient"] = None
            st.session_state["actual_admitted"] = None

    demo_rec = st.session_state.get("demo_patient")

    def get_default(key, fallback=""):
        """Get value from demo patient or defaults."""
        if demo_rec and key in demo_rec and pd.notna(demo_rec[key]):
            return demo_rec[key]
        return defaults.get(key, fallback)

    def get_default_index(key, options):
        """Get index for selectbox from demo patient or defaults."""
        val = str(get_default(key, options[0]))
        if val in options:
            return options.index(val)
        return 0

    # --- Form columns ---
    col1, col2, col3 = st.columns(3)

    # ── Column 1: Patient + Arrival ──
    with col1:
        st.subheader("👤 Patient")
        age = st.number_input("Age (years)", min_value=0, max_value=120,
                              value=int(get_default("age", 45)), step=1)
        gender = st.selectbox("Gender", cats["gender"],
                              index=get_default_index("gender", cats["gender"]))
        insurance = st.selectbox("Insurance", cats["insurance_status"],
                                 index=get_default_index("insurance_status", cats["insurance_status"]))
        marital = st.selectbox("Marital status", cats["marital_status"],
                               index=get_default_index("marital_status", cats["marital_status"]))
        employment = st.selectbox("Employment", cats["employment_status"],
                                  index=get_default_index("employment_status", cats["employment_status"]))
        race = st.selectbox("Race", cats["race"],
                            index=get_default_index("race", cats["race"]))
        ethnicity = st.selectbox("Ethnicity", cats["ethnicity"],
                                 index=get_default_index("ethnicity", cats["ethnicity"]))
        language = st.selectbox("Language", cats["language"],
                                index=get_default_index("language", cats["language"]))

        st.subheader("🚑 Arrival & Triage")
        esi_options = ["1", "2", "3", "4", "5"]
        esi_val = str(int(get_default("esi_level", 3)))
        esi = st.selectbox("Triage level (ESI)", esi_options,
                           index=esi_options.index(esi_val) if esi_val in esi_options else 2)
        complaint = st.selectbox("Chief complaint", cats["chief_complaint"],
                                 index=get_default_index("chief_complaint", cats["chief_complaint"]))
        arrival_mode = st.selectbox("Arrival mode", cats["arrival_mode"],
                                    index=get_default_index("arrival_mode", cats["arrival_mode"]))
        ed_site = st.selectbox("ED site", cats["ed_site"],
                               index=get_default_index("ed_site", cats["ed_site"]))
        arrival_day = st.selectbox("Arrival day", cats["arrival_day"],
                                   index=get_default_index("arrival_day", cats["arrival_day"]))
        arrival_time = st.selectbox("Arrival time", cats["arrival_hour_bin"],
                                    index=get_default_index("arrival_hour_bin", cats["arrival_hour_bin"]))
        arrival_month = st.selectbox("Arrival month", cats["arrival_month"],
                                     index=get_default_index("arrival_month", cats["arrival_month"]))

    # ── Column 2: Vitals + History ──
    with col2:
        st.subheader("💓 Triage Vital Signs")
        st.caption("Leave blank (set to 0) if not measured — the model handles missing values.")
        vitals = {}
        for key, label, low, high in VITALS:
            default_val = get_default(key, defaults.get(key, 0))
            if pd.isna(default_val):
                default_val = 0.0
            vitals[key] = st.number_input(label, min_value=0.0, max_value=float(high * 2),
                                          value=float(default_val), step=1.0,
                                          format="%.1f" if key == "temperature_c" else "%.0f")

        supp_o2_default = bool(int(get_default("on_supplemental_oxygen", 0))) if pd.notna(get_default("on_supplemental_oxygen", 0)) else False
        supp_o2 = st.checkbox("On supplemental oxygen", value=supp_o2_default)

        st.subheader("📋 Hospital History")
        history = {}
        for key, label in HISTORY:
            default_val = get_default(key, 0)
            if pd.isna(default_val):
                default_val = 0
            history[key] = st.number_input(label, min_value=0, max_value=999,
                                           value=int(float(default_val)), step=1)
        prev_dispo = st.selectbox("Previous ED outcome", cats["previous_disposition"],
                                  index=get_default_index("previous_disposition", cats["previous_disposition"]))

    # ── Column 3: Conditions + Predict ──
    with col3:
        st.subheader("🏷️ Known Conditions")
        conditions = {}
        for key, label in CONDITIONS:
            default_cond = bool(int(get_default(key, 0))) if pd.notna(get_default(key, 0)) else False
            conditions[key] = st.checkbox(label, value=default_cond)

        st.divider()

        # ── PREDICT BUTTON ──
        predict_clicked = st.button("🔍 **Predict Admission**", type="primary",
                                     use_container_width=True)

    # ── Run prediction ──
    if predict_clicked:
        patient = {
            "age": float(age), "gender": gender, "insurance_status": insurance,
            "marital_status": marital, "employment_status": employment,
            "race": race, "ethnicity": ethnicity, "language": language,
            "esi_level": float(esi), "chief_complaint": complaint,
            "arrival_mode": arrival_mode, "ed_site": ed_site,
            "arrival_day": arrival_day, "arrival_hour_bin": arrival_time,
            "arrival_month": arrival_month, "previous_disposition": prev_dispo,
            "on_supplemental_oxygen": int(supp_o2),
        }
        # Vitals: treat 0 as not measured (NaN)
        for key, label, low, high in VITALS:
            val = vitals[key]
            patient[key] = np.nan if val == 0 else float(val)
        # History
        for key, label in HISTORY:
            patient[key] = float(history[key])
        # Conditions
        for key, label in CONDITIONS:
            patient[key] = int(conditions[key])

        result = predict_patient(model, info, patient)

        # ── Display result ──
        st.divider()
        band = result["band"]
        color = BAND_COLORS[band]
        emoji = BAND_EMOJI[band]

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f'<div style="background-color:{color}; color:white; padding:30px; '
                f'border-radius:12px; text-align:center;">'
                f'<h1 style="margin:0; color:white;">{emoji} {band}</h1>'
                f'<h3 style="margin:5px 0 0 0; color:white;">'
                f'{"ADMISSION PREDICTED" if result["admit"] else "DISCHARGE PREDICTED"}'
                f'</h3></div>', unsafe_allow_html=True)

        with r2:
            st.metric("Model Risk Score", f'{result["score"]:.2f}',
                       delta=None)
            st.progress(min(result["score"], 1.0))
            st.markdown(f"**Estimated chance of admission:** {result['probability']:.0%}")
            st.info(f"**Recommended action:** {result['action']}")

            # Show actual outcome if demo patient
            actual = st.session_state.get("actual_admitted")
            if actual is not None:
                correct = bool(actual) == result["admit"]
                actual_text = "ADMITTED" if actual else "DISCHARGED"
                icon = "✅" if correct else "❌"
                st.markdown(f"{icon} **Real outcome:** {actual_text} "
                            f"(*{'model correct' if correct else 'model wrong'}*)")

        st.caption(f"Decision threshold: {info['threshold']:.2f} · "
                   f"Score ≥ threshold → admission predicted · "
                   f"Probability is corrected for class weighting used in training.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Dataset 1 — USA, 20,000 Test Visits")
    try:
        df_comp = pd.read_csv(os.path.join(METRIC_DIR, "model_comparison.csv"))
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning("model_comparison.csv not found.")

    st.subheader("📈 Dataset 2 — Korea KTAS, 5-Fold Cross-Validation")
    try:
        df_ktas = pd.read_csv(os.path.join(METRIC_DIR, "ktas_model_comparison.csv"))
        st.dataframe(df_ktas, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning("ktas_model_comparison.csv not found.")

    st.divider()
    st.subheader("📊 Charts")

    if os.path.exists(FIG_DIR):
        figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
        if figs:
            selected_fig = st.selectbox("Select a chart:", figs,
                                         index=11 if len(figs) > 11 else 0)
            st.image(os.path.join(FIG_DIR, selected_fig), use_container_width=True)
    else:
        st.warning("Figures directory not found.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown(f"""
**CT-264 Programming for Artificial Intelligence — Complex Computing Activity**
NED University of Engineering and Technology, Department of CSIT

### 👥 Group Members
| Roll No. | Name |
|---|---|
| CT-24033 | Muhammad Hamza Siddiqui |
| CT-24044 | Saif Ur Rehman |
| CT-24312 | Muhammad Talha Bin Suhail |

### 🎯 Purpose
Predict at triage whether an emergency patient will be admitted, so beds and
staff can be prepared earlier and ED crowding is reduced.

### 📂 Datasets
1. **ED Patient Flow** — Yale New Haven Health (Hong et al., PLOS ONE 2018),
   100,000 visits sampled from 560,486 visits.
2. **Korean Triage and Acuity Scale** (Moon et al., PLOS ONE 2019), 1,267 visits.

### 🤖 Final Model
**{info['model_name']}** trained on **{info['train_size']:,}** visits,
tested on **{info['test_size']:,}** unseen visits
(ROC-AUC **{info['test_metrics']['ROC-AUC']:.3f}**,
Recall **{info['test_metrics']['Recall (sensitivity)']:.3f}**).

### ⚠️ Disclaimer
Decision-support prototype for education only. It must not replace
clinical judgement and would need local validation before real use.
""")

