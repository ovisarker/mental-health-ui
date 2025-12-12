# ============================
# Student Mental Health UI
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    preprocess = joblib.load(ARTIFACT_DIR / "preprocess_pipeline.joblib")

    models = {
        "Anxiety": (
            joblib.load(ARTIFACT_DIR / "Anxiety_model_A.joblib"),
            joblib.load(ARTIFACT_DIR / "Anxiety_model_B.joblib"),
        ),
        "Stress": (
            joblib.load(ARTIFACT_DIR / "Stress_model_A.joblib"),
            joblib.load(ARTIFACT_DIR / "Stress_model_B.joblib"),
        ),
        "Depression": (
            joblib.load(ARTIFACT_DIR / "Depression_model_A.joblib"),
            joblib.load(ARTIFACT_DIR / "Depression_model_B.joblib"),
        ),
    }
    return preprocess, models


@st.cache_resource
def load_schema():
    with open(ARTIFACT_DIR / "schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


preprocess, models = load_artifacts()
schema = load_schema()

demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
all_cols = demo_cols + q_cols

# ----------------------------
# UI
# ----------------------------
st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.caption(
    "Predicts Anxiety / Stress / Depression as Present or Absent with confidence (%)"
)

# ----------------------------
# Demographics
# ----------------------------
st.subheader("👤 Demographic Information")

user_input = {}

user_input["1. Age"] = st.number_input("Age", min_value=15, max_value=60, value=20)
user_input["2. Gender"] = st.selectbox("Gender", ["Male", "Female"])
user_input["3. University"] = st.text_input("University", "Unknown")
user_input["4. Department"] = st.text_input("Department", "CSE")
user_input["5. Academic Year"] = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
user_input["6. Current CGPA"] = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
user_input["7. Did you receive a waiver or scholarship at your university?"] = st.selectbox(
    "Scholarship / Waiver", ["Yes", "No"]
)

# ----------------------------
# Questionnaire
# ----------------------------
st.subheader("📝 Questionnaire")

scale_map = {
    0: "Never",
    1: "Sometimes",
    2: "Often",
    3: "Almost Always"
}

for q in q_cols:
    val = st.selectbox(
        q,
        options=list(scale_map.keys()),
        format_func=lambda x: f"{x} — {scale_map[x]}"
    )
    user_input[q] = val

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Mental Health Risk"):

    # 1️⃣ Build RAW dataframe (NO preprocessing here)
    input_df = pd.DataFrame([{c: user_input.get(c, np.nan) for c in all_cols}])

    # 🔐 Critical fix
    input_df = input_df.reindex(columns=preprocess.feature_names_in_)

    # 2️⃣ Preprocess (ONLY via pipeline)
    X_trans = preprocess.transform(input_df)

    st.subheader("📊 Prediction Result")

    scores = {}

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(X_trans)[0][1]
        p2 = m2.predict_proba(X_trans)[0][1]
        avg = (p1 + p2) / 2

        scores[target] = {
            "status": "Present" if avg >= 0.5 else "Absent",
            "percent": round(avg * 100, 2)
        }

    dominant = max(scores.items(), key=lambda x: x[1]["percent"])

    st.markdown("---")
    for k, v in scores.items():
        st.write(f"**{k}** : {v['status']} ({v['percent']}%)")

    st.markdown("---")
    st.success(
        f"🚨 Dominant Condition: **{dominant[0]} ({dominant[1]['percent']}%)**"
    )
