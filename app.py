import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Student Mental Health Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Student Mental Health Risk Predictor")

# ---------------------------
# Load Models (PIPELINES)
# ---------------------------
BASE_DIR = Path(__file__).parent

models = {
    "Depression": (
        joblib.load(BASE_DIR / "depression_model_1.joblib"),
        joblib.load(BASE_DIR / "depression_model_2.joblib"),
    ),
    "Anxiety": (
        joblib.load(BASE_DIR / "anxiety_model_1.joblib"),
        joblib.load(BASE_DIR / "anxiety_model_2.joblib"),
    ),
    "Stress": (
        joblib.load(BASE_DIR / "stress_model_1.joblib"),
        joblib.load(BASE_DIR / "stress_model_2.joblib"),
    ),
}

# ---------------------------
# Feature Names (MUST MATCH TRAINING)
# ---------------------------
FEATURES = models["Depression"][0].feature_names_in_

# ---------------------------
# Value Meaning (UI clarity)
# ---------------------------
SCALE = {
    "0 — Never": 0,
    "1 — Almost Never": 1,
    "2 — Sometimes": 2,
    "3 — Fairly Often": 3,
    "4 — Very Often": 4,
}

# ---------------------------
# Questionnaire UI
# ---------------------------
st.subheader("📋 Questionnaire")

user_row = {}

for col in FEATURES:
    user_row[col] = st.selectbox(
        col,
        list(SCALE.keys()),
        index=0
    )

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    # Convert to numeric
    for k in user_row:
        user_row[k] = SCALE[user_row[k]]

    input_df = pd.DataFrame([user_row])

    st.subheader("📊 Prediction Result")

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        if avg < 0.33:
            level = "🟢 Low Risk"
        elif avg < 0.66:
            level = "🟡 Moderate Risk"
        else:
            level = "🔴 High Risk"

        st.metric(
            label=target,
            value=f"{level}",
            delta=f"{avg*100:.1f}% probability"
        )
