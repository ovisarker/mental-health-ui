import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Student Mental Health Risk Prediction", layout="wide")

BASE_DIR = Path(__file__).parent

# -------------------------------
# Load schema
# -------------------------------
with open(BASE_DIR / "schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
all_cols = demo_cols + q_cols

# -------------------------------
# Load artifacts (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    preprocess = joblib.load(BASE_DIR / "preprocess_pipeline.joblib")

    models = {
        "Anxiety": (
            joblib.load(BASE_DIR / "Anxiety_model_A.joblib"),
            joblib.load(BASE_DIR / "Anxiety_model_B.joblib")
        ),
        "Stress": (
            joblib.load(BASE_DIR / "Stress_model_A.joblib"),
            joblib.load(BASE_DIR / "Stress_model_B.joblib")
        ),
        "Depression": (
            joblib.load(BASE_DIR / "Depression_model_A.joblib"),
            joblib.load(BASE_DIR / "Depression_model_B.joblib")
        )
    }
    return preprocess, models

preprocess, models = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.write("Answer all questions honestly. Scale: **0 (Never) – 3 (Very Often)**")

user_input = {}

st.subheader("Demographic Information")
for col in demo_cols:
    user_input[col] = st.text_input(col)

st.subheader("Questionnaire")
for col in q_cols:
    user_input[col] = st.selectbox(col, [0, 1, 2, 3])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Mental Health Risk"):
    input_df = pd.DataFrame([{col: user_input[col] for col in all_cols}])

    X_trans = preprocess.transform(input_df)

    st.subheader("📊 Prediction Result")

    risk_scores = {}

    for target, (mA, mB) in models.items():
        pA = mA.predict_proba(X_trans)
        pB = mB.predict_proba(X_trans)

        avg_prob = (pA + pB) / 2
        pred = np.argmax(avg_prob[0])
        conf = avg_prob[0][pred] * 100

        status = "Present" if pred == 1 else "Absent"
        risk_scores[target] = conf

        st.write(f"**{target}**: {status} ({conf:.2f}%)")

    dominant = max(risk_scores, key=risk_scores.get)

    st.markdown("---")
    st.success(f"🚨 **Dominant Condition:** {dominant} ({risk_scores[dominant]:.2f}%)")
