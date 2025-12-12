# ==========================================
# Student Mental Health Risk Prediction UI
# Hybrid ML (Deployment Only)
# ==========================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ------------------------------------------
# App Config
# ------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Risk Prediction (Hybrid ML)",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.caption(
    "Outputs Anxiety / Stress / Depression as Present or Absent with % confidence "
    "and identifies the dominant condition."
)

# ------------------------------------------
# Paths (ROOT SAFE)
# ------------------------------------------
BASE_DIR = Path(__file__).parent

# ------------------------------------------
# Load Schema
# ------------------------------------------
def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

schema = load_schema(BASE_DIR / "schema.json")

demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
numeric_demo = set(schema.get("numeric_demo", []))

# ------------------------------------------
# Load Models & Preprocess Pipeline
# ------------------------------------------
@st.cache_resource
def load_artifacts(base_dir: Path):
    preprocess = joblib.load(base_dir / "preprocess_pipeline.joblib")

    models = {
        "Anxiety": (
            joblib.load(base_dir / "Anxiety_model_A.joblib"),
            joblib.load(base_dir / "Anxiety_model_B.joblib"),
        ),
        "Stress": (
            joblib.load(base_dir / "Stress_model_A.joblib"),
            joblib.load(base_dir / "Stress_model_B.joblib"),
        ),
        "Depression": (
            joblib.load(base_dir / "Depression_model_A.joblib"),
            joblib.load(base_dir / "Depression_model_B.joblib"),
        ),
    }
    return preprocess, models

preprocess, models = load_artifacts(BASE_DIR)

# ------------------------------------------
# UI – Demographics
# ------------------------------------------
st.subheader("👤 Demographic Information")

user_input = {}

for col in demo_cols:
    if col in numeric_demo:
        user_input[col] = st.number_input(col, value=0.0)
    else:
        user_input[col] = st.text_input(col)

# ------------------------------------------
# UI – Questionnaire (33 questions)
# ------------------------------------------
st.subheader("📝 Mental Health Questionnaire")
st.caption("Answer each question on a scale similar to the original survey (0–4).")

for q in q_cols:
    user_input[q] = st.selectbox(
        q,
        options=[0, 1, 2, 3, 4],
        index=0
    )

# ------------------------------------------
# Prediction Logic
# ------------------------------------------
def hybrid_predict(row_df: pd.DataFrame, modelA, modelB):
    p1 = modelA.predict_proba(row_df)[0]
    p2 = modelB.predict_proba(row_df)[0]
    avg = (p1 + p2) / 2.0

    classes = modelA.classes_
    idx = int(np.argmax(avg))
    return classes[idx], float(avg[idx] * 100)

# ------------------------------------------
# Run Prediction
# ------------------------------------------
if st.button("🔍 Analyze Mental Health Risk"):
    input_df = pd.DataFrame([user_input])

    # Preprocess
    X = preprocess.transform(input_df)

    st.subheader("📊 Prediction Results")

    results = {}
    dominance = []

    for target, (mA, mB) in models.items():
        label, confidence = hybrid_predict(X, mA, mB)
        status = "Present" if label == 1 else "Absent"

        results[target] = {
            "status": status,
            "confidence": confidence,
            "models": f"{type(mA).__name__} + {type(mB).__name__}"
        }

        dominance.append((target, confidence))

        st.write(
            f"**{target}** : {status}  |  "
            f"{confidence:.2f}%  |  Hybrid: {results[target]['models']}"
        )

    dom_target, dom_conf = max(dominance, key=lambda x: x[1])

    st.markdown("---")
    st.error(f"🚨 **Dominant Condition:** {dom_target} ({dom_conf:.2f}%)")

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Hybrid ML system | Research & Deployment Ready")
