import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# ---------------------------
# Load models (PIPELINES)
# ---------------------------
@st.cache_resource
def load_models():
    return {
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

models = load_models()

# ---------------------------
# Column order (MUST MATCH TRAINING)
# ---------------------------
FEATURE_COLS = joblib.load(ARTIFACT_DIR / "feature_names.joblib")

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
**Scale meaning**
- 0 — Never  
- 1 — Almost Never  
- 2 — Sometimes  
- 3 — Fairly Often  
- 4 — Very Often  
""")

user_input = {}

for col in FEATURE_COLS:
    user_input[col] = st.selectbox(
        col,
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — Never",
            1: "1 — Almost Never",
            2: "2 — Sometimes",
            3: "3 — Fairly Often",
            4: "4 — Very Often",
        }[x],
    )

# ---------------------------
# Predict
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    input_df = pd.DataFrame([user_input])

    st.subheader("📊 Prediction Result")

    for target, (m1, m2) in models.items():
        # ⚠️ DO NOT PREPROCESS — PIPELINE DOES IT
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(
            label=f"{target} Risk",
            value=f"{avg*100:.2f}%",
        )
