import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------------------------
# Paths (point to repo root)
# ---------------------------
BASE_DIR = Path(__file__).parent

# ---------------------------
# Load models (full pipelines)
# ---------------------------
@st.cache_resource
def load_models():
    return {
        "Anxiety": (
            joblib.load(BASE_DIR / "Anxiety_model_A.joblib"),
            joblib.load(BASE_DIR / "Anxiety_model_B.joblib"),
        ),
        "Stress": (
            joblib.load(BASE_DIR / "Stress_model_A.joblib"),
            joblib.load(BASE_DIR / "Stress_model_B.joblib"),
        ),
        "Depression": (
            joblib.load(BASE_DIR / "Depression_model_A.joblib"),
            joblib.load(BASE_DIR / "Depression_model_B.joblib"),
        ),
    }

models = load_models()

# ---------------------------
# Load expected feature names (33)
# ---------------------------
with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

# Response scale labels for radio buttons
response_labels = {
    0: "0 — Never",
    1: "1 — Almost Never",
    2: "2 — Sometimes",
    3: "3 — Fairly Often",
    4: "4 — Very Often"
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
This tool predicts **Anxiety**, **Stress**, and **Depression** risks based on your self-reported responses.
Please answer all 33 items below (7 demographics + 26 semester questions).
""")

# Collect user input
user_input = {}
for col in FEATURE_COLS:
    # If it's a demographic field, provide a text or numeric input
    if col in ["Age", "Current CGPA"]:
        user_input[col] = st.number_input(col, min_value=0.0, max_value=100.0, value=0.0)
    elif col in ["Gender"]:
        user_input[col] = st.selectbox(col, options=["Male", "Female", "Non-binary", "Prefer not to say"])
    elif col in ["University", "Department", "Academic Year", "Received Waiver or Scholarship"]:
        user_input[col] = st.text_input(col)
    else:
        # All other columns are Likert-scale questions (0–4)
        user_input[col] = st.selectbox(col, options=[0, 1, 2, 3, 4], format_func=lambda x: response_labels[x])

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    # Build DataFrame in correct order
    df = pd.DataFrame([user_input])
    input_df = df.reindex(columns=FEATURE_COLS)
    
    st.subheader("📊 Prediction Results")
    for target, (m1, m2) in models.items():
        # Make sure we transform input_df through each model pipeline
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(f"{target} Risk", f"{avg * 100:.2f}%")
