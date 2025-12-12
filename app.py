import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------------------------
# Load models
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
# Load UI questions
# ---------------------------
with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    UI_FEATURES = json.load(f)  # ✅ LIST ONLY

# ---------------------------
# Get true model feature order
# ---------------------------
sample_model = list(models.values())[0][0]
MODEL_FEATURES = list(sample_model.feature_names_in_)

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
### Response Scale
- Never
- Almost Never
- Sometimes
- Fairly Often
- Very Often
""")

SCALE = {
    "Never": 0,
    "Almost Never": 1,
    "Sometimes": 2,
    "Fairly Often": 3,
    "Very Often": 4,
}

user_input = {}

for question in UI_FEATURES:
    choice = st.selectbox(question, list(SCALE.keys()))
    user_input[question] = SCALE[choice]

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    raw_df = pd.DataFrame([user_input])

    # Align with model
    input_df = raw_df.reindex(columns=MODEL_FEATURES, fill_value=0)

    st.subheader("📊 Prediction Results")

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(target, f"{avg * 100:.2f}%")
