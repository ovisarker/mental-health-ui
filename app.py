import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------------------------
# Base directory (ROOT)
# ---------------------------
BASE_DIR = Path(__file__).parent

# ---------------------------
# Load feature columns (JSON)
# ---------------------------
with open(BASE_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

# ---------------------------
# Load FULL models (with preprocess inside)
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
# UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
### Response Scale
- **0** — Never  
- **1** — Almost Never  
- **2** — Sometimes  
- **3** — Fairly Often  
- **4** — Very Often  

Please answer all questions honestly based on your experience in the current semester.
""")

# ---------------------------
# User Input
# ---------------------------
user_input = {}

scale_map = {
    0: "0 — Never",
    1: "1 — Almost Never",
    2: "2 — Sometimes",
    3: "3 — Fairly Often",
    4: "4 — Very Often",
}

for col in FEATURE_COLS:
    user_input[col] = st.selectbox(
        col,
        list(scale_map.keys()),
        format_func=lambda x: scale_map[x]
    )

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    input_df = pd.DataFrame([user_input])

    st.subheader("📊 Prediction Result")

    scores = {}

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        scores[target] = avg

        st.metric(
            label=f"{target} Risk",
            value=f"{avg * 100:.2f}%"
        )

    # ---------------------------
    # Dominant Condition
    # ---------------------------
    dominant = max(scores, key=scores.get)

    st.markdown("---")
    st.subheader("🧠 Dominant Mental Health Condition")
    st.success(f"**{dominant}** ({scores[dominant] * 100:.2f}%)")
