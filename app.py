import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Student Mental Health Risk Prediction",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ===============================
# Load Schema (SOURCE OF TRUTH)
# ===============================
with open(BASE_DIR / "schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
all_cols = demo_cols + q_cols

# ===============================
# Load Artifacts (Cached)
# ===============================
@st.cache_resource
def load_artifacts():
    preprocess = joblib.load(BASE_DIR / "preprocess_pipeline.joblib")

    models = {
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
    return preprocess, models

preprocess, models = load_artifacts()

# ===============================
# UI HEADER
# ===============================
st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.write(
    "Answer all questions honestly. "
    "Scale is based on standard psychological assessment."
)

st.markdown("""
**Response Scale**
- 0 — Never  
- 1 — Sometimes  
- 2 — Often  
- 3 — Very Often  
""")

# ===============================
# Scale Mapping (UI clarity)
# ===============================
scale_map = {
    "0": "Never",
    "1": "Sometimes",
    "2": "Often",
    "3": "Very Often"
}

# ===============================
# Collect User Input
# ===============================
user_input = {}

st.subheader("🧾 Demographic Information")
for col in demo_cols:
    user_input[col] = st.text_input(col)

st.subheader("📝 Questionnaire")

for col in q_cols:
    choice = st.selectbox(
        col,
        options=list(scale_map.keys()),
        format_func=lambda x: f"{x} — {scale_map[x]}"
    )
    user_input[col] = int(choice)

# ===============================
# Prediction Button
# ===============================
if st.button("🔍 Predict Mental Health Risk"):

    # ---------------------------
    # Build input_df SAFELY
    # ---------------------------
    input_df = pd.DataFrame(columns=all_cols)
    input_df.loc[0] = [user_input[c] for c in all_cols]

    # ---------------------------
    # Preprocess
    # ---------------------------
    X_trans = preprocess.transform(input_df)

    st.subheader("📊 Prediction Result")

    risk_scores = {}

    # ---------------------------
    # Hybrid Prediction (Per Target)
    # ---------------------------
    for target, (model_A, model_B) in models.items():

        prob_A = model_A.predict_proba(X_trans)
        prob_B = model_B.predict_proba(X_trans)

        avg_prob = (prob_A + prob_B) / 2.0
        pred_class = np.argmax(avg_prob[0])
        confidence = avg_prob[0][pred_class] * 100

        status = "Present" if pred_class == 1 else "Absent"
        risk_scores[target] = confidence

        st.write(
            f"**{target}** : {status} "
            f"({confidence:.2f}%)"
        )

    # ---------------------------
    # Dominant Condition
    # ---------------------------
    dominant = max(risk_scores, key=risk_scores.get)

    st.markdown("---")
    st.success(
        f"🚨 **Dominant Condition:** "
        f"{dominant} ({risk_scores[dominant]:.2f}%)"
    )
