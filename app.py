import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Paths (ROOT folder)
# ---------------------------
BASE_DIR = Path(__file__).parent

# ---------------------------
# Load models (FULL PIPELINES)
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
# Load RAW feature names (33)
# ---------------------------
FEATURE_COLS = joblib.load(BASE_DIR / "feature_names.joblib")

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
""")

user_input = {}

for col in FEATURE_COLS:
    user_input[col] = st.selectbox(
        col,
        [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — Never",
            1: "1 — Almost Never",
            2: "2 — Sometimes",
            3: "3 — Fairly Often",
            4: "4 — Very Often",
        }[x]
    )

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):
    input_df = pd.DataFrame([user_input])

    st.subheader("📊 Prediction Result")

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(
            label=f"{target} Risk",
            value=f"{avg * 100:.2f}%"
        )
