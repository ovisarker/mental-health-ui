import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
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
# Load UI feature labels (for display only)
# ---------------------------
with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    UI_FEATURES = json.load(f)   # ✅ list

# ---------------------------
# Extract TRUE model features (ground truth)
# ---------------------------
# Take from first pipeline
sample_model = list(models.values())[0][0]
MODEL_FEATURES = list(sample_model.feature_names_in_)

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
**Response Scale**
- 0 — Never
- 1 — Almost Never
- 2 — Sometimes
- 3 — Fairly Often
- 4 — Very Often
""")

user_input = {}

for col in UI_FEATURES:
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
    raw_df = pd.DataFrame([user_input])

    # 🔑 ALIGN TO MODEL FEATURES (THIS IS THE MAGIC FIX)
    input_df = raw_df.reindex(columns=MODEL_FEATURES, fill_value=0)

    st.subheader("📊 Prediction Results")

    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(f"{target} Risk", f"{avg * 100:.2f}%")
