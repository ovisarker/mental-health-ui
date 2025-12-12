import streamlit as st
import pandas as pd
import joblib
import json
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
# Load feature list (33 columns)
# ---------------------------
with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)["features"]

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("""
### Response Scale (for questionnaire items)
- **0** — Never  
- **1** — Almost Never  
- **2** — Sometimes  
- **3** — Fairly Often  
- **4** — Very Often  
""")

# Collect user input
user_input = {}

for col in FEATURE_COLS:
    # For demographic entries that require text (e.g., Age or CGPA), you can use text_input or number_input
    if "Age" in col or "CGPA" in col or "University" in col or "Department" in col or "Academic Year" in col or "Gender" in col:
        user_input[col] = st.text_input(col, "")
    else:
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
    # Create DataFrame for user input
    input_df = pd.DataFrame([user_input])
    # Ensure column order and presence match the training
    input_df = input_df.reindex(columns=FEATURE_COLS)

    st.subheader("📊 Prediction Results")

    for target, (m1, m2) in models.items():
        # Each model is a pipeline; calling predict_proba runs preprocessing then classification
        p1 = m1.predict_proba(input_df)[0][1]
        p2 = m2.predict_proba(input_df)[0][1]
        avg = (p1 + p2) / 2

        st.metric(
            label=f"{target} Risk",
            value=f"{avg * 100:.2f}%"
        )
