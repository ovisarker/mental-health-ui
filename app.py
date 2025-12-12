import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

BASE_DIR = Path(__file__).parent

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

with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

LIKERT = {
    "Never": 0,
    "Almost Never": 1,
    "Sometimes": 2,
    "Fairly Often": 3,
    "Very Often": 4
}

st.title("🧠 Student Mental Health Risk Predictor")

user_input = {}

# -------- Demographics --------
user_input["1. Age"] = st.number_input("Age", 15, 60, 22)
user_input["2. Gender"] = st.selectbox("Gender", ["Male", "Female", "Other"])
user_input["3. University"] = st.text_input("University")
user_input["4. Department"] = st.text_input("Department")
user_input["5. Academic Year"] = st.selectbox("Academic Year", [1, 2, 3, 4])
user_input["6. Current CGPA"] = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
user_input["7. Did you receive a waiver or scholarship at your university?"] = st.selectbox(
    "Waiver / Scholarship", ["Yes", "No"]
)

# -------- Questionnaire --------
for col in FEATURE_COLS[7:]:
    label = st.selectbox(col, list(LIKERT.keys()))
    user_input[col] = LIKERT[label]

if st.button("🔍 Predict Mental Health Risk"):
    df = pd.DataFrame([user_input])

    st.subheader("📊 Prediction Results")
    for target, (m1, m2) in models.items():
        p = (m1.predict_proba(df)[0][1] + m2.predict_proba(df)[0][1]) / 2
        st.metric(target, f"{p*100:.2f}%")
