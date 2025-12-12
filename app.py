# =========================================
# Student Mental Health Risk Prediction UI
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Student Mental Health Risk Prediction",
    layout="wide"
)

st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.caption(
    "Predicts Anxiety / Stress / Depression (Present / Absent) with % confidence"
)

# ---------------------------
# Load schema
# ---------------------------
def load_schema(path="schema.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

schema = load_schema()
demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
all_cols = demo_cols + q_cols

# ---------------------------
# Load artifacts
# ---------------------------
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

BASE_DIR = Path(".")
preprocess, models = load_artifacts(BASE_DIR)

# ---------------------------
# Likert Scale (User Friendly)
# ---------------------------
LIKERT_LABELS = {
    "Never (0)": 0,
    "Almost Never (1)": 1,
    "Sometimes (2)": 2,
    "Fairly Often (3)": 3,
    "Very Often (4)": 4,
}

# ---------------------------
# UI Input
# ---------------------------
st.header("📝 Questionnaire")

user_input = {}

with st.form("mental_health_form"):
    st.subheader("Demographics")

    for col in demo_cols:
        user_input[col] = st.text_input(col, value="")

    st.subheader("Questions")
    for col in q_cols:
        user_input[col] = st.selectbox(col, list(LIKERT_LABELS.keys()))

    submitted = st.form_submit_button("🔍 Predict Mental Health Risk")

# ---------------------------
# Prediction
# ---------------------------
if submitted:
    row = {}

    # ---- FIX 1: Convert demographics safely ----
    for col in demo_cols:
        val = user_input.get(col, "").strip()
        if val == "":
            row[col] = np.nan          # 🔥 critical fix
        else:
            try:
                row[col] = float(val)
            except:
                row[col] = np.nan

    # ---- FIX 2: Convert Likert to numeric ----
    for col in q_cols:
        row[col] = LIKERT_LABELS[user_input[col]]

    input_df = pd.DataFrame([row])

    # ---- FIX 3: Align columns with training ----
    input_df = input_df.reindex(
        columns=preprocess.feature_names_in_,
        fill_value=np.nan
    )

    # ---- Preprocess ----
    X_trans = preprocess.transform(input_df)

    # ---------------------------
    # Results
    # ---------------------------
    st.subheader("📊 Prediction Result")

    scores = {}
    for target, (m1, m2) in models.items():
        p1 = m1.predict_proba(X_trans)[0][1]
        p2 = m2.predict_proba(X_trans)[0][1]

        avg = (p1 + p2) / 2
        status = "Present" if avg >= 0.5 else "Absent"

        scores[target] = avg

        st.write(f"**{target}** : {status} ({avg*100:.2f}%)")

    dominant = max(scores, key=scores.get)

    st.markdown("---")
    st.success(
        f"🚨 **Dominant Condition:** {dominant} ({scores[dominant]*100:.2f}%)"
    )
