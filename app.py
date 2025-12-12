import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Student Mental Health Risk (Hybrid ML)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def load_schema(schema_path: str = "schema.json") -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_artifacts(models_dir: str):
    models_dir = Path(models_dir)

    preprocess = joblib.load(models_dir / "preprocess_pipeline.joblib")

    loaded = {}
    for tgt in ["Anxiety", "Stress", "Depression"]:
        A = joblib.load(models_dir / f"{tgt}_model_A.joblib")
        B = joblib.load(models_dir / f"{tgt}_model_B.joblib")
        loaded[tgt] = (A, B)

    return preprocess, loaded

def hybrid_predict_proba(pipeA, pipeB, user_df) -> float:
    # Binary classifier proba for "Present" class = [:,1]
    pA = float(pipeA.predict_proba(user_df)[:, 1][0])
    pB = float(pipeB.predict_proba(user_df)[:, 1][0])
    return (pA + pB) / 2.0

def present_absent(percent: float, thr: float = 50.0) -> str:
    return "Present" if percent >= thr else "Absent"

# ----------------------------
# UI Header
# ----------------------------
st.title("🧠 Student Mental Health Risk Prediction (Hybrid ML)")
st.caption("Outputs Anxiety / Stress / Depression as Present/Absent with % confidence, plus Dominant condition. (Deployment-only UI)")

# ----------------------------
# Load schema + artifacts
# ----------------------------
schema = load_schema("schema.json")
demo_cols = schema["demo_cols"]
q_cols = schema["q_cols"]
numeric_demo = set(schema.get("numeric_demo", []))
cat_demo = schema.get("cat_demo", [])
QMIN = int(schema.get("question_scale_min", 0))
QMAX = int(schema.get("question_scale_max", 4))

with st.sidebar:
    st.header("⚙️ Settings")
    models_dir = st.text_input("Models folder", value="models")
    decision_threshold = st.slider("Present threshold (%)", 0, 100, 50)
    st.write("Tip: keep 50% for now.")

preprocess, models = load_artifacts(models_dir)

# ----------------------------
# Input Form
# ----------------------------
st.subheader("📝 Input (33 fields)")

tab1, tab2 = st.tabs(["Manual Form", "Upload 1-row CSV"])

user_df = None

with tab1:
    with st.form("mh_form"):
        st.markdown("### Demographics (7)")
        dcols = st.columns(3)
        demo_values = {}

        for i, col in enumerate(demo_cols):
            with dcols[i % 3]:
                if col in numeric_demo:
                    demo_values[col] = st.number_input(col, value=0.0, step=1.0)
                else:
                    # Free text is safe because OneHotEncoder(handle_unknown="ignore")
                    demo_values[col] = st.text_input(col, value="")

        st.markdown("### Questionnaire (26)")
        st.caption(f"Answer scale: {QMIN} to {QMAX} (adjust in schema.json if your real scale differs)")

        q_values = {}
        for idx, q in enumerate(q_cols, start=1):
            q_values[q] = st.slider(f"Q{idx}: {q}", QMIN, QMAX, QMIN)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            row = {**demo_values, **q_values}
            user_df = pd.DataFrame([row], columns=demo_cols + q_cols)

with tab2:
    st.write("Upload a CSV containing exactly **one row** with the same columns as your training raw data (demo + questions).")
    up = st.file_uploader("Upload 1-row CSV", type=["csv"])
    if up is not None:
        temp = pd.read_csv(up)
        missing = [c for c in (demo_cols + q_cols) if c not in temp.columns]
        if missing:
            st.error(f"Missing columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
        else:
            user_df = temp[demo_cols + q_cols].iloc[[0]].copy()
            st.success("✅ CSV accepted (1 row). Click Predict below.")
            if st.button("🔍 Predict from Uploaded Row"):
                pass  # user_df already set

# ----------------------------
# Prediction
# ----------------------------
if user_df is not None:
    st.divider()
    st.subheader("✅ Prediction Results")

    results = {}
    risk_scores = {}

    for tgt in ["Anxiety", "Stress", "Depression"]:
        A, B = models[tgt]
        p = hybrid_predict_proba(A, B, user_df)
        percent = round(p * 100, 2)
        status = present_absent(percent, decision_threshold)

        results[tgt] = {"status": status, "percent": percent}
        risk_scores[tgt] = percent

    dominant = max(risk_scores, key=risk_scores.get)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anxiety", f"{results['Anxiety']['status']}", f"{results['Anxiety']['percent']}%")
    c2.metric("Stress", f"{results['Stress']['status']}", f"{results['Stress']['percent']}%")
    c3.metric("Depression", f"{results['Depression']['status']}", f"{results['Depression']['percent']}%")

    st.info(f"🚨 **Dominant Condition:** **{dominant}** ({risk_scores[dominant]}%)")

    with st.expander("Show details"):
        st.write("Threshold:", f"{decision_threshold}%")
        st.write(results)

    st.caption("Note: If you see extreme 100% values, it can happen when the input strongly matches a learned pattern; try different responses to verify behavior.")
