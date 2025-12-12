# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict Mental Health Risk"):

    # 1️⃣ Build single-row input from UI
    row = {}
    for col in all_cols:
        row[col] = user_input.get(col, 0)

    input_df = pd.DataFrame([row])

    # 2️⃣ 🔥 CRITICAL FIX: align columns with training pipeline
    input_df = input_df.reindex(
        columns=preprocess.feature_names_in_,
        fill_value=0
    )

    # 3️⃣ Transform using trained preprocess pipeline
    X_trans = preprocess.transform(input_df)

    # ---------------------------
    # Prediction
    # ---------------------------
    st.subheader("📊 Prediction Result")

    results = {}
    for target, model_pair in models.items():
        m1, m2 = model_pair

        p1 = m1.predict_proba(X_trans)[0][1]
        p2 = m2.predict_proba(X_trans)[0][1]

        avg_prob = (p1 + p2) / 2
        status = "Present" if avg_prob >= 0.5 else "Absent"

        results[target] = {
            "status": status,
            "percent": round(avg_prob * 100, 2)
        }

        st.write(f"**{target}** : {status} ({avg_prob*100:.2f}%)")
