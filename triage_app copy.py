#streamlit run triage_app.py
# pip install streamlit
# %%

# app.py
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

sys.path.append(
    '/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Assets'
)
from Normalize import add_symptom_columns, split_symptoms_and_treatments, map_symptoms_from_binary



from model_pipeline import (
    load_csv, build_ohe, map_constructs, construct_scores,
    fit_acuity_minmax, train_rf, evaluate_all,
    make_centroids, make_predictor,
    triage_thresholds_from_proportions, triage_thresholds_fixed,
    save_artifacts, load_artifacts
)

st.set_page_config(page_title="Symptom → Disease + Triage", layout="wide")
st.title("🩺 Symptom → Disease Classifier + Triage (Streamlit App)")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_path = st.sidebar.text_input("Or path to CSV", "All_data_disease_symptom.csv")

st.sidebar.header("Training")
min_cases = st.sidebar.slider("Min cases per disease", 50, 1000, 200, 50)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.35, 0.05)
n_estimators = st.sidebar.slider("RF n_estimators", 50, 600, 200, 50)
max_depth = st.sidebar.slider("RF max_depth", 4, 24, 12, 1)
use_calibration = st.sidebar.checkbox("Use isotonic calibration (prediction UI)", value=True)

st.sidebar.header("Triage")
use_proportion = st.sidebar.checkbox("Use proportion-based thresholds", value=True)
emerg_prop = st.sidebar.slider("Emergency proportion", 0.0, 0.2, 0.10, 0.01)
urgent_prop = st.sidebar.slider("Urgent proportion", 0.0, 0.5, 0.30, 0.01)

st.sidebar.header("Low-signal smoothing")
use_smoothing = st.sidebar.checkbox("Enable smoothing", value=True)
signal_threshold = st.sidebar.slider("Signal threshold", 0.0, 0.2, 0.05, 0.005)
prior_mix = st.sidebar.slider("Prior mix (centroid vs global)", 0.0, 1.0, 0.7, 0.05)
sim_temp = st.sidebar.slider("Centroid softmax temperature", 0.1, 2.0, 0.5, 0.1)

# -------------------------
# Data
# -------------------------
with st.spinner("Loading data..."):
    if uploaded is not None:
        df = pd.read_csv(uploaded, low_memory=False)
        if "_id" not in df.columns:
            df["_id"] = np.arange(len(df))
    else:
        df = load_csv(csv_path)
st.write("**Data shape:**", df.shape)

# -------------------------
# Feature engineering
# -------------------------
with st.spinner("Building features..."):
    ohe = build_ohe(df)
    mapping = map_constructs(ohe.columns)
    FS, construct_models = construct_scores(ohe, mapping)

with st.spinner("Fitting Acuity (train-only) & splitting..."):
    split = fit_acuity_minmax(FS, df, min_cases=min_cases, test_size=test_size)
    FS = split["FS"]
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    train_min, train_max = split["train_min"], split["train_max"]

st.success("Features ready.")

# -------------------------
# Train
# -------------------------
with st.spinner("Training RandomForest..."):
    clf = train_rf(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

# -------------------------
# Triage thresholds
# -------------------------
if use_proportion:
    q_emerg, q_urgent, triage_func = triage_thresholds_from_proportions(X_train["Acuity"], emerg_prop, urgent_prop)
else:
    q_emerg, q_urgent, triage_func = triage_thresholds_fixed(X_train)

# -------------------------
# Evaluate
# -------------------------
with st.spinner("Evaluating..."):
    eval_res = evaluate_all(clf, X_test, y_test, X_train=X_train, y_train=y_train)

st.subheader("📈 Evaluation")
st.text(eval_res["report_text"])
st.write("Confusion matrix shape:", eval_res["cm_shape"])
if eval_res["dropped_for_topk"] > 0:
    st.warning(f"Dropped {eval_res['dropped_for_topk']} test rows for Top-k metrics (labels unseen in training).")
st.write("Top-k (uncalibrated):", {k: round(v,3) for k,v in eval_res["topk"].items()})
st.write("MRR (uncalibrated):", round(eval_res["mrr"], 3))
if eval_res["topk_cal"] is not None:
    st.write("Top-k (calibrated):", {k: round(v,3) for k,v in eval_res["topk_cal"].items()})
    st.write("MRR (calibrated):", round(eval_res["mrr_cal"], 3))

st.subheader("🚦 Triage thresholds in use")
st.write({"Emergency": q_emerg, "Urgent": q_urgent})

# -------------------------
# Predictor prep
# -------------------------
construct_only_cols, centroids_df, centroids_unit = make_centroids(X_train, y_train, clf.classes_)
class_prior = y_train.value_counts().reindex(clf.classes_).fillna(0).astype(float)
class_prior = (class_prior / class_prior.sum()).to_numpy()

predictor_model = eval_res["cal_model"] if (use_calibration and eval_res["cal_model"] is not None) else clf

predict_patient = make_predictor(
    ohe_columns=ohe.columns.tolist(),
    construct_models=construct_models,
    clf=predictor_model,
    train_min=train_min,
    train_max=train_max,
    train_cols=X_train.columns.tolist(),
    triage_func=triage_func,
    use_smoothing=use_smoothing,
    signal_threshold=signal_threshold,
    prior_mix=prior_mix,
    sim_temp=sim_temp,
    class_prior=class_prior,
    centroids_unit=centroids_unit,
    construct_only=construct_only_cols
)

# -------------------------
# UI: Predict
# -------------------------
st.subheader("🧪 Try a prediction")
sym_in = st.text_input("Symptoms (comma-separated)", value="Fever, Cough, Palpitations")
if st.button("Predict"):
    symptoms = [s.strip() for s in sym_in.split(",") if s.strip()]
    out = predict_patient(symptoms)
    st.write("**Symptoms:**", out["symptoms"])
    if out["unseen_symptoms"]:
        st.warning(f"Unseen symptoms ignored: {out['unseen_symptoms']}")
    st.write("**Predicted disease:**", out["predicted_disease"])
    st.write("**Top predictions:**")
    st.table(pd.DataFrame(out["top_diseases"], columns=["Disease", "Probability"]))
    st.write("**Acuity:**", round(out["acuity"], 6), "→ **Triage:**", out["triage_level"])
    st.caption(f"Signal={round(out['signal'], 6)} | Smoothing β={round(out['beta_used'], 3)}")

# -------------------------
# Save / Load artifacts
# -------------------------
st.subheader("💾 Save / Load")
c1, c2 = st.columns(2)
with c1:
    if st.button("Save artifacts → triage_model_artifacts.joblib"):
        save_artifacts(
            "triage_model_artifacts.joblib",
            clf=clf,
            cal_clf=eval_res["cal_model"],
            ohe_vocab=ohe.columns.tolist(),
            construct_models=construct_models,
            train_cols=X_train.columns.tolist(),
            train_min=train_min,
            train_max=train_max,
            q_emerg=q_emerg,
            q_urgent=q_urgent,
            class_prior=class_prior,
            centroids_unit=centroids_unit,
            construct_only=construct_only_cols,
            use_calibration=use_calibration,
            use_smoothing=use_smoothing,
            signal_threshold=signal_threshold,
            prior_mix=prior_mix,
            sim_temp=sim_temp
        )
        st.success("Saved triage_model_artifacts.joblib")

with c2:
    up = st.file_uploader("Load .joblib", type=["joblib"], key="art_up")
    if up is not None:
        art = load_artifacts(up)
        st.success("Artifacts loaded.")
        st.json({
            "has_calibrated": art.get("cal_clf") is not None,
            "triage_thresholds": {"Emergency": art.get("q_emerg"), "Urgent": art.get("q_urgent")},
            "smoothing_enabled": art.get("use_smoothing", False)
        })
