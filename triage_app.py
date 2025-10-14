# streamlit run triage_app.py
# pip install streamlit
# python -m streamlit run triage_app.py

# %%

# app.py
from model_pipeline import (
    load_csv, build_ohe, map_constructs, construct_scores,
    fit_acuity_minmax, train_rf, evaluate_all,
    make_centroids, make_predictor,
    triage_thresholds_from_proportions, triage_thresholds_fixed,
    save_artifacts, load_artifacts
)
#from Normalize import add_symptom_columns, split_symptoms_and_treatments, map_symptoms_from_binary
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


st.set_page_config(page_title="Symptom → Disease + Triage", layout="wide")
st.title("🩺 Symptom → Disease Classifier + Triage (Streamlit App)")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_path = st.sidebar.text_input(
    "Or path to CSV", "All_data_disease_symptom.csv")

st.sidebar.header("Training")
min_cases = st.sidebar.slider("Min cases per disease", 50, 1000, 200, 50)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.35, 0.05)
n_estimators = st.sidebar.slider("RF n_estimators", 50, 600, 200, 50)
max_depth = st.sidebar.slider("RF max_depth", 4, 24, 12, 1)
use_calibration = st.sidebar.checkbox(
    "Use isotonic calibration (prediction UI)", value=True)

st.sidebar.header("Triage")
use_proportion = st.sidebar.checkbox(
    "Use proportion-based thresholds", value=True)
emerg_prop = st.sidebar.slider("Emergency proportion", 0.0, 0.2, 0.10, 0.01)
urgent_prop = st.sidebar.slider("Urgent proportion", 0.0, 0.5, 0.30, 0.01)

st.sidebar.header("Low-signal smoothing")
use_smoothing = st.sidebar.checkbox("Enable smoothing", value=True)
signal_threshold = st.sidebar.slider("Signal threshold", 0.0, 0.2, 0.05, 0.005)
prior_mix = st.sidebar.slider(
    "Prior mix (centroid vs global)", 0.0, 1.0, 0.7, 0.05)
sim_temp = st.sidebar.slider(
    "Centroid softmax temperature", 0.1, 2.0, 0.5, 0.1)


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
    clf = train_rf(X_train, y_train, n_estimators=n_estimators,
                   max_depth=max_depth)

# -------------------------
# Triage thresholds
# -------------------------
if use_proportion:
    q_emerg, q_urgent, triage_func = triage_thresholds_from_proportions(
        X_train["Acuity"], emerg_prop, urgent_prop)
else:
    q_emerg, q_urgent, triage_func = triage_thresholds_fixed(X_train)


# =======================
# Evaluation + Tables
# =======================
with st.spinner("Evaluating..."):
    eval_res = evaluate_all(clf, X_test, y_test,
                            X_train=X_train, y_train=y_train)

st.subheader("📈 Evaluation")

# --- Summary tables (NEW) ---
# Prefer dict from evaluate_all; fallback to recompute if missing.
if "report_dict" in eval_res:
    rep = eval_res["report_dict"]
    acc_overall = eval_res.get("accuracy", rep.get("accuracy", None))
else:
    from sklearn.metrics import classification_report
    _pred_tmp = clf.predict(X_test)
    rep = classification_report(y_test, _pred_tmp, output_dict=True)
    acc_overall = float(rep.get("accuracy", 0.0))

topk_uncal = eval_res["topk"]
mrr_uncal = eval_res["mrr"]
topk_cal = eval_res.get("topk_cal")
mrr_cal = eval_res.get("mrr_cal")

summary_rows = [
    ["Accuracy", round(acc_overall, 3)],
    ["Top-1 (uncal)", round(topk_uncal.get("Top-1", float("nan")), 3)],
    ["Top-3 (uncal)", round(topk_uncal.get("Top-3", float("nan")), 3)],
    ["Top-5 (uncal)", round(topk_uncal.get("Top-5", float("nan")), 3)],
    ["MRR (uncal)", round(mrr_uncal, 3)],
]
if topk_cal is not None and mrr_cal is not None:
    summary_rows += [
        ["Top-1 (cal)", round(topk_cal.get("Top-1", float("nan")), 3)],
        ["Top-3 (cal)", round(topk_cal.get("Top-3", float("nan")), 3)],
        ["Top-5 (cal)", round(topk_cal.get("Top-5", float("nan")), 3)],
        ["MRR (cal)", round(mrr_cal, 3)],
    ]
summary_rows += [
    ["Emergency threshold", round(q_emerg, 6)],
    ["Urgent threshold", round(q_urgent, 6)],
]
summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
st.markdown("**Summary table**")
st.table(summary_df)

# Macro / weighted averages (NEW)
macro = rep.get("macro avg", {})
weighted = rep.get("weighted avg", {})
avg_df = pd.DataFrame.from_dict(
    {
        "macro avg": {
            "precision": round(macro.get("precision", float("nan")), 3),
            "recall":    round(macro.get("recall", float("nan")), 3),
            "f1-score":  round(macro.get("f1-score", float("nan")), 3),
        },
        "weighted avg": {
            "precision": round(weighted.get("precision", float("nan")), 3),
            "recall":    round(weighted.get("recall", float("nan")), 3),
            "f1-score":  round(weighted.get("f1-score", float("nan")), 3),
        },
        "overall": {
            "precision": float("nan"),
            "recall":    float("nan"),
            # show accuracy in this row
            "f1-score":  round(rep.get("accuracy", float("nan")), 3),
        }
    },
    orient="index"
)
st.markdown("**Averages**")
st.table(avg_df)

# --- Keep your original full text report & details below (optional) ---
st.markdown("**Full classification report (text)**")
st.text(eval_res["report_text"])

st.write("Confusion matrix shape:", eval_res["cm_shape"])
if eval_res["dropped_for_topk"] > 0:
    st.warning(
        f"Dropped {eval_res['dropped_for_topk']} test rows for Top-k metrics (labels unseen in training).")

st.write("Top-k (uncalibrated):", {k: round(v, 3)
         for k, v in eval_res["topk"].items()})
st.write("MRR (uncalibrated):", round(eval_res["mrr"], 3))
if eval_res["topk_cal"] is not None:
    st.write("Top-k (calibrated):", {k: round(v, 3)
             for k, v in eval_res["topk_cal"].items()})
    st.write("MRR (calibrated):", round(eval_res["mrr_cal"], 3))

st.subheader("🚦 Triage thresholds in use")
st.write({"Emergency": q_emerg, "Urgent": q_urgent})

# (Optional) Per-class table (big) behind an expander
with st.expander("Per-class metrics (precision / recall / f1 / support)"):
    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    class_rows = {k: v for k, v in rep.items() if k not in skip_keys}
    per_class_df = (
        pd.DataFrame(class_rows)
        .T[["precision", "recall", "f1-score", "support"]]
        .sort_index()
    )
    per_class_df[["precision", "recall", "f1-score"]
                 ] = per_class_df[["precision", "recall", "f1-score"]].round(3)
    st.dataframe(per_class_df, height=500)



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
st.success("Predictor ready.")


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
