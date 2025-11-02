
# streamlit run triage_app.py
# pip install streamlit
# python -m streamlit run triage_app.py
# source .venv/bin/activate
# python -m streamlit run triage_app.py --server.port 5000

# triage_app.py

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from model_pipeline import (
    load_csv, build_ohe, map_constructs, construct_scores,
    fit_acuity_minmax, train_rf, evaluate_all,
    make_centroids, make_predictor,
    triage_thresholds_from_proportions, triage_thresholds_fixed,
    save_artifacts, load_artifacts as _joblib_load_artifacts  # your helper
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Symptom → Disease + Triage", page_icon="🩺", layout="wide")
st.title("🩺 Symptom → Disease Classifier + Triage")

# ──────────────────────────────────────────────────────────────────────────────
# (A) Cloud-safe local artifact loader
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_local_artifacts():
    """
    Try to load triage_model_artifacts.joblib from the app's working dir.
    Returns a dict (artifacts) or None. Never crashes if missing.
    """
    p = Path("triage_model_artifacts.joblib")
    if not p.exists():
        st.info("No local model artifacts found → running in TRAIN/DEMO mode unless you upload one below.")
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.warning(f"Failed to load local artifacts: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Helper: build triage thresholds even if old artifacts lack them
# ──────────────────────────────────────────────────────────────────────────────
def build_triage_from_artifacts(art):
    """
    Returns (q_emerg, q_urgent, triage_func). If the artifact doesn't contain
    thresholds, fallback to defaults (Emergency≥0.90, Urgent≥0.60).
    """
    q_e = art.get("q_emerg") if isinstance(art, dict) else None
    q_u = art.get("q_urgent") if isinstance(art, dict) else None

    if (q_e is not None) and (q_u is not None):
        q_e, q_u = float(q_e), float(q_u)
        def triage_func(a: float) -> str:
            if a >= q_e:  return "Emergency"
            if a >= q_u:  return "Urgent"
            return "Routine"
        return q_e, q_u, triage_func

    # Fallback
    q_e_default, q_u_default = 0.90, 0.60
    st.info("Artifacts missing triage thresholds — using defaults (Emergency≥0.90, Urgent≥0.60). "
            "Train once and re-save artifacts to persist real thresholds.")
    def triage_func(a: float) -> str:
        if a >= q_e_default:  return "Emergency"
        if a >= q_u_default:  return "Urgent"
        return "Routine"
    return q_e_default, q_u_default, triage_func

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Mode")
use_prebuilt = st.sidebar.checkbox("Use prebuilt artifacts (skip training)", value=True)

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

# ──────────────────────────────────────────────────────────────────────────────
# (B) Mode select: attempt to load artifacts if requested
# ──────────────────────────────────────────────────────────────────────────────
artifacts_local = load_local_artifacts() if use_prebuilt else None

# Flags/holders used across branches
is_training_path = False
eval_res = None
predict_patient = None
q_emerg = None
q_urgent = None

# For Save/Load section variables (set in either path)
_save_vars = {}

# ──────────────────────────────────────────────────────────────────────────────
# (C) Main flow: either use artifacts OR run full train/eval pipeline
# ──────────────────────────────────────────────────────────────────────────────
if artifacts_local is not None:
    # =========================
    # PATH 1: Use prebuilt artifacts
    # =========================
    st.success("Using prebuilt artifacts (skipping training).")

    # Minimal required keys for inference
    required = ["clf", "ohe_vocab", "construct_models", "train_cols", "train_min", "train_max"]
    missing = [k for k in required if k not in artifacts_local]
    if missing:
        st.error(f"Artifacts are missing required entries: {missing}. "
                 "Re-train locally and click 'Save artifacts' to create a complete bundle.")
        st.stop()

    # Unpack (use .get for optional fields)
    clf_raw          = artifacts_local.get("clf")
    cal_clf         = artifacts_local.get("cal_clf")               # may be None
    ohe_vocab        = artifacts_local.get("ohe_vocab", [])
    construct_models = artifacts_local.get("construct_models", {})
    train_cols       = artifacts_local.get("train_cols", [])
    train_min        = artifacts_local.get("train_min", pd.Series(dtype=float))
    train_max        = artifacts_local.get("train_max", pd.Series(dtype=float))
    class_prior      = artifacts_local.get("class_prior", None)
    centroids_unit   = artifacts_local.get("centroids_unit", None)
    construct_only   = artifacts_local.get("construct_only", None)
    saved_use_cal    = bool(artifacts_local.get("use_calibration", False))
    saved_use_smooth = bool(artifacts_local.get("use_smoothing", True))
    saved_signal_thr = float(artifacts_local.get("signal_threshold", 0.05))
    saved_prior_mix  = float(artifacts_local.get("prior_mix", 0.7))
    saved_sim_temp   = float(artifacts_local.get("sim_temp", 0.5))

    # Pick model for inference (respect sidebar + availability)
    predictor_model = cal_clf if (use_calibration and cal_clf is not None) else clf_raw

    # Thresholds (robust to missing keys)
    q_emerg, q_urgent, triage_func = build_triage_from_artifacts(artifacts_local)

    # Build predictor (no CSV/training needed)
    predict_patient = make_predictor(
        ohe_columns=ohe_vocab,
        construct_models=construct_models,
        clf=predictor_model,
        train_min=train_min,
        train_max=train_max,
        train_cols=train_cols,
        triage_func=triage_func,
        use_smoothing=use_smoothing if use_smoothing is not None else saved_use_smooth,
        signal_threshold=signal_threshold if signal_threshold is not None else saved_signal_thr,
        prior_mix=prior_mix if prior_mix is not None else saved_prior_mix,
        sim_temp=sim_temp if sim_temp is not None else saved_sim_temp,
        class_prior=class_prior,
        centroids_unit=centroids_unit,
        construct_only=construct_only
    )

    st.success("Predictor ready (from artifacts).")

    # Save variables for "Save / Load" section
    _save_vars = dict(
        clf=clf_raw,
        cal_clf=cal_clf,
        ohe_vocab=ohe_vocab,
        construct_models=construct_models,
        train_cols=train_cols,
        train_min=train_min,
        train_max=train_max,
        q_emerg=q_emerg,
        q_urgent=q_urgent,
        class_prior=class_prior,
        centroids_unit=centroids_unit,
        construct_only=construct_only,
        use_calibration=use_calibration,
        use_smoothing=use_smoothing,
        signal_threshold=signal_threshold,
        prior_mix=prior_mix,
        sim_temp=sim_temp
    )

else:
    # =========================
    # PATH 2: Train pipeline (original flow)
    # =========================
    is_training_path = True

    # Data
    with st.spinner("Loading data..."):
        if uploaded is not None:
            df = pd.read_csv(uploaded, low_memory=False)
            if "_id" not in df.columns:
                df["_id"] = np.arange(len(df))
        else:
            df = load_csv(csv_path)
    st.write("**Data shape:**", df.shape)

    # Feature engineering
    with st.spinner("Building features..."):
        ohe = build_ohe(df)
        mapping = map_constructs(ohe.columns)
        FS, construct_models = construct_scores(ohe, mapping)

    # Acuity + split
    with st.spinner("Fitting Acuity (train-only) & splitting..."):
        split = fit_acuity_minmax(FS, df, min_cases=min_cases, test_size=test_size)
        FS = split["FS"]
        X_train, y_train = split["X_train"], split["y_train"]
        X_test,  y_test  = split["X_test"],  split["y_test"]
        train_min, train_max = split["train_min"], split["train_max"]
    st.success("Features ready.")

    # Train RF
    with st.spinner("Training RandomForest..."):
        clf = train_rf(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

    # Triage thresholds
    if use_proportion:
        q_emerg, q_urgent, triage_func = triage_thresholds_from_proportions(
            X_train["Acuity"], emerg_prop, urgent_prop
        )
    else:
        q_emerg, q_urgent, triage_func = triage_thresholds_fixed(X_train)

    # Evaluation
    with st.spinner("Evaluating..."):
        eval_res = evaluate_all(clf, X_test, y_test, X_train=X_train, y_train=y_train)

    st.subheader("📈 Evaluation")
    # Summary tables
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

    # Macro / weighted averages
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
                "f1-score":  round(rep.get("accuracy", float("nan")), 3),
            }
        },
        orient="index"
    )
    st.markdown("**Averages**")
    st.table(avg_df)

    st.markdown("**Full classification report (text)**")
    st.text(eval_res["report_text"])
    st.write("Confusion matrix shape:", eval_res["cm_shape"])
    if eval_res["dropped_for_topk"] > 0:
        st.warning(
            f"Dropped {eval_res['dropped_for_topk']} test rows for Top-k metrics (labels unseen in training)."
        )
    st.write("Top-k (uncalibrated):", {k: round(v, 3) for k, v in eval_res["topk"].items()})
    st.write("MRR (uncalibrated):", round(eval_res["mrr"], 3))
    if eval_res["topk_cal"] is not None:
        st.write("Top-k (calibrated):", {k: round(v, 3) for k, v in eval_res["topk_cal"].items()})
        st.write("MRR (calibrated):", round(eval_res["mrr_cal"], 3))

    st.subheader("🚦 Triage thresholds in use")
    st.write({"Emergency": q_emerg, "Urgent": q_urgent})

    # Predictor prep (training path)
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

    # Save variables for "Save / Load" section
    _save_vars = dict(
        clf=clf,
        cal_clf=eval_res.get("cal_model"),
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

# ──────────────────────────────────────────────────────────────────────────────
# (D) UI: Try a prediction (works in both modes)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🧪 Try a prediction")
sym_in = st.text_input("Symptoms (comma-separated)", value="Fever, Cough, Palpitations")

if st.button("Predict"):
    if predict_patient is None:
        st.error("Predictor is not ready. Please load artifacts or run training.")
    else:
        symptoms = [s.strip() for s in sym_in.split(",") if s.strip()]
        out = predict_patient(symptoms)
        st.write("**Symptoms:**", out["symptoms"])
        if out.get("unseen_symptoms"):
            st.warning(f"Unseen symptoms ignored: {out['unseen_symptoms']}")
        st.write("**Predicted disease:**", out["predicted_disease"])
        st.write("**Top predictions:**")
        st.table(pd.DataFrame(out["top_diseases"], columns=["Disease", "Probability"]))
        st.write("**Acuity:**", round(out["acuity"], 6), "→ **Triage:**", out["triage_level"])
        if "signal" in out and "beta_used" in out:
            st.caption(f"Signal={round(out['signal'], 6)} | Smoothing β={round(out['beta_used'], 3)}")

# ──────────────────────────────────────────────────────────────────────────────
# (E) Save / Load artifacts (works in both modes)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("💾 Save / Load")
c1, c2 = st.columns(2)

with c1:
    if st.button("Save artifacts → triage_model_artifacts.joblib"):
        if not _save_vars:
            st.error("Nothing to save yet. Train a model or load artifacts first.")
        else:
            save_artifacts(
                "triage_model_artifacts.joblib",
                clf=_save_vars["clf"],
                cal_clf=_save_vars["cal_clf"],
                ohe_vocab=_save_vars["ohe_vocab"],
                construct_models=_save_vars["construct_models"],
                train_cols=_save_vars["train_cols"],
                train_min=_save_vars["train_min"],
                train_max=_save_vars["train_max"],
                q_emerg=_save_vars["q_emerg"],
                q_urgent=_save_vars["q_urgent"],
                class_prior=_save_vars["class_prior"],
                centroids_unit=_save_vars["centroids_unit"],
                construct_only=_save_vars["construct_only"],
                use_calibration=_save_vars["use_calibration"],
                use_smoothing=_save_vars["use_smoothing"],
                signal_threshold=_save_vars["signal_threshold"],
                prior_mix=_save_vars["prior_mix"],
                sim_temp=_save_vars["sim_temp"]
            )
            st.success("Saved triage_model_artifacts.joblib (local to app working dir).")

with c2:
    up = st.file_uploader("Load .joblib", type=["joblib"], key="art_up")
    if up is not None:
        try:
            art = _joblib_load_artifacts(up)  # use your helper
            st.success("Artifacts loaded.")
            st.json({
                "has_calibrated": art.get("cal_clf") is not None,
                "triage_thresholds": {"Emergency": art.get("q_emerg"), "Urgent": art.get("q_urgent")},
                "smoothing_enabled": art.get("use_smoothing", False)
            })
            st.info("Reload the page with 'Use prebuilt artifacts' enabled to use these without retraining.")
        except Exception as e:
            st.error(f"Failed to load uploaded artifacts: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# (F) If we skipped training, give a small note where evaluation would be
# ──────────────────────────────────────────────────────────────────────────────
if not is_training_path:
    st.info("Running in artifacts mode: training metrics are not shown. Switch off 'Use prebuilt artifacts' to train/evaluate here.")
