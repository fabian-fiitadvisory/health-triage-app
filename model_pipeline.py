# model_pipeline.py
# End-to-end helpers: load → OHE → construct scores → train RF → triage thresholds → predictor

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

# -------------------------
# Construct keyword map
# -------------------------
CONSTRUCT_KEYWORDS: Dict[str, list] = {
    "Cardio/Resp": ["chest", "palpit", "breath", "dysp", "cough", "wheeze", "tachy", "orthop", "cyanosis", "hemoptysis"],
    "Neuro": ["seiz", "confus", "stiff", "photophobia", "focal", "weak", "numb", "paral", "aphas", "headache", "migra", "dizzi", "syncope"],
    "GI": ["nausea", "vomit", "diarr", "abdominal", "abdomen", "constip", "hematem", "melena", "jaund", "appetite", "peristalsis", "pyloric", "mass"],
    "GU/Reproductive": ["vaginal", "dysuria", "urinary", "hematuria", "flank", "scrot", "testic", "hernia", "menstru", "pelvic"],
    "Systemic/Metabolic": ["fever", "chills", "rigor", "myalgia", "fatigue", "malaise", "sweat", "weight", "thirst", "dehydra", "acidosis", "kidney"],
    "Skin": ["rash", "lesion", "itch", "urticar", "hive", "erythema", "redness", "tenderness", "cellulitis", "skin"],
    "ENT/Voice": ["hoars", "vocal", "throat", "sinus", "rhino", "ear", "hearing"],
    "Eye": ["eye", "vision", "double", "depth", "tilting", "photophobia"],
    "MSK/Trauma": ["fracture", "sprain", "back", "joint", "arthral", "swelling", "bruise", "deform", "movement"],
    "Psych": ["anxiety", "panic", "fear", "depress", "insomnia", "psychosis"]
}

# -------------------------
# Safe CSV loading (cloud-friendly)
# -------------------------
def safe_load_csv(uploaded_file, csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV from an uploaded file-like or a local path.
    If not found, return a tiny demo DataFrame so the app can still boot.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            p = Path(csv_path)
            if p.exists():
                df = pd.read_csv(p, low_memory=False)
            else:
                # Minimal demo DF with one symptom + label
                df = pd.DataFrame({"Symptom_1": ["Fever"], "Disease": ["demo"]})
        if "_id" not in df.columns:
            df = df.copy()
            df["_id"] = np.arange(len(df))
        return df
    except Exception:
        # Last-resort demo DF if reading fails for any reason
        df = pd.DataFrame({"Symptom_1": ["Fever"], "Disease": ["demo"]})
        df["_id"] = np.arange(len(df))
        return df

def load_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Strict CSV loader (raises if file not found). Prefer safe_load_csv in Streamlit apps."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "_id" not in df.columns:
        df = df.copy()
        df["_id"] = np.arange(len(df))
    return df

# -------------------------
# One-hot encoding of symptoms
# -------------------------
def build_ohe(df: pd.DataFrame) -> pd.DataFrame:
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    if not symptom_cols:
        raise ValueError("No columns starting with 'Symptom_' found.")
    long = df.melt(id_vars=["_id"], value_vars=symptom_cols, var_name="slot", value_name="symptom")
    long["symptom"] = long["symptom"].astype(str).str.strip()
    long = long.replace({"symptom": {"nan": np.nan}}).dropna(subset=["symptom"])
    long["symptom"] = (
        long["symptom"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.title()
    )
    ohe = pd.crosstab(long["_id"], long["symptom"])
    ohe = (ohe > 0).astype(int)
    ohe = ohe.reindex(df["_id"], fill_value=0)
    ohe.index.name = "_id"
    return ohe

def map_constructs(ohe_cols) -> dict:
    assign = {}
    for s in ohe_cols:
        s_l = s.lower()
        hit = None
        for cons, kws in CONSTRUCT_KEYWORDS.items():
            if any(kw in s_l for kw in kws):
                hit = cons
                break
        assign[s] = hit if hit is not None else "Other/Unmapped"
    mapping = {}
    for s, cons in assign.items():
        mapping.setdefault(cons, []).append(s)
    return mapping

# -------------------------
# Construct scores (per-construct factor)
# -------------------------
def construct_scores(ohe: pd.DataFrame, mapping: dict):
    """
    Returns:
      FS: DataFrame with columns ['_id', <constructs...>] and NO 'Acuity' yet
      construct_models: dict describing how each construct score was computed
    """
    construct_models, scores = {}, {}
    for cons, syms in mapping.items():
        syms = [s for s in syms if s in ohe.columns]
        X = ohe[syms].astype(float) if syms else pd.DataFrame(index=ohe.index)
        keep = [c for c in X.columns if X[c].nunique() > 1]
        X = X[keep]
        if len(keep) == 0:
            s = pd.Series(0.0, index=ohe.index)
            construct_models[cons] = {"type": "zero", "cols": []}
        elif len(keep) == 1:
            col = keep[0]
            mu, sd = X[col].mean(), X[col].std() + 1e-6
            s = (X[col] - mu) / sd
            construct_models[cons] = {"type": "z1", "cols": [col], "mu": float(mu), "sd": float(sd)}
        else:
            svd = TruncatedSVD(n_components=1, random_state=42)
            z = svd.fit_transform(X.values)
            s = pd.Series(z[:, 0], index=ohe.index)
            construct_models[cons] = {"type": "svd1", "cols": keep, "svd": svd}
        scores[cons] = s
    FS = pd.DataFrame(scores, index=ohe.index).fillna(0.0).reset_index()
    return FS, construct_models

# -------------------------
# Acuity + split
# -------------------------
def fit_acuity_minmax(FS: pd.DataFrame, df_labels: pd.DataFrame, min_cases: int, test_size: float, seed=42):
    """
    Computes Acuity via min–max scaling on TRAIN ONLY. Returns train/test splits and scaling params.
    """
    construct_cols = [c for c in FS.columns if c not in ("Other/Unmapped", "Acuity", "_id")]
    data_full = FS.merge(df_labels[["_id", "Disease"]], on="_id", how="left").dropna(subset=["Disease"]).copy()
    counts = data_full["Disease"].value_counts()
    valid = counts[counts >= min_cases].index
    data_full = data_full[data_full["Disease"].isin(valid)].copy()

    tr_ids, te_ids = train_test_split(
        data_full["_id"], test_size=test_size, random_state=seed, stratify=data_full["Disease"]
    )
    train_slice = FS.loc[FS["_id"].isin(tr_ids), construct_cols]
    train_min = train_slice.min()
    train_max = train_slice.max()

    def compute_acuity_row(row):
        cols = [c for c in construct_cols if c in row.index]
        if not cols:
            return 0.0
        x = (row[cols] - train_min[cols]) / (train_max[cols] - train_min[cols] + 1e-6)
        return float(x.mean())

    FS = FS.copy()
    FS["Acuity"] = FS.apply(compute_acuity_row, axis=1)

    data = FS.merge(df_labels[["_id", "Disease"]], on="_id", how="left").dropna(subset=["Disease"]).copy()
    data = data[data["Disease"].isin(valid)].copy()

    mask_tr = data["_id"].isin(tr_ids)
    X_train = data.loc[mask_tr].drop(columns=["_id", "Disease"])
    y_train = data.loc[mask_tr, "Disease"].astype(str)
    X_test  = data.loc[~mask_tr].drop(columns=["_id", "Disease"])
    y_test  = data.loc[~mask_tr, "Disease"].astype(str)

    return {
        "FS": FS, "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "train_min": train_min, "train_max": train_max,
        "valid_classes": valid
    }

# -------------------------
# Train & evaluate RF
# -------------------------
def train_rf(X_train, y_train, n_estimators=200, max_depth=12, seed=42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1, class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_all(clf, X_test, y_test, X_train=None, y_train=None):
    pred = clf.predict(X_test)
    report_text = classification_report(y_test, pred, digits=3)
    report_dict = classification_report(y_test, pred, output_dict=True)
    acc = float((y_test.to_numpy() == pred).mean())
    cm = confusion_matrix(y_test, pred, labels=clf.classes_)

    # Filter for Top-k metrics (ensure y_test ⊆ clf.classes_)
    mask = y_test.isin(clf.classes_)
    X_topk = X_test.loc[mask]
    y_topk = y_test.loc[mask]
    dropped = int((~mask).sum())

    def topk_report(proba, classes, ks=(1, 3, 5)):
        vals = {}
        for k in ks:
            vals[f"Top-{k}"] = float(top_k_accuracy_score(y_topk, proba, k=k, labels=classes))
        # MRR
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[y] for y in y_topk])
        ranks = proba.argsort(axis=1)[:, ::-1]
        pos = (ranks == y_idx[:, None]).argmax(axis=1) + 1
        mrr = float(np.mean(1.0 / pos))
        return vals, mrr

    proba = clf.predict_proba(X_topk)
    topk, mrr = topk_report(proba, clf.classes_)

    # Optional calibrated metrics
    cal = None
    if (X_train is not None) and (y_train is not None):
        try:
            cal = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            cal.fit(X_train, y_train)
            proba_cal = cal.predict_proba(X_topk)
            topk_cal, mrr_cal = topk_report(proba_cal, cal.classes_)
        except Exception:
            cal, topk_cal, mrr_cal = None, None, None
    else:
        topk_cal, mrr_cal = None, None

    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "accuracy": acc,
        "cm_shape": cm.shape,
        "dropped_for_topk": dropped,
        "topk": topk, "mrr": mrr,
        "cal_model": cal,
        "topk_cal": topk_cal, "mrr_cal": mrr_cal
    }

# -------------------------
# Centroids & Predictor
# -------------------------
def make_centroids(X_train, y_train, classes, drop_cols=("Acuity",)):
    construct_only = [c for c in X_train.columns if c not in drop_cols]
    centroids = (
        X_train[construct_only]
        .assign(_y=y_train.values)
        .groupby("_y")
        .mean()
        .reindex(classes)
        .fillna(0.0)
    )
    arr = centroids.to_numpy()
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    unit = arr / norm
    return construct_only, centroids, unit

def make_predictor(ohe_columns, construct_models, clf, train_min, train_max, train_cols,
                   triage_func,
                   # smoothing knobs
                   use_smoothing=True, signal_threshold=0.05, prior_mix=0.7, sim_temp=0.5,
                   class_prior=None, centroids_unit=None, construct_only=None):
    classes_local = clf.classes_
    prior_local = class_prior
    centroids_unit_local = centroids_unit
    construct_only_local = construct_only or [c for c in train_cols if c != "Acuity"]

    def softmax(x, temp=1.0):
        z = (x / max(temp, 1e-9))
        z = z - z.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def predict_patient(symptoms):
        row = pd.DataFrame(0, index=[0], columns=ohe_columns)
        unseen = []
        for s in symptoms:
            s_norm = s.strip().title()
            if s_norm in row.columns:
                row.loc[0, s_norm] = 1
            else:
                unseen.append(s)

        # construct scores
        scores = {}
        for cons, meta in construct_models.items():
            t = meta["type"]; cols = meta["cols"]
            if t == "zero" or len(cols) == 0:
                scores[cons] = 0.0
            elif t == "z1":
                col = cols[0]; mu, sd = meta["mu"], meta["sd"]
                x = float(row[col].iloc[0]) if col in row.columns else 0.0
                scores[cons] = (x - mu) / sd
            elif t == "svd1":
                x_vec = np.array([[float(row[c].iloc[0]) if c in row.columns else 0.0 for c in cols]])
                z = meta["svd"].transform(x_vec)
                scores[cons] = float(z[0, 0])
            else:
                scores[cons] = 0.0

        FS_row = pd.DataFrame([scores])
        for c in train_min.index:
            if c not in FS_row.columns:
                FS_row[c] = 0.0
        acuity = ((FS_row[train_min.index] - train_min) / (train_max - train_min + 1e-6)).mean(axis=1).iloc[0]
        FS_row["Acuity"] = float(acuity)

        FS_row = FS_row.reindex(columns=train_cols, fill_value=0.0)
        proba = clf.predict_proba(FS_row)[0]

        # smoothing for low-signal
        signal = float(np.linalg.norm(FS_row[construct_only_local].to_numpy(dtype=float)))
        beta = float(np.clip((signal_threshold - signal) / max(signal_threshold, 1e-9), 0.0, 1.0)) if use_smoothing else 0.0
        if beta > 0 and prior_local is not None and centroids_unit_local is not None:
            pv = FS_row[construct_only_local].to_numpy(dtype=float).ravel()
            pv_unit = pv / (np.linalg.norm(pv) + 1e-9)
            sim = centroids_unit_local @ pv_unit
            # centroid prior
            prior_centroid = softmax(sim, temp=sim_temp)
            # global prior must match class order
            prior_blend = (1.0 - prior_mix) * prior_local + prior_mix * prior_centroid
            proba = (1.0 - beta) * proba + beta * prior_blend
            proba = proba / (proba.sum() + 1e-12)

        top_idx = np.argsort(proba)[::-1][:5]
        top_diseases = [(classes_local[i], float(proba[i])) for i in top_idx]

        return {
            "symptoms": symptoms,
            "unseen_symptoms": unseen,
            "predicted_disease": classes_local[top_idx[0]],
            "top_diseases": top_diseases,
            "acuity": float(acuity),
            "triage_level": triage_func(float(acuity)),
            "signal": signal,
            "beta_used": beta
        }

    return predict_patient

# -------------------------
# Triage thresholds
# -------------------------
def triage_thresholds_from_proportions(acuity_train: pd.Series, emerg_prop=0.10, urgent_prop=0.30):
    acuity = acuity_train.dropna()
    q_emerg  = acuity.quantile(1 - emerg_prop) if emerg_prop > 0 else float("inf")
    q_urgent = acuity.quantile(1 - (emerg_prop + urgent_prop)) if urgent_prop > 0 else float("-inf")
    if q_urgent > q_emerg:
        q_urgent, q_emerg = q_emerg, q_urgent
    def triage_func(a: float) -> str:
        if a >= q_emerg:  return "Emergency"
        if a >= q_urgent: return "Urgent"
        return "Routine"
    return float(q_emerg), float(q_urgent), triage_func

def triage_thresholds_fixed(X_train: pd.DataFrame, p_emerg=0.90, p_urgent=0.60):
    q_emerg = float(X_train["Acuity"].quantile(p_emerg))
    q_urgent = float(X_train["Acuity"].quantile(p_urgent))
    def triage_func(a: float) -> str:
        if a >= q_emerg:  return "Emergency"
        if a >= q_urgent: return "Urgent"
        return "Routine"
    return q_emerg, q_urgent, triage_func

# -------------------------
# Save / Load artifacts
# -------------------------
def save_artifacts(path: Union[str, Path], **kwargs):
    joblib.dump(kwargs, path)

def load_artifacts(path: Union[str, Path]):
    return joblib.load(path)

__all__ = [
    "safe_load_csv", "load_csv",
    "build_ohe", "map_constructs", "construct_scores",
    "fit_acuity_minmax", "train_rf", "evaluate_all",
    "make_centroids", "make_predictor",
    "triage_thresholds_from_proportions", "triage_thresholds_fixed",
    "save_artifacts", "load_artifacts",
    "CONSTRUCT_KEYWORDS",
]
