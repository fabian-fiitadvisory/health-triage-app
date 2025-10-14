# %%
# triage_pipeline.py
# End-to-end: load → wrangle symptoms → construct scoring → train RF → triage thresholds → predictor

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 0) Load data (single, robust)
# -----------------------------
CSV_PATH = Path("All_data_disease_symptom.csv")
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Loaded:", CSV_PATH, "shape:", df.shape)
print(df.head(2))

# ------------------------------------------------
# 1) Wide → long → one-hot of symptoms per patient
# ------------------------------------------------
# Identify symptom columns
symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
if not symptom_cols:
    raise ValueError("No columns starting with 'Symptom_' found.")

# Add row id
df = df.copy()
df["_id"] = np.arange(len(df))

# Long format
long = df.melt(
    id_vars=["_id"],
    value_vars=symptom_cols,
    var_name="slot",
    value_name="symptom"
)
# Clean symptom strings
long["symptom"] = long["symptom"].astype(str).str.strip()
long = long.replace({"symptom": {"nan": np.nan}}).dropna(subset=["symptom"])
long["symptom"] = (
    long["symptom"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

# One-hot: rows = patients, cols = symptoms
ohe = pd.crosstab(long["_id"], long["symptom"])
ohe = (ohe > 0).astype(int)
# Ensure all rows
ohe = ohe.reindex(index=df["_id"], fill_value=0)
print("One-hot shape:", ohe.shape)

# ---------------------------------------
# 2) Keyword-based construct assignments
# ---------------------------------------
construct_keywords = {
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

# Assign each symptom to a construct
assign = {}
for s in ohe.columns:
    s_l = s.lower()
    hit = None
    for cons, kws in construct_keywords.items():
        if any(kw in s_l for kw in kws):
            hit = cons
            break
    assign[s] = hit if hit is not None else "Other/Unmapped"

# Build mapping: construct -> list of symptoms
mapping = {}
for s, cons in assign.items():
    mapping.setdefault(cons, []).append(s)

# Optional: save mapping
with open("datadriven_constructs.json", "w") as f:
    json.dump(mapping, f, indent=2)
print("Constructs:", list(mapping.keys()))




# Ensure df has a clean _id column
if "_id" not in df.columns:
    df = df.copy()
    df["_id"] = np.arange(len(df))

# Ensure ohe index is aligned to df["_id"] and named "_id"
# (This guarantees every row in df exists in ohe, even if all-zeros)
ohe = ohe.reindex(df["_id"], fill_value=0)
ohe.index.name = "_id"

# ---------------------------
# 3) Construct scores + models
# ---------------------------
construct_models = {}
scores = {}

for cons, syms in mapping.items():
    # keep only known symptom columns
    syms = [s for s in syms if s in ohe.columns]
    X = ohe[syms].astype(float) if syms else pd.DataFrame(index=ohe.index)

    # drop zero-variance cols
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
        z = svd.fit_transform(X.values)  # (n_samples, 1)
        s = pd.Series(z[:, 0], index=ohe.index)
        construct_models[cons] = {"type": "svd1", "cols": keep, "svd": svd}

    scores[cons] = s

# Build FS with SAME index as ohe (_id)
FS = pd.DataFrame(scores, index=ohe.index).fillna(0.0)

# Make _id a plain column (not index). After this, FS DEFINITELY has an "_id" column.
FS = FS.reset_index()               # brings index named "_id" out as a column
assert "_id" in FS.columns, "Failed to materialize _id as a column in FS"

# --------------------------------------------
# 4) Build Acuity (min–max fit on TRAIN only)
# --------------------------------------------
# Exclude non-construct columns from Acuity
construct_cols = [c for c in FS.columns if c not in ("Other/Unmapped", "Acuity", "_id")]

# Merge labels, drop rows without Disease
if "Disease" not in df.columns:
    raise ValueError("df must contain a 'Disease' column for supervised training.")
data_full = FS.merge(df[["_id", "Disease"]], on="_id", how="left").dropna(subset=["Disease"]).copy()

# Filter rare diseases BEFORE the split
min_cases = 200 #50
valid = data_full["Disease"].value_counts()
valid = valid[valid >= min_cases].index
data_full = data_full[data_full["Disease"].isin(valid)].copy()
print(f"Kept {len(valid)} diseases (>= {min_cases}). Rows: {len(data_full):,}")


# Split on IDs (so we can fit scalers on TRAIN only)
tr_ids, te_ids = train_test_split(
    data_full["_id"], test_size=0.25, random_state=42, stratify=data_full["Disease"]
)

# Fit min–max on TRAIN constructs only (aligned columns)
train_slice = FS.loc[FS["_id"].isin(tr_ids), construct_cols]
train_min = train_slice.min()
train_max = train_slice.max()

def compute_acuity_row(row: pd.Series) -> float:
    cols = [c for c in construct_cols if c in row.index]
    if not cols:
        return 0.0
    x = (row[cols] - train_min[cols]) / (train_max[cols] - train_min[cols] + 1e-6)
    return float(x.mean())

FS["Acuity"] = FS.apply(compute_acuity_row, axis=1)

# Rebuild labeled data and split features/labels
data = FS.merge(df[["_id", "Disease"]], on="_id", how="left").dropna(subset=["Disease"]).copy()

mask_tr = data["_id"].isin(tr_ids)
X_train = data.loc[mask_tr].drop(columns=["_id", "Disease"])
y_train = data.loc[mask_tr, "Disease"].astype(str)

X_test  = data.loc[~mask_tr].drop(columns=["_id", "Disease"])
y_test  = data.loc[~mask_tr, "Disease"].astype(str)

print("Sanity — FS has _id column:", "_id" in FS.columns)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Acuity stats (train):\n", X_train["Acuity"].describe())
print("Acuity stats (test):\n", X_test["Acuity"].describe())
# -------------------------
# 5) Train RF + evaluation
# -------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("\nDisease classification report:")
print(classification_report(y_test, pred, digits=3))
cm = confusion_matrix(y_test, pred, labels=clf.classes_)
print("Confusion matrix shape:", cm.shape)

# ==========================================
# 6) Evaluate (Top-1 + Top-k + MRR)
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.calibration import CalibratedClassifierCV

pred = clf.predict(X_test)
print("\nDisease classification report (Top-1):")
print(classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred, labels=clf.classes_)
print("Confusion matrix shape:", cm.shape)

# --- Filter test rows to classes known by the model (fix for top_k_accuracy_score) ---
valid_mask_topk = y_test.isin(clf.classes_)
n_dropped = (~valid_mask_topk).sum()
if n_dropped > 0:
    print(f"[Top-k] Dropping {n_dropped} test rows with unseen classes (not in clf.classes_)")
X_test_topk = X_test.loc[valid_mask_topk]
y_test_topk = y_test.loc[valid_mask_topk]


def topk_report(y_true, proba, classes, ks=(1,3,5)):
    # Top-k accuracies
    for k in ks:
        acc_k = top_k_accuracy_score(y_true, proba, k=k, labels=classes)
        print(f"Top-{k} accuracy: {acc_k:.3f}")
    # Mean Reciprocal Rank (MRR)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true])
    ranks = proba.argsort(axis=1)[:, ::-1]  # descending
    pos = (ranks == y_idx[:, None]).argmax(axis=1) + 1  # 1-based
    mrr = np.mean(1.0 / pos)
    print(f"MRR: {mrr:.3f}")
    # Hit@k table
    tbl = []
    for k in ks:
        hits = (pos <= k).mean()
        tbl.append([k, round(hits, 4)])
    df_hits = pd.DataFrame(tbl, columns=["k", "hit@k"])
    print("\nHit@k:\n", df_hits.to_string(index=False))

# Uncalibrated
proba = clf.predict_proba(X_test_topk)
print("\n== Uncalibrated probabilities ==")
topk_report(y_test_topk.to_numpy(), proba, clf.classes_, ks=(1,3,5))

# Optional: calibrated
try:
    cal_clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    cal_clf.fit(X_train, y_train)
    proba_cal = cal_clf.predict_proba(X_test_topk)
    print("\n== Calibrated probabilities (isotonic) ==")
    topk_report(y_test_topk.to_numpy(), proba_cal, cal_clf.classes_, ks=(1,3,5))
except Exception as e:
    print("\n[Calibration skipped]", e)

# ==========================================
# 7) Triage thresholds
# ==========================================
# (A) Fixed percentile cutoffs from TRAIN (90%/60%)

q_emerg_fixed  = float(X_train["Acuity"].quantile(0.90))  # top 10%
q_urgent_fixed = float(X_train["Acuity"].quantile(0.60))  # next 30%

def triage_from_acuity_fixed(a: float) -> str:
    if a >= q_emerg_fixed:
        return "Emergency"
    if a >= q_urgent_fixed:
        return "Urgent"
    return "Routine"

tmp = X_test.copy()
tmp["triage_level"] = tmp["Acuity"].apply(triage_from_acuity_fixed)
print("\nSample triage levels on test (fixed 90/60):")
print(tmp[["Acuity", "triage_level"]].head(10))

# (B) Proportion-based helper
def make_triage_by_proportion(acuity_train: pd.Series, emerg_prop: float = 0.10, urgent_prop: float = 0.30):
    if not (0 <= emerg_prop < 1) or not (0 <= urgent_prop < 1):
        raise ValueError("Proportions must be in [0, 1).")
    if emerg_prop + urgent_prop >= 1:
        raise ValueError("emerg_prop + urgent_prop must be < 1.")
    acuity = acuity_train.dropna()
    q_emerg  = acuity.quantile(1 - emerg_prop) if emerg_prop > 0 else float("inf")
    q_urgent = acuity.quantile(1 - (emerg_prop + urgent_prop)) if urgent_prop > 0 else float("-inf")
    if q_urgent > q_emerg:
        q_urgent, q_emerg = q_emerg, q_urgent
    def triage_from_acuity(a: float) -> str:
        if a >= q_emerg:  return "Emergency"
        if a >= q_urgent: return "Urgent"
        return "Routine"
    return float(q_emerg), float(q_urgent), triage_from_acuity

q_emerg_prop, q_urgent_prop, triage_from_acuity = make_triage_by_proportion(
    X_train["Acuity"], emerg_prop=0.10, urgent_prop=0.30
)
print(f"\nProportion-based thresholds → Emergency: {q_emerg_prop:.6f}, Urgent: {q_urgent_prop:.6f}")

train_triage = X_train["Acuity"].apply(triage_from_acuity).value_counts(normalize=True)
test_triage  = X_test["Acuity"].apply(triage_from_acuity).value_counts(normalize=True)
print("Train triage %:\n", (train_triage*100).round(2))
print("Test triage %:\n", (test_triage*100).round(2))


# ==========================================
# 8) Patient-level predictor (production)
# ==========================================
train_cols = X_train.columns.tolist()   # exact RF feature order
ohe_vocab  = ohe.columns.tolist()       # known symptom vocab

def make_patient_predictor(ohe_columns, construct_models, clf, train_min, train_max, train_cols,
                           triage_func=triage_from_acuity_fixed):
    def predict_patient(symptoms):
        # 1) OHE row
        row = pd.DataFrame(0, index=[0], columns=ohe_columns)
        unseen = []
        for s in symptoms:
            s_norm = s.strip().title()
            if s_norm in row.columns:
                row.loc[0, s_norm] = 1
            else:
                unseen.append(s)

        # 2) Construct scores using SAME transformers
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
                z = meta["svd"].transform(x_vec)  # (1,1)
                scores[cons] = float(z[0, 0])
            else:
                scores[cons] = 0.0

        FS_row = pd.DataFrame([scores])

        # 3) Acuity via TRAIN min–max across construct_cols
        for c in train_min.index:
            if c not in FS_row.columns:
                FS_row[c] = 0.0
        acuity = ((FS_row[train_min.index] - train_min) / (train_max - train_min + 1e-6)).mean(axis=1).iloc[0]
        FS_row["Acuity"] = float(acuity)

        # 4) Align to training feature order
        FS_row = FS_row.reindex(columns=train_cols, fill_value=0.0)

        # 5) Predict disease + top-3
        proba = clf.predict_proba(FS_row)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_diseases = [(clf.classes_[i], float(proba[i])) for i in top_idx]

        return {
            "symptoms": symptoms,
            "unseen_symptoms": unseen,
            "predicted_disease": clf.classes_[top_idx[0]],
            "top_diseases": top_diseases,
            "acuity": float(acuity),
            "triage_level": triage_func(float(acuity))
        }
    return predict_patient

# Choose which triage function you prefer in production:
# - triage_from_acuity_fixed (90/60)
# - triage_from_acuity (proportion-based)
predict_patient = make_patient_predictor(
    ohe_vocab, construct_models, clf, train_min, train_max, train_cols,
    triage_func=triage_from_acuity_fixed  # or triage_from_acuity
)

# Smoke tests
_examples = [
    ["Fever", "Cough", "Palpitations"],
    ["Hoarseness", "Vocal Changes", "Vocal Fatigue"],
    ["Pain", "Swelling", "Bruising", "Deformity", "Difficulty moving"],
]
for ex in _examples:
    print("\nPredict for:", ex)
    print(predict_patient(ex))



# ==========================================
# 9) (Optional) Persist artifacts
# ==========================================
try:
    import joblib
    joblib.dump({
        "clf": clf,
        "ohe_vocab": ohe_vocab,
        "construct_models": construct_models,
        "train_cols": train_cols,
        "train_min": train_min,
        "train_max": train_max,
        "q_emerg_fixed": q_emerg_fixed,
        "q_urgent_fixed": q_urgent_fixed
    }, "triage_model_artifacts.joblib")
    print("\nSaved triage_model_artifacts.joblib")
except Exception as e:
    print("Skipping artifact save:", e)


#######################

# -------------------------------
# 6) Data-driven triage thresholds
# -------------------------------
q_emerg  = float(X_train["Acuity"].quantile(0.90))  # top 10%
q_urgent = float(X_train["Acuity"].quantile(0.60))  # next 30%

def triage_from_acuity(a: float) -> str:
    if a >= q_emerg:
        return "Emergency"
    if a >= q_urgent:
        return "Urgent"
    return "Routine"

# Quick preview on test set
tmp = X_test.copy()
tmp["triage_level"] = tmp["Acuity"].apply(triage_from_acuity)
print("\nSample triage levels on test:")
print(tmp[["Acuity", "triage_level"]].head(10))

# ------------------------------------------
# 7) Patient-level predictor (production)
# ------------------------------------------
train_cols = X_train.columns.tolist()   # exact feature order for RF
ohe_vocab  = ohe.columns.tolist()       # known symptom vocabulary

def make_patient_predictor(ohe_columns, construct_models, clf, train_min, train_max, train_cols):
    def predict_patient(symptoms):
        # 1) One-hot encode the patient
        row = pd.DataFrame(0, index=[0], columns=ohe_columns)
        unseen = []
        for s in symptoms:
            s_norm = s.strip().title()
            if s_norm in row.columns:
                row.loc[0, s_norm] = 1
            else:
                unseen.append(s)

        # 2) Construct scores using the SAME transformers as training
        scores = {}
        for cons, meta in construct_models.items():
            t = meta["type"]
            cols = meta["cols"]
            if t == "zero" or len(cols) == 0:
                scores[cons] = 0.0
            elif t == "z1":
                col = cols[0]
                mu, sd = meta["mu"], meta["sd"]
                x = float(row[col].iloc[0]) if col in row.columns else 0.0
                scores[cons] = (x - mu) / sd
            elif t == "svd1":
                x_vec = np.array([[float(row[c].iloc[0]) if c in row.columns else 0.0 for c in cols]])
                z = meta["svd"].transform(x_vec)  # (1,1)
                scores[cons] = float(z[0, 0])
            else:
                # placeholder for any other transformer types
                scores[cons] = 0.0

        FS_row = pd.DataFrame([scores])

        # 3) Compute Acuity via TRAIN min–max over construct_cols
        for c in train_min.index:
            if c not in FS_row.columns:
                FS_row[c] = 0.0
        acuity = ((FS_row[train_min.index] - train_min) / (train_max - train_min + 1e-6)).mean(axis=1).iloc[0]
        FS_row["Acuity"] = float(acuity)

        # 4) Align features to RF training order
        FS_row = FS_row.reindex(columns=train_cols, fill_value=0.0)

        # 5) Predict disease + top-3
        proba = clf.predict_proba(FS_row)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_diseases = [(clf.classes_[i], float(proba[i])) for i in top_idx]

        return {
            "symptoms": symptoms,
            "unseen_symptoms": unseen,
            "predicted_disease": clf.classes_[top_idx[0]],
            "top_diseases": top_diseases,
            "acuity": float(acuity),
            "triage_level": triage_from_acuity(float(acuity))
        }
    return predict_patient

predict_patient = make_patient_predictor(
    ohe_vocab, construct_models, clf, train_min, train_max, train_cols
)

# ----------------------------
# 8) Quick manual test inputs
# ----------------------------
examples = [
    ["Fever", "Cough", "Palpitations"],
    ["Hoarseness", "Vocal Changes", "Vocal Fatigue"],
    ["Pain", "Swelling", "Bruising", "Deformity", "Difficulty moving"],
]
for ex in examples:
    print("\nExample:", ex)
    print(predict_patient(ex))

# --------------------------------
# 9) (Optional) Persist artifacts
# --------------------------------
try:
    import joblib
    joblib.dump({
        "clf": clf,
        "ohe_vocab": ohe_vocab,
        "construct_models": construct_models,
        "train_cols": train_cols,
        "train_min": train_min,
        "train_max": train_max,
        "q_emerg": q_emerg,
        "q_urgent": q_urgent,
    }, "triage_model_artifacts.joblib")
    print("\nSaved triage_model_artifacts.joblib")
except Exception as e:
    print("Skipping artifact save:", e)





# Load your data
df = pd.read_csv("All_data_disease_symptom.csv")
csv_all_data_disease_symptom = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/'
    'All_data_disease_symptom.csv'
)
df = pd.read_csv(csv_all_data_disease_symptom)
print(df.head())

# Combine symptom columns into a list
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
# 2) Identify symptom columns
symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
print("Symptom columns:", symptom_cols)

# 3) Add row id
df["_id"] = np.arange(len(df))

# 4) Reshape from wide (Symptom_1…Symptom_28) to long format
long = df.melt(
    id_vars=["_id"],
    value_vars=symptom_cols,
    var_name="slot",
    value_name="symptom"
)

# 5) Clean symptom strings
long["symptom"] = long["symptom"].astype(str).str.strip()
long = long.replace({"symptom": {"nan": np.nan}}).dropna(subset=["symptom"])
long["symptom"] = (
    long["symptom"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)   # collapse multiple spaces
    .str.title()                            # title case for consistency
)

# 6) Convert to one-hot: rows = patients, columns = symptom names
ohe = pd.crosstab(long["_id"], long["symptom"])
ohe = (ohe > 0).astype(int)

print("One-hot shape:", ohe.shape)
print(ohe.head())



# 1) Define keyword rules for constructs
construct_keywords = {
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

# 2) Assign each symptom column to a construct
assign = {}
for s in ohe.columns:   # iterate over symptom names from Step 1
    s_l = s.lower()
    hit = None
    for cons, kws in construct_keywords.items():
        if any(kw in s_l for kw in kws):
            hit = cons
            break
    assign[s] = hit if hit is not None else "Other/Unmapped"

# 3) Build mapping dict: construct -> list of symptoms
mapping = {}
for s, cons in assign.items():
    mapping.setdefault(cons, []).append(s)

# 4) Save mapping (optional, to edit later)
with open("datadriven_constructs.json", "w") as f:
    json.dump(mapping, f, indent=2)

print(json.dumps(mapping, indent=2)[:800])  # preview first part


##### STEP 3  

# 1) Load mapping from Step 2
with open("datadriven_constructs.json") as f:
    mapping = json.load(f)

scores = {}  # dict of construct -> factor scores

# 2) For each construct, compute factor score
for cons, syms in mapping.items():
    # Keep only symptoms that exist in the one-hot dataframe
    syms = [s for s in syms if s in ohe.columns]
    X = ohe[syms].astype(float)
    
    # Drop zero-variance columns (otherwise FactorAnalysis fails)
    keep = [c for c in X.columns if X[c].nunique() > 1]
    X = X[keep]

    if X.shape[1] == 0:
        # No valid columns → flat score
        s = pd.Series(0.0, index=ohe.index)
    elif X.shape[1] == 1:
        # Only one column → standardize
        col = X.columns[0]
        s = (X[col] - X[col].mean()) / (X[col].std() + 1e-6)
    else:
        # Factor Analysis (1 factor)
        fa = FactorAnalysis(n_components=1, random_state=42)
        z = fa.fit_transform(X.values)
        s = pd.Series(z[:, 0], index=ohe.index)
    
    scores[cons] = s

# 3) Collect construct-level scores
FS = pd.DataFrame(scores).fillna(0.0)

# 4) Second-order latent variable: Acuity
construct_cols = [c for c in FS.columns if c != "Other/Unmapped"]
FSz = (FS[construct_cols] - FS[construct_cols].mean()) / (FS[construct_cols].std() + 1e-6)
FS["Acuity"] = FSz.mean(axis=1)

print(FS.head())



### STeP 4

# If df / FS aren't already in memory, uncomment and load:
# import pandas as pd
# df = pd.read_csv("All_data_disease_symptom.csv", low_memory=False)
# FS = pd.read_csv("sem_factor_scores_python.csv")  # or ..._subset.csv if you used a subset

# 1) Recompute Acuity from construct scores currently in FS
construct_cols = [c for c in FS.columns if c not in ("Other/Unmapped", "Acuity")]

# Simple min-max scaling across constructs
train_min = FS[construct_cols].min()
train_max = FS[construct_cols].max()

def compute_acuity(FS_row):
    FS_scaled = (FS_row[construct_cols] - train_min) / (train_max - train_min + 1e-6)
    return FS_scaled.mean()

FS["Acuity"] = FS.apply(compute_acuity, axis=1)

print("Construct scores + Acuity (preview):")
print(FS.head())

# 2) Ensure _id is a clean column (not both index and column)
if "_id" in FS.columns:
    if FS.index.name == "_id":
        FS = FS.reset_index(drop=True)
else:
    FS = FS.reset_index().rename(columns={"index": "_id"})

# Make sure df has matching _id values
df = df.copy()
df["_id"] = np.arange(len(df))

# 3) Merge labels (Disease) onto FS
data = FS.merge(df[["_id", "Disease"]], on="_id", how="left").dropna(subset=["Disease"])

# 🔧 NEW: filter out very rare diseases (too few samples)
min_cases = 50  # adjust depending on dataset size
counts = data["Disease"].value_counts()
valid_diseases = counts[counts >= min_cases].index
data = data[data["Disease"].isin(valid_diseases)]

# 4) Train/test split
X = data.drop(columns=["_id", "Disease"])
y = data["Disease"].astype(str)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=None
    )

# 5) Train a compact Random Forest
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train)

# 6) Evaluate disease prediction
pred = clf.predict(X_test)
print("\nDisease classification report:")
print(classification_report(y_test, pred))

# 7) Apply triage from Acuity
def triage_from_acuity(a: float) -> str:
    if a >= 0.7:
        return "Emergency"
    if a >= 0.3:
        return "Urgent"
    return "Routine"

data["triage_level"] = data["Acuity"].apply(triage_from_acuity)

print("\nSample triage assignments:")
print(data[["Disease", "Acuity", "triage_level"]].head(10))






# STEP 5

def triage_from_acuity(a: float) -> str:
    if a >= 1.0:
        return "Emergency"
    if a >= 0.3:
        return "Urgent"
    return "Routine"

def make_patient_predictor(ohe_columns, construct_mapping, clf, train_means, train_stds):
    """
    Returns a function predict_patient(symptoms) that uses training distribution for Acuity.
    
    ohe_columns: list of all possible symptom names
    construct_mapping: dict {construct: [symptom names]} from Step 2
    clf: trained classifier from Step 4
    train_means, train_stds: Series from training set construct scores
    """
    construct_cols = train_means.index.tolist()

    def predict_patient(symptoms):
        # 1) One-hot encode the patient input
        row = pd.DataFrame(0, index=[0], columns=ohe_columns)
        for s in symptoms:
            s_norm = s.strip().title()
            if s_norm in row.columns:
                row.loc[0, s_norm] = 1

        # 2) Compute construct scores
        scores = {}
        for cons, syms in construct_mapping.items():
            syms = [s for s in syms if s in row.columns]
            if len(syms) == 0:
                scores[cons] = 0.0
            else:
                scores[cons] = row[syms].mean(axis=1).iloc[0]

        FS = pd.DataFrame([scores])

        # 3) Compute Acuity using training distribution
        FSz = (FS[construct_cols] - train_means) / train_stds
        FS["Acuity"] = FSz.mean(axis=1)

        # 4) Predict disease
        disease_pred = clf.predict(FS)[0]
        proba = clf.predict_proba(FS)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_diseases = [(clf.classes_[i], float(proba[i])) for i in top_idx]

        # 5) Predict triage
        acuity = FS["Acuity"].iloc[0]
        triage_level = triage_from_acuity(acuity)

        return {
            "symptoms": symptoms,
            "predicted_disease": disease_pred,
            "top_diseases": top_diseases,
            "acuity": float(acuity),
            "triage_level": triage_level
        }

    return predict_patient




# construct_cols = all feature columns used for classifier (Step 4, before training)
construct_cols = [c for c in X_train.columns if c != "Acuity"]

# Save training distribution from Step 4
train_means = X_train[construct_cols].mean()
train_stds = X_train[construct_cols].std() + 1e-6

# Build patient predictor
predict_patient = make_patient_predictor(ohe.columns.tolist(), mapping, clf, train_means, train_stds)


example_symptoms = ["Fever", "Cough", "Palpitations"]
result = predict_patient(example_symptoms)
print(result)


example_symptoms = ["Hoarseness", "Vocal Changes", "Vocal Fatigue"]
result = predict_patient(example_symptoms)
print(result)


example_symptoms = ["Pain", "Swelling", "Bruising", "Deformity", "Difficulty moving"]
result = predict_patient(example_symptoms)
print(result)