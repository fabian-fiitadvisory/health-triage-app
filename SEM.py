# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.decomposition import FactorAnalysis
import json
from sklearn.metrics import classification_report
# Advanced balancing with SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
#import sys
#!{sys.executable} -m pip install imbalanced-learn


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