# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

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
df['symptom_list'] = df[symptom_cols].apply(lambda row: list(filter(pd.notnull, row)), axis=1)

# Drop rows with no symptoms or no disease label
df = df.dropna(subset=['Disease'])
df = df[df['symptom_list'].apply(len) > 0]

# Filter out diseases with only 1 occurrence
counts = df['Disease'].value_counts()
df = df[df['Disease'].isin(counts[counts > 1].index)]

# Encode symptoms and diseases
mlb = MultiLabelBinarizer(sparse_output=True)
X = mlb.fit_transform(df['symptom_list'])

le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost classifier
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

# Only use classes present in y_test
labels_in_test = sorted(set(y_test))
target_names_in_test = [le.classes_[i] for i in labels_in_test]

print(classification_report(
    y_test, y_pred,
    labels=labels_in_test,
    target_names=target_names_in_test,
    zero_division=0
))


# Evaluate
#y_pred = model.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))





# Samples
print(df['Disease'].value_counts())

min_samples = 10  # You can adjust this threshold
disease_counts = df['Disease'].value_counts()
rare_diseases = disease_counts[disease_counts < min_samples].index
df['Disease_grouped'] = df['Disease'].apply(lambda x: x if x not in rare_diseases else 'Other')

from sklearn.utils import resample

balanced_df = pd.DataFrame()
for disease in df['Disease_grouped'].unique():
    subset = df[df['Disease_grouped'] == disease]
    n_samples = min(len(subset), 100)  # max 100 samples per disease
    balanced_df = pd.concat([balanced_df, resample(subset, n_samples=n_samples, random_state=42)])
df = balanced_df.reset_index(drop=True)

df['num_symptoms'] = df['symptom_list'].apply(len)


# Use only a subset of your data (e.g., 10,000 rows)
df_small = df.sample(n=10000, random_state=42)
# Re-run your encoding and train-test split on df_small


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
# %%



###### UPDATE
# Use only a subset of your data (e.g., 10,000 rows)
df_small = df.sample(n=10000, random_state=42)
# Re-run your encoding and train-test split on df_small

model = RandomForestClassifier(n_estimators=10, random_state=42)

model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)

# Select top N symptoms
top_symptoms = df['symptom_list'].explode().value_counts().index[:50]
df['symptom_list'] = df['symptom_list'].apply(lambda x: [s for s in x if s in top_symptoms])

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))