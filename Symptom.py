# %%
import sys
import os
import re
import requests
import pandas as pd
# pip3 install pandas requests
# pip3 install requests

# Replace this with your actual absolute path to the Assets folder
sys.path.append(
    '/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Assets'
)
from Normalize import add_symptom_columns, split_symptoms_and_treatments, map_symptoms_from_binary


### 
#
# Load the dataset DISEASE SYMPTOM AND PATIENT PROFILE DATASET
#
###
url_disease_symptom_and_patient_profile_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms/'
    'Disease_symptom_and_patient_profile_dataset.csv'
)
df_disease_symptom_and_patient_profile_datase = pd.read_csv(url_disease_symptom_and_patient_profile_dataset)
print(df_disease_symptom_and_patient_profile_datase.head())

# List of symptom columns to check for 'Yes'
symptom_col_names = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
df_disease_symptom_and_patient_profile_datase, symptom_col_names = add_symptom_columns(
     df_disease_symptom_and_patient_profile_datase, symptom_col_names)
df_symptoms_only = df_disease_symptom_and_patient_profile_datase.drop_duplicates()



### 
# 
# Load the dataset for diseases and symptoms
#
###
url_disease_and_symptoms_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms/'
    'DiseaseAndSymptoms.csv'
)
df_disease_and_symptoms = pd.read_csv(url_disease_and_symptoms_dataset)
df_disease_and_symptoms = df_disease_and_symptoms.drop_duplicates()



###
#
# Load the dataset for diseases and symptoms with additional information

####
url_disease_symptoms_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms/'
    'Diseases_Symptoms.csv'
)
df_disease_symptoms = pd.read_csv(url_disease_symptoms_dataset)
df_name_symptoms, df_name_treatments = split_symptoms_and_treatments(df_disease_symptoms)






####
#
# Load the dataset for respiratory symptoms and treatment
#
####
url_respiratory_symptoms_and_treatment_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    '/respiratory_symptoms_and_treatment.csv'
)
df_respiratory_symptoms_and_treatment = pd.read_csv(
    url_respiratory_symptoms_and_treatment_dataset
)
# Only keep the 'Symptoms' and 'Disease' columns and rename 'Symptoms' to 'Symptom_1'
df_respiratory_symptoms_and_treatment = df_respiratory_symptoms_and_treatment[['Disease', 'Symptoms']].rename(columns={'Symptoms': 'Symptom_1'})



####
#
# Load the training data for diseases and symptoms
#
#####
url_training_data = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    '/training_data.csv'
)
df_training_data = pd.read_csv(
    url_training_data
)
print(df_training_data.head())

# Assume your DataFrame is named df_training_data and is already loaded

# Get all symptom columns (exclude 'prognosis' and any non-symptom columns)
symptom_columns = [col for col in df_training_data.columns if col not in ['prognosis']]

df_symptoms_mapped = map_symptoms_from_binary(df_training_data)



####
#
# Load the final augmented dataset for diseases and symptoms
#
#####
url_final_augmented_dataset_diseases_and_symptoms_data = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms/'
    'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
)
df_url_final_augmented_dataset_diseases_and_symptoms_data = pd.read_csv(url_final_augmented_dataset_diseases_and_symptoms_data)
print(df_url_final_augmented_dataset_diseases_and_symptoms_data.head())
df_final_augmented = map_symptoms_from_binary(df_url_final_augmented_dataset_diseases_and_symptoms_data)



####
#
# Load the Disease Symptom Knowledge Base from a URL
#
#####
url = "https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/"
response = requests.get(url)

html_content = response.text

# Parse all tables in the HTML
tables = pd.read_html(html_content)

# If there are multiple tables, inspect them and pick the right one
for i, table in enumerate(tables):
    print(f"Table {i} shape: {table.shape}")
    print(table.head())

# Example: If the first table is the one you want
df_disease_symptom_kb = tables[0]
print(df_disease_symptom_kb.head())


# Drop the header row and reset index
df_raw = df_disease_symptom_kb.iloc[1:].reset_index(drop=True)
df_raw.columns = ['Disease', 'Count', 'Symptom']

# Forward-fill the Disease and Count columns
df_raw['Disease'] = df_raw['Disease'].ffill()
df_raw['Count'] = df_raw['Count'].ffill()

# Drop rows where Symptom is NaN or empty
df_raw = df_raw[df_raw['Symptom'].notna() & (df_raw['Symptom'] != '')]

# Group symptoms by Disease
df_grouped = df_raw.groupby(['Disease', 'Count'])['Symptom'].apply(list).reset_index()

# Expand symptoms into separate columns
max_symptoms = df_grouped['Symptom'].apply(len).max()
for i in range(max_symptoms):
    df_grouped[f'Symptom_{i+1}'] = df_grouped['Symptom'].apply(lambda x: x[i] if i < len(x) else '')

# Drop the original 'Symptom' list column
df_grouped = df_grouped.drop(columns=['Symptom'])
df_grouped = df_grouped.drop(columns=['Count'])

print(df_grouped.head())

def remove_umls_prefix(value):
    if isinstance(value, str):
        return re.sub(r'UMLS:[A-Z0-9]+_', '', value)
    return value

# Apply to all string columns in df_grouped
for col in df_grouped.columns:
    if df_grouped[col].dtype == object:
        df_grouped[col] = df_grouped[col].apply(remove_umls_prefix)

print(df_grouped.head())









####
# # Symptom class definition
   #     return {
   #         'symptom_id': self.symptom_id,
   #         'name': self.name,
   #         'description': self.description
   #     }

    #@staticmethod
    #def from_dict(data: dict):
    #    return Symptom(
    #        symptom_id=data['symptom_id'],
    #        name=data['name'],
    #        description=data['description']
    #    )

    #@staticmethod
    #def from_dataframe(df: pd.DataFrame):
    #    return [Symptom.from_dict(row) for _, row in df.iterrows()]
    
    
    
    
# --- Combine all disease-symptom DataFrames into one main DataFrame ---

# 1. Patient profile dataset
df_symptoms_only_main = df_symptoms_only.copy()
if 'Name' in df_symptoms_only_main.columns:
    df_symptoms_only_main = df_symptoms_only_main.rename(columns={'Name': 'Disease'})
symptom_cols_1 = [col for col in df_symptoms_only_main.columns if col.startswith('Symptom_')]
df_symptoms_only_main = df_symptoms_only_main[['Disease'] + symptom_cols_1]

# 2. DiseaseAndSymptoms.csv
df_disease_and_symptoms_main = df_disease_and_symptoms.copy()
if 'Name' in df_disease_and_symptoms_main.columns:
    df_disease_and_symptoms_main = df_disease_and_symptoms_main.rename(columns={'Name': 'Disease'})
symptom_cols_2 = [col for col in df_disease_and_symptoms_main.columns if col.startswith('Symptom_')]
if symptom_cols_2:
    df_disease_and_symptoms_main = df_disease_and_symptoms_main[['Disease'] + symptom_cols_2]

# 3. Diseases_Symptoms.csv (split_symptoms_and_treatments)
df_name_symptoms_main = df_name_symptoms.copy()
if 'Name' in df_name_symptoms_main.columns:
    df_name_symptoms_main = df_name_symptoms_main.rename(columns={'Name': 'Disease'})
symptom_cols_3 = [col for col in df_name_symptoms_main.columns if col.startswith('Symptom_')]
df_name_symptoms_main = df_name_symptoms_main[['Disease'] + symptom_cols_3]

# 4. Respiratory symptoms and treatment
df_respiratory_main = df_respiratory_symptoms_and_treatment.copy()
symptom_cols_4 = [col for col in df_respiratory_main.columns if col.startswith('Symptom_')]
df_respiratory_main = df_respiratory_main[['Disease'] + symptom_cols_4]

# 5. Training data (binary to mapped)
df_symptoms_mapped_main = df_symptoms_mapped.copy()
symptom_cols_5 = [col for col in df_symptoms_mapped_main.columns if col.startswith('Symptom_')]
df_symptoms_mapped_main = df_symptoms_mapped_main[['Disease'] + symptom_cols_5]

# 6. Final augmented dataset
df_final_augmented_main = df_final_augmented.copy()
symptom_cols_6 = [col for col in df_final_augmented_main.columns if col.startswith('Symptom_')]
df_final_augmented_main = df_final_augmented_main[['Disease'] + symptom_cols_6]

# 7. Disease Symptom Knowledge Base (df_grouped)
df_kb_main = df_grouped.copy()
symptom_cols_7 = [col for col in df_kb_main.columns if col.startswith('Symptom_')]
df_kb_main = df_kb_main[['Disease'] + symptom_cols_7]

# --- Concatenate all, align columns, and drop duplicates ---
all_dfs = [
    df_symptoms_only_main,
    df_disease_and_symptoms_main,
    df_name_symptoms_main,
    df_respiratory_main,
    df_symptoms_mapped_main,
    df_final_augmented_main,
    df_kb_main
]

# Find all unique symptom columns across all DataFrames
all_symptom_cols = set()
for df in all_dfs:
    all_symptom_cols.update([col for col in df.columns if col.startswith('Symptom_')])
all_symptom_cols = sorted(all_symptom_cols)

# Reindex all DataFrames to have the same columns
for i in range(len(all_dfs)):
    all_dfs[i] = all_dfs[i].reindex(columns=['Disease'] + all_symptom_cols)

# Concatenate and clean
main_df = pd.concat(all_dfs, ignore_index=True)
main_df = main_df.fillna('')
main_df = main_df.drop_duplicates(subset=['Disease'] + all_symptom_cols).reset_index(drop=True)

# Rearrange main_df columns to start with Symptom_1, Symptom_2, ... then Disease
symptom_cols_sorted = sorted([col for col in main_df.columns if col.startswith('Symptom_')],
                             key=lambda x: int(x.split('_')[1]))
cols_order = symptom_cols_sorted + ['Disease']
main_df = main_df[cols_order]

print(main_df.head())
main_df.to_csv('All_data_disease_symptom.csv', index=False)
print("Combined main_df saved as All_data_disease_symptom.csv with reordered columns")
