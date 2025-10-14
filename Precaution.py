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
# Load the dataset for diseases and symptoms with additional information

####
url_disease_symptoms_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms/'
    'Diseases_Symptoms.csv'
)
df_disease_symptoms = pd.read_csv(url_disease_symptoms_dataset)
df_name_symptoms, df_name_treatments = split_symptoms_and_treatments(df_disease_symptoms)
# Rename 'Name' to 'Disease' in df_name_treatments if present
if 'Name' in df_name_treatments.columns:
    df_name_treatments = df_name_treatments.rename(columns={'Name': 'Disease'})


####
#
# Load the dataset for respiratory symptoms and treatment
#
####
url_disease_precaution_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_precaution'
    '/Disease precaution.csv')
df_disease_precaution = pd.read_csv(url_disease_precaution_dataset)

# Rename columns Precaution_1, Precaution_2, ... to treatment_1, treatment_2, ...
precaution_cols = [col for col in df_disease_precaution.columns if col.lower().startswith('precaution_')]
rename_dict = {col: f"treatment_{i+1}" for i, col in enumerate(precaution_cols)}
df_disease_precaution = df_disease_precaution.rename(columns=rename_dict)
print(df_disease_precaution.head())


####
#
# Load the training data for diseases and symptoms
#
#####
url_independent_medical_reviews_data = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_precaution'
    '/Independent_Medical_Reviews.csv'
)
df_independent_medical_reviews_data = pd.read_csv(url_independent_medical_reviews_data)
print(df_independent_medical_reviews_data.head())

# Create a DataFrame with only category, diagnosis, and treatment columns if they exist
columns_to_keep = []
for col in ['Diagnosis Category', 'Treatment Category']:
    if col in df_independent_medical_reviews_data.columns:
        columns_to_keep.append(col)

df_diag_treat = df_independent_medical_reviews_data[columns_to_keep]

# Rename columns for consistency
df_diag_treat = df_diag_treat.rename(columns={
    'Diagnosis Category': 'Disease',
    'Treatment Category': 'treatment_1'
})
print(df_diag_treat.head())



####
#
# Load the final augmented dataset for diseases and symptoms
#
#####
url_medquad_data = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_precaution/'
    'medquad.csv'
)
df_medquad_data = pd.read_csv(url_medquad_data)
print(df_medquad_data.head())

# Create a DataFrame with only category, diagnosis, and treatment columns if they exist
columns_to_keep = []
for col in ['focus_area', 'answer']:
    if col in df_medquad_data.columns:
        columns_to_keep.append(col)

df_medquad = df_medquad_data[columns_to_keep]

# Rename columns for consistency
df_medquad = df_medquad.rename(columns={
    'focus_area': 'Disease',
    'answer': 'treatment_1'
})
print(df_medquad.head())




###########

# --- Combine all treatment/precaution DataFrames into one main DataFrame ---

# Ensure all DataFrames have the same columns: Disease, treatment_1, treatment_2, ...
dfs = []

# 1. From split_symptoms_and_treatments (df_name_treatments)
treat_cols_1 = [col for col in df_name_treatments.columns if col.startswith('treatment_')]
df_name_treatments_main = df_name_treatments[['Disease'] + treat_cols_1]
dfs.append(df_name_treatments_main)

# 2. From Disease precaution.csv
treat_cols_2 = [col for col in df_disease_precaution.columns if col.startswith('treatment_')]
df_disease_precaution_main = df_disease_precaution[['Disease'] + treat_cols_2]
dfs.append(df_disease_precaution_main)

# 3. From Independent Medical Reviews
treat_cols_3 = [col for col in df_diag_treat.columns if col.startswith('treatment_')]
df_diag_treat_main = df_diag_treat[['Disease'] + treat_cols_3]
dfs.append(df_diag_treat_main)

# 4. From medquad.csv
treat_cols_4 = [col for col in df_medquad.columns if col.startswith('treatment_')]
df_medquad_main = df_medquad[['Disease'] + treat_cols_4]
dfs.append(df_medquad_main)

# Find all unique treatment columns across all DataFrames
all_treat_cols = set()
for df in dfs:
    all_treat_cols.update([col for col in df.columns if col.startswith('treatment_')])
all_treat_cols = sorted(all_treat_cols, key=lambda x: int(x.split('_')[1]))

# Reindex all DataFrames to have the same columns
for i in range(len(dfs)):
    dfs[i] = dfs[i].reindex(columns=['Disease'] + all_treat_cols)

# Concatenate and clean
main_treatment_df = pd.concat(dfs, ignore_index=True)
main_treatment_df = main_treatment_df.fillna('')
main_treatment_df = main_treatment_df.drop_duplicates(subset=['Disease'] + all_treat_cols).reset_index(drop=True)

print(main_treatment_df.head())
main_treatment_df.to_csv('All_data_disease_treatment.csv', index=False)
print("Combined main_treatment_df saved as All_data_disease_treatment.csv")