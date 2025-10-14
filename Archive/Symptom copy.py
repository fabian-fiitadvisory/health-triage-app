# %%
import sys
import os
import pandas as pd
# pip3 install pandas requests
# pip3 install requests

# Replace this with your actual absolute path to the Assets folder
sys.path.append(
    '/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Assets'
)
from Normalize import convert_categorical_columns


### Load the dataset DISEASE SYMPTOM AND PATIENT PROFILE DATASET
url_disease_symptom_and_patient_profile_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    'Disease_symptom_and_patient_profile_dataset.csv'
)
df_disease_symptom_and_patient_profile_datase = pd.read_csv(
    url_disease_symptom_and_patient_profile_dataset
)
print(df_disease_symptom_and_patient_profile_datase.head())


# List of symptom columns to check for 'Yes'
symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']

# Function to extract symptom names for 'Yes' values
def extract_symptoms(row):
    symptoms = [col for col in symptom_cols if row[col] == 'Yes']
    if row['Blood Pressure'] == 'High':
        symptoms.append('Blood Pressure')
    if row['Cholesterol Level'] == 'High':
        symptoms.append('Cholesterol Level')
    return symptoms

# Apply the function to each row and expand into separate columns
symptom_lists = df_disease_symptom_and_patient_profile_datase.apply(extract_symptoms, axis=1)
max_symptoms = symptom_lists.apply(len).max()
for i in range(max_symptoms):
    df_disease_symptom_and_patient_profile_datase[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

# Create a new DataFrame with only the new Symptom columns
symptom_col_names = [f'Symptom_{i+1}' for i in range(max_symptoms)]
df_symptoms_only = df_disease_symptom_and_patient_profile_datase[['Disease'] + symptom_col_names]
# Remove duplicate rows from the symptoms-only DataFrame
df_symptoms_only = df_symptoms_only.drop_duplicates()



url_disease_and_symptoms_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    'DiseaseAndSymptoms.csv'
)
df_disease_and_symptoms = pd.read_csv(
    url_disease_and_symptoms_dataset
)
df_disease_and_symptoms = df_disease_and_symptoms.drop_duplicates()


url_disease_symptoms_dataset = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    '/Diseases_Symptoms.csv'
)
df_disease_symptoms = pd.read_csv(
    url_disease_symptoms_dataset
)
# --- Name and Symptoms ---
# Split Symptoms column by comma
symptom_lists = df_disease_symptoms['Symptoms'].str.split(',\s*')
max_symptoms = symptom_lists.apply(len).max()
symptom_cols = [f'Symptom_{i+1}' for i in range(max_symptoms)]
df_name_symptoms = pd.DataFrame()
df_name_symptoms['Name'] = df_disease_symptoms['Name']
for i in range(max_symptoms):
    df_name_symptoms[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

# --- Name and Treatments ---
# Split Treatments column by comma, handling NaN values
treatment_lists = df_disease_symptoms['Treatments'].fillna('').str.split(',\s*')
max_treatments = treatment_lists.apply(len).max()
treatment_cols = [f'Treatment_{i+1}' for i in range(max_treatments)]
df_name_treatments = pd.DataFrame()
df_name_treatments['Name'] = df_disease_symptoms['Name']
for i in range(max_treatments):
    df_name_treatments[f'Treatment_{i+1}'] = treatment_lists.apply(lambda x: x[i] if i < len(x) else '')

print(df_name_treatments.head())
# Show the results
print(df_name_symptoms.head())
print(df_name_treatments.head())





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
print(df_respiratory_symptoms_and_treatment.head())



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

# Function to extract symptom names where value is 1
def extract_symptoms(row):
    return [col for col in symptom_columns if row[col] == 1]

# Apply the function to each row
symptom_lists = df_training_data.apply(extract_symptoms, axis=1)
max_symptoms = symptom_lists.apply(len).max()

# Create new columns: Symptom_1, Symptom_2, ...
for i in range(max_symptoms):
    df_training_data[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

# Create the new DataFrame with 'prognosis' as 'Disease' and symptom columns
symptom_col_names = [f'Symptom_{i+1}' for i in range(max_symptoms)]
df_symptoms_mapped = df_training_data[['prognosis'] + symptom_col_names].rename(columns={'prognosis': 'Disease'})

print(df_symptoms_mapped.head())




url_final_augmented_dataset_diseases_and_symptoms_data = (
    r'/Users/Fabian/Library/CloudStorage/GoogleDrive-'
    'fabian.francisco@fiitadvisory.nl/Mijn Drive/Projects/Data/Disease_symptoms'
    '/Final_Augmented_dataset_Diseases_and_Symptoms.csv'
)
df_url_final_augmented_dataset_diseases_and_symptoms_data = pd.read_csv(
    url_final_augmented_dataset_diseases_and_symptoms_data
)
print(df_url_final_augmented_dataset_diseases_and_symptoms_data.head())


# Get all symptom columns (exclude 'diseases' or 'Disease' column)
symptom_columns = [col for col in df_url_final_augmented_dataset_diseases_and_symptoms_data.columns if col.lower() not in ['diseases', 'disease']]

# Function to extract symptom names where value is 1
def extract_symptoms(row):
    return [col for col in symptom_columns if row[col] == 1]

# Apply the function to each row
symptom_lists = df_url_final_augmented_dataset_diseases_and_symptoms_data.apply(extract_symptoms, axis=1)
max_symptoms = symptom_lists.apply(len).max()

# Create new columns: Symptom_1, Symptom_2, ...
for i in range(max_symptoms):
    df_url_final_augmented_dataset_diseases_and_symptoms_data[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

# Create the new DataFrame with 'diseases' as 'Disease' and symptom columns
disease_col = 'diseases' if 'diseases' in df_url_final_augmented_dataset_diseases_and_symptoms_data.columns else 'Disease'
symptom_col_names = [f'Symptom_{i+1}' for i in range(max_symptoms)]
df_final_augmented = df_url_final_augmented_dataset_diseases_and_symptoms_data[[disease_col] + symptom_col_names].rename(columns={disease_col: 'Disease'})

print(df_final_augmented.head())


import requests

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

import re

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