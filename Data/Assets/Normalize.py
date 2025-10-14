
import re
import requests
import pandas as pd

# Display unique values in the 'Blood Pressure' column
# 0 Low
# 2 Normal  
# 1 High

def convert_categorical_columns(df):
    # Convert 'No'/'Yes' values to 0/1
    yes_no_columns = [
        col for col in df.columns 
        if set(df[col].unique()) == {'No', 'Yes'}
    ]
    for col in yes_no_columns:
        df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Convert 'Low'/'Normal'/'High' values to -1/0/1
    low_normal_high_columns = [
        col for col in df.columns
        if set(df[col].unique()) == {'Low', 'Normal', 'High'}
    ]
    for col in low_normal_high_columns:
        df[col] = df[col].map({'Low': -1, 'Normal': 0, 'High': 1})

    # Convert 'Positive'/'Negative' values if needed (mapping logic can be adjusted)
    variable_positive_columns = [
        col for col in df.columns
        if set(df[col].unique()) == {'Positive', 'Negative'}
    ]
    for col in variable_positive_columns:
        df[col] = df[col].map({'Negative': 0, 'Positive': 1})

    return df


def add_symptom_columns(df, symptom_cols):
    """
    For each row in df, extract symptoms where value is 'Yes' in symptom_cols,
    and add 'Blood Pressure'/'Cholesterol Level' if 'High'.
    Adds new columns Symptom_1, Symptom_2, ... to the DataFrame.
    Returns the modified DataFrame and the list of new symptom column names.
    """
    def extract_symptoms(row):
        symptoms = [col for col in symptom_cols if row[col] == 'Yes']
        if row.get('Blood Pressure', None) == 'High':
            symptoms.append('Blood Pressure')
        if row.get('Cholesterol Level', None) == 'High':
            symptoms.append('Cholesterol Level')
        return symptoms

    symptom_lists = df.apply(extract_symptoms, axis=1)
    max_symptoms = symptom_lists.apply(len).max()
    for i in range(max_symptoms):
        df[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')
    symptom_col_names = [f'Symptom_{i+1}' for i in range(max_symptoms)]
    return df, symptom_col_names


def split_symptoms_and_treatments(df_disease_symptoms):
    """
    Splits the 'Symptoms' and 'Treatments' columns into separate columns per item.
    Returns two DataFrames: (df_name_symptoms, df_name_treatments)
    """
    # --- Name and Symptoms ---
    symptom_lists = df_disease_symptoms['Symptoms'].str.split(',\s*')
    max_symptoms = symptom_lists.apply(len).max()
    df_name_symptoms = pd.DataFrame()
    df_name_symptoms['Name'] = df_disease_symptoms['Name']
    for i in range(max_symptoms):
        df_name_symptoms[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

    # --- Name and Treatments ---
    treatment_lists = df_disease_symptoms['Treatments'].fillna('').str.split(',\s*')
    max_treatments = treatment_lists.apply(len).max()
    df_name_treatments = pd.DataFrame()
    df_name_treatments['Name'] = df_disease_symptoms['Name']
    for i in range(max_treatments):
        df_name_treatments[f'Treatment_{i+1}'] = treatment_lists.apply(lambda x: x[i] if i < len(x) else '')

    return df_name_symptoms, df_name_treatments


def map_symptoms_from_binary(df, disease_col_candidates=['prognosis', 'diseases', 'Disease']):
    """
    Converts a DataFrame with binary symptom columns to a mapped DataFrame
    with 'Disease' and Symptom_1, Symptom_2, ... columns.
    Automatically detects the disease column from the provided candidates.
    """
    # Find the disease column
    disease_col = None
    for col in disease_col_candidates:
        if col in df.columns:
            disease_col = col
            break
    if disease_col is None:
        raise ValueError("No disease column found in DataFrame.")

    # Get all symptom columns (exclude disease column)
    symptom_columns = [col for col in df.columns if col.lower() not in [c.lower() for c in disease_col_candidates]]

    # Function to extract symptom names where value is 1
    def extract_symptoms(row):
        return [col for col in symptom_columns if row[col] == 1]

    # Apply the function to each row
    symptom_lists = df.apply(extract_symptoms, axis=1)
    max_symptoms = symptom_lists.apply(len).max()

    # Create new columns: Symptom_1, Symptom_2, ...
    for i in range(max_symptoms):
        df[f'Symptom_{i+1}'] = symptom_lists.apply(lambda x: x[i] if i < len(x) else '')

    # Create the new DataFrame with disease column as 'Disease' and symptom columns
    symptom_col_names = [f'Symptom_{i+1}' for i in range(max_symptoms)]
    df_symptoms_mapped = df[[disease_col] + symptom_col_names].rename(columns={disease_col: 'Disease'})
    return df_symptoms_mapped


def fetch_and_parse_disease_symptom_kb(url, show_tables=False):
    """
    Fetches and parses the DiseaseSymptomKB HTML tables from the given URL, cleans, and expands symptoms into columns.
    Removes UMLS prefixes from symptom names.
    Returns a DataFrame with columns: Disease, Symptom_1, Symptom_2, ...
    """
    response = requests.get(url)
    html_content = response.text

    # Parse all tables in the HTML
    tables = pd.read_html(html_content)

    if show_tables:
        for i, table in enumerate(tables):
            print(f"Table {i} shape: {table.shape}")
            print(table.head())

    # Use the first table by default
    df_disease_symptom_kb = tables[0]

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

    # Drop the original 'Symptom' list and 'Count' columns
    df_grouped = df_grouped.drop(columns=['Symptom', 'Count'])

    # Remove UMLS prefix from all string columns
    def remove_umls_prefix(value):
        if isinstance(value, str):
            return re.sub(r'UMLS:[A-Z0-9]+_', '', value)
        return value

    for col in df_grouped.columns:
        if df_grouped[col].dtype == object:
            df_grouped[col] = df_grouped[col].apply(remove_umls_prefix)

    return df_grouped