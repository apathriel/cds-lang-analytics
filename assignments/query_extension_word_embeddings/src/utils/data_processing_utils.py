import difflib
import os
import pandas as pd

from utils.utilities import remove_punctuation_from_list, escape_punctuation_in_list

def load_csv_to_df(file_path: str):
    print(f"[SYSTEM] Attempting to load dataset {os.path.basename(file_path)}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"[SYSTEM] Dataset {os.path.basename(file_path)} has been loaded successfully!")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
def load_existing_df_or_create_new_df(data_path, columns=[]):
    if os.path.isfile(data_path):
        return load_csv_to_df(data_path)
    else:
        return pd.DataFrame(columns=columns)

def find_closest_match(df, column_name: str, query: str, n=1):
    unique_values = df[column_name].unique()
    closest_match = difflib.get_close_matches(query, unique_values, n=n)
    return closest_match

def get_unique_row_values_by_column(df, column_name: str):
    return df[column_name].unique()
    
def get_num_rows(df):
    return df.shape[0]

def filter_df_rows_by_column_value(df, column_name: str, value, validate=True):
    filtered_df = df.loc[df[column_name] == value]

    if validate and filtered_df.empty:
        raise ValueError(f"No rows found with {column_name} equal to {value}")

    return filtered_df

def drop_empty_rows(df):
    df = df.dropna(how='all')
    return df

def filter_df_by_term_occurance(df, column_name: str, term_list: list, remove_punctuation=False):
    if remove_punctuation:
        pattern = '|'.join(remove_punctuation_from_list(term_list))
    else:
        pattern = '|'.join(escape_punctuation_in_list(term_list))

    return df[df[column_name].str.contains(pattern, na=False)]

def write_csv_to_file(df, dir_path, file_name, remove_empty_rows=True):
    os.makedirs(dir_path, exist_ok=True)
    if remove_empty_rows:
        df = drop_empty_rows(df)
    df.to_csv(os.path.join(dir_path, file_name), index=False)