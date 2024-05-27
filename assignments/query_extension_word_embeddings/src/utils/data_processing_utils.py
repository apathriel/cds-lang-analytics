import difflib
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from .logging_utils import get_logger

from utils.utilities import (
    remove_punctuation_from_list,
    escape_punctuation_in_list,
)

logger = get_logger(__name__)


def load_csv_to_df(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    logger.info(f"Attempting to load dataset {file_path.name}...")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset {file_path.name} has been loaded successfully!")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the dataset: {e}")


def load_existing_df_or_create_new_df(
    data_path: Path, columns: List[str] = []
) -> pd.DataFrame:
    """
    Load an existing DataFrame from a CSV file if it exists, otherwise create a new DataFrame with the specified columns.

    Parameters:
        data_path (Path): The path to the CSV file.
        columns (List[str], optional): The list of column names for the new DataFrame. Defaults to an empty list.

    Returns:
        pd.DataFrame: The loaded DataFrame if the file exists, otherwise a new DataFrame with the specified columns.
    """
    if data_path.is_file():
        return load_csv_to_df(data_path)
    else:
        return pd.DataFrame(columns=columns)


def find_closest_match(
    df: pd.DataFrame, column_name: str, query: str, n: int = 1
) -> list[str]:
    """
    Finds the closest matches to a given query in a DataFrame column. If multiple matches are found, returns the top n matches. If no matches are found, returns an empty list.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to search.
        column_name (str): The name of the column to search.
        query (str): The query string to find matches for.
        n (int, optional): The number of closest matches to return. Defaults to 1.

    Returns:
        list[str]: A list of closest matches to the query.
    """
    unique_values = df[column_name].unique()
    closest_match = difflib.get_close_matches(query, unique_values, n=n)
    return closest_match


def get_unique_row_values_by_column(df: pd.DataFrame, column_name: str) -> np.ndarray:
    """
    Returns an array of unique values from a specific column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to extract values from.
        column_name (str): The name of the column to extract values from.

    Returns:
        np.ndarray: An array of unique values from the specified column.
    """
    return df[column_name].unique()


def get_num_rows(df: pd.DataFrame) -> int:
    """
    Returns the number of rows in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to calculate the number of rows.

    Returns:
    int: The number of rows in the DataFrame.
    """
    return df.shape[0]


def filter_df_rows_by_column_value(
    df: pd.DataFrame, column_name: str, value: Any, validate: bool = True
) -> pd.DataFrame:
    """
    Filter rows of a DataFrame based on a specific column value.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        column_name (str): The name of the column to filter on.
        value (Any): The value to filter for in the specified column.
        validate (bool, optional): Whether to validate if any rows are found. Defaults to True.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If validate is True and no rows are found with the specified column value.
    """
    filtered_df = df.loc[df[column_name] == value]

    if validate and filtered_df.empty:
        raise ValueError(f"No rows found with {column_name} equal to {value}")

    return filtered_df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with missing values from the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with empty rows dropped.
    """
    df = df.dropna(how="all")
    return df


def filter_df_by_term_occurance(
    df: pd.DataFrame,
    column_name: str,
    term_list: list,
    remove_punctuation: bool = False,
) -> pd.DataFrame:
    """
    Filter a DataFrame based on the occurrence of terms in a specific column.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        column_name (str): The name of the column to search for term occurrences.
        term_list (list): A list of terms to search for in the specified column.
        remove_punctuation (bool, optional): Whether to remove punctuation from the term list before searching. Defaults to False.

    Returns:
        pd.DataFrame: The filtered DataFrame containing rows where the specified terms occur in the specified column.
    """
    if df.empty:
        return df
    
    if remove_punctuation:
        pattern = "|".join(remove_punctuation_from_list(term_list))
    else:
        pattern = "|".join(escape_punctuation_in_list(term_list))

    return df[df[column_name].str.contains(pattern, na=False)]


def write_csv_to_file(
    df: pd.DataFrame, dir_path: Path, file_name: str, remove_empty_rows: bool = True
) -> None:
    """
    Write a DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to be written to the CSV file.
        dir_path (Path): The directory path where the CSV file will be saved.
        file_name (str): The name of the CSV file.
        remove_empty_rows (bool, optional): Whether to remove empty rows from the DataFrame before writing to the file. Defaults to True.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    if remove_empty_rows:
        df = drop_empty_rows(df)
    df.to_csv(dir_path / file_name, index=False)
