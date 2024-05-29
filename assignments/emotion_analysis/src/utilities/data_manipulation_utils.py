from pathlib import Path
from typing import Any, List

import pandas as pd

from .logger_utils import get_logger

logger = get_logger(__name__)


def save_first_n_rows_to_csv(df: pd.DataFrame, num_rows: int, file_path: Path) -> None:
    if not file_path:
        file_path = Path(".") / f"first_{num_rows}_rows.csv"
    df.head(num_rows).to_csv(file_path, index=False)


def load_csv_as_df(file_path: Path) -> pd.DataFrame:
    logger.info(f"Attempting to load csv from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded csv from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred when trying to load csv: {e}")
        raise e


def export_df_as_csv(df: pd.DataFrame, directory: Path, filename: str) -> None:
    """
    Export a pandas DataFrame as a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to be exported.
        directory (Path): The directory where the CSV file will be saved.
        filename (str): The name of the CSV file.

    Raises:
        PermissionError: If the function does not have permission to create the directory or write the file.
        OSError: If an OS error occurs when trying to create the directory.
        Exception: If an unexpected error occurs when trying to write to the file.
    """
    try:
        # Create the directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when trying to create directory: {directory}")
        return
    except OSError as e:
        logger.error(f"OS error occurred when trying to create directory: {e}")
        return
    
    # Check if filename ends with .csv, if not, append it
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Create the full file path
    file_path = directory / filename

    logger.info(f"Trying to export DataFrame to CSV: {file_path}")
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully exported DataFrame to CSV: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied when trying to write to file: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred when trying to write to file: {e}")


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    return df[column].unique()


def get_column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts()


def get_column_value_counts_by_group_as_percentage(
    df: pd.DataFrame, column_to_group: str, value_to_group_by: str
) -> pd.Series:
    return df.groupby(column_to_group)[value_to_group_by].value_counts(normalize=True)


def get_filenames_in_dir(directory: Path, list_sub_dirs=False) -> List[str]:
    directory = Path(directory)
    return (
        [x.name for x in directory.iterdir() if x.is_dir()]
        if list_sub_dirs
        else [x.name for x in directory.iterdir() if x.is_file()]
    )


def convert_column_to_data_type(
    df: pd.DataFrame, column: str, data_type: type
) -> pd.DataFrame:
    df[column] = df[column].astype(data_type)
    return df
