from pathlib import Path

import os
import pandas as pd

from logger_utils import get_logger

LOGGER = get_logger(__name__)


def save_first_n_rows_to_csv(df: pd.DataFrame, num_rows: int, file_path: Path) -> None:
    if not file_path:
        file_path = Path(".") / f"first_{num_rows}_rows.csv"
    df.head(num_rows).to_csv(file_path, index=False)


def load_csv(file_path: Path) -> pd.DataFrame:
    LOGGER.info(f"Attempting to load csv from {file_path}...")
    df = pd.read_csv(file_path)
    LOGGER.info(f"Successfully loaded csv from {file_path}")
    return df


def save_df_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    LOGGER.info(f"Attempting to save csv from pandas dataframe...")
    df.to_csv(file_path, index=False)
    LOGGER.info(f"CSV has been successfully saved to {file_path}!")


def get_unique_values(df: pd.DataFrame, column: str) -> list:
    return df[column].unique()


def get_column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts()


def get_column_value_counts_by_group_as_percentage(
    df: pd.DataFrame, column_to_group: str, value_to_group_by: str
) -> pd.Series:
    return df.groupby(column_to_group)[value_to_group_by].value_counts(normalize=True)


def get_filenames_in_dir(directory, list_sub_dirs=False):
    return (
        os.listdir(directory)
        if list_sub_dirs
        else [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
    )


def convert_column_to_data_type(
    df: pd.DataFrame, column: str, data_type: type
) -> pd.DataFrame:
    df[column] = df[column].astype(data_type)
    return df
