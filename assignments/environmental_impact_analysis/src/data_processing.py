from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import glob

from logger_utils import get_logger

logger = get_logger(__name__)

def load_csv_as_df(file_path: Path, logging_enabled: bool = False) -> pd.DataFrame:
    if logging_enabled:
        logger.info(f"Attempting to load csv from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if logging_enabled:
            logger.info(f"Successfully loaded csv from {file_path}")
        return df
    except FileNotFoundError:
        if logging_enabled:
            logger.error(f"File not found: {file_path}")
    except Exception as e:
        if logging_enabled:
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

def load_csv_as_df_from_directory(
    directory: Path, return_filenames: bool = False
) -> Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
    """
    Load CSV files from a directory into pandas DataFrames.

    Parameters:
        directory (Path): The directory path where the CSV files are located.
        return_filenames (bool, optional): Whether to return a dictionary with filenames as keys and DataFrames as values.
            If False, a list of DataFrames will be returned. Defaults to False.

    Returns:
        Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]: A dictionary or list of pandas DataFrames containing the loaded CSV data.

    Raises:
        FileNotFoundError: If the specified directory is not found.
        Exception: If an unexpected error occurs while loading the CSV files.

    """
    logger.info(f"Attempting to load CSV files from path: {directory}...")
    try:
        # Get a list of all CSV files in the directory
        csv_files = glob.glob(f"{directory}/*.csv")

        # Load each CSV file into a DataFrame and store in a dictionary or list
        if return_filenames:
            dataframes = {Path(file).name: load_csv_as_df(file, logging_enabled=False) for file in csv_files}
        else:
            dataframes = [load_csv_as_df(file, logging_enabled=False) for file in csv_files]

        logger.info(f"Successfully loaded {len(dataframes)} CSV files from path: {directory}")
        return dataframes
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading CSV files from path: {e}")
        return []

def combine_similar_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine a list of pandas DataFrames into one DataFrame.

    Parameters:
        dfs (List[pd.DataFrame]): The list of DataFrames to be combined.

    Returns:
        pd.DataFrame: The combined DataFrame.

    Raises:
        ValueError: If not all dataframes have the same number of columns.
    """
    # Ensure that all dataframes have the same number of columns
    first_num_columns = dfs[0].shape[1]
    if any(df.shape[1] != first_num_columns for df in dfs):
        raise ValueError("All dataframes must have the same number of columns.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def main():
    # Instantiate the input data directory
    input_data_directory = Path(__file__).resolve().parents[1] / "in"
    # Load all csv files from input directory to list of dfs
    emission_dataframes = load_csv_as_df_from_directory(input_data_directory)
    # Combine all dfs into one, check if match in columns
    combined_emissions = combine_similar_dataframes(emission_dataframes)
    # Export the combined df as a csv file
    export_df_as_csv(combined_emissions, input_data_directory, "combined_emissions")

if __name__ == "__main__":
    main()