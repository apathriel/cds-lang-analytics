from pathlib import Path
import re

import chardet
import pandas as pd

from utilities import get_logger

logger = get_logger(__name__)

def export_df_as_csv(df: pd.DataFrame, directory: Path, filename: str) -> None:
    """
    Export a pandas DataFrame as a CSV file.

    Paramters:
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

    # Create the full file path
    file_path = directory / filename

    try:
        df.to_csv(file_path, index=False)
    except PermissionError:
        logger.error(f"Permission denied when trying to write to file: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred when trying to write to file: {e}")

def remove_html_tags(text: str) -> str:
    """
    This function uses a regular expression to remove all occurrences of tags in text.

    Parameters:
        text (str): The text to clean.

    Returns:
        str: The cleaned text with all tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def load_text_file(file_path: Path) -> str:
    """
    Opens and reads a text file, returning its contents as a string.

    The function assumed UTF-8 encoding, if it fails, attempts to determine the encoding using chardet module, try loading text file with new encoding.

    Paramters:
        filepath (str): The path to the file to open.

    Returns:
        str: The contents of the file as a string.

    Raises:
        UnicodeDecodeError: If the file cannot be decoded using UTF-8 encoding.
    """
    # Read the file as bytes in order to provide more robust encoding-independent file loading
    file_bytes = file_path.read_bytes()
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        get_encoding = chardet.detect(file_bytes)
        try:
            text = file_bytes.decode(get_encoding["encoding"])
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return ""
    return remove_html_tags(text)

