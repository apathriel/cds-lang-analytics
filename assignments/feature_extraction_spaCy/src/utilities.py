import logging
from pathlib import Path
import re

import chardet

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with the specified name.

    Parameters:
        name (str): The name of the logger. Convention is to use __name__.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] - %(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger

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
