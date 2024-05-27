from pathlib import Path
from typing import Any, List, Tuple
from statistics import mean

from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from utilities import get_logger

logger = get_logger(__name__)


def load_labeled_data_as_df(path_to_data: Path) -> pd.DataFrame:
    """
    Load labeled data from a CSV file into a pandas DataFrame.

    Parameters:
        path_to_data (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The labeled data as a pandas DataFrame.
    """
    return pd.read_csv(path_to_data)


def save_cross_validated_scores_to_txt(
    scores: List[float], output_dir: Path, file_name: str, decimals: int = 3
):
    """
    Save cross-validated scores to a text file.

    Parameters:
        scores (List[float]): The list of cross-validated scores.
        output_dir (Path): The directory where the output file will be saved.
        file_name (str): The name of the output file (without extension).
        decimals (int, optional): The number of decimal places to round the scores to. Defaults to 3.

    Raises:
        Exception: If there is an error while saving the scores.

    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{file_name}.txt", "w") as file:
            file.write("Cross-validated scores:\n")
            for i, score in enumerate(scores, 1):
                file.write(f"Fold {i}: {round(score, decimals)}\n")
            file.write(f"Mean score: {round(mean(scores), decimals)}\n")
    except Exception as e:
        logger.error(f"Failed to save cross-validated scores: {e}")


def save_classification_report_to_txt(
    classification_report: str, output_dir: Path, file_name: str
) -> None:
    """
    Saves the classification report to a text file.

    Parameters:
        classification_report (str): The classification report to be saved.
        output_dir (Path): The directory where the text file will be saved.
        file_name (str): The name of the text file.

    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_file_path = output_dir / f"{file_name}.txt"
        with full_file_path.open("w") as file:
            file.write(classification_report)
        logger.info(f"Classification report saved as {full_file_path.name}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")


def load_object_from_joblib(file_path: Path) -> Any:
    """
    Loads an object from a joblib file.

    Parameters:
        file_path (Path): The path to the joblib file.

    Returns:
        Any: The loaded object.

    Raises:
        Exception: If there is an error loading the object.
    """
    try:
        return load(file_path)
    except Exception as e:
        logger.error(f"Failed to load object from {file_path}: {e}")


def save_object_as_joblib(
    object_to_save: Any, output_dir: Path, file_stem: str, object_name: str
) -> None:
    """
    Saves an object to a joblib file.

    Parameters:
        object_to_save (Any): The object to be saved.
        output_dir (Path): The directory where the file will be saved.
        file_stem (str): The stem of the file name.
        object_name (str): The name of the object being saved.

    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = output_dir / f"{file_stem}_{object_name}.joblib"
        dump(object_to_save, file_name)
        logger.info(f"{object_name.capitalize()} saved as {file_name.name}")
    except Exception as e:
        logger.error(f"Failed to save {object_name}: {e}")


def load_and_split_training_data(
    data: pd.DataFrame,
    text_col: str,
    label_col: str,
    train_test_size: float = 0.2,
    seed: int = 24,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load and split the training data into train and test sets.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data.
        text_col (str): The name of the column containing the text data.
        label_col (str): The name of the column containing the label data.
        train_test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.2.
        seed (int, optional): The random seed for reproducibility. Defaults to 24.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: A tuple containing the train-test split of the text and label data.
    """
    logger.info(
        f"Splitting data into train and test sets with test size {train_test_size} and seed {seed}."
    )
    X = data[text_col]
    y = data[label_col]
    return train_test_split(
        X, y, test_size=train_test_size, random_state=seed, stratify=y
    )


def prepare_data_for_model_training(
    data: pd.DataFrame,
    text_col: str,
    label_col: str,
    vectorizer: TfidfVectorizer,
    train_test_size: float = 0.2,
    seed: int = 24,
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    """
    Prepares data for model training and testing.

    Parameters:
        data (pd.DataFrame): The input data containing text and label columns.
        text_col (str): The name of the column containing the text data.
        label_col (str): The name of the column containing the label data.
        vectorizer (TfidfVectorizer): The vectorizer used to transform text data into feature vectors.
        train_test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.
        seed (int, optional): The random seed for splitting the data. Defaults to 24.

    Returns:
        Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]: A tuple containing the feature vectors for training and testing, and the corresponding label series for training and testing.
    """
    logger.info("Preparing data for model training...")

    if text_col not in data.columns or label_col not in data.columns:
        raise ValueError(f"Columns {text_col} or {label_col} not found in data.")

    X_train, X_test, y_train, y_test = load_and_split_training_data(
        data, text_col, label_col, train_test_size, seed
    )

    logger.info("Transforming text data into feature vectors.")
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    logger.info("Data preparation complete!")
    return X_train_feats, X_test_feats, y_train, y_test
