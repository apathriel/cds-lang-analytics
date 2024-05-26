from pathlib import Path
from typing import Any, Tuple
from statistics import mean

from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from utilities import get_logger

logger = get_logger(__name__)


def load_labeled_data_as_df(path_to_data: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_data)

def save_cross_validated_scores_to_txt(scores, output_dir: Path, file_name: str):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{file_name}.txt", "w") as file:
            file.write("Cross-validated scores:\n")
            for i, score in enumerate(scores, 1):
                file.write(f"Fold {i}: {score}\n")
            file.write(f"Mean score: {mean(scores)}\n")
    except Exception as e:
        logger.error(f"Failed to save cross-validated scores: {e}")

def save_classification_report_to_txt(
    classification_report: str, output_dir: Path, file_name: str
) -> None:
    """Saves the classification report to a text file."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_file_path = output_dir / f"{file_name}.txt"
        with full_file_path.open("w") as file:
            file.write(classification_report)
        logger.info(f"Classification report saved as {full_file_path.name}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")


def save_object_as_joblib(
    object_to_save: Any, output_dir: Path, file_stem: str, object_name: str
) -> None:
    """Saves an object to a joblib file."""
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
    logger.info(
        f"Splitting data into train and test sets with test size {train_test_size} and seed {seed}."
    )
    X = data[text_col]
    y = data[label_col]
    return train_test_split(X, y, test_size=train_test_size, random_state=seed)



def prepare_data_for_model_training(
    data: pd.DataFrame,
    text_col: str,
    label_col: str,
    vectorizer: TfidfVectorizer,
    train_test_size: float = 0.2,
    seed: int = 24,
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    """Prepares data for training and testing."""
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
