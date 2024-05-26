from pathlib import Path
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from data_processing_utilities import (
    load_labeled_data_as_df,
    save_classification_report_to_txt,
    save_cross_validated_scores_to_txt,
    save_object_as_joblib,
    prepare_data_for_model_training,
)
from utilities import get_logger

logger = get_logger(__name__)


def train_neural_network_classifier_model(
    X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]
) -> MLPClassifier:
    logger.info("Training neural network classifier...")
    return MLPClassifier(
        activation="logistic", hidden_layer_sizes=(20,), max_iter=1000, random_state=42
    ).fit(X_train, y_train)


def neural_network_news_classification_pipeline(
    data: pd.DataFrame,
    text_col: str,
    label_col: str,
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    report_path: Path,
    model_path: Path,
    train_test_size: float = 0.2,
    seed: int = 24,
    cross_validate: bool = False,
    cv_fold: int = 10,
):
    X_train_feats, X_test_feats, y_train, y_test = prepare_data_for_model_training(
        data, text_col, label_col, vectorizer, train_test_size, seed
    )
    classifier = train_neural_network_classifier_model(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)

    if cross_validate:
        logger.info(f"Cross-validating with {cv_fold} folds...")
        scores = cross_val_score(classifier, X_train_feats, y_train, cv=cv_fold)
        logger.info(
            f"Cross-validation complete. Cross-validated mean score: {round(mean(scores), 2)}"
        )
        save_cross_validated_scores_to_txt(
            scores, report_path, "neural_network_cross_validated_scores"
        )

    save_classification_report_to_txt(
        classification_report=classification_report(y_test, y_pred),
        output_dir=report_path,
        file_name="neural_network_classification_report",
    )

    save_object_as_joblib(
        object_to_save=vectorizer,
        output_dir=model_path,
        file_stem="neural_network",
        object_name="vectorizer",
    )
    save_object_as_joblib(
        object_to_save=classifier,
        output_dir=model_path,
        file_stem="neural_network",
        object_name="classifier",
    )


def main():
    # Initialize input/output paths
    input_data_path = Path(__file__).parent / ".." / "in" / "fake_or_real_news.csv"
    report_data_path = Path(__file__).parent / ".." / "out"
    model_data_path = Path(__file__).parent / ".." / "out" / "models" / "neural_network"

    # Load the labeled data
    news_dataset = load_labeled_data_as_df(input_data_path)

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        max_df=0.95,
        min_df=0.05,
        max_features=100,
    )

    # Run the neural network pipeline, training the model and saving the report and model
    neural_network_news_classification_pipeline(
        data=news_dataset,
        text_col="text",
        label_col="label",
        vectorizer=vectorizer,
        report_path=report_data_path,
        model_path=model_data_path,
    )


if __name__ == "__main__":
    main()
