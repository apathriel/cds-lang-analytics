from pathlib import Path
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data_processing_utilities import (
    export_df_as_csv,
    load_labeled_data_as_df,
    save_classification_report_to_txt,
    save_cross_validated_scores_to_csv,
    save_object_as_joblib,
    prepare_data_for_model_training,
)
from emission_tracker_class import SingletonEmissionsTracker
from utilities import get_logger

# Initialize the emissions tracker
emissions_tracker = SingletonEmissionsTracker(
    project_name=Path(__file__).stem,
    experiment_id="logistic_regression_news_classification",
    output_dir=Path(__file__).parent / ".." / "out",
)

logger = get_logger(__name__)


def train_logistic_regression_classifier_model(
    X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]
) -> LogisticRegression:
    logger.info("Training logistic regression classifier...")
    classifier = LogisticRegression(random_state=24).fit(X_train, y_train)
    return classifier


def logistic_regression_news_classification_pipeline(
    data: pd.DataFrame,
    text_col: str,
    label_col: str,
    vectorizer: Union[TfidfVectorizer, CountVectorizer],
    report_path: Path,
    model_path: Path,
    train_test_size: float = 0.2,
    seed: int = 24,
    cross_validate: bool = False,
    cv_fold: int = 10,
) -> None:

    SingletonEmissionsTracker.start_task("split_data_and_fit_vectorizer")
    X_train_feats, X_test_feats, y_train, y_test = prepare_data_for_model_training(
        data, text_col, label_col, vectorizer, train_test_size, seed
    )
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("train_logistic_regression_classifier")
    classifier = train_logistic_regression_classifier_model(X_train_feats, y_train)
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("predict_test_data")
    y_pred = classifier.predict(X_test_feats)
    SingletonEmissionsTracker.stop_current_task()

    if cross_validate:
        SingletonEmissionsTracker.start_task(f"cross_validate_logistic_regression_{cv_fold}folds")
        logger.info(f"Cross-validating with {cv_fold} folds...")
        scores = cross_val_score(classifier, X_train_feats, y_train, cv=cv_fold)
        logger.info(
            f"Cross-validation complete. Cross-validated mean score: {round(mean(scores), 2)}"
        )
        save_cross_validated_scores_to_csv(
            scores, report_path, "logistic_regression_cross_validated_scores"
        )
        SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("save_model_and_report")
    save_classification_report_to_txt(
        classification_report=classification_report(y_test, y_pred),
        output_dir=report_path,
        file_name="logistic_regression_classification_report",
    )

    save_object_as_joblib(
        object_to_save=vectorizer,
        output_dir=model_path,
        file_stem="logistic_regression",
        object_name="vectorizer",
    )
    save_object_as_joblib(
        object_to_save=classifier,
        output_dir=model_path,
        file_stem="logistic_regression",
        object_name="classifier",
    )
    SingletonEmissionsTracker.stop_current_task()


def main():
    SingletonEmissionsTracker.start_task("initialize_paths")
    # Initialize input/output paths
    input_data_path = Path(__file__).parent / ".." / "in" / "fake_or_real_news.csv"
    report_data_path = Path(__file__).parent / ".." / "out"
    model_data_path = (
        Path(__file__).parent / ".." / "out" / "models" / "logistic_regression"
    )
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("load_dataset")
    # Load the labeled data
    news_dataset = load_labeled_data_as_df(input_data_path)
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("instantiate_vectorizer")
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        max_df=0.95,
        min_df=0.05,
        max_features=2000,
    )
    SingletonEmissionsTracker.stop_current_task()

    # Run the logistic regression pipeline
    logistic_regression_news_classification_pipeline(
        data=news_dataset,
        text_col="text",
        label_col="label",
        vectorizer=vectorizer,
        report_path=report_data_path,
        model_path=model_data_path,
        cross_validate=True,
        cv_fold=10,
    )


if __name__ == "__main__":
    main()
    SingletonEmissionsTracker.log_task_results()
    df = SingletonEmissionsTracker.create_dataframe_from_task_results()
    export_df_as_csv(
        df, Path(__file__).parent / ".." / "out", "logistic_emission_results_process.csv"
    )
