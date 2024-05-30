from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from data_processing_utilities import (
    export_df_as_csv,
    load_labeled_data_as_df,
    save_classification_report_to_txt,
    save_cross_validated_scores_to_csv,
    save_object_as_joblib,
    prepare_data_for_model_training,
)
from utilities import get_logger
from emission_tracker_class import SingletonEmissionsTracker

logger = get_logger(__name__)

# Initialize the emissions tracker
emissions_tracker = SingletonEmissionsTracker(
    project_name=Path(__file__).stem,
    experiment_id="neural_network_news_classification",
    output_dir=Path(__file__).parent / ".." / "out",
)


def train_neural_network_classifier_model(
    output_dir: Path,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    clf_parameters: Dict[str, Any],
    use_grid_search: bool = False,
    grid_search_params: Optional[Dict[str, Any]] = None,
    grid_search_folds: int = 5,
    cross_validate: bool = False,
    cv_fold: int = 10,
) -> MLPClassifier:
    """
    Trains a neural network classifier model. Saves the trained model, vectroizer, and classification report.

    Parameters:
        output_dir (Path): The output directory to save the trained model and results.
        X_train (Union[pd.DataFrame, np.ndarray]): The input features for training.
        y_train (Union[pd.Series, np.ndarray]): The target labels for training.
        clf_parameters (Dict[str, Any]): The parameters for the MLPClassifier model.
        use_grid_search (bool, optional): Whether to perform grid search for hyperparameters. Defaults to False.
        grid_search_params (Optional[Dict[str, Any]], optional): The grid search parameters. Defaults to None.
        cross_validate (bool, optional): Whether to perform cross-validation. Defaults to False.
        cv_fold (int, optional): The number of cross-validation folds. Defaults to 10.

    Returns:
        MLPClassifier: The trained MLPClassifier model.
    """
    logger.info("Training neural network classifier...")

    if use_grid_search and grid_search_params is not None:
        num_combinations = np.prod([len(v) for v in grid_search_params.values()])
        num_fits = num_combinations * grid_search_folds
        SingletonEmissionsTracker.start_task(f"grid_search_{num_fits}_fits_cv_5_folds")
        logger.info("Performing grid search for hyperparameters...")
        # Define the model
        mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)

        # Set up the grid search
        clf = GridSearchCV(mlp, grid_search_params, n_jobs=-1, cv=grid_search_folds, verbose=3)

        # Fit the model and find the best hyperparameters
        clf.fit(X_train, y_train)

        logger.info(f"Best parameters found: {clf.best_params_}")

        best_estimator = clf.best_estimator_
        SingletonEmissionsTracker.stop_current_task()
    else:
        unpacked_clf_parameters = {
            key: value[0] for key, value in clf_parameters.items()
        }
        best_estimator = MLPClassifier(
            **unpacked_clf_parameters,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ).fit(X_train, y_train)

    if cross_validate:
        SingletonEmissionsTracker.start_task(f"cross_validate_neural_network_{cv_fold}_folds")
        logger.info(f"Cross-validating with {cv_fold} folds...")
        scores = cross_val_score(best_estimator, X_train, y_train, cv=cv_fold)
        logger.info(
            f"Cross-validation complete. Cross-validated mean score: {round(np.mean(scores), 2)}"
        )
        SingletonEmissionsTracker.stop_current_task()

        SingletonEmissionsTracker.start_task("save_cross_validated_scores")
        save_cross_validated_scores_to_csv(scores, output_dir, "neural_network_cross_validated_scores")
        SingletonEmissionsTracker.stop_current_task()

    return best_estimator


def neural_network_news_classification_pipeline(
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    report_path: Path,
    model_path: Path,
    classifier: MLPClassifier, 
    X_test_feats: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> None:
    """
    Runs a news classification pipeline using a neural network classifier.

    Parameters:
        vectorizer (Union[CountVectorizer, TfidfVectorizer]): The vectorizer used to transform the input data.
        report_path (Path): The path to save the classification report.
        model_path (Path): The path to save the trained model.
        classifier (MLPClassifier): The neural network classifier.
        X_test_feats (Union[pd.DataFrame, np.ndarray]): The features of the test data.
        y_test (Union[pd.Series, np.ndarray]): The labels of the test data.

    """
    SingletonEmissionsTracker.start_task("predict_test_data")
    y_pred = classifier.predict(X_test_feats)
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("save_model_and_report")
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
    SingletonEmissionsTracker.stop_current_task()


def main():
    SingletonEmissionsTracker.start_task("initialize_paths")
    # Initialize input/output paths
    input_data_path = Path(__file__).parent / ".." / "in" / "fake_or_real_news.csv"
    report_data_path = Path(__file__).parent / ".." / "out"
    model_data_path = Path(__file__).parent / ".." / "out" / "models" / "neural_network"
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

    SingletonEmissionsTracker.start_task("perepare_data_for_model_training")
    # Prepare the data for model training
    X_train_feats, X_test_feats, y_train, y_test = prepare_data_for_model_training(
        data=news_dataset,
        text_col="text",
        label_col="label",
        vectorizer=vectorizer,
        train_test_size=0.20,
        seed=24,
    )
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("instantiate_hyperparameters")
    # Define the grid of hyperparameters to search
    parameter_space = {
        "hidden_layer_sizes": [(50, 100, 50)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.05],
        "learning_rate": ["constant"],
    }
    SingletonEmissionsTracker.stop_current_task()

    SingletonEmissionsTracker.start_task("instantiate_grid_search_parameters")
    grid_search_parameters = {
        "hidden_layer_sizes": [(50, 100, 50), (100, 100, 100), (100, 200, 100)],
        "alpha": [0.05, 0.01, 0.001],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "learning_rate": ["constant", "adaptive"],
    }
    SingletonEmissionsTracker.stop_current_task()


    # Track emissions in function to seperate task logic
    # Train the neural network classifier
    mlp_classifier = train_neural_network_classifier_model(
        output_dir=report_data_path,
        X_train=X_train_feats,
        y_train=y_train,
        clf_parameters=parameter_space,
        use_grid_search=False,
        grid_search_params=grid_search_parameters,
        cross_validate=True,
        cv_fold=10,
    )

    # Run the neural network pipeline, training the model and saving the report and model
    neural_network_news_classification_pipeline(
        vectorizer=vectorizer,
        classifier=mlp_classifier,
        X_test_feats=X_test_feats,
        y_test=y_test,
        report_path=report_data_path,
        model_path=model_data_path,
    )


if __name__ == "__main__":
    main()
    df = SingletonEmissionsTracker.create_dataframe_from_task_results()
    export_df_as_csv(
        df, Path(__file__).parent / ".." / "out", "neural_emission_results.csv"
    )
