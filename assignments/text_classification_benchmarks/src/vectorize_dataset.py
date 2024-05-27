from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from data_processing_utilities import (
    save_object_as_joblib,
    load_labeled_data_as_df,
)


def main():
    # Initialize input/output paths
    input_data_path = Path(__file__).parent / ".." / "in" / "fake_or_real_news.csv"
    output_data_path = Path(__file__).parent / ".." / "in"

    # Load the labeled data
    news_dataset = load_labeled_data_as_df(input_data_path)

    # Instantiate vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        max_df=0.95,
        min_df=0.05,
        max_features=100,
    )

    X = news_dataset["text"]

    X_train_feats = vectorizer.fit_transform(X)

    save_object_as_joblib(
        object_to_save=X_train_feats,
        output_dir=output_data_path,
        file_stem="fake_or_real_news",
        object_name="vectorized_data",
    )
if __name__ == "__main__":
    main()