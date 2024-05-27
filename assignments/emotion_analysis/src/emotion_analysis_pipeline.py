from pathlib import Path
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, Pipeline

from cli_decorator import cli_options
from data_manipulation_utils import (
    export_df_as_csv,
    load_csv_as_df,
    convert_column_to_data_type,
)

from logger_utils import get_logger

logger = get_logger(__name__)


def emotion_analysis_pipeline(
    df: pd.DataFrame,
    classifier: Pipeline,
    raw_text_column: str,
    emotion_column_title: str,
    score_column_title: str,
) -> pd.DataFrame:

    # Initialize a progress bar with tqdm
    tqdm.pandas()

    # Run the classifier on each element in the defined column in the DataFrame
    try:
        results = df[raw_text_column].progress_apply(lambda x: classifier(x)[0][0])
    except Exception as e:
        print(f"Failed to classify text: {e}")
        return df

    # Extract the labels and scores from the results
    df[emotion_column_title] = results.apply(lambda x: x["label"])
    df[score_column_title] = results.apply(lambda x: x["score"])

    return df


@click.command()
@cli_options
def main(
    input_csv_path: str,
    save_results_to_csv: bool,
    output_csv_path: Optional[str],
    raw_text_column: str,
    hf_model: str,
    emotion_column_title: str,
    score_column_title: str,
) -> None:
    # Initialize CSV paths for input and output
    input_csv_path = (
        f"{input_csv_path}.csv"
        if not input_csv_path.endswith(".csv")
        else input_csv_path
    )
    input_data_path = Path(__file__).parent / ".." / "in" / input_csv_path
    output_data_path = (
        Path(__file__).parent / ".." / output_csv_path if output_csv_path else None
    )

    # Load the Hugging Face model. Initialize the text classifier pipeline.
    text_classifier = pipeline(
        task="text-classification",
        model=hf_model,
        top_k=1,
        framework="tf",
    )

    # Load CSV file
    df = load_csv_as_df(input_data_path)
    # Convert the column to the appropriate data type
    df = convert_column_to_data_type(df, "Sentence", str)
    # Run the emotion analysis pipeline
    df = emotion_analysis_pipeline(
        df=df,
        classifier=text_classifier,
        raw_text_column=raw_text_column,
        emotion_column_title=emotion_column_title,
        score_column_title=score_column_title,
    )
    # Save the results to a new CSV file
    if save_results_to_csv:
        export_df_as_csv(df, output_data_path)


if __name__ == "__main__":
    main()
