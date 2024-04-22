import click
import os
from pathlib import Path

from data_manipulation_utils import save_df_to_csv, load_csv, convert_column_to_data_type
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from logger_utils import get_logger

LOGGER = get_logger(__name__)

def emotion_analysis_pipeline(
    df: pd.DataFrame,
    classifier: pipeline,
    raw_text_column: str,
    emotion_column_title: str,
    score_column_title: str
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
    df[emotion_column_title] = results.apply(lambda x: x['label'])
    df[score_column_title] = results.apply(lambda x: x['score'])

    return df

@click.command()
@click.option("--input_csv_path", "-i", help="Path to the input CSV file relative to this scripts parent folder", prompt=True, required=True)
@click.option("--save_results_to_csv", "-s", help="Save the results to a new CSV file", default=True, type=bool)
@click.option("--output_csv_path", "-o", help="Path to the output CSV file", default=None)
@click.option("--raw_text_column", "-rtc", help="Name of the column containing raw text data", default="Sentence")
@click.option("--hf_model", "-m", help="Name of the Hugging Face model to use for classification", default="j-hartmann/emotion-english-distilroberta-base")
@click.option("--emotion_column_title", "-ect", help="Name of the column to store the predicted emotion", default="Emotion")
@click.option("--score_column_title", "-sct", help="Name of the column to store the prediction score", default="Score")

def main(input_csv_path, save_results_to_csv, output_csv_path, raw_text_column, hf_model, emotion_column_title, score_column_title):
    # Initialize CSV paths for input and output
    input_data_path = Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", input_csv_path)
    )
    output_data_path = Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", output_csv_path)
    ) if output_csv_path else None

    print(input_data_path)
    # Load the Hugging Face model. Initialize the text classifier pipeline.
    text_classifier = pipeline(
        task="text-classification",
        model=hf_model,
        top_k=1,
        framework="tf",
    )

    # Load CSV file
    df = load_csv(input_data_path)
    # Convert the column to the appropriate data type
    df = convert_column_to_data_type(df, "Sentence", str)
    # Run the emotion analysis pipeline
    df = emotion_analysis_pipeline(df=df, classifier=text_classifier, raw_text_column=raw_text_column, emotion_column_title=emotion_column_title, score_column_title=score_column_title)
    # Save the results to a new CSV file
    if save_results_to_csv:
        save_df_to_csv(df, output_data_path)

if __name__ == "__main__":
    main()