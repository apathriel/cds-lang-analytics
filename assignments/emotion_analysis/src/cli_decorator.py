# cli_decorator.py
import click

def cli_options(func):
    decorators = [
        click.option(
            "--input_csv_path",
            "-i",
            help="Path to the input CSV file relative to this scripts parent folder",
            prompt=True,
            required=True,
        ),
        click.option(
            "--save_results_to_csv",
            "-s",
            help="Save the results to a new CSV file",
            default=True,
            type=bool,
        ),
        click.option(
            "--output_csv_path", "-o", help="Path to the output CSV file", default=None
        ),
        click.option(
            "--raw_text_column",
            "-rtc",
            help="Name of the column containing raw text data",
            default="Sentence",
        ),
        click.option(
            "--hf_model",
            "-m",
            help="Name of the Hugging Face model to use for classification",
            default="j-hartmann/emotion-english-distilroberta-base",
        ),
        click.option(
            "--emotion_column_title",
            "-ect",
            help="Name of the column to store the predicted emotion",
            default="Emotion",
        ),
        click.option(
            "--score_column_title",
            "-sct",
            help="Name of the column to store the prediction score",
            default="Score",
        ),
    ]
    # Apply each decorator in reverse order, reverse to maintain order
    for decorator in reversed(decorators):
        func = decorator(func)
    return func