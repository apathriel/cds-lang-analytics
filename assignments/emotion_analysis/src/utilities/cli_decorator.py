# cli_decorator.py
import click


def cli_options(func):
    decorators = [
        click.option(
            "--input_csv_path",
            "-i",
            help="Path to the input CSV file relative to this scripts parent folder",
            default="Game_of_Thrones_Script.csv",
            prompt=True,
            required=True,
        ),
        click.option(
            "--output_csv_path",
            "-o",
            help="Path to the output CSV file",
            default=None,
            type=str,
        ),
        click.option(
            "--output_plot_path",
            "-op",
            help="Path to the output plot file. If not provided, plots will not be saved",
            type=str,
            default=None,
        ),
        click.option(
            "--processed_data_path",
            "-pdp",
            help="Path to the processed csv file. If not provided, the pipeline will process the input data and save it to a new file.",
            type=str,
            default=None,
        ),
        click.option(
            "--filter_out_neutral_tag",
            "-f",
            help="If true, disregard neutral emotion tags in visualization",
            is_flag=True,
            default=False,
        ),
        click.option(
            "--rescale-y-axis_for_fluctuation_plot",
            "-ry",
            help="If true, rescale the y-axis 0-1 for the fluctuation plot",
            is_flag=True,
            default=False,
        ),
        click.option(
            "--hf_model",
            "-m",
            help="Name of the Hugging Face model to use for classification",
            default="j-hartmann/emotion-english-distilroberta-base",
        ),
        click.option(
            "--raw_text_column",
            "-rtc",
            help="Name of the column containing raw text data",
            default="Sentence",
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
