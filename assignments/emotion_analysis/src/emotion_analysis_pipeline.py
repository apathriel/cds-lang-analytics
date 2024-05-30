from pathlib import Path
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, Pipeline

from utilities.cli_decorator import cli_options
from utilities.data_manipulation_utils import (
    export_df_as_csv,
    load_csv_as_df,
    convert_column_to_data_type,
    get_column_value_counts_by_group_as_percentage,
)

from utilities.logger_utils import get_logger
from utilities.plotting_utilities import (
    visualize_relative_emotion_distribution_by_season,
    visualize_emotion_flunctuations_across_seasons,
)


logger = get_logger(__name__)


def emotion_analysis_pipeline(
    df: pd.DataFrame,
    classifier: Pipeline,
    raw_text_column: str,
    emotion_column_title: str,
    score_column_title: str,
) -> pd.DataFrame:

    # Initialize a progress bar with tqdm
    tqdm.pandas(desc="Classifying text")

    # Run the classifier on each element in the defined column in the DataFrame
    try:
        results = df[raw_text_column].progress_apply(lambda x: classifier(x)[0][0])
    except Exception as e:
        logger.error(f"Failed to classify text: {e}")
        return df

    # Extract the labels and scores from the results
    df[emotion_column_title] = results.apply(lambda x: x["label"])
    df[score_column_title] = results.apply(lambda x: x["score"])

    return df


@click.command()
@cli_options
def main(
    input_csv_path: str,
    output_csv_path: Optional[str],
    output_plot_path: Optional[str],
    processed_data_path: Optional[str],
    filter_out_neutral_tag: bool,
    rescale_y_axis_for_fluctuation_plot: bool,
    hf_model: str,
    raw_text_column: str,
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
    output_data_plot_path = (
        Path(__file__).parent / ".." / output_plot_path if output_plot_path else None
    )

    if processed_data_path:
        processed_data_path = (
            f"{processed_data_path}.csv"
            if not processed_data_path.endswith(".csv")
            else processed_data_path
        )
        processed_data_path = Path(__file__).parent / ".." / "out" / processed_data_path
        df = load_csv_as_df(Path(processed_data_path))
        print(df.head())
    else:
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
        if output_data_path:
            export_df_as_csv(
                df, output_data_path, f"{input_data_path.stem}_emotion_classification"
            )

    # Filter out neutral emotion tags if disregard_neutral_tag is True
    df = df[df["Emotion"] != "neutral"] if filter_out_neutral_tag else df

    # Visualize the results, show plots if no output path is provided
    counts_by_season_title = (
        "emotion_counts_by_season" if output_data_plot_path else None
    )
    counts_across_seasons_title = (
        "emotion_flunctuations_across_seasons" if output_data_plot_path else None
    )

    emotion_counts_by_season = get_column_value_counts_by_group_as_percentage(
        df, "Season", "Emotion"
    )

    colors_for_plots = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    # Plot the emotion counts by season
    visualize_relative_emotion_distribution_by_season(
        normalized_counts_by_category=emotion_counts_by_season,
        num_subplots_columns=3,
        plot_title="Distribution of emotion labels per season",
        plot_colors=colors_for_plots,
        output_dir=output_data_plot_path,
        plot_output_title=counts_by_season_title,
        plot_output_format="png",
    )

    # Plot the relative frequency of emotion labels across total lines of season
    visualize_emotion_flunctuations_across_seasons(
        normalized_counts_across_timeseries=emotion_counts_by_season.unstack(level=0),
        num_subplots_columns=3,
        plot_title="Relative emotion flunctuations across seasons",
        plot_colors=colors_for_plots,
        output_dir=output_data_plot_path,
        plot_output_title=counts_across_seasons_title,
        plot_output_format="png",
        rescale_y_axis=rescale_y_axis_for_fluctuation_plot,
    )

if __name__ == "__main__":
    main()
