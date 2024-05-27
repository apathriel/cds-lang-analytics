import math
from pathlib import Path

from logger_utils import get_logger
from data_manipulation_utils import (
    load_csv_as_df,
    get_column_value_counts_by_group_as_percentage,
    convert_column_to_data_type,
)
import matplotlib.pyplot as plt
import pandas as pd


logger = get_logger(__name__)


def visualize_relative_emotion_distribution_by_season(
    normalized_counts_by_category: pd.Series,
    num_subplots_columns: int,
    plot_title: str,
    plot_colors: list,
    output_dir: Path,
    plot_output_title: str = None,
    plot_output_format: str = "png",
) -> None:
    # Calculate the number of rows needed for the grid
    num_of_subplots = len(normalized_counts_by_category.index.levels[0])
    num_rows = math.ceil(num_of_subplots / num_subplots_columns)

    # Create a subplot for each emotion in a grid
    fig, axs = plt.subplots(num_rows, num_subplots_columns, figsize=(20, 5 * num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    for ax in axs[num_of_subplots:]:
        fig.delaxes(ax)

    fig.suptitle(plot_title.capitalize(), fontsize=16)
    fig.subplots_adjust(hspace=1)

    for i, season in enumerate(normalized_counts_by_category.index.levels[0]):
        # Get the counts for the current season
        counts = normalized_counts_by_category.loc[season]

        # Sort the counts by emotion label
        counts = counts.sort_index()

        # Create a bar plot for the current season
        counts.plot(kind="bar", ax=axs[i], color=plot_colors[i % len(plot_colors)])

        # Set the title and labels
        axs[i].set_title(f"Emotion Counts for {season}")
        axs[i].set_xlabel("Emotion")
        axs[i].set_ylabel("Percentage")
        axs[i].legend([f"{emotion}: {count:.2f}" for emotion, count in counts.items()])

        # Rotate X axis labels by 45 degrees
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

    if plot_output_title:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{plot_output_title}.{plot_output_format}"
        plt.savefig(fname=save_path)
        logger.info(f"Plot saved as {save_path}")
    else:
        plt.show()


def visualize_emotion_flunctuations_across_seasons(
    normalized_counts_across_timeseries: pd.DataFrame,
    num_subplots_columns: int,
    plot_title: str,
    plot_colors: list,
    output_dir: Path,
    plot_output_title: str = None,
    plot_output_format: str = "png",
) -> None:
    num_of_subplots = len(normalized_counts_across_timeseries.index)
    num_rows = math.ceil(num_of_subplots / num_subplots_columns)

    # Create a subplot for each emotion in a 2-column grid
    fig, axs = plt.subplots(num_rows, num_subplots_columns, figsize=(20, 5 * num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    for ax in axs[num_of_subplots:]:
        fig.delaxes(ax)

    # Get all labels from the X axis

    fig.suptitle(plot_title.capitalize(), fontsize=16)
    fig.subplots_adjust(hspace=1)

    # Iterate over each emotion
    for i, emotion in enumerate(normalized_counts_across_timeseries.index):
        # Plot the emotion percentages for each season
        normalized_counts_across_timeseries.loc[emotion].plot(
            kind="line", ax=axs[i], color=plot_colors[i % len(plot_colors)]
        )

        # Get the X axis labels
        labels = [item.get_text() for item in axs[i].get_xticklabels()]

        # Replace 'Season ' with 'S' in each label
        labels = [
            "S" + label.split(" ")[1] if " " in label else label for label in labels
        ]
        # Set the title and labels
        axs[i].set_title(f"{emotion.capitalize()} Percentage by Season")
        axs[i].set_xlabel("Season")
        axs[i].set_ylabel("Percentage")
        axs[i].set_xticklabels(labels)

        # Rotate X axis labels by 45 degrees
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

    # Handle plot output
    if plot_output_title:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{plot_output_title}.{plot_output_format}"
        plt.savefig(fname=save_path)
        logger.info(f"Plot saved as {save_path}")
    else:
        plt.show()


def main():
    input_data_path = Path(__file__).parent / ".." / "out"
    output_plot_path = Path(__file__).parent / ".." / "out" / "plots"

    input_file_path = input_data_path / "Game_of_Thrones_Script_with_Classification.csv"

    df = load_csv_as_df(input_file_path)
    df = convert_column_to_data_type(df, "Sentence", str)

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
        output_dir=output_plot_path,
        plot_output_title="emotion_counts_by_season",
        plot_output_format="png",
    )

    # Plot the relative frequency of emotion labels across total lines of season
    visualize_emotion_flunctuations_across_seasons(
        normalized_counts_across_timeseries=emotion_counts_by_season.unstack(level=0),
        num_subplots_columns=3,
        plot_title="Relative emotion flunctuations across seasons",
        plot_colors=colors_for_plots,
        output_dir=output_plot_path,
        plot_output_title="emotion_flunctuations_across_seasons",
        plot_output_format="png",
    )


if __name__ == "__main__":
    main()
