import math
import os
from pathlib import Path

from logger_utils import get_logger
from data_manipulation_utils import load_csv, get_column_value_counts_by_group_as_percentage, convert_column_to_data_type
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


LOGGER = get_logger(__name__)


def visualize_relative_emotion_distribution_by_season(
    normalized_counts_by_category: pd.Series,
    num_subplots_columns: int,
    plot_title: str,
    plot_colors: list,
):
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

    # Show the plot
    plt.show()

def visualize_emotion_flunctuations_across_seasons(
    normalized_counts_across_timeseries: pd.DataFrame,
    num_subplots_columns: int,
    plot_title: str,
    plot_colors: list,
):
    num_of_subplots = len(normalized_counts_across_timeseries.index)
    num_rows = math.ceil(num_of_subplots / num_subplots_columns)

    # Create a subplot for each emotion in a 2-column grid
    fig, axs = plt.subplots(num_rows, num_subplots_columns, figsize=(20, 5 * num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    for ax in axs[num_of_subplots:]:
        fig.delaxes(ax)

    fig.suptitle(plot_title.capitalize(), fontsize=16)
    fig.subplots_adjust(hspace=1)

    # Iterate over each emotion
    for i, emotion in enumerate(normalized_counts_across_timeseries.index):
        # Plot the emotion percentages for each season
        normalized_counts_across_timeseries.loc[emotion].plot(
            kind="line", ax=axs[i], color=plot_colors[i % len(plot_colors)]
        )

        # Set the title and labels
        axs[i].set_title(f"{emotion.capitalize()} Percentage by Season")
        axs[i].set_xlabel("Season")
        axs[i].set_ylabel("Percentage")

        # Rotate X axis labels by 45 degrees
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

    # Show the plot
    plt.show()

def main():
    data_path = Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "in")
    )

    df = load_csv(data_path / "Game_of_Thrones_Script_with_Classification.csv")
    df = convert_column_to_data_type(df, "Sentence", str)

    classifier = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        framework="tf",
    )

    emotion_counts_by_season = get_column_value_counts_by_group_as_percentage(
        df, "Season", "Emotion"
    )

    PLOT_COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    # Plot the emotion counts by season
    visualize_relative_emotion_distribution_by_season(
        normalized_counts_by_category=emotion_counts_by_season,
        num_subplots_columns=3,
        plot_title="Distribution of emotion labels per season",
        plot_colors=PLOT_COLORS,
    )

    # Plot the relative frequency of emotion labels across total lines of season
    visualize_emotion_flunctuations_across_seasons(
        normalized_counts_across_timeseries=emotion_counts_by_season.unstack(level=0),
        num_subplots_columns=3,
        plot_title="Relative emotion flunctuations across seasons",
        plot_colors=PLOT_COLORS,
    )