from tqdm import tqdm
from transformers import pipeline
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "in"))

def get_filenames_in_dir(directory, list_sub_dirs=False):
    return os.listdir(directory) if list_sub_dirs else [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def save_first_n_rows_to_csv(df: pd.DataFrame, num_rows: int, file_path: Path) -> None:
    if not file_path:
        file_path = data_path / f"first_{num_rows}_rows.csv"
    df.head(num_rows).to_csv(file_path, index=False)

def load_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def get_unique_values(df: pd.DataFrame, column: str) -> list:
    return df[column].unique()

def get_column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts()

def get_column_value_counts_by_group_as_percentage(df: pd.DataFrame, column_to_group: str, value_to_group_by: str) -> pd.Series:
    return df.groupby(column_to_group)[value_to_group_by].value_counts(normalize=True) #/ df.groupby(column_to_group)[value_to_group_by].count()

def convert_column_to_data_type(df: pd.DataFrame, column: str, data_type: type) -> pd.DataFrame:
    df[column] = df[column].astype(data_type)
    return df

df = load_csv(data_path / "first_10_rows.csv")
df = convert_column_to_data_type(df, 'Sentence', str)

classifier = pipeline(task="text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k=1,
                      framework='tf'
                      )


results = [classifier(sentence)[0][0] for sentence in tqdm(df['Sentence'])]

# Extract the labels and scores from the results
labels, scores = zip(*[(result['label'], result['score']) for result in results])

# Add the labels and scores to the DataFrame
df['Emotion'], df['Score'] = labels, scores

emotion_counts_by_season = get_column_value_counts_by_group_as_percentage(df, 'Season', 'Emotion')

PLOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
num_seasons = len(emotion_counts_by_season.index.levels[0])

def temp_plot(num_of_subplots: int, plot_title: str, plot_colors: list):
    fig, axs = plt.subplots(num_of_subplots, 1, figsize=(10, 5*num_of_subplots))
    fig.suptitle(plot_title, fontsize=16)
    fig.subplots_adjust(hspace=0.5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(1)

    for i, season in enumerate(emotion_counts_by_season.index.levels[0]):
        # Get the counts for the current season
        counts = emotion_counts_by_season.loc[season]

        # Sort the counts by emotion label
        counts = counts.sort_index()

        # Create a bar plot for the current season
        counts.plot(kind='bar', ax=axs[i], color=plot_colors[i % len(plot_colors)])

        # Set the title and labels
        axs[i].set_title(f'Emotion Counts for {season}')
        axs[i].set_xlabel('Emotion')
        axs[i].set_ylabel('Percentage')
        axs[i].legend([f'{emotion}: {count:.2f}' for emotion, count in counts.items()])
    # Adjust the spacing between the subplots
        # Move the y-axis label to the right side
        # axs[i].yaxis.set_label_position("right")

    # Show the plot
    plt.show()

import math

def plot_emotion_by_season(df):
    # Calculate the number of rows needed for the grid
    num_rows = math.ceil(df.shape[0] / 3)

    # Create a subplot for each emotion in a 2-column grid
    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 7*num_rows))

    # Flatten the axes array and remove extra subplots
    axs = axs.flatten()[:df.shape[0]]

    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(1)
    fig.subplots_adjust(hspace=0.5)

    # Iterate over each emotion
    for i, emotion in enumerate(df.index):
        # Plot the emotion percentages for each season
        df.loc[emotion].plot(kind='line', ax=axs[i], color='blue')

        # Set the title and labels
        axs[i].set_title(f'{emotion.capitalize()} Percentage by Season')
        axs[i].set_xlabel('Season')
        axs[i].set_ylabel('Percentage')

    # Adjust the spacing between the subplots
    fig.tight_layout()

    # Show the plot
    plt.show()

# Plot the emotion counts by season
temp_plot(num_seasons, 'Distribution of emotion labels per season', PLOT_COLORS)

# Plot the relative frequency of emotion labels across total lines of season
plot_emotion_by_season(emotion_counts_by_season.unstack(level=0))
# Plot the relative frequency of emotion labels by season from total num of label occurances

# Save the DataFrame back to a CSV file
df.to_csv(data_path / "Game_of_Thrones_Script_with_Classification.csv", index=False)