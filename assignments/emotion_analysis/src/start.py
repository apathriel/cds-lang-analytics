from tqdm import tqdm
from transformers import pipeline
import pandas as pd
from pathlib import Path
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "in"))

def get_filenames_in_dir(directory, list_sub_dirs=False):
    return os.listdir(directory) if list_sub_dirs else [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def save_first_n_rows_to_csv(df: pd.DataFrame, num_rows: int, file_path: Path) -> None:
    if not file_path:
        file_path = data_path / f"first_{num_rows}_rows.csv"
    df.head(num_rows).to_csv(file_path, index=False)

df = pd.read_csv(data_path / "Game_of_Thrones_Script_10.csv")

classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k=1,
                      framework='tf'
                      )

results = [classifier(sentence)[0] for sentence in tqdm(df['Sentence'])]

# Extract the labels and scores from the results
labels, scores = zip(*[(result['label'], result['score']) for result in results])

# Add the labels and scores to the DataFrame
df['Emotion'], df['Score'] = labels, scores

# Save the DataFrame back to a CSV file
df.to_csv(data_path / "Game_of_Thrones_Script_with_Classification.csv", index=False)