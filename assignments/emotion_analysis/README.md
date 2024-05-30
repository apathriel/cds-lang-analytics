# 🎭 Emotion analysis with pre-trained language models
This project utilizes pre-trained language models to perform computational text analysis, specifically emotion analysis. This project looks at culturally-relevant data, the entire script of the Game of Thrones television show, and performs emotion classification for each sentence spoken. The project's results contain a csv file containing emotion classification for the entire script, and visualizations aiding in interpreting these results. This computational methodology presents reproducable analysis pipelines, which can be incorporated in wider cultural studies.

## 📈 Data

### 📋 Dataset
This project utilizes the [Game of Thrones Script All Seasons](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons) dataset for performing emotion analysis using a pre-trained language model. The dataset contains the entire script of the Game of Thrones series, split by sentence, totalling over 23.000 sentences. The dataset, loaded through a csv file sourced from Kaggle, should be placed in the `in` directory, and can be sourced from [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons).

Alternatively, the dataset can be sourced through the `kaggle_dataset_downloader.py` script, which interfaces the Kaggle API. Note that Kaggle credentials are required. Please see the [usage section](#-usage) for more details.

### 🤖 Model
This project primarily utilizes the [emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) BERT model from j-hartmann for constructing the emotion analysis pipeline. The model is intended to classify emotions in english text data: It follows the framework of Ekman's 6 basic emotions (and neutral) for its classification categories. The pre-trained model is accessed through the Hugging Face transformers module's pipeline method.

## ⚙️ Setup
To set up the project, you need to create a virtual environment and install the required packages. You can do this by running the appropriate setup script for your operating system.

### 🐍 Dependencies
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.3`

### 💾 Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-vis-analytics
```
2. Navigate to the project directory
```sh
cd assignments
cd emotion_analysis
```
3. Run the setup script to install dependencies, depending on OS.
```sh
bash setup_unix.sh
```
3.5 (Optional) download the dataset from Kaggle (need to have kaggle.json in project).
```sh
python src/kaggle_dataset_downloader.py -i in -u https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons
```
1. Activate virtual environment (OS-specific) and run main py scripts.
```sh
source env/bin/activate
python src/face_detection.py
```
## 🚀 Usage
Once the virtual environment is set up and dependencies are installed, execute the main script `emotion_analysis_pipeline.py` with your preferred command-line options.

If the script is run without providing an argument for the `-pdp` flag, the script executions the following sequence: Paths are instantiated from CLI args. Hugging Face pipeline is instantiated, set to return the emotion with the highest probability/presence. The data is loaded, and the classifier is ran on each sentence through the `emotion_analysis_pipeline` function. 

The results are optionally saved to a csv file, and plots visualizing emotion distribution by season, and emotion fluctuations across entire show are displayed, optionally saved as well.

Specifying an input for the `-pdp` flag targets a csv already containing an emotion classification column, and visualizes the resutls accordingly. 

### 🧰 Utilities
- ``cli_decorator.py``: Contains decorators wrapper for the command-line interface (CLI) click options.
- ``data_manipulation_utils.py``: Contains functions for manipulating data, such as loading, manipulating, and exporting dataframes with pandas.
- `logger_utils.py`: Contains functions for setting up and getting a logger.
- ``plotting_utilities.py``: Handles visualizing data, contains helper functions for modularity. 

### 📥 Kaggle Dataset Downloader
This script is designed to download datasets from Kaggle. It uses the Kaggle API, asyncio for asynchronous operations, and click for a simple CLI implementation. 

The script is primarily interfaced through the `KaggleDatasetManager` class, which handles authenticating the Kaggle API, downloading datasets, and administrating other classes, like `KaggleCredentialsManager`, which is responsible for managing API authentication and `DirectoryManipulator`, which handles all directory-related operations.

The script would benefit from a more cohesive OOP design. 

### 💻 CLI Reference
The CLI is implemented through the click module. The CLI is implemented throguh a custom decorator, which wraps the click options, this is done for readability. The script `emotion_analysis_pipeline.py` can be run from the command line with several options:

| Option | Short | Default | Type | Description |
| --- | --- | --- | --- | --- |
| `--input_csv_path` | `-i` | "Game_of_Thrones_Script.csv" | str | Path to the input CSV file relative to this script's parent folder |
| `--output_csv_path` | `-o` | None | str | Path to the output CSV file |
| `--output_plot_path` | `-op` | None | str | Path to the output plot file. If not provided, plots will not be saved |
| `--processed_data_path` | `-pdp` | None | str | Path to the processed csv file. If not provided, the pipeline will process the input data and save it to a new file |
| `--filter_out_neutral_tag` | `-f` | False | bool | If true, disregard neutral emotion tags in visualization |
| `--rescale-y-axis_for_fluctuation_plot` | `-ry` | False | bool | If true, rescale the y-axis 0-1 for the fluctuation plot |
| `--hf_model` | `-m` | "j-hartmann/emotion-english-distilroberta-base" | str | Name of the Hugging Face model to use for classification |
| `--raw_text_column` | `-rtc` | "Sentence" | str | Name of the column containing raw text data |
| `--emotion_column_title` | `-ect` | "Emotion" | str | Name of the column to store the predicted emotion |
| `--score_column_title` | `-sct` | "Score" | str | Name of the column to store the prediction score |

## 📊 Results
This project's results consist of a csv containing classifications for each sentence, and visualizations depicting the distribution of emotions for each season (relative frequency) in a bar plot, and the trend of emotions across the entire series, visualized as a line plot for visual clarity.

### 🔍 Emotion Distribution by Season
With the intent of providing a more insightful analysis, this discussion will consider the visualization which has filtered the neutral emotion. 

The plot was produced by running the following command:
```py
python src/emotion_analysis_pipeline.py -pdp Game_of_thrones_Script_emotion_classification.csv -op "out/plots" -f
```

Please refer to the plots below: 

<p float="left">
  <figure style="display: inline-block;">
    <img src="/out/plots/emotion_counts_by_season_base.png" width="400" alt="Emotion Counts by Season with neutral tags" />
    <figcaption style="text-align: center">Fig.1 - Emotion Counts by Season.</figcaption>
  </figure>
  <figure style="display: inline-block;">
    <img src="/out/plots/emotion_counts_by_season_filtered.png" width="400" alt="Emotion counts by season with neutral tags filtered out" />
    <figcaption style="text-align: center">Fig.2 - Emotion Counts by Season (no neutral).</figcaption>
  </figure>
</p>

Looking at the two plots, there's an immediate difference when the neutral classifications have been filtered out - This could provide a more salient insight into the emotions at play of each individual seasons.

The overwhelming presence of neutral classifications might be due to the model defaulting to neutral, or simply because dialogue typical of the high fantasy genre exhibit this tendency. 

Immediately from looking at the plot, there's some clear trends. For every season, anger is the most prominent emotion depicted in the script. Season 8 contains the highest percentage of sentences classified as anger, perhaps mirroring the sentiment of the fans towards their subversion of expectations. 

Additionally disgust and surprise are also expressed quite frequently. Disgust was especially prominent in earlier seasons, peaking in season 3, and steadily trending downwards after this peak.

Fear, and especially joy, are generally uncharacteristic of this series.

Overall, I think the distributions are quite uniform across all seasons, with some notable exceptions like season 1-3 disgust. This might suggest cohesion, or even a formulaeric emotion arcs pertaining to the series. 

It would've been interesting to visualize emotion distributions across episodes, perhaps yielding insights as to the emotional nature of the individual episode, and the development of emotions across the season.

### 🌱 Emotion Flunctuations across Series 

<p float="left">
  <figure style="display: inline-block;">
    <img src="/out/plots/emotion_flunctuations_across_seasons_base.png" width="400" alt="Line plot of emotion flunctuations by season" />
    <figcaption style="text-align: center">Fig.3 - Emotions flunctuations by season.</figcaption>
  </figure>
  <figure style="display: inline-block;">
    <img src="/out/plots/emotion_flunctuations_across_seasons_scaled.png" width="400" alt="Line plot depicting emotions flunctuations by season with y-axis scaled between 0 and 1" />
    <figcaption style="text-align: center">Fig.4 - Emotions flunctuations by season (scaled).</figcaption>
  </figure>
</p>

This visualization provides insight into the trends of each emotion across the series' runtime. The sub plot's y-axes are based on the min-max, which is somewhat of a limitation, as it masks the degree of change.

However, some tendencies can be observed from the visualizations. In accordance to the previous section's analysis, the trajectory of sentences classified as disgust is clearly depicted here. Additionally, there's a clear spike in joy during seasons 3-6, with season 5 being the peak, and season 6 denoting a sharp drop-off. 

In retrospect, it might've been optimal to visualize all emotions within the same plot, this might've provided a more suitable analytical aid, adding some visual clarity for easier comparison between emotions in the same representational vector space.

### 🔮 Conclusion 
By default, the classifier pipeline provides certainty scores for each emotion, which are also written to the csv file. These are not taken into account when visualizing the results. This reflects a greater are of concern with the reliability of the analysis, namely the noticable lack of certainty regarding the results. The model card on HF specifies an evaluation accuracy of 66%, which needs to be considered when appreciating the analysis. Additionally, classifying emotions in non-trivial, and the fact that the model's "understanding" of emotions is based on the framework of Ekman's 6 emotions needs to be considered as well.

Yet, the methodology presents a quantitative approach to analysing cultural data. A large corpus was provided, and novel insights were actualized. The specific methodology of emotion analysis presents an interesting approach for making sense of how emotions intersect with culture, and perhaps about emotions in general.
## 📖 References
[Game of Thrones Script All Seasons dataset](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons)