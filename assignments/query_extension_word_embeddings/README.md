# 🕵🏻 Query expansion with word embeddings
This project examines the presence of a specified concept within a given artist's corpus. In order to gain a more nuanced exploration of the concept's frequency within the corpus, word embeddings are utilied to expand to query to include similiar words. The word embeddings are extracted from pre-trained GloVe vector representations, loaded from ``glove-wiki-gigaword-50``, which was trained on 2B tweets, 27B tokens, 1.2M vocab.

## 📈 Data

### 📋 Dataset
This project utilizes the dataset [57.650 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs) uploaded by user joebeachcapital, sourced from Kaggle. The corpus is comprised of lyrics from 57.650 english-language songs spanning 643 different artists. The dataset, loaded through a csv file, should be placed in the `in` directory. Please refer to the dataset [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs).

|      artist       |        song        |        link        |        text        |
|:-----------------:|:------------------:|:------------------:|:------------------:|
| 643 unique values | 44824 unique values| 57650 unique values| 57494 unique values|


### 🤖 Model
By default, the project utilizes the [glove-wiki-gigaword-50](https://huggingface.co/fse/glove-wiki-gigaword-50) model from gensim. The script accepts user-defined gensim model names with the `--model` flag, which the script attempts to load using the `gensim.downloader` module.


## 📂 Project structure
```
└── query_extension_word_embeddings
	├── in/
	│   └── spotify_million_dataset.csv
	│
	├── out/
	│   └── query_results.csv    
	│
	├── src/
	│   ├── query_expansion.py
    │   └── utils/
	│        ├── cli_utils.py
	│        ├── data_processing_utils.py
	│        ├── emission_tracker_class.py
	│        ├── logging_utils.py
	│        ├── model_utils.py
	│        └── utilities.py
	│
	├── README.md
	├── requirements.txt
	├── setup_unix.sh
	└── setup_win.sh
```

## ⚙️ Setup
To set up the project, you need to create a virtual environment and install the required packages. You can do this by running the appropriate setup script for your operating system.

### 🐍 Dependencies
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.3`

### 💾 Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-lang-analytics
```
2. Navigate to the project directory
```sh
cd assignments
cd query_extension_word_embeddings
```
3. Run the setup script to install dependencies, depending on OS.
```sh
bash setup_unix.sh
```
4. Activate the virtual environment (OS-specific)
``` sh
source env/bin/activate
```
5. Run the main script through the CLI.
```sh
python src/query_expansion.py -a [ARTIST] -q [QUERY]
```

## 🚀 Usage 
The project is run from the `query_expansion.py` script, which manages the CLI. Please refer to the [CLI reference below](#-cli-reference) for documentation of the CLI's functionality. Generally, the script accepts a list of artists and a query to expand. The query is expanded through word embeddings. A list of the n most similar words (default is 10) is compared against each song within the specified artists's corpus. The percentage of the artists's corpus containing elements from the word embeddings are printed and optionally saved to a .csv file. If csv file with specified `output` name already exists, csv will be loaded in as `df`, and result will be written to that csv. Script checks if artist + query already exists.

### 💻 CLI Reference
Please use the `--help` flag for an overview of the CLI functionality.

|flag     |shorthand|description                                               |type|required|
|---------|---------|----------------------------------------------------------|----|--------|
|\--artist|\-a      |Takes one or more artists in dataset to filter by         |list|TRUE    |
|\--query |\-q      |Takes one query from which to explore the artists's corpus|str |TRUE    |
|\--model |\-m      |Optional user-defined gensim model                        |str |FALSE   |
|\--save  |\-s      |Whether to save result to csv file                        |bool|FALSE   |
|\--output|\-o      |Output file name                                          |str |FALSE   |

### 🧰 Utilities
- cli_utils.py: This module provides utilities for handling command-line interface (CLI) interactions with click.
- data_processing_utils.py: This module contains functions for processing and manipulating data.
- logging_utils.py: This module provides utilities intantiate logger through logging module.
model_utils.py: This module contains utilities for handling and utilizing gensim models and getting word embeddings.
- utilities.py: This module provides additional general-purpose utilities.

## 📊 Results
Please refer to the table below for an overview of the project's results from [query_results.py](./out/query_results.csv).

| Percent | Artist      | Query |
|---------|-------------|-------|
| 98.93   | Cher        | love  |
| 23.53   | Cher        | hate  |
| 97.6    | Depeche Mode| love  |
| 17.37   | Depeche Mode| hate  |
| 98.19   | Korn        | love  |
| 69.88   | Korn        | hate  |

Looking at this table, we can surmise that all three artists overwhelmingly sing about love (and who can blame them!), but only Korn frequently incorporate concepts related to hate in their music.

As (briefly) demonstrated above, this method can be used to compare the corpus of multiple artists. Taking this idea further, one could identify crucial/essential concepts from specific genres by examining the corpus of influential/iconic artists to that genre (Korn for nü-metal or Depeche Mode for synth pop).

Additionally, you could examine an artist's musical evolution, if you query the concept by album. This could provide an overview of the most important concepts (if repeatedly queried) for each album from an artist's corpus.


## 📖 References
- [57.650 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs)
- [glove-wiki-gigaword-50](https://huggingface.co/fse/glove-wiki-gigaword-50)
- [Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)