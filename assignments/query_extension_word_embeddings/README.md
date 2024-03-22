# 🕵🏻 Query expansion with word embeddings

This project is part of the portfolio for the course 'Language Analytics' - Bachelor's elective in Cultural Data Science.

The project examines the presence of a specified concept within a specified artist's corpus. In order to gain a more nuanced exploration of the concept's frequency, word embeddings are used to expand to query to include similiar words.

## 📈 Data

### 📋 Dataset
The project utilizes a dataset [57.650 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs) sourced from Kaggle. The corpus is comprised of lyrics 57.650 english-language songs. Please refer to the dataset [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs).

|      artist       |        song        |        link        |        text        |
|:-----------------:|:------------------:|:------------------:|:------------------:|
| 643 unique values | 44824 unique values| 57650 unique values| 57494 unique values|


### 🤖 Model
By default, the project utilizes the 'glove-wiki-gigaword-50' model from gensim. The script accepts user-defined gensim model names with the `--model` flag, which the script attempts to load using the `gensim.downloader` module.


## 📂 Project structure
```
└── query_extension_word_embeddings
	├── in
	│   └── spotify_million_dataset.csv
	├── out
	│   └── query_results.csv    
	├── src
	│   ├── query_expansion.py
	│       └── utils
	│        ├── cli_utils.py
	│        ├── data_processing_utils.py
	│        ├── model_utils.py
	│        └── utilities.py
	├── README.md
	├── requirements.txt
	├── setup_unix.sh
	└── setup_win.sh
```

## ⚙️ Setup

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
``` sh
bash setup_unix.sh
```
4. Run the main script through the CLI.
```sh
python src/query_expansion.py -a -q
```

### ⚒️ Usage 
The project is run from the `query_expansion.py` script, which manages the CLI. Please refer to the reference below for documentation of the CLI's functionality. Generally, the script accepts a list of artists and a query to expand. The query is expanded through word embeddings. A list of the 10 most similar words is compared to each song within the specified artists's corpus. The percentage of the artists's corpus containing elements from the word embeddings are printed and optionally saved to a .csv file.

## 💻 CLI Reference
Please use the `--help` flag for an overview of the CLI functionality.

|flag     |shorthand|description                                               |type|required|
|---------|---------|----------------------------------------------------------|----|--------|
|\--artist|\-a      |Takes one or more artists in dataset to filter by         |list|TRUE    |
|\--query |\-q      |Takes one query from which to explore the artists's corpus|str |TRUE    |
|\--model |\-m      |Optional user-defined gensim model                        |str |FALSE   |
|\--save  |\-s      |Whether to save result to csv file                        |bool|FALSE   |
|\--output|\-o      |Output file name                                          |str |FALSE   |
