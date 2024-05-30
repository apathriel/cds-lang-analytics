# 🪐 Extracting lnguistic information from a corpus using spaCy
This project is designed to perform linguistic analysis on a corpus of text files using the `spaCy` library. The main functionalities of the project are encapsulated in the Python scripts located in the `src/` directory.

The project concerns processing a number of inputs, in the form of text files, specifically extracting linguistic information using spaCy. The extracted information is written to a pandas dataframe, before being output as a csv file.

The project also supports command-line options for customizing the execution of the main script.


## 📈 Data

### 📋 Dataset
This project utilizes [The Uppsala Student English Corpus (USE)](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457) dataset for feature extraction. The dataset's individual sub folders (a1, a2, etc.) should be placed in the `in` directory, and can be sourced from [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2457/USEcorpus.zip?sequence=5&isAllowed=y).

### 🤖 Model
This project primarily uses the `en_core_web_md` model from `spaCy`. The `en_core_web_md` model supports all core capabilities of `spacy`, includingpart-of-speech (POS) tagging and named entity recognition (NER). The setup scripts automate its installation for immediate use. However, the project's design allows for flexibility. You can install and use other `spaCy` models via the Command Line Interface (CLI).

## 📂 Project structure
```
└── feature_extraction_spaCy
	├── in/
	│	├── a1/
	│	├── ...
	│	└── c1/
	│
	├── out/
	│	├── a1_table.csv
	│	├── ...
	│	└── c1_table.csv
	│
	├── src/
	│   ├── cli_utilities.py
	│	├── data_processing_utilities.py
	│	├── emission_tracker_class.py
	│	├── linguistic_analysis.py
	│	└── utilities.py
	│
	├── README.md
	├── requirements.txt
	├── setup_unix.sh
	└── setup_unix.sh
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
cd assignment
cd feature_extraction_spaCy
```
3. Run the setup script to install dependencies, depending on OS.
```sh
bash setup_unix.sh
```

4. Activate virtual environment (OS-specific) and run main py scripts.
```sh
source env/bin/activate
python src/linguistic_analysis.py
```

## 🚀 Usage 
Once the virtual environment is set up and dependencies are installed, you can execute the main script `linguistic_analysis.py` from the `src/` directory. This script performs linguistic analysis on a corpus of text files using the spaCy library.

Please refer to the example below for running the script:

```py
python src/linguistic_analysis.py -i in -o out --model en_core_web_md
```

Refer to the [CLI Reference](#-cli-reference) section for more information on the available command-line options.

## 💻 CLI Reference
The CLI is implemented through the `argparse` module. The script `linguistic_analysis.py` can be run from the command line with several options:

| Option | Short | Default | Type | Description |
| --- | --- | --- | --- | --- |
| `--input_path` | `-i` | "in" | str | Directory containing the input text files |
| `--output_path` | `-o` | "out" | str | Directory to save the output CSV file |
| `--model` | `-s` | "en_core_web_md" | str | spaCy model to use for linguistic analysis |

The paths are instantiated from a relative path from the root directory: 
```py
input_folder_path = Path(__file__).parent / ".." / cli_args.input_path
output_folder_path = Path(__file__).parent / ".." / cli_args.output_path
```

## 📊 Results
This project generates and exports a csv file containing the extracted linguistic information. The main script generates a csv file for each sub-directory, with a row for each essay txt file. Please refer to the table below for an example.

| Filename    | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | No. Unique PER | No. Unique LOC | No. Unique ORG |
| ----------- | ------------ | ------------ | ----------- | ----------- | -------------- | -------------- | -------------- |
| 0100.a1.txt | 1518.99      | 1223.63      | 815.75      | 534.46      | 0              | 0              | 0              |
| 0101.a1.txt | 1166.88      | 1242.16      | 589.71      | 840.65      | 1              | 0              | 0              |
| 0102.a1.txt | 1485.03      | 1197.6       | 682.63      | 479.04      | 1              | 0              | 0              |
| 0103.a1.txt | 1097.56      | 1363.64      | 598.67      | 576.5       | 1              | 0              | 1              |
| 0104.a1.txt | 1320.99      | 1197.53      | 567.9       | 679.01      | 0              | 1              | 2              |

By default, the relative frequency is calculated per 10000 tokens, with punctuation removed. Punctuation is removed since it presents noise to the methodology of our linguistic analysis - POS tagging & NER.

The distribution of POS (Part-of-Speech) tags within a given essay might inform us about the grammatical structure and complexity of the text. By computationally analyzing the frequency and patterns of different POS tags, we can gain insights into the the overall linguistic characteristics of the essay. This would optimally be coupled with qualitative insights to provide a more comprehensive analysis.

## 📖 References
- [The Uppsala Student English Corpus (USE)](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457)
- [en_core_web_md model from spaCy](https://spacy.io/models/en)