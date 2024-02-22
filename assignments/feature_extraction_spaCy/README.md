# Extracting lnguistic information from a corpus using spaCy

This project is part of the portfolio for the course 'Language Analytics' - Bachelor's elective in Cultural Data Science.

The project concerns processing a number of inputs, in the form of text files, specifically extracting linguistic information using spaCy. The extracted information is written to a pandas dataframe, before being outputted as a csv file.

The project utilizes the 'en_core_web_md' model from spaCy, which is made available through the setup.sh script, but other models can be installed and specified.

## Project structure
```
└── feature_extraction_spaCy
	├── data
	│   ├── input
	│   └── output
	├── src
	│   └── process_corpus.py
	├── setup.sh
	├── requirements.txt
	├── README.md
```

## Setup
***Dependencies***
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.2`

### Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-lang-analytics
```
1. Navigate to the project directory
```sh
cd assignment
cd feature_extraction_spaCy
```
1. Run the setup script to install dependencies + spaCy model.
``` sh
bash setup.sh
```
1. Run the main script
```sh
python src/process_corpus.py
```

### Usage 
The `process_folders()` function of main script takes three arguments: an input folder path, an output folder path, and a loaded spaCy model. These are predefined with the intent of showcasing the script's functionality. Thus, the data intended to be processed, needs to be sources from the 'input' directory. By default, punctuation is *not* considered when calculating the relative frequency of POS tags.
