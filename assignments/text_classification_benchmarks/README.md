# Text classification benchmarks

This project is part of the portfolio for the course 'Language Analytics' - Bachelor's elective in Cultural Data Science.

The project contains two python scripts, both use the scikit-learn module to perform text classification on tbe labeled data contained in the "in" data folder. Added option for cross-validation. The outputs consits of a a .txt file containing the classification report located in the "out" folder. Additionally, classifier model and vectorizer are saved in the "models" folder.


## Project structure
```
└── text_classification_benchmarks
	├── in
	│   └── fake_or_real_news.csv
    ├── out
    │   ├── logistic_regression_report.txt
    │   └── neural_network_report.txt
    ├── models
    │    ├── logistic_regression
    │    │    ├── logistic_regression_classifier.jotlib
    │    │    └── logistic_regression_vectorizer.jotlib
    │    └── neural_network
    │         ├── neural_network_classifier.jotlib
    │         └── neural_network_vectorizer.jotlib        
	├── src
	│   ├── logistic_regression.py
    │    └── neural_network.py
	├── setup.sh
	├── requirements.txt
	└── README.md
```

## Setup
***Dependencies***
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.2`

### Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-lang-analytics
```
2. Navigate to the project directory
```sh
cd assignments
cd text_classification_benchmarks
```
3. Run the setup script to install dependencies
``` sh
bash setup.sh
```
4. Run the main scripts
```sh
python src/logistic_regression.py
python src/neural_network.py
```

### Usage 
The main functionality of the script entails defining an 'input path' to the labeled dataset, additionally this could've been accomplished through command line arguments - output paths are defined as well. Vectorizer is created - Tfidfvectorizer is chosen, CountVectorizer could be used as well. Data is loaded and prepared, classifier is fitted. Predictions on test split is performed, and output is computed.