# ⏱️ Text classification benchmarks

This project is designed to perform text classification tasks using two different machine learning models: Logistic Regression and Neural Network. Both scripts utilize the architectures afforded by the `scikit-learn` module to perform binary classification on labeled data. The project utilizes supervised machine learning with the dataset [Fake or Real News](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) constituting the training data. A classifier is trained using the 2 different model architectures, and their performance are evaluated through a classifcation report, which is exported to a .txt file. Additionally, the project utilizes the `joblib` module, to export the classifier model and vectorizer to the `out/models` directory.

## 📈 Data

### 📋 Dataset
The project utilizes the [Fake or Real News](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) dataset for text classification. The dataset, loaded through a csv file, should be placed in the `in` directory, and can be sourced from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). Please see an example of the dataset below:

| # | Title      | Text                  | Label   |
|---|------------|-----------------------|---------|
| 8476 | You Can Smell Hillary’s Fear | Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing... | FAKE |


### 🤖 Model
This project uses two different machine learning models for text classification: `LogisticRegression` and `MLPClassifier`, each representing logistic regression- and neural network models respectiely. These models are implemented in the scripts logistic_regression.py and neural_network.py respectively.

## 📂 Project structure
```
└── text_classification_benchmarks
	├── in/
	│   └── fake_or_real_news.csv
    │
    ├── out/
    │    ├── models/
    │    │    ├── logistic_regression/
    │    │    │    ├── logistic_regression_classifier.joblib
    │    │    │    └── logistic_regression_vectorizer.jotblib
    │    │    │
    │    │    └── neural_network/
    │    │         ├── neural_network_classifier.joblib
    │    │         └── neural_network_vectorizer.joblib
    │    │
    │    ├── logistic_regression_report.txt
    │    ├── logistic_regression_cross_validated_scores.csv
    │    ├── neural_network_report.txt
    │    └── neural_network_cross_validated_scroes.csv
    │ 
	├── src/
	│   ├── data_processing_utilities.py
	│   ├── emission_tracker_class.py
	│   ├── logistic_regression.py
	│   ├── neural_network
	│   ├── utilities.py
	│   └── vectorize_dataset.py
    │
	├── setup.sh
	├── requirements.txt
	└── README.md
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
cd text_classification_benchmarks
```
3. Run the setup script to install dependencies, depending on OS.
```sh
bash setup_unix.sh
```
4. Activate virtual environment (OS-specific) and run main py scripts.
```sh
source env/bin/activate
python src/logistic_regression.py
```

## 🚀 Usage
Once dependencies are installed and your environment is set up, you can run the project scripts. The main scripts use pre-set hyperparameters for the `LogisticRegression` and `MLPClassifier` models. These parameters were determined through iterative testing and a grid search using scikit-learn. Running the scripts will use these default parameters. Optionally, the `neural_network.py` script has grid search functionality through the `GridSearchCV` method from `scikit-learn`. 

### 🧰 Utilities
- ``data_processing_utilities.py``: This module handles data loading, preprocessing, splitting, and model training preparation. It also handles saving classification reports, cross-validated scores, and trained models & vectorizers.
- ``utilities.py``: This module contains the get_logger function, which is used to set up logging for the project.
- ``vectorize_dataset.py``: This module is responsible for vectorizing the dataset. It can use either the ``CountVectorizer`` or ``TfidfVectorizer`` from `sciit-learn`

## 📊 Results
The results of the binary classification task are saved to the `out` directory for both model scripts. Out-of-the-box, the models perform similarly: After tweaking the hyperparameters, they still perform similarly. Both have a f1-score & 10-fold cross validated mean score of 0.91. This means that both models have a high degree of precision and recall in their predictions, and they generalize well to the training data.

## 📖 References
[Fake or Real News dataset by jillanisofttech](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)
[LogisticRegression class from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[MLPClassifier class from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)