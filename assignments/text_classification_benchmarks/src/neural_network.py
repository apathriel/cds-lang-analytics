import os
from statistics import mean
import sys
sys.path.append("..")

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump

def load_labeled_data_as_df(path_to_data):
    return pd.read_csv(path_to_data)

def instantiate_vectorizer(vectorizer, base_ngram_range=(1,2), base_lowercase=True, base_max_df=0.95, base_min_df=0.05, base_max_features=100):
    return vectorizer(ngram_range = base_ngram_range, lowercase =  base_lowercase, max_df = base_max_df, min_df = base_min_df, max_features = base_max_features)

def save_classification_report(classification_report, file_path):
    with open(file_path, "w") as file:
        file.write(classification_report)
    print(f"[INFO] Classification report saved as {os.path.basename(file_path)}")

def save_object(object_to_save, dir_path, object_name):
    file_name = f'{os.path.join(dir_path, os.path.basename(__file__).split(".")[0])}_{object_name}.joblib'
    dump(object_to_save, file_name)
    print(f"[INFO] {object_name.capitalize()} saved as {os.path.basename(file_name)}")

def cross_validation(classifier, X_train_features, y_train, cv_fold=10):
    return cross_val_score(classifier, X_train_features, y_train, cv=cv_fold) 

def prepare_data(data, text_col, label_col, vectorizer, train_test_size=0.2, seed=24):
    print("[INFO] Preparing data...")
    X = data[text_col]
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, random_state=seed)
    
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    return X_train_feats, X_test_feats, y_train, y_test

def train_model(X_train, y_train):
    print("[INFO] Training neural network classifier...")
    return MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42).fit(X_train, y_train)

def train_neural_network(data, text_col, label_col, report_path, model_path, train_test_size=0.2, seed=24, cross_validate=False, cv_fold=10):
    vectorizer = instantiate_vectorizer(TfidfVectorizer, base_max_features=100)
    X_train_feats, X_test_feats, y_train, y_test = prepare_data(data, text_col, label_col, vectorizer, train_test_size, seed)
    classifier = train_model(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)

    if cross_validate:
        scores = cross_validation(classifier, X_train_feats, y_train, cv_fold)
        print(f"[INFO] Cross-validated mean score: {round(mean(scores), 2)}")

    save_classification_report(metrics.classification_report(y_test, y_pred), report_path)
    save_object(vectorizer, model_path, 'vectorizer')
    save_object(classifier, model_path, 'classifier')


def main():
    # define input/output paths
    input_data_path = os.path.join("in", "fake_or_real_news.csv")
    report_data_path = os.path.join("out", "neural_network_report.txt")
    model_data_path = os.path.join("models", "neural_network")

    train_neural_network(load_labeled_data_as_df(input_data_path), "text", "label", report_path = report_data_path, model_path=model_data_path)

if __name__ == "__main__":
    main()