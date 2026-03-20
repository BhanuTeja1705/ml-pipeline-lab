# train.py - Lab 1: Versioned Training Pipeline
# TODO: Complete all sections marked ### YOUR CODE ###

import hashlib, json, os, pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
    "test_size": 0.2,
    "model_version": "v1.0.0",
}


def compute_data_hash(X, y):
    # ### YOUR CODE ###
    # Convert dataset into bytes → ensures exact data representation
    data_bytes = X.tobytes() + y.tobytes()

    # Generate SHA-256 hash → used for tracking dataset version
    # Same data → same hash (important for reproducibility)
    return hashlib.sha256(data_bytes).hexdigest()


def load_and_split_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    # ### YOUR CODE ###
    # Splitting dataset into training and testing
    # test_size = 20% of data used for testing
    # random_state ensures SAME split every time (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    # Returning both split and full dataset (for hashing)
    return X_train, X_test, y_train, y_test, X, y


def train_model(X_train, y_train):
    # ### YOUR CODE ###
    # Creating RandomForest model using CONFIG parameters
    model = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],  # number of trees
        max_depth=CONFIG["max_depth"],        # limits tree depth → avoids overfitting
        random_state=CONFIG["random_state"]   # ensures same training result
    )

    # Training the model
    model.fit(X_train, y_train)

    # Returning trained model
    return model


def evaluate_model(model, X_test, y_test):
    # ### YOUR CODE ###
    # Predicting outputs for test data
    preds = model.predict(X_test)

    # Calculating evaluation metrics
    # Accuracy → overall correctness
    # F1 Score → balance between precision & recall
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average='weighted')
    }

    return metrics


def run_training():
    print("[INFO] Starting training pipeline")

    X_train, X_test, y_train, y_test, X, y = load_and_split_data()

    if X_train is None:
        print("[ERROR] load_and_split_data() not implemented!")
        return

    print("[INFO] Train:", len(X_train), "Test:", len(X_test))

    # Compute dataset hash
    data_hash = compute_data_hash(X, y)
    print("[INFO] Data hash:", data_hash)

    # Train model
    model = train_model(X_train, y_train)

    if model is None:
        print("[ERROR] train_model() not implemented!")
        return

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("[INFO] Metrics:", metrics)

    # ### YOUR CODE ###
    # Saving trained model to file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Saving metadata for tracking experiment
    # Includes version, metrics, dataset hash, and config
    metadata = {
        "model_version": CONFIG["model_version"],
        "metrics": metrics,
        "data_hash": data_hash,
        "config": CONFIG
    }

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Accuracy:", metrics.get("accuracy", 0))

    return model, metrics, data_hash


if __name__ == '__main__':
    run_training()