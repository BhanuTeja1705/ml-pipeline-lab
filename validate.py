# validate.py - Lab 2: Model Validation Gates
# TODO: Implement all 4 gates + run_all_gates()

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

THRESHOLDS = {
    "min_accuracy": 0.85,
    "min_f1": 0.80,
    "regression_tolerance": 0.02,
    "min_per_class_recall": 0.70,
    "expected_feature_count": 4,
}
PROD_BASELINE = {"accuracy": 0.88, "f1": 0.87}


def load_test_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def gate_schema_validation(X_test):
    # ### YOUR CODE ###
    # Checking if number of features matches expected value (should be 4 for Iris dataset)
    if X_test.shape[1] != THRESHOLDS["expected_feature_count"]:
        return False, f"Expected {THRESHOLDS['expected_feature_count']} features, got {X_test.shape[1]}"

    # Checking if dataset contains any NaN (missing values)
    if np.isnan(X_test).any():
        return False, "NaN values detected in dataset"

    # If both checks pass → schema is valid
    return True, "Schema valid"


def gate_performance(model, X_test, y_test):
    # ### YOUR CODE ###
    # Predicting using trained model
    preds = model.predict(X_test)

    # Calculating performance metrics
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    metrics = {"accuracy": accuracy, "f1_score": f1}

    # Checking minimum thresholds
    if accuracy < THRESHOLDS["min_accuracy"]:
        return False, metrics, f"Accuracy too low: {accuracy:.4f}"

    if f1 < THRESHOLDS["min_f1"]:
        return False, metrics, f"F1 score too low: {f1:.4f}"

    return True, metrics, "Performance OK"


def gate_regression(new_metrics):
    # ### YOUR CODE ###
    # Ensuring new model is not worse than production model beyond tolerance
    new_accuracy = new_metrics["accuracy"]
    baseline = PROD_BASELINE["accuracy"]
    tolerance = THRESHOLDS["regression_tolerance"]

    # Allowed minimum accuracy
    min_allowed = baseline - tolerance

    if new_accuracy < min_allowed:
        return False, f"Regression detected: {new_accuracy:.4f} < {min_allowed:.4f}"

    return True, "No regression"


def gate_fairness(model, X_test, y_test):
    # ### YOUR CODE ###
    # Predicting outputs
    preds = model.predict(X_test)

    # Calculating recall for each class separately
    # average=None → gives recall per class (important for fairness)
    per_class = recall_score(y_test, preds, average=None)

    # Checking if any class has low recall
    for i, r in enumerate(per_class):
        if r < THRESHOLDS["min_per_class_recall"]:
            return False, dict(enumerate(per_class)), f"Class {i} recall too low: {r:.4f}"

    return True, dict(enumerate(per_class)), "Fairness OK"


def run_all_gates(model=None):
    # ### YOUR CODE ###
    # Run all 4 gates IN ORDER. Stop and return on first FAIL.
    # Print: "[GATE N] Name: PASS/FAIL - message"
    # Return: {"status": "PASS"/"FAIL", "failed_gate": str, "reason": str, "metrics": dict}

    print('=' * 50)
    print('VALIDATION PIPELINE')
    print('=' * 50)

    from sklearn.ensemble import RandomForestClassifier

    X_test, y_test = load_test_data()

    # If no model is passed, train a default one
    if model is None:
        iris = load_iris()
        X_tr, _, y_tr, _ = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # IMPORTANT: keep same config as training (max_depth)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_tr, y_tr)

    # Default result
    result = {"status": "FAIL", "failed_gate": "none", "reason": "", "metrics": {}}

    # ---------------- GATE 1: SCHEMA ----------------
    passed, msg = gate_schema_validation(X_test)
    print(f"[GATE 1] Schema: {'PASS' if passed else 'FAIL'} - {msg}")

    if not passed:
        result["failed_gate"] = "schema"
        result["reason"] = msg
        return result

    # ---------------- GATE 2: PERFORMANCE ----------------
    passed, metrics, msg = gate_performance(model, X_test, y_test)
    print(f"[GATE 2] Performance: {'PASS' if passed else 'FAIL'} - {msg}")

    if not passed:
        result["failed_gate"] = "performance"
        result["reason"] = msg
        return result

    # ---------------- GATE 3: REGRESSION ----------------
    passed, msg = gate_regression(metrics)
    print(f"[GATE 3] Regression: {'PASS' if passed else 'FAIL'} - {msg}")

    if not passed:
        result["failed_gate"] = "regression"
        result["reason"] = msg
        return result

    # ---------------- GATE 4: FAIRNESS ----------------
    passed, per_class, msg = gate_fairness(model, X_test, y_test)
    print(f"[GATE 4] Fairness: {'PASS' if passed else 'FAIL'} - {msg}")

    if not passed:
        result["failed_gate"] = "fairness"
        result["reason"] = msg
        return result

    # If all gates pass
    result["status"] = "PASS"
    result["metrics"] = metrics

    return result


if __name__ == '__main__':
    result = run_all_gates()
    print("\nFINAL:", result["status"])