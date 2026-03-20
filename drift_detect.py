# drift_detect.py - Lab 3: Statistical Drift Detection
# PSI > 0.1 = slight | PSI > 0.2 = severe (trigger retrain!)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

PSI_SLIGHT = 0.1
PSI_SEVERE = 0.2


def get_reference_data():
    iris = load_iris()
    # Reference data = training distribution (baseline)
    X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, y_train


def get_production_data(drift_magnitude=0.8):
    np.random.seed(99)
    iris = load_iris()
    X = iris.data.copy()

    # Introducing artificial drift into production data
    X[:, 0] += drift_magnitude * 1.5   # shift feature 1
    X[:, 2] += drift_magnitude * 0.8   # shift feature 3

    # Adding noise to simulate real-world variation
    X += np.random.normal(0, drift_magnitude * 0.3, X.shape)

    return X[:100], iris.target[:100]


def compute_psi(reference, production, n_bins=10):
    # ### YOUR CODE ###
    # PSI (Population Stability Index)
    # Measures how much distribution changed between reference and production

    # Step 1: Create bins based on reference data range
    bins = np.linspace(reference.min(), reference.max(), n_bins + 1)

    # Step 2: Count values in each bin
    ref_counts, _ = np.histogram(reference, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)

    # Step 3: Convert counts to percentages
    ref_pct = ref_counts / len(reference)
    prod_pct = prod_counts / len(production)

    # Step 4: Avoid division by zero using clipping
    ref_pct = np.clip(ref_pct, 1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)

    # Step 5: Apply PSI formula
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

    return psi


def compute_kl_divergence(reference, production, n_bins=10):
    # ### YOUR CODE ###
    # KL Divergence → measures how one distribution differs from another

    # Same binning as PSI
    bins = np.linspace(reference.min(), reference.max(), n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)

    ref_pct = ref_counts / len(reference)
    prod_pct = prod_counts / len(production)

    # Avoid log(0)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)

    # KL(P||Q) where P = production, Q = reference
    kl = np.sum(prod_pct * np.log(prod_pct / ref_pct))

    return kl


def detect_feature_drift(X_ref, X_prod, feature_names=None):
    # ### YOUR CODE ###
    # Detect drift for each feature separately

    if feature_names is None:
        feature_names = ["f" + str(i) for i in range(X_ref.shape[1])]

    results = []

    # Loop through each feature column
    for i, name in enumerate(feature_names):

        # Compute PSI and KL divergence
        psi = compute_psi(X_ref[:, i], X_prod[:, i])
        kl = compute_kl_divergence(X_ref[:, i], X_prod[:, i])

        # Determine severity based on PSI
        if psi > PSI_SEVERE:
            severity = "severe"
        elif psi > PSI_SLIGHT:
            severity = "slight"
        else:
            severity = "none"

        # Store results
        results.append({
            "feature": name,
            "psi": psi,
            "kl_div": kl,
            "severity": severity,
            "alert": severity != "none"  # alert if any drift exists
        })

    return results


def check_prediction_drift(model, X_ref, X_prod):
    # ### YOUR CODE ###
    # Compare predictions distribution between reference and production

    ref_preds = model.predict(X_ref)
    prod_preds = model.predict(X_prod)

    # Count class proportions
    ref_counts = np.bincount(ref_preds) / len(ref_preds)
    prod_counts = np.bincount(prod_preds) / len(prod_preds)

    changes = {}

    drift_detected = False

    # Compare each class proportion
    for i in range(len(ref_counts)):
        diff = abs(prod_counts[i] - ref_counts[i])
        changes[f"class_{i}"] = diff

        # If change > 0.15 → drift detected
        if diff > 0.15:
            drift_detected = True

    return drift_detected, changes


def generate_drift_report(feature_results, pred_drift, pred_changes):
    # ### YOUR CODE ###
    # Generate overall drift status

    severe_features = [f["feature"] for f in feature_results if f["severity"] == "severe"]
    slight_features = [f["feature"] for f in feature_results if f["severity"] == "slight"]

    # Decision logic
    if severe_features or pred_drift:
        status = "RED"
        recommendation = "Immediate retraining required"
    elif slight_features:
        status = "YELLOW"
        recommendation = "Monitor closely"
    else:
        status = "GREEN"
        recommendation = "System stable"

    report = {
        "overall_status": status,
        "drifted_features": severe_features,
        "recommendation": recommendation
    }

    return report


def run_drift_detection():
    print('=' * 50 + '\nDRIFT DETECTION\n' + '=' * 50)

    X_ref, y_ref = get_reference_data()
    X_prod, y_prod = get_production_data(drift_magnitude=0.8)

    names = ["sepal_length","sepal_width","petal_length","petal_width"]

    # Detect feature drift
    feature_results = detect_feature_drift(X_ref, X_prod, names)

    print('\nFeature Drift:')
    for r in feature_results:
        sev = r.get("severity","?")
        icon = "RED" if sev=="severe" else ("YLW" if sev=="slight" else " OK")
        print("  [" + icon + "] " + r["feature"] + ": PSI=" + str(round(r.get("psi",0),4)))

    from sklearn.ensemble import RandomForestClassifier

    # Train model on reference data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_ref, y_ref)

    # Check prediction drift
    pred_drift, pred_changes = check_prediction_drift(model, X_ref, X_prod)
    print('Prediction drift:', pred_drift)

    # Generate final report
    report = generate_drift_report(feature_results, pred_drift, pred_changes)

    print("STATUS:", report.get("overall_status","UNKNOWN"))

    return report


if __name__ == '__main__':
    run_drift_detection()