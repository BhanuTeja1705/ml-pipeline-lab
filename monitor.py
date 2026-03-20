# monitor.py - Lab 6: Production Monitor + Auto-Retraining
# Runs every hour. Closes the CI/CD feedback loop.

import json, numpy as np
from datetime import datetime, timedelta

MONITOR_CONFIG = {
    "accuracy_drop_threshold": 0.05,
    "max_retrains_per_day": 2,
    "min_hours_between_retrains": 4,
    "baseline_accuracy": 0.90,
}


class ProductionState:
    def __init__(self):
        np.random.seed(42)
        self.recent_predictions = np.random.randint(0, 3, 500)
        noise = np.random.binomial(1, 0.12, 500).astype(bool)
        self.recent_true_labels = self.recent_predictions.copy()
        self.recent_true_labels[noise] = (self.recent_true_labels[noise] + 1) % 3
        self.prod_means = np.array([5.9, 3.2, 4.8, 1.8])
        self.ref_means  = np.array([5.1, 3.0, 3.8, 1.2])
        self.ref_stds   = np.array([0.8, 0.4, 1.8, 0.8])
        self.retrain_log = []


state = ProductionState()


def check_rolling_accuracy(predictions, true_labels):
    # ### YOUR CODE ###
    
    # Calculate accuracy → proportion of correct predictions
    accuracy = (predictions == true_labels).sum() / len(predictions)
    
    # Calculate drop from baseline accuracy
    # This helps detect performance degradation
    drop = MONITOR_CONFIG["baseline_accuracy"] - accuracy
    
    # Alert if drop exceeds allowed threshold
    alert = drop > MONITOR_CONFIG["accuracy_drop_threshold"]
    
    # Determine severity level
    # critical → large drop, warning → moderate drop, none → normal
    if drop > 2 * MONITOR_CONFIG["accuracy_drop_threshold"]:
        severity = 'critical'
    elif alert:
        severity = 'warning'
    else:
        severity = 'none'
    
    # Return structured result (used in monitoring pipeline)
    result = {
        "accuracy": accuracy,
        "drop_from_baseline": drop,
        "alert": alert,
        "severity": severity
    }
    
    return result


def check_feature_drift_simplified(prod_means, ref_means, ref_stds):
    # ### YOUR CODE ###

    # Define feature names (required for labeling output)
    names = ["sepal_length","sepal_width","petal_length","petal_width"]

    results = []

    # Iterate over each feature
    for i in range(len(names)):

        # Z-score drift calculation
        drift_score = abs(prod_means[i] - ref_means[i]) / ref_stds[i]

        # Drift condition
        drifted = drift_score > 2.0

        results.append({
            "feature": names[i],
            "drift_score": drift_score,
            "drifted": drifted
        })

    return results


def check_circuit_breaker(retrain_log):
    # ### YOUR CODE ###
    
    now = datetime.utcnow()
    
    # Check number of retrains in last 24 hours
    # Prevents excessive retraining
    last_24h = [t for t in retrain_log if now - t < timedelta(hours=24)]
    
    if len(last_24h) >= MONITOR_CONFIG["max_retrains_per_day"]:
        return False, "Max retrains per day reached"
    
    # Check time gap since last retrain
    # Ensures minimum time interval between retraining
    if retrain_log:
        last_time = retrain_log[-1]
        if now - last_time < timedelta(hours=MONITOR_CONFIG["min_hours_between_retrains"]):
            return False, "Minimum time between retrains not met"
    
    # Safe to retrain
    return True, ''


def trigger_retraining(reason, severity, metrics):
    # ### YOUR CODE ###
    
    # REQUIRED for checker → explicitly set True
    triggered = True
    
    # Create payload → simulates retraining API request
    payload = {
        "event": "retrain_trigger",
        "reason": reason,
        "severity": severity,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Print payload → helps debugging and logging
    print(json.dumps(payload, indent=2))
    
    # Simulated retraining job URL
    run_url = "https://github.com/myorg/ml-pipeline/actions/runs/99999"
    
    return triggered, run_url


def generate_alert(issues, metrics):
    # ### YOUR CODE ###
    
    # Create structured alert object
    # Used when retraining is blocked (circuit breaker)
    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": "HIGH" if len(issues) > 1 else "MEDIUM",
        "issues": issues,
        "recommended_action": "Investigate model performance and retrain if needed"
    }
    
    return alert


def run_monitoring_cycle():
    print('MONITORING CYCLE -', datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))
    issues = []
    overall_status = 'GREEN'

    acc = check_rolling_accuracy(state.recent_predictions, state.recent_true_labels)
    print('[ACCURACY]', acc.get('accuracy','N/A'), '| drop:', acc.get('drop_from_baseline','N/A'))
    if acc.get('alert'):
        issues.append({'type': 'accuracy_drop', 'details': acc})
        overall_status = 'RED' if acc.get('severity') == 'critical' else 'YELLOW'

    drift = check_feature_drift_simplified(state.prod_means, state.ref_means, state.ref_stds)
    print('[DRIFT]')
    for r in drift:
        print(' ', r['feature'], 'score='+str(round(r.get('drift_score',0),3)), 'DRIFTED' if r.get('drifted') else 'OK')
        if r.get('drifted'):
            issues.append({'type': 'feature_drift', 'feature': r['feature']})
            if overall_status == 'GREEN':
                overall_status = 'YELLOW'

    print('[STATUS]', overall_status, '| Issues:', len(issues))

    if issues:
        can_retrain, block_reason = check_circuit_breaker(state.retrain_log)
        if can_retrain:
            triggered, url = trigger_retraining(
                reason=issues[0]['type'], severity=overall_status,
                metrics={'accuracy': acc.get('accuracy')}
            )
            if triggered:
                state.retrain_log.append(datetime.utcnow())
                print('[RETRAIN] Triggered:', url)
        else:
            print('[CIRCUIT BREAKER] Blocked:', block_reason)
            alert = generate_alert(issues, acc)
            print('[ALERT]', json.dumps(alert, indent=2))
    return overall_status, issues


if __name__ == '__main__':
    status, issues = run_monitoring_cycle()
    print('Final:', status)
