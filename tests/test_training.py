import pytest
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_selection import select_top_features
from src.model_training import evaluate_and_tune_models


@pytest.fixture
def mock_training_data():
    """
    Fixture to create mock training data for testing.
    """
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ])
    y = np.array([0, 1, 0, 1])  # Binary classification labels
    feature_names = ["feature1", "feature2", "feature3"]
    return X, y, feature_names

def test_select_top_features(mock_training_data):
    """
    Test the feature selection logic.
    """
    X, y, feature_names = mock_training_data

    # Call the function
    X_reduced, selected_features = select_top_features(X, y, feature_names, top_percent=0.66)

    # Assert outputs
    assert len(selected_features) == 2, "Top 66% of 3 features should result in 2 features."
    assert X_reduced.shape[1] == 2, "Reduced feature matrix should have 2 features."
    assert all(f in feature_names for f in selected_features), "Selected features must exist in original features."

def test_evaluate_and_tune_models(mock_training_data):
    """
    Test model evaluation and hyperparameter tuning logic.
    """
    X, y, _ = mock_training_data

    # Mock model configurations
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(),
            "params": {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [10, 20],
                "max_depth": [None, 5],
                "max_features": ["sqrt"]
            }
        }
    }

    # Split data into train/validation sets
    X_train, X_val = X[:3], X[3:]
    y_train, y_val = y[:3], y[3:]

    # Call the function
    results = evaluate_and_tune_models(model_configs, X_train, y_train, X_val, y_val)

    # Assert outputs
    assert "Logistic Regression" in results, "Logistic Regression should be evaluated."
    assert "Random Forest" in results, "Random Forest should be evaluated."

    for model_name, result in results.items():
        assert "best_model" in result, f"{model_name} should have a 'best_model' key."
        assert "validation_metrics" in result, f"{model_name} should have 'validation_metrics'."
        assert all(metric in result["validation_metrics"] for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]), \
            f"{model_name} metrics should include Accuracy, Precision, Recall, and F1-Score."

def test_model_metrics():
    """
    Test that model metrics are calculated correctly.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

    # Assert correctness
    assert metrics["Accuracy"] == 0.5, "Accuracy should be 0.5"
    assert metrics["Precision"] == 0.5, "Precision should be 0.5"
    assert metrics["Recall"] == 0.5, "Recall should be 0.5"
    assert metrics["F1-Score"] == 0.5, "F1-Score should be 0.5"
