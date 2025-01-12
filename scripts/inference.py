import os
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Get the directory of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the base directory to the parent directory (../)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Define default paths relative to BASE_DIR
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "breast_cancer_data.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "data", "models", "feature_names.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, "data", "models", "selected_feature_names.pkl")

def load_test_data(test_data_dir=PROCESSED_DATA_DIR):
    """
    Load test data from JSON files.

    Parameters:
        test_data_dir (str): Directory containing the test data JSON files.

    Returns:
        X_test (array): Test features.
        y_test (array): Test labels.
        feature_names (list): List of selected feature names.
    """
    try:
        with open(os.path.join(PROCESSED_DATA_DIR, "X_test.json"), "r") as f:
            X_test_dict = json.load(f)

        with open(os.path.join(PROCESSED_DATA_DIR, "y_test.json"), "r") as f:
            y_test = np.array(json.load(f))

        feature_names = list(X_test_dict.keys())
        X_test = np.array([X_test_dict[feature] for feature in feature_names]).T

        return X_test, y_test, feature_names

    except Exception as e:
        raise Exception(f"Error loading test data: {str(e)}")


def load_model(model_path=os.path.join(MODELS_DIR, "best_model.pkl")):
    """
    Load the trained model from a pickle file.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        model: Loaded model.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate the model's predictions with various metrics.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        y_prob (array): Predicted probabilities.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_prob),
    }
    return metrics


def main():
    try:
        # Load test data
        X_test, y_test, selected_features = load_test_data()

        # Load the trained model
        model = load_model()

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification

        # Evaluate predictions
        metrics = evaluate_model(y_test, y_pred, y_prob)
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"Error in inference pipeline: {str(e)}")


if __name__ == "__main__":
    main()
