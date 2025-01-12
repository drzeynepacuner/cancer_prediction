import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
from src.data_preprocessing import load_and_preprocess_data, resample_data
from src.feature_selection import select_top_features
from src.model_training import evaluate_and_tune_models
from src.explainability import explain_model
import joblib
import json
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Get the directory of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the base directory to the parent directory (../)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Validate BASE_DIR
if not os.path.isdir(BASE_DIR):
    raise ValueError(f"Invalid BASE_DIR: {BASE_DIR}")

logger.info(f"Base directory set to: {BASE_DIR}")

# Define default paths relative to BASE_DIR
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "breast_cancer_data.csv")

# Ensure necessary directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# Configure logger
logger.add(
    "logs/training.log",  # Save training logs in logs/training.log
    rotation="1 MB",
    retention="10 days",
    compression="zip"
)

def save_test_data(X_test, y_test, feature_names, output_dir="data/processed"):
    """
    Save the test data (X_test and y_test) to JSON files.

    Parameters:
        X_test (array): The test features.
        y_test (array): The test labels.
        feature_names (list): List of feature names used in the test data.
        output_dir (str): Directory to save the JSON files.
    """
    try:
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert X_test to a dictionary
        X_test_dict = {feature: X_test[:, idx].tolist() for idx, feature in enumerate(feature_names)}

        # Save X_test.json
        with open(os.path.join(output_dir, "X_test.json"), "w") as f:
            json.dump(X_test_dict, f, indent=4)

        # Save y_test.json
        with open(os.path.join(output_dir, "y_test.json"), "w") as f:
            json.dump(y_test.tolist(), f, indent=4)

        print(f"Test data saved in {output_dir}")

    except Exception as e:
        print(f"Error saving test data: {str(e)}")


def main():
    try:
        data_path = "data/raw/breast_cancer_data.csv"

        # Step 1: Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_data(data_path)

        joblib.dump(feature_names, "data/models/feature_names.pkl")

        # Step 2: Resample data
        X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

        # Step 3: Feature selection
        X_train_reduced, selected_features = select_top_features(X_train_resampled, y_train_resampled, feature_names)
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_val_reduced = X_val[:, selected_indices]
        X_test_reduced = X_test[:, selected_indices]

        # Save processed test data
        save_test_data(X_test_reduced, y_test, selected_features)

        class_ratio = len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])

        # Step 4: Define model configs and train
        model_configs = {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "max_features": ["sqrt", "log2"],
                    "class_weight": ["balanced"]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "scale_pos_weight": [class_ratio]
                }
            }
        }
        results = evaluate_and_tune_models(model_configs, X_train_reduced, y_train_resampled, X_val_reduced, y_val)

        # Step 5: Save the best model
        best_model_name = max(results, key=lambda x: results[x]["validation_metrics"]["F1-Score"])
        best_model = results[best_model_name]["best_model"]
        best_metrics = results[best_model_name]["validation_metrics"]

        # Log the best model and metrics
        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Hyperparameters: {results[best_model_name]['best_params']}")
        logging.info(f"Validation Metrics: {best_metrics}")

        # Print details for terminal feedback
        print("\nBest Model Details:")
        print(f"Model Name: {best_model_name}")
        print(f"Best Hyperparameters: {results[best_model_name]['best_params']}")
        print("Validation Metrics:")
        for metric, value in best_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save the best model
        joblib.dump(best_model, "data/models/best_model.pkl")
        joblib.dump(best_model, "data/models/best_model.pkl")

        logger.info("Best model saved to data/models/best_model.pkl")

        # Step 6: Explain model predictions
        explain_model(best_model, X_test_reduced, selected_features)

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
