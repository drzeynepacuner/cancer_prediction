import os
import joblib
import pandas as pd

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


def make_prediction(input_data, model_path=MODELS_DIR, scaler_path=SCALER_PATH, feature_path=FEATURE_PATH):
    """
    Make a prediction using the trained model and provided input features.

    Parameters:
        input_data (dict): Input features as a dictionary.
        model_path (str): Path to the trained model.
        scaler_path (str): Path to the scaler.
        feature_path (str): Path to the selected feature names.

    Returns:
        dict: Prediction result including class and probabilities.
    """

    try:
        # Load model, scaler, and selected features
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selected_features = joblib.load(feature_path)

        # Sort features alphabetically
        selected_features = sorted(selected_features)

        # Validate and align input features
        filtered_features = {key: input_data.get(key, 0.0) for key in selected_features}
        features_df = pd.DataFrame([filtered_features], columns=selected_features)

        # Scale and predict
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]

        # Format the response
        return {
            "prediction": int(prediction),
            "probabilities": {
                "benign": float(prediction_proba[0]),
                "malignant": float(prediction_proba[1])
            }
        }

    except Exception as e:
        return {"error": str(e)}
