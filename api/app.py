from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import os

# Ensure the logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure loguru to save logs in the logs folder
logger.add(
    "logs/app.log",  # Save logs to logs/app.log
    rotation="1 MB",  # Rotate log file after it reaches 1 MB
    retention="10 days",  # Keep logs for 10 days
    compression="zip"  # Compress old logs
)

# Load model, scaler, and selected features
try:
    model = joblib.load("data/models/best_model.pkl")
    scaler = joblib.load("data/models/scaler.pkl")
    selected_feature_names = joblib.load("data/models/selected_feature_names.pkl")
    selected_feature_names = sorted(selected_feature_names)  # Alphabetically sort feature names
    logger.info("Model, scaler, and selected feature names loaded successfully.")
    logger.info(f"Alphabetically sorted features: {selected_feature_names}")
except Exception as e:
    logger.error(f"Error loading artifacts: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON payload with feature values and returns the prediction.
    """
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")

        if not data or "features" not in data:
            logger.error("Invalid input: 'features' key is missing.")
            return jsonify({"error": "Invalid input. 'features' key is missing."}), 400

        # Validate and filter features
        input_features = data["features"]
        filtered_features = {key: input_features[key] for key in selected_feature_names if key in input_features}

        # Check for missing features
        missing_features = [feature for feature in selected_feature_names if feature not in filtered_features]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return jsonify({"error": "Missing required features", "missing_features": missing_features}), 400

        # Add missing features with default values
        for feature in selected_feature_names:
            if feature not in filtered_features:
                filtered_features[feature] = 0.0

        # Convert to DataFrame with correct feature order
        features_df = pd.DataFrame([filtered_features], columns=selected_feature_names)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Response
        response = {
            "prediction": int(prediction),
            "probabilities": {
                "benign": float(prediction_proba[0]),
                "malignant": float(prediction_proba[1])
            }
        }
        logger.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
