from flask import Flask, request, jsonify
import joblib
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

        # Extract raw features from input
        input_features = data["features"]

        # Add missing features with default values (0.0)
        for feature in selected_feature_names:
            if feature not in input_features:
                input_features[feature] = 0.0

        # Convert input features to a DataFrame
        all_features_df = pd.DataFrame([input_features])

        # Scale the features
        try:
            scaled_features = scaler.transform(all_features_df)
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            return jsonify({"error": "Error during feature scaling", "details": str(e)}), 500

        # Convert scaled features to a DataFrame
        scaled_features_df = pd.DataFrame(scaled_features, columns=all_features_df.columns)

        # Filter scaled features based on the selected feature names
        filtered_features_df = scaled_features_df[selected_feature_names]

        # Make prediction
        try:
            prediction = model.predict(filtered_features_df)[0]
            prediction_proba = model.predict_proba(filtered_features_df)[0]
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            return jsonify({"error": "Error during model prediction", "details": str(e)}), 500

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
