import pytest
from flask import Flask
import main  # Import your Flask app

@pytest.fixture
def client():
    """
    Fixture to create a test client for the Flask app.
    """
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """
    Test the /health endpoint.
    """
    response = client.get('/health')
    assert response.status_code == 200, "Health check endpoint should return status 200."
    assert response.get_json() == {"status": "healthy"}, "Health check response should indicate 'healthy'."

def test_predict_valid(client):
    """
    Test the /predict endpoint with valid input.
    """
    # Mock valid input features
    valid_input = {
        "features": {
            "radius_mean": 17.99,
            "texture_mean": 10.38,
            "perimeter_mean": 122.8,
            "area_mean": 1001.0,
            "smoothness_mean": 0.1184,
            "compactness_mean": 0.2776,
            "concavity_mean": 0.3001,
            "concave points_mean": 0.1471,
            "symmetry_mean": 0.2419,
            "fractal_dimension_mean": 0.07871,
            "radius_se": 1.095,
            "texture_se": 0.9053,
            "perimeter_se": 8.589,
            "area_se": 153.4,
            "smoothness_se": 0.006399,
            "compactness_se": 0.04904,
            "concavity_se": 0.05373,
            "concave points_se": 0.01587,
            "symmetry_se": 0.03003,
            "fractal_dimension_se": 0.006193,
            "radius_worst": 25.38,
            "texture_worst": 17.33
        }
    }

    response = client.post('/predict', json=valid_input)
    assert response.status_code == 200, "Predict endpoint should return status 200 for valid input."

    # Check the response contains expected keys
    data = response.get_json()
    assert "prediction" in data, "Response should contain 'prediction'."
    assert "probabilities" in data, "Response should contain 'probabilities'."
    assert "benign" in data["probabilities"] and "malignant" in data["probabilities"], \
        "Probabilities should include 'benign' and 'malignant'."

def test_predict_invalid(client):
    """
    Test the /predict endpoint with invalid input.
    """
    # Mock invalid input (missing features)
    invalid_input = {"invalid_key": {}}

    response = client.post('/predict', json=invalid_input)
    assert response.status_code == 400, "Predict endpoint should return status 400 for invalid input."

    # Check the error message in the response
    data = response.get_json()
    assert "error" in data, "Response should contain 'error'."
    assert data["error"] == "Invalid input. 'features' key is missing.", \
        "Error message should indicate missing 'features' key."

def test_predict_missing_features(client):
    """
    Test the /predict endpoint with missing required features.
    """
    # Mock input missing some features
    missing_features_input = {
        "features": {
            "radius_mean": 17.99,
            "texture_mean": 10.38
            # Missing other required features
        }
    }

    response = client.post('/predict', json=missing_features_input)
    assert response.status_code == 400, "Predict endpoint should return status 400 for missing features."

    # Check the response contains details about missing features
    data = response.get_json()
    assert "error" in data, "Response should contain 'error'."
    assert "missing_features" in data, "Response should include missing features list."
    assert len(data["missing_features"]) > 0, "There should be at least one missing feature in the response."
