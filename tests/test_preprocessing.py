import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preprocessing import load_and_preprocess_data, resample_data


@pytest.fixture
def mock_data():
    """
    Fixture to create mock breast cancer data for testing.
    """
    data = {
        "id": [1, 2, 3, 4],
        "diagnosis": ["M", "B", "B", "M"],
        "radius_mean": [17.99, 13.54, 12.45, 20.12],
        "texture_mean": [10.38, 14.36, 15.78, 19.82],
        "perimeter_mean": [122.8, 88.4, 84.1, 132.4],
        "area_mean": [1001.0, 521.0, 458.0, 1458.0],
        "compactness_mean": [0.2776, 0.1599, 0.1235, 0.3456],
        "symmetry_mean": [0.2419, 0.2239, 0.2138, 0.3254],
    }
    return pd.DataFrame(data)

def test_load_and_preprocess_data(mock_data, monkeypatch):
    """
    Test the data loading and preprocessing function.
    """
    # Mock the `pd.read_csv` function
    def mock_read_csv(filepath, *args, **kwargs):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    # Call the function
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_data("mock_path.csv")

    # Assert outputs
    assert X_train.shape[1] == len(feature_names), "Feature names count should match number of columns in X_train"
    assert X_train.shape[0] > 0, "X_train should not be empty"
    assert y_train.nunique() == 2, "y_train should have two classes"
    assert isinstance(feature_names, list), "Feature names should be a list"
    assert all(isinstance(f, str) for f in feature_names), "Feature names should be strings"

def test_resample_data():
    """
    Test the data resampling function.
    """
    # Mock training data
    X_train = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [3.5, 1.5]])
    y_train = np.array([0, 1, 1, 0])

    # Call the function
    X_resampled, y_resampled = resample_data(X_train, y_train)

    # Assert outputs
    assert X_resampled.shape[0] == y_resampled.shape[0], "X_resampled and y_resampled should have the same number of rows"
    assert y_resampled.nunique() == 2, "Resampled data should still have two classes"
    assert X_resampled.shape[1] == X_train.shape[1], "Number of features should remain the same"

def test_scaling():
    """
    Test scaling with StandardScaler.
    """
    scaler = StandardScaler()
    X_train = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    X_train_scaled = scaler.fit_transform(X_train)

    # Assert scaling
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0), "Mean of scaled features should be approximately 0"
    assert np.allclose(np.std(X_train_scaled, axis=0), 1), "Standard deviation of scaled features should be approximately 1"
