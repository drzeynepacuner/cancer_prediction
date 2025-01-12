# Cancer Prediction Using Machine Learning

This repository provides a machine learning solution for predicting breast cancer based on digitized images obtained from fine needle aspirates (FNA) of breast masses. The project aims to distinguish between benign and malignant tumors, facilitating early detection and treatment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Run with Docker](#run-with-docker)
- [Prediction Request](#prediction-request)
- [Contributing](#contributing)
- [License](#license)

## Overview

Breast cancer is one of the most common cancers affecting women worldwide. This project leverages machine learning to analyze cellular features from FNA-derived images to predict whether a tumor is benign or malignant.

The predictive model is trained using a dataset of breast cancer cases and uses feature engineering and advanced classification techniques to achieve high accuracy.

## Features

- **Data Preprocessing**: Handles missing values, scales features, and splits the data into training/testing sets.
- **Model Training**: Uses Scikit-learn classifiers (e.g., Random Forest, SVM) to classify tumors as benign or malignant.
- **Performance Evaluation**: Reports accuracy, precision, recall, and F1-score.
- **Interactive API**: Integrates a Flask-based API for real-time predictions.
- **Visualization Tools**: Provides scripts for visualizing feature distributions and model performance.

## Project Structure

```
cancer_prediction/
│
├── main.py              # Main code that launches the app.
├── data/                # Dataset files (training and testing data)
├── scripts/             # Scripts for data processing, model training, and inference
├── src/                 # Core source code for the project
├── tests/               # Unit and integration tests
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

## Requirements

The project requires the following dependencies:

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Flask (for API functionality)
- Jupyter Notebook (for exploratory data analysis)
- Loguru (for logging)

All dependencies can be installed using the `requirements.txt` file.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/drzeynepacuner/cancer_prediction.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd cancer_prediction
   ```

3. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Model Training Script**
   

   You might need to set your path for Python first.

   ```bash
   export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
   ```

   ```bash
   python scripts/train.py
   ```

   This will preprocess the data, train the model, and save the trained model to the `models/` directory.

2. **Inference**

   ```bash
   python scripts/inference.py
   ```

   This script will load the trained model and make predictions on new or test data. The results will be saved or displayed based on the configuration.

3. **Optional: Run the API for Real-Time Predictions**

   ```bash
   python api/app.py
   ```

   Access the API at `http://localhost:5000`.

4. **Jupyter Notebook**

   For exploratory data analysis, open the provided Jupyter notebooks in the `notebooks/` directory:

   ```bash
   jupyter notebook
   ```

## Run with Docker

You can containerize the project using Docker to simplify setup and ensure consistent environments.

1. **Build the Docker Image**

   Navigate to the project directory and build the Docker image:

   ```bash
   docker build -t cancer_prediction .
   ```

2. **Run the Docker Container**

   Run the container, exposing port `5000` for the API:

   ```bash
   docker run -p 5000:5000 cancer_prediction
   ```

3. **Access the API**

   After running the container, the API will be accessible at:

   ```
   http://localhost:5000
   ```

   You can test the API endpoints (e.g., `/predict`) using tools like `curl` or Postman.

4. **Stopping the Container**

   To stop the running container, locate the container ID using:

   ```bash
   docker ps
   ```

   Then stop the container:

   ```bash
   docker stop <container_id>
   ```

### Notes:
- Ensure that Docker is installed and running on your system before proceeding.
- If additional dependencies or configurations are required for the Docker image, update the `Dockerfile` accordingly.

## Prediction Request

The API provides a `/predict` endpoint for making real-time predictions using the trained model. Here's how to send a request:

### Endpoint

```
POST /predict
```

### Request Format

The request payload must include a `features` key, containing a dictionary of feature names and their corresponding values. Missing features will be automatically set to `0.0`.

### Required Features

The following features are expected in the input JSON:

- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`
- `radius_se`
- `texture_se`
- `perimeter_se`
- `area_se`
- `smoothness_se`
- `compactness_se`
- `concavity_se`
- `concave points_se`
- `symmetry_se`
- `fractal_dimension_se`
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

### Example Request

```json
{
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
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

### Example CURL Request

You can use `curl` to test the `/predict` endpoint. Features need to be alphabetically listed. Here’s an example:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "features": {
    "area_mean": 1001.0,
    "area_se": 153.4,
    "area_worst": 2019.0,
    "compactness_mean": 0.2776,
    "compactness_se": 0.04904,
    "compactness_worst": 0.6656,
    "concave points_mean": 0.1471,
    "concave points_se": 0.01587,
    "concave points_worst": 0.2654,
    "concavity_mean": 0.3001,
    "concavity_se": 0.05373,
    "concavity_worst": 0.7119,
    "fractal_dimension_mean": 0.07871,
    "fractal_dimension_se": 0.006193,
    "fractal_dimension_worst": 0.1189,
    "perimeter_mean": 122.8,
    "perimeter_se": 8.589,
    "perimeter_worst": 184.6,
    "radius_mean": 17.99,
    "radius_se": 1.095,
    "radius_worst": 25.38,
    "smoothness_mean": 0.1184,
    "smoothness_se": 0.006399,
    "smoothness_worst": 0.1622,
    "symmetry_mean": 0.2419,
    "symmetry_se": 0.03003,
    "symmetry_worst": 0.4601,
    "texture_mean": 10.38,
    "texture_se": 0.9053,
    "texture_worst": 17.33
  }
}'
```

### Response Format

The response includes the predicted class (`benign` or `malignant`) and the associated probabilities.

### Example Response

```json
{
  "prediction": 1,
  "probabilities": {
    "benign": 0.12,
    "malignant": 0.88
  }
}
```

### Error Handling

- If the `features` key is missing or improperly formatted:

  ```json
  {
    "error": "Invalid input. 'features' key is missing."
  }
  ```

- If there is an issue during feature scaling or prediction:

  ```json
  {
    "error": "Error during feature scaling",
    "details": "Detailed error message here."
  }
  ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](