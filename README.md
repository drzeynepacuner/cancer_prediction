# Cancer Prediction Using Machine Learning

This repository provides a machine learning solution for predicting breast cancer based on digitized images obtained from fine needle aspirates (FNA) of breast masses. The project aims to distinguish between benign and malignant tumors, facilitating early detection and treatment.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

Breast cancer is one of the most common cancers affecting women worldwide. This project leverages machine learning to analyze cellular features from FNA-derived images to predict whether a tumor is benign or malignant.

The predictive model is trained using a dataset of breast cancer cases and uses feature engineering and advanced classification techniques to achieve high accuracy.

## Features

- **Data Preprocessing**: Cleans and prepares the dataset for training and evaluation.
- **Model Training**: Implements machine learning algorithms to classify tumor types.
- **Evaluation Metrics**: Provides metrics like accuracy, precision, recall, and F1-score to measure model performance.
- **Interactive API**: Offers endpoints for integration with external systems for real-time predictions (optional).

## Project Structure

```plaintext
cancer_prediction/
│
├── api/                 # Code for the API layer (if applicable)
├── data/                # Dataset files (training and testing data)
├── scripts/             # Scripts for data processing and model training
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
3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**  
   - Place the dataset files in the `data/` directory.  
   - Ensure the dataset is properly formatted as specified in the repository.

2. **Run the Model Training Script**  
   ```bash
   python scripts/train_model.py
   ```  
   This will preprocess the data, train the model, and save the trained model to the `models/` directory.

3. **Evaluate the Model**  
   ```bash
   python scripts/evaluate_model.py
   ```  
   This script will generate performance metrics and visualizations.

4. **Optional: Run the API for Real-Time Predictions**  
   ```bash
   python api/app.py
   ```  
   Access the API at `http://localhost:5000`.

5. **Jupyter Notebook**  
   For exploratory data analysis, open the provided Jupyter notebooks in the `notebooks/` directory:  
   ```bash
   jupyter notebook
   ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
