import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import joblib
from loguru import logger


def load_and_preprocess_data(file_path):
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    df.drop(columns=['Unnamed: 32'], inplace=True, errors='ignore')

    X = df.drop(columns=['id', 'diagnosis'])
    y = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode target
    X = X[sorted(X.columns)]  # Alphabetically sort features

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "data/models/scaler.pkl")  # Save the scaler
    logger.info("Scaler saved to models/scaler.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train.columns.tolist()


def resample_data(X_train, y_train):
    logger.info("Applying SMOTE + Tomek Links for resampling...")
    smote_tomek = SMOTETomek(random_state=42)
    return smote_tomek.fit_resample(X_train, y_train)
