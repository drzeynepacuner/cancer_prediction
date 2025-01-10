import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from loguru import logger


def select_top_features(X, y, feature_names, top_percent=0.75):
    logger.info(f"Selecting top {top_percent * 100}% features...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [(feature_names[i], importances[i]) for i in sorted_indices]

    top_n = int(len(feature_names) * top_percent)
    selected_features = sorted([name for name, _ in sorted_features[:top_n]])

    joblib.dump(selected_features, "data/models/selected_feature_names.pkl")
    logger.info(f"Selected features saved to models/selected_feature_names.pkl")
    return X[:, [feature_names.index(f) for f in selected_features]], selected_features
