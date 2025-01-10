from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from loguru import logger


def evaluate_and_tune_models(model_configs, X_train, y_train, X_val, y_val):
    results = {}
    for model_name, config in model_configs.items():
        logger.info(f"Tuning {model_name}...")
        grid_search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=50,
            cv=5,
            scoring="f1",
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_val_pred = best_model.predict(X_val)

        results[model_name] = {
            "best_model": best_model,
            "best_params": grid_search.best_params_,
            "validation_metrics": {
                "Accuracy": accuracy_score(y_val, y_val_pred),
                "Precision": precision_score(y_val, y_val_pred),
                "Recall": recall_score(y_val, y_val_pred),
                "F1-Score": f1_score(y_val, y_val_pred)
            }
        }
        logger.info(f"Best {model_name} parameters: {grid_search.best_params_}")
    return results
