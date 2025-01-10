from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from loguru import logger


def evaluate_and_tune_models(model_configs, X_train, y_train, X_val, y_val):
    results = {}
    for model_name, config in model_configs.items():
        logger.info(f"Tuning {model_name} with cross-validation...")

        # Define a scorer for F1-Score to standardize across models
        f1_scorer = make_scorer(f1_score)

        # Perform randomized search with cross-validation
        grid_search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=50,
            cv=5,  # 5-fold cross-validation
            scoring=f1_scorer,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        grid_search.fit(X_train, y_train)

        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Cross-validate the selected model on the training data
        cross_val_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        logger.info(f"Cross-validated F1-Score for {model_name}: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")

        # Evaluate the best model on the validation data
        y_val_pred = best_model.predict(X_val)
        validation_metrics = {
            "Accuracy": accuracy_score(y_val, y_val_pred),
            "Precision": precision_score(y_val, y_val_pred),
            "Recall": recall_score(y_val, y_val_pred),
            "F1-Score": f1_score(y_val, y_val_pred)
        }

        # Save results
        results[model_name] = {
            "best_model": best_model,
            "best_params": best_params,
            "cross_val_metrics": {
                "F1-Score Mean": cross_val_scores.mean(),
                "F1-Score Std": cross_val_scores.std()
            },
            "validation_metrics": validation_metrics
        }

        logger.info(f"Best {model_name} parameters: {best_params}")
        logger.info(f"Validation Metrics for {model_name}: {validation_metrics}")
    return results
