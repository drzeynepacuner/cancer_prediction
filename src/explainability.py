import shap
import matplotlib.pyplot as plt
from loguru import logger


def explain_model(model, X_sample, feature_names, output_file=None):
    """
    Generate SHAP values and create a summary plot for explainability.

    Parameters:
        model (object): Trained model.
        X_sample (array): Sample feature matrix (e.g., X_test).
        feature_names (list): List of feature names.
        output_file (str): Path to save the SHAP summary plot. If None, plot is shown interactively.
    """
    try:
        logger.info("Generating SHAP explanations...")

        # Use SHAP's TreeExplainer (for tree-based models like XGBoost, Random Forest, etc.)
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Generate summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

        # Save or display the plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {output_file}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {str(e)}")
        raise
