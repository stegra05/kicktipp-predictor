"""Generates and saves SHAP summary plots for model interpretability.

This module uses the SHAP (SHapley Additive exPlanations) library to explain the
output of the trained XGBoost models. It creates summary plots that visualize
feature importance and their impact on predictions for the outcome classifier
and goal regressors.
"""

from __future__ import annotations

import os

from ..metrics import ensure_dir

try:
    import matplotlib.pyplot as plt
    import shap
except ImportError:
    shap = None
    plt = None


def run_shap_for_predictor(
    predictor, sample_X, out_dir: str = os.path.join("data", "predictions", "shap")
) -> str | None:
    """Computes and saves SHAP summary plots for the predictor's models.

    This function generates SHAP summary plots for the outcome classifier and the
    home/away goal regressors. The plots are saved to the specified directory.
    If SHAP or Matplotlib are not installed, the function will do nothing.

    Args:
        predictor: A trained `MatchPredictor` instance.
        sample_X: A DataFrame of sample data for which to compute SHAP values.
                  A smaller sample size (e.g., 2000 rows) is recommended for
                  performance reasons.
        out_dir: The directory where the SHAP plots will be saved.

    Returns:
        The output directory path if plots were generated, otherwise None.
    """
    if shap is None or plt is None:
        print("SHAP analysis skipped: `shap` and `matplotlib` are not installed.")
        return None

    ensure_dir(out_dir)
    X = sample_X.copy()
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)

    _generate_outcome_model_plots(predictor, X, out_dir)
    _generate_goal_model_plot(predictor.home_goals_model, X, out_dir, "home_goals")
    _generate_goal_model_plot(predictor.away_goals_model, X, out_dir, "away_goals")

    return out_dir


def _generate_outcome_model_plots(predictor, X, out_dir):
    """Generates SHAP plots for the multiclass outcome model."""
    if predictor.outcome_model is None:
        return

    try:
        explainer = shap.TreeExplainer(predictor.outcome_model)
        shap_values = explainer.shap_values(X)

        if not isinstance(shap_values, list):
            shap_values = [shap_values]

        class_names = predictor.label_encoder.classes_
        for i, class_shap_values in enumerate(shap_values):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            plt.figure()
            shap.summary_plot(class_shap_values, X, show=False)
            plt.title(f"SHAP Summary - Outcome: {class_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"shap_summary_outcome_{class_name}.png"))
            plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot for outcome model: {e}")


def _generate_goal_model_plot(model, X, out_dir, model_name):
    """Generates a SHAP summary plot for a single goal regression model."""
    if model is None:
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary - {model_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_summary_{model_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot for {model_name}: {e}")
