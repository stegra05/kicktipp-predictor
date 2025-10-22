from __future__ import annotations

import os

from ..metrics import ensure_dir

# Optional dependencies for SHAP analysis. These are not required for the core
# prediction logic and are handled gracefully if not installed.
try:  # pragma: no cover
    import matplotlib.pyplot as plt
    import shap
except ImportError:  # pragma: no cover
    shap = None
    plt = None


def run_shap_for_predictor(
    predictor, sample_X, out_dir: str = os.path.join("data", "predictions", "shap")
) -> str | None:
    """Computes and saves SHAP summary plots for the predictor's models.

    This function generates SHAP summary plots for the outcome classifier and
    goal regressors if the `shap` and `matplotlib` libraries are installed.
    The plots are saved to the specified output directory.

    Args:
        predictor: The predictor instance containing trained models.
        sample_X: A sample of input features (pandas DataFrame) for the explainer.
        out_dir: The directory where the SHAP plots will be saved.

    Returns:
        The output directory path if SHAP plots were generated, otherwise None.
    """
    if shap is None or plt is None:
        print("SHAP or matplotlib not installed. Skipping SHAP analysis.")
        return None
    ensure_dir(out_dir)

    # Use a smaller sample for performance reasons if the input data is large.
    if len(sample_X) > 2000:
        X = sample_X.sample(2000, random_state=42)
    else:
        X = sample_X.copy()

    # The following blocks are wrapped in try-except to gracefully handle cases
    # where a model is not present in the predictor or if SHAP analysis fails.

    # Generate SHAP plot for the outcome classifier.
    try:
        model = getattr(predictor, "outcome_model", None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)

            # For multiclass classifiers, SHAP returns a list of arrays.
            # We generate a separate plot for each class.
            if isinstance(values, list):
                class_names = getattr(predictor.label_encoder, "classes_", [])
                for i, val in enumerate(values):
                    class_name = class_names[i] if i < len(class_names) else f"class_{i}"
                    plt.figure()
                    shap.summary_plot(val, X, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"shap_result_summary_{class_name}.png"))
                    plt.close()
            else:
                plt.figure()
                shap.summary_plot(values, X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap_result_summary.png"))
                plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot for outcome model: {e}")

    # Generate SHAP plot for the home goals regressor.
    try:
        model = getattr(predictor, "home_goals_model", None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            plt.figure()
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_home_goals_summary.png"))
            plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot for home goals model: {e}")

    # Generate SHAP plot for the away goals regressor.
    try:
        model = getattr(predictor, "away_goals_model", None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            plt.figure()
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_away_goals_summary.png"))
            plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot for away goals model: {e}")

    return out_dir


# Backward-compatible alias.
def run_shap_for_mlpredictor(
    ml_predictor, sample_X, out_dir: str = os.path.join("data", "predictions", "shap")
) -> str | None:
    """Alias for `run_shap_for_predictor` for backward compatibility."""
    return run_shap_for_predictor(ml_predictor, sample_X, out_dir)
