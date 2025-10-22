from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..metrics import ensure_dir

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None  # type: ignore
    plt = None  # type: ignore


def run_shap_for_predictor(predictor, sample_X, out_dir: str = os.path.join('data', 'predictions', 'shap')) -> Optional[str]:
    """
    Compute SHAP summary plots for the trained XGBoost models if dependencies are available.
    Saves summary plots for outcome classifier and goal regressors.
    """
    if shap is None or plt is None:
        return None
    ensure_dir(out_dir)

    # Ensure sample size is reasonable
    X = sample_X.copy()
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)

    # Outcome classifier (multiclass): generate per-class summary plots if values is a list
    try:
        model = getattr(predictor, 'outcome_model', None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            # If multiclass, values is a list of arrays (one per class)
            if isinstance(values, list):
                class_names = getattr(getattr(predictor, 'label_encoder', None), 'classes_', None)
                for i, val in enumerate(values):
                    shap.summary_plot(val, X, show=False)
                    plt.tight_layout()
                    name = None
                    try:
                        if class_names is not None and i < len(class_names):
                            name = str(class_names[i])
                    except Exception:
                        name = None
                    fname = f"shap_result_summary_class_{name or i}.png"
                    plt.savefig(os.path.join(out_dir, fname))
                    plt.close()
            else:
                shap.summary_plot(values, X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'shap_result_summary.png'))
                plt.close()
    except Exception:
        pass

    # Home goals regressor
    try:
        model = getattr(predictor, 'home_goals_model', None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_home_goals_summary.png'))
            plt.close()
    except Exception:
        pass

    # Away goals regressor
    try:
        model = getattr(predictor, 'away_goals_model', None)
        if model is not None:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_away_goals_summary.png'))
            plt.close()
    except Exception:
        pass

    return out_dir

# Backward-compatible alias
def run_shap_for_mlpredictor(ml_predictor, sample_X, out_dir: str = os.path.join('data', 'predictions', 'shap')) -> Optional[str]:
    return run_shap_for_predictor(ml_predictor, sample_X, out_dir)


