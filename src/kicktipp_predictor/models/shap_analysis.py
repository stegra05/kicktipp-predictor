from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .metrics import ensure_dir

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None  # type: ignore
    plt = None  # type: ignore


def run_shap_for_mlpredictor(ml_predictor, sample_X, out_dir: str = os.path.join('data', 'predictions', 'shap')) -> Optional[str]:
    """
    Compute SHAP summary plots for ML models if dependencies are available.
    Saves summary plots for result classifier and goal regressors.
    """
    if shap is None or plt is None:
        return None
    ensure_dir(out_dir)

    # Ensure sample size is reasonable
    X = sample_X.copy()
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)

    # Result classifier
    try:
        if getattr(ml_predictor, 'result_model', None) is not None:
            explainer = shap.TreeExplainer(ml_predictor.result_model)
            values = explainer.shap_values(X)
            # Multiclass: values is list
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_result_summary.png'))
            plt.close()
    except Exception:
        pass

    # Home goals regressor
    try:
        if getattr(ml_predictor, 'score_model_home', None) is not None:
            explainer = shap.TreeExplainer(ml_predictor.score_model_home)
            values = explainer.shap_values(X)
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_home_goals_summary.png'))
            plt.close()
    except Exception:
        pass

    # Away goals regressor
    try:
        if getattr(ml_predictor, 'score_model_away', None) is not None:
            explainer = shap.TreeExplainer(ml_predictor.score_model_away)
            values = explainer.shap_values(X)
            shap.summary_plot(values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_away_goals_summary.png'))
            plt.close()
    except Exception:
        pass

    return out_dir


