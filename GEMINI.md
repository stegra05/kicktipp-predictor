# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand and effectively assist with the **3. Liga Match Predictor** project.

## Project Overview

This is a Python-based machine learning project designed to predict the outcomes of football matches in Germany's 3. Bundesliga. The primary goal is to maximize points in a fantasy football league like Kicktipp.

The architecture follows a two-step "Predictor-Selector" model:
1.  **Selector**: An XGBoost classifier predicts the match outcome (Home Win, Draw, or Away Win).
2.  **Predictor**: XGBoost regressors predict the number of goals for each team, and a Poisson grid search selects the most likely scoreline that matches the predicted outcome.

The project includes a comprehensive feature engineering pipeline, a command-line interface (CLI) for all major operations, and a Flask-based web interface for viewing predictions.

### Key Technologies

*   **Backend**: Python 3.10+
*   **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
*   **CLI**: Typer
*   **Web**: Flask
*   **Hyperparameter Tuning**: Optuna
*   **Model Analysis**: SHAP
*   **Linting/Formatting**: Ruff

## Building and Running

The project is packaged with `setuptools` and can be installed in editable mode.

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kicktipp-predictor

# Install the project and its core dependencies
pip install -e .

# Install optional dependencies for tuning and plotting
pip install -e .[tuning,plots]
```

### Key Commands

The main entry point is the `kicktipp-predictor` CLI script.

*   **Train the models:**
    ```bash
    kicktipp_predictor train --seasons-back 5
    ```

*   **Generate predictions for the next 7 days:**
    ```bash
    kicktipp_predictor predict --days 7
    ```

*   **Evaluate model performance on the current season:**
    ```bash
    kicktipp_predictor evaluate
    ```

*   **Run the web interface:**
    ```bash
    kicktipp_predictor web --host 127.0.0.1 --port 8000
    ```

*   **Run hyperparameter tuning:**
    ```bash
    kicktipp_predictor tune --n-trials 100 --workers 4 --storage "sqlite:///data/kicktipp_study.db?timeout=60"
    ```

*   **Run tests:**
    ```bash
    pytest
    ```

## Development Conventions

*   **Code Style**: The project uses `ruff` for linting and formatting, with a line length of 88 characters. Configuration is in `pyproject.toml`.
*   **Typing**: The project uses type hints, but they are not strictly enforced at this time (as seen in `mypy` configuration).
*   **Configuration**: All model and application configuration is centralized in `.yaml` files within the `config/` directory. The main configuration is loaded from `config/best_params.yaml`.
*   **Modularity**: The code is organized into distinct modules for data handling (`data.py`), prediction (`predictor.py`), evaluation (`evaluate.py`), and the web app (`web/app.py`).
*   **CLI**: All user-facing actions are exposed via the `kicktipp-predictor` command, defined in `src/kicktipp_predictor/cli.py`.
