# Kicktipp Predictor [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A sophisticated football match prediction model and web application designed to forecast match outcomes and scorelines, primarily for platforms like Kicktipp.

## Architecture Overview

The Kicktipp Predictor employs a robust two-stage, predictor-selector architecture to achieve high accuracy and interpretability in football match forecasting.

### Overview

1.  **Outcome Prediction (Selector):** An XGBoost classifier is utilized to predict the fundamental match result: Home Win, Draw, or Away Win.
2.  **Scoreline Selection (Predictor):** Two XGBoost regressors estimate the expected goals for each team. These expected goal values (lambda) are then fed into a Poisson distribution to determine the most probable scoreline for the match.

### Rationale

This architectural choice balances predictive accuracy with model interpretability. By separating outcome prediction from scoreline selection, specialized models can be trained for each task. The Poisson distribution provides a probabilistic framework well-suited for modeling goal counts.

### Feature Engineering

The project incorporates a comprehensive feature engineering pipeline that generates rich predictive signals from historical match data. Key features include:

-   **Elo Ratings**: Tamed Elo ratings are used to control for bias and incorporate uncertainty, providing a dynamic measure of team strength.
-   **Form Metrics**: Various performance indicators such as points, wins, draws, losses, goals scored, and goals conceded are calculated over different historical windows (e.g., last 3, 5, 10 matches).
-   **Strength of Schedule**: Opponent rank is used to weight prior points, offering a more nuanced assessment of team strength against varying competition.
-   **Venue-Specific Deltas**: Features capturing the comparative strength of teams when playing at home versus away are included to account for home-field advantage.

### Model Training

Both the outcome classifier and the goal regressors are trained using XGBoost, a powerful gradient boosting framework. Training methodologies include:

-   **Time-Decay Weighting**: More recent matches are given higher weight during training to ensure the model is sensitive to current team form and trends.
-   **Class-Balanced Weights**: The outcome classifier uses class-balanced weights, with an additional boost for the 'Draw' class to mitigate its underrepresentation in football results.
-   **Cross-Validation**: Time-series cross-validation is employed for hyperparameter tuning, ensuring robust performance on sequential data characteristic of sports seasons.

### Prediction and Scoreline Selection

The prediction process involves several steps:

1.  **Outcome Probabilities**: The outcome classifier generates initial probabilities for Home Win, Draw, and Away Win. These can be blended with Poisson-derived probabilities using a configurable hybrid weight.
2.  **Expected Goals**: Home and away goal regressors predict the expected number of goals for each team.
3.  **Scoreline Selection**: A Poisson grid is used to select the most probable scoreline based on predicted outcomes and expected goals. Optionally, an Expected Points (EP) maximizing approach can be used to select the scoreline that yields the highest fantasy football points.

### Evaluation

Model performance is rigorously evaluated using a dynamic, expanding-window procedure. The model is retrained every N matchdays on all historical data up to that point and then evaluated on the finished matches of the current season. Key metrics tracked include:

-   **Brier Score**: Measures the accuracy of probabilistic predictions.
-   **Log Loss**: Penalizes inaccurate and confident predictions.
-   **Ranked Probability Score (RPS)**: A proper scoring rule that considers the distance between predicted and actual outcomes.
-   **Accuracy**: Overall prediction accuracy.
-   **Kicktipp Points**: Simulated fantasy football points earned.
-   **Expected Calibration Error (ECE)**: Assesses how well predicted probabilities align with observed frequencies.

All evaluation results are output to the console and saved as concise JSON/CSV artifacts.

## Features

The Kicktipp Predictor offers both a command-line interface (CLI) and a web application for interacting with the prediction model.

### CLI Features

-   **`train`**: Trains the match predictor on historical data.
    -   Configurable number of past seasons for training.
-   **`predict`**: Generates predictions for upcoming matches.
    -   Predicts for a specified number of days ahead or a specific matchday.
    -   Options for probability source (classifier, Poisson, hybrid) and Poisson parameters.
    -   Supports parallel processing for scoreline selection.
-   **`evaluate`**: Evaluates model performance across the current season using expanding-window retraining.
    -   Configurable retraining frequency and probability source options.
-   **`web`**: Starts the Flask web application.
    -   Configurable host and port.
-   **`tune`**: Runs Optuna hyperparameter tuning with selectable objectives.
    -   Supports multi-worker parallel tuning with shared storage.
    -   Various objectives (e.g., PPG, logloss, brier, accuracy).
    -   Pruning strategies (median, hyperband).
-   **`shap`**: Performs SHAP (SHapley Additive exPlanations) analysis on trained models to explain feature importance.
    -   Configurable number of seasons and samples for analysis.

### Web Application Features

-   **Home Page (`/`)**: Displays upcoming match predictions.
-   **API Endpoints**:
    -   **`/api/upcoming_predictions`**: Fetches predictions for upcoming matches (configurable days ahead).
    -   **`/api/current_matchday`**: Retrieves predictions for the current matchday.
    -   **`/api/table`**: Provides the current league table, including form and EWMA points.
    -   **`/api/model_quality`**: Returns model quality metrics (Brier Score, Log Loss, Accuracy, etc.) from the latest evaluation.
    -   **`/api/match/<match_id>`**: Displays detailed prediction and feature information for a specific match.
-   **Frontend Pages**:
    -   **`/statistics`**: Dedicated page for model statistics.
    -   **`/table`**: Dedicated page for the league table.
    -   **`/match/<match_id>`**: Detailed view for individual match predictions.

## Installation

To set up the Kicktipp Predictor locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/kicktipp-predictor.git
    cd kicktipp-predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    If you plan to run SHAP analysis or advanced tuning, install with extras:
    ```bash
    pip install -e ".[dev,shap,tune]"
    ```

4.  **Environment Variables (Optional):**
    The application might use environment variables for API keys or database connections. Check `src/kicktipp_predictor/config/config.py` for details.
    Example:
    ```bash
    export KICKTIPP_API_KEY="your_api_key_here"
    ```

## Configuration

The project uses YAML files for configuration, located in the `src/kicktipp_predictor/config/` directory.

-   `all_features.yaml`: Defines all available features for the model.
-   `best_params_baseline.yaml`: Stores the best hyperparameters found during tuning for the baseline model.
-   `best_params_ppg.yaml`: Stores the best hyperparameters found during tuning for the PPG objective.
-   `config.py`: Python module for loading and managing configurations.
-   `kept_features.yaml`: Specifies features to be kept.
-   `minimal_features.yaml`: Defines a minimal set of features.

You can modify these files to adjust model parameters, feature sets, and other application settings.

## Usage

### Command-Line Interface (CLI)

All CLI commands are accessed via `kicktipp-predictor`.

-   **Train the model:**
    ```bash
    kicktipp-predictor train --seasons-back 5
    ```
    This will train the model using data from the last 5 seasons.

-   **Make predictions for the next 7 days:**
    ```bash
    kicktipp-predictor predict --days 7
    ```

-   **Evaluate model performance:**
    ```bash
    kicktipp-predictor evaluate --retrain-every 1
    ```
    This evaluates the model, retraining it every matchday.

-   **Run hyperparameter tuning:**
    ```bash
    kicktipp-predictor tune --n-trials 100 --objective ppg --workers 4 --storage "sqlite:///optuna_study.db"
    ```
    This runs 100 Optuna trials for the PPG objective using 4 workers and stores results in `optuna_study.db`.

-   **Run SHAP analysis:**
    ```bash
    kicktipp-predictor shap --seasons-back 3 --sample 5000
    ```
    This performs SHAP analysis using data from the last 3 seasons and 5000 samples.

### Web Application

To start the web application:

```bash
kicktipp-predictor web --host 0.0.0.0 --port 8000
```
Then, open your browser and navigate to `http://localhost:8000`.

#### Screenshots/GIFs

*(Placeholder for screenshots/GIFs demonstrating the web application's home page, league table, and match detail views.)*

## Development

### Contribution Guidelines

(Placeholder for detailed contribution guidelines, e.g., coding standards, pull request process.)

### Testing Procedures

Tests are located in the `tests/` directory. To run tests:

```bash
pytest
```

### Build/Deployment Processes

(Placeholder for build and deployment instructions, e.g., Docker, CI/CD.)

## License

This project is licensed under the [MIT License](LICENSE).

---
**Note:** This README is a living document. For the most up-to-date information, please refer to the source code and project documentation.
