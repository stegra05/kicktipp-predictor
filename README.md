# Kicktipp Predictor

A Python-based tool for predicting football match outcomes, with a focus on the German football league system. It provides a command-line interface (CLI) for training models, making predictions, and evaluating performance, as well as a Flask web application for serving the predictions.

## Features

*   **End-to-End Machine Learning Pipeline:** From data fetching and feature engineering to model training, evaluation, and prediction.
*   **Command-Line Interface:** A powerful CLI built with Typer for easy interaction with the prediction pipeline.
*   **Web Application:** A Flask-based web interface to view predictions and model status.
*   **Automated Data Fetching:** Fetches historical and upcoming match data from the OpenLigaDB API.
*   **Advanced Feature Engineering:** Creates a rich set of features, including team form, historical performance, and Elo ratings.
*   **XGBoost Model:** Uses an XGBoost Regressor to predict the goal difference between two teams.
*   **Probabilistic Predictions:** Derives home win, draw, and away win probabilities from the predicted goal difference.
*   **Comprehensive Evaluation:** Includes a dynamic, expanding-window evaluation process to simulate a realistic prediction scenario over a season.
*   **Hyperparameter Tuning:** Integrated with Optuna for automated hyperparameter tuning to find the best model configuration.
*   **Code Quality and Testing:** Adheres to modern Python development standards with linting, formatting, and a suite of tests.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/kicktipp-predictor.git
    cd kicktipp-predictor
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the project in editable mode:**

    ```bash
    pip install -e .
    ```

4.  **Install optional dependencies for tuning and analysis:**

    ```bash
    pip install -e .[tuning,plots]
    ```

## Usage

The primary way to interact with the Kicktipp Predictor is through its CLI.

### Training the Model

To train the prediction model on historical data, run the `train` command:

```bash
kicktipp-predictor train
```

You can specify the number of past seasons to use for training with the `--seasons-back` option:

```bash
kicktipp-predictor train --seasons-back 5
```

### Making Predictions

To predict outcomes for upcoming matches, use the `predict` command:

```bash
kicktipp-predictor predict
```

By default, it predicts matches for the next 7 days. You can change this with the `--days` option:

```bash
kicktipp-predictor predict --days 14
```

You can also predict matches for a specific matchday:

```bash
kicktipp-predictor predict --matchday 10
```

### Evaluating the Model

To evaluate the model's performance over a season, use the `evaluate` command:

```bash
kicktipp-predictor evaluate
```

This command performs a dynamic, expanding-window evaluation, which means it retrains the model periodically as the season progresses to simulate a real-world prediction scenario. You can control the retraining frequency with the `--retrain-every` option.

### Tuning Hyperparameters

To find the best hyperparameters for the model, use the `tune` command:

```bash
kicktipp-predictor tune
```

This will run an Optuna study to optimize the model's parameters. The results will be saved in `src/kicktipp_predictor/config/best_params.yaml`.

### Web Application

The project also includes a Flask web application to display predictions. To run the web server, use the `web` command:

```bash
kicktipp-predictor web
```

The application will be available at `http://127.0.0.1:8000`.

## Architecture

The project is structured as a Python package with the main source code in the `src/kicktipp_predictor` directory.

### Data Pipeline

The data pipeline is managed by the `DataLoader` class in `src/kicktipp_predictor/data.py`. It is responsible for:

1.  **Fetching Data:** It fetches match data from the OpenLigaDB API and caches it locally.
2.  **Feature Engineering:** It creates a wide range of features for the model, including:
    *   **Team Form:** Rolling averages of goals scored, goals conceded, and points.
    *   **Weighted Form:** Form metrics weighted by the opponent's rank.
    *   **Elo Ratings:** A dynamic Elo rating system that is updated after each match.
    *   **Historical Performance:** Averages of goals, points, and other metrics over multiple seasons.

### Model

The prediction model is a `GoalDifferencePredictor` class in `src/kicktipp_predictor/predictor.py`. It uses an XGBoost Regressor to predict the goal difference in a match. From the predicted goal difference, it derives the probabilities for a home win, draw, and away win using a normal distribution.

### Evaluation

The evaluation process, located in `src/kicktipp_predictor/evaluate.py`, is designed to provide a realistic assessment of the model's performance. The `run_season_dynamic_evaluation` function simulates a full season, retraining the model at regular intervals and making predictions on upcoming matches. This provides a more robust evaluation than a simple train-test split.

## Development

### Running Tests

The project uses `pytest` for testing. To run the tests, execute:

```bash
pytest
```

### Code Style and Linting

The project uses `ruff` for code formatting and linting. The configuration is in `pyproject.toml`. To check the code style, run:

```bash
ruff check .
```

To format the code, run:

```bash
ruff format .
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically check and format the code before each commit. To install the hooks, run:

```bash
pre-commit install
```

## Future Work

*   **More Advanced Models:** Experiment with different model architectures, such as Poisson or Negative Binomial models, which are often used for modeling football scores.
*   **More Features:** Incorporate additional data sources, such as player statistics, team news, or betting odds.
*   **Improved Web Interface:** Enhance the web interface with more detailed statistics, visualizations, and user accounts.
*   **Cloud Deployment:** Deploy the application to a cloud platform to make it publicly accessible.
