# Architecture

This document formally defines the current "winning" architecture of the Football Prediction Model.

## Overview

The model follows a two-stage, predictor-selector architecture:

1.  **Outcome Prediction (Selector):** An XGBoost classifier predicts the match result (Home Win, Draw, or Away Win).
2.  **Scoreline Selection (Predictor):** Two XGBoost regressors estimate the expected goals for each team. These lambda values are then used with a Poisson distribution to find the most probable scoreline.

## Rationale

This architecture was chosen for its balance of accuracy and interpretability. The two-stage process allows for specialized models to be trained for each sub-problem, and the use of a Poisson distribution for scoreline selection provides a probabilistic framework that is well-suited to the task.

## Feature Engineering

The feature engineering pipeline is designed to create a rich set of predictive signals from historical match data. Key aspects include:

-   **Elo Ratings**: Tamed Elo ratings are used to control for bias and incorporate uncertainty.
-   **Form Metrics**: Various form metrics are calculated, including points, wins, draws, losses, goals scored, and goals conceded over different windows (e.g., last 3, 5, 10 matches).
-   **Strength of Schedule**: Opponent rank is used to weight prior points, providing a more nuanced measure of team strength.
-   **Venue-Specific Deltas**: Features capturing the comparative strength of teams at home versus away are included.

## Model Training

Both the outcome classifier and the goal regressors are trained using XGBoost. Training incorporates several techniques to enhance model performance and robustness:

-   **Time-Decay Weighting**: More recent matches are given higher weight during training to capture current team form.
-   **Class-Balanced Weights**: The outcome classifier uses class-balanced weights, with an additional boost for the 'Draw' class to address its underrepresentation.
-   **Cross-Validation**: Time-series cross-validation is used for hyperparameter tuning to ensure robust performance on sequential data.

## Prediction and Scoreline Selection

During prediction, the following steps are executed:

1.  **Outcome Probabilities**: The outcome classifier provides initial probabilities for Home Win, Draw, and Away Win. These probabilities can be blended with Poisson-derived probabilities using a configurable hybrid weight.
2.  **Expected Goals**: The home and away goal regressors predict the expected number of goals for each team.
3.  **Scoreline Selection**: Based on the predicted outcome and expected goals, a Poisson grid is used to select the most probable scoreline. Optionally, an Expected Points (EP) maximizing approach can be used to select the scoreline that yields the highest fantasy football points.

## Evaluation

Model performance is evaluated using a dynamic, expanding-window procedure. The model is retrained every N matchdays on all historical data up to that point, and then evaluated on the finished matches of the current season. Key metrics include:

-   **Brier Score**: Measures the accuracy of probabilistic predictions.
-   **Log Loss**: Penalizes inaccurate and confident predictions.
-   **Ranked Probability Score (RPS)**: A proper scoring rule that considers the distance between predicted and actual outcomes.
-   **Accuracy**: Overall prediction accuracy.
-   **Kicktipp Points**: Simulated fantasy football points earned.
-   **Expected Calibration Error (ECE)**: Assesses how well predicted probabilities align with observed frequencies.

All evaluation results are output to the console and saved as concise JSON/CSV artifacts, focusing on data rather than image plots.
