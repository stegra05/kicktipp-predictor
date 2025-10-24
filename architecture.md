# Architecture

This document formally defines the current "winning" architecture of the Football Prediction Model.

## Overview

The model follows a two-stage, predictor-selector architecture:

1.  **Outcome Prediction (Selector):** An XGBoost classifier predicts the match result (Home Win, Draw, or Away Win).
2.  **Scoreline Selection (Predictor):** Two XGBoost regressors estimate the expected goals for each team. These lambda values are then used with a Poisson distribution to find the most probable scoreline.

## Rationale

This architecture was chosen for its balance of accuracy and interpretability. The two-stage process allows for specialized models to be trained for each sub-problem, and the use of a Poisson distribution for scoreline selection provides a probabilistic framework that is well-suited to the task.
