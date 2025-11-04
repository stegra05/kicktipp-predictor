# Kicktipp Predictor V3

**A modern, data-driven approach to predicting football match outcomes using ordinal regression.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project provides a complete framework for predicting football match outcomes, built around a powerful and elegant **goal difference regression** model. It moves beyond traditional classification approaches to better capture the ordinal nature of football results (Win, Draw, Loss).

The system is architected for robustness and ease of use, featuring:
- **A single, powerful `XGBoost` model** at its core.
- **A "Probabilistic Bridge"** to translate regression output into accurate H/D/A probabilities.
- **A flexible CLI** for training, prediction, and evaluation.
- **Comprehensive evaluation metrics** to assess model performance.

## Project Status

This project is a functional and well-documented proof-of-concept. The V3 architecture represents a significant improvement over previous versions, delivering more accurate and reliable predictions.

## Installation

To get started, clone the repository and install the dependencies using a virtual environment:

```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and generating plots, install the `dev` and `plots` extras:
```bash
pip install -e '.[dev,plots]'
```

## Usage

The project is controlled via a command-line interface.

### Training the Model

To train the model on the latest data:
```bash
kicktipp-predictor train
```

### Making Predictions

To generate predictions for upcoming matches:
```bash
kicktipp-predictor predict
```

### Evaluating Performance

To run a full back-testing evaluation of the model:
```bash
kicktipp-predictor evaluate
```

## The V3 Architecture

The V3 architecture is a strategic redesign focused on simplicity and predictive power. It replaces a complex, multi-model pipeline with a single `XGBRegressor` trained to predict the **goal difference** of a match.

This approach has several key advantages:
- **Solves Class Imbalance:** The "draw" outcome is no longer a rare class but a natural result of a predicted goal difference near zero.
- **Improves Accuracy:** By modeling the ordinal relationship between outcomes, the model learns a more nuanced representation of team strength.
- **Reduces Complexity:** A single model is easier to tune, debug, and interpret.

For a more detailed explanation of the architecture, please see the [Architecture Documentation](docs/architecture.md).

## Results

The model's performance is continuously evaluated using a rigorous back-testing methodology. Here are the latest results:

- **Accuracy:** 40.8%
- **Average Points:** 1.177
- **Log Loss:** 1.0999

**Note:** The current model configuration does not predict any draws. This is a known issue and a key area for future improvement. 

For a more detailed analysis of the results, including SHAP plots and performance metrics, please see the [Results Documentation](docs/results.md).
