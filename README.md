# Kicktipp Predictor V4

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning-powered football match prediction system using a cascaded two-stage architecture. Predicts match outcomes (Home/Draw/Away) with calibrated probabilities using XGBoost classifiers.

## 🎯 Overview

V4 introduces a **cascaded, two-stage classifier** that models match outcomes more directly and robustly than previous versions:

* **Stage 1 (Gatekeeper):** Binary classifier predicts `Draw` vs `NotDraw`
* **Stage 2 (Finisher):** Binary classifier predicts `HomeWin` vs `AwayWin` (conditioned on NotDraw)

Final probabilities are combined via the law of total probability to produce calibrated H/D/A predictions. This design targets realistic draw rates (18-28%) while improving accuracy and interpretability.

### Key Features

- ✅ **Cascaded Architecture:** Two-stage binary classification for robust predictions
- ✅ **Feature Engineering:** Elo ratings, team form, schedule context, and more
- ✅ **Dynamic Evaluation:** Expanding-window retraining simulating real-world usage
- ✅ **Hyperparameter Tuning:** Optuna-based optimization for both classifiers
- ✅ **Rich CLI:** Easy-to-use commands for training, prediction, and evaluation
- ✅ **Web Interface:** Flask-based UI for interactive predictions (optional)

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Commands](#-cli-commands)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kicktipp-predictor.git
cd kicktipp-predictor

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base dependencies
pip install -e .
```

### Installation with Optional Extras

```bash
# For development (linting, type checking, pre-commit hooks)
pip install -e ".[dev]"

# For SHAP value analysis and plots
pip install -e ".[plots]"

# For hyperparameter tuning with Optuna
pip install -e ".[tuning]"

# Install everything
pip install -e ".[dev,plots,tuning]"
```

**Note for zsh users:** Quote the brackets to avoid globbing:
```bash
pip install -e '.[dev,plots,tuning]'
```

---

## ⚡ Quick Start

### 1. Train the Model

Train on the last 5 seasons of data:

```bash
python -m kicktipp_predictor.cli train --seasons-back 5
```

This will:
- Fetch historical match data
- Engineer features (Elo, form, schedule context)
- Train two XGBoost classifiers (draw gatekeeper + win finisher)
- Save models to `data/models/`

### 2. Make Predictions

Predict upcoming matches in the next 7 days:

```bash
python -m kicktipp_predictor.cli predict --days 7
```

Or predict a specific matchday:

```bash
python -m kicktipp_predictor.cli predict --matchday 15
```

### 3. Evaluate Performance

Run expanding-window evaluation on a season:

```bash
python -m kicktipp_predictor.cli evaluate --retrain-every 1
```

This simulates real-world usage by retraining before each matchday and evaluating predictions.

---

## 🛠️ CLI Commands

### `train`

Train the cascaded predictor on historical data.

```bash
python -m kicktipp_predictor.cli train [OPTIONS]
```

**Options:**
- `--seasons-back INTEGER`: Number of past seasons to use for training (default: 5)
- `--validate / --no-validate`: Run 3-fold CV diagnostics after training (default: `--no-validate`)

**Example:**
```bash
python -m kicktipp_predictor.cli train --seasons-back 3
```

Optionally include CV diagnostics during development:
```bash
python -m kicktipp_predictor.cli train --seasons-back 3 --validate
```

---

### `predict`

Make predictions for upcoming matches.

```bash
python -m kicktipp_predictor.cli predict [OPTIONS]
```

**Options:**
- `--days INTEGER`: Days ahead to predict (default: 7)
- `--matchday INTEGER`: Specific matchday to predict (optional)

**Examples:**
```bash
# Predict next 14 days
python -m kicktipp_predictor.cli predict --days 14

# Predict matchday 20
python -m kicktipp_predictor.cli predict --matchday 20
```

---

### `evaluate`

Evaluate season performance with expanding-window retraining.

```bash
python -m kicktipp_predictor.cli evaluate [OPTIONS]
```

**Options:**
- `--season INTEGER`: Season to evaluate (default: current season)
- `--retrain-every INTEGER`: Retrain model every N matchdays (default: 1)
- `--start-matchday INTEGER`: Starting matchday for evaluation (default: 10)

**Example:**
```bash
python -m kicktipp_predictor.cli evaluate --season 2023 --retrain-every 2
```

This outputs:
- Overall metrics (accuracy, log loss, Brier score, RPS)
- Per-matchday performance
- Confusion matrix
- Predicted vs actual outcome distribution

---

### `tune`

Run Optuna-based hyperparameter tuning.

```bash
python -m kicktipp_predictor.cli tune [OPTIONS]
```

**Options:**
- `--n-trials INTEGER`: Number of Optuna trials (default: 100)
- `--seasons-back INTEGER`: Seasons for training data (default: 5)
- `--model-to-tune TEXT`: Which model to tune: "draw", "win", or "both" (default: "both")

**Example:**
```bash
python -m kicktipp_predictor.cli tune --n-trials 200 --seasons-back 4
```

**Note:** Requires the `tuning` extra to be installed.

---

### `web`

Launch the Flask web interface (if implemented).

```bash
python -m kicktipp_predictor.cli web
```

---

## 🏗️ Architecture

### Cascaded Prediction Pipeline

```
┌──────────────────────────┐
│  Historical Match Data   │
│  (Bundesliga, etc.)      │
└──────────┬───────────────┘
           │
           v
┌──────────────────────────┐
│  Feature Engineering     │
│  • Elo Ratings           │
│  • Team Form (5-match)   │
│  • Schedule Context      │
│  • Head-to-Head Stats    │
└──────────┬───────────────┘
           │
           v
┌──────────────────────────┐
│  Stage 1: Draw Classifier│
│  P(Draw) vs P(NotDraw)   │
│  (XGBoost Binary)        │
└──────────┬───────────────┘
           │
           v
┌──────────────────────────┐
│  Stage 2: Win Classifier │
│  P(Home|ND) vs P(Away|ND)│
│  (XGBoost Binary)        │
└──────────┬───────────────┘
           │
           v
┌──────────────────────────┐
│  Probability Combination │
│  P(H) = P(ND)*P(H|ND)    │
│  P(A) = P(ND)*P(A|ND)    │
│  P(D) = P(Draw)          │
└──────────┬───────────────┘
           │
           v
┌──────────────────────────┐
│  Final H/D/A Predictions │
│  (Normalized)            │
└──────────────────────────┘
```

### Why Cascaded?

1. **Realistic Draw Rates:** Previous models often under-predicted draws (V3: 0.8%). The gatekeeper explicitly models draws, achieving 18-28% predicted draw rates.
2. **Specialized Classifiers:** Each stage focuses on a specific binary decision, improving accuracy.
3. **Better Calibration:** Combining probabilities via law of total probability ensures valid probability distributions.
4. **Interpretability:** Easier to analyze where predictions fail (draw detection vs. home/away distinction).

---

## ⚙️ Configuration

Configuration is managed in `src/kicktipp_predictor/config.py` using Python dataclasses.

### Key Parameters

#### Draw Classifier (Stage 1)
```python
draw_n_estimators: int = 400
draw_max_depth: int = 5
draw_learning_rate: float = 0.05
draw_subsample: float = 0.7
draw_colsample_bytree: float = 0.7
draw_scale_pos_weight: float = 3.0  # Balances class imbalance
```

#### Win Classifier (Stage 2)
```python
win_n_estimators: int = 800
win_max_depth: int = 6
win_learning_rate: float = 0.1
win_subsample: float = 0.8
win_colsample_bytree: float = 0.8
```

### Modifying Configuration

Edit `src/kicktipp_predictor/config.py` or use environment variables/config files as needed.

---

## 📊 Performance

V4 targets the following performance metrics on Bundesliga data:

| Metric | Target | V3 Baseline |
|--------|--------|-------------|
| **Accuracy** | 40-50% | 31.7% |
| **Predicted Draw Rate** | 18-28% | 0.8% |
| **Log Loss** | < 1.0 | 1.15 |
| **Brier Score** | < 0.5 | 0.58 |
| **RPS** | < 0.25 | 0.28 |

**Note:** Performance varies by league, season, and training data size. Run `evaluate` to benchmark on your data.

### Confusion Matrix (Typical)

```
           Predicted
Actual    H    D    A
  H      [45]  12   8
  D       9  [18]  10
  A       7   11  [42]
```

The model excels at distinguishing home wins from away wins, with draws being the most challenging outcome (as expected in football).

---

## 👨‍💻 Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Code Quality Tools

- **Linting:** `ruff check src/ tests/`
- **Formatting:** `ruff format src/ tests/`
- **Type Checking:** `mypy src/`
- **Tests:** `pytest tests/`

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:
- Ruff formatting and linting
- Trailing whitespace removal
- YAML/JSON validation

### Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
kicktipp-predictor/
├── src/
│   └── kicktipp_predictor/
│       ├── __init__.py
│       ├── cli.py              # CLI commands
│       ├── predictor.py        # CascadedPredictor class
│       ├── data.py             # Data loading & feature engineering
│       ├── config.py           # Configuration management
│       ├── evaluate.py         # Evaluation logic
│       ├── tune.py             # Optuna tuning
│       ├── metrics.py          # Custom metrics
│       └── web/                # Flask web interface (optional)
├── tests/                      # Unit and integration tests
├── data/
│   ├── cache/                  # Cached match data
│   └── models/                 # Trained model artifacts
├── docs/                       # Additional documentation
├── scripts/                    # Utility scripts
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── BLUEPRINT.md                # Detailed technical architecture
├── ROADMAP.md                  # Future development plans
└── LICENSE
```

---

## 📚 Additional Documentation

- **[BLUEPRINT.md](BLUEPRINT.md):** Detailed technical architecture and implementation guide
- **[ROADMAP.md](ROADMAP.md):** Planned features and improvements
- **[docs/](docs/):** Additional guides and examples

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Ensure all tests pass and code is formatted with Ruff.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Data sourced from public football statistics APIs
- Built with XGBoost, scikit-learn, and Typer
- Inspired by modern football analytics and probabilistic modeling

---

## 📞 Support

For questions, issues, or feature requests, please:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review `BLUEPRINT.md` for technical details

---

**Happy Predicting! ⚽📈**
