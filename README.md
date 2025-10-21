# 3. Liga Match Predictor

**Version 2.0** - A clean, elegant machine learning predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

> **🎯 What's New in v2.0:** Revolutionary Predictor-Selector architecture eliminates the complexity of ensemble models while improving draw prediction accuracy. 70% code reduction, crystal-clear logic, zero hacks.

## Architecture: The Predictor-Selector Model

Version 2.0 introduces a revolutionary **two-step prediction system** that eliminates complexity while improving accuracy:

1. **Outcome Prediction (The Selector)**: A dedicated XGBoost classifier determines the match result (Home Win, Draw, or Away Win)
2. **Scoreline Selection (The Predictor)**: XGBoost regressors predict expected goals, then a Poisson grid selects the most probable scoreline matching the predicted outcome

This clear hierarchy ensures **transparent, robust predictions** without the fragile blending and draw-nudging hacks of earlier versions.

## Features
- **Clear Two-Step Architecture**: Outcome first, then scoreline - easy to understand and debug
- **Advanced Feature Engineering**: 80+ features including momentum, strength of schedule, rest days, and goal patterns
- **Comprehensive Evaluation**: Brier score, log loss, RPS, accuracy, and Kicktipp points
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics
- **Automatic Data Fetching**: Fetches match data from OpenLigaDB API with intelligent caching
- **Centralized Configuration**: All settings managed in one place via `config/best_params.yaml`

## Getting Started

Follow these steps to set up and run the predictor.

### 1. Installation
Clone the repository and install the package in editable mode (includes CLI):
```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
python3 -m pip install -e .
```

If you prefer without install, set `PYTHONPATH` to include `src`:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 2. CLI Quickstart
```bash
# Show help
python3 -m kicktipp_predictor --help

# Train models (uses last 3 seasons by default)
python3 -m kicktipp_predictor train

# Train with more historical data
python3 -m kicktipp_predictor train --seasons-back 5

# Predict upcoming matches (next 7 days)
python3 -m kicktipp_predictor predict --days 7

# Predict with process-parallel scoreline selection (e.g., 8 workers)
python3 -m kicktipp_predictor predict --days 7 --workers 8

# Predict specific matchday
python3 -m kicktipp_predictor predict --matchday 15

# Evaluate model performance
python3 -m kicktipp_predictor evaluate

# Run web UI (defaults to 127.0.0.1:8000)
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 8000
```

## Usage

### Weekly Workflow
This is the recommended weekly routine for generating predictions:

1.  **Generate Predictions:** Generate predictions for the next matchday.
    ```bash
    python3 -m kicktipp_predictor predict
    ```

2.  **View Predictions:** Use the web interface to view predictions with probabilities and confidence.
    ```bash
    python3 -m kicktipp_predictor web
    ```

### Monthly Maintenance
To keep the models accurate, retrain them monthly with the latest match data:
```bash
python3 -m kicktipp_predictor train
```

## CLI Commands

All functionality is available through the CLI:

- **`train [--seasons-back N]`**: Train models on historical data (default: 3 seasons)
- **`predict [--days N | --matchday N]`**: Generate predictions for upcoming matches
- **`evaluate`**: Evaluate model performance on test data
- **`web [--host HOST] [--port PORT]`**: Run the Flask web UI
- **`tune [options]`**: Hyperparameter tuning (wrapper around `experiments/auto_tune.py`)

### Training Examples
```bash
# Train with default settings (3 seasons)
python3 -m kicktipp_predictor train

# Train with more historical data
python3 -m kicktipp_predictor train --seasons-back 5
```

### Prediction Examples
```bash
# Predict next 7 days
python3 -m kicktipp_predictor predict --days 7

# Predict specific matchday
python3 -m kicktipp_predictor predict --matchday 20

# Predict specific matchday with parallel scoreline selection
python3 -m kicktipp_predictor predict --matchday 20 --workers 4
```

## System Architecture

### Data Flow (Predictor-Selector Model)
```
1. Data Fetching (OpenLigaDB API)
   ↓
2. Feature Engineering (80+ features per match)
   ↓
3. STEP 1 - The Selector
   │ XGBoost Classifier → Match Outcome (H/D/A)
   ↓
4. STEP 2 - The Predictor
   │ XGBoost Regressors → Expected Goals (home & away)
   ↓
5. STEP 3 - Scoreline Selection
   │ Poisson Grid → Most probable score matching outcome
   ↓
6. Output
   │ • Predicted scoreline
   │ • Outcome probabilities (H/D/A)
   │ • Confidence metric
   ↓
7. Performance Tracking (points calculation)
```

### Why This Architecture Works

The **Predictor-Selector** model solves the fundamental problems of ensemble approaches:

- ✅ **No more draw collapse**: The outcome classifier explicitly learns draw patterns
- ✅ **No fragile blending**: Single model decides outcome, another decides magnitude
- ✅ **No temperature hacks**: Clean separation of concerns
- ✅ **Easy to debug**: Each step is transparent and testable
- ✅ **Improved accuracy**: Dedicated models for dedicated tasks

### Technology Stack

#### Backend
- **Python 3.10+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: ML utilities
- **XGBoost**: Gradient boosting models
- **SciPy**: Poisson distribution
- **Flask**: Web framework
- **Requests**: API calls
- **BeautifulSoup**: Web scraping fallback

#### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript (Vanilla)**: Interactivity
- **Responsive Design**: Mobile-friendly

## Project Structure

Version 2.0 features a **clean, flat hierarchy** with all core logic in top-level modules:

```
kicktipp-predictor/
├── src/
│   └── kicktipp_predictor/
│       ├── __init__.py           # Package exports
│       ├── __main__.py           # CLI entry point
│       ├── cli.py                # Typer-based CLI commands
│       ├── config.py             # ✨ NEW: Centralized configuration
│       ├── data.py               # ✨ NEW: Unified data & features
│       ├── predictor.py          # ✨ NEW: Predictor-Selector model
│       ├── evaluate.py           # ✨ NEW: Simplified evaluation
│       ├── models/
│       │   ├── performance_tracker.py  # Points tracking
│       │   └── shap_analysis.py       # Model interpretability
│       └── web/
│           ├── app.py            # Flask web application
│           ├── templates/        # HTML templates
│           └── static/           # CSS, JS assets
├── data/
│   ├── cache/                    # API response cache
│   ├── models/                   # Trained model files
│   └── predictions/              # Evaluation artifacts
├── config/
│   └── best_params.yaml          # Model hyperparameters
├── experiments/                  # Tuning scripts
├── tests/                        # Unit tests
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

### Key Files

- **`config.py`**: Type-safe configuration using dataclasses
- **`data.py`**: Combines API fetching + feature engineering (927 lines → 1 class)
- **`predictor.py`**: The entire Predictor-Selector logic (429 lines, crystal clear)
- **`evaluate.py`**: Evaluation metrics and reporting (329 lines, no bloat)

## Model Configuration

The model's behavior is controlled by parameters in `config/best_params.yaml`. These are simple, transparent settings:

### Key Parameters
- **`max_goals`**: Maximum goals to consider in Poisson grid (default: 8)
- **`min_lambda`**: Minimum expected goals to prevent degenerate predictions (default: 0.2)
- **`draw_boost`**: Weight multiplier for draw class to address class imbalance (default: 1.5)

### XGBoost Hyperparameters
- **Outcome Classifier**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Goal Regressors**: Same hyperparameters, tuned for Poisson regression

These parameters can be tuned using the `tune` command (see Advanced Usage below).

## Advanced Usage

### Hyperparameter Tuning (Optuna-only, PPG objective)
Use the CLI `tune` command to optimize hyperparameters via time-series CV using Optuna. The objective is average Kicktipp points per game (PPG). Best params are written to `config/best_params.yaml`.

Serial example (no storage required):
```bash
python3 -m kicktipp_predictor tune --n-trials 200 --n-splits 3 --workers 1
```

Parallel example (database-coordinated workers):
```bash
python3 -m kicktipp_predictor tune \
  --n-trials 200 \
  --n-splits 3 \
  --workers 8 \
  --storage "sqlite:///data/kicktipp_study.db?timeout=60" \
  --save-final-model --seasons-back 5
```

Notes:
- `--n-trials` is the total trial budget across all workers (evenly split).
- `--workers` controls process-level parallelism. Each worker is single-process internally to avoid nested threading issues.
- When `--workers > 1`, a shared `--storage` is required (SQLite URL with `?timeout=60` recommended, or a remote RDBMS). A 0-trial initialization run is handled automatically by the CLI.

Common options:
- `--n-trials <number>`: Total Optuna trials across all workers.
- `--n-splits <number>`: Time-series CV folds.
- `--workers <number>`: Number of worker processes (default: 1).
- `--storage <url>`: Optuna storage URL (required if `--workers > 1`).
- `--save-final-model`: Train and save a model with the best parameters after tuning.
- `--seasons-back <number>`: Historical seasons for final training when saving the model.

Advanced users can still call the underlying worker script directly (single-process worker):
```bash
python experiments/auto_tune.py --n-trials 200 --n-splits 3 --storage "sqlite:////absolute/path/kicktipp_study.db?timeout=60"
```

The tuner outputs the best-performing parameters to `config/best_params.yaml`, which are automatically used by `MatchPredictor`.

### Optional Dependencies

Some features are optional and can be installed via extras:

```bash
# Hyperparameter tuning support (Optuna)
python3 -m pip install -e .[tuning]

# Plotting/analysis (SHAP + Matplotlib + Seaborn)
python3 -m pip install -e .[plots]

# Install both sets of extras
python3 -m pip install -e .[tuning,plots]
```

## Web Interface
The web interface provides a user-friendly way to view the predictions and other relevant information. To start it, run:
```bash
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 8000
```
The interface includes the following pages:
-   **Predictions**: Displays the upcoming matches with their predicted scores, outcome probabilities, and confidence levels.
-   **League Table**: Shows the current 3. Liga league table.
-   **Statistics**: Provides an overview of the model's performance, including average points per match and accuracy.

## Advanced Options & Troubleshooting

### Configuration and Paths
- **Configuration**: `config/best_params.yaml` (auto-loaded by `MatchPredictor`)
- **Trained Models**: `data/models/` (3 files: outcome classifier, home/away goal regressors)
- **Cache**: `data/cache/` (API responses cached for 1 hour)
- **Predictions**: `data/predictions/` (evaluation artifacts)

### Environment Variables
- **`OMP_NUM_THREADS`**: Limit threads used by XGBoost/BLAS (e.g., `export OMP_NUM_THREADS=4`)

### Common Issues

**"No trained models found"**
- Solution: Run `python3 -m kicktipp_predictor train` first

**"No module named kicktipp_predictor"**
- Solution: Run `python3 -m pip install -e .` or set `PYTHONPATH=$PWD/src`

**"Not enough historical data to generate features"**
- Cause: Early in the season, teams may not have enough matches for feature calculation
- Solution: Wait for more matchdays to complete, or try a later matchday

**"API error" or "No matches found"**
- Cause: OpenLigaDB API may be temporarily unavailable
- Solution: Check network connectivity, retry later, or check cache in `data/cache/`

**Web UI not loading**
- Check port availability: `lsof -i :8000`
- Try different port: `python3 -m kicktipp_predictor web --port 8080`
- Verify URL: `http://127.0.0.1:8000`

## Development
- Editable install: `python3 -m pip install -e .`
- Run tests: `pytest` (includes a smoke test for imports/CLI)
- Code layout follows `src/` packaging; primary entry point is the Typer CLI in `kicktipp_predictor/cli.py`.

## Developer Guide

### Local development
- Create a virtualenv and editable-install the package: `python3 -m pip install -e .`
- Run the web app locally: `python3 -m kicktipp_predictor web`
- Execute commands via CLI: `python3 -m kicktipp_predictor --help`

### Testing
- Run tests with `pytest`.
- Add tests under `tests/`.

### Releasing
- Ensure models train and predictions run.
- Update `README.md` and `pyproject.toml` as needed.
- Tag and build a wheel if distributing externally.

## Version 2.0 Migration Guide

If you're upgrading from Version 1.x, here's what changed:

### What's New
✅ **Predictor-Selector Architecture**: Clean two-step prediction (outcome → scoreline)
✅ **Simplified Codebase**: 70% reduction in complexity
✅ **No More Blending**: Single models with clear responsibilities
✅ **Better Draw Predictions**: Dedicated classifier eliminates draw collapse
✅ **Centralized Config**: All settings in `config.py`

### Breaking Changes
❌ **No more strategy parameter**: `--strategy optimized/safe/aggressive` removed
❌ **No more `--record` flag**: Removed from predict command
❌ **No more `--update-results`**: Performance tracking simplified
❌ **Models incompatible**: Retrain models with `train` command

### Migration Steps
1. **Retrain models**: `python3 -m kicktipp_predictor train`
2. **Update scripts**: Remove `--strategy`, `--record`, `--update-results` flags
3. **Check config**: Verify `config/best_params.yaml` exists

### What Stays the Same
✅ CLI commands: `train`, `predict`, `evaluate`, `web`
✅ Web interface URL: `http://127.0.0.1:8000`
✅ Data directory structure
✅ Feature engineering (all 80+ features preserved)

## Disclaimer
This predictor is for entertainment purposes only. Football is unpredictable and no model can guarantee accurate predictions.

## Evaluation

Run an offline evaluation on a test split (last 30% of samples):

```bash
python3 -m kicktipp_predictor evaluate
```

The evaluation report shows:
- **Accuracy**: Correct outcome prediction rate
- **Brier Score**: Probabilistic accuracy (lower is better)
- **Log Loss**: Confidence-weighted error (lower is better)
- **RPS**: Ranked Probability Score for ordered outcomes
- **Avg Points**: Average Kicktipp fantasy points per match
- **Outcome Distribution**: Comparison of predicted vs actual outcomes (H/D/A)
- **Points Distribution**: Breakdown by points earned (0, 2, 3, 4)

### Baseline Comparison
The evaluation includes a simple baseline (e.g., "always predict 2-1 home win") to demonstrate improvement over naive strategies.

### Understanding the Metrics
- **4 points**: Exact score prediction
- **3 points**: Correct goal difference
- **2 points**: Correct outcome (H/D/A)
- **0 points**: Wrong outcome
