# 3. Liga Match Predictor

**Version 2.0** - A clean, transparent machine learning predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

> **What's New in v2.0:** Predictor-Selector architecture with selectable outcome probability source (classifier, Poisson-derived, or hybrid) and clearer evaluation tooling.

## Architecture: The Predictor-Selector Model

Version 2.0 introduces a clear **two-step prediction system**:

1. **Outcome Prediction (The Selector)**: A dedicated XGBoost classifier determines the match result (Home Win, Draw, or Away Win)
2. **Scoreline Selection (The Predictor)**: XGBoost regressors predict expected goals, then a Poisson grid selects the most probable scoreline matching the predicted outcome

Outcome probabilities used for evaluation can be sourced from the classifier, derived from the goal Poisson model, or blended via a simple hybrid weight.

## Features
- **Clear Two-Step Architecture**: Outcome first, then scoreline - easy to understand and debug
- **Selectable Outcome Probabilities**: Use classifier probabilities, Poisson-derived probabilities, or a hybrid blend
- **Advanced Feature Engineering**: 80+ features including momentum, strength of schedule, rest days, and goal patterns
- **EWMA Recency Features (new)**: Leakage-safe exponentially weighted moving averages for recent form (goals for/against, goal diff, and points per match) precomputed once and merged into match features
- **Comprehensive Evaluation**: Brier score, log loss, RPS, accuracy, and Kicktipp points
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics
- **Automatic Data Fetching**: Fetches match data from OpenLigaDB API with intelligent caching
- **Centralized Configuration**: All settings managed in one place via `config/best_params.yaml`

### Performance

The feature engineering pipeline is fully vectorized with Pandas (`groupby` + `expanding/rolling` + `merge_asof`), reducing complexity from quadratic to near-linear. Training feature construction typically completes in seconds instead of minutes.

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

# Evaluate model performance (default probability source)
python3 -m kicktipp_predictor evaluate

# Evaluate with Poisson-derived probabilities and detailed plots
python3 -m kicktipp_predictor evaluate --prob-source poisson --detailed

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
- **`evaluate [--season] [--dynamic] [--retrain-every N]`**: Evaluate on a test split or across the current season; with `--dynamic`, retrain every N matchdays (default: 1)
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

# Predict using Poisson-derived probabilities
python3 -m kicktipp_predictor predict --days 7 --prob-source poisson --proba-grid-max-goals 12

# Predict using hybrid probabilities (blend classifier with Poisson)
python3 -m kicktipp_predictor predict --days 7 --prob-source hybrid --hybrid-poisson-weight 0.5
```

### Season Evaluation (Static vs Dynamic)
```bash
# Static season evaluation (train-once predictor state; evaluates finished matchdays)
python3 -m kicktipp_predictor evaluate --season

# Dynamic season evaluation (expanding window): retrain before each evaluated matchday
python3 -m kicktipp_predictor evaluate --season --dynamic

# Dynamic with less frequent retraining (e.g., every 4 matchdays)
python3 -m kicktipp_predictor evaluate --season --dynamic --retrain-every 4
```

Notes:
- `--dynamic` mirrors real usage more closely by incrementally retraining as the season unfolds.
- This is significantly slower (retraining many times). Use `--retrain-every N` to control cadence.

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
   │ • Outcome probabilities (H/D/A) from configured source (classifier / Poisson / hybrid)
   │ • Confidence metric
   ↓
7. Performance Tracking (points calculation)
```

### EWMA Recency Features

To capture short-term momentum without leakage, the feature pipeline precomputes exponentially weighted moving averages (span=5, adjust=False) on prior match values per team and merges them into match-level features. Added columns include:

- `home_goals_for_ewm5`, `away_goals_for_ewm5`
- `home_goals_against_ewm5`, `away_goals_against_ewm5`
- `home_goal_diff_ewm5`, `away_goal_diff_ewm5`
- `home_points_ewm5`, `away_points_ewm5`

Early-season gaps are filled with global means to handle cold starts. These features are computed once across the dataset and reused for both training and prediction paths.

### Benefits

- Clear separation of concerns between outcome decision and score magnitude
- Option to derive H/D/A probabilities directly from the Poisson goal model
- Flexible evaluation: choose the probability source that optimizes Brier/LogLoss

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
- **`max_goals`**: Maximum goals to consider in Poisson grid for scoreline selection (default: 8)
- **`proba_grid_max_goals`**: Grid cap for Poisson-derived probabilities (default: 12)
- **`min_lambda`**: Minimum expected goals to prevent degenerate predictions (default: 0.2)
- **`prob_source`**: Outcome probability source: `classifier` | `poisson` | `hybrid` (default: `classifier`)
- **`hybrid_poisson_weight`**: When `prob_source=hybrid`, weight of Poisson probabilities in [0,1] (default: 0.5)
- **`proba_temperature`**: Temperature scaling for classifier probabilities (default: 1.0)
- **`prior_blend_alpha`**: Empirical-prior blending for classifier probabilities (applies only when `prob_source='classifier'`)
- **`draw_boost`**: Class weight multiplier for draws during classifier training

### XGBoost Hyperparameters
- **Outcome Classifier**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Goal Regressors**: Same hyperparameters, tuned for Poisson regression

These parameters can be tuned using the `tune` command (see Advanced Usage below).

## Advanced Usage

### Hyperparameter Tuning (Optuna, recommended objective)
Use the CLI `tune` command to optimize hyperparameters via time-series CV using Optuna. Starting with v2.1, the outcome classifier is trained with class-balanced weights by default, combined with time-decay and an optional `draw_boost`. We recommend optimizing for weighted Kicktipp points (PPG). The tuner writes per-objective YAMLs (e.g., `config/best_params_logloss.yaml`) and copies the winner (by weighted PPG) to `config/best_params.yaml`. The search space includes `prob_source`, `hybrid_poisson_weight`, and `proba_grid_max_goals`.

Serial example (no storage required):
```bash
python3 -m kicktipp_predictor tune --n-trials 200 --n-splits 3 --workers 1 --objective ppg
```

Parallel example (database-coordinated workers):
```bash
python3 -m kicktipp_predictor tune \
  --n-trials 200 \
  --n-splits 3 \
  --workers 8 \
  --storage "sqlite:///data/kicktipp_study.db?timeout=60" \
  --objective ppg

### Recommended study run
After enabling the balanced trainer, run a longer PPG-focused study with modest parallelism to reduce DB lock contention:
```bash
python3 -m kicktipp_predictor tune \
  --n-trials 500 \
  --n-splits 3 \
  --workers 8 \
  --storage "sqlite:///data/study_balanced_ppg.db?timeout=120" \
  --objective ppg
```
```

Notes:
- `--n-trials` is the total trial budget across all workers (evenly split).
- `--workers` controls process-level parallelism. Each worker is single-threaded internally to avoid nested parallelism.
- When `--workers > 1`, a storage URL is required (SQLite with `?timeout=60` recommended, or a remote RDBMS). In compare mode with SQLite, the CLI automatically derives objective-specific files (e.g., `..._logloss.db`).

Common options:
- `--n-trials <number>`: Total Optuna trials across all workers.
- `--n-splits <number>`: Time-series CV folds.
- `--workers <number>`: Number of worker processes (default: 1).
- `--storage <url>`: Optuna storage URL (required if `--workers > 1`).
- `--objective <name>`: ppg|ppg_unweighted|logloss|brier|balanced_accuracy|accuracy|rps
- `--direction <dir>`: auto|maximize|minimize (auto resolves from objective)
- `--compare <list>`: comma-separated objectives to run and compare (ignores `--objective`)

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
