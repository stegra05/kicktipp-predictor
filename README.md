# 3. Liga Match Predictor

**Version 2.0** - A clean, transparent machine learning predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

> What's New: Dynamic season evaluation only, with a richer console report powered by `rich`. Plots are no longer produced by evaluation; artifacts are concise JSON/CSV. 

## Architecture: The Predictor-Selector Model

Version 2.0 introduces a clear two-step prediction system:

1. Outcome Prediction (The Selector): A dedicated XGBoost classifier determines the match result (Home Win, Draw, or Away Win)
2. Scoreline Selection (The Predictor): XGBoost regressors predict expected goals, then a Poisson grid selects the most probable scoreline matching the predicted outcome

Outcome probabilities used for evaluation can be sourced from the classifier, derived from the goal Poisson model, or blended via a simple hybrid weight.

## Features
- **Clear Two-Step Architecture**: Outcome first, then scoreline - easy to understand and debug
- **Selectable Outcome Probabilities**: Use classifier probabilities, Poisson-derived probabilities, or a hybrid blend
- **Dynamic Feature Management**: Comes with 80+ features, but you can use the built-in **Feature Ablation Study** to find the optimal subset for performance and simplicity.
- **EWMA Recency Features**: Leakage-safe exponentially weighted moving averages capture recent form.
- **Comprehensive Evaluation**: Brier score, log loss, RPS, accuracy, and Kicktipp points.
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned.
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics.
- **Automatic Data Fetching**: Fetches match data from OpenLigaDB API with intelligent caching.
- **Centralized Configuration**: All settings managed in one place via `config/best_params.yaml`.

### Performance

The feature engineering pipeline is fully vectorized with Pandas (`groupby` + `expanding/rolling` + `merge_asof`), reducing complexity from quadratic to near-linear. Training feature construction typically completes in seconds instead of minutes.

## Getting Started

### Installation
```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
pip install -e .
```

If you prefer without install, set `PYTHONPATH` to include `src`:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 2. CLI Quickstart
```bash
# Show help
kicktipp_predictor --help

# Train models (uses last 3 seasons by default)
kicktipp_predictor train

# Train with more historical data
kicktipp_predictor train --seasons-back 5

# Predict upcoming matches (next 7 days)
kicktipp_predictor predict --days 7

# Predict with process-parallel scoreline selection (e.g., 8 workers)
kicktipp_predictor predict --days 7 --workers 8

# Predict specific matchday
kicktipp_predictor predict --matchday 15

# Run web UI (defaults to 127.0.0.1:8000)
kicktipp_predictor web --host 0.0.0.0 --port 8000
```

## Usage

### Weekly Workflow
This is the recommended weekly routine for generating predictions:

1.  **Generate Predictions:** Generate predictions for the next matchday.
    ```bash
    kicktipp_predictor predict
    ```

2.  **View Predictions:** Use the web interface to view predictions with probabilities and confidence.
    ```bash
    kicktipp_predictor web
    ```

### Monthly Maintenance
To keep the models accurate, retrain them monthly with the latest match data:
```bash
kicktipp_predictor train
```

## CLI Commands

All functionality is available through the CLI:

- **`train [--seasons-back N]`**: Train models on historical data (default: 3 seasons).
- **`predict [--days N | --matchday N]`**: Generate predictions for upcoming matches.
- **`evaluate [--retrain-every N]`**: Evaluate on a test split or across the current season; with `--dynamic`, retrain every N matchdays (default: 1).
- **`web [--host HOST] [--port PORT]`**: Run the Flask web UI.
- **`tune [options]`**: Hyperparameter tuning (wrapper around `experiments/auto_tune.py`).
- **`shap`**: Run SHAP analysis to understand feature importance.

## Season Evaluation (Dynamic Only)

The `evaluate` command now uses a single dynamic, expanding-window procedure:
- Retrains every N matchdays (default: 1) on all historical + season-to-date matches
- Evaluates finished matchdays in the current season
- Writes:
  - `data/predictions/metrics_season.json`
  - `data/predictions/per_matchday_metrics_season.csv`


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

The project features a clean hierarchy with all core logic organized into modules:

```
kicktipp-predictor/
├── src/
│   └── kicktipp_predictor/
│       ├── __init__.py           # Package exports
│       ├── __main__.py           # CLI entry point
│       ├── cli.py                # Typer-based CLI commands
│       ├── config.py             # Centralized configuration
│       ├── data.py               # Data loading & feature engineering
│       ├── predictor.py          # Predictor-Selector model
│       ├── evaluate.py           # Evaluation logic
│       ├── analysis/
│       │   └── shap_analysis.py  # SHAP analysis
│       └── web/
│           ├── app.py            # Flask web application
│           ├── templates/        # HTML templates
│           └── static/           # CSS, JS assets
├── data/
│   ├── cache/                    # API response cache
│   ├── models/                   # Trained model files
│   ├── feature_selection/        # (Optional) Kept features list
│   └── feature_ablation/         # Study artifacts
├── config/
│   └── best_params.yaml          # Model hyperparameters
├── experiments/                  # Tuning & ablation scripts
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
- `max_goals`: Maximum goals considered in Poisson scoreline selection (default: 8)
- `proba_grid_max_goals`: Grid cap for Poisson-derived probabilities (default: 12)
- Note: Expected goals clamp uses an internal constant `MIN_LAMBDA=0.2` and is not tunable via config.
- **`prob_source`**: Outcome probability source: `classifier` | `poisson` | `hybrid` (default: `hybrid`)
- **`hybrid_poisson_weight`**: When `prob_source=hybrid`, fixed weight of Poisson probabilities in [0,1] (default: 0.0525).
- **`proba_temperature`**: Temperature scaling for classifier probabilities (default: 1.0)

- **`draw_boost`**: Class weight multiplier for draws during classifier training
- **`use_ep_selection`**: Enable EP-maximizing scoreline selection (default: True)
- **`use_time_decay`**: Apply recency weighting during training (default: True)
- **`time_decay_half_life_days`**: Half-life in days for time-decay weights (default: 330, typically overridden via YAML)

### XGBoost Hyperparameters
- **Outcome Classifier**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Goal Regressors**: Same hyperparameters, tuned for Poisson regression

These parameters can be tuned using the `tune` command (see Advanced Usage below).

## Advanced Usage

### Hyperparameter Tuning (Optuna)
Use the CLI `tune` command to optimize hyperparameters via time-series CV using Optuna.

Serial example (no storage required):
```bash
kicktipp_predictor tune --n-trials 200 --n-splits 3 --workers 1
```

Parallel example (database-coordinated workers):
```bash
kicktipp_predictor tune \
  --n-trials 200 \
  --n-splits 3 \
  --workers 8 \
  --storage "sqlite:///data/kicktipp_study.db?timeout=60"
```

### Feature Ablation Study
To find the optimal balance between model complexity and performance, run the feature ablation study. This script systematically tests different feature subsets by removing categories one by one and analyzing the impact.

**Usage:**
```bash
./run_feature_study.sh
```

The study will:
1.  **Run Experiments**: Test various scenarios, including a baseline with all features, removing feature categories (e.g., `ewma_recency`, `venue_specific`), and testing a minimal core set.
2.  **Generate Report**: Output a summary report with performance metrics (accuracy, avg points, Brier score) for each experiment.
3.  **Provide Recommendations**: Suggest optimal feature sets for both **best performance** and **best simplification** (i.e., the smallest feature set with minimal performance loss).

**Interpreting the Results:**

The script will print a detailed report to the console and save the results to `data/feature_ablation/`. The most important section is the **"RECOMMENDED CONFIGURATIONS"**, which provides copy-pasteable YAML output for `kept_features.yaml`.

-   **Best Performance**: The feature set that achieved the highest average points.
-   **Best Simplification**: A smaller, more efficient feature set that performs almost as well as the baseline.

**Workflow:**

1.  **Run the study:** `./run_feature_study.sh`
2.  **Review the report:** Analyze the results in your terminal.
3.  **Update feature set:** Copy the recommended feature list from the report into a new file at `data/feature_selection/kept_features.yaml`.
4.  **Retrain the model:** `python3 -m kicktipp_predictor train`
5.  **Evaluate:** `python3 -m kicktipp_predictor evaluate --season`

This workflow allows you to maintain a lean, high-performing feature set without manual trial and error.

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
See probabilities, confidence and predictions in a simple web UI:
```bash
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 8000
```

## Development
- Editable install: `python3 -m pip install -e .`
- Run tests: `pytest`

## Disclaimer
This predictor is for entertainment purposes only; no model can guarantee accurate predictions.
