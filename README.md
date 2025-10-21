# 3. Liga Match Predictor

A hybrid machine learning and statistical predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

## Features
- **Advanced Hybrid Prediction Model**: Combines XGBoost ML models with Dixon-Coles enhanced Poisson models
- **Probability Calibration**: Isotonic regression calibration for ML expected goals, blended outcome probabilities
- **Enhanced Confidence Metrics**: Margin-based confidence calculation with entropy-based alternatives
- **Advanced Feature Engineering**: 80+ features including momentum, strength of schedule, rest days, and goal patterns
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis by outcome/confidence
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics
- **Data Scraping**: Automatic fetching of match data from OpenLigaDB API

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

# Train models
python3 -m kicktipp_predictor train

# Predict upcoming (7 days)
python3 -m kicktipp_predictor predict --days 7

# Predict and record to data/predictions
python3 -m kicktipp_predictor predict --record

# Evaluate whole season
python3 -m kicktipp_predictor evaluate --season

# Run web UI
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 5000
```

## Usage

### Weekly Workflow
This is the recommended weekly routine for generating predictions.

1.  **Update Results:** After a matchday is complete, update the performance tracker with the actual scores.
    ```bash
    python3 -m kicktipp_predictor predict --update-results
    ```

2.  **Generate New Predictions:** Generate and save predictions for the next matchday.
    ```bash
    python3 -m kicktipp_predictor predict --record
    ```

3.  **View Predictions:** Use the web interface to view the latest predictions.
    ```bash
    python3 -m kicktipp_predictor web
    ```

### Monthly Maintenance
To keep the models accurate, retrain them monthly with the latest match data.
```bash
python3 -m kicktipp_predictor train
```

## CLI Commands

All functionality is available through the CLI:

- `train`: Train models on historical data
- `predict [--days N | --matchday N] [--record] [--update-results]`: Generate predictions
- `evaluate [--season]`: Evaluate on test split or entire season
- `web [--host HOST] [--port PORT]`: Run the Flask web UI
- `tune [options]`: Hyperparameter tuning (wrapper around `experiments/auto_tune.py`)

Examples:
```bash
# Grid-based tuning with refinement
python3 -m kicktipp_predictor tune \
  --max-trials 100 --n-splits 3 --objective points \
  --n-jobs 4 --omp-threads 2 --refine

# Optuna-based tuning (requires `optuna`)
python3 -m kicktipp_predictor tune --optuna 50
```

## System Architecture

### Data Flow
```
1. Data Fetching (OpenLigaDB API)
   ↓
2. Feature Engineering (80+ features per match)
   ↓
3. Model Prediction
   ├── Poisson Model (statistical)
   ├── ML Model (XGBoost)
   └── Hybrid Ensemble (weighted average)
   ↓
4. Strategy Optimization
   ↓
5. Output (Predictions + Probabilities)
   ↓
6. Performance Tracking (points calculation)
```

### Technology Stack

#### Backend
- **Python 3.8+**
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
```
kicktipp-predictor/
├── src/
│   └── kicktipp_predictor/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── core/
│       │   ├── features/
│       │   └── scraper/
│       ├── models/
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── evaluate.py
│       └── web/
│           ├── app.py
│           └── templates/ | static/
├── data/
│   ├── cache/
│   ├── models/
│   └── predictions/
├── experiments/
├── config/
├── tests/
├── pyproject.toml
└── README.md
```

## Prediction Strategy

The model's prediction strategy is determined by the parameters in `config/best_params.json`. These parameters are automatically optimized when you run the `auto_tune.py` script.

### Key Parameters
- `ml_weight`: The weight given to the machine learning model in the hybrid ensemble.
- `prob_blend_alpha`: The blending factor between the Poisson-based probability grid and the ML classifier probabilities.
- `min_lambda`: The minimum expected goals value to prevent degenerate predictions.
- `goal_temperature`: A scaling factor to adjust the predicted goals to match observed distributions.
- `confidence_threshold`: The confidence level below which a "safe" prediction strategy is applied.
- `strategy`: The prediction strategy to use. The default is `optimized`, which is designed to maximize points.

### Prediction Strategies
- **Optimized**: This is the default and recommended strategy. It uses the best-performing parameters found by `auto_tune.py` to maximize fantasy football points.
- **Balanced**: A well-rounded strategy that provides a good balance between risk and reward.
- **Conservative**: This strategy favors lower-scoring predictions, making it suitable for defensive or tight matches.
- **Aggressive**: A high-risk, high-reward strategy that aims for exact score predictions.
- **Safe**: This strategy prioritizes predicting the correct winner over the exact score, using common scorelines like 1-0, 2-1, and 1-1.

## Advanced Usage

### Hyperparameter Tuning
Use the CLI `tune` command to optimize hyperparameters via time-series CV. Best params are written to `config/best_params.yaml` (or `.json`).

To run the tuner, for grid-based search:
```bash
python3 -m kicktipp_predictor tune --max-trials 100 --n-splits 3 --objective points --refine
```

**Arguments:**
- `--max-trials <number>`: The maximum number of parameter combinations to evaluate. A higher number will take longer but may yield better results.
- `--n-splits <number>`: The number of time-series cross-validation folds to use.
- `--objective <name>`: The optimization objective. Can be `points` (default) or `composite` (balances points with realism).
- `--refine`: Enable a second refinement step to zoom in on the best-performing parameter configurations.
- `--save-final-model`: Train and save a new model using the best-found parameters.
 - `--optuna <N>`: Run Optuna with N trials instead of grid (requires `optuna`).

Advanced users can still call the underlying script directly:
```bash
python experiments/auto_tune.py --max-trials 100 --n-splits 3
```

The tuner will output the best-performing parameters to `config/best_params.yaml` (or `.json`), which are automatically used by the `HybridPredictor`.

## Web Interface
The web interface provides a user-friendly way to view the predictions and other relevant information. To start it, run:
```bash
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 5000
```
The interface includes the following pages:
-   **Predictions**: Displays the upcoming matches with their predicted scores, outcome probabilities, and confidence levels.
-   **League Table**: Shows the current 3. Liga league table.
-   **Statistics**: Provides an overview of the model's performance, including average points per match and accuracy.

## Advanced Options & Troubleshooting

### Configuration and paths
- Best parameters are read from `config/best_params.yaml` or `config/best_params.json` (auto-loaded by the `HybridPredictor`).
- Models are saved/loaded under `data/models`.
- Predictions and performance logs are stored under `data/predictions`.

### Environment variables
- `OMP_NUM_THREADS`: caps threads used by XGBoost/BLAS (e.g. `export OMP_NUM_THREADS=4`).

### Common issues
- **"No trained models found"**: Run `python3 -m kicktipp_predictor train` first.
- **"No module named kicktipp_predictor"**: Run `python3 -m pip install -e .` or set `PYTHONPATH=$PWD/src`.
- **"Not enough historical data"**: The API may be unavailable; retry later or check connectivity.
- **Web UI not loading**: Verify port 5000 availability. Try `http://127.0.0.1:5000`.

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
This predictor is for entertainment purposes only. Football is unpredicted and no model can guarantee accurate predictions.
