# 3. Liga Match Predictor

**Version 2.0** - A clean, transparent machine learning predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

> What's New: Dynamic season evaluation only, with a richer console report powered by `rich`. Plots are no longer produced by evaluation; artifacts are concise JSON/CSV.

## Architecture: The Predictor-Selector Model

Version 2.0 introduces a clear two-step prediction system:

1. Outcome Prediction (The Selector): A dedicated XGBoost classifier determines the match result (Home Win, Draw, or Away Win)
2. Scoreline Selection (The Predictor): XGBoost regressors predict expected goals, then a Poisson grid selects the most probable scoreline matching the predicted outcome

Outcome probabilities used for evaluation can be sourced from the classifier, derived from the goal Poisson model, or blended via a simple hybrid weight.

## Features
- Clear two-step architecture: outcome first, then scoreline
- Selectable outcome probabilities: classifier, Poisson-derived, or hybrid
- Advanced feature engineering: 80+ features (momentum, schedule strength, rest, goal patterns)
- EWMA recency features: leakage-safe exponentially weighted moving averages
- Comprehensive evaluation: Brier, Log Loss, RPS, accuracy, Kicktipp points
- Performance tracking and a simple Flask web UI

## Getting Started

### Installation
```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
python3 -m pip install -e .
```

### CLI Quickstart
```bash
# Show help
python3 -m kicktipp_predictor --help

# Train models (uses last 3 seasons by default)
python3 -m kicktipp_predictor train

# Predict upcoming matches (next 7 days)
python3 -m kicktipp_predictor predict --days 7

# Evaluate performance across current season (dynamic, expanding window)
python3 -m kicktipp_predictor evaluate --retrain-every 1

# Run web UI (defaults to 127.0.0.1:8000)
python3 -m kicktipp_predictor web --host 0.0.0.0 --port 8000
```

## CLI Commands
- `train [--seasons-back N]`: Train models on historical data (default: 3 seasons)
- `predict [--days N | --matchday N]`: Generate predictions for upcoming matches
- `evaluate [--retrain-every N] [prob options]`: Dynamic season evaluation (expanding window)
  - Probability options: `--prob-source classifier|poisson|hybrid`, `--hybrid-poisson-weight`, `--proba-grid-max-goals`, `--poisson-draw-rho`
- `web [--host HOST] [--port PORT]`: Run the Flask web UI
- `tune [options]`: Hyperparameter tuning (wrapper around `experiments/auto_tune.py`)

## Season Evaluation (Dynamic Only)

The `evaluate` command now uses a single dynamic, expanding-window procedure:
- Retrains every N matchdays (default: 1) on all historical + season-to-date matches
- Evaluates finished matchdays in the current season
- Writes:
  - `data/predictions/metrics_season.json`
  - `data/predictions/per_matchday_metrics_season.csv`

### Rich Console Report
The console output includes:
- Season metrics panel: matches, avg/total points, accuracy, Brier, Log Loss, RPS
- Prediction quality: exact scores, correct goal difference, correct result
- Baseline comparison (always 2-1 home): average points and accuracy deltas
- Outcome distribution: actual vs predicted H/D/A counts and percents
- Points distribution: counts for 0/2/3/4 points
- Top scorelines: top 5 predicted vs top 5 actual
- Confidence buckets: count, avg points, accuracy, avg confidence (numeric only)
- Confusion matrix and per-class precision/recall
- Per-matchday summary table: n, average points, baseline, delta, accuracy, exact/diff/result

Plots (calibration curves, confusion PNGs) are no longer produced in evaluation; use the CSV/JSON for downstream analysis or the web UI for visualization.

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
