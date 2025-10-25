# Kicktipp Predictor

Hey there! This is my little side project while juggling studies – a football match predictor for the German 3. Liga. I built it to dominate my Kicktipp games with friends using some machine learning wizardry: an XGBoost outcome classifier, goal regressors, and Poisson scoreline selection. It comes with a handy CLI, caching, evaluation tools, and an API. :)

## Features
- Train outcome and goal models from OpenLigaDB season data
- Predict upcoming matches with scorelines and H/D/A probabilities
- Hybrid probability blending: classifier + Poisson with fixed weight
- Configurable feature selection via `kept_features.yaml`
- Season-long dynamic evaluation with expanding-window retraining
- Simple persistence: `data/models/*.joblib` + `metadata.joblib` ^_^

## Installation
- Requirements: Python `>=3.10`
- Create and activate a virtual environment, then install:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  # Optional extras
  pip install -e .[dev]        # ruff, mypy, pre-commit
  pip install -e .[tuning]     # optuna
  pip install -e .[plots]      # shap
  ```
- The CLI entrypoint is installed as `kicktipp-predictor`. You can also run:
  ```bash
  python -m kicktipp_predictor --help
  ```

## Quickstart (CLI)
- Show help:
  ```bash
  kicktipp-predictor --help
  ```
- Train models (past seasons window):
  ```bash
  kicktipp-predictor train --seasons-back 5
  ```
- Predict upcoming matches (next N days or a specific matchday):
  ```bash
  # Next 7 days
  kicktipp-predictor predict --days 7
  
  # Or predict a specific matchday
  kicktipp-predictor predict --matchday 12
  ```
  Notes:
  - Outcome probabilities use hybrid blending (classifier + Poisson) with a fixed internal weight.
  - Scorelines are selected via Expected Points (EP) maximization under a Poisson grid.
- Evaluate a current season with expanding-window retraining:
  ```bash
  kicktipp-predictor evaluate --retrain-every 1
  ```
  Outputs season metrics and per-matchday breakdowns to `data/predictions/`.

## Configuration
Configuration is centralized in `src/kicktipp_predictor/config.py` and loaded optionally from YAML.

- YAML location: by default `config/best_params.yaml` at project root. Override via env var:
  ```bash
  export KTP_CONFIG_FILE=path/to/your_params.yaml
  ```
- Paths (`PathConfig`):
  - `data_dir`: `data/`
  - `models_dir`: `data/models/`
  - `cache_dir`: `data/cache/`
  - `config_dir`: `config/` (create this directory at project root to use YAMLs)
- API (`APIConfig`): `base_url`, `league_code` (default `bl3`), `cache_ttl`, `request_timeout`.
- Model (`ModelConfig`): key options (see code for full list)
  - Outcome classifier: `outcome_n_estimators`, `outcome_max_depth`, `outcome_learning_rate`, `outcome_subsample`, `outcome_reg_lambda`, `outcome_min_child_weight`
  - Goal regressors: `goals_n_estimators`, `goals_max_depth`, `goals_learning_rate`, `goals_subsample`, `goals_reg_lambda`, `goals_min_child_weight`, `goals_early_stopping_rounds`
  - Training: `random_state`, `val_fraction`, `min_training_matches`
  - Feature/recency: `use_time_decay`, `time_decay_half_life_days`, `form_last_n`
  - Probability config: `draw_boost` (training class weight), `proba_temperature` (reserved)
  - Scoreline selection: `max_goals`
  - Feature selection file: `selected_features_file` (default `kept_features.yaml`)

Example YAML (`config/best_params.yaml`):
```yaml
# Training and features
use_time_decay: true
time_decay_half_life_days: 330.0
form_last_n: 5

# Outcome classifier
outcome_n_estimators: 1150
outcome_max_depth: 6
outcome_learning_rate: 0.1
outcome_subsample: 0.8
outcome_reg_lambda: 1.0
outcome_min_child_weight: 0.1587

# Goal regressors
goals_n_estimators: 800
goals_max_depth: 6
goals_learning_rate: 0.1
goals_subsample: 0.8
goals_reg_lambda: 1.0
goals_min_child_weight: 1.6919

# Scorelines
max_goals: 8

# Class weighting & temperature
draw_boost: 1.7
proba_temperature: 1.0
```

Feature selection (`kept_features.yaml`): place this file under `config/` at the project root. If present, only essential columns and the listed features are kept for training and prediction feature frames.

## Programmatic API
Train, save, load, and predict in Python:
```python
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor
from kicktipp_predictor.config import get_config

# Optional: tweak config in code
cfg = get_config()

loader = DataLoader()
current = loader.get_current_season()
all_matches = loader.fetch_historical_seasons(current - 5, current)
train_df = loader.create_features_from_matches(all_matches)

predictor = MatchPredictor()
predictor.train(train_df)
predictor.save_models()

# Later: load and predict upcoming
predictor.load_models()
upcoming = loader.get_upcoming_matches(days=7)
historical = loader.fetch_season_matches(current)
features = loader.create_prediction_features(upcoming, historical)
preds = predictor.predict(features)
for p in preds:
    print(p["home_team"], "vs", p["away_team"], p["predicted_home_score"], "-", p["predicted_away_score"])  # noqa: E501
```

## Evaluation Outputs
The `evaluate` command writes season artifacts to `data/predictions/`:
- `metrics_season.json`: overall metrics (Brier, log loss, RPS, accuracy, points, bootstrap CI)
- `per_matchday_metrics_season.csv`: per-matchday breakdown
- `blend_debug.csv`: optional diagnostics if available (classifier vs. Poisson probs and blend weight)

## Web Command (optional)
A `web` command exists in the CLI and expects a Flask app at `kicktipp_predictor.web.app`. If this module is not present in your local checkout, running `kicktipp-predictor web` will fail. Some distributions may include the subpackage.

## Paths & Artifacts
- Models are saved under `data/models/`:
  - `outcome_classifier.joblib`, `home_goals_regressor.joblib`, `away_goals_regressor.joblib`
  - `metadata.joblib` (feature columns, label encoder)
- API cache uses `data/cache/`.

## Contributing
As a student side project, I'm thrilled if anyone wants to contribute! 
- Setup:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[dev]
  pre-commit install
  ```
- Run checks:
  ```bash
  ruff check .
  ruff format .
  pytest -q
  ```
- Guidelines:
  - Keep changes focused and minimal
  - Match existing style; prefer small, composable functions
  - Avoid unrelated fixes in PRs; mention known issues separately

I'm super thankful for every bit of time you spend looking at or improving this project – it means the world to me! :)

## Troubleshooting
- "No trained models found": run `kicktipp-predictor train` first.
- "Could not create features": early-season data may be insufficient; try a later matchday.
- OpenLigaDB errors/timeouts: retry later; cached data may be used if available.
- `web` import error: ensure the `kicktipp_predictor.web` subpackage exists; not all checkouts include it.
- Evaluate probability options: the `evaluate` command currently uses config defaults; adjust via YAML or programmatic config before running.

## License
Licensed under the MIT License. See `LICENSE` for details.

## Thanks!
A huge shoutout to everyone who checks out this project. Your interest keeps me motivated during late-night coding sessions. If it helps you win your Kicktipp group, drop me a line – I'd love to hear! :)