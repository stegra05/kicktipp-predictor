### Kicktipp Predictor V4: Cascaded Architecture Overview

This document describes the V4 Cascaded Architecture and serves as a user-facing overview of how the predictor works in this release. All content related to the former V3 goal-difference regressor has been removed.

**Installation**

Use a virtual environment and install via `pyproject.toml` extras:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[plots,dev]
```

- The `plots` extra includes `shap` for SHAP value analysis.
- The `dev` extra includes `ruff`, `mypy`, and `pre-commit`.
- For Optuna-based tuning, install the `tuning` extra: `pip install -e .[tuning]`.
- zsh users: quote extras to avoid globbing: `pip install -e '.[plots,dev]'`.

#### Executive Summary

V4 introduces a cascaded, two-stage classifier that models match outcomes more directly and robustly:

* Stage 1 (Gatekeeper): A binary classifier predicts `Draw` vs `NotDraw`.
* Stage 2 (Finisher): Conditioned on `NotDraw`, a binary classifier predicts `HomeWin` vs `AwayWin`.

Final probabilities are combined via the law of total probability to produce calibrated `H/D/A` probabilities. This design targets realistic draw rates while improving accuracy and interpretability versus prior versions.

---

#### 1. Architecture and Key Components (V4)

**Core Class:** `CascadedPredictor`

* **Draw Classifier (Gatekeeper):** `XGBClassifier` trained to predict `Draw` vs `NotDraw` on all matches.
* **Win Classifier (Finisher):** `XGBClassifier` trained to predict `HomeWin` vs `AwayWin` on the non-draw subset.
* **Feature Pipeline:** Reuses the feature engineering from `data.py` (Elo, form metrics, schedule context).
* **Evaluation:** `evaluate.py` runs dynamic season evaluation with expanding-window retraining and rich console reports.
* **CLI:** `cli.py` provides `train`, `predict`, `evaluate`, and `tune` commands.

---

#### 2. Data Flow (V4)

The cascaded pipeline transforms features into calibrated H/D/A probabilities:

```
Features (data.py) ──▶ Draw Classifier (P(Draw), P(NotDraw))
                         │
                         └── if NotDraw ──▶ Win Classifier (P(Home|NotDraw), P(Away|NotDraw))

Combine via law of total probability:
  P(H) = P(NotDraw) * P(Home|NotDraw)
  P(A) = P(NotDraw) * (1 - P(Home|NotDraw))
  P(D) = P(Draw)

Post-process:
  - Normalize rows to sum to 1
  - Simple scoreline heuristic for points-per-game
```

#### 3. Performance Expectations

While performance varies by league and season, V4 targets:

* **Realistic draw rate:** 18–28% predicted draws across evaluation periods.
* **Accuracy:** Higher than the V3 goal-difference baseline; approaching prior “never-draw” heuristic while keeping draw realism.
* **Log Loss / RPS / Brier:** Improved calibration due to explicit modeling of draw vs non-draw.
* **Stability:** Robust across matchdays under expanding-window retraining.

---

#### 4. Migration Considerations (V3 → V4)

* **Predictor class:** `GoalDifferencePredictor` replaced by `CascadedPredictor` in `predictor.py`.
* **Config:** `gd_*` parameters deprecated; new `draw_*` and `win_*` sections with `draw_params` and `win_params` helpers.
* **Persistence:** Models saved as `draw_classifier.joblib`, `win_classifier.joblib`, with `metadata_v4.joblib`.
* **CLI:** Commands unchanged (`train`, `predict`, `evaluate`, `tune`), but help/docs reflect cascaded behavior.
* **Evaluation:** Continues to consume `H/D/A` probabilities; improved calibration expected out-of-the-box.

---

#### 5. Getting Started

**Training**

```
python -m kicktipp_predictor.cli train --seasons-back 5
```

**Predicting Upcoming Matches**

```
python -m kicktipp_predictor.cli predict --days 7
```

**Evaluating Season Performance**

```
python -m kicktipp_predictor.cli evaluate --retrain-every 1
```

**Tuning (Optuna)**

```
python -m kicktipp_predictor.cli tune --n-trials 200 --seasons-back 5
```

For a deep technical reference, see `BLUEPRINT.md` (V4 Cascaded Predictor).