Of course. Here is a detailed, step-by-step implementation plan for the V4.0 "Cascaded Predictor" architecture. This will serve as a comprehensive technical reference for the rebuild.

***

### Architectural Overview and Key Components (V4)

**Predictor Core:** `CascadedPredictor`

- **Draw Classifier (Gatekeeper):** Binary classifier (`Draw` vs `NotDraw`) trained on all matches.
- **Win Classifier (Finisher):** Binary classifier (`HomeWin` vs `AwayWin`) trained on the non-draw subset.
- **Feature Pipeline:** Reuse `data.py` engineered features (Elo, form, schedule/context).
- **Evaluation:** Expanding-window dynamic season evaluation in `evaluate.py` with rich console reporting.
- **CLI:** `train`, `predict`, `evaluate`, and `tune` commands in `cli.py`.

### Data Flow Diagram

```
            +----------------------+
            |  Features (data.py)  |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |  Draw Classifier     |
            |  P(Draw), P(NotDraw) |
            +----------+-----------+
                       |
          if NotDraw   |
                       v
            +---------------------------+
            |  Win Classifier           |
            |  P(Home|NotDraw), P(Away) |
            +------------+--------------+
                         |
                         v
           +-------------------------------+
           |  Combine Probabilities         |
           |  P(H) = P(ND)*P(H|ND)         |
           |  P(A) = P(ND)*(1-P(H|ND))     |
           |  P(D) = P(Draw)               |
           +-------------------------------+
                         |
                         v
           +-------------------------------+
           |  Normalize & Scoreline Heur.  |
           +-------------------------------+
```

### Performance Expectations

- **Draw Rate:** 18–28% predicted draws under season evaluation.
- **Accuracy:** Higher than V3 goal-difference baseline while maintaining draw realism.
- **Calibration:** Improved `log_loss`, `Brier`, and `RPS` due to explicit draw modeling.
- **Stability:** Consistent across matchdays with expanding-window retraining.

### Migration Considerations from V3

- **Predictor Class:** Replace `GoalDifferencePredictor` with `CascadedPredictor`.
- **Configuration:** Deprecate `gd_*`. Add `draw_*` and `win_*` with `draw_params` and `win_params` helpers.
- **Artifacts:** Save `draw_classifier.joblib`, `win_classifier.joblib`, and `metadata_v4.joblib`.
- **CLI:** Keep `train`, `predict`, `evaluate`, `tune`. Update help/docs to mention cascaded architecture while retaining discoverability.
- **Tests:** Update smoke tests to import `CascadedPredictor` and ensure CLI help remains functional.

***

### Technical Blueprint: V4.0 Cascaded Predictor

#### **Phase 1: Project Scaffolding and Configuration**

**Goal:** Set up the new `CascadedPredictor` class and adapt the configuration to support two distinct models.

**Step 1.1: Branch and Versioning**
*   **Action:** Create a new Git branch: `feature/v4-cascaded-model`.
*   **Action:** Update `__version__` in `src/kicktipp_predictor/__init__.py` to `"4.0.0a1"`.

**Step 1.2: Refactor `config.py`**
*   **Action:** In the `ModelConfig` dataclass, remove the `gd_*` parameters.
*   **Action:** Add two new sections of hyperparameters, one for each model. The names should be explicit.

```python
# In src/kicktipp_predictor/config.py within ModelConfig

# --- V4 Cascaded Model: Draw Classifier ---
draw_n_estimators: int = 400
draw_max_depth: int = 5
draw_learning_rate: float = 0.05
draw_subsample: float = 0.7
draw_colsample_bytree: float = 0.7
# This is crucial for imbalanced binary classification
draw_scale_pos_weight: float = 3.0 # (Roughly num_not_draw / num_draw)

# --- V4 Cascaded Model: Win Classifier (H vs A) ---
win_n_estimators: int = 800
win_max_depth: int = 6
win_learning_rate: float = 0.1
win_subsample: float = 0.8
win_colsample_bytree: float = 0.8

# Action: Add two new properties to ModelConfig to easily get these params
@property
def draw_params(self) -> dict[str, Any]:
    return {
        "n_estimators": self.draw_n_estimators,
        "max_depth": self.draw_max_depth,
        # ... and so on for all draw_* params
        "scale_pos_weight": self.draw_scale_pos_weight,
        "random_state": self.random_state,
        "n_jobs": self.n_jobs,
    }

@property
def win_params(self) -> dict[str, Any]:
    return {
        "n_estimators": self.win_n_estimators,
        "max_depth": self.win_max_depth,
        # ... and so on for all win_* params
        "random_state": self.random_state,
        "n_jobs": self.n_jobs,
    }
```

**Step 1.3: Create the New `predictor.py`**
*   **Action:** Create a new class `CascadedPredictor`.
*   **Action:** Define its attributes. It will manage two models and two label encoders.

```python
# In src/kicktipp_predictor/predictor.py

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class CascadedPredictor:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.draw_model: XGBClassifier | None = None
        self.win_model: XGBClassifier | None = None
        self.feature_columns: list[str] = []
        
        # Encoder for the draw model: 0=NotDraw, 1=Draw
        self.draw_label_encoder = LabelEncoder()
        
        # Encoder for the win model: 0=AwayWin, 1=HomeWin
        self.win_label_encoder = LabelEncoder()

    # Stub out train, predict, save_models, load_models methods for now
```

---

#### **Phase 2: Core Logic Implementation**

**Goal:** Implement the data preparation, training, and prediction pipeline for the two-stage model.

**Step 2.1: Prepare Data with New Targets**
*   **This logic can live inside the `train` method.** It doesn't require changes to `data.py`.

```python
# Inside CascadedPredictor.train(self, matches_df: pd.DataFrame)

# 1. Prepare Draw Model Target
# The target is binary: 1 if the match was a draw, 0 otherwise.
matches_df['is_draw'] = (matches_df['result'] == 'D').astype(int)
y_draw = matches_df['is_draw']
# Fit the encoder. It should map {'NotDraw': 0, 'Draw': 1} or similar
self.draw_label_encoder.fit([0, 1]) # Simple binary case

# 2. Prepare Win Model Data (Subset and Target)
# Filter out all draw matches for the second model's training set
non_draw_df = matches_df[matches_df['is_draw'] == 0].copy()

# The target is binary: 1 for Home Win, 0 for Away Win
# This avoids needing a separate LabelEncoder if we stick to 0/1
non_draw_df['is_home_win'] = (non_draw_df['result'] == 'H').astype(int)
y_win = non_draw_df['is_home_win']
self.win_label_encoder.fit(['A', 'H']) # Explicitly fit for clarity

# Prepare feature matrices for both
X_all = matches_df[self.feature_columns]
X_non_draw = non_draw_df[self.feature_columns]
```
*   **Note:** You'll need to define `self.feature_columns` before this, using the same logic as before to select numeric columns from `matches_df`.

**Step 2.2: Implement the `train` Method**
*   **Action:** Flesh out the training logic using the data prepared above.

```python
# Continuing inside CascadedPredictor.train()

# 3. Train Draw Model
print("Training Draw Classifier (Gatekeeper)...")
self.draw_model = XGBClassifier(**self.config.model.draw_params)
self.draw_model.fit(X_all, y_draw) # Use all data

# 4. Train Win Model
print("Training Win Classifier (Finisher)...")
self.win_model = XGBClassifier(**self.config.model.win_params)
self.win_model.fit(X_non_draw, y_win) # Use only non-draw data
```

**Step 2.3: Implement the `predict` Method**
*   **Action:** This is the most critical part. Implement the two-stage prediction and probability combination.

```python
# Inside CascadedPredictor.predict(self, features_df: pd.DataFrame)

# 1. Prepare feature matrix X (same as before)
X = features_df.reindex(columns=self.feature_columns).fillna(0.0)

# 2. Get probabilities from Draw Model
# predict_proba returns shape (n_samples, n_classes), e.g., [[P(NotDraw), P(Draw)]]
draw_probs = self.draw_model.predict_proba(X)
# Get the index of the "Draw" class (should be 1)
draw_class_idx = self.draw_label_encoder.transform([1])[0] 
p_draw = draw_probs[:, draw_class_idx]

# 3. Get probabilities from Win Model
# predict_proba returns shape (n_samples, n_classes), e.g., [[P(Away), P(Home)]]
win_probs = self.win_model.predict_proba(X)
# Get the index of the "Home Win" class
home_win_class_idx = self.win_label_encoder.transform(['H'])[0]
p_h_given_not_draw = win_probs[:, home_win_class_idx]

# 4. Combine probabilities using the law of total probability
p_not_draw = 1 - p_draw
final_p_home = p_not_draw * p_h_given_not_draw
final_p_away = p_not_draw * (1 - p_h_given_not_draw)
final_p_draw = p_draw

# 5. Assemble the final probability matrix [P(H), P(D), P(A)]
# Note the order matters for your evaluation suite!
final_probs = np.vstack([final_p_home, final_p_draw, final_p_away]).T
# Normalize to ensure rows sum to 1, correcting for any minor float precision errors
final_probs /= final_probs.sum(axis=1, keepdims=True)

# 6. Format the output dictionaries (reuse logic from V3)
# You'll also need a simple scoreline heuristic again for PPG calculation.
# ... format predictions into a list of dicts ...
return predictions
```

---

#### **Phase 3: Persistence and Tooling**

**Goal:** Ensure the new two-model architecture can be saved, loaded, and tuned correctly.

**Step 3.1: Implement `save_models` and `load_models`**
*   **Action:** These methods now need to handle two model objects and their metadata.

```python
# In CascadedPredictor

def save_models(self):
    # Save draw model
    joblib.dump(self.draw_model, self.config.paths.models_dir / "draw_classifier.joblib")
    # Save win model
    joblib.dump(self.win_model, self.config.paths.models_dir / "win_classifier.joblib")
    
    metadata = {
        "feature_columns": self.feature_columns,
        "draw_label_encoder": self.draw_label_encoder,
        "win_label_encoder": self.win_label_encoder,
    }
    joblib.dump(metadata, self.config.paths.models_dir / "metadata_v4.joblib")

def load_models(self):
    self.draw_model = joblib.load(self.config.paths.models_dir / "draw_classifier.joblib")
    self.win_model = joblib.load(self.config.paths.models_dir / "win_classifier.joblib")
    
    metadata = joblib.load(self.config.paths.models_dir / "metadata_v4.joblib")
    self.feature_columns = metadata["feature_columns"]
    self.draw_label_encoder = metadata["draw_label_encoder"]
    self.win_label_encoder = metadata["win_label_encoder"]
```

**Step 3.2: Update `tune.py`**
*   **Action:** The tuning process becomes more complex. The simplest approach is to tune the models sequentially.
*   **Recommended Tuning Strategy:**
    1.  **Tune Draw Model:** Create an Optuna study that calls a modified `objective` function. This function only trains and evaluates the `draw_model`. The metric to optimize could be `roc_auc` or `f1_score` for the draw class. Run this study and save the best `draw_*` hyperparameters.
    2.  **Tune Win Model:** Create a second Optuna study. This one uses the *best fixed parameters* for the draw model. Its objective function trains both models but only tunes the `win_*` hyperparameters. The optimization metric can be the overall `accuracy` or `log_loss` on the final combined probabilities.
*   **Implementation:** You might need to add flags to your `tune` command, e.g., `kicktipp-predictor tune --model-to-tune=draw`.

---

#### **Phase 4: Evaluation and Iteration**

**Goal:** Establish a new baseline for the V4 architecture and analyze its performance.

**Step 4.1: Initial Run**
*   **Action:** Manually set some reasonable default hyperparameters in `config.py`.
*   **Action:** Run `kicktipp-predictor train`.
*   **Action:** Run `kicktipp-predictor evaluate`.

**Step 4.2: Critical Analysis**
*   **Check the "Outcome Distribution":** Is the predicted draw rate now reasonable (e.g., 15-30%)?
*   **Check the "Season Metrics":** What is the overall accuracy? Is it higher than the V3 model's 31.7%? Is it approaching the V3 "never-draw" model's 47.5%?
*   **Check the "Confusion Matrix":** How well is the `draw_model` performing? Look at the "D" column (Precision) and the "D" row (Recall). How well is the `win_model` doing at distinguishing H vs. A on the non-draws?

This detailed blueprint provides a clear and actionable path to implementing the V4 Cascaded Predictor. It directly addresses the shortcomings of the previous architectures by specializing the modeling tasks.