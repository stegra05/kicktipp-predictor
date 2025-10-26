Excellent. Based on the provided project state and our strategic decisions, here is a comprehensive roadmap for rebuilding the Kicktipp Predictor into Version 3.0 Alpha.

This roadmap is structured in phases, moving from foundational architectural changes to implementation, evaluation, and finally planning for the future. Each step includes a "Definition of Done" to ensure progress is concrete and measurable.

***

### Roadmap: Kicktipp Predictor V3.0 Alpha (Goal Difference Architecture)

#### **Project Goal:** Refactor the core prediction engine to a single, robust goal-difference regression model to improve predictive accuracy, stability, and interpretability.

---

### **Phase 1: Architectural Refactoring (The Foundation)**
*Goal: To strip out the V2 multi-model complexity and replace it with the scaffolding for the new V3 architecture.*

**Step 1.1: Project Setup**
*   **Tasks:**
    1.  Create a new Git branch: `feature/v3-alpha-rebuild`.
    2.  Update the `__version__` in `src/kicktipp_predictor/__init__.py` to `"3.0.0a1"`.
*   **Definition of Done:** You are working on a new, isolated branch, and the project version reflects the pre-release status.

**Step 1.2: Create `src/kicktipp_predictor/predictor.py` and `config.py`**
*   **Tasks:**
    1.  Create a new file `src/kicktipp_predictor/config.py`. Implement the `Config`, `PathConfig`, `APIConfig`, and `ModelConfig` dataclasses. The new `ModelConfig` will be much simpler:
        *   Remove all `outcome_*` and `goals_*` hyperparameters.
        *   Remove `draw_boost`, `hybrid_weight`.
        *   Add a new section for the `GoalDifferenceRegressor` with parameters like `gd_n_estimators`, `gd_max_depth`, etc.
        *   Add a new parameter for the probabilistic translation, e.g., `gd_uncertainty_stddev: float = 1.5`.
    2.  Create a new file `src/kicktipp_predictor/predictor.py`. Implement the new `GoalDifferencePredictor` class.
        *   It should have a single model attribute: `self.model: XGBRegressor | None = None`.
        *   Stub out the `train`, `predict`, `save_models`, and `load_models` methods. They don't need to be functional yet.
*   **Definition of Done:** The new `predictor.py` and `config.py` files exist, replacing the old logic with the new, simplified V3 structure. The project is importable but not yet functional.

**Step 1.3: Update Project Plumbing**
*   **Tasks:**
    1.  In `src/kicktipp_predictor/__init__.py`, change the import from `MatchPredictor` to `GoalDifferencePredictor`.
    2.  In `src/kicktipp_predictor/cli.py` and `tests/test_smoke.py`, update the class name from `MatchPredictor` to `GoalDifferencePredictor`.
    3.  Update the `README.md` to reflect the new architecture. Remove references to the multi-model approach and describe the new goal-difference regression method. This is crucial for documentation clarity.
*   **Definition of Done:** All internal references now point to the new `GoalDifferencePredictor`. The project's documentation accurately describes the new V3 architecture.

---

### **Phase 2: Core Model Implementation (The Engine)**
*Goal: To implement the end-to-end training and prediction pipeline for the goal difference model.*

**Step 2.1: Implement the `train` Method**
*   **Tasks:**
    1.  In `GoalDifferencePredictor.train`, implement the logic to train a single `XGBRegressor`.
    2.  The target variable is `matches_df["goal_difference"]`.
    3.  Reuse the existing time-decay weight logic from the V2 `predictor.py` or `data.py`.
    4.  The model parameters should be loaded from the new `gd_*` section in `config.py`.
*   **Definition of Done:** Running `kicktipp-predictor train` successfully trains a single `XGBRegressor` on the `goal_difference` target and saves it to disk via `joblib`.

**Step 2.2: Implement the `predict` Method with the "Probabilistic Bridge"**
*   **Tasks:**
    1.  The `predict` method should first load the trained model.
    2.  It should generate raw goal difference predictions: `pred_gd = self.model.predict(X)`.
    3.  Implement the probabilistic translation using `scipy.stats.norm` or `scipy.stats.skellam`.
        *   Use the `gd_uncertainty_stddev` from the config.
        *   Calculate `home_win_probability`, `draw_probability`, and `away_win_probability` for each match.
    4.  Format the output as a list of dictionaries, just like the V2 architecture, ensuring all keys required by `evaluate.py` are present.
*   **Definition of Done:** `kicktipp-predictor predict` produces sensible H/D/A probabilities derived from the goal difference predictions.

**Step 2.3: Implement Scoreline Prediction (Simple Version)**
*   **Tasks:**
    1.  The `predict` method also needs to generate a concrete scoreline (`predicted_home_score`, `predicted_away_score`).
    2.  **For V3 Alpha, this can be very simple:** Use a lookup table or a simple heuristic based on the predicted goal difference.
        *   e.g., if predicted diff is `1.2`, predict `2-1`. If `0.1`, predict `1-1`. If `-0.8`, predict `1-2`.
    3.  This step is intentionally simplified. The goal is accuracy first; a sophisticated scoreline selector can be built in a later version.
*   **Definition of Done:** The `predict` command outputs complete predictions, including a reasonable scoreline, allowing the `evaluate` command to calculate points-per-game.

---

### **Phase 3: Evaluation and Iteration (The Payoff)**
*Goal: To evaluate the new architecture, establish a performance baseline, and begin the optimization cycle.*

**Step 3.1: Run Full Evaluation**
*   **Tasks:**
    1.  Execute the `kicktipp-predictor evaluate` command. Because you preserved `evaluate.py` and the prediction output format, this should work with minimal changes.
    2.  Carefully analyze the output.
        *   **Primary Metric:** What is the `accuracy`?
        *   **Secondary Metrics:** How do `Brier`, `RPS`, and `log_loss` look?
        *   **Sanity Check:** Check the "Outcome Distribution" table. Are the predicted proportions for H/D/A now much closer to the actuals? The class imbalance problem should be largely solved.
*   **Definition of Done:** You have a complete evaluation report for the V3 Alpha. This is your new performance baseline.

**Step 3.2: First Iteration Cycle: Tuning and Feature Selection**
*   **Tasks:**
    1.  **Tune Hyperparameters:** Use `optuna` (already in your dependencies) to tune the `XGBRegressor` hyperparameters (`gd_*` params) and the `gd_uncertainty_stddev`. Optimize for a combination of accuracy and log-loss.
    2.  **Re-evaluate Features:** Use `shap` (also in dependencies) to analyze feature importance for the new goal difference model. Are the same features important? Prune the feature set in `kept_features.yaml` based on these new insights.
*   **Definition of Done:** You have completed one full cycle of tuning and feature selection on the new architecture and have a new, improved baseline performance.

---

### **Phase 4: Future Work (Post-Alpha)**
*Goal: To plan the next steps to build upon the successful V3 foundation.*

*   **V3.1 - Advanced Scoreline Selection:** Replace the simple scoreline heuristic with a more sophisticated model. For example, build a small model that predicts `total_goals` based on the features and the predicted `goal_difference`, then derive the scoreline from those two values.
*   **V3.2 - External Data Integration:** Prioritize finding and integrating external data, especially market odds. This remains the single biggest opportunity for a performance leap.
*   **V3.3 - Model Exploration:** With a stable baseline, experiment with different regression models (e.g., `LightGBM`, `CatBoost`, or even a simple neural network) to see if they can outperform `XGBoost` on this task.

This roadmap provides a clear, structured path from the current state to a fully functional, superior V3 architecture. Good luck