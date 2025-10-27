### Roadmap: Kicktipp Predictor V4.0 Alpha (Cascaded Architecture)

This roadmap outlines the phased delivery of the V4 cascaded predictor. Each phase includes a clear “Definition of Done.”

***

#### Project Goal
Deliver a robust two-stage classifier that produces calibrated H/D/A probabilities with realistic draw rates and improved accuracy over V3.

---

### Phase 1: Architectural Setup
*Goal: Establish the cascaded predictor and configuration scaffolding.*

**Tasks**
1. Create/confirm branch `feature/v4-cascaded-model`.
2. Update `__version__` in `src/kicktipp_predictor/__init__.py` to `"4.0.0a1"`.
3. Refactor `ModelConfig` (`config.py`): remove `gd_*`; add `draw_*` and `win_*` with `draw_params` and `win_params` helpers.
4. Implement `CascadedPredictor` class with `train`, `predict`, `save_models`, `load_models` stubs and feature preparation.
5. Update CLI imports and help/docs to reflect V4.

**Definition of Done**
* Package imports work; CLI exposes `train`, `predict`, `evaluate`, `tune`; predictor class ready for training.

---

### Phase 2: Core Logic Implementation
*Goal: Implement end-to-end training and prediction.*

**Tasks**
1. Implement draw target (`is_draw`) and win target (`is_home_win`) preparation inside `train`.
2. Train `draw_model` on all matches; train `win_model` on non-draw subset.
3. Implement combined probability computation (`P(H)`, `P(D)`, `P(A)`).
4. Add simple scoreline heuristic for PPG evaluation.

**Definition of Done**
* `train` saves both models; `predict` returns calibrated H/D/A probabilities and a reasonable scoreline.

---

### Phase 3: Persistence and Tuning
*Goal: Ensure models can be saved/loaded and tuned.*

**Tasks**
1. Save models as `draw_classifier.joblib` and `win_classifier.joblib` with `metadata_v4.joblib`.
2. Add tuning entry-points for sequential draw → win tuning; expose CLI `tune` options (`model_to_tune`, metrics, parallel).

**Definition of Done**
* Models load successfully; tuning runs and produces best-params artifacts.

---

### Phase 4: Evaluation Baseline
*Goal: Establish season performance under expanding-window retraining.*

**Tasks**
1. Run dynamic season evaluation; capture metrics (`accuracy`, `log_loss`, `RPS`, `Brier`).
2. Validate draw rate realism (target range 18–28%).

**Definition of Done**
* Season metrics demonstrate improved calibration and competitive accuracy versus V3.

---

### Phase 5: Backward Compatibility and Documentation
*Goal: Maintain compatibility where required and finalize docs.*

**Tasks**
1. Ensure CLI remains stable and discoverable; help text mentions V4 cascaded architecture but retains `Kicktipp Predictor CLI` phrase.
2. Update README and BLUEPRINT with architecture, data flow, performance expectations, and migration notes.

**Definition of Done**
* Documentation reflects V4; users can upgrade from V3 following migration guidance.

---

### Future Work (Post-Alpha)
* Advanced scoreline modeling (total-goals + difference).
* External data integration (market odds) for further gains.
* Model exploration (LightGBM, CatBoost, NN) on the cascaded formulation.