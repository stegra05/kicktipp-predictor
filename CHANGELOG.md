# Changelog

## 2025-10-24

- Update default hyperparameters in `ModelConfig` to tuned values:
  - `draw_boost`: 1.7
  - `hybrid_poisson_weight`: 0.0525
  - `time_decay_half_life_days`: 330.0
  - `momentum_decay`: 0.83
  - `outcome_n_estimators`: 1150
  - `outcome_gamma`: 4.57e-07
  - `outcome_min_child_weight`: 0.1587
  - `goals_min_child_weight`: 1.6919
- Documentation updated to match new defaults.
- Unit tests passed (8 tests), warnings unrelated to changes.
- CLI training smoke test succeeded using `--seasons-back 1`.