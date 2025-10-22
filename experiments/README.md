# Experiments

This directory contains experimental scripts for hyperparameter tuning and model optimization.

## Scripts

### `auto_tune.py`

Optuna-based hyperparameter tuning for the MatchPredictor.

**Usage:**
```bash
python experiments/auto_tune.py --trials 100 --objective avg_points
```

**Features:**
- Multi-objective optimization
- Cross-validation
- Comparison mode for evaluating multiple objectives
- Parallel execution support

### `feature_ablation.py` ⭐ NEW

Automated feature ablation study to identify optimal feature subsets.

**Usage:**
```bash
# Run full study
python experiments/feature_ablation.py

# Or use the helper script
./run_feature_study.sh
```

**What it does:**
1. **Baseline**: Tests current feature set (62 features)
2. **Category Ablation**: Removes each feature category and measures impact
3. **Percentage Pruning**: Tests removing 10%, 20%, 30%, 40% of features
4. **Minimal Core**: Tests a minimal feature set (~30-40 features)

**Output:**
- `data/feature_ablation/ablation_results.csv` - Full results
- Console report with recommendations
- Identifies low-impact features safe to remove

**Expected Runtime:** ~15-30 minutes (depends on dataset size)

---

## Feature Ablation Study - What to Expect

### Phase 1: Baseline
Tests current 62-feature setup to establish benchmark.

### Phase 2: Category Ablation
Tests impact of removing entire feature categories:
- **Interaction ratios** (6 features): attack/defense ratios, form ratios, etc.
- **Venue-specific** (6 features): home/away performance metrics
- **EWMA recency** (8 features): exponentially weighted moving averages
- **Momentum** (6 features): momentum-based features
- **Form** (18 features): last-N-games form metrics
- **Base stats** (12 features): long-term averages
- **Derived diffs** (4 features): difference features

**Key Question:** Which categories can be removed with minimal performance loss?

### Phase 3: Percentage Pruning
Simulates importance-based feature selection by removing:
- 10% (6 features) - removes interaction ratios
- 20% (12 features) - adds venue-specific
- 30% (19 features) - adds some EWMA
- 40% (25 features) - more aggressive pruning

**Key Question:** What's the sweet spot between complexity and performance?

### Phase 4: Minimal Core
Tests a hand-selected minimal set of ~30-40 most essential features.

**Key Question:** How much can we simplify while staying competitive?

---

## Interpreting Results

### Success Criteria

A simplified feature set is **successful** if:
- `avg_points` drop < 0.10 (less than 1 point per 10 matches)
- `accuracy` drop < 2%
- `brier_score` increase < 0.02

### Reading the Report

**Category Impact:**
- 🟢 **Low impact** (< 0.05 pts): Safe to remove
- 🟡 **Medium impact** (0.05-0.15 pts): Consider for simplification
- 🔴 **High impact** (> 0.15 pts): Keep these features

**Recommended Actions:**

Example output:
```
🎯 Best Simplification (<0.1 pts loss):
   prune_20pct: 50 features (-19%)
   Avg Points: 1.895 (-0.055)
   Accuracy: 0.485 (-0.015)
```

This means: **Removing 20% of features (12 features) only costs 0.055 points/match**

---

## Next Steps After Study

### 1. Review Results
```bash
cat data/feature_ablation/ablation_results.csv
```

### 2. Identify Best Configuration
Look for the configuration with:
- Fewest features
- Performance drop < 0.10 avg_points

### 3. Apply the Simplification

If study recommends removing interaction ratios and venue features:

```bash
# Edit kept_features.yaml - remove recommended features
nano data/feature_selection/kept_features.yaml

# Retrain with new feature set
python -m kicktipp_predictor train

# Validate
python -m kicktipp_predictor evaluate --season
```

### 4. Monitor Production Performance

After deploying simplified model, track:
- Points per matchday
- Prediction accuracy
- Brier score trends

If performance degrades, rollback to previous feature set.

---

## Tips for Best Results

1. **Run during off-hours**: Study takes 15-30 minutes
2. **Clean cache first**: Ensure fresh data
   ```bash
   rm -rf data/cache/*
   ```
3. **Check data quality**: Verify recent seasons loaded
   ```bash
   python -m kicktipp_predictor train --verbose
   ```
4. **Multiple runs**: Run study 2-3 times to verify consistency

---

## Example Session

```bash
# 1. Run study
python experiments/feature_ablation.py

# Expected output:
# 📊 Loading data...
#    Train: 1247 samples
#    Test:  534 samples
#
# ================================================================================
# PHASE 1: BASELINE
# ================================================================================
# Experiment: baseline_all_features
# Features: 62
# ✅ Results:
#    Accuracy:    0.485
#    Avg Points:  1.950
#    ...
#
# [... more phases ...]
#
# ================================================================================
# FEATURE ABLATION STUDY - SUMMARY REPORT
# ================================================================================
# 
# TOP CONFIGURATIONS (by avg_points):
# Rank   Experiment                     Features   Δ Pts      Δ Acc      Brier     
# --------------------------------------------------------------------------------
# 1      baseline_all_features          62         +0.000     +0.000     0.5234
# 2      ablate_interaction_ratios      56         -0.035     -0.010     0.5256
# 3      prune_10pct                    56         -0.042     -0.012     0.5261
# ...
#
# RECOMMENDED CONFIGURATIONS:
#
# 🎯 Best Simplification (<0.1 pts loss):
#    ablate_interaction_ratios: 56 features (-10%)
#    Avg Points: 1.915 (-0.035)
#    Accuracy: 0.475 (-0.010)
#
# 📈 CATEGORY ABLATION INSIGHTS:
# Category                  Impact          Features   Δ Pts     
# ----------------------------------------------------------------------
# form                      🔴 High         18         -0.234
# base_stats                🔴 High         12         -0.189
# momentum                  🟡 Medium       6          -0.098
# ewma_recency              🟡 Medium       8          -0.087
# derived_diffs             🟢 Low          4          -0.045
# venue_specific            🟢 Low          6          -0.038
# interaction_ratios        🟢 Low          6          -0.035
#
# ✅ Safe to remove (< 0.05 pts impact):
#    - derived_diffs (4 features)
#    - venue_specific (6 features)
#    - interaction_ratios (6 features)

# 2. Review CSV for details
cat data/feature_ablation/ablation_results.csv

# 3. Apply recommendation - remove 16 features (ratios + venue + diffs)
# Edit data/feature_selection/kept_features.yaml
# Remove the 16 low-impact features

# 4. Retrain and validate
python -m kicktipp_predictor train
python -m kicktipp_predictor evaluate --season
```

---

## Troubleshooting

**Study crashes during training:**
- Reduce dataset size in script (edit `start_season`)
- Check memory usage
- Verify all dependencies installed

**Inconsistent results:**
- Run multiple times (random seed varies)
- Check for data quality issues
- Verify cache is fresh

**All experiments fail:**
- Check that base model trains successfully
- Verify feature names match exactly
- Review error messages in output

---

## Advanced: Custom Experiments

### Test Specific Feature Combinations

Edit `feature_ablation.py` and add custom experiment:

```python
def run_custom_test(self, train_df, test_df):
    # Your custom feature list
    custom_features = [
        'home_avg_points', 'away_avg_points',
        'home_form_points', 'away_form_points',
        # ... add features
    ]
    
    return self._train_and_evaluate(
        custom_features,
        train_df,
        test_df,
        "custom_experiment"
    )
```

Then call in `run_full_study()`.

---

## Questions?

See `FEATURE_COMPLEXITY_ANALYSIS.md` for detailed feature documentation.

