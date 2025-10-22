# Recommended Features to Remove

## Decision: Don't Trust the Ablation Results

The ablation study results are **invalid** because all models were severely undertrained (25.7% accuracy vs your model's 50.2%).

Instead, I'm recommending removals based on:
1. **Domain knowledge** - What tree models learn automatically
2. **Existing correlation analysis** - Already removed 7 highly correlated features
3. **Conservative approach** - Remove only obviously redundant features

---

## Phase 1: Remove Interaction Ratios (7 features)

### Why Remove These?

XGBoost **automatically learns feature interactions** through tree splits.

Example: The model can learn `attack_defense_form_ratio` by splitting on:
1. First split: `home_form_avg_goals_scored`
2. Second split: `away_form_avg_goals_conceded`
3. Result: Ratio learned implicitly

### Features to Remove:

```yaml
# These are all manually computed ratios:
- attack_defense_form_ratio_home
- attack_defense_form_ratio_away
- attack_defense_long_ratio_home
- attack_defense_long_ratio_away
- form_points_pg_ratio
- momentum_score_ratio
- ewm_points_ratio
```

### Expected Impact:

- **Risk:** Low
- **Expected loss:** < 0.05 avg_points
- **Benefit:** 61 → 54 features (11% reduction)

---

## Phase 2: If Phase 1 Succeeds, Remove Venue Deltas (3 features)

### Features to Remove:

```yaml
# These might be redundant with venue-specific per-game stats
- venue_points_delta
- venue_goals_delta
- venue_conceded_delta
```

Because we already have:
- `home_points_pg_at_home`, `away_points_pg_away` (same information)
- `home_goals_pg_at_home`, `away_goals_pg_away`
- `home_goals_conceded_pg_at_home`, `away_goals_conceded_pg_away`

### Expected Impact:

- **Risk:** Low-Medium
- **Expected loss:** 0.02-0.08 avg_points
- **Benefit:** 54 → 51 features (16% total reduction)

---

## Phase 3: If Still Good, Remove Absolute Differences (2 features)

### Features to Remove:

```yaml
# Keep the signed versions, remove absolute
- abs_form_points_diff
- abs_momentum_score_diff
```

Keep: `form_points_difference`, `momentum_score_difference`

The model can learn "magnitude" from signed differences.

### Expected Impact:

- **Risk:** Medium
- **Expected loss:** 0.05-0.10 avg_points
- **Benefit:** 51 → 49 features (20% total reduction)

---

## Implementation Plan

### Step 1: Backup

```bash
cd /Users/stef/Documents/Programmieren/unfinished/kicktipp-predictor
cp data/feature_selection/kept_features.yaml data/feature_selection/kept_features_backup.yaml
```

### Step 2: Edit Feature List (Phase 1)

```bash
nano data/feature_selection/kept_features.yaml
```

Remove these 7 lines:
```
- attack_defense_form_ratio_home
- attack_defense_form_ratio_away
- attack_defense_long_ratio_home
- attack_defense_long_ratio_away
- form_points_pg_ratio
- momentum_score_ratio
- ewm_points_ratio
```

### Step 3: Retrain

```bash
python -m kicktipp_predictor train
```

### Step 4: Evaluate

```bash
python -m kicktipp_predictor evaluate --season > phase1_results.txt
cat phase1_results.txt | grep "Avg Points"
```

Compare to your current model:
- **Current:** ~1.38 avg_points (from earlier eval)
- **Target:** > 1.33 avg_points (< 0.05 drop)

### Step 5: Decision

**If avg_points > 1.33:**
✅ Success! Keep the changes and proceed to Phase 2

**If avg_points < 1.33:**
❌ Rollback:
```bash
cp data/feature_selection/kept_features_backup.yaml data/feature_selection/kept_features.yaml
python -m kicktipp_predictor train
```

---

## Alternative: Wait for Better Ablation

If you want data-driven results, re-run the improved ablation script:

```bash
# I've updated it to use 4 seasons instead of 2
./run_feature_study.sh
```

This should give more reliable results with ~1400 training samples instead of 608.

---

## My Recommendation: Do Phase 1 Now

The 7 interaction ratio features are **highly likely to be redundant**.

This is a safe, conservative first step that should:
- ✅ Reduce complexity (11% fewer features)
- ✅ Maintain performance (< 0.05 points loss expected)
- ✅ Speed up training slightly
- ✅ Give you confidence to proceed to Phase 2 & 3

Start with this and see how it goes!

