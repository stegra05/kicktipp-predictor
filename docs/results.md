# Model Performance and Results

This document provides a detailed analysis of the model's performance, including key metrics and visualizations.

## Evaluation Metrics

The model is evaluated on the following metrics:

- **Accuracy:** 40.8%
- **Average Points:** 1.177
- **Log Loss:** 1.0999
- **Brier Score:** 0.6675
- **Rank Probability Score (RPS):** 0.2385

As noted in the main `README.md`, the model currently does not predict any draws. This is a key area for future improvement and is likely due to the `draw_margin` parameter being too small.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to explain the output of the model. The following plots show the most important features and their impact on the model's predictions.

### Feature Importance

The following plot shows the mean absolute SHAP value for each feature, which is a measure of its importance.

![SHAP Summary Plot](../images/summary_beeswarm.png)

### Dependence Plots

These plots show how the model's prediction for a single feature changes as the feature's value changes.

![Dependence Plot for away_avg_goals_against](../images/dependence_away_avg_goals_against.png)
![Dependence Plot for away_matches_played](../images/dependence_away_matches_played.png)
![Dependence Plot for tanh_tamed_elo](../images/dependence_tanh_tamed_elo.png)
![Dependence Plot for away_avg_goals_for](../images/dependence_away_avg_goals_for.png)
![Dependence Plot for home_wform_points_L3](../images/dependence_home_wform_points_L3.png)
![Dependence Plot for home_wform_points_L10](../images/dependence_home_wform_points_L10.png)
![Dependence Plot for home_form_points_weighted_by_opponent_rank](../images/dependence_home_form_points_weighted_by_opponent_rank.png)