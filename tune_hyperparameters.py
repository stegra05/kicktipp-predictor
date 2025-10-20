#!/usr/bin/env python3
"""
Script to tune hyperparameters using cross-validation.
Optimizes weights and parameters to maximize prediction points.
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor


def calculate_points(predictions, actuals):
    """Calculate total points for a set of predictions."""
    total_points = 0

    for pred, actual in zip(predictions, actuals):
        pred_home = pred['predicted_home_score']
        pred_away = pred['predicted_away_score']
        actual_home = actual['home_score']
        actual_away = actual['away_score']

        # Exact score: 4 points
        if pred_home == actual_home and pred_away == actual_away:
            total_points += 4
            continue

        # Correct goal difference: 3 points
        if (pred_home - pred_away) == (actual_home - actual_away):
            total_points += 3
            continue

        # Correct winner: 2 points
        pred_winner = 'H' if pred_home > pred_away else ('A' if pred_away > pred_home else 'D')
        actual_winner = 'H' if actual_home > actual_away else ('A' if actual_away > actual_home else 'D')

        if pred_winner == actual_winner:
            total_points += 2

    return total_points


def evaluate_weights(ml_weight, prob_blend_alpha, min_lambda,
                    train_features, train_matches, test_features, test_matches):
    """Evaluate a specific set of hyperparameters."""

    predictor = HybridPredictor()
    predictor.ml_weight = ml_weight
    predictor.poisson_weight = 1 - ml_weight
    predictor.prob_blend_alpha = prob_blend_alpha
    predictor.min_lambda = min_lambda

    # Train on training data
    predictor.train(train_matches)

    # Predict on test data
    predictions = predictor.predict(test_features)

    # Calculate points
    points = calculate_points(predictions, test_matches.to_dict('records'))

    return points / len(test_matches) if len(test_matches) > 0 else 0


def grid_search():
    """Perform grid search over hyperparameters."""

    print("="*80)
    print("HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()

    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2  # Use last 2-3 seasons

    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    print(f"Loaded {len(all_matches)} matches")

    # Create features
    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} feature samples")

    # Define parameter grid
    ml_weights = [0.5, 0.55, 0.6, 0.65, 0.7]
    prob_blend_alphas = [0.6, 0.65, 0.7, 0.75]
    min_lambdas = [0.2, 0.25, 0.3]

    best_score = 0
    best_params = {}

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    print("\nStarting grid search...")
    print(f"Total combinations: {len(ml_weights) * len(prob_blend_alphas) * len(min_lambdas)}")
    print()

    iteration = 0
    total_iterations = len(ml_weights) * len(prob_blend_alphas) * len(min_lambdas)

    for ml_weight in ml_weights:
        for prob_blend_alpha in prob_blend_alphas:
            for min_lambda in min_lambdas:
                iteration += 1

                cv_scores = []

                for fold, (train_idx, test_idx) in enumerate(tscv.split(features_df)):
                    train_features = features_df.iloc[train_idx]
                    test_features = features_df.iloc[test_idx]

                    # Filter out targets for test features
                    test_features_only = test_features.drop(
                        columns=['home_score', 'away_score', 'goal_difference', 'result'],
                        errors='ignore'
                    )

                    score = evaluate_weights(
                        ml_weight, prob_blend_alpha, min_lambda,
                        train_features, train_features,
                        test_features_only, test_features
                    )

                    cv_scores.append(score)

                avg_score = np.mean(cv_scores)

                print(f"[{iteration}/{total_iterations}] "
                      f"ml_weight={ml_weight:.2f}, "
                      f"prob_blend_alpha={prob_blend_alpha:.2f}, "
                      f"min_lambda={min_lambda:.2f} "
                      f"-> Avg score: {avg_score:.3f} pts/match")

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'ml_weight': ml_weight,
                        'prob_blend_alpha': prob_blend_alpha,
                        'min_lambda': min_lambda
                    }

    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*80)
    print(f"ML Weight: {best_params['ml_weight']:.2f}")
    print(f"Probability Blend Alpha: {best_params['prob_blend_alpha']:.2f}")
    print(f"Minimum Lambda: {best_params['min_lambda']:.2f}")
    print(f"Average Score: {best_score:.3f} points per match")
    print(f"Expected Season Total: {best_score * 38:.1f} points")
    print()

    print("To use these parameters, update the values in")
    print("src/models/hybrid_predictor.py __init__ method:")
    print(f"  self.ml_weight = {best_params['ml_weight']}")
    print(f"  self.prob_blend_alpha = {best_params['prob_blend_alpha']}")
    print(f"  self.min_lambda = {best_params['min_lambda']}")
    print()


if __name__ == "__main__":
    grid_search()
