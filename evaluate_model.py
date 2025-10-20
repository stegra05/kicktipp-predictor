#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Provides detailed metrics, confusion matrices, and performance analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor


def calculate_detailed_metrics(predictions, actuals):
    """Calculate comprehensive evaluation metrics."""

    metrics = {
        'total_matches': len(predictions),
        'exact_scores': 0,
        'correct_differences': 0,
        'correct_results': 0,
        'incorrect': 0,
        'total_points': 0,
        'by_outcome': defaultdict(lambda: {'total': 0, 'correct': 0, 'points': 0}),
        'by_confidence': defaultdict(lambda: {'total': 0, 'correct_result': 0, 'points': 0}),
        'score_predictions': defaultdict(int),
        'score_actuals': defaultdict(int),
    }

    for pred, actual in zip(predictions, actuals):
        pred_home = pred['predicted_home_score']
        pred_away = pred['predicted_away_score']
        actual_home = actual['home_score']
        actual_away = actual['away_score']

        # Determine actual outcome
        if actual_home > actual_away:
            actual_outcome = 'H'
        elif actual_away > actual_home:
            actual_outcome = 'A'
        else:
            actual_outcome = 'D'

        # Determine predicted outcome
        if pred_home > pred_away:
            pred_outcome = 'H'
        elif pred_away > pred_home:
            pred_outcome = 'A'
        else:
            pred_outcome = 'D'

        # Track by actual outcome
        metrics['by_outcome'][actual_outcome]['total'] += 1

        # Track score distributions
        score_str = f"{pred_home}-{pred_away}"
        actual_str = f"{actual_home}-{actual_away}"
        metrics['score_predictions'][score_str] += 1
        metrics['score_actuals'][actual_str] += 1

        # Confidence bucket
        confidence = pred.get('confidence', pred.get('max_probability', 0))
        if confidence < 0.4:
            conf_bucket = 'low'
        elif confidence < 0.6:
            conf_bucket = 'medium'
        else:
            conf_bucket = 'high'

        metrics['by_confidence'][conf_bucket]['total'] += 1

        # Calculate points
        points = 0

        # Exact score: 4 points
        if pred_home == actual_home and pred_away == actual_away:
            metrics['exact_scores'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 4
            metrics['by_confidence'][conf_bucket]['points'] += 4
            points = 4
        # Correct goal difference: 3 points
        elif (pred_home - pred_away) == (actual_home - actual_away):
            metrics['correct_differences'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 3
            metrics['by_confidence'][conf_bucket]['points'] += 3
            points = 3
        # Correct winner: 2 points
        elif pred_outcome == actual_outcome:
            metrics['correct_results'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 2
            metrics['by_confidence'][conf_bucket]['points'] += 2
            metrics['by_confidence'][conf_bucket]['correct_result'] += 1
            points = 2
        else:
            metrics['incorrect'] += 1

        metrics['total_points'] += points

    return metrics


def print_evaluation_report(metrics):
    """Print a comprehensive evaluation report."""

    n = metrics['total_matches']

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Overall metrics
    print("\n--- OVERALL PERFORMANCE ---")
    print(f"Total Matches: {n}")
    print(f"Total Points: {metrics['total_points']}")
    print(f"Average Points per Match: {metrics['total_points']/n:.3f}")
    print(f"Expected Season Total (38 matches): {(metrics['total_points']/n)*38:.1f} points")
    print()

    # Accuracy breakdown
    print("--- ACCURACY BREAKDOWN ---")
    print(f"Exact Scores (4pts):       {metrics['exact_scores']:4d} ({metrics['exact_scores']/n*100:5.1f}%)")
    print(f"Correct Differences (3pts): {metrics['correct_differences']:4d} ({metrics['correct_differences']/n*100:5.1f}%)")
    print(f"Correct Results (2pts):    {metrics['correct_results']:4d} ({metrics['correct_results']/n*100:5.1f}%)")
    print(f"Incorrect (0pts):          {metrics['incorrect']:4d} ({metrics['incorrect']/n*100:5.1f}%)")
    print()

    # Points distribution
    total_possible = n * 4
    efficiency = (metrics['total_points'] / total_possible) * 100
    print(f"Point Efficiency: {efficiency:.1f}% (of maximum possible)")
    print()

    # By outcome
    print("--- PERFORMANCE BY ACTUAL OUTCOME ---")
    for outcome in ['H', 'D', 'A']:
        outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        data = metrics['by_outcome'][outcome]

        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            avg_pts = data['points'] / data['total']
            print(f"{outcome_name:10s}: {data['total']:3d} matches, "
                  f"{data['correct']:3d} correct ({acc:5.1f}%), "
                  f"avg {avg_pts:.2f} pts")

    print()

    # By confidence
    print("--- PERFORMANCE BY CONFIDENCE LEVEL ---")
    for conf_level in ['low', 'medium', 'high']:
        data = metrics['by_confidence'][conf_level]

        if data['total'] > 0:
            result_acc = data['correct_result'] / data['total'] * 100
            avg_pts = data['points'] / data['total']
            print(f"{conf_level.capitalize():8s}: {data['total']:3d} matches, "
                  f"result accuracy {result_acc:5.1f}%, "
                  f"avg {avg_pts:.2f} pts")

    print()

    # Most common predictions
    print("--- TOP 10 PREDICTED SCORES ---")
    sorted_preds = sorted(metrics['score_predictions'].items(),
                         key=lambda x: x[1], reverse=True)[:10]
    for score, count in sorted_preds:
        print(f"{score:6s}: {count:3d} times ({count/n*100:5.1f}%)")

    print()

    # Most common actual scores
    print("--- TOP 10 ACTUAL SCORES ---")
    sorted_actuals = sorted(metrics['score_actuals'].items(),
                           key=lambda x: x[1], reverse=True)[:10]
    for score, count in sorted_actuals:
        print(f"{score:6s}: {count:3d} times ({count/n*100:5.1f}%)")

    print()
    print("="*80)


def main():
    """Run comprehensive model evaluation."""

    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()

    # Load trained models
    if not predictor.load_models("hybrid"):
        print("ERROR: No trained models found. Run train_model.py first.")
        return

    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2

    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    finished = [m for m in all_matches if m['is_finished']]

    print(f"Loaded {len(finished)} finished matches")

    # Create features
    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} samples")

    # Use last 30% as test set
    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]

    print(f"Using {len(test_df)} matches for evaluation")

    # Prepare test features (without targets)
    test_features = test_df.drop(
        columns=['home_score', 'away_score', 'goal_difference', 'result'],
        errors='ignore'
    )

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(test_features)

    # Calculate metrics
    print("Calculating metrics...")
    actuals = test_df.to_dict('records')
    metrics = calculate_detailed_metrics(predictions, actuals)

    # Print report
    print_evaluation_report(metrics)

    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    strategies = ['balanced', 'conservative', 'aggressive', 'safe']
    strategy_results = {}

    for strategy in strategies:
        preds = predictor.predict_optimized(test_features, strategy=strategy)
        metrics = calculate_detailed_metrics(preds, actuals)
        strategy_results[strategy] = metrics

    print(f"\n{'Strategy':<15} {'Avg Pts':<10} {'Exact':<8} {'Diff':<8} {'Result':<8}")
    print("-" * 60)

    for strategy in strategies:
        m = strategy_results[strategy]
        n = m['total_matches']
        avg_pts = m['total_points'] / n
        exact_pct = m['exact_scores'] / n * 100
        diff_pct = m['correct_differences'] / n * 100
        result_pct = m['correct_results'] / n * 100

        print(f"{strategy.capitalize():<15} {avg_pts:<10.3f} "
              f"{exact_pct:<8.1f} {diff_pct:<8.1f} {result_pct:<8.1f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
