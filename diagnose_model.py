#!/usr/bin/env python3
"""
Diagnostic script for model analysis.
Provides detailed insights into model behavior, calibration, and prediction patterns.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor


def analyze_goal_distributions(predictions, actuals):
    """Analyze predicted vs actual goal distributions."""
    print("\n" + "="*80)
    print("GOAL DISTRIBUTION ANALYSIS")
    print("="*80)

    # Collect goals
    pred_home_goals = [p['predicted_home_score'] for p in predictions]
    pred_away_goals = [p['predicted_away_score'] for p in predictions]
    actual_home_goals = [a['home_score'] for a in actuals]
    actual_away_goals = [a['away_score'] for a in actuals]

    # Statistics
    print("\n--- Goal Statistics ---")
    print(f"{'Metric':<25} {'Predicted':<15} {'Actual':<15} {'Difference'}")
    print("-" * 70)

    pred_home_mean = np.mean(pred_home_goals)
    actual_home_mean = np.mean(actual_home_goals)
    print(f"{'Home goals (mean)':<25} {pred_home_mean:<15.3f} {actual_home_mean:<15.3f} {pred_home_mean - actual_home_mean:+.3f}")

    pred_away_mean = np.mean(pred_away_goals)
    actual_away_mean = np.mean(actual_away_goals)
    print(f"{'Away goals (mean)':<25} {pred_away_mean:<15.3f} {actual_away_mean:<15.3f} {pred_away_mean - actual_away_mean:+.3f}")

    pred_total_mean = pred_home_mean + pred_away_mean
    actual_total_mean = actual_home_mean + actual_away_mean
    print(f"{'Total goals (mean)':<25} {pred_total_mean:<15.3f} {actual_total_mean:<15.3f} {pred_total_mean - actual_total_mean:+.3f}")

    # Distributions
    print("\n--- Goal Distribution (0-5+) ---")
    print(f"{'Goals':<10} {'Pred Home':<12} {'Actual Home':<12} {'Pred Away':<12} {'Actual Away'}")
    print("-" * 60)

    pred_home_dist = Counter(pred_home_goals)
    actual_home_dist = Counter(actual_home_goals)
    pred_away_dist = Counter(pred_away_goals)
    actual_away_dist = Counter(actual_away_goals)

    for i in range(6):
        if i == 5:
            # 5+ goals
            pred_h = sum(pred_home_dist.get(j, 0) for j in range(5, 10))
            actual_h = sum(actual_home_dist.get(j, 0) for j in range(5, 10))
            pred_a = sum(pred_away_dist.get(j, 0) for j in range(5, 10))
            actual_a = sum(actual_away_dist.get(j, 0) for j in range(5, 10))
            print(f"{'5+':<10} {pred_h:<12d} {actual_h:<12d} {pred_a:<12d} {actual_a}")
        else:
            pred_h = pred_home_dist.get(i, 0)
            actual_h = actual_home_dist.get(i, 0)
            pred_a = pred_away_dist.get(i, 0)
            actual_a = actual_away_dist.get(i, 0)
            print(f"{i:<10} {pred_h:<12d} {actual_h:<12d} {pred_a:<12d} {actual_a}")


def analyze_score_predictions(predictions, actuals):
    """Analyze most common predicted vs actual scores."""
    print("\n" + "="*80)
    print("SCORELINE ANALYSIS")
    print("="*80)

    # Collect scorelines
    pred_scores = [f"{p['predicted_home_score']}-{p['predicted_away_score']}" for p in predictions]
    actual_scores = [f"{int(a['home_score'])}-{int(a['away_score'])}" for a in actuals]

    pred_counter = Counter(pred_scores)
    actual_counter = Counter(actual_scores)

    print("\n--- Top 15 Predicted Scorelines ---")
    print(f"{'Score':<10} {'Count':<10} {'Percentage'}")
    print("-" * 40)
    for score, count in pred_counter.most_common(15):
        pct = count / len(pred_scores) * 100
        print(f"{score:<10} {count:<10} {pct:5.1f}%")

    print("\n--- Top 15 Actual Scorelines ---")
    print(f"{'Score':<10} {'Count':<10} {'Percentage'}")
    print("-" * 40)
    for score, count in actual_counter.most_common(15):
        pct = count / len(actual_scores) * 100
        print(f"{score:<10} {count:<10} {pct:5.1f}%")

    # Check for over-prediction of specific scores
    print("\n--- Over/Under-Predicted Scores (Top 10) ---")
    all_scores = set(pred_scores) | set(actual_scores)
    differences = []

    for score in all_scores:
        pred_pct = pred_counter.get(score, 0) / len(pred_scores) * 100
        actual_pct = actual_counter.get(score, 0) / len(actual_scores) * 100
        diff = pred_pct - actual_pct
        differences.append((score, diff, pred_pct, actual_pct))

    differences.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Score':<10} {'Diff %':<10} {'Predicted %':<12} {'Actual %'}")
    print("-" * 50)
    for score, diff, pred_pct, actual_pct in differences[:10]:
        sign = "+" if diff > 0 else ""
        print(f"{score:<10} {sign}{diff:<9.1f} {pred_pct:<12.1f} {actual_pct:.1f}")


def analyze_outcome_distribution(predictions, actuals):
    """Analyze H/D/A outcome distributions."""
    print("\n" + "="*80)
    print("OUTCOME DISTRIBUTION ANALYSIS")
    print("="*80)

    def get_outcome(home, away):
        if home > away:
            return 'H'
        elif away > home:
            return 'A'
        else:
            return 'D'

    pred_outcomes = [get_outcome(p['predicted_home_score'], p['predicted_away_score'])
                    for p in predictions]
    actual_outcomes = [get_outcome(a['home_score'], a['away_score'])
                      for a in actuals]

    pred_counter = Counter(pred_outcomes)
    actual_counter = Counter(actual_outcomes)

    print("\n--- Outcome Distribution ---")
    print(f"{'Outcome':<15} {'Predicted':<15} {'Actual':<15} {'Difference'}")
    print("-" * 60)

    for outcome in ['H', 'D', 'A']:
        outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        pred_count = pred_counter.get(outcome, 0)
        actual_count = actual_counter.get(outcome, 0)
        pred_pct = pred_count / len(pred_outcomes) * 100
        actual_pct = actual_count / len(actual_outcomes) * 100
        diff = pred_pct - actual_pct

        print(f"{outcome_name:<15} {pred_pct:5.1f}% ({pred_count:3d}) "
              f"{actual_pct:5.1f}% ({actual_count:3d}) {diff:+6.1f}%")

    # Confusion matrix
    print("\n--- Outcome Confusion Matrix ---")
    print("Rows: Actual, Columns: Predicted")
    print(f"{'Actual \\ Pred':<15} {'H':<10} {'D':<10} {'A':<10}")
    print("-" * 45)

    confusion = defaultdict(lambda: defaultdict(int))
    for pred_out, actual_out in zip(pred_outcomes, actual_outcomes):
        confusion[actual_out][pred_out] += 1

    for actual in ['H', 'D', 'A']:
        actual_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[actual]
        row = []
        total = sum(confusion[actual].values())
        for pred in ['H', 'D', 'A']:
            count = confusion[actual][pred]
            pct = count / total * 100 if total > 0 else 0
            row.append(f"{count:3d} ({pct:4.1f}%)")

        print(f"{actual_name:<15} {row[0]:<10} {row[1]:<10} {row[2]:<10}")


def analyze_confidence_calibration(predictions, actuals):
    """Analyze confidence calibration."""
    print("\n" + "="*80)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("="*80)

    # Group by confidence bins
    bins = [(0, 0.4), (0.4, 0.6), (0.6, 1.0)]
    bin_names = ['Low (<40%)', 'Medium (40-60%)', 'High (>60%)']

    print("\n--- Calibration by Confidence Bin ---")
    print(f"{'Confidence':<20} {'N':<8} {'Correct':<12} {'Points/Match'}")
    print("-" * 60)

    for (low, high), name in zip(bins, bin_names):
        bin_preds = []
        bin_actuals = []

        for pred, actual in zip(predictions, actuals):
            conf = pred.get('confidence', 0.5)
            if low <= conf < high:
                bin_preds.append(pred)
                bin_actuals.append(actual)

        if not bin_preds:
            continue

        correct = 0
        points = 0

        for pred, actual in zip(bin_preds, bin_actuals):
            pred_h = pred['predicted_home_score']
            pred_a = pred['predicted_away_score']
            actual_h = actual['home_score']
            actual_a = actual['away_score']

            if pred_h == actual_h and pred_a == actual_a:
                correct += 1
                points += 4
            elif (pred_h - pred_a) == (actual_h - actual_a):
                correct += 1
                points += 3
            else:
                pred_winner = 'H' if pred_h > pred_a else ('A' if pred_a > pred_h else 'D')
                actual_winner = 'H' if actual_h > actual_a else ('A' if actual_a > actual_h else 'D')
                if pred_winner == actual_winner:
                    correct += 1
                    points += 2

        n = len(bin_preds)
        correct_pct = correct / n * 100
        avg_points = points / n

        print(f"{name:<20} {n:<8} {correct:3d} ({correct_pct:5.1f}%) {avg_points:5.2f}")


def main():
    """Run comprehensive diagnostics."""
    print("="*80)
    print("MODEL DIAGNOSTICS")
    print("="*80)

    # Load data and models
    print("\nLoading data and models...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()

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

    # Use last 30% as test set
    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]

    print(f"Using {len(test_df)} matches for diagnostics\n")

    # Prepare test features
    test_features = test_df.drop(
        columns=['home_score', 'away_score', 'goal_difference', 'result'],
        errors='ignore'
    )

    # Generate predictions
    print("Generating predictions...")
    predictions = predictor.predict(test_features)
    actuals = test_df.to_dict('records')

    # Run analyses
    analyze_goal_distributions(predictions, actuals)
    analyze_score_predictions(predictions, actuals)
    analyze_outcome_distribution(predictions, actuals)
    analyze_confidence_calibration(predictions, actuals)

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
