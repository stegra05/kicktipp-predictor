#!/usr/bin/env python3
"""
Script to evaluate predictor performance for the entire current season.
"""

import sys
from collections import defaultdict, Counter
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor
from src.models.performance_tracker import PerformanceTracker

def main():
    """
    Main function to run the season evaluation.
    """
    print("="*80)
    print("SEASON PERFORMANCE EVALUATION")
    print("="*80)
    print()

    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()
    # Using a temporary tracker to not interfere with recorded predictions
    tracker = PerformanceTracker(storage_dir="data/predictions_season_eval")

    # Load trained models
    print("Loading models...")
    if not predictor.load_models("hybrid"):
        print("\nERROR: No trained models found.")
        print("Please run train_model.py first to train the models.")
        sys.exit(1)
    print("Models loaded successfully!\n")

    # Get current season data
    current_season = data_fetcher.get_current_season()
    print(f"Fetching data for current season: {current_season}")
    season_matches = data_fetcher.fetch_season_matches(current_season)

    finished_matches = [m for m in season_matches if m['is_finished']]
    if not finished_matches:
        print("No finished matches found for the current season.")
        sys.exit(0)

    # Determine matchdays to evaluate
    first_matchday = min(m['matchday'] for m in finished_matches)
    last_matchday = max(m['matchday'] for m in finished_matches)

    print(f"Evaluating matchdays from {first_matchday} to {last_matchday}\n")

    all_predictions = []

    # Get historical data for feature context
    print("Fetching historical data for context...")
    historical_matches = data_fetcher.fetch_season_matches(current_season)
    if len(historical_matches) < 50:
        print("Fetching additional historical data from previous season...")
        prev_season_matches = data_fetcher.fetch_season_matches(current_season - 1)
        historical_matches.extend(prev_season_matches)

    for matchday in range(first_matchday, last_matchday + 1):
        print(f"--- Processing Matchday {matchday} ---")

        # Get matches for the current matchday
        matchday_matches = [m for m in finished_matches if m['matchday'] == matchday]
        if not matchday_matches:
            print(f"No finished matches for matchday {matchday}.")
            continue

        # Create features
        features_df = feature_engineer.create_prediction_features(
            matchday_matches, historical_matches
        )

        if features_df.empty:
            print(f"Could not generate features for matchday {matchday}.")
            continue

        # Train Poisson component on finished historical matches for realistic λs
        import pandas as pd
        hist_df = pd.DataFrame([m for m in historical_matches if m['is_finished']])
        predictor.poisson_predictor.train(hist_df)

        # Generate predictions (using default 'safe' strategy with expected-points optimization)
        predictions = predictor.predict_optimized(
            features_df,
            strategy='safe',
            optimize_for_points=True
        )

        # Record predictions and update results immediately
        for pred, actual in zip(predictions, matchday_matches):
            points = tracker._calculate_points(
                pred['predicted_home_score'], pred['predicted_away_score'],
                actual['home_score'], actual['away_score']
            )
            pred['actual_home_score'] = actual['home_score']
            pred['actual_away_score'] = actual['away_score']
            pred['points_earned'] = points
            pred['is_evaluated'] = True
            pred['matchday'] = matchday
            all_predictions.append(pred)

        print(f"Generated and evaluated {len(predictions)} predictions for matchday {matchday}")

    if not all_predictions:
        print("\nNo predictions could be generated for the season.")
        sys.exit(0)

    # --- AGGREGATE AND PRINT COMPREHENSIVE REPORT ---

    # Overall metrics
    total_matches = len(all_predictions)
    total_points = sum(p['points_earned'] for p in all_predictions)
    avg_points = total_points / total_matches if total_matches > 0 else 0

    # Detailed metrics calculation
    metrics = {
        'by_outcome': defaultdict(lambda: {'total': 0, 'correct': 0, 'points': 0}),
        'by_confidence': defaultdict(lambda: {'total': 0, 'points': 0}),
        'score_predictions': Counter(),
        'score_actuals': Counter(),
        'matchday_stats': defaultdict(lambda: {'points': 0, 'matches': 0})
    }
    points_dist = defaultdict(int)

    for pred in all_predictions:
        points_dist[pred['points_earned']] += 1

        ph, pa = pred['predicted_home_score'], pred['predicted_away_score']
        ah, aa = pred['actual_home_score'], pred['actual_away_score']

        # Outcome
        actual_outcome = 'D' if ah == aa else ('H' if ah > aa else 'A')
        pred_outcome = 'D' if ph == pa else ('H' if ph > pa else 'A')

        metrics['by_outcome'][actual_outcome]['total'] += 1
        metrics['by_outcome'][actual_outcome]['points'] += pred['points_earned']
        if pred_outcome == actual_outcome:
            metrics['by_outcome'][actual_outcome]['correct'] += 1

        # Confidence
        confidence = pred.get('confidence', 0)
        conf_bucket = 'low' if confidence < 0.4 else ('medium' if confidence < 0.6 else 'high')
        metrics['by_confidence'][conf_bucket]['total'] += 1
        metrics['by_confidence'][conf_bucket]['points'] += pred['points_earned']

        # Scores
        metrics['score_predictions'][f"{ph}-{pa}"] += 1
        metrics['score_actuals'][f"{ah}-{aa}"] += 1

        # Matchday
        metrics['matchday_stats'][pred['matchday']]['points'] += pred['points_earned']
        metrics['matchday_stats'][pred['matchday']]['matches'] += 1

    # Print Report
    print("\n" + "="*80)
    print("SEASON PERFORMANCE SUMMARY")
    print("="*80)

    print("\n--- OVERALL PERFORMANCE ---")
    print(f"Evaluated Matchdays: {first_matchday} - {last_matchday}")
    print(f"Total Matches Evaluated: {total_matches}")
    print(f"Total Points Earned: {total_points}")
    print(f"Average Points per Match: {avg_points:.3f}")
    print(f"Projected Season Total (38 matchdays): {avg_points * 38:.1f} points")
    print(f"Point Efficiency: {(total_points / (total_matches * 4)) * 100:.1f}% (of maximum possible)")

    print("\n--- ACCURACY BREAKDOWN ---")
    exact_scores = points_dist[4]
    correct_diffs = points_dist[3]
    correct_results = points_dist[2]
    incorrect = points_dist[0]

    print(f"Exact Scores (4pts):       {exact_scores:4d} ({exact_scores/total_matches*100:5.1f}%)")
    print(f"Correct Differences (3pts): {correct_diffs:4d} ({correct_diffs/total_matches*100:5.1f}%)")
    print(f"Correct Results (2pts):    {correct_results:4d} ({correct_results/total_matches*100:5.1f}%)")
    print(f"Incorrect (0pts):          {incorrect:4d} ({incorrect/total_matches*100:5.1f}%)")

    print("\n--- PERFORMANCE BY ACTUAL OUTCOME ---")
    for outcome, data in sorted(metrics['by_outcome'].items()):
        name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        avg_pts_outcome = data['points'] / data['total'] if data['total'] > 0 else 0
        print(f"{name:10s}: {data['total']:3d} matches, {data['correct']:3d} correct ({acc:5.1f}%), avg {avg_pts_outcome:.2f} pts")

    print("\n--- PERFORMANCE BY CONFIDENCE ---")
    for conf, data in sorted(metrics['by_confidence'].items()):
        avg_pts_conf = data['points'] / data['total'] if data['total'] > 0 else 0
        print(f"{conf.capitalize():8s}: {data['total']:3d} matches, avg {avg_pts_conf:.2f} pts")

    print("\n--- PERFORMANCE BY MATCHDAY ---")
    print(f"{'Matchday':<10} {'Matches':<10} {'Points':<10} {'Avg Pts':<10}")
    print("-" * 42)
    for md, data in sorted(metrics['matchday_stats'].items()):
        avg_pts_md = data['points'] / data['matches'] if data['matches'] > 0 else 0
        print(f"{md:<10} {data['matches']:<10} {data['points']:<10} {avg_pts_md:<10.2f}")

    print("\n--- TOP 5 PREDICTED SCORES ---")
    for score, count in metrics['score_predictions'].most_common(5):
        print(f"{score:6s}: {count:3d} times ({count/total_matches*100:5.1f}%)")

    print("\n--- TOP 5 ACTUAL SCORES ---")
    for score, count in metrics['score_actuals'].most_common(5):
        print(f"{score:6s}: {count:3d} times ({count/total_matches*100:5.1f}%)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
