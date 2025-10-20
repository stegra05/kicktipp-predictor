#!/usr/bin/env python3
"""
Script to generate predictions for upcoming matches.
Run this weekly to get predictions for the next matchday.
"""

import sys
import argparse
from datetime import datetime
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor
from src.models.performance_tracker import PerformanceTracker


def print_predictions(predictions):
    """Print predictions in a nice format."""
    print("\n" + "="*80)
    print("MATCH PREDICTIONS")
    print("="*80)

    for pred in predictions:
        print(f"\n{pred['home_team']} vs {pred['away_team']}")
        print(f"Predicted Score: {pred['predicted_home_score']}:{pred['predicted_away_score']}")
        print(f"Probabilities: Home {pred['home_win_probability']*100:.1f}% | "
              f"Draw {pred['draw_probability']*100:.1f}% | "
              f"Away {pred['away_win_probability']*100:.1f}%")
        print(f"Confidence: {pred.get('confidence', 0)*100:.1f}%")
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Generate 3. Liga match predictions')
    parser.add_argument('--matchday', type=int, help='Specific matchday to predict')
    parser.add_argument('--days', type=int, default=7, help='Number of days ahead to predict (default: 7)')
    parser.add_argument('--record', action='store_true', help='Record predictions for performance tracking')
    parser.add_argument('--update-results', action='store_true',
                       help='Update previous predictions with actual results')

    args = parser.parse_args()

    print("="*80)
    print("3. LIGA PREDICTOR")
    print("="*80)
    print()

    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()
    tracker = PerformanceTracker()

    # Load trained models
    print("Loading models...")
    if not predictor.load_models("hybrid"):
        print("\nERROR: No trained models found.")
        print("Please run train_model.py first to train the models.")
        sys.exit(1)

    print("Models loaded successfully!\n")

    # Update results if requested
    if args.update_results:
        print("Updating previous predictions with actual results...")
        current_season = data_fetcher.get_current_season()
        finished_matches = [m for m in data_fetcher.fetch_season_matches(current_season)
                           if m['is_finished']]

        updated_count = tracker.update_results(finished_matches)
        print(f"Updated {updated_count} predictions with results\n")

        if updated_count > 0:
            tracker.print_summary()

    # Get matches to predict
    if args.matchday:
        print(f"Fetching matchday {args.matchday}...")
        current_season = data_fetcher.get_current_season()
        upcoming_matches = data_fetcher.fetch_matchday(args.matchday, current_season)
        upcoming_matches = [m for m in upcoming_matches if not m['is_finished']]
        matchday = args.matchday
    else:
        print(f"Fetching upcoming matches (next {args.days} days)...")
        upcoming_matches = data_fetcher.get_upcoming_matches(days=args.days)
        matchday = upcoming_matches[0]['matchday'] if upcoming_matches else None

    if not upcoming_matches:
        print("No upcoming matches found.")
        sys.exit(0)

    print(f"Found {len(upcoming_matches)} upcoming matches")

    # Get historical data for features
    print("Fetching historical data for context...")
    current_season = data_fetcher.get_current_season()
    historical_matches = data_fetcher.fetch_season_matches(current_season)

    # If not enough data in current season, add previous seasons
    if len(historical_matches) < 50:
        print("Fetching additional historical data...")
        prev_season_matches = data_fetcher.fetch_season_matches(current_season - 1)
        historical_matches.extend(prev_season_matches)

    print(f"Using {len([m for m in historical_matches if m['is_finished']])} historical matches for context")

    # Create features
    print("\nGenerating features...")
    features_df = feature_engineer.create_prediction_features(
        upcoming_matches, historical_matches
    )

    # Train Poisson component on finished historical matches for realistic λs
    import pandas as pd
    hist_df = pd.DataFrame([m for m in historical_matches if m['is_finished']])
    predictor.poisson_predictor.train(hist_df)

    # Generate predictions
    print("Generating predictions...\n")
    predictions = predictor.predict_optimized(
        features_df,
    )

    # Print predictions
    print_predictions(predictions)

    # Record predictions if requested
    if args.record:
        print("\nRecording predictions for performance tracking...")
        tracker.record_predictions(predictions, matchday=matchday)
        print("Predictions recorded successfully!")

    # Show current performance stats if available
    stats = tracker.get_current_stats()
    if stats['total_predictions'] > 0:
        print("\n" + "="*80)
        print("CURRENT PERFORMANCE")
        print("="*80)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Average Points per Match: {stats['avg_points_per_match']:.2f}")
        print(f"Projected Season Total: {stats['avg_points_per_match'] * 38:.1f} points")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
