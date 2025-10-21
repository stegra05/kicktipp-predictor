from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
from kicktipp_predictor.models.hybrid_predictor import HybridPredictor
from kicktipp_predictor.models.performance_tracker import PerformanceTracker
from kicktipp_predictor.models.confidence_selector import extract_display_confidence


def run() -> None:
    import sys
    from collections import defaultdict, Counter
    import pandas as pd

    print("="*80)
    print("SEASON PERFORMANCE EVALUATION")
    print("="*80)
    print()

    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()
    tracker = PerformanceTracker(storage_dir="data/predictions_season_eval")

    # Load trained models
    print("Loading models...")
    if not predictor.load_models("hybrid"):
        print("\nERROR: No trained models found.")
        print("Please run training first to train the models.")
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

    # Get historical data for feature context (finished-only; augment with previous season if sparse)
    print("Fetching historical data for context...")
    historical_matches_all = data_fetcher.fetch_season_matches(current_season)
    historical_matches = [m for m in historical_matches_all if m.get('is_finished')]
    if len(historical_matches) < 50:
        print("Fetching additional finished matches from previous season...")
        prev_season_matches = data_fetcher.fetch_season_matches(current_season - 1)
        historical_matches.extend([m for m in prev_season_matches if m.get('is_finished')])

    # Fit goal temperature once on prior finished matches (train Poisson first)
    try:
        hist_df_for_temp = pd.DataFrame([m for m in historical_matches if m.get('date') is not None])
        if not hist_df_for_temp.empty:
            predictor.poisson_predictor.train(hist_df_for_temp)
            if hasattr(predictor, 'fit_goal_temperature'):
                predictor.fit_goal_temperature(hist_df_for_temp)
    except Exception:
        pass

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

        # Train Poisson on finished historical matches strictly before this matchday's earliest match
        md_min_date = min(m['date'] for m in matchday_matches)
        hist_prior = [m for m in historical_matches if m['is_finished'] and m.get('date') is not None and m['date'] < md_min_date]
        hist_df = pd.DataFrame(hist_prior)
        predictor.poisson_predictor.train(hist_df)

        # Generate predictions
        predictions = predictor.predict(features_df)

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

    # Overall metrics
    total_matches = len(all_predictions)
    total_points = sum(p['points_earned'] for p in all_predictions)
    avg_points = total_points / total_matches if total_matches > 0 else 0
    print(f"\nTotal Matches Evaluated: {total_matches}")
    print(f"Total Points Earned: {total_points}")
    print(f"Average Points per Match: {avg_points:.3f}")


