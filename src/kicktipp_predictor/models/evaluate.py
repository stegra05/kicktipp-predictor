from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
from kicktipp_predictor.models.hybrid_predictor import HybridPredictor


def run_evaluation(season: bool = False) -> None:
    if season:
        from kicktipp_predictor.evaluate_season_entry import run as season_eval
        season_eval()
        return

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
        print("ERROR: No trained models found. Run training first.")
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

    # Train Poisson only on matches strictly before the test set's first date
    id_to_date = {m['match_id']: m['date'] for m in all_matches if m.get('match_id') is not None and m.get('date') is not None}
    test_ids = list(test_df['match_id']) if 'match_id' in test_df.columns else []
    test_dates = [id_to_date.get(mid) for mid in test_ids]
    test_dates = [d for d in test_dates if d is not None]
    if test_dates:
        first_test_date = min(test_dates)
        hist_prior = [m for m in finished if m.get('date') is not None and m['date'] < first_test_date]
    else:
        hist_prior = finished
    import pandas as pd
    hist_df = pd.DataFrame(hist_prior)
    predictor.poisson_predictor.train(hist_df)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(test_features)

    # Calculate metrics (reuse HybridPredictor.evaluate printing)
    predictor.evaluate(test_df, test_features)


