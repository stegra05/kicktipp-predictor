from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
from kicktipp_predictor.models.hybrid_predictor import HybridPredictor
import os


def run_training(config_path: str = "config/best_params.json") -> None:
    import sys
    print("="*60)
    print("3. LIGA PREDICTOR - MODEL TRAINING")
    print("="*60)
    print()

    # Resolve and set thread count for downstream libraries (XGBoost reads OMP_NUM_THREADS)
    try:
        cpu_count = os.cpu_count() or 1
        omp_threads = int(os.getenv("OMP_NUM_THREADS", str(cpu_count)))
        if omp_threads < 1:
            omp_threads = 1
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        # Common BLAS/numexpr knobs to avoid oversubscription
        os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(omp_threads))
        print(f"Using {omp_threads} threads for training (OMP_NUM_THREADS)")
    except Exception:
        pass

    # Initialize components
    print("Initializing components...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()

    # Fetch historical data
    print("\nFetching historical data...")
    current_season = data_fetcher.get_current_season()

    # Fetch multiple seasons for training (last 3-4 seasons)
    start_season = current_season - 3
    print(f"Fetching seasons from {start_season}/{start_season+1} to {current_season}/{current_season+1}")

    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)

    print(f"\nTotal matches fetched: {len(all_matches)}")
    finished_matches = [m for m in all_matches if m['is_finished']]
    print(f"Finished matches: {len(finished_matches)}")

    if len(finished_matches) < 100:
        print("\nERROR: Not enough historical data to train models.")
        print("Need at least 100 finished matches.")
        sys.exit(1)

    # Create features
    print("\nCreating features from matches...")
    features_df = feature_engineer.create_features_from_matches(all_matches)

    print(f"Feature dataset created with {len(features_df)} samples")
    print(f"Features: {len(features_df.columns)} columns")

    # Split data for evaluation
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df[:split_idx]
    test_df = features_df[split_idx:]

    print(f"\nTraining set: {len(train_df)} matches")
    print(f"Test set: {len(test_df)} matches")

    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    predictor.train(train_df)

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    test_features = test_df.drop(columns=['home_score', 'away_score', 'goal_difference', 'result'],
                                  errors='ignore')

    # Auto-fit separate home/away goal temperatures using validation split
    try:
        predictor.fit_goal_temperatures_from_validation(test_df, test_features)
    except Exception:
        pass
    evaluation = predictor.evaluate(test_df, test_features)

    # Validate goal distributions
    print("\n" + "="*60)
    print("GOAL DISTRIBUTION VALIDATION")
    print("="*60)

    # Get predictions for validation (already fitted temps)
    test_predictions = predictor.predict(test_features)

    # Calculate actual vs predicted goal statistics
    actual_home_goals = test_df['home_score'].mean()
    actual_away_goals = test_df['away_score'].mean()
    actual_total_goals = test_df['home_score'].sum() + test_df['away_score'].sum()

    # Lightweight aggregation via list->numpy arrays
    try:
        import numpy as _np
        _home_arr = _np.fromiter((float(p['home_expected_goals']) for p in test_predictions), dtype=float)
        _away_arr = _np.fromiter((float(p['away_expected_goals']) for p in test_predictions), dtype=float)
        pred_home_goals = float(_home_arr.mean()) if _home_arr.size else 0.0
        pred_away_goals = float(_away_arr.mean()) if _away_arr.size else 0.0
        pred_total_goals = float((_home_arr + _away_arr).sum())
    except Exception:
        pred_home_goals = sum(p['home_expected_goals'] for p in test_predictions) / len(test_predictions)
        pred_away_goals = sum(p['away_expected_goals'] for p in test_predictions) / len(test_predictions)
        pred_total_goals = sum(p['home_expected_goals'] + p['away_expected_goals'] for p in test_predictions)

    print(f"\nActual average goals per match:")
    print(f"  Home: {actual_home_goals:.3f}")
    print(f"  Away: {actual_away_goals:.3f}")
    print(f"  Total: {actual_home_goals + actual_away_goals:.3f}")

    print(f"\nPredicted average expected goals per match:")
    print(f"  Home: {pred_home_goals:.3f}")
    print(f"  Away: {pred_away_goals:.3f}")
    print(f"  Total: {pred_home_goals + pred_away_goals:.3f}")

    home_diff_pct = abs(pred_home_goals - actual_home_goals) / actual_home_goals * 100
    away_diff_pct = abs(pred_away_goals - actual_away_goals) / actual_away_goals * 100

    print(f"\nDifference:")
    print(f"  Home: {home_diff_pct:.1f}%")
    print(f"  Away: {away_diff_pct:.1f}%")

    if home_diff_pct > 15 or away_diff_pct > 15:
        print("\nWARNING: Goal prediction mismatch > 15%!")
        print("   Consider adjusting goal_temperature parameter")

    # Save models
    print("\nSaving trained models...")
    predictor.save_models("hybrid")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to data/models/")


