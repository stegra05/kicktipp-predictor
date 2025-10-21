import typer

app = typer.Typer(help="Kicktipp Predictor CLI")


@app.command()
def train(
    seasons_back: int = typer.Option(3, help="Number of past seasons to use for training"),
):
    """Train the match predictor on historical data."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor

    print("=" * 80)
    print("TRAINING MATCH PREDICTOR")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    loader = DataLoader()
    current_season = loader.get_current_season()
    start_season = current_season - seasons_back

    print(f"Fetching seasons {start_season} to {current_season}...")
    all_matches = loader.fetch_historical_seasons(start_season, current_season)
    print(f"Loaded {len(all_matches)} matches")

    # Create features
    print("Creating features...")
    features_df = loader.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} training samples with {len(features_df.columns)} columns")

    # Train predictor
    print("\nTraining predictor...")
    predictor = MatchPredictor()
    predictor.train(features_df)

    # Save models
    print("\nSaving models...")
    predictor.save_models()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


@app.command()
def predict(
    days: int = typer.Option(7, help="Days ahead to predict"),
    matchday: int | None = typer.Option(None, help="Specific matchday to predict"),
):
    """Make predictions for upcoming matches."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor

    print("=" * 80)
    print("MATCH PREDICTIONS")
    print("=" * 80)
    print()

    # Load data
    loader = DataLoader()
    predictor = MatchPredictor()

    # Load trained models
    try:
        predictor.load_models()
    except FileNotFoundError:
        print("ERROR: No trained models found. Run 'train' command first.")
        raise typer.Exit(code=1)

    # Get upcoming matches
    if matchday is not None:
        print(f"Getting matches for matchday {matchday}...")
        upcoming_matches = loader.fetch_matchday(matchday)
    else:
        print(f"Getting upcoming matches (next {days} days)...")
        upcoming_matches = loader.get_upcoming_matches(days=days)

    if not upcoming_matches:
        print("No upcoming matches found.")
        return

    print(f"Found {len(upcoming_matches)} upcoming matches")

    # Get historical data for context
    current_season = loader.get_current_season()
    historical_matches = loader.fetch_season_matches(current_season)

    # Create features
    features_df = loader.create_prediction_features(upcoming_matches, historical_matches)

    if len(features_df) == 0:
        print("Could not create features (insufficient data). Try a later matchday.")
        return

    # Make predictions
    predictions = predictor.predict(features_df)

    # Display predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    for pred in predictions:
        home = pred['home_team']
        away = pred['away_team']
        score = f"{pred['predicted_home_score']}-{pred['predicted_away_score']}"
        outcome = pred['predicted_result']
        confidence = pred['confidence']

        print(f"\n{home} vs {away}")
        print(f"  Predicted Score: {score} ({outcome})")
        print(f"  Probabilities: H={pred['home_win_probability']:.2%} "
              f"D={pred['draw_probability']:.2%} A={pred['away_win_probability']:.2%}")
        print(f"  Confidence: {confidence:.3f}")


@app.command()
def evaluate():
    """Evaluate predictor performance on test data."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor
    from kicktipp_predictor.evaluate import print_evaluation_report, simple_benchmark

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print()

    # Load data
    loader = DataLoader()
    predictor = MatchPredictor()

    # Load trained models
    try:
        predictor.load_models()
    except FileNotFoundError:
        print("ERROR: No trained models found. Run 'train' command first.")
        raise typer.Exit(code=1)

    # Get data
    current_season = loader.get_current_season()
    start_season = current_season - 2

    print(f"Loading data from seasons {start_season} to {current_season}...")
    all_matches = loader.fetch_historical_seasons(start_season, current_season)
    features_df = loader.create_features_from_matches(all_matches)

    # Use last 30% as test set
    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]

    print(f"Evaluating on {len(test_df)} test samples...")
    print()

    # Evaluate
    metrics = predictor.evaluate(test_df)

    # Compute benchmark
    benchmark = simple_benchmark(test_df, strategy='home_win')

    # Print report
    print_evaluation_report(metrics, benchmark)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    from kicktipp_predictor.web.app import app as flask_app
    flask_app.run(host=host, port=port)


@app.command()
def tune(
    max_trials: int = typer.Option(0, help="Max parameter combos to evaluate (0 = full grid)"),
    n_splits: int = typer.Option(3, help="TimeSeriesSplit folds"),
    progress_interval: int = typer.Option(10, help="Trials per progress print"),
    objective: str = typer.Option("points", help="Optimization objective", case_sensitive=False),
    n_jobs: int = typer.Option(0, help="Parallel processes (0 = half CPUs)"),
    omp_threads: int = typer.Option(1, help="Threads per process for BLAS/OMP"),
    refine: bool = typer.Option(False, help="Enable zoom-in refinement around top configs"),
    refine_top_k: int = typer.Option(8, help="Top-K coarse configs to refine"),
    refine_steps: int = typer.Option(5, help="Points per parameter during local sweep"),
    span_ml_weight: float = typer.Option(0.03, help="Local span for ml_weight"),
    span_prob_alpha: float = typer.Option(0.05, help="Local span for prob_blend_alpha"),
    span_min_lambda: float = typer.Option(0.05, help="Local span for min_lambda"),
    span_goal_temp: float = typer.Option(0.10, help="Local span for goal_temperature"),
    span_conf_thr: float = typer.Option(0.05, help="Local span for confidence_threshold"),
    save_final_model: bool = typer.Option(False, help="Train final model on full data and save"),
    seasons_back: int = typer.Option(3, help="Past seasons to include for final training"),
    optuna: int = typer.Option(0, help="Run Optuna with N trials instead of grid (requires optuna)"),
):
    """Run hyperparameter tuning (wrapper around experiments/auto_tune.py)."""
    import sys
    import subprocess
    from pathlib import Path

    # Locate experiments/auto_tune.py relative to this file
    pkg_root = Path(__file__).resolve().parents[2]  # repo root
    autotune_path = pkg_root / "experiments" / "auto_tune.py"
    if not autotune_path.exists():
        typer.echo(f"Could not find {autotune_path}. Run from a checkout with experiments present.")
        raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        str(autotune_path),
        "--max-trials", str(max_trials),
        "--n-splits", str(n_splits),
        "--progress-interval", str(progress_interval),
        "--objective", str(objective),
        "--n-jobs", str(n_jobs),
        "--omp-threads", str(omp_threads),
    ]
    if refine:
        cmd.append("--refine")
    cmd += [
        "--refine-top-k", str(refine_top_k),
        "--refine-steps", str(refine_steps),
        "--span-ml-weight", str(span_ml_weight),
        "--span-prob-alpha", str(span_prob_alpha),
        "--span-min-lambda", str(span_min_lambda),
        "--span-goal-temp", str(span_goal_temp),
        "--span-conf-thr", str(span_conf_thr),
    ]
    if save_final_model:
        cmd.append("--save-final-model")
    cmd += ["--seasons-back", str(seasons_back)]
    if optuna and optuna > 0:
        cmd += ["--optuna", str(optuna)]

    # Stream output
    proc = subprocess.Popen(cmd, cwd=str(pkg_root))
    proc.wait()
    raise typer.Exit(code=proc.returncode)


if __name__ == "__main__":
    app()


