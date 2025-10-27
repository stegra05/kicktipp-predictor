import typer

app = typer.Typer(help="Kicktipp Predictor CLI")


@app.command()
def train(
    seasons_back: int = typer.Option(
        5, help="Number of past seasons to use for training"
    ),
):
    """Train the match predictor on historical data."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import GoalDifferencePredictor

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
    print(
        f"Created {len(features_df)} training samples with {len(features_df.columns)} columns"
    )

    # Train predictor
    print("\nTraining predictor...")
    predictor = GoalDifferencePredictor()
    predictor.train(features_df)

    # Save models
    print("\nSaving models...")
    predictor.save_model()

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
    from kicktipp_predictor.predictor import GoalDifferencePredictor

    print("=" * 80)
    print("MATCH PREDICTIONS")
    print("=" * 80)
    print()


    # Load data
    loader = DataLoader()
    predictor = GoalDifferencePredictor()

    # Load trained models
    try:
        predictor.load_model()
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
    features_df = loader.create_prediction_features(
        upcoming_matches, historical_matches
    )

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
        home = pred["home_team"]
        away = pred["away_team"]
        score = f"{pred['predicted_home_score']}-{pred['predicted_away_score']}"
        outcome = pred["predicted_result"]

        print(f"\n{home} vs {away}")
        print(f"  Predicted Score: {score} ({outcome})")
        print(
            f"  Probabilities: H={pred['home_win_probability']:.2%} "
            f"D={pred['draw_probability']:.2%} A={pred['away_win_probability']:.2%}"
        )


@app.command()
def evaluate(
    retrain_every: int = typer.Option(
        1, help="Retrain every N matchdays during dynamic season evaluation"
    )
):
    """Evaluate performance across the current season using expanding-window retraining."""
    from kicktipp_predictor.evaluate import run_season_dynamic_evaluation

    print("=" * 80)
    print("MODEL EVALUATION (Dynamic Season)")
    print("=" * 80)
    print()


    # Run dynamic season evaluation
    run_season_dynamic_evaluation(retrain_every=retrain_every)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    from kicktipp_predictor.web.app import app as flask_app

    flask_app.run(host=host, port=port)

@app.command()
def tune(
    n_trials: int = typer.Option(100, help="Number of Optuna trials to run"),
    seasons_back: int = typer.Option(5, help="Number of seasons back for training"),
    storage: str | None = typer.Option(
        None,
        help="Optuna storage URL (e.g., sqlite:///path/to/optuna.db). Defaults to project data dir.",
    ),
    study_name: str = typer.Option("v4_cascaded_sequential", help="Optuna study name base"),
    timeout: int | None = typer.Option(None, help="Timeout in seconds for optimize()"),
    arch: str = typer.Option("v4", help="Model architecture to tune: v3 or v4"),
    model_to_tune: str = typer.Option(
        "both",
        help="Which model to tune for v4: draw, win, or both",
    ),
    draw_metric: str = typer.Option(
        "roc_auc",
        help="Phase 1 draw metric: roc_auc or f1",
    ),
    win_metric: str = typer.Option(
        "accuracy",
        help="Phase 2 win metric: accuracy or log_loss",
    ),
):
    """Run Optuna tuning.

    - v4 (default): Sequential tuning for CascadedPredictor (Phase 1 draw, Phase 2 win).
    - v3: Legacy multi-objective tuning for GoalDifferencePredictor.

    When selecting v4 with model_to_tune=win, Phase 2 uses fixed draw_* params
    from best_params.yaml (produced in Phase 1), ensuring sequential dependency.
    """
    from kicktipp_predictor.tune import run_tuning, run_tuning_v4_sequential

    print("=" * 80)
    print("OPTUNA TUNING")
    print("=" * 80)
    print()

    try:
        if arch.lower() == "v3":
            run_tuning(
                n_trials=n_trials,
                seasons_back=seasons_back,
                storage=storage,
                study_name=study_name,
                timeout=timeout,
            )
        else:
            run_tuning_v4_sequential(
                n_trials=n_trials,
                seasons_back=seasons_back,
                storage=storage,
                study_name=study_name,
                timeout=timeout,
                model_to_tune=model_to_tune,
                draw_metric=draw_metric,
                win_metric=win_metric,
            )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
