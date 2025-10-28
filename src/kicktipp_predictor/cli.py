import typer

app = typer.Typer(help="Kicktipp Predictor CLI")


@app.command()
def train(
    seasons_back: int = typer.Option(
        5, help="Number of past seasons to use for training"
    ),
):
    """
    Train the match predictor on historical data.

    This command fetches historical match data, creates features, and trains the
    goal difference predictor. The trained model is then saved to disk.

    Args:
        seasons_back: The number of past seasons to use for training.
            Defaults to 5.

    Example:
        To train the model on the last 3 seasons of data:
        $ kicktipp-predictor train --seasons-back 3
    """
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
    """
    Make predictions for upcoming matches.

    This command loads the trained model and makes predictions for upcoming
    matches. You can specify a number of days ahead to predict or a specific
    matchday.

    Args:
        days: The number of days ahead to predict. Defaults to 7.
        matchday: The specific matchday to predict. If provided, 'days' is
            ignored.

    Example:
        To predict matches for the next 10 days:
        $ kicktipp-predictor predict --days 10

        To predict matches for matchday 12:
        $ kicktipp-predictor predict --matchday 12
    """
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
    ),
    season: int | None = typer.Option(
        None, help="Season to evaluate (e.g., 2024 for 2024/2025). Defaults to current season."
    ),
):
    """
    Evaluate performance across a season using expanding-window retraining.

    This command simulates a full season of predictions, retraining the model
    at a specified frequency to provide a realistic performance evaluation.

    Args:
        retrain_every: The number of matchdays after which the model is
            retrained. Defaults to 1.
        season: The season to evaluate. Defaults to the current season.

    Example:
        To evaluate the model on the 2023/2024 season, retraining every 3
        matchdays:
        $ kicktipp-predictor evaluate --season 2023 --retrain-every 3
    """
    from kicktipp_predictor.evaluate import run_season_dynamic_evaluation

    print("=" * 80)
    print("MODEL EVALUATION (Dynamic Season)")
    print("=" * 80)
    print()


    # Run dynamic season evaluation
    run_season_dynamic_evaluation(retrain_every=retrain_every, season=season)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    """
    Launch the Flask web application.

    This command starts a local web server to provide a user interface for the
    Kicktipp Predictor.

    Args:
        host: The hostname to bind to. Defaults to '127.0.0.1'.
        port: The port to listen on. Defaults to 8000.

    Example:
        To run the web server on all available network interfaces on port 5000:
        $ kicktipp-predictor web --host 0.0.0.0 --port 5000
    """
    from kicktipp_predictor.web import create_app

    flask_app = create_app()
    flask_app.run(host=host, port=port)

@app.command()
def tune(
    n_trials: int = typer.Option(100, help="Number of Optuna trials to run"),
    seasons_back: int = typer.Option(5, help="Number of seasons back for training"),
    storage: str | None = typer.Option(
        None,
        help="Optuna storage URL (e.g., sqlite:///path/to/optuna.db). Defaults to project data dir.",
    ),
    study_name: str = typer.Option("gd_v3_tuning", help="Optuna study name"),
    timeout: int | None = typer.Option(None, help="Timeout in seconds for optimize()"),
):
    """
    Run Optuna multi-objective tuning for the V3 goal-difference model.

    This command uses Optuna to search for the best hyperparameters for the
    goal difference model. The results are saved to
    `src/kicktipp_predictor/config/best_params.yaml`.

    Args:
        n_trials: The number of trials to run. Defaults to 100.
        seasons_back: The number of past seasons to use for training the
            tuning model. Defaults to 5.
        storage: The Optuna storage URL. Defaults to a SQLite database in the
            project's data directory.
        study_name: The name of the Optuna study.
        timeout: An optional timeout in seconds for the optimization process.

    Example:
        To run 200 tuning trials with a timeout of 1 hour:
        $ kicktipp-predictor tune --n-trials 200 --timeout 3600
    """
    from kicktipp_predictor.tune import run_tuning

    print("=" * 80)
    print("OPTUNA TUNING")
    print("=" * 80)
    print()

    try:
        run_tuning(
            n_trials=n_trials,
            seasons_back=seasons_back,
            storage=storage,
            study_name=study_name,
            timeout=timeout,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
