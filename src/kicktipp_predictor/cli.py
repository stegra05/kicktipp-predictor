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
    print(
        f"Created {len(features_df)} training samples with {len(features_df.columns)} columns"
    )

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
    workers: int = typer.Option(
        1, help="Process workers for scoreline selection ( >1 enables parallelism)"
    ),
    prob_source: str = typer.Option(
        "hybrid", help="Outcome prob source: classifier|poisson|hybrid"
    ),
    hybrid_poisson_weight: float | None = typer.Option(
        None, help="When prob_source=hybrid: weight of Poisson probabilities [0,1] (default from config)"
    ),
    proba_grid_max_goals: int = typer.Option(
        12, help="Grid cap for Poisson-derived probabilities (not scoreline grid)"
    ),
    poisson_draw_rho: float = typer.Option(
        0.0, help="Diagonal bump for draws in Poisson probs: multiply diag by exp(rho)"
    ),
):
    """Make predictions for upcoming matches."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor

    print("=" * 80)
    print("MATCH PREDICTIONS")
    print("=" * 80)
    print()

    # Apply probability-source options to config
    from kicktipp_predictor.config import get_config

    cfg = get_config()
    cfg.model.prob_source = str(prob_source).strip().lower()
    if hybrid_poisson_weight is not None:
        cfg.model.hybrid_poisson_weight = float(hybrid_poisson_weight)
    cfg.model.proba_grid_max_goals = int(proba_grid_max_goals)
    cfg.model.poisson_draw_rho = float(poisson_draw_rho)
    # prior_blend_alpha applies only when prob_source=classifier

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
    features_df = loader.create_prediction_features(
        upcoming_matches, historical_matches
    )

    if len(features_df) == 0:
        print("Could not create features (insufficient data). Try a later matchday.")
        return

    # Make predictions
    predictions = predictor.predict(features_df, workers=workers)

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

    # Apply probability-source options to config
    from kicktipp_predictor.config import get_config

    cfg = get_config()
    cfg.model.prob_source = str(prob_source).strip().lower()
    if hybrid_poisson_weight is not None:
        cfg.model.hybrid_poisson_weight = float(hybrid_poisson_weight)
    cfg.model.proba_grid_max_goals = int(proba_grid_max_goals)
    cfg.model.poisson_draw_rho = float(poisson_draw_rho)

    # Run dynamic season evaluation
    run_season_dynamic_evaluation(retrain_every=retrain_every)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    from kicktipp_predictor.web.app import app as flask_app

    flask_app.run(host=host, port=port)

if __name__ == "__main__":
    app()
