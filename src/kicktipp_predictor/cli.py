import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

app = typer.Typer(help="Kicktipp Predictor CLI (V4 Cascaded Architecture)")
console = Console()


@app.command()
def train(
    seasons_back: int = typer.Option(
        5, help="Number of past seasons to use for training"
    ),
    validate: bool = typer.Option(False, help="Run cross-validation diagnostics after training"),
):
    """Train the V4 cascaded match predictor on historical data.

    Trains two binary classifiers in cascade:
    - Draw vs NotDraw (gatekeeper)
    - HomeWin vs AwayWin (finisher conditioned on NotDraw)
    """
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import CascadedPredictor

    console.print(
        Panel(
            "[bold cyan]TRAINING MATCH PREDICTOR[/bold cyan]\n"
            "[dim]Training two binary classifiers in cascade: Draw vs NotDraw, HomeWin vs AwayWin[/dim]",
            border_style="cyan",
            expand=False,
        )
    )

    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    loader = DataLoader()
    current_season = loader.get_current_season()
    start_season = current_season - seasons_back

    console.print(f"[dim]Fetching seasons {start_season} to {current_season}...[/dim]")
    all_matches = loader.fetch_historical_seasons(start_season, current_season)
    console.print(f"[green]✓ Loaded {len(all_matches)} matches[/green]")

    # Create features
    console.print("[cyan]Creating features...[/cyan]")
    features_df = loader.create_features_from_matches(all_matches)
    console.print(
        f"[green]✓ Created {len(features_df)} training samples with {len(features_df.columns)} columns[/green]"
    )

    # Train predictor
    console.print("")
    predictor = CascadedPredictor()
    predictor.train(features_df)

    if validate:
        console.print("")
        console.print("[cyan]Running cross-validation diagnostics...[/cyan]")
        predictor.run_cv_diagnostics(features_df)

    # Save models
    console.print("")
    console.print("[cyan]Saving models...[/cyan]")
    predictor.save_models()
    console.print("[green]✓ Models saved successfully[/green]")

    console.print("")
    console.print(
        Panel(
            "[bold green]TRAINING COMPLETE[/bold green]",
            border_style="green",
            expand=False,
        )
    )


@app.command()
def predict(
    days: int = typer.Option(7, help="Days ahead to predict"),
    matchday: int | None = typer.Option(None, help="Specific matchday to predict"),
):
    """Make predictions for upcoming matches using the V4 cascaded predictor.

    Produces calibrated H/D/A probabilities by combining draw and win-stage outputs.
    """
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import CascadedPredictor

    console.print(
        Panel(
            "[bold yellow]MATCH PREDICTIONS[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )

    # Load data
    loader = DataLoader()
    predictor = CascadedPredictor()

    # Load trained models
    try:
        console.print("[cyan]Loading trained models...[/cyan]")
        predictor.load_models()
        console.print("[green]✓ Models loaded successfully[/green]")
    except FileNotFoundError:
        console.print("[bold red]ERROR: No trained models found. Run 'train' command first.[/bold red]")
        raise typer.Exit(code=1)

    # Get upcoming matches
    if matchday is not None:
        console.print(f"[dim]Getting matches for matchday {matchday}...[/dim]")
        upcoming_matches = loader.fetch_matchday(matchday)
    else:
        console.print(f"[dim]Getting upcoming matches (next {days} days)...[/dim]")
        upcoming_matches = loader.get_upcoming_matches(days=days)

    if not upcoming_matches:
        console.print("[yellow]No upcoming matches found.[/yellow]")
        return

    console.print(f"[green]✓ Found {len(upcoming_matches)} upcoming matches[/green]")

    # Get historical data for context
    current_season = loader.get_current_season()
    historical_matches = loader.fetch_season_matches(current_season)

    # Create features
    console.print("[cyan]Creating prediction features...[/cyan]")
    features_df = loader.create_prediction_features(
        upcoming_matches, historical_matches
    )

    if len(features_df) == 0:
        console.print("[red]Could not create features (insufficient data). Try a later matchday.[/red]")
        return

    # Make predictions
    predictions = predictor.predict(features_df)

    # Display predictions in a table
    console.print("")
    table = Table(title="Predictions", show_header=True, header_style="bold magenta")
    table.add_column("Match", style="cyan", width=30)
    table.add_column("Score", justify="center")
    table.add_column("Outcome", justify="center", style="bold")
    table.add_column("H %", justify="right")
    table.add_column("D %", justify="right")
    table.add_column("A %", justify="right")
    
    for pred in predictions:
        home = pred.get("home_team", "?")
        away = pred.get("away_team", "?")
        match_display = f"{home} vs {away}"
        score = f"{pred['predicted_home_score']}-{pred['predicted_away_score']}"
        outcome = pred.get("predicted_outcome") or pred.get("predicted_result")
        
        # Color the outcome
        if outcome == "H":
            outcome_display = f"[green]{outcome}[/green]"
        elif outcome == "A":
            outcome_display = f"[red]{outcome}[/red]"
        else:
            outcome_display = f"[yellow]{outcome}[/yellow]"
        
        table.add_row(
            match_display,
            score,
            outcome_display,
            f"{pred['home_win_probability']:.1%}",
            f"{pred['draw_probability']:.1%}",
            f"{pred['away_win_probability']:.1%}",
        )
    
    console.print(table)


@app.command()
def evaluate(
    retrain_every: int = typer.Option(
        1, help="Retrain every N matchdays during dynamic season evaluation"
    )
):
    """Evaluate season performance (expanding-window) with the V4 cascaded predictor.

    Retrains periodically and reports accuracy, log_loss, RPS, Brier, and draw-rate realism.
    """
    from kicktipp_predictor.evaluate import run_season_dynamic_evaluation

    console.print(
        Panel(
            "[bold blue]MODEL EVALUATION (Dynamic Season)[/bold blue]\n"
            f"[dim]Retrain every {retrain_every} matchday(s)[/dim]",
            border_style="blue",
            expand=False,
        )
    )

    # Run dynamic season evaluation
    run_season_dynamic_evaluation(retrain_every=retrain_every)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    """Start the web application for match predictions."""
    from kicktipp_predictor.web.app import app as flask_app

    console.print(
        Panel(
            f"[bold green]Starting web application...[/bold green]\n"
            f"[dim]Access at: http://{host}:{port}[/dim]",
            border_style="green",
            expand=False,
        )
    )
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
    model_to_tune: str = typer.Option(
        "both",
        help="Which model to tune: draw, win, or both",
    ),
    draw_metric: str = typer.Option(
        "roc_auc",
        help="Phase 1 draw metric: roc_auc or f1",
    ),
    win_metric: str = typer.Option(
        "accuracy",
        help="Phase 2 win metric: accuracy or log_loss",
    ),
    parallel: bool = typer.Option(False, help="Run tuning in parallel across multiple workers"),
    workers: int = typer.Option(0, help="Number of parallel workers (0 = auto)"),
    bench_trials: int = typer.Option(0, help="Optional small sequential benchmark trial count"),
    log_level: str = typer.Option("warning", help="Logging level: debug/info/warning/error"),
    reset_storage: bool = typer.Option(
        False,
        help="Reset Optuna storage before tuning (WARNING: This will delete existing studies)",
    ),
):
    """Run Optuna tuning for the V4 cascaded predictor (sequential draw → win).

    Supports phase-specific metrics and optional parallel workers.
    """
    console.rule("[bold]OPTUNA TUNING[/bold]")

    try:
        # Explicit storage reset control with confirmation
        if reset_storage:
            console.print("[yellow]WARNING:[/yellow] Resetting storage will [bold]delete[/bold] existing Optuna studies.")
            typer.confirm("Proceed with storage reset?", abort=True)
        if parallel:
            from kicktipp_predictor.tune import run_tuning_v4_parallel
            run_tuning_v4_parallel(
                n_trials=n_trials,
                seasons_back=seasons_back,
                storage=storage,
                study_name=(study_name if study_name else "v4_cascaded_parallel"),
                timeout=timeout,
                model_to_tune=model_to_tune,
                draw_metric=draw_metric,
                win_metric=win_metric,
                workers=(None if workers <= 0 else workers),
                bench_trials=(None if bench_trials <= 0 else bench_trials),
                log_level=log_level,
                reset_storage=reset_storage,
            )
        else:
            from kicktipp_predictor.tune import run_tuning_v4_sequential
            run_tuning_v4_sequential(
                n_trials=n_trials,
                seasons_back=seasons_back,
                storage=storage,
                study_name=(study_name if study_name else "v4_cascaded_sequential"),
                timeout=timeout,
                model_to_tune=model_to_tune,
                draw_metric=draw_metric,
                win_metric=win_metric,
                reset_storage=reset_storage,
            )
    except RuntimeError as e:
        console.print(f"[red]ERROR[/red]: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
