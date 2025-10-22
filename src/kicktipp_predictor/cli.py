import typer

app = typer.Typer(help="Kicktipp Predictor CLI")


@app.command()
def train(
    seasons_back: int = typer.Option(
        3, help="Number of past seasons to use for training"
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
        "classifier", help="Outcome prob source: classifier|poisson|hybrid"
    ),
    hybrid_poisson_weight: float = typer.Option(
        0.5, help="When prob_source=hybrid: weight of Poisson probabilities [0,1]"
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
        confidence = pred["confidence"]

        print(f"\n{home} vs {away}")
        print(f"  Predicted Score: {score} ({outcome})")
        print(
            f"  Probabilities: H={pred['home_win_probability']:.2%} "
            f"D={pred['draw_probability']:.2%} A={pred['away_win_probability']:.2%}"
        )
        print(f"  Confidence: {confidence:.3f}")


@app.command()
def evaluate(
    detailed: bool = typer.Option(
        False, help="Run detailed evaluation with calibration and plots"
    ),
    season: bool = typer.Option(
        False, help="Evaluate performance across the current season (finished matches)"
    ),
    dynamic: bool = typer.Option(
        False, help="Enable expanding-window retraining during season evaluation"
    ),
    retrain_every: int = typer.Option(
        1, help="Retrain every N matchdays when --dynamic is set"
    ),
    prob_source: str = typer.Option(
        "classifier", help="Outcome prob source: classifier|poisson|hybrid"
    ),
    hybrid_poisson_weight: float = typer.Option(
        0.5, help="When prob_source=hybrid: weight of Poisson probabilities [0,1]"
    ),
    proba_grid_max_goals: int = typer.Option(
        12, help="Grid cap for Poisson-derived probabilities (not scoreline grid)"
    ),
    poisson_draw_rho: float = typer.Option(
        0.0, help="Diagonal bump for draws in Poisson probs: multiply diag by exp(rho)"
    ),
):
    """Evaluate predictor performance on test data."""
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.evaluate import (
        print_evaluation_report,
        run_evaluation,
        simple_benchmark,
    )
    from kicktipp_predictor.predictor import MatchPredictor

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print()

    # Apply probability-source options to config
    from kicktipp_predictor.config import get_config

    cfg = get_config()
    cfg.model.prob_source = str(prob_source).strip().lower()
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

    if season:
        run_evaluation(season=True, dynamic=dynamic, retrain_every=retrain_every)
        return

    if detailed:
        run_evaluation(season=False)
        return

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
    benchmark = simple_benchmark(test_df, strategy="home_win")

    # Print report
    print_evaluation_report(metrics, benchmark)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    from kicktipp_predictor.web.app import app as flask_app

    flask_app.run(host=host, port=port)


@app.command()
def tune(
    n_trials: int = typer.Option(100, help="Total Optuna trials across all workers"),
    n_splits: int = typer.Option(3, help="TimeSeriesSplit folds"),
    workers: int = typer.Option(1, help="Number of parallel worker processes"),
    objective: str = typer.Option(
        "ppg",
        help="Objective (PPG recommended): ppg|ppg_unweighted|logloss|brier|balanced_accuracy|accuracy|rps",
    ),
    direction: str = typer.Option("auto", help="Direction: auto|maximize|minimize"),
    compare: str | None = typer.Option(
        None,
        help="Comma-separated objectives to compare; overrides --objective (not recommended with balanced trainer)",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose inner logs during tuning"),
    storage: str | None = typer.Option(
        None,
        help="Optuna storage URL for multi-process tuning (e.g., sqlite:////abs/path/study.db?timeout=60)",
    ),
    study_name: str | None = typer.Option(
        None, help="Optuna study name when using storage"
    ),
    pruner: str = typer.Option("median", help="Pruner: none|median|hyperband"),
    pruner_startup_trials: int = typer.Option(
        20, help="Trials before enabling pruning (median)"
    ),
):
    """Run Optuna tuning with selectable objectives and optional compare mode.

    When workers > 1, a shared storage is required. In compare mode, separate sqlite files are
    automatically created per objective if a sqlite storage URL is provided.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Locate experiments/auto_tune.py relative to this file
    pkg_root = Path(__file__).resolve().parents[2]  # repo root
    autotune_path = pkg_root / "experiments" / "auto_tune.py"
    if not autotune_path.exists():
        typer.echo(
            f"Could not find {autotune_path}. Run from a checkout with experiments present."
        )
        raise typer.Exit(code=1)

    if workers < 1:
        workers = 1

    # Require storage for multi-worker coordination
    if workers > 1 and not storage:
        typer.echo(
            "When --workers > 1, you must provide --storage (e.g., sqlite:////abs/path/study.db?timeout=60)"
        )
        raise typer.Exit(code=2)

    def base_args(trials: int) -> list[str]:
        args = [
            sys.executable,
            str(autotune_path),
            "--n-trials",
            str(max(0, int(trials))),
            "--n-splits",
            str(n_splits),
            "--verbose" if verbose else None,
        ]
        args = [a for a in args if a is not None]
        # propagate inner options
        if pruner:
            args += ["--pruner", pruner]
        if pruner_startup_trials is not None:
            args += ["--pruner-startup-trials", str(pruner_startup_trials)]
        if direction:
            args += ["--direction", direction]
        return args

    # Helper: build sqlite URL per objective
    def sqlite_url_for(obj: str) -> str | None:
        if not storage:
            return None
        if not storage.startswith("sqlite:"):
            return storage
        # split before query
        base, *q = storage.split("?", 1)
        query = ("?" + q[0]) if q else ""
        # derive filename suffix
        if base.endswith(".db"):
            return base.replace(".db", f"_{obj}.db") + query
        return base + f"_{obj}" + query

    env = os.environ.copy()
    # tell worker not to delete DB when coordinated by CLI
    env["KTP_TUNE_COORDINATED"] = "1"

    # Determine objectives list
    objectives = (
        [o.strip() for o in compare.split(",") if o.strip()] if compare else [objective]
    )

    # Serial execution (workers==1)
    if workers == 1:
        procs: list[subprocess.Popen] = []
        exit_code = 0
        for obj in objectives:
            cmd = base_args(n_trials) + ["--objective", obj]
            # storage handling
            url = sqlite_url_for(obj)
            if url:
                cmd += ["--storage", url]
            if study_name:
                cmd += [
                    "--study-name",
                    f"{study_name}-{obj}" if compare else study_name,
                ]
            p = subprocess.Popen(cmd, cwd=str(pkg_root), env=env)
            procs.append(p)
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode
        raise typer.Exit(code=exit_code)

    # Multi-worker execution requires storage; for compare we iterate objectives
    if workers > 1 and compare:
        # For each objective, run a full multi-worker round using per-objective sqlite
        for obj in objectives:
            url = sqlite_url_for(obj)
            if not url:
                typer.echo(
                    "When --workers > 1, a storage URL is required (sqlite or RDBMS)"
                )
                raise typer.Exit(code=2)
            # 1) Initialize the study/db with a 0-trial run
            init_cmd = base_args(0) + ["--objective", obj, "--storage", url]
            if study_name:
                init_cmd += ["--study-name", f"{study_name}-{obj}"]
            init_proc = subprocess.Popen(init_cmd, cwd=str(pkg_root), env=env)
            init_proc.wait()
            if init_proc.returncode != 0:
                typer.echo(
                    f"Failed to initialize storage for objective {obj}. Aborting."
                )
                raise typer.Exit(code=init_proc.returncode)

            # 2) Split total trials evenly across workers
            base = n_trials // workers
            rem = n_trials % workers
            allocations = [base + (1 if i < rem else 0) for i in range(workers)]

            # 3) Launch worker subprocesses
            procs: list[subprocess.Popen] = []
            for trials in allocations:
                if trials <= 0:
                    continue
                cmd = base_args(trials) + ["--objective", obj, "--storage", url]
                if study_name:
                    cmd += ["--study-name", f"{study_name}-{obj}"]
                p = subprocess.Popen(cmd, cwd=str(pkg_root), env=env)
                procs.append(p)

            # 4) Wait for all workers
            exit_code = 0
            for p in procs:
                p.wait()
                if p.returncode != 0 and exit_code == 0:
                    exit_code = p.returncode
            if exit_code != 0:
                raise typer.Exit(code=exit_code)
        raise typer.Exit(code=0)

    # Multi-worker single objective
    if workers > 1:
        if not storage:
            typer.echo(
                "When --workers > 1, a storage URL is required (sqlite or RDBMS)"
            )
            raise typer.Exit(code=2)
        # 1) Initialize the study/db with a 0-trial run
        init_cmd = base_args(0) + ["--objective", objectives[0], "--storage", storage]
        if study_name:
            init_cmd += ["--study-name", study_name]
        init_proc = subprocess.Popen(init_cmd, cwd=str(pkg_root), env=env)
        init_proc.wait()
        if init_proc.returncode != 0:
            typer.echo("Failed to initialize study/storage. Aborting.")
            raise typer.Exit(code=init_proc.returncode)

        # 2) Split total trials evenly across workers
        base = n_trials // workers
        rem = n_trials % workers
        allocations = [base + (1 if i < rem else 0) for i in range(workers)]

        # 3) Launch worker subprocesses
        procs: list[subprocess.Popen] = []
        for trials in allocations:
            if trials <= 0:
                continue
            cmd = base_args(trials) + [
                "--objective",
                objectives[0],
                "--storage",
                storage,
            ]
            if study_name:
                cmd += ["--study-name", study_name]
            p = subprocess.Popen(cmd, cwd=str(pkg_root), env=env)
            procs.append(p)

        # 4) Wait for all workers
        exit_code = 0
        for p in procs:
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode
        raise typer.Exit(code=exit_code)


@app.command()
def shap(
    seasons_back: int = typer.Option(
        3, help="Number of seasons back to sample training-like data"
    ),
    sample: int = typer.Option(2000, help="Max samples for SHAP computation"),
):
    """Run SHAP analysis on the trained models and save summary plots."""
    from kicktipp_predictor.analysis.shap_analysis import run_shap_for_predictor
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor

    print("=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80)
    print()

    loader = DataLoader()
    predictor = MatchPredictor()

    try:
        predictor.load_models()
    except FileNotFoundError:
        print("ERROR: No trained models found. Run 'train' first.")
        raise typer.Exit(code=1)

    current = loader.get_current_season()
    start = current - max(1, seasons_back - 1)
    print(f"Loading seasons {start} to {current} for SHAP background...")
    matches = loader.fetch_historical_seasons(start, current)
    df = loader.create_features_from_matches(matches)
    if df is None or len(df) == 0:
        print("No data available for SHAP background.")
        raise typer.Exit(code=1)

    # Build X aligned to trained schema
    import pandas as pd

    X = pd.DataFrame(df, copy=True)
    # drop labels if present
    for col in ["home_score", "away_score", "goal_difference", "result"]:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)
    # Ensure columns present and ordered
    for col in predictor.feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[predictor.feature_columns].fillna(0)
    if len(X) > sample:
        X = X.sample(sample, random_state=42)

    out_dir = run_shap_for_predictor(predictor, X)
    if out_dir is None:
        print("SHAP/matplotlib not installed; install extras to enable this command.")
        raise typer.Exit(code=2)
    print(f"SHAP plots saved to: {out_dir}")


if __name__ == "__main__":
    app()
