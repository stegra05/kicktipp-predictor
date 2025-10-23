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
    retrain_every: int = typer.Option(
        1, help="Retrain every N matchdays during dynamic season evaluation"
    ),
    prob_source: str = typer.Option(
        "hybrid", help="Outcome prob source: classifier|poisson|hybrid"
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
    cfg.model.hybrid_poisson_weight = float(hybrid_poisson_weight)
    cfg.model.proba_grid_max_goals = int(proba_grid_max_goals)
    cfg.model.poisson_draw_rho = float(poisson_draw_rho)

    # Run dynamic season evaluation
    run_season_dynamic_evaluation(retrain_every=retrain_every)


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

    # Warn about SQLite limitations with high worker counts
    if workers > 1 and storage and storage.startswith("sqlite:"):
        if workers > 20:
            typer.echo(
                f"⚠️  Warning: Using {workers} workers with SQLite may cause database lock errors. "
                "Consider using fewer workers (≤20) or a PostgreSQL/MySQL database for better concurrency."
            )
        elif workers > 10:
            typer.echo(
                f"ℹ️  Note: Using {workers} workers with SQLite. For optimal performance, consider using ≤10 workers or a PostgreSQL database."
            )

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

        # Ensure optimal parameters are set for SQLite URLs to handle database locks
        if query:
            # Check if timeout is already specified
            if "timeout=" not in query:
                query += "&timeout=300"  # 5 minutes timeout
            if "check_same_thread=" not in query:
                query += "&check_same_thread=false"
            if "isolation_level=" not in query:
                query += "&isolation_level=IMMEDIATE"
        else:
            query = "?timeout=300&check_same_thread=false&isolation_level=IMMEDIATE"

        # derive filename suffix
        if base.endswith(".db"):
            return base.replace(".db", f"_{obj}.db") + query
        return base + f"_{obj}" + query

    env = os.environ.copy()
    # tell worker not to delete DB when coordinated by CLI
    env["KTP_TUNE_COORDINATED"] = "1"
    # Propagate BLAS/OMP thread caps to subprocesses to prevent OpenBLAS oversubscription
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["XGBOOST_NUM_THREADS"] = "1"

    # Serial execution (workers==1)
    if workers == 1:
        cmd = base_args(n_trials)
        url = sqlite_url_for(objective)
        if url:
            cmd += ["--storage", url]
        if study_name:
            cmd += ["--study-name", study_name]
        p = subprocess.Popen(cmd, cwd=str(pkg_root), env=env)
        p.wait()
        raise typer.Exit(code=p.returncode or 0)

    # Multi-worker execution requires storage
    url = sqlite_url_for(objective)
    if not url:
        typer.echo("When --workers > 1, a storage URL is required (sqlite or RDBMS)")
        raise typer.Exit(code=2)

    # 1) Initialize the study/db with a 0-trial run
    init_cmd = base_args(0)
    init_cmd += ["--storage", url]
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
    for worker_idx, trials in enumerate(allocations):
        if trials <= 0:
            continue
        cmd = base_args(trials) + ["--storage", url]
        if study_name:
            cmd += ["--study-name", study_name]

        # Set worker environment variables for coordination
        worker_env = env.copy()
        worker_env["OPTUNA_WORKER_ID"] = str(worker_idx)
        worker_env["OPTUNA_TOTAL_WORKERS"] = str(workers)

        p = subprocess.Popen(cmd, cwd=str(pkg_root), env=worker_env)
        procs.append(p)

    # 4) Central progress bar and summary
    exit_code = 0
    try:
        import time as _time

        import optuna as _optuna
        from optuna.importance import get_param_importances as _get_param_importances
        from optuna.visualization import (
            plot_param_importances as _plot_param_importances,
        )
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        # Silence Optuna logs for a cleaner console
        try:
            _optuna.logging.disable_default_handler()
            _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        except Exception:
            pass

        # Determine study name used by workers (defaults to tune_baseline.py default)
        study_name_local = study_name or "baseline-ppg"
        total_trials = n_trials

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} trials"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Baseline tuning (PPG)", total=total_trials)
            # Poll study storage until all workers finish
            while True:
                # Update completed count
                try:
                    study = _optuna.load_study(study_name=study_name_local, storage=url)
                    trials = study.get_trials(deepcopy=False)
                    completed = sum(
                        1
                        for t in trials
                        if t.state == _optuna.trial.TrialState.COMPLETE
                    )
                    progress.update(task_id, completed=min(completed, total_trials))
                except Exception:
                    # If storage not ready yet, just keep spinner spinning
                    pass
                # Check workers
                alive = any(p.poll() is None for p in procs)
                if not alive:
                    break
                _time.sleep(0.5)

        # Finalize and compute exit code
        for p in procs:
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode

        # Central best params and importances logging
        try:
            study = _optuna.load_study(study_name=study_name_local, storage=url)
            trials = study.get_trials(deepcopy=False)
            completed = sum(
                1 for t in trials if t.state == _optuna.trial.TrialState.COMPLETE
            )
            if completed == 0:
                typer.echo("No completed trials; skipping best params and importances.")
            else:
                # Save best params
                best_params = dict(study.best_params)
                cfg_dir = pkg_root / "config"
                cfg_dir.mkdir(parents=True, exist_ok=True)
                out_yaml = cfg_dir / "best_params_baseline.yaml"
                try:
                    import yaml as _yaml  # type: ignore
                except Exception:
                    _yaml = None
                if _yaml is not None:
                    with open(out_yaml, "w", encoding="utf-8") as f:
                        _yaml.safe_dump(best_params, f, sort_keys=True)
                    typer.echo(f"Best parameters saved to {out_yaml}")
                else:
                    import json as _json

                    with open(
                        out_yaml.with_suffix(".json"), "w", encoding="utf-8"
                    ) as f:
                        _json.dump(best_params, f, indent=2)
                    typer.echo(
                        f"Best parameters saved to {out_yaml.with_suffix('.json')}"
                    )

                # Save importances plot if enough trials
                if completed >= 2:
                    try:
                        _ = _get_param_importances(study)  # validate availability
                        fig = _plot_param_importances(study)
                        out_dir = pkg_root / "data" / "optuna"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        plot_path = out_dir / "baseline_param_importances.html"
                        fig.write_html(str(plot_path))
                        typer.echo(f"Param importances saved to {plot_path}")
                    except Exception as e:
                        typer.echo(
                            f"Warning: Could not compute/save param importances: {e}"
                        )
                else:
                    typer.echo("Not enough trials to compute importances (need ≥2).")
        except Exception:
            # Swallow central save errors; users will still have worker outputs
            pass

        raise typer.Exit(code=exit_code)
    except Exception:
        # Fallback: no rich/optuna available; just wait for workers
        exit_code = 0
        for p in procs:
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode


@app.command()
def tune_baseline(
    n_trials: int = typer.Option(100, help="Total Optuna trials across all workers"),
    seasons_back: int = typer.Option(
        5, help="Number of seasons back for training window"
    ),
    workers: int = typer.Option(1, help="Number of parallel worker processes"),
    storage: str | None = typer.Option(
        None,
        help="Optuna storage URL for multi-process tuning (e.g., sqlite:////abs/path/study.db?timeout=60)",
    ),
    study_name: str | None = typer.Option(
        None, help="Optuna study name when using storage"
    ),
    pruner: str = typer.Option("median", help="Pruner: none|median|hyperband"),
    pruner_startup_trials: int = typer.Option(
        15, help="Trials before enabling pruning (median)"
    ),
    omp_threads: int = typer.Option(1, help="Threads per worker for BLAS/OMP"),
    verbose: bool = typer.Option(False, help="Enable verbose inner logs during tuning"),
    save_plot: str | None = typer.Option(
        None, help="Path to save param importances plot (HTML)"
    ),
):
    """Run baseline Optuna tuning on a fixed train/test split (PPG objective).

    When workers > 1, a shared storage is required. Trials are evenly split across
    workers and coordinated via storage, similar to the 'tune' command.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Locate experiments/tune_baseline.py relative to this file
    pkg_root = Path(__file__).resolve().parents[2]
    baseline_path = pkg_root / "experiments" / "tune_baseline.py"
    if not baseline_path.exists():
        typer.echo(
            f"Could not find {baseline_path}. Run from a checkout with experiments present."
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

    # Warn about SQLite limitations with high worker counts
    if workers > 1 and storage and storage.startswith("sqlite:"):
        if workers > 20:
            typer.echo(
                f"⚠️  Warning: Using {workers} workers with SQLite may cause database lock errors. "
                "Consider using fewer workers (≤20) or a PostgreSQL/MySQL database for better concurrency."
            )
        elif workers > 10:
            typer.echo(
                f"ℹ️  Note: Using {workers} workers with SQLite. For optimal performance, consider using ≤10 workers or a PostgreSQL database."
            )

    def base_args(trials: int) -> list[str]:
        args = [
            sys.executable,
            str(baseline_path),
            "--n-trials",
            str(max(0, int(trials))),
            "--seasons-back",
            str(seasons_back),
            "--omp-threads",
            str(omp_threads),
            "--verbose" if verbose else None,
        ]
        args = [a for a in args if a is not None]
        # propagate inner options
        if pruner:
            args += ["--pruner", pruner]
        if pruner_startup_trials is not None:
            args += ["--pruner-startup-trials", str(pruner_startup_trials)]
        if save_plot:
            args += ["--save-plot", str(save_plot)]
        return args

    # Helper: build sqlite URL with concurrency parameters
    def sqlite_url() -> str | None:
        if not storage:
            return None
        if not storage.startswith("sqlite:"):
            return storage
        base, *q = storage.split("?", 1)
        query = ("?" + q[0]) if q else ""

        # Ensure optimal parameters are set for SQLite URLs to handle database locks
        if query:
            if "timeout=" not in query:
                query += "&timeout=300"  # 5 minutes timeout
            if "check_same_thread=" not in query:
                query += "&check_same_thread=false"
            if "isolation_level=" not in query:
                query += "&isolation_level=IMMEDIATE"
        else:
            query = "?timeout=300&check_same_thread=false&isolation_level=IMMEDIATE"
        return base + query

    env = os.environ.copy()
    # tell worker not to delete DB when coordinated by CLI
    env["KTP_TUNE_COORDINATED"] = "1"
    # Propagate BLAS/OMP thread caps to subprocesses to prevent OpenBLAS oversubscription
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    env["NUMEXPR_NUM_THREADS"] = str(omp_threads)
    env["XGBOOST_NUM_THREADS"] = str(omp_threads)

    # Serial execution (workers==1)
    if workers == 1:
        cmd = base_args(n_trials)
        url = sqlite_url()
        if url:
            cmd += ["--storage", url]
        if study_name:
            cmd += ["--study-name", study_name]
        p = subprocess.Popen(cmd, cwd=str(pkg_root), env=env)
        p.wait()
        raise typer.Exit(code=p.returncode or 0)

    # Multi-worker execution requires storage
    url = sqlite_url()
    if not url:
        typer.echo("When --workers > 1, a storage URL is required (sqlite or RDBMS)")
        raise typer.Exit(code=2)

    # 1) Initialize the study/db with a 0-trial run
    init_cmd = base_args(0)
    init_cmd += ["--storage", url]
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
    for worker_idx, trials in enumerate(allocations):
        if trials <= 0:
            continue
        cmd = base_args(trials) + ["--storage", url]
        if study_name:
            cmd += ["--study-name", study_name]

        # Set worker environment variables for coordination
        worker_env = env.copy()
        worker_env["OPTUNA_WORKER_ID"] = str(worker_idx)
        worker_env["OPTUNA_TOTAL_WORKERS"] = str(workers)

        p = subprocess.Popen(cmd, cwd=str(pkg_root), env=worker_env)
        procs.append(p)

    # 4) Central progress bar and summary
    exit_code = 0
    try:
        import time as _time

        import optuna as _optuna
        from optuna.importance import get_param_importances as _get_param_importances
        from optuna.visualization import (
            plot_param_importances as _plot_param_importances,
        )
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        # Silence Optuna logs for a cleaner console
        try:
            _optuna.logging.disable_default_handler()
            _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        except Exception:
            pass

        # Determine study name used by workers (defaults to tune_baseline.py default)
        study_name_local = study_name or "baseline-ppg"
        total_trials = n_trials

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} trials"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Baseline tuning (PPG)", total=total_trials)
            # Poll study storage until all workers finish
            while True:
                # Update completed count
                try:
                    study = _optuna.load_study(study_name=study_name_local, storage=url)
                    trials = study.get_trials(deepcopy=False)
                    completed = sum(
                        1
                        for t in trials
                        if t.state == _optuna.trial.TrialState.COMPLETE
                    )
                    progress.update(task_id, completed=min(completed, total_trials))
                except Exception:
                    # If storage not ready yet, just keep spinner spinning
                    pass
                # Check workers
                alive = any(p.poll() is None for p in procs)
                if not alive:
                    break
                _time.sleep(0.5)

        # Finalize and compute exit code
        for p in procs:
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode

        # Central best params and importances logging
        try:
            study = _optuna.load_study(study_name=study_name_local, storage=url)
            trials = study.get_trials(deepcopy=False)
            completed = sum(
                1 for t in trials if t.state == _optuna.trial.TrialState.COMPLETE
            )
            if completed == 0:
                typer.echo("No completed trials; skipping best params and importances.")
            else:
                # Save best params
                best_params = dict(study.best_params)
                cfg_dir = pkg_root / "config"
                cfg_dir.mkdir(parents=True, exist_ok=True)
                out_yaml = cfg_dir / "best_params_baseline.yaml"
                try:
                    import yaml as _yaml  # type: ignore
                except Exception:
                    _yaml = None
                if _yaml is not None:
                    with open(out_yaml, "w", encoding="utf-8") as f:
                        _yaml.safe_dump(best_params, f, sort_keys=True)
                    typer.echo(f"Best parameters saved to {out_yaml}")
                else:
                    import json as _json

                    with open(
                        out_yaml.with_suffix(".json"), "w", encoding="utf-8"
                    ) as f:
                        _json.dump(best_params, f, indent=2)
                    typer.echo(
                        f"Best parameters saved to {out_yaml.with_suffix('.json')}"
                    )

                # Save importances plot if enough trials
                if completed >= 2:
                    try:
                        _ = _get_param_importances(study)  # validate availability
                        fig = _plot_param_importances(study)
                        out_dir = pkg_root / "data" / "optuna"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        plot_path = (
                            Path(save_plot)
                            if save_plot
                            else (out_dir / "baseline_param_importances.html")
                        )
                        fig.write_html(str(plot_path))
                        typer.echo(f"Param importances saved to {plot_path}")
                    except Exception as e:
                        typer.echo(
                            f"Warning: Could not compute/save param importances: {e}"
                        )
                else:
                    typer.echo("Not enough trials to compute importances (need ≥2).")
        except Exception:
            # Swallow central save errors; users will still have worker outputs
            pass

        raise typer.Exit(code=exit_code)
    except Exception:
        # Fallback: no rich/optuna available; just wait for workers
        exit_code = 0
        for p in procs:
            p.wait()
            if p.returncode != 0 and exit_code == 0:
                exit_code = p.returncode
        raise typer.Exit(code=exit_code)


@app.command()
def shap(
    seasons_back: int = typer.Option(
        5, help="Number of seasons back to sample training-like data"
    ),
    sample: int = typer.Option(2000, help="Max samples for SHAP computation"),
):
    """Run SHAP analysis on the trained models and save summary plots."""
    # Print header immediately, before any heavy imports
    print("=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80)
    print()

    # Lazily import dependencies for this command
    print("Loading dependencies for SHAP analysis (this can take a moment)...")

    from kicktipp_predictor.data import DataLoader

    print("Loaded data")
    from kicktipp_predictor.predictor import MatchPredictor

    print("Loaded predictor")
    from kicktipp_predictor.models.shap_analysis import run_shap_for_predictor

    print("Loaded shap_analysis")
    print("Dependencies loaded.")

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
