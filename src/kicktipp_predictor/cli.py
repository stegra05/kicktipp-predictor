import typer

app = typer.Typer(help="Kicktipp Predictor CLI")


@app.command()
def train(
    config_path: str = "config/best_params.json",
):
    from kicktipp_predictor.models.train import run_training
    run_training(config_path)


@app.command()
def predict(
    record: bool = typer.Option(False, help="Record predictions for performance tracking"),
    days: int = typer.Option(7, help="Days ahead to predict"),
    matchday: int | None = typer.Option(None, help="Specific matchday to predict"),
    update_results: bool = typer.Option(False, help="Update previous predictions with results"),
):
    from kicktipp_predictor.models.predict import run_predictions
    run_predictions(record=record, days=days, matchday=matchday, update_results=update_results)


@app.command()
def evaluate(season: bool = typer.Option(False, help="Evaluate over entire current season")):
    from kicktipp_predictor.models.evaluate import run_evaluation
    run_evaluation(season=season)


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


