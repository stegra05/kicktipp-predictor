"""Season dynamic evaluation (expanding-window) with rich console output.

This module keeps only the season-long dynamic evaluation:
- Expanding-window retraining (configurable cadence)
- Rich-based console report (no image plots)
- Minimal artifacts: metrics_season.json and per_matchday_metrics_season.csv
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import (
    LABELS_ORDER,
    bin_by_confidence,
    brier_score_multiclass,
    compute_points,
    confusion_matrix_stats,
    ensure_dir,
    expected_calibration_error,
    log_loss_multiclass,
    ranked_probability_score_3c,
    save_json,
)
from kicktipp_predictor.predictor import CascadedPredictor

# ===========================================================================
# Helper Functions
# ===========================================================================


def _process_prediction(
    pred: dict, actual: dict
) -> dict:
    """Process a single prediction and add evaluation metrics.

    Args:
        pred: Raw prediction dictionary from predictor
        actual: Actual match data with results

    Returns:
        Enhanced prediction with evaluation metrics
    """
    pred_home = int(pred.get("predicted_home_score", 0))
    pred_away = int(pred.get("predicted_away_score", 0))
    actual_home = int(actual.get("home_score", 0))
    actual_away = int(actual.get("away_score", 0))

    # Determine actual winner
    actual_winner = (
        "H" if actual_home > actual_away
        else ("A" if actual_away > actual_home else "D")
    )

    # Get predicted winner from probabilities
    prob_home = float(pred.get("home_win_probability", 1.0 / 3))
    prob_draw = float(pred.get("draw_probability", 1.0 / 3))
    prob_away = float(pred.get("away_win_probability", 1.0 / 3))
    probs = np.array([prob_home, prob_draw, prob_away], dtype=float)
    if probs.sum() > 0:
        probs = probs / probs.sum()

    pred_idx = int(np.argmax(probs))
    winner_pred = LABELS_ORDER[pred_idx]
    winner_pred_prob = float(probs[pred_idx])
    winner_correct = bool(winner_pred == actual_winner)

    # Calculate points based on scoreline prediction
    if pred_home == actual_home and pred_away == actual_away:
        points = 4
    elif (pred_home - pred_away) == (actual_home - actual_away):
        points = 3
    else:
        pred_winner_score = (
            "H" if pred_home > pred_away
            else ("A" if pred_away > pred_home else "D")
        )
        points = 2 if pred_winner_score == actual_winner else 0

    # Update prediction with evaluation data
    pred["actual_home_score"] = actual_home
    pred["actual_away_score"] = actual_away
    pred["points_earned"] = points
    pred["is_evaluated"] = True
    pred["matchday"] = actual["matchday"]
    pred["winner_true"] = actual_winner
    pred["winner_pred"] = winner_pred
    pred["winner_pred_prob"] = winner_pred_prob
    pred["winner_correct"] = winner_correct

    return pred


def _compute_overall_metrics(
    predictions: list[dict],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute overall evaluation metrics from predictions.

    Args:
        predictions: List of evaluated predictions

    Returns:
        Tuple of (metrics_dict, pred_home, pred_away, actual_home,
                 actual_away, true_labels)
    """
    actual_home = np.array(
        [int(p.get("actual_home_score", 0)) for p in predictions], dtype=int
    )
    actual_away = np.array(
        [int(p.get("actual_away_score", 0)) for p in predictions], dtype=int
    )

    # Determine true outcome labels
    true_labels = [
        "H" if actual_home[i] > actual_away[i]
        else ("A" if actual_away[i] > actual_home[i] else "D")
        for i in range(len(predictions))
    ]

    # Extract and normalize probability matrix
    prob_matrix = np.array(
        [
            [
                float(p.get("home_win_probability", 1 / 3)),
                float(p.get("draw_probability", 1 / 3)),
                float(p.get("away_win_probability", 1 / 3)),
            ]
            for p in predictions
        ],
        dtype=float,
    )
    prob_matrix = np.clip(prob_matrix, 1e-15, 1.0)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    pred_home = np.array(
        [int(p.get("predicted_home_score", 0)) for p in predictions], dtype=int
    )
    pred_away = np.array(
        [int(p.get("predicted_away_score", 0)) for p in predictions], dtype=int
    )

    points = compute_points(pred_home, pred_away, actual_home, actual_away)

    # Compute main metrics
    metrics = {
        "brier": float(brier_score_multiclass(true_labels, prob_matrix)),
        "log_loss": float(log_loss_multiclass(true_labels, prob_matrix)),
        "rps": float(ranked_probability_score_3c(true_labels, prob_matrix)),
        "ece": expected_calibration_error(true_labels, prob_matrix, n_bins=10),
        "avg_points": float(np.mean(points)) if len(points) else 0.0,
        "total_points": int(np.sum(points)),
        "accuracy": float(
            np.mean(
                np.argmax(prob_matrix, axis=1)
                == np.array(
                    [{"H": 0, "D": 1, "A": 2}[t] for t in true_labels]
                )
            )
        ),
        "n": int(len(predictions)),
    }

    # Add winner statistics
    winner_correct = sum(p.get("winner_correct", False) for p in predictions)
    metrics["winner_accuracy_prob"] = float(metrics["accuracy"])
    metrics["winner_correct_count"] = winner_correct
    metrics["winner_incorrect_count"] = metrics["n"] - winner_correct

    # Per-class winner statistics
    y_true_arr = np.array(true_labels)
    per_class_stats = {}
    for j, label in enumerate(LABELS_ORDER):
        idx_label = np.where(np.argmax(prob_matrix, axis=1) == j)[0]
        n_label = len(idx_label)
        if n_label > 0:
            mean_prob = float(np.mean(prob_matrix[idx_label, j]))
            accuracy = float(np.mean(y_true_arr[idx_label] == label))
        else:
            mean_prob = float("nan")
            accuracy = float("nan")
        per_class_stats[label] = {
            "n_predicted": n_label,
            "mean_pred_prob": mean_prob,
            "accuracy": accuracy,
        }
    metrics["winner_prob_per_class"] = per_class_stats

    return (
        metrics,
        pred_home,
        pred_away,
        actual_home,
        actual_away,
        true_labels,
    )


def _compute_baseline_comparison(
    n_matches: int,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    points: np.ndarray,
) -> dict:
    """Compute baseline (2-1 home) comparison metrics.

    Args:
        n_matches: Number of matches
        actual_home: Actual home scores
        actual_away: Actual away scores
        points: Model points earned

    Returns:
        Dictionary with baseline metrics and bootstrap CI
    """
    true_labels = [
        "H" if actual_home[i] > actual_away[i]
        else ("A" if actual_away[i] > actual_home[i] else "D")
        for i in range(n_matches)
    ]

    # Baseline: always predict 2-1 home
    baseline_home = np.full(n_matches, 2, dtype=int)
    baseline_away = np.full(n_matches, 1, dtype=int)
    baseline_points = compute_points(
        baseline_home, baseline_away, actual_home, actual_away
    )

    baseline = {
        "avg_points": float(np.mean(baseline_points))
        if len(baseline_points)
        else 0.0,
        "total_points": int(np.sum(baseline_points)),
        "accuracy": float(np.mean(np.array(true_labels) == "H")),
    }

    # Bootstrap CI for PPG delta (paired)
    try:
        rng = np.random.default_rng(42)
        deltas = []
        if n_matches > 0:
            diff = (points - baseline_points).astype(float)
            for _ in range(2000):
                sample_idx = rng.choice(n_matches, size=n_matches, replace=True)
                deltas.append(np.mean(diff[sample_idx]))
            baseline["bootstrap_ci"] = {
                "lo": float(np.percentile(deltas, 2.5)),
                "hi": float(np.percentile(deltas, 97.5)),
                "B": 2000,
            }
        else:
            baseline["bootstrap_ci"] = {
                "lo": float("nan"),
                "hi": float("nan"),
                "B": 2000,
            }
    except Exception:
        baseline["bootstrap_ci"] = {
            "lo": float("nan"),
            "hi": float("nan"),
            "B": 2000,
        }

    return baseline


# ===========================================================================
# Main Evaluation Function
# ===========================================================================


def run_season_dynamic_evaluation(retrain_every: int = 1) -> None:
    """Evaluate current season with expanding-window retraining.

    Retrains the predictor every N matchdays using all matches finished so far
    (plus a few previous seasons for a warm start), then evaluates finished
    matches in the current season. Produces:
      - Rich console report
      - data/predictions/metrics_season.json
      - data/predictions/per_matchday_metrics_season.csv
    """
    console = Console()

    console.rule("SEASON PERFORMANCE EVALUATION")

    data_loader = DataLoader()
    predictor = CascadedPredictor()

    console.print("[bold]Loading models...[/bold]")
    try:
        predictor.load_models()
        console.print("[green]Models loaded successfully![/green]\n")
    except FileNotFoundError:
        console.print("[red]No trained models found. Run training first.[/red]")
        raise SystemExit(1)

    current_season = data_loader.get_current_season()
    console.print(f"Fetching data for current season: [bold]{current_season}[/bold]")
    season_matches = data_loader.fetch_season_matches(current_season)

    finished_matches = [m for m in season_matches if m.get("is_finished")]
    if not finished_matches:
        console.print(
            "[yellow]No finished matches found for the current season.[/yellow]"
        )
        raise SystemExit(0)

    first_matchday = min(m["matchday"] for m in finished_matches)
    last_matchday = max(m["matchday"] for m in finished_matches)
    console.print(f"Evaluating matchdays from {first_matchday} to {last_matchday}\n")

    all_predictions: list[dict] = []

    console.print("Preparing historical warm start (previous seasons)...")
    all_historical_matches = data_loader.fetch_historical_seasons(
        current_season - 5, current_season - 1
    )
    cumulative_training_matches = list(all_historical_matches)
    console.print(
        f"Initialized training set with [bold]{len(cumulative_training_matches)}[/bold] matches from previous seasons."
    )

    for matchday in range(first_matchday, last_matchday + 1):
        console.print(Panel.fit(f"Processing Matchday {matchday}", style="cyan"))

        if (matchday - first_matchday) % max(1, int(retrain_every)) == 0:
            train_df = data_loader.create_features_from_matches(
                cumulative_training_matches
            )
            predictor.train(train_df)
            console.print("[green]Model retrained.[/green]")

        matchday_matches = [m for m in finished_matches if m["matchday"] == matchday]
        if not matchday_matches:
            continue

        features_df = data_loader.create_prediction_features(
            matchday_matches, cumulative_training_matches
        )
        if features_df is None or len(features_df) == 0:
            console.print("[yellow]Insufficient features; skipping.[/yellow]")
            continue

        preds_today = predictor.predict(features_df)
        for pred, actual in zip(preds_today, matchday_matches):
            processed_pred = _process_prediction(pred, actual)
            all_predictions.append(processed_pred)

        cumulative_training_matches.extend(matchday_matches)

    if not all_predictions:
        console.print(
            "[yellow]No predictions could be generated for the season.[/yellow]"
        )
        raise SystemExit(0)

    # Compute overall metrics
    (
        metrics,
        pred_home,
        pred_away,
        actual_home,
        actual_away,
        true_labels,
    ) = _compute_overall_metrics(all_predictions)

    # Compute points for various analyses
    points = compute_points(pred_home, pred_away, actual_home, actual_away)

    # Diagnostics output directory
    out_dir = os.path.join("data", "predictions")
    ensure_dir(out_dir)

    # Save blend debug CSV with per-match diagnostics
    try:
        debug_rows = []
        for pred in all_predictions:
            row = {
                "match_id": pred.get("match_id"),
                "matchday": pred.get("matchday"),
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "predicted_home_score": pred.get("predicted_home_score"),
                "predicted_away_score": pred.get("predicted_away_score"),
                "home_win_probability": pred.get("home_win_probability"),
                "draw_probability": pred.get("draw_probability"),
                "away_win_probability": pred.get("away_win_probability"),
                "actual_home_score": pred.get("actual_home_score"),
                "actual_away_score": pred.get("actual_away_score"),
                "points_earned": pred.get("points_earned"),
                "winner_true": pred.get("winner_true"),
                "winner_pred": pred.get("winner_pred"),
                "winner_correct": pred.get("winner_correct"),
                "winner_pred_prob": pred.get("winner_pred_prob"),
                "predicted_goal_difference": pred.get("predicted_goal_difference"),
                "uncertainty_stddev": pred.get("uncertainty_stddev"),
                "draw_margin_used": pred.get("draw_margin_used"),
            }
            debug_rows.append(row)
        if debug_rows:
            pd.DataFrame(debug_rows).to_csv(
                os.path.join(out_dir, "blend_debug.csv"), index=False
            )
            console.print(
                f"Blend diagnostics written to "
                f"[bold]{os.path.join(out_dir, 'blend_debug.csv')}[/bold]"
            )
    except Exception as e:  # pragma: no cover
        console.print(
            f"[yellow]Warning: could not write blend_debug.csv: {e}[/yellow]"
        )

    # Overall quality breakdown
    exact_count = int(
        np.sum((pred_home == actual_home) & (pred_away == actual_away))
    )
    diff_count = int(
        np.sum(
            ((pred_home - pred_away) == (actual_home - actual_away))
            & ~((pred_home == actual_home) & (pred_away == actual_away))
        )
    )
    result_count = int(
        np.sum(
            ((pred_home > pred_away) & (actual_home > actual_away))
            | ((pred_home == pred_away) & (actual_home == actual_away))
            | ((pred_home < pred_away) & (actual_home < actual_away))
        )
        - exact_count
        - diff_count
    )

    # Baseline comparison and bootstrap CI
    baseline = _compute_baseline_comparison(
        metrics["n"], actual_home, actual_away, points
    )
    metrics["bootstrap_ci_ppg_delta"] = baseline.pop("bootstrap_ci")

    # Extract probability matrix for reporting
    prob_matrix = np.array(
        [
            [
                float(p.get("home_win_probability", 1 / 3)),
                float(p.get("draw_probability", 1 / 3)),
                float(p.get("away_win_probability", 1 / 3)),
            ]
            for p in all_predictions
        ],
        dtype=float,
    )
    prob_matrix = np.clip(prob_matrix, 1e-15, 1.0)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    # Distributions
    label_counts = {lab: true_labels.count(lab) for lab in LABELS_ORDER}
    pred_labels = [LABELS_ORDER[i] for i in np.argmax(prob_matrix, axis=1)]
    pred_counts = {lab: pred_labels.count(lab) for lab in LABELS_ORDER}

    # Scoreline top-k
    pred_scores = Counter(
        [f"{h}-{a}" for h, a in zip(pred_home.tolist(), pred_away.tolist())]
    )
    actual_scores = Counter(
        [f"{h}-{a}" for h, a in zip(actual_home.tolist(), actual_away.tolist())]
    )

    # Confusion & per-class stats
    cm_stats = confusion_matrix_stats(true_labels, prob_matrix)
    cm = np.array(cm_stats["matrix"], dtype=int)
    per_class = (
        cm_stats.get("per_class", {}) if isinstance(cm_stats, dict) else {}
    )

    # Confidence buckets (numeric only)
    max_prob = np.max(prob_matrix, axis=1)
    sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    confidence = 0.6 * max_prob + 0.4 * margin
    conf_df = bin_by_confidence(
        confidence, true_labels, prob_matrix, points, n_bins=5
    )

    # Save season metrics
    save_json({"main": metrics}, os.path.join(out_dir, "metrics_season.json"))

    # Per-matchday breakdown
    rows = []
    if all_predictions:
        matchdays = [int(p.get("matchday", -1)) for p in all_predictions]
        matchday_indices = {}
        for i, matchday in enumerate(matchdays):
            if matchday < 0:
                continue
            matchday_indices.setdefault(matchday, []).append(i)

        for matchday in sorted(matchday_indices):
            indices = np.array(matchday_indices[matchday], dtype=int)
            if len(indices) == 0:
                continue

            prob_md = prob_matrix[indices]
            labels_md = [true_labels[i] for i in indices]
            pred_home_md = pred_home[indices]
            pred_away_md = pred_away[indices]
            actual_home_md = actual_home[indices]
            actual_away_md = actual_away[indices]
            points_md = points[indices]

            n_matches = len(indices)
            total_pts = int(np.sum(points_md))
            avg_pts = float(np.mean(points_md)) if n_matches else 0.0
            points_0 = int(np.sum(points_md == 0))
            points_2 = int(np.sum(points_md == 2))
            points_3 = int(np.sum(points_md == 3))
            points_4 = int(np.sum(points_md == 4))
            accuracy_md = float(
                np.mean(
                    np.argmax(prob_md, axis=1)
                    == np.array(
                        [{"H": 0, "D": 1, "A": 2}[t] for t in labels_md]
                    )
                )
            )

            exact_md = int(
                np.sum(
                    (pred_home_md == actual_home_md)
                    & (pred_away_md == actual_away_md)
                )
            )
            diff_md = int(
                np.sum(
                    ((pred_home_md - pred_away_md) == (actual_home_md - actual_away_md))
                    & ((pred_home_md != actual_home_md) | (pred_away_md != actual_away_md))
                )
            )
            result_md = int(
                np.sum(
                    ((pred_home_md > pred_away_md) & (actual_home_md > actual_away_md))
                    | ((pred_home_md == pred_away_md) & (actual_home_md == actual_away_md))
                    | ((pred_home_md < pred_away_md) & (actual_home_md < actual_away_md))
                )
                - exact_md
                - diff_md
            )

            baseline_home_md = np.full(n_matches, 2, dtype=int)
            baseline_away_md = np.full(n_matches, 1, dtype=int)
            baseline_pts_md = compute_points(
                baseline_home_md, baseline_away_md, actual_home_md, actual_away_md
            )
            baseline_total = int(np.sum(baseline_pts_md))
            baseline_avg = (
                float(np.mean(baseline_pts_md)) if n_matches else 0.0
            )

            rows.append(
                {
                    "matchday": matchday,
                    "n": n_matches,
                    "avg_points": avg_pts,
                    "total_points": total_pts,
                    "points_0": points_0,
                    "points_2": points_2,
                    "points_3": points_3,
                    "points_4": points_4,
                    "accuracy": accuracy_md,
                    "exact_count": exact_md,
                    "diff_count": diff_md,
                    "result_count": result_md,
                    "brier": float(brier_score_multiclass(labels_md, prob_md)),
                    "log_loss": float(log_loss_multiclass(labels_md, prob_md)),
                    "rps": float(ranked_probability_score_3c(labels_md, prob_md)),
                    "baseline_avg_points": baseline_avg,
                    "baseline_total_points": baseline_total,
                    "delta_avg_points": avg_pts - baseline_avg,
                    "delta_total_points": total_pts - baseline_total,
                }
            )

    per_md_df = pd.DataFrame(rows)
    per_md_csv = os.path.join(out_dir, "per_matchday_metrics_season.csv")
    if len(per_md_df) > 0:
        try:
            per_md_df.sort_values("matchday").to_csv(per_md_csv, index=False)
            console.print(f"Per-matchday metrics written to [bold]{per_md_csv}[/bold]")
        except Exception as e:  # pragma: no cover
            console.print(f"[red]Failed to write per-matchday CSV: {e}[/red]")

    # -------------------- Rich Console Report --------------------
    # Summary panel
    summary = Table.grid(expand=False)
    summary.add_column(justify="right")
    summary.add_column(justify="left")
    summary.add_row("Matches", f"{metrics['n']}")
    summary.add_row("Avg points", f"{metrics['avg_points']:.3f}")
    summary.add_row("Total points", f"{metrics['total_points']}")
    summary.add_row("Accuracy", f"{metrics['accuracy']:.3f}")
    summary.add_row("Brier", f"{metrics['brier']:.4f}")
    summary.add_row("Log loss", f"{metrics['log_loss']:.4f}")
    summary.add_row("RPS", f"{metrics['rps']:.4f}")

    quality = Table.grid(expand=False)
    quality.add_column(justify="right")
    quality.add_column(justify="left")
    quality.add_row("Exact scores", f"{exact_count}")
    quality.add_row("Correct diff", f"{diff_count}")
    quality.add_row("Correct result", f"{result_count}")

    base = Table.grid(expand=False)
    base.add_column(justify="right")
    base.add_column(justify="left")
    base.add_row("Baseline avg", f"{baseline['avg_points']:.3f}")
    base.add_row("Δ avg", f"{metrics['avg_points'] - baseline['avg_points']:+.3f}")
    base.add_row("Baseline acc", f"{baseline['accuracy']:.3f}")
    base.add_row("Δ acc", f"{metrics['accuracy'] - baseline['accuracy']:+.3f}")

    console.print(
        Columns(
            [
                Panel(
                    summary,
                    title="Season Metrics",
                    border_style="green",
                    box=box.ROUNDED,
                ),
                Panel(
                    quality,
                    title="Prediction Quality",
                    border_style="blue",
                    box=box.ROUNDED,
                ),
                Panel(
                    base,
                    title="Baseline (2-1 H)",
                    border_style="magenta",
                    box=box.ROUNDED,
                ),
            ],
            equal=True,
        )
    )

    # Outcome distributions
    dist_table = Table(title="Outcome Distribution", box=box.SIMPLE_HEAVY)
    dist_table.add_column("Label", justify="center")
    dist_table.add_column("Actual", justify="right")
    dist_table.add_column("Actual %", justify="right")
    dist_table.add_column("Predicted", justify="right")
    dist_table.add_column("Predicted %", justify="right")
    total = max(1, metrics["n"])
    for lab in LABELS_ORDER:
        a = int(label_counts.get(lab, 0))
        p = int(pred_counts.get(lab, 0))
        dist_table.add_row(lab, str(a), f"{a/total:.1%}", str(p), f"{p/total:.1%}")

    # Points distribution
    points_table = Table(title="Points Distribution", box=box.SIMPLE_HEAVY)
    points_table.add_column("Points", justify="center")
    points_table.add_column("Count", justify="right")
    points_table.add_column("Percent", justify="right")
    for point_value in [0, 2, 3, 4]:
        count = int(np.sum(points == point_value))
        points_table.add_row(str(point_value), str(count), f"{(count/total):.1%}")

    console.print(Columns([dist_table, points_table], equal=True))

    # Additional monitoring: draw statistics and uncertainty scaling
    try:
        total_matches = max(1, metrics["n"])
        actual_draw_rate = float(label_counts.get("D", 0)) / float(total_matches)
        predicted_draw_rate = float(pred_counts.get("D", 0)) / float(total_matches)
        avg_draw_prob = float(np.mean(prob_matrix[:, 1]))
        target_ok = 0.15 <= predicted_draw_rate <= 0.30

        draw_table = Table(title="Draw Stats", box=box.SIMPLE_HEAVY)
        draw_table.add_column("Metric", justify="left")
        draw_table.add_column("Value", justify="right")
        draw_table.add_row("Actual draw rate", f"{actual_draw_rate:.3f}")
        draw_table.add_row("Predicted draw rate", f"{predicted_draw_rate:.3f}")
        draw_table.add_row("Avg draw probability", f"{avg_draw_prob:.3f}")
        draw_table.add_row("Within 15–30% target", "Yes" if target_ok else "No")

        # Uncertainty scaling stats from predictions
        gd_abs = np.array([
            abs(float(p.get("predicted_goal_difference", 0.0))) for p in all_predictions
        ], dtype=float)
        stddevs = np.array([
            float(p.get("uncertainty_stddev", float("nan"))) for p in all_predictions
        ], dtype=float)
        if len(gd_abs) > 1 and np.all(np.isfinite(stddevs)):
            corr = float(np.corrcoef(gd_abs, stddevs)[0, 1])
        else:
            corr = float("nan")
        std_mean = float(np.nanmean(stddevs))
        std_min = float(np.nanmin(stddevs))
        std_max = float(np.nanmax(stddevs))
        # draw margin used (assumes static)
        dm_values = [p.get("draw_margin_used") for p in all_predictions]
        dm_values = [float(x) for x in dm_values if x is not None]
        draw_margin_used = float(dm_values[0]) if dm_values else float("nan")

        unc_table = Table(title="Uncertainty Stats", box=box.SIMPLE_HEAVY)
        unc_table.add_column("Metric", justify="left")
        unc_table.add_column("Value", justify="right")
        unc_table.add_row("Stddev mean", f"{std_mean:.3f}")
        unc_table.add_row("Stddev min", f"{std_min:.3f}")
        unc_table.add_row("Stddev max", f"{std_max:.3f}")
        unc_table.add_row("corr(|pred GD|, stddev)", f"{corr:.3f}")
        unc_table.add_row("Draw margin used", f"{draw_margin_used:.3f}")

        console.print(Columns([draw_table, unc_table], equal=True))
    except Exception:
        pass

    # Top scorelines
    top_table = Table(
        title="Top Scorelines (Predicted vs Actual)", box=box.SIMPLE_HEAVY
    )
    top_table.add_column("Rank", justify="right")
    top_table.add_column("Predicted", justify="left")
    top_table.add_column("Count", justify="right")
    top_table.add_column("Actual", justify="left")
    top_table.add_column("Count", justify="right")
    top_pred = pred_scores.most_common(5)
    top_act = actual_scores.most_common(5)
    for i in range(5):
        ps = top_pred[i][0] if i < len(top_pred) else "-"
        pc = top_pred[i][1] if i < len(top_pred) else 0
        as_ = top_act[i][0] if i < len(top_act) else "-"
        ac = top_act[i][1] if i < len(top_act) else 0
        top_table.add_row(str(i + 1), ps, str(pc), as_, str(ac))
    console.print(top_table)

    # Confidence buckets table
    if isinstance(conf_df, pd.DataFrame) and len(conf_df) > 0:
        conf_table = Table(title="Confidence Buckets", box=box.SIMPLE_HEAVY)
        conf_table.add_column("Bin", justify="left")
        conf_table.add_column("Count", justify="right")
        conf_table.add_column("Avg Points", justify="right")
        conf_table.add_column("Accuracy", justify="right")
        conf_table.add_column("Avg Confidence", justify="right")
        for _, r in conf_df.iterrows():
            conf_table.add_row(
                str(r.get("bin")),
                str(int(r.get("count", 0))),
                f"{float(r.get('avg_points', float('nan'))):.3f}",
                f"{float(r.get('accuracy', float('nan'))):.3f}",
                f"{float(r.get('avg_confidence', float('nan'))):.3f}",
            )
        console.print(conf_table)

    # Winner accuracy summary (probabilities only)
    win_acc_table = Table(title="Winner Accuracy (Probabilities)", box=box.SIMPLE_HEAVY)
    win_acc_table.add_column("Metric", justify="left")
    win_acc_table.add_column("Value", justify="right")
    win_acc_table.add_row("Matches", f"{metrics['n']}")
    win_acc_table.add_row("Correct winners", f"{metrics['winner_correct_count']}")
    win_acc_table.add_row("Incorrect winners", f"{metrics['winner_incorrect_count']}")
    win_acc_table.add_row("Accuracy", f"{metrics['winner_accuracy_prob']:.3f}")
    console.print(win_acc_table)

    # Per-class probability vs accuracy
    prob_pc_table = Table(
        title="Per-class Winner Prob vs Accuracy", box=box.SIMPLE_HEAVY
    )
    prob_pc_table.add_column("Class", justify="center")
    prob_pc_table.add_column("Pred Count", justify="right")
    prob_pc_table.add_column("Mean Prob", justify="right")
    prob_pc_table.add_column("Accuracy", justify="right")
    for lab in LABELS_ORDER:
        s = metrics["winner_prob_per_class"].get(lab, {})
        n_pred = int(s.get("n_predicted", 0))
        mean_prob = float(s.get("mean_pred_prob", float("nan")))
        acc_lab = float(s.get("accuracy", float("nan")))
        prob_pc_table.add_row(lab, str(n_pred), f"{mean_prob:.3f}", f"{acc_lab:.3f}")
    console.print(prob_pc_table)

    # Max-probability calibration buckets (probabilities only)
    prob_conf_df = bin_by_confidence(
        max_prob, true_labels, prob_matrix, points, n_bins=5
    )
    if isinstance(prob_conf_df, pd.DataFrame) and len(prob_conf_df) > 0:
        prob_conf_table = Table(
            title="Max-Prob Calibration Buckets", box=box.SIMPLE_HEAVY
        )
        prob_conf_table.add_column("Bin", justify="left")
        prob_conf_table.add_column("Count", justify="right")
        prob_conf_table.add_column("Accuracy", justify="right")
        prob_conf_table.add_column("Avg Max-Prob", justify="right")
        prob_conf_table.add_column("Avg Points", justify="right")
        for _, r in prob_conf_df.iterrows():
            prob_conf_table.add_row(
                str(r.get("bin")),
                str(int(r.get("count", 0))),
                f"{float(r.get('accuracy', float('nan'))):.3f}",
                f"{float(r.get('avg_confidence', float('nan'))):.3f}",
                f"{float(r.get('avg_points', float('nan'))):.3f}",
            )
        console.print(prob_conf_table)

    # Confusion matrix + per-class
    cm_table = Table(title="Confusion Matrix", box=box.SIMPLE_HEAVY)
    cm_table.add_column("Actual \\ Pred")
    for lab in LABELS_ORDER:
        cm_table.add_column(lab, justify="right")
    for i, lab in enumerate(LABELS_ORDER):
        row = [lab] + [str(int(cm[i, j])) for j in range(3)]
        cm_table.add_row(*row)
    console.print(cm_table)

    if isinstance(per_class, dict) and per_class:
        pc_table = Table(title="Per-class Precision/Recall", box=box.SIMPLE_HEAVY)
        pc_table.add_column("Class", justify="center")
        pc_table.add_column("Precision", justify="right")
        pc_table.add_column("Recall", justify="right")
        for lab in LABELS_ORDER:
            stats = per_class.get(lab, {})
            pr = float(stats.get("precision", float("nan")))
            rc = float(stats.get("recall", float("nan")))
            pc_table.add_row(lab, f"{pr:.3f}", f"{rc:.3f}")
        console.print(pc_table)

    # Per-matchday compact summary (console)
    if len(per_md_df) > 0:
        md_table = Table(title="Per-Matchday Summary", box=box.SIMPLE_HEAVY)
        md_table.add_column("MD", justify="right")
        md_table.add_column("n", justify="right")
        md_table.add_column("avg", justify="right")
        md_table.add_column("base", justify="right")
        md_table.add_column("Δavg", justify="right")
        md_table.add_column("acc", justify="right")
        md_table.add_column("ex/diff/res", justify="center")
        for _, r in per_md_df.sort_values("matchday").iterrows():
            ex_dr = f"{int(r['exact_count'])}/{int(r['diff_count'])}/{int(r['result_count'])}"
            md_table.add_row(
                f"{int(r['matchday'])}",
                f"{int(r['n'])}",
                f"{float(r['avg_points']):.2f}",
                f"{float(r['baseline_avg_points']):.2f}",
                f"{float(r['delta_avg_points']):+.2f}",
                f"{float(r['accuracy']):.2f}",
                ex_dr,
            )
        console.print(md_table)

    # ECE by class (if dict)
    ece = metrics.get("ece")
    if isinstance(ece, dict):
        ece_table = Table(title="ECE by Class (n_bins=10)", box=box.SIMPLE_HEAVY)
        ece_table.add_column("Class", justify="center")
        ece_table.add_column("ECE", justify="right")
        for lab in LABELS_ORDER:
            e = float(ece.get(lab, float("nan")))
            ece_table.add_row(lab, f"{e:.4f}")
        console.print(ece_table)

    console.rule("Evaluation Complete")
    console.print(Text(f"Artifacts written to {out_dir}", style="bold"))
