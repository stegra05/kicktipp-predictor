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
from kicktipp_predictor.predictor import MatchPredictor


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
    predictor = MatchPredictor()

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
            ph = int(pred.get("predicted_home_score", 0))
            pa = int(pred.get("predicted_away_score", 0))
            ah = int(actual.get("home_score", 0))
            aa = int(actual.get("away_score", 0))

            # Actual outcome label from final scores
            actual_winner = "H" if ah > aa else ("A" if aa > ah else "D")

            # Predicted outcome based on probabilities only (not scores)
            pH = float(pred.get("home_win_probability", 1.0 / 3))
            pD = float(pred.get("draw_probability", 1.0 / 3))
            pA = float(pred.get("away_win_probability", 1.0 / 3))
            probs = np.array([pH, pD, pA], dtype=float)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            pred_idx = int(np.argmax(probs))
            winner_pred = LABELS_ORDER[pred_idx]
            winner_pred_prob = float(probs[pred_idx])
            winner_correct = bool(winner_pred == actual_winner)

            # Points remain based on the scoreline prediction
            if ph == ah and pa == aa:
                points = 4
            elif (ph - pa) == (ah - aa):
                points = 3
            else:
                pred_winner_score = "H" if ph > pa else ("A" if pa > ph else "D")
                points = 2 if pred_winner_score == actual_winner else 0

            pred["actual_home_score"] = ah
            pred["actual_away_score"] = aa
            pred["points_earned"] = points
            pred["is_evaluated"] = True
            pred["matchday"] = matchday

            # Store probability-based winner info
            pred["winner_true"] = actual_winner
            pred["winner_pred"] = winner_pred
            pred["winner_pred_prob"] = winner_pred_prob
            pred["winner_correct"] = winner_correct

            all_predictions.append(pred)

        cumulative_training_matches.extend(matchday_matches)

    if not all_predictions:
        console.print(
            "[yellow]No predictions could be generated for the season.[/yellow]"
        )
        raise SystemExit(0)

    preds_all = list(all_predictions)
    ah = np.asarray([int(p.get("actual_home_score", 0)) for p in preds_all], dtype=int)
    aa = np.asarray([int(p.get("actual_away_score", 0)) for p in preds_all], dtype=int)

    y_true: list[str] = []
    for i in range(len(preds_all)):
        if ah[i] > aa[i]:
            y_true.append("H")
        elif aa[i] > ah[i]:
            y_true.append("A")
        else:
            y_true.append("D")

    P = np.array(
        [
            [
                float(p.get("home_win_probability", 1 / 3)),
                float(p.get("draw_probability", 1 / 3)),
                float(p.get("away_win_probability", 1 / 3)),
            ]
            for p in preds_all
        ],
        dtype=float,
    )
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    ph = np.asarray(
        [int(p.get("predicted_home_score", 0)) for p in preds_all], dtype=int
    )
    pa = np.asarray(
        [int(p.get("predicted_away_score", 0)) for p in preds_all], dtype=int
    )

    pts = compute_points(ph, pa, ah, aa)

    # Metrics
    metrics = {
        "brier": float(brier_score_multiclass(y_true, P)),
        "log_loss": float(log_loss_multiclass(y_true, P)),
        "rps": float(ranked_probability_score_3c(y_true, P)),
        "ece": expected_calibration_error(y_true, P, n_bins=10),
        "avg_points": float(np.mean(pts)) if len(pts) else 0.0,
        "total_points": int(np.sum(pts)),
        "accuracy": float(
            np.mean(
                np.argmax(P, axis=1)
                == np.array([{"H": 0, "D": 1, "A": 2}[t] for t in y_true])
            )
        ),
        "n": int(len(preds_all)),
    }

    # Diagnostics output directory
    out_dir = os.path.join("data", "predictions")
    ensure_dir(out_dir)

    # Save blend debug CSV with per-match diagnostics if available
    try:
        debug_rows: list[dict] = []
        for p in preds_all:
            row = {
                "match_id": p.get("match_id"),
                "matchday": p.get("matchday"),
                "home_team": p.get("home_team"),
                "away_team": p.get("away_team"),
                "predicted_home_score": p.get("predicted_home_score"),
                "predicted_away_score": p.get("predicted_away_score"),
                "home_expected_goals": p.get("home_expected_goals"),
                "away_expected_goals": p.get("away_expected_goals"),
                "home_win_probability": p.get("home_win_probability"),
                "draw_probability": p.get("draw_probability"),
                "away_win_probability": p.get("away_win_probability"),
                "entropy": p.get("entropy"),
                "blend_weight": p.get("blend_weight"),
                "cls_p_H": p.get("cls_p_H"),
                "cls_p_D": p.get("cls_p_D"),
                "cls_p_A": p.get("cls_p_A"),
                "pois_p_H": p.get("pois_p_H"),
                "pois_p_D": p.get("pois_p_D"),
                "pois_p_A": p.get("pois_p_A"),
                "actual_home_score": p.get("actual_home_score"),
                "actual_away_score": p.get("actual_away_score"),
                "points_earned": p.get("points_earned"),
                "winner_true": p.get("winner_true"),
                "winner_pred": p.get("winner_pred"),
                "winner_correct": p.get("winner_correct"),
                "winner_pred_prob": p.get("winner_pred_prob"),
            }
            debug_rows.append(row)
        if len(debug_rows) > 0:
            pd.DataFrame(debug_rows).to_csv(
                os.path.join(out_dir, "blend_debug.csv"), index=False
            )
            console.print(
                f"Blend diagnostics written to [bold]{os.path.join(out_dir, 'blend_debug.csv')}[/bold]"
            )
    except Exception as e:  # pragma: no cover
        console.print(f"[yellow]Warning: could not write blend_debug.csv: {e}[/yellow]")

    # Overall quality breakdown
    exact_count = int(np.sum((ph == ah) & (pa == aa)))
    diff_count = int(np.sum(((ph - pa) == (ah - aa)) & ~((ph == ah) & (pa == aa))))
    result_count = int(
        np.sum(
            ((ph > pa) & (ah > aa))
            | ((ph == pa) & (ah == aa))
            | ((ph < pa) & (ah < aa))
        )
        - exact_count
        - diff_count
    )

    # Baseline (2-1 home) comparison
    base_ph = np.full(metrics["n"], 2, dtype=int)
    base_pa = np.full(metrics["n"], 1, dtype=int)
    base_pts = compute_points(base_ph, base_pa, ah, aa)
    baseline = {
        "avg_points": float(np.mean(base_pts)) if len(base_pts) else 0.0,
        "total_points": int(np.sum(base_pts)),
        "accuracy": float(np.mean(np.array(y_true) == "H")),
    }

    # Bootstrap CI for PPG delta (paired)
    try:
        rng = np.random.default_rng(42)
        B = 2000
        idx_all = (
            np.arange(metrics["n"]) if metrics["n"] > 0 else np.array([], dtype=int)
        )
        deltas = []
        if len(idx_all) > 0:
            diff = (pts - base_pts).astype(float)
            for _ in range(B):
                bs = rng.choice(idx_all, size=len(idx_all), replace=True)
                deltas.append(np.mean(diff[bs]))
            ci_lo, ci_hi = (
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            )
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        metrics["bootstrap_ci_ppg_delta"] = {"lo": ci_lo, "hi": ci_hi, "B": B}
    except Exception as e:  # pragma: no cover
        console.print(f"[yellow]Bootstrap CI computation failed: {e}[/yellow]")

    # Distributions
    label_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    pred_labels = [LABELS_ORDER[i] for i in np.argmax(P, axis=1)]
    pred_counts = {lab: pred_labels.count(lab) for lab in LABELS_ORDER}

    # Probability-based winner correctness and per-class stats
    y_pred_idx = np.argmax(P, axis=1)
    y_true_arr = np.array(y_true)
    correct_winner_count = int(
        np.sum(np.array([LABELS_ORDER[i] for i in y_pred_idx]) == y_true_arr)
    )
    incorrect_winner_count = int(metrics["n"] - correct_winner_count)

    per_class_prob_acc: dict[str, dict] = {}
    for j, lab in enumerate(LABELS_ORDER):
        idx_lab = np.where(y_pred_idx == j)[0]
        n_lab = int(len(idx_lab))
        if n_lab > 0:
            mean_prob_lab = float(np.mean(P[idx_lab, j]))
            acc_lab = float(np.mean(y_true_arr[idx_lab] == lab))
        else:
            mean_prob_lab = float("nan")
            acc_lab = float("nan")
        per_class_prob_acc[lab] = {
            "n_predicted": n_lab,
            "mean_pred_prob": mean_prob_lab,
            "accuracy": acc_lab,
        }

    metrics["winner_accuracy_prob"] = float(metrics["accuracy"])  # same definition
    metrics["winner_correct_count"] = correct_winner_count
    metrics["winner_incorrect_count"] = incorrect_winner_count
    metrics["winner_prob_per_class"] = per_class_prob_acc

    # Scoreline top-k
    pred_scores = Counter([f"{h}-{a}" for h, a in zip(ph.tolist(), pa.tolist())])
    actual_scores = Counter([f"{h}-{a}" for h, a in zip(ah.tolist(), aa.tolist())])

    # Confusion & per-class stats
    cm_stats = confusion_matrix_stats(y_true, P)
    cm = np.array(cm_stats["matrix"], dtype=int)
    per_class = cm_stats.get("per_class", {}) if isinstance(cm_stats, dict) else {}

    # Confidence buckets (numeric only)
    max_prob = np.max(P, axis=1)
    sorted_probs = np.sort(P, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    confidence = 0.6 * max_prob + 0.4 * margin
    conf_df = bin_by_confidence(confidence, y_true, P, pts, n_bins=5)

    # Save season metrics
    save_json({"main": metrics}, os.path.join(out_dir, "metrics_season.json"))

    # Per-matchday breakdown (also saved as CSV)
    rows: list[dict] = []
    if len(preds_all) > 0:
        matchdays = [int(p.get("matchday", -1)) for p in preds_all]
        md_to_idx: dict[int, list[int]] = {}
        for i, md in enumerate(matchdays):
            if md < 0:
                continue
            md_to_idx.setdefault(md, []).append(i)
        for md in sorted(md_to_idx):
            idx = np.array(md_to_idx[md], dtype=int)
            if len(idx) == 0:
                continue
            P_md = P[idx]
            y_md = [y_true[i] for i in idx]
            ph_md, pa_md = ph[idx], pa[idx]
            ah_md, aa_md = ah[idx], aa[idx]
            pts_md = pts[idx]

            n_md = len(idx)
            total_pts = int(np.sum(pts_md))
            avg_pts = float(np.mean(pts_md)) if n_md else 0.0
            points_0 = int(np.sum(pts_md == 0))
            points_2 = int(np.sum(pts_md == 2))
            points_3 = int(np.sum(pts_md == 3))
            points_4 = int(np.sum(pts_md == 4))
            acc = float(
                np.mean(
                    np.argmax(P_md, axis=1)
                    == np.array([{"H": 0, "D": 1, "A": 2}[t] for t in y_md])
                )
            )

            exact_md = int(np.sum((ph_md == ah_md) & (pa_md == aa_md)))
            diff_md = int(
                np.sum(
                    ((ph_md - pa_md) == (ah_md - aa_md))
                    & ((ph_md != ah_md) | (pa_md != aa_md))
                )
            )
            result_md = int(
                np.sum(
                    ((ph_md > pa_md) & (ah_md > aa_md))
                    | ((ph_md == pa_md) & (ah_md == aa_md))
                    | ((ph_md < pa_md) & (ah_md < aa_md))
                )
                - exact_md
                - diff_md
            )

            base_ph = np.full(n_md, 2, dtype=int)
            base_pa = np.full(n_md, 1, dtype=int)
            base_pts = compute_points(base_ph, base_pa, ah_md, aa_md)
            base_total = int(np.sum(base_pts))
            base_avg = float(np.mean(base_pts)) if n_md else 0.0

            rows.append(
                {
                    "matchday": md,
                    "n": n_md,
                    "avg_points": avg_pts,
                    "total_points": total_pts,
                    "points_0": points_0,
                    "points_2": points_2,
                    "points_3": points_3,
                    "points_4": points_4,
                    "accuracy": float(acc),
                    "exact_count": exact_md,
                    "diff_count": diff_md,
                    "result_count": result_md,
                    "brier": float(brier_score_multiclass(y_md, P_md)),
                    "log_loss": float(log_loss_multiclass(y_md, P_md)),
                    "rps": float(ranked_probability_score_3c(y_md, P_md)),
                    "baseline_avg_points": base_avg,
                    "baseline_total_points": base_total,
                    "delta_avg_points": avg_pts - base_avg,
                    "delta_total_points": total_pts - base_total,
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
    for p_val in [0, 2, 3, 4]:
        c = int(np.sum(pts == p_val))
        points_table.add_row(str(p_val), str(c), f"{(c/total):.1%}")

    console.print(Columns([dist_table, points_table], equal=True))

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
    prob_conf_df = bin_by_confidence(max_prob, y_true, P, pts, n_bins=5)
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
