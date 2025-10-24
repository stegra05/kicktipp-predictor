#!/usr/bin/env python3
"""
Run a fixed train/test baseline.

- Loads historical data from N seasons back up to the most recent full season year.
- Uses the most recent full season (current_season - 1) as the test set.
- Trains on all prior seasons in that window.
- Uses classifier-only probabilities (no calibration, no prior anchoring).
- Prints an evaluation metrics summary.

Example:
  If current season is 2025, test on 2024 and train on 2020–2023.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

# Ensure local src/ is on sys.path for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from kicktipp_predictor.config import get_config
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import LABELS_ORDER, confusion_matrix_stats
from kicktipp_predictor.predictor import MatchPredictor


def run_baseline(
    seasons_back: int = 5,
    quiet: bool = False,
    prob_source: str = "classifier",
    hybrid_poisson_weight: float = 0.5,
    use_ep_selection: bool = False,
) -> None:
    console = Console()

    # Initialize loader and determine seasons
    loader = DataLoader()
    current_season = loader.get_current_season()
    test_season = current_season - 1
    start_season = max(test_season - seasons_back + 1, 2005)  # safety lower bound
    train_end_season = test_season - 1

    if train_end_season < start_season:
        raise ValueError(
            f"Invalid season window: start={start_season}, train_end={train_end_season}, test={test_season}"
        )

    console.print(
        f"[bold]Fixed Split[/bold] — Train: {start_season}–{train_end_season}, Test: {test_season}"
    )

    # Configure baseline: probability source from args, no calibration, no prior anchoring
    cfg = get_config()
    cfg.model.calibrator_enabled = False
    cfg.model.prior_anchor_enabled = False
    cfg.model.prob_source = str(prob_source).strip().lower()
    if cfg.model.prob_source == "hybrid":
        # Use fixed weighting to respect hybrid_poisson_weight
        cfg.model.hybrid_scheme = "fixed"
        cfg.model.hybrid_poisson_weight = float(hybrid_poisson_weight)
    # Wire EP scoreline selection
    cfg.model.use_ep_selection = bool(use_ep_selection)

    # Load data
    console.status("Loading historical training seasons...")
    train_matches = loader.fetch_historical_seasons(start_season, train_end_season)
    console.status("Loading test season matches...")
    test_matches = loader.fetch_season_matches(test_season)

    # Create features
    console.status("Building training features...")
    train_df = loader.create_features_from_matches(train_matches)
    console.status("Building test features...")
    test_df = loader.create_features_from_matches(test_matches)

    if train_df.empty:
        raise RuntimeError("No training data available after feature engineering.")
    if test_df.empty:
        raise RuntimeError(
            "No test data available after feature engineering for the selected test season."
        )

    console.print(f"Training matches: {len(train_df)}, Test matches: {len(test_df)}")

    # Train simplified predictor
    predictor = MatchPredictor(quiet=quiet)
    predictor.train(train_df)

    # Evaluate on fixed test set
    metrics = predictor.evaluate(test_df)

    # Print a compact top-line summary
    ece = metrics.get("ece")
    if isinstance(ece, dict):
        try:
            ece_value = float(
                np.nanmean(
                    [float(ece.get(lab, float("nan"))) for lab in ("H", "D", "A")]
                )
            )
        except Exception:
            ece_value = float("nan")
    else:
        ece_value = float(ece) if ece is not None else float("nan")

    top = {
        "season_test": test_season,
        "seasons_train_start": start_season,
        "seasons_train_end": train_end_season,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "brier": float(metrics.get("brier", float("nan"))),
        "log_loss": float(metrics.get("log_loss", float("nan"))),
        "rps": float(metrics.get("rps", float("nan"))),
        "ece": ece_value,
        "avg_points": float(metrics.get("avg_points", float("nan"))),
        "total_points": float(metrics.get("total_points", float("nan"))),
        "accuracy": float(metrics.get("accuracy", float("nan"))),
    }
    console.print("\n[bold]Top-Line Metrics[/bold]")
    for k, v in top.items():
        console.print(f"- {k}: {v}")

    # Detailed diagnostics for the fixed 2024 test season
    preds = predictor.predict(test_df)

    # Actual outcomes
    ah = test_df["home_score"].astype(int).to_numpy()
    aa = test_df["away_score"].astype(int).to_numpy()
    y_true = np.where(ah > aa, "H", np.where(aa > ah, "A", "D")).tolist()

    # Probability matrix (H, D, A)
    P = np.array(
        [
            [
                float(p.get("home_win_probability", 1 / 3)),
                float(p.get("draw_probability", 1 / 3)),
                float(p.get("away_win_probability", 1 / 3)),
            ]
            for p in preds
        ],
        dtype=float,
    )
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    # Outcome Distribution Table
    label_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    pred_labels = [LABELS_ORDER[i] for i in np.argmax(P, axis=1)]
    pred_counts = {lab: pred_labels.count(lab) for lab in LABELS_ORDER}
    total = max(1, len(preds))

    dist_table = Table(title="Outcome Distribution", box=box.SIMPLE_HEAVY)
    dist_table.add_column("Label", justify="center")
    dist_table.add_column("Actual", justify="right")
    dist_table.add_column("Actual %", justify="right")
    dist_table.add_column("Predicted", justify="right")
    dist_table.add_column("Predicted %", justify="right")
    for lab in LABELS_ORDER:
        a = int(label_counts.get(lab, 0))
        p = int(pred_counts.get(lab, 0))
        dist_table.add_row(lab, str(a), f"{a/total:.1%}", str(p), f"{p/total:.1%}")
    console.print(dist_table)

    # Confusion Matrix
    cm_stats = confusion_matrix_stats(y_true, P)
    cm = np.array(cm_stats["matrix"], dtype=int)
    cm_table = Table(title="Confusion Matrix", box=box.SIMPLE_HEAVY)
    cm_table.add_column("Actual \\ Pred")
    for lab in LABELS_ORDER:
        cm_table.add_column(lab, justify="right")
    for i, lab in enumerate(LABELS_ORDER):
        row = [lab] + [str(int(cm[i, j])) for j in range(3)]
        cm_table.add_row(*row)
    console.print(cm_table)

    # Per-class Precision/Recall
    per_class = cm_stats.get("per_class", {}) if isinstance(cm_stats, dict) else {}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline with fixed train/test split"
    )
    parser.add_argument(
        "--seasons-back",
        type=int,
        default=5,
        help="Number of seasons back from current to include (train uses all but last).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training/evaluation logging.",
    )
    parser.add_argument(
        "--prob-source",
        type=str,
        default="classifier",
        choices=["classifier", "poisson", "hybrid"],
        help="Outcome probability source.",
    )
    parser.add_argument(
        "--hybrid-poisson-weight",
        type=float,
        default=0.5,
        help="When prob_source=hybrid, weight of Poisson probabilities in [0,1].",
    )
    parser.add_argument(
        "--use-ep-selection",
        action="store_true",
        help="Enable EP scoreline selection (maximize expected Kicktipp points).",
    )
    args = parser.parse_args()

    run_baseline(
        seasons_back=args.seasons_back,
        quiet=args.quiet,
        prob_source=args.prob_source,
        hybrid_poisson_weight=args.hybrid_poisson_weight,
        use_ep_selection=args.use_ep_selection,
    )
