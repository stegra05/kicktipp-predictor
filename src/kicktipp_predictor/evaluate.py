"""Evaluation module for match predictor.

This module provides metrics and evaluation functions for the MatchPredictor.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Iterable
from pathlib import Path

# Label order for consistency
LABELS_ORDER = ('H', 'D', 'A')


def compute_points(pred_home: Iterable[int], pred_away: Iterable[int],
                   act_home: Iterable[int], act_away: Iterable[int]) -> np.ndarray:
    """Compute Kicktipp points for predictions.

    Points system:
    - 4 points: Exact score
    - 3 points: Correct goal difference
    - 2 points: Correct outcome (H/D/A)
    - 0 points: Wrong outcome

    Args:
        pred_home: Predicted home scores.
        pred_away: Predicted away scores.
        act_home: Actual home scores.
        act_away: Actual away scores.

    Returns:
        Array of points for each prediction.
    """
    ph = np.asarray(list(pred_home), dtype=int)
    pa = np.asarray(list(pred_away), dtype=int)
    ah = np.asarray(list(act_home), dtype=int)
    aa = np.asarray(list(act_away), dtype=int)

    points = np.zeros_like(ph, dtype=int)

    # 4 points: exact score
    exact = (ph == ah) & (pa == aa)
    points[exact] = 4

    # 3 points: correct goal difference (but not exact)
    not_exact = ~exact
    gd_ok = ((ph - pa) == (ah - aa)) & not_exact
    points[gd_ok] = 3

    # 2 points: correct outcome (but not goal difference)
    rem = ~(exact | gd_ok)
    pred_outcome = np.where(ph > pa, "H", np.where(pa > ph, "A", "D"))
    act_outcome = np.where(ah > aa, "H", np.where(aa > ah, "A", "D"))
    points[(pred_outcome == act_outcome) & rem] = 2

    return points


def brier_score_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    """Compute Brier score for multiclass predictions.

    Args:
        y_true: True labels (H/D/A).
        proba: Probability matrix (n_samples, 3) for [H, D, A].

    Returns:
        Brier score (lower is better).
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)

    # Ensure proba is valid
    P = np.clip(proba, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    # One-hot encode true labels
    n = len(y)
    Y = np.zeros_like(P)
    valid = (y >= 0)
    Y[np.arange(n)[valid], y[valid]] = 1.0

    # Compute Brier score
    diffs = (P - Y)[valid]
    if len(diffs) == 0:
        return float('nan')

    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def log_loss_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    """Compute log loss for multiclass predictions.

    Args:
        y_true: True labels (H/D/A).
        proba: Probability matrix (n_samples, 3) for [H, D, A].

    Returns:
        Log loss (lower is better).
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)

    # Ensure proba is valid
    P = np.clip(proba, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    valid = (y >= 0)
    idx = np.arange(len(y))[valid]
    if len(idx) == 0:
        return float('nan')

    p_true = P[idx, y[valid]]
    return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))


def ranked_probability_score(y_true: List[str], proba: np.ndarray) -> float:
    """Compute Ranked Probability Score (RPS) for ordered outcomes.

    RPS measures the quality of probabilistic predictions for ordered categories.

    Args:
        y_true: True labels (H/D/A).
        proba: Probability matrix (n_samples, 3) for [H, D, A].

    Returns:
        RPS (lower is better).
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)

    # Ensure proba is valid
    P = np.clip(proba, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    valid = (y >= 0)
    if not np.any(valid):
        return float('nan')

    rps_scores = []
    for i in np.where(valid)[0]:
        true_idx = y[i]
        # Cumulative distributions
        cdf_pred = np.cumsum(P[i, :])
        cdf_true = np.zeros(3)
        cdf_true[true_idx:] = 1.0

        # RPS for this sample
        rps = np.sum((cdf_pred - cdf_true) ** 2) / 2.0  # Normalize by K-1 = 2
        rps_scores.append(rps)

    return float(np.mean(rps_scores))


def evaluate_predictor(predictor, test_df: pd.DataFrame) -> Dict:
    """Evaluate predictor on test data.

    Args:
        predictor: MatchPredictor instance.
        test_df: DataFrame with features and actual results.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Prepare test features
    test_features = test_df.drop(
        columns=['home_score', 'away_score', 'goal_difference', 'result'],
        errors='ignore'
    )

    # Generate predictions
    predictions = predictor.predict(test_features)

    # Extract ground truth
    y_true = test_df['result'].tolist()
    actual_home = test_df['home_score'].values
    actual_away = test_df['away_score'].values

    # Extract predictions
    pred_home = np.array([p['predicted_home_score'] for p in predictions])
    pred_away = np.array([p['predicted_away_score'] for p in predictions])

    # Build probability matrix [H, D, A]
    proba = np.array([
        [p['home_win_probability'], p['draw_probability'], p['away_win_probability']]
        for p in predictions
    ])

    # Predicted outcomes
    pred_outcomes = np.array([LABELS_ORDER[i] for i in np.argmax(proba, axis=1)])

    # Compute metrics
    points = compute_points(pred_home, pred_away, actual_home, actual_away)

    # Accuracy
    accuracy = float(np.mean(pred_outcomes == np.array(y_true)))

    # Probabilistic metrics
    brier = brier_score_multiclass(y_true, proba)
    logloss = log_loss_multiclass(y_true, proba)
    rps = ranked_probability_score(y_true, proba)

    # Points statistics
    avg_points = float(np.mean(points))
    total_points = int(np.sum(points))

    # Count by outcome
    label_counts = {lab: int(np.sum(np.array(y_true) == lab)) for lab in LABELS_ORDER}
    pred_counts = {lab: int(np.sum(pred_outcomes == lab)) for lab in LABELS_ORDER}

    # Count by points
    points_dist = {
        '0pt': int(np.sum(points == 0)),
        '2pt': int(np.sum(points == 2)),
        '3pt': int(np.sum(points == 3)),
        '4pt': int(np.sum(points == 4)),
    }

    metrics = {
        'accuracy': accuracy,
        'brier_score': brier,
        'log_loss': logloss,
        'rps': rps,
        'avg_points': avg_points,
        'total_points': total_points,
        'n_samples': len(test_df),
        'label_distribution': label_counts,
        'predicted_distribution': pred_counts,
        'points_distribution': points_dist,
    }

    return metrics


def print_evaluation_report(metrics: Dict, benchmark: Dict | None = None):
    """Print a formatted evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics.
        benchmark: Optional benchmark metrics for comparison.
    """
    print("=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print()

    print("PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Samples:           {metrics['n_samples']}")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Avg Points:        {metrics['avg_points']:.3f}")
    print(f"Total Points:      {metrics['total_points']}")
    print()
    print(f"Brier Score:       {metrics['brier_score']:.4f}")
    print(f"Log Loss:          {metrics['log_loss']:.4f}")
    print(f"RPS:               {metrics['rps']:.4f}")
    print()

    print("OUTCOME DISTRIBUTION")
    print("-" * 80)
    label_dist = metrics['label_distribution']
    pred_dist = metrics['predicted_distribution']
    total = metrics['n_samples']

    print("Actual:   ", end="")
    for lab in LABELS_ORDER:
        count = label_dist.get(lab, 0)
        pct = count / total if total > 0 else 0
        print(f"{lab}={count:3d} ({pct:.1%})  ", end="")
    print()

    print("Predicted:", end="")
    for lab in LABELS_ORDER:
        count = pred_dist.get(lab, 0)
        pct = count / total if total > 0 else 0
        print(f"{lab}={count:3d} ({pct:.1%})  ", end="")
    print()
    print()

    print("POINTS DISTRIBUTION")
    print("-" * 80)
    points_dist = metrics['points_distribution']
    for pt_label in ['0pt', '2pt', '3pt', '4pt']:
        count = points_dist.get(pt_label, 0)
        pct = count / total if total > 0 else 0
        print(f"{pt_label}: {count:3d} ({pct:.1%})")
    print()

    if benchmark:
        print("BENCHMARK COMPARISON")
        print("-" * 80)
        print(f"Accuracy:    {metrics['accuracy']:.3f}  vs  {benchmark.get('accuracy', 0):.3f} (benchmark)")
        print(f"Avg Points:  {metrics['avg_points']:.3f}  vs  {benchmark.get('avg_points', 0):.3f} (benchmark)")
        improvement = metrics['avg_points'] - benchmark.get('avg_points', 0)
        print(f"Improvement: {improvement:+.3f} points/match")
        print()

    print("=" * 80)


def simple_benchmark(test_df: pd.DataFrame, strategy: str = 'home_win') -> Dict:
    """Generate a simple baseline prediction for benchmarking.

    Args:
        test_df: DataFrame with actual results.
        strategy: Benchmark strategy ('home_win', 'draw', 'most_common').

    Returns:
        Dictionary with benchmark metrics.
    """
    actual_home = test_df['home_score'].values
    actual_away = test_df['away_score'].values
    y_true = test_df['result'].tolist()

    if strategy == 'home_win':
        # Always predict 2-1 home win
        pred_home = np.full(len(test_df), 2)
        pred_away = np.full(len(test_df), 1)
        pred_outcomes = np.full(len(test_df), 'H')
    elif strategy == 'draw':
        # Always predict 1-1 draw
        pred_home = np.full(len(test_df), 1)
        pred_away = np.full(len(test_df), 1)
        pred_outcomes = np.full(len(test_df), 'D')
    elif strategy == 'most_common':
        # Predict the most common outcome in training data
        from collections import Counter
        most_common_outcome = Counter(y_true).most_common(1)[0][0]
        pred_outcomes = np.full(len(test_df), most_common_outcome)
        if most_common_outcome == 'H':
            pred_home = np.full(len(test_df), 2)
            pred_away = np.full(len(test_df), 1)
        elif most_common_outcome == 'A':
            pred_home = np.full(len(test_df), 1)
            pred_away = np.full(len(test_df), 2)
        else:
            pred_home = np.full(len(test_df), 1)
            pred_away = np.full(len(test_df), 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Compute metrics
    points = compute_points(pred_home, pred_away, actual_home, actual_away)
    accuracy = float(np.mean(pred_outcomes == np.array(y_true)))

    return {
        'strategy': strategy,
        'accuracy': accuracy,
        'avg_points': float(np.mean(points)),
        'total_points': int(np.sum(points)),
    }
