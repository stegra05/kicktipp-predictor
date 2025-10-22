"""Evaluation module for match predictor.

This module provides metrics and evaluation functions for the MatchPredictor.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from pathlib import Path

from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import (
    LABELS_ORDER,
    brier_score_multiclass,
    log_loss_multiclass,
    ranked_probability_score_3c,
    expected_calibration_error,
    reliability_diagram,
    plot_reliability_curve,
    confusion_matrix_stats,
    plot_confusion_matrix,
    bin_by_confidence,
    plot_confidence_buckets,
    compute_points,
    ensure_dir,
    save_json,
)
from kicktipp_predictor.predictor import MatchPredictor


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
    rps = ranked_probability_score_3c(y_true, proba)

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


def run_evaluation(season: bool = False, dynamic: bool = False, retrain_every: int = 1) -> None:
    """Run comprehensive model evaluation.

    Args:
        season: If True, run season evaluation instead of test split evaluation.
        dynamic: If True and season=True, use dynamic retraining during season.
        retrain_every: How often to retrain when using dynamic evaluation.
    """
    if season:
        _run_season_evaluation(dynamic=dynamic, retrain_every=retrain_every)
        return

    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    loader = DataLoader()
    predictor = MatchPredictor()

    # Load trained models
    try:
        predictor.load_models()
    except FileNotFoundError:
        print("ERROR: No trained models found. Run training first.")
        return

    current_season = loader.get_current_season()
    start_season = current_season - 2

    all_matches = loader.fetch_historical_seasons(start_season, current_season)
    finished = [m for m in all_matches if m['is_finished']]
    print("\n" + "-"*80)
    print("DATASET")
    print("-"*80)
    print(f"Seasons range       : {start_season} - {current_season}")
    print(f"Total matches       : {len(all_matches)} (finished: {len(finished)})")
    uniq_teams = sorted({m.get('home_team') for m in all_matches} | {m.get('away_team') for m in all_matches} - {None})
    print(f"Unique teams        : {len(uniq_teams)}")

    # Create features
    print("Creating features...")
    features_df = loader.create_features_from_matches(all_matches)
    print("\n" + "-"*80)
    print("FEATURES")
    print("-"*80)
    print(f"Samples             : {len(features_df)}")
    try:
        print(f"Feature columns     : {len(features_df.columns)}")
        example_cols = [c for c in list(features_df.columns) if c not in ['home_score','away_score','goal_difference','result']][:10]
        if example_cols:
            print(f"Example columns     : {', '.join(example_cols)}")
    except Exception:
        pass

    # Use last 30% as test set
    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]
    train_len = split_idx
    print("\n" + "-"*80)
    print("SPLIT")
    print("-"*80)
    print(f"Train/Test sizes    : {train_len} / {len(test_df)} (70/30 split)")
    if 'date' in features_df.columns:
        try:
            tr_start = str(features_df.iloc[0]['date'])
            tr_end = str(features_df.iloc[max(0, split_idx-1)]['date'])
            te_start = str(test_df.iloc[0]['date']) if len(test_df) else "-"
            te_end = str(test_df.iloc[-1]['date']) if len(test_df) else "-"
            print(f"Train dates         : {tr_start} -> {tr_end}")
            print(f"Test dates          : {te_start} -> {te_end}")
        except Exception:
            pass
    print(f"Using {len(test_df)} matches for evaluation")

    # Prepare test features (without targets)
    test_features = test_df.drop(
        columns=['home_score', 'away_score', 'goal_difference', 'result'],
        errors='ignore'
    )

    # Helper: extract ground truth labels and arrays
    def _actual_labels(df: pd.DataFrame) -> List[str]:
        if 'result' in df.columns and df['result'].notna().all():
            return [str(x) for x in df['result'].tolist()]
        labels: List[str] = []
        for _, r in df.iterrows():
            if r['home_score'] > r['away_score']:
                labels.append('H')
            elif r['away_score'] > r['home_score']:
                labels.append('A')
            else:
                labels.append('D')
        return labels

    def _proba_from_preds(preds: List[Dict]) -> np.ndarray:
        P = []
        for p in preds:
            P.append([
                float(p.get('home_win_probability', 1/3)),
                float(p.get('draw_probability', 1/3)),
                float(p.get('away_win_probability', 1/3)),
            ])
        arr = np.asarray(P, dtype=float)
        arr = np.clip(arr, 1e-15, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr

    def _scores_from_preds(preds: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        ph = np.asarray([int(p.get('predicted_home_score', 0)) for p in preds], dtype=int)
        pa = np.asarray([int(p.get('predicted_away_score', 0)) for p in preds], dtype=int)
        return ph, pa

    def _confidence_bundle(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_prob = np.max(P, axis=1)
        sorted_probs = np.sort(P, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(P * np.log(P), axis=1)
        entropy_conf = 1.0 - (entropy / np.log(3))
        combined = 0.6 * max_prob + 0.4 * margin
        return max_prob, margin, entropy_conf, combined

    # Generate predictions
    preds = predictor.predict(test_features)

    # Ground truth
    y_true = _actual_labels(test_df)
    ah = np.asarray(test_df['home_score'], dtype=int)
    aa = np.asarray(test_df['away_score'], dtype=int)

    # Build proba matrix
    P = _proba_from_preds(preds)

    # Label distribution and sanity checks
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTIONS (Test)")
    print("-"*80)
    lab_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    total_labels = sum(lab_counts.values()) or 1
    print("Actual              : " + "  ".join([f"{lab}={lab_counts.get(lab,0)} ({lab_counts.get(lab,0)/total_labels:.2%})" for lab in LABELS_ORDER]))
    pred_labels = [LABELS_ORDER[i] for i in np.argmax(P, axis=1)]
    pred_counts = {lab: pred_labels.count(lab) for lab in LABELS_ORDER}
    print("Predicted           : " + "  ".join([f"{lab}={pred_counts.get(lab,0)} ({pred_counts.get(lab,0)/max(1,len(pred_labels)):.2%})" for lab in LABELS_ORDER]))

    # Predicted scores and points
    ph, pa = _scores_from_preds(preds)

    # Points per match
    pts = compute_points(ph, pa, ah, aa)

    # Metrics
    def _all_metrics(P: np.ndarray, points_vec: np.ndarray) -> Dict[str, object]:
        return {
            'brier': brier_score_multiclass(y_true, P),
            'log_loss': log_loss_multiclass(y_true, P),
            'rps': ranked_probability_score_3c(y_true, P),
            'ece': expected_calibration_error(y_true, P, n_bins=10),
            'avg_points': float(np.mean(points_vec)) if len(points_vec) else 0.0,
            'total_points': int(np.sum(points_vec)),
            'accuracy': float(np.mean(np.argmax(P, axis=1) == np.array([{'H':0,'D':1,'A':2}[t] for t in y_true]))),
        }

    metrics_main = _all_metrics(P, pts)

    # Metrics table in console
    print("\n" + "-"*80)
    print("METRICS")
    print("-"*80)
    print(f"avg_pts={metrics_main['avg_points']:.3f}  acc={metrics_main['accuracy']:.3f}  "
          f"brier={metrics_main['brier']:.4f}  logloss={metrics_main['log_loss']:.4f}  rps={metrics_main['rps']:.4f}")

    # Confidence bundle
    max_prob, margin, entropy_h, combined = _confidence_bundle(P)

    # Save artifacts
    out_dir = os.path.join('data', 'predictions')
    ensure_dir(out_dir)

    plot_dir = os.path.join('src', 'kicktipp_predictor', 'web', 'static', 'plots')
    ensure_dir(plot_dir)

    # metrics.json
    save_json({'main': metrics_main}, os.path.join(out_dir, 'metrics.json'))

    # Debug CSV for main predictions
    debug_rows: List[Dict] = []
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    for i in range(len(test_df)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true = float(P[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows.append({
            'match_id': int(test_df.iloc[i]['match_id']) if 'match_id' in test_df.columns else None,
            'home_team': test_features.iloc[i]['home_team'] if 'home_team' in test_features.columns else None,
            'away_team': test_features.iloc[i]['away_team'] if 'away_team' in test_features.columns else None,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P[i]))],
            'pH': float(P[i, 0]),
            'pD': float(P[i, 1]),
            'pA': float(P[i, 2]),
            'p_true': p_true,
            'points': int(pts[i]),
            'confidence': float(combined[i]),
            'margin': float(margin[i]),
            'entropy_conf': float(entropy_h[i]),
        })
    pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, 'debug_eval.csv'), index=False)

    # Calibration plots
    curve_H = reliability_diagram(y_true, P, 'H', n_bins=10)
    curve_D = reliability_diagram(y_true, P, 'D', n_bins=10)
    curve_A = reliability_diagram(y_true, P, 'A', n_bins=10)
    plot_reliability_curve(curve_H, 'H', os.path.join(plot_dir, 'calibration_curve_home.png'))
    plot_reliability_curve(curve_D, 'D', os.path.join(plot_dir, 'calibration_curve_draw.png'))
    plot_reliability_curve(curve_A, 'A', os.path.join(plot_dir, 'calibration_curve_away.png'))

    # Confusion matrix
    cm_stats = confusion_matrix_stats(y_true, P)
    cm = np.array(cm_stats['matrix'], dtype=int)
    plot_confusion_matrix(cm, os.path.join(plot_dir, 'confusion_matrix.png'))
    print("\nCONFUSION MATRIX")
    header = "         Pred ->   H     D     A"
    print(header)
    for i, lab in enumerate(LABELS_ORDER):
        row = cm[i]
        print(f"Actual {lab}       : {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    print(f"Overall accuracy   : {cm_stats.get('accuracy', float('nan')):.3f}")
    per_class = cm_stats.get('per_class', {})
    if isinstance(per_class, dict):
        for lab in LABELS_ORDER:
            stats = per_class.get(lab, {})
            pr = float(stats.get('precision', float('nan')))
            rc = float(stats.get('recall', float('nan')))
            print(f"Class {lab}         : precision={pr:.3f} recall={rc:.3f}")

    # Confidence bucket analysis
    _, _, _, combined_all = _confidence_bundle(P)
    conf_df = bin_by_confidence(combined_all, y_true, P, pts, n_bins=5)
    conf_df.to_csv(os.path.join(out_dir, 'confidence_buckets.csv'), index=False)
    plot_confidence_buckets(conf_df, os.path.join(out_dir, 'confidence_buckets.png'))
    if len(conf_df) > 0:
        print("\nCONFIDENCE BUCKETS")
        for _, r in conf_df.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # ECE snapshot
    ece = metrics_main.get('ece')
    if isinstance(ece, dict):
        print("\nECE by class: " + ", ".join([f"{lab}={float(ece.get(lab, float('nan'))):.4f}" for lab in LABELS_ORDER]))

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"avg_pts={metrics_main['avg_points']:.3f}  acc={metrics_main['accuracy']:.3f}  "
          f"brier={metrics_main['brier']:.4f}  logloss={metrics_main['log_loss']:.4f}  rps={metrics_main['rps']:.4f}")
    print(f"Artifacts written to {out_dir}")


def _run_season_evaluation(dynamic: bool = False, retrain_every: int = 1) -> None:
    """Run season-long evaluation with optional dynamic retraining.

    Args:
        dynamic: If True, retrain model as season progresses.
        retrain_every: How many matchdays between retraining.
    """
    import sys

    print("="*80)
    print("SEASON PERFORMANCE EVALUATION")
    print("="*80)
    print()

    # Initialize components
    data_loader = DataLoader()
    predictor = MatchPredictor()

    # Load trained models
    print("Loading models...")
    try:
        predictor.load_models()
        print("Models loaded successfully!\n")
    except FileNotFoundError:
        print("\nERROR: No trained models found.")
        print("Please run training first to train the models.")
        sys.exit(1)

    # Get current season data
    current_season = data_loader.get_current_season()
    print(f"Fetching data for current season: {current_season}")
    season_matches = data_loader.fetch_season_matches(current_season)

    finished_matches = [m for m in season_matches if m['is_finished']]
    if not finished_matches:
        print("No finished matches found for the current season.")
        sys.exit(0)

    # Determine matchdays to evaluate
    first_matchday = min(m['matchday'] for m in finished_matches)
    last_matchday = max(m['matchday'] for m in finished_matches)
    print(f"Evaluating matchdays from {first_matchday} to {last_matchday}\n")

    all_predictions: list[dict] = []

    # Get historical data for feature context (finished-only; augment with previous season if sparse)
    print("Fetching historical data for context...")
    historical_matches_all = data_loader.fetch_season_matches(current_season)
    historical_finished = [m for m in historical_matches_all if m.get('is_finished')]
    if len(historical_finished) < 50:
        print("Fetching additional finished matches from previous season...")
        prev_season_matches = data_loader.fetch_season_matches(current_season - 1)
        historical_finished.extend([m for m in prev_season_matches if m.get('is_finished')])

    if not dynamic:
        # Original train-once prediction loop using a fixed historical context
        historical_matches = list(historical_finished)
        for matchday in range(first_matchday, last_matchday + 1):
            print(f"--- Processing Matchday {matchday} ---")

            matchday_matches = [m for m in finished_matches if m['matchday'] == matchday]
            if not matchday_matches:
                print(f"No finished matches for matchday {matchday}.")
                continue

            features_df = data_loader.create_prediction_features(
                matchday_matches, historical_matches
            )

            if features_df.empty:
                print(f"Could not generate features for matchday {matchday}.")
                continue

            predictions = predictor.predict(features_df)

            for pred, actual in zip(predictions, matchday_matches):
                ph = int(pred['predicted_home_score'])
                pa = int(pred['predicted_away_score'])
                ah = int(actual['home_score'])
                aa = int(actual['away_score'])
                if ph == ah and pa == aa:
                    points = 4
                elif (ph - pa) == (ah - aa):
                    points = 3
                else:
                    pred_winner = 'H' if ph > pa else ('A' if pa > ph else 'D')
                    actual_winner = 'H' if ah > aa else ('A' if aa > ah else 'D')
                    points = 2 if pred_winner == actual_winner else 0
                pred['actual_home_score'] = actual['home_score']
                pred['actual_away_score'] = actual['away_score']
                pred['points_earned'] = points
                pred['is_evaluated'] = True
                pred['matchday'] = matchday
                all_predictions.append(pred)

            print(f"Generated and evaluated {len(predictions)} predictions for matchday {matchday}")
    else:
        # --- DYNAMIC expanding-window evaluation ---
        print("\nRunning DYNAMIC expanding-window evaluation...")

        # 1. Separate historical base from the season to be evaluated
        all_historical_matches = data_loader.fetch_historical_seasons(current_season - 3, current_season - 1)

        # The matches we will loop through and evaluate
        evaluation_season_matches = [m for m in finished_matches if m['is_finished']]

        # The training set starts with all prior seasons and will grow
        cumulative_training_matches = list(all_historical_matches)
        print(f"Initialized training set with {len(cumulative_training_matches)} matches from previous seasons.")

        for matchday in range(first_matchday, last_matchday + 1):
            print(f"\n--- Processing Matchday {matchday} ---")

            # 2. Retrain on schedule using the full cumulative dataset
            if (matchday - first_matchday) % max(1, int(retrain_every)) == 0:
                print(f"Retraining model with {len(cumulative_training_matches)} total matches...")
                train_df = data_loader.create_features_from_matches(cumulative_training_matches)
                predictor.train(train_df)
                print("Model retrained successfully.")

            # 3. Predict on the current matchday's matches
            matchday_matches_to_predict = [m for m in evaluation_season_matches if m['matchday'] == matchday]
            if not matchday_matches_to_predict:
                continue

            # Use the cumulative data as context for feature generation
            features_df = data_loader.create_prediction_features(matchday_matches_to_predict, cumulative_training_matches)

            predictions = predictor.predict(features_df)
            for pred, actual in zip(predictions, matchday_matches_to_predict):
                ph = int(pred['predicted_home_score'])
                pa = int(pred['predicted_away_score'])
                ah = int(actual['home_score'])
                aa = int(actual['away_score'])
                if ph == ah and pa == aa:
                    points = 4
                elif (ph - pa) == (ah - aa):
                    points = 3
                else:
                    pred_winner = 'H' if ph > pa else ('A' if pa > ph else 'D')
                    actual_winner = 'H' if ah > aa else ('A' if aa > ah else 'D')
                    points = 2 if pred_winner == actual_winner else 0
                pred['actual_home_score'] = actual['home_score']
                pred['actual_away_score'] = actual['away_score']
                pred['points_earned'] = points
                pred['is_evaluated'] = True
                pred['matchday'] = matchday
                all_predictions.append(pred)

            # 4. Add the now-finished matchday to the training pool for the next iteration
            cumulative_training_matches.extend(matchday_matches_to_predict)

    if not all_predictions:
        print("\nNo predictions could be generated for the season.")
        sys.exit(0)

    # Overall basic metrics snapshot
    total_matches = len(all_predictions)
    total_points = sum(p['points_earned'] for p in all_predictions)
    avg_points = total_points / total_matches if total_matches > 0 else 0
    print(f"\nTotal Matches Evaluated: {total_matches}")
    print(f"Total Points Earned: {total_points}")
    print(f"Average Points per Match: {avg_points:.3f}")

    # ------------------------------------------------------------------
    # Expanded season analysis (align with non-season evaluation depth)
    # ------------------------------------------------------------------
    use_dynamic_preds = bool(dynamic)

    def _proba_from_preds(preds: list) -> np.ndarray:
        mat = []
        for p in preds:
            mat.append([
                float(p.get('home_win_probability', 1/3)),
                float(p.get('draw_probability', 1/3)),
                float(p.get('away_win_probability', 1/3)),
            ])
        arr = np.asarray(mat, dtype=float)
        arr = np.clip(arr, 1e-15, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr

    def _scores_from_preds(preds: list) -> tuple:
        ph = np.asarray([int(p.get('predicted_home_score', 0)) for p in preds], dtype=int)
        pa = np.asarray([int(p.get('predicted_away_score', 0)) for p in preds], dtype=int)
        return ph, pa

    if use_dynamic_preds:
        preds_all = list(all_predictions)
        # Build ground truth and probabilities from stored predictions
        ah = np.asarray([int(p.get('actual_home_score', 0)) for p in preds_all], dtype=int)
        aa = np.asarray([int(p.get('actual_away_score', 0)) for p in preds_all], dtype=int)
        y_true = []
        for i in range(len(preds_all)):
            if ah[i] > aa[i]:
                y_true.append('H')
            elif aa[i] > ah[i]:
                y_true.append('A')
            else:
                y_true.append('D')

        P = np.array([
            [float(p.get('home_win_probability', 1/3)), float(p.get('draw_probability', 1/3)), float(p.get('away_win_probability', 1/3))]
            for p in preds_all
        ], dtype=float)
        P = np.clip(P, 1e-15, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

        ph = np.asarray([int(p.get('predicted_home_score', 0)) for p in preds_all], dtype=int)
        pa = np.asarray([int(p.get('predicted_away_score', 0)) for p in preds_all], dtype=int)
        pts = compute_points(ph, pa, ah, aa)
        features_like = pd.DataFrame({
            'match_id': [p.get('match_id') for p in preds_all],
            'matchday': [p.get('matchday') for p in preds_all],
            'home_team': [p.get('home_team') for p in preds_all],
            'away_team': [p.get('away_team') for p in preds_all],
        })
    else:
        # Build a unified features dataframe for all finished matches
        season_df = pd.DataFrame(finished_matches)
        features_all = data_loader.create_prediction_features(
            finished_matches, historical_finished
        )
        if features_all is None or len(features_all) == 0:
            print("\nUnable to build features for season-level analysis.")
            return

        # Generate season-wide predictions
        print("\nGenerating season-wide predictions...")
        preds_all = predictor.predict(features_all)

        # Ground truth aligned to features_all order
        id_to_actual = {m['match_id']: (m['home_score'], m['away_score']) for m in finished_matches if m.get('match_id') is not None}
        actual_home_list = []
        actual_away_list = []
        y_true = []
        for _, row in features_all.iterrows():
            mid = int(row['match_id']) if 'match_id' in features_all.columns else None
            if mid is None or mid not in id_to_actual:
                actual_home_list.append(np.nan)
                actual_away_list.append(np.nan)
                y_true.append('D')
            else:
                ah_i, aa_i = id_to_actual[mid]
                actual_home_list.append(int(ah_i))
                actual_away_list.append(int(aa_i))
                if ah_i > aa_i:
                    y_true.append('H')
                elif aa_i > ah_i:
                    y_true.append('A')
                else:
                    y_true.append('D')
        ah = np.asarray(actual_home_list, dtype=int)
        aa = np.asarray(actual_away_list, dtype=int)

        # Probability matrix
        P = _proba_from_preds(preds_all)

        # Scores and points
        ph, pa = _scores_from_preds(preds_all)
        pts = compute_points(ph, pa, ah, aa)
        features_like = features_all.copy()

    # Label distribution and sanity
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTIONS (Season)")
    print("-"*80)
    lab_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    total_labels = sum(lab_counts.values()) or 1
    print("Actual              : " + "  ".join([f"{lab}={lab_counts.get(lab,0)} ({lab_counts.get(lab,0)/total_labels:.2%})" for lab in LABELS_ORDER]))
    pred_labels = [LABELS_ORDER[i] for i in np.argmax(P, axis=1)]
    pred_counts = {lab: pred_labels.count(lab) for lab in LABELS_ORDER}
    print("Predicted           : " + "  ".join([f"{lab}={pred_counts.get(lab,0)} ({pred_counts.get(lab,0)/max(1,len(pred_labels)):.2%})" for lab in LABELS_ORDER]))

    # Scores and points
    ph, pa = _scores_from_preds(preds_all)
    pts = compute_points(ph, pa, ah, aa)

    def _all_metrics(P: np.ndarray, points_vec: np.ndarray) -> dict:
        return {
            'brier': brier_score_multiclass(y_true, P),
            'log_loss': log_loss_multiclass(y_true, P),
            'rps': ranked_probability_score_3c(y_true, P),
            'ece': expected_calibration_error(y_true, P, n_bins=10),
            'avg_points': float(np.mean(points_vec)) if len(points_vec) else 0.0,
            'total_points': int(np.sum(points_vec)),
            'accuracy': float(np.mean(np.argmax(P, axis=1) == np.array([{'H':0,'D':1,'A':2}[t] for t in y_true]))),
        }

    metrics = _all_metrics(P, pts)

    # Metrics table
    print("\n" + "-"*80)
    print("SEASON METRICS")
    print("-"*80)
    print(f"avg_pts={metrics['avg_points']:.3f}  acc={metrics['accuracy']:.3f}  "
          f"brier={metrics['brier']:.4f}  logloss={metrics['log_loss']:.4f}  rps={metrics['rps']:.4f}")

    # Points distribution
    print("\nPOINTS DISTRIBUTION (Season)")
    unique_pts = [0, 2, 3, 4]
    pt_counts = {p: int(np.sum(pts == p)) for p in unique_pts}
    print("Counts by points    : " + ", ".join([f"{p}p={pt_counts[p]}" for p in unique_pts]))
    print(f"Avg points          : {np.mean(pts) if len(pts) else 0.0:.3f}  Total: {int(np.sum(pts))}")

    # Prediction quality breakdown
    print("\n" + "-"*80)
    print("PREDICTION QUALITY (Season)")
    print("-"*80)
    exact_scores = int(np.sum((ph==ah) & (pa==aa)))
    correct_diffs = int(np.sum((ph-pa)==(ah-aa)))
    correct_results = int(np.sum(((ph>pa)&(ah>aa)) | ((ph==pa)&(ah==aa)) | ((ph<pa)&(ah<aa))))
    print(f"Exact scores        : {exact_scores:3d} ({exact_scores/len(pts)*100:.1f}%)")
    print(f"Correct differences : {correct_diffs:3d} ({correct_diffs/len(pts)*100:.1f}%)")
    print(f"Correct results     : {correct_results:3d} ({correct_results/len(pts)*100:.1f}%)")

    # Confidence buckets
    def _confidence_bundle(P_: np.ndarray) -> tuple:
        max_prob = np.max(P_, axis=1)
        sorted_probs = np.sort(P_, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(P_ * np.log(P_), axis=1)
        entropy_conf = 1.0 - (entropy / np.log(3))
        combined = 0.6 * max_prob + 0.4 * margin
        return max_prob, margin, entropy_conf, combined

    _, _, _, combined_all = _confidence_bundle(P)
    out_dir = os.path.join('data', 'predictions')
    ensure_dir(out_dir)
    conf_df = bin_by_confidence(combined_all, y_true, P, pts, n_bins=5)
    conf_season_csv = os.path.join(out_dir, 'confidence_buckets_season.csv')
    conf_df.to_csv(conf_season_csv, index=False)
    try:
        plot_confidence_buckets(conf_df, os.path.join(out_dir, 'confidence_buckets_season.png'))
    except Exception:
        pass
    if len(conf_df) > 0:
        print("\nCONFIDENCE BUCKETS (Season)")
        for _, r in conf_df.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # Calibration curves and confusion matrices
    try:
        curve_H = reliability_diagram(y_true, P, 'H', n_bins=10)
        curve_D = reliability_diagram(y_true, P, 'D', n_bins=10)
        curve_A = reliability_diagram(y_true, P, 'A', n_bins=10)
        plot_reliability_curve(curve_H, 'H', os.path.join(out_dir, 'calibration_home_season.png'))
        plot_reliability_curve(curve_D, 'D', os.path.join(out_dir, 'calibration_draw_season.png'))
        plot_reliability_curve(curve_A, 'A', os.path.join(out_dir, 'calibration_away_season.png'))
    except Exception:
        pass

    cm_stats = confusion_matrix_stats(y_true, P)
    cm = np.array(cm_stats['matrix'], dtype=int)
    try:
        plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix_season.png'))
    except Exception:
        pass
    print("\nCONFUSION MATRIX (Season)")
    header = "         Pred ->   H     D     A"
    print(header)
    for i, lab in enumerate(LABELS_ORDER):
        row = cm[i]
        print(f"Actual {lab}       : {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    print(f"Overall accuracy   : {cm_stats.get('accuracy', float('nan')):.3f}")
    per_class = cm_stats.get('per_class', {})
    if isinstance(per_class, dict):
        for lab in LABELS_ORDER:
            stats = per_class.get(lab, {})
            pr = float(stats.get('precision', float('nan')))
            rc = float(stats.get('recall', float('nan')))
            print(f"Class {lab}         : precision={pr:.3f} recall={rc:.3f}")

    # Save consolidated artifacts
    season_metrics = {'main': metrics}
    save_json(season_metrics, os.path.join(out_dir, 'metrics_season.json'))
    with open(os.path.join(out_dir, 'metrics_table_season.txt'), 'w', encoding='utf-8') as f:
        f.write(f"main      avg_pts={metrics['avg_points']:.3f}  acc={metrics['accuracy']:.3f}  "
                f"brier={metrics['brier']:.4f}  logloss={metrics['log_loss']:.4f}  rps={metrics['rps']:.4f}\n")

    # Build and save debug CSV for season
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    debug_rows = []
    for i in range(len(preds_all)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true = float(P[i, true_idx]) if true_idx >= 0 else float('nan')
        mid_val = features_like.iloc[i]['match_id'] if 'match_id' in features_like.columns else None
        md_val = features_like.iloc[i]['matchday'] if 'matchday' in features_like.columns else None
        ht_val = features_like.iloc[i]['home_team'] if 'home_team' in features_like.columns else None
        at_val = features_like.iloc[i]['away_team'] if 'away_team' in features_like.columns else None
        debug_rows.append({
            'match_id': int(mid_val) if mid_val is not None and not pd.isna(mid_val) else None,
            'matchday': int(md_val) if md_val is not None and not pd.isna(md_val) else None,
            'home_team': ht_val,
            'away_team': at_val,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P[i]))],
            'pH': float(P[i, 0]),
            'pD': float(P[i, 1]),
            'pA': float(P[i, 2]),
            'p_true': p_true,
            'points': int(pts[i]),
        })
    pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, 'debug_season.csv'), index=False)
    print(f"\nSeason artifacts written to {out_dir}")

    # ------------------------------------------------------------------
    # Per-matchday metrics with baseline comparison and plots
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    # Build matchday -> indices mapping (aligned with features_like / preds_all / P / pts)
    if 'matchday' not in features_like.columns:
        print("\nWARNING: No 'matchday' column in features; skipping per-matchday breakdown.")
        return

    matchdays: list[int] = []
    md_to_idx: dict[int, list[int]] = {}
    for i, md_val in enumerate(features_like['matchday'].tolist()):
        try:
            md = int(md_val)
        except Exception:
            continue
        if md not in md_to_idx:
            md_to_idx[md] = []
            matchdays.append(md)
        md_to_idx[md].append(i)

    matchdays = sorted(matchdays)

    # Helper for true-class prob per row
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y_idx_all = np.array([mapping.get(lbl, -1) for lbl in y_true], dtype=int)

    rows = []
    cum_points = 0
    cum_points_baseline = 0

    avg_pts_series = []
    avg_pts_series_baseline = []
    cum_pts_series = []
    cum_pts_series_baseline = []

    for md in matchdays:
        idx_list = md_to_idx.get(md, [])
        if not idx_list:
            continue

        idx = np.array(idx_list, dtype=int)

        # Slices
        P_md = P[idx]
        y_md = [y_true[i] for i in idx_list]
        y_idx_md = y_idx_all[idx]
        ph_md = ph[idx]
        pa_md = pa[idx]
        ah_md = ah[idx]
        aa_md = aa[idx]
        pts_md = pts[idx]

        n_md = len(idx)

        # Points
        total_pts = int(np.sum(pts_md))
        avg_pts = float(np.mean(pts_md)) if n_md else 0.0
        points_0 = int(np.sum(pts_md == 0))
        points_2 = int(np.sum(pts_md == 2))
        points_3 = int(np.sum(pts_md == 3))
        points_4 = int(np.sum(pts_md == 4))

        # Classification accuracy
        acc = float(np.mean(np.argmax(P_md, axis=1) == y_idx_md)) if n_md else float('nan')

        # Exact/diff/result counts
        exact_mask = (ph_md == ah_md) & (pa_md == aa_md)
        diff_mask = ((ph_md - pa_md) == (ah_md - aa_md)) & (~exact_mask)
        result_mask = (((ph_md > pa_md) & (ah_md > aa_md)) | ((ph_md == pa_md) & (ah_md == aa_md)) | ((ph_md < pa_md) & (ah_md < aa_md))) & (~exact_mask) & (~diff_mask)
        exact_count = int(np.sum(exact_mask))
        diff_count = int(np.sum(diff_mask))
        result_count = int(np.sum(result_mask))

        # Probabilistic metrics
        brier = brier_score_multiclass(y_md, P_md)
        logloss = log_loss_multiclass(y_md, P_md)
        rps = ranked_probability_score_3c(y_md, P_md)

        # Confidence: avg max-prob, avg true-class prob
        max_prob = float(np.mean(np.max(P_md, axis=1))) if n_md else float('nan')
        p_true_vals = []
        for i_row in range(n_md):
            ti = y_idx_md[i_row]
            if ti >= 0:
                p_true_vals.append(float(P_md[i_row, ti]))
        avg_true_prob = float(np.mean(p_true_vals)) if p_true_vals else float('nan')

        # Score errors (MAE)
        mae_home = float(np.mean(np.abs(ph_md - ah_md))) if n_md else float('nan')
        mae_away = float(np.mean(np.abs(pa_md - aa_md))) if n_md else float('nan')
        mae_gd = float(np.mean(np.abs((ph_md - pa_md) - (ah_md - aa_md)))) if n_md else float('nan')

        # Baseline: always 2-1 home
        base_ph = np.full(n_md, 2, dtype=int)
        base_pa = np.full(n_md, 1, dtype=int)
        base_pts = compute_points(base_ph, base_pa, ah_md, aa_md)
        base_total = int(np.sum(base_pts))
        base_avg = float(np.mean(base_pts)) if n_md else 0.0
        base_acc = float(np.mean(np.array(y_md) == 'H')) if n_md else float('nan')

        # Deltas
        d_avg = avg_pts - base_avg
        d_total = total_pts - base_total
        d_acc = acc - base_acc

        # Cumulative
        cum_points += total_pts
        cum_points_baseline += base_total

        avg_pts_series.append(avg_pts)
        avg_pts_series_baseline.append(base_avg)
        cum_pts_series.append(cum_points)
        cum_pts_series_baseline.append(cum_points_baseline)

        rows.append({
            'matchday': md,
            'n': n_md,
            'avg_points': avg_pts,
            'total_points': total_pts,
            'points_0': points_0,
            'points_2': points_2,
            'points_3': points_3,
            'points_4': points_4,
            'accuracy': float(acc),
            'exact_count': exact_count,
            'diff_count': diff_count,
            'result_count': result_count,
            'brier': float(brier),
            'log_loss': float(logloss),
            'rps': float(rps),
            'avg_max_prob': max_prob,
            'avg_true_prob': avg_true_prob,
            'mae_home': mae_home,
            'mae_away': mae_away,
            'mae_gd': mae_gd,
            'baseline_avg_points': base_avg,
            'baseline_total_points': base_total,
            'baseline_accuracy': float(base_acc),
            'delta_avg_points': d_avg,
            'delta_total_points': d_total,
            'delta_accuracy': d_acc,
            'cum_points': cum_points,
            'cum_points_baseline': cum_points_baseline,
        })

    # Save CSV
    per_md_df = pd.DataFrame(rows)
    per_md_csv = os.path.join(out_dir, 'per_matchday_metrics_season.csv')
    try:
        per_md_df.sort_values('matchday').to_csv(per_md_csv, index=False)
        print(f"Per-matchday metrics written to {per_md_csv}")
    except Exception as e:
        print(f"Failed to write per-matchday CSV: {e}")

    # Console table
    if len(rows) > 0:
        print("\n" + "-"*80)
        print("PER-MATCHDAY SUMMARY")
        print("-"*80)
        print(f"{'MD':>3}  {'n':>2}  {'avg':>5}  {'tot':>4}  {'base':>5}  {'Δavg':>5}  {'acc':>5}  {'ex/diff/res':>10}")
        for r in sorted(rows, key=lambda x: x['matchday']):
            ex_dr = f"{r['exact_count']}/{r['diff_count']}/{r['result_count']}"
            print(f"{int(r['matchday']):3d}  {int(r['n']):2d}  {r['avg_points']:.2f}  {int(r['total_points']):4d}  "
                  f"{r['baseline_avg_points']:.2f}  {r['delta_avg_points']:+.2f}  {r['accuracy']:.2f}  {ex_dr:>10}")

    # Plots
    if plt is not None and len(matchdays) == len(avg_pts_series):
        try:
            # Per-matchday average points
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(matchdays, avg_pts_series, label='Model', marker='o')
            ax.plot(matchdays, avg_pts_series_baseline, label='Baseline (2-1 H)', marker='o')
            ax.set_xlabel('Matchday')
            ax.set_ylabel('Avg points')
            ax.set_title('Per-matchday Avg Points (Season)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'per_matchday_points_season.png'))
            plt.close(fig)

            # Cumulative points
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(matchdays, cum_pts_series, label='Model', marker='o')
            ax2.plot(matchdays, cum_pts_series_baseline, label='Baseline (2-1 H)', marker='o')
            ax2.set_xlabel('Matchday')
            ax2.set_ylabel('Cumulative points')
            ax2.set_title('Cumulative Points (Season)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'per_matchday_points_cum_season.png'))
            plt.close(fig2)
        except Exception:
            pass
