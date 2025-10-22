from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import (
    ensure_dir,
    LABELS_ORDER,
    brier_score_multiclass,
    log_loss_multiclass,
    ranked_probability_score_3c,
    expected_calibration_error,
    reliability_diagram,
    confusion_matrix_stats,
    plot_reliability_curve,
    plot_confusion_matrix,
    bin_by_confidence,
    plot_confidence_buckets,
)
from kicktipp_predictor.predictor import MatchPredictor

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple


def run_evaluation(season: bool = False, dynamic: bool = False, retrain_every: int = 1) -> None:
    if season:
        from kicktipp_predictor.evaluate_season_entry import run as season_eval
        season_eval(dynamic=dynamic, retrain_every=retrain_every)
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
    from kicktipp_predictor.evaluate import compute_points  # reuse simple points
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
    from kicktipp_predictor.metrics import save_json
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



