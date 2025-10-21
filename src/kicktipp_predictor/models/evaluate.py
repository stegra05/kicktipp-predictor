from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
from kicktipp_predictor.models.hybrid_predictor import HybridPredictor
from kicktipp_predictor.models.metrics import (
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
    compute_points,
    save_json,
)
from kicktipp_predictor.models.confidence_selector import extract_display_confidence
from kicktipp_predictor.models.shap_analysis import run_shap_for_mlpredictor

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple


def run_evaluation(season: bool = False) -> None:
    if season:
        from kicktipp_predictor.evaluate_season_entry import run as season_eval
        season_eval()
        return

    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()

    # Load trained models
    if not predictor.load_models("hybrid"):
        print("ERROR: No trained models found. Run training first.")
        return

    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2

    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    finished = [m for m in all_matches if m['is_finished']]
    print(f"Loaded {len(finished)} finished matches")

    # Create features
    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} samples")

    # Use last 30% as test set
    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]
    print(f"Using {len(test_df)} matches for evaluation")

    # Prepare test features (without targets)
    test_features = test_df.drop(
        columns=['home_score', 'away_score', 'goal_difference', 'result'],
        errors='ignore'
    )

    # Train Poisson only on matches strictly before the test set's first date
    id_to_date = {m['match_id']: m['date'] for m in all_matches if m.get('match_id') is not None and m.get('date') is not None}
    test_ids = list(test_df['match_id']) if 'match_id' in test_df.columns else []
    test_dates = [id_to_date.get(mid) for mid in test_ids]
    test_dates = [d for d in test_dates if d is not None]
    if test_dates:
        first_test_date = min(test_dates)
        hist_prior = [m for m in finished if m.get('date') is not None and m['date'] < first_test_date]
    else:
        hist_prior = finished
    import pandas as pd
    hist_df = pd.DataFrame(hist_prior)
    predictor.poisson_predictor.train(hist_df)

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
        # numeric safety
        arr = np.clip(arr, 1e-15, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr

    def _scores_from_preds(preds: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        ph = np.asarray([int(p.get('predicted_home_score', 0)) for p in preds], dtype=int)
        pa = np.asarray([int(p.get('predicted_away_score', 0)) for p in preds], dtype=int)
        return ph, pa

    def _confidence_bundle(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # max prob, margin (top minus second), entropy_confidence, combined
        max_prob = np.max(P, axis=1)
        # top two
        sorted_probs = np.sort(P, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        # entropy based (normalized)
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(P * np.log(P), axis=1)
        entropy_conf = 1.0 - (entropy / np.log(3))
        combined = 0.6 * max_prob + 0.4 * margin
        return max_prob, margin, entropy_conf, combined

    # Generate predictions for three variants
    print("\nGenerating predictions (hybrid, ml-only, poisson-only)...")
    hybrid_preds = predictor.predict(test_features)
    ml_preds = predictor.ml_predictor.predict(test_features)
    # Poisson-only batch
    matches = [(row['home_team'], row['away_team']) for _, row in test_features.iterrows()]
    poisson_preds = predictor.poisson_predictor.predict_batch(matches)
    # Attach identifiers to poisson preds
    for i, p in enumerate(poisson_preds):
        p['match_id'] = int(test_features.iloc[i]['match_id']) if 'match_id' in test_features.columns else None
        p['home_team'] = test_features.iloc[i]['home_team']
        p['away_team'] = test_features.iloc[i]['away_team']

    # Ground truth
    y_true = _actual_labels(test_df)
    ah = np.asarray(test_df['home_score'], dtype=int)
    aa = np.asarray(test_df['away_score'], dtype=int)

    # Build proba matrices
    P_h = _proba_from_preds(hybrid_preds)
    P_m = _proba_from_preds(ml_preds)
    P_p = _proba_from_preds(poisson_preds)

    # Predicted scores
    ph_h, pa_h = _scores_from_preds(hybrid_preds)
    ph_m, pa_m = _scores_from_preds(ml_preds)
    ph_p, pa_p = _scores_from_preds(poisson_preds)

    # Points per match
    pts_h = compute_points(ph_h, pa_h, ah, aa)
    pts_m = compute_points(ph_m, pa_m, ah, aa)
    pts_p = compute_points(ph_p, pa_p, ah, aa)

    # Metrics per variant
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

    metrics_h = _all_metrics(P_h, pts_h)
    metrics_m = _all_metrics(P_m, pts_m)
    metrics_p = _all_metrics(P_p, pts_p)

    # Save artifacts
    out_dir = os.path.join('data', 'predictions')
    ensure_dir(out_dir)

    # metrics.json with all variants
    save_json({'hybrid': metrics_h, 'ml': metrics_m, 'poisson': metrics_p}, os.path.join(out_dir, 'metrics.json'))

    # metrics_table.txt concise
    with open(os.path.join(out_dir, 'metrics_table.txt'), 'w', encoding='utf-8') as f:
        def _line(name: str, md: Dict[str, object]) -> str:
            return (f"{name:8s}  avg_pts={md['avg_points']:.3f}  acc={md['accuracy']:.3f}  "
                    f"brier={md['brier']:.4f}  logloss={md['log_loss']:.4f}  rps={md['rps']:.4f}")
        f.write(_line('hybrid', metrics_h) + "\n")
        f.write(_line('ml', metrics_m) + "\n")
        f.write(_line('poisson', metrics_p) + "\n")

    # Debug CSV (for hybrid, drives confidence selector)
    max_prob_h, margin_h, entropy_h, combined_h = _confidence_bundle(P_h)
    debug_rows: List[Dict] = []
    for i in range(len(test_df)):
        debug_rows.append({
            'match_id': int(test_df.iloc[i]['match_id']) if 'match_id' in test_df.columns else None,
            'actual': y_true[i],
            'pred': LABELS_ORDER[int(np.argmax(P_h[i]))],
            'pH': float(P_h[i, 0]),
            'pD': float(P_h[i, 1]),
            'pA': float(P_h[i, 2]),
            'points': int(pts_h[i]),
            'confidence': float(combined_h[i]),
            'margin': float(margin_h[i]),
            'entropy_conf': float(entropy_h[i]),
        })
    pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, 'debug_eval.csv'), index=False)

    # Calibration plots (hybrid only for canonical filenames)
    curve_H = reliability_diagram(y_true, P_h, 'H', n_bins=10)
    curve_D = reliability_diagram(y_true, P_h, 'D', n_bins=10)
    curve_A = reliability_diagram(y_true, P_h, 'A', n_bins=10)
    plot_reliability_curve(curve_H, 'H', os.path.join(out_dir, 'calibration_home.png'))
    plot_reliability_curve(curve_D, 'D', os.path.join(out_dir, 'calibration_draw.png'))
    plot_reliability_curve(curve_A, 'A', os.path.join(out_dir, 'calibration_away.png'))

    # Confusion matrix (hybrid)
    cm_stats = confusion_matrix_stats(y_true, P_h)
    cm = np.array(cm_stats['matrix'], dtype=int)
    plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix.png'))

    # Confidence bucket analysis (hybrid)
    _, _, _, combined_all = _confidence_bundle(P_h)
    conf_df = bin_by_confidence(combined_all, y_true, P_h, pts_h, n_bins=5)
    conf_df.to_csv(os.path.join(out_dir, 'confidence_buckets.csv'), index=False)
    plot_confidence_buckets(conf_df, os.path.join(out_dir, 'confidence_buckets.png'))

    # SHAP analysis (optional; safe no-op if deps missing)
    try:
        # Use the exact features used for ML models
        ml_X = test_features[predictor.ml_predictor.feature_columns].fillna(0)
        run_shap_for_mlpredictor(predictor.ml_predictor, ml_X)
    except Exception:
        pass

    # Print concise summary
    print("\n=== Evaluation Summary ===")
    for name, md in [('Hybrid', metrics_h), ('ML-only', metrics_m), ('Poisson-only', metrics_p)]:
        print(f"{name:13s} avg_pts={md['avg_points']:.3f} acc={md['accuracy']:.3f} "
              f"brier={md['brier']:.4f} logloss={md['log_loss']:.4f} rps={md['rps']:.4f}")
    print(f"Artifacts written to {out_dir}")


