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
    print("\n" + "-"*80)
    print("DATASET")
    print("-"*80)
    print(f"Seasons range       : {start_season} - {current_season}")
    print(f"Total matches       : {len(all_matches)} (finished: {len(finished)})")
    uniq_teams = sorted({m.get('home_team') for m in all_matches} | {m.get('away_team') for m in all_matches} - {None})
    print(f"Unique teams        : {len(uniq_teams)}")

    # Create features
    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
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

    # Label distribution and quick sanity checks
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTIONS (Test)")
    print("-"*80)
    lab_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    total_labels = sum(lab_counts.values()) or 1
    print("Actual              : " + "  ".join([f"{lab}={lab_counts.get(lab,0)} ({lab_counts.get(lab,0)/total_labels:.2%})" for lab in LABELS_ORDER]))
    pred_h_labels = [LABELS_ORDER[i] for i in np.argmax(P_h, axis=1)]
    pred_counts_h = {lab: pred_h_labels.count(lab) for lab in LABELS_ORDER}
    print("Hybrid predicted    : " + "  ".join([f"{lab}={pred_counts_h.get(lab,0)} ({pred_counts_h.get(lab,0)/max(1,len(pred_h_labels)):.2%})" for lab in LABELS_ORDER]))
    # ML-only
    pred_m_labels = [LABELS_ORDER[i] for i in np.argmax(P_m, axis=1)]
    pred_counts_m = {lab: pred_m_labels.count(lab) for lab in LABELS_ORDER}
    print("ML predicted        : " + "  ".join([f"{lab}={pred_counts_m.get(lab,0)} ({pred_counts_m.get(lab,0)/max(1,len(pred_m_labels)):.2%})" for lab in LABELS_ORDER]))
    # Poisson-only
    pred_p_labels = [LABELS_ORDER[i] for i in np.argmax(P_p, axis=1)]
    pred_counts_p = {lab: pred_p_labels.count(lab) for lab in LABELS_ORDER}
    print("Poisson predicted   : " + "  ".join([f"{lab}={pred_counts_p.get(lab,0)} ({pred_counts_p.get(lab,0)/max(1,len(pred_p_labels)):.2%})" for lab in LABELS_ORDER]))

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

    # Metrics table in console
    print("\n" + "-"*80)
    print("VARIANT METRICS")
    print("-"*80)
    def _print_metrics_row(name: str, md: Dict[str, object]) -> None:
        print(f"{name:13s} avg_pts={md['avg_points']:.3f}  acc={md['accuracy']:.3f}  "
              f"brier={md['brier']:.4f}  logloss={md['log_loss']:.4f}  rps={md['rps']:.4f}")
    _print_metrics_row('Hybrid', metrics_h)
    _print_metrics_row('ML-only', metrics_m)
    _print_metrics_row('Poisson-only', metrics_p)

    # Points distribution (hybrid)
    print("\nPOINTS DISTRIBUTION (Hybrid)")
    unique_pts = [0, 2, 3, 4]
    pt_counts = {p: int(np.sum(pts_h == p)) for p in unique_pts}
    print("Counts by points    : " + ", ".join([f"{p}p={pt_counts[p]}" for p in unique_pts]))
    print(f"Avg points          : {np.mean(pts_h) if len(pts_h) else 0.0:.3f}  Total: {int(np.sum(pts_h))}")
    # Points distribution (ML)
    print("\nPOINTS DISTRIBUTION (ML-only)")
    pt_counts_m = {p: int(np.sum(pts_m == p)) for p in unique_pts}
    print("Counts by points    : " + ", ".join([f"{p}p={pt_counts_m[p]}" for p in unique_pts]))
    print(f"Avg points          : {np.mean(pts_m) if len(pts_m) else 0.0:.3f}  Total: {int(np.sum(pts_m))}")
    # Points distribution (Poisson)
    print("\nPOINTS DISTRIBUTION (Poisson-only)")
    pt_counts_p = {p: int(np.sum(pts_p == p)) for p in unique_pts}
    print("Counts by points    : " + ", ".join([f"{p}p={pt_counts_p[p]}" for p in unique_pts]))
    print(f"Avg points          : {np.mean(pts_p) if len(pts_p) else 0.0:.3f}  Total: {int(np.sum(pts_p))}")

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
    # Additional calibration plots for ML-only and Poisson-only (variant-specific filenames)
    curve_H_m = reliability_diagram(y_true, P_m, 'H', n_bins=10)
    curve_D_m = reliability_diagram(y_true, P_m, 'D', n_bins=10)
    curve_A_m = reliability_diagram(y_true, P_m, 'A', n_bins=10)
    plot_reliability_curve(curve_H_m, 'H', os.path.join(out_dir, 'calibration_home_ml.png'))
    plot_reliability_curve(curve_D_m, 'D', os.path.join(out_dir, 'calibration_draw_ml.png'))
    plot_reliability_curve(curve_A_m, 'A', os.path.join(out_dir, 'calibration_away_ml.png'))
    curve_H_p = reliability_diagram(y_true, P_p, 'H', n_bins=10)
    curve_D_p = reliability_diagram(y_true, P_p, 'D', n_bins=10)
    curve_A_p = reliability_diagram(y_true, P_p, 'A', n_bins=10)
    plot_reliability_curve(curve_H_p, 'H', os.path.join(out_dir, 'calibration_home_poisson.png'))
    plot_reliability_curve(curve_D_p, 'D', os.path.join(out_dir, 'calibration_draw_poisson.png'))
    plot_reliability_curve(curve_A_p, 'A', os.path.join(out_dir, 'calibration_away_poisson.png'))

    # Confusion matrix (hybrid)
    cm_stats = confusion_matrix_stats(y_true, P_h)
    cm = np.array(cm_stats['matrix'], dtype=int)
    plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix.png'))
    # Console view of confusion matrix and per-class metrics
    print("\nCONFUSION MATRIX (Hybrid)")
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

    # Confusion matrix (ML-only)
    cm_stats_m = confusion_matrix_stats(y_true, P_m)
    cm_m = np.array(cm_stats_m['matrix'], dtype=int)
    plot_confusion_matrix(cm_m, os.path.join(out_dir, 'confusion_matrix_ml.png'))
    print("\nCONFUSION MATRIX (ML-only)")
    print(header)
    for i, lab in enumerate(LABELS_ORDER):
        row = cm_m[i]
        print(f"Actual {lab}       : {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    print(f"Overall accuracy   : {cm_stats_m.get('accuracy', float('nan')):.3f}")
    per_class_m = cm_stats_m.get('per_class', {})
    if isinstance(per_class_m, dict):
        for lab in LABELS_ORDER:
            stats = per_class_m.get(lab, {})
            pr = float(stats.get('precision', float('nan')))
            rc = float(stats.get('recall', float('nan')))
            print(f"Class {lab}         : precision={pr:.3f} recall={rc:.3f}")

    # Confusion matrix (Poisson-only)
    cm_stats_p = confusion_matrix_stats(y_true, P_p)
    cm_p = np.array(cm_stats_p['matrix'], dtype=int)
    plot_confusion_matrix(cm_p, os.path.join(out_dir, 'confusion_matrix_poisson.png'))
    print("\nCONFUSION MATRIX (Poisson-only)")
    print(header)
    for i, lab in enumerate(LABELS_ORDER):
        row = cm_p[i]
        print(f"Actual {lab}       : {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    print(f"Overall accuracy   : {cm_stats_p.get('accuracy', float('nan')):.3f}")
    per_class_p = cm_stats_p.get('per_class', {})
    if isinstance(per_class_p, dict):
        for lab in LABELS_ORDER:
            stats = per_class_p.get(lab, {})
            pr = float(stats.get('precision', float('nan')))
            rc = float(stats.get('recall', float('nan')))
            print(f"Class {lab}         : precision={pr:.3f} recall={rc:.3f}")

    # Confidence bucket analysis (hybrid)
    _, _, _, combined_all = _confidence_bundle(P_h)
    conf_df = bin_by_confidence(combined_all, y_true, P_h, pts_h, n_bins=5)
    conf_df.to_csv(os.path.join(out_dir, 'confidence_buckets.csv'), index=False)
    plot_confidence_buckets(conf_df, os.path.join(out_dir, 'confidence_buckets.png'))
    # Console: confidence buckets
    if len(conf_df) > 0:
        print("\nCONFIDENCE BUCKETS (Hybrid)")
        for _, r in conf_df.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # Confidence bucket analysis (ML-only)
    _, _, _, combined_all_m = _confidence_bundle(P_m)
    conf_df_m = bin_by_confidence(combined_all_m, y_true, P_m, pts_m, n_bins=5)
    conf_df_m.to_csv(os.path.join(out_dir, 'confidence_buckets_ml.csv'), index=False)
    plot_confidence_buckets(conf_df_m, os.path.join(out_dir, 'confidence_buckets_ml.png'))
    if len(conf_df_m) > 0:
        print("\nCONFIDENCE BUCKETS (ML-only)")
        for _, r in conf_df_m.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # Confidence bucket analysis (Poisson-only)
    _, _, _, combined_all_p = _confidence_bundle(P_p)
    conf_df_p = bin_by_confidence(combined_all_p, y_true, P_p, pts_p, n_bins=5)
    conf_df_p.to_csv(os.path.join(out_dir, 'confidence_buckets_poisson.csv'), index=False)
    plot_confidence_buckets(conf_df_p, os.path.join(out_dir, 'confidence_buckets_poisson.png'))
    if len(conf_df_p) > 0:
        print("\nCONFIDENCE BUCKETS (Poisson-only)")
        for _, r in conf_df_p.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # SHAP analysis (optional; safe no-op if deps missing)
    try:
        # Use the exact features used for ML models
        ml_X = test_features[predictor.ml_predictor.feature_columns].fillna(0)
        run_shap_for_mlpredictor(predictor.ml_predictor, ml_X)
    except Exception:
        pass

    # Build detailed rows for examples (Hybrid)
    max_prob_h, margin_h, entropy_h, combined_h = _confidence_bundle(P_h)
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    # Extend debug rows to include teams and p_true
    debug_rows_h: List[Dict] = []
    for i in range(len(test_df)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true = float(P_h[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows_h.append({
            'match_id': int(test_df.iloc[i]['match_id']) if 'match_id' in test_df.columns else None,
            'home_team': test_features.iloc[i]['home_team'] if 'home_team' in test_features.columns else None,
            'away_team': test_features.iloc[i]['away_team'] if 'away_team' in test_features.columns else None,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P_h[i]))],
            'pH': float(P_h[i, 0]),
            'pD': float(P_h[i, 1]),
            'pA': float(P_h[i, 2]),
            'p_true': p_true,
            'points': int(pts_h[i]),
            'confidence': float(combined_h[i]),
            'margin': float(margin_h[i]),
            'entropy_conf': float(entropy_h[i]),
        })
    # Save enriched debug for hybrid (canonical filename)
    pd.DataFrame(debug_rows_h).to_csv(os.path.join(out_dir, 'debug_eval.csv'), index=False)

    # Top/bottom cases for Hybrid
    def _safe_key(v: float) -> float:
        try:
            return float(v)
        except Exception:
            return float('nan')
    correct = [r for r in debug_rows_h if r['actual'] == r['pred']]
    wrong = [r for r in debug_rows_h if r['actual'] != r['pred']]
    correct_sorted = sorted(correct, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    wrong_sorted = sorted(wrong, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    worst_ptrue = sorted(debug_rows_h, key=lambda r: _safe_key(r['p_true']))[:5]

    print("\n" + "-"*80)
    print("EXAMPLES (Hybrid)")
    print("-"*80)
    if correct_sorted:
        print("Top 5 confident CORRECT predictions:")
        for r in correct_sorted:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  points={r['points']}")
    if wrong_sorted:
        print("\nTop 5 confident WRONG predictions:")
        for r in wrong_sorted:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  p_true={r['p_true']:.3f}")
    if worst_ptrue:
        print("\nLowest probability assigned to TRUE outcome (worst log-loss cases):")
        for r in worst_ptrue:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  act={r['actual']}  p_true={r['p_true']:.3f}  pred={r['pred']}  conf={r['confidence']:.3f}")

    # Build detailed rows and examples for ML-only
    max_prob_m, margin_m, entropy_m, combined_m = _confidence_bundle(P_m)
    debug_rows_m: List[Dict] = []
    for i in range(len(test_df)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true_m = float(P_m[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows_m.append({
            'match_id': int(test_df.iloc[i]['match_id']) if 'match_id' in test_df.columns else None,
            'home_team': test_features.iloc[i]['home_team'] if 'home_team' in test_features.columns else None,
            'away_team': test_features.iloc[i]['away_team'] if 'away_team' in test_features.columns else None,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P_m[i]))],
            'pH': float(P_m[i, 0]),
            'pD': float(P_m[i, 1]),
            'pA': float(P_m[i, 2]),
            'p_true': p_true_m,
            'points': int(pts_m[i]),
            'confidence': float(combined_m[i]),
            'margin': float(margin_m[i]),
            'entropy_conf': float(entropy_m[i]),
        })
    pd.DataFrame(debug_rows_m).to_csv(os.path.join(out_dir, 'debug_eval_ml.csv'), index=False)

    correct_m = [r for r in debug_rows_m if r['actual'] == r['pred']]
    wrong_m = [r for r in debug_rows_m if r['actual'] != r['pred']]
    correct_sorted_m = sorted(correct_m, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    wrong_sorted_m = sorted(wrong_m, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    worst_ptrue_m = sorted(debug_rows_m, key=lambda r: _safe_key(r['p_true']))[:5]

    print("\n" + "-"*80)
    print("EXAMPLES (ML-only)")
    print("-"*80)
    if correct_sorted_m:
        print("Top 5 confident CORRECT predictions:")
        for r in correct_sorted_m:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  points={r['points']}")
    if wrong_sorted_m:
        print("\nTop 5 confident WRONG predictions:")
        for r in wrong_sorted_m:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  p_true={r['p_true']:.3f}")
    if worst_ptrue_m:
        print("\nLowest probability assigned to TRUE outcome (worst log-loss cases):")
        for r in worst_ptrue_m:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  act={r['actual']}  p_true={r['p_true']:.3f}  pred={r['pred']}  conf={r['confidence']:.3f}")

    # Build detailed rows and examples for Poisson-only
    max_prob_p, margin_p, entropy_p, combined_p = _confidence_bundle(P_p)
    debug_rows_p: List[Dict] = []
    for i in range(len(test_df)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true_p = float(P_p[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows_p.append({
            'match_id': int(test_df.iloc[i]['match_id']) if 'match_id' in test_df.columns else None,
            'home_team': test_features.iloc[i]['home_team'] if 'home_team' in test_features.columns else None,
            'away_team': test_features.iloc[i]['away_team'] if 'away_team' in test_features.columns else None,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P_p[i]))],
            'pH': float(P_p[i, 0]),
            'pD': float(P_p[i, 1]),
            'pA': float(P_p[i, 2]),
            'p_true': p_true_p,
            'points': int(pts_p[i]),
            'confidence': float(combined_p[i]),
            'margin': float(margin_p[i]),
            'entropy_conf': float(entropy_p[i]),
        })
    pd.DataFrame(debug_rows_p).to_csv(os.path.join(out_dir, 'debug_eval_poisson.csv'), index=False)

    correct_p = [r for r in debug_rows_p if r['actual'] == r['pred']]
    wrong_p = [r for r in debug_rows_p if r['actual'] != r['pred']]
    correct_sorted_p = sorted(correct_p, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    wrong_sorted_p = sorted(wrong_p, key=lambda r: _safe_key(r['confidence']), reverse=True)[:5]
    worst_ptrue_p = sorted(debug_rows_p, key=lambda r: _safe_key(r['p_true']))[:5]

    print("\n" + "-"*80)
    print("EXAMPLES (Poisson-only)")
    print("-"*80)
    if correct_sorted_p:
        print("Top 5 confident CORRECT predictions:")
        for r in correct_sorted_p:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  points={r['points']}")
    if wrong_sorted_p:
        print("\nTop 5 confident WRONG predictions:")
        for r in wrong_sorted_p:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  pred={r['pred']} act={r['actual']}  conf={r['confidence']:.3f}  p_true={r['p_true']:.3f}")
    if worst_ptrue_p:
        print("\nLowest probability assigned to TRUE outcome (worst log-loss cases):")
        for r in worst_ptrue_p:
            teams = f"{r.get('home_team','?')} - {r.get('away_team','?')}"
            print(f"  {teams:30s}  act={r['actual']}  p_true={r['p_true']:.3f}  pred={r['pred']}  conf={r['confidence']:.3f}")

    # Calibration (ECE) snapshot for hybrid
    ece_h = metrics_h.get('ece')
    if isinstance(ece_h, dict):
        print("\nECE by class (Hybrid): " + ", ".join([f"{lab}={float(ece_h.get(lab, float('nan'))):.4f}" for lab in LABELS_ORDER]))
    # ECE for ML-only
    ece_m = metrics_m.get('ece')
    if isinstance(ece_m, dict):
        print("ECE by class (ML-only): " + ", ".join([f"{lab}={float(ece_m.get(lab, float('nan'))):.4f}" for lab in LABELS_ORDER]))
    # ECE for Poisson-only
    ece_p = metrics_p.get('ece')
    if isinstance(ece_p, dict):
        print("ECE by class (Poisson-only): " + ", ".join([f"{lab}={float(ece_p.get(lab, float('nan'))):.4f}" for lab in LABELS_ORDER]))

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    _print_metrics_row('Hybrid', metrics_h)
    _print_metrics_row('ML-only', metrics_m)
    _print_metrics_row('Poisson-only', metrics_p)
    print(f"Artifacts written to {out_dir}")


