from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
from kicktipp_predictor.models.hybrid_predictor import HybridPredictor
from kicktipp_predictor.models.performance_tracker import PerformanceTracker
from kicktipp_predictor.models.confidence_selector import extract_display_confidence
from kicktipp_predictor.models.metrics import (
    ensure_dir,
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
    save_json,
)


def run() -> None:
    import sys
    from collections import defaultdict, Counter
    import pandas as pd
    import numpy as np
    import os

    print("="*80)
    print("SEASON PERFORMANCE EVALUATION")
    print("="*80)
    print()

    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()
    tracker = PerformanceTracker(storage_dir="data/predictions_season_eval")

    # Load trained models
    print("Loading models...")
    if not predictor.load_models("hybrid"):
        print("\nERROR: No trained models found.")
        print("Please run training first to train the models.")
        sys.exit(1)
    print("Models loaded successfully!\n")

    # Get current season data
    current_season = data_fetcher.get_current_season()
    print(f"Fetching data for current season: {current_season}")
    season_matches = data_fetcher.fetch_season_matches(current_season)

    finished_matches = [m for m in season_matches if m['is_finished']]
    if not finished_matches:
        print("No finished matches found for the current season.")
        sys.exit(0)

    # Determine matchdays to evaluate
    first_matchday = min(m['matchday'] for m in finished_matches)
    last_matchday = max(m['matchday'] for m in finished_matches)
    print(f"Evaluating matchdays from {first_matchday} to {last_matchday}\n")

    all_predictions = []

    # Get historical data for feature context (finished-only; augment with previous season if sparse)
    print("Fetching historical data for context...")
    historical_matches_all = data_fetcher.fetch_season_matches(current_season)
    historical_matches = [m for m in historical_matches_all if m.get('is_finished')]
    if len(historical_matches) < 50:
        print("Fetching additional finished matches from previous season...")
        prev_season_matches = data_fetcher.fetch_season_matches(current_season - 1)
        historical_matches.extend([m for m in prev_season_matches if m.get('is_finished')])

    # Fit goal temperatures once on prior finished matches (train Poisson first)
    try:
        hist_df_for_temp = pd.DataFrame([m for m in historical_matches if m.get('date') is not None])
        if not hist_df_for_temp.empty:
            predictor.poisson_predictor.train(hist_df_for_temp)
            if hasattr(predictor, 'fit_goal_temperature'):
                predictor.fit_goal_temperature(hist_df_for_temp)
    except Exception:
        pass

    for matchday in range(first_matchday, last_matchday + 1):
        print(f"--- Processing Matchday {matchday} ---")

        # Get matches for the current matchday
        matchday_matches = [m for m in finished_matches if m['matchday'] == matchday]
        if not matchday_matches:
            print(f"No finished matches for matchday {matchday}.")
            continue

        # Create features
        features_df = feature_engineer.create_prediction_features(
            matchday_matches, historical_matches
        )

        if features_df.empty:
            print(f"Could not generate features for matchday {matchday}.")
            continue

        # Train Poisson on finished historical matches strictly before this matchday's earliest match
        md_min_date = min(m['date'] for m in matchday_matches)
        hist_prior = [m for m in historical_matches if m['is_finished'] and m.get('date') is not None and m['date'] < md_min_date]
        hist_df = pd.DataFrame(hist_prior)
        predictor.poisson_predictor.train(hist_df)

        # Generate predictions
        predictions = predictor.predict(features_df)

        # Record predictions and update results immediately
        for pred, actual in zip(predictions, matchday_matches):
            points = tracker._calculate_points(
                pred['predicted_home_score'], pred['predicted_away_score'],
                actual['home_score'], actual['away_score']
            )
            pred['actual_home_score'] = actual['home_score']
            pred['actual_away_score'] = actual['away_score']
            pred['points_earned'] = points
            pred['is_evaluated'] = True
            pred['matchday'] = matchday
            all_predictions.append(pred)

        print(f"Generated and evaluated {len(predictions)} predictions for matchday {matchday}")

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
    # Build a unified features dataframe for all finished matches
    season_df = pd.DataFrame(finished_matches)
    features_all = feature_engineer.create_prediction_features(
        finished_matches, historical_matches
    )
    if features_all is None or len(features_all) == 0:
        print("\nUnable to build features for season-level analysis.")
        return

    # Train Poisson on available historical finished matches (for season-level baselines)
    try:
        hist_df_all = pd.DataFrame([m for m in historical_matches if m.get('date') is not None])
        if not hist_df_all.empty:
            predictor.poisson_predictor.train(hist_df_all)
    except Exception:
        pass

    # Helper builders
    def _actual_labels(df) -> list:
        labels = []
        for _, r in df.iterrows():
            if r['home_score'] > r['away_score']:
                labels.append('H')
            elif r['away_score'] > r['home_score']:
                labels.append('A')
            else:
                labels.append('D')
        return labels

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

    # Generate predictions per variant based on the same match set/order
    print("\nGenerating season-wide predictions (hybrid, ml-only, poisson-only)...")
    hybrid_preds_all = predictor.predict(features_all)
    ml_preds_all = predictor.ml_predictor.predict(features_all)
    matches_all = [(row['home_team'], row['away_team']) for _, row in features_all.iterrows()]
    poisson_preds_all = predictor.poisson_predictor.predict_batch(matches_all)
    # Attach identifiers to poisson preds to align
    for i, p in enumerate(poisson_preds_all):
        p['match_id'] = int(features_all.iloc[i]['match_id']) if 'match_id' in features_all.columns else None
        p['home_team'] = features_all.iloc[i]['home_team']
        p['away_team'] = features_all.iloc[i]['away_team']

    # Ground truth aligned to features_all order
    id_to_actual = {m['match_id']: (m['home_score'], m['away_score']) for m in finished_matches if m.get('match_id') is not None}
    actual_home_list = []
    actual_away_list = []
    y_true = []
    for _, row in features_all.iterrows():
        mid = int(row['match_id']) if 'match_id' in features_all.columns else None
        if mid is None or mid not in id_to_actual:
            # Skip if actual not found (shouldn't happen); keep arrays aligned by adding NaNs placeholder
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

    # Probability matrices
    P_h = _proba_from_preds(hybrid_preds_all)
    P_m = _proba_from_preds(ml_preds_all)
    P_p = _proba_from_preds(poisson_preds_all)

    # Label distribution and sanity
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTIONS (Season)")
    print("-"*80)
    lab_counts = {lab: y_true.count(lab) for lab in LABELS_ORDER}
    total_labels = sum(lab_counts.values()) or 1
    print("Actual              : " + "  ".join([f"{lab}={lab_counts.get(lab,0)} ({lab_counts.get(lab,0)/total_labels:.2%})" for lab in LABELS_ORDER]))
    pred_h_labels = [LABELS_ORDER[i] for i in np.argmax(P_h, axis=1)]
    pred_counts_h = {lab: pred_h_labels.count(lab) for lab in LABELS_ORDER}
    print("Hybrid predicted    : " + "  ".join([f"{lab}={pred_counts_h.get(lab,0)} ({pred_counts_h.get(lab,0)/max(1,len(pred_h_labels)):.2%})" for lab in LABELS_ORDER]))
    pred_m_labels = [LABELS_ORDER[i] for i in np.argmax(P_m, axis=1)]
    pred_counts_m = {lab: pred_m_labels.count(lab) for lab in LABELS_ORDER}
    print("ML predicted        : " + "  ".join([f"{lab}={pred_counts_m.get(lab,0)} ({pred_counts_m.get(lab,0)/max(1,len(pred_m_labels)):.2%})" for lab in LABELS_ORDER]))
    pred_p_labels = [LABELS_ORDER[i] for i in np.argmax(P_p, axis=1)]
    pred_counts_p = {lab: pred_p_labels.count(lab) for lab in LABELS_ORDER}
    print("Poisson predicted   : " + "  ".join([f"{lab}={pred_counts_p.get(lab,0)} ({pred_counts_p.get(lab,0)/max(1,len(pred_p_labels)):.2%})" for lab in LABELS_ORDER]))

    # Scores and points per variant
    ph_h, pa_h = _scores_from_preds(hybrid_preds_all)
    ph_m, pa_m = _scores_from_preds(ml_preds_all)
    ph_p, pa_p = _scores_from_preds(poisson_preds_all)
    pts_h = compute_points(ph_h, pa_h, ah, aa)
    pts_m = compute_points(ph_m, pa_m, ah, aa)
    pts_p = compute_points(ph_p, pa_p, ah, aa)

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

    metrics_h = _all_metrics(P_h, pts_h)
    metrics_m = _all_metrics(P_m, pts_m)
    metrics_p = _all_metrics(P_p, pts_p)

    # Metrics table
    print("\n" + "-"*80)
    print("VARIANT METRICS (Season)")
    print("-"*80)
    def _print_metrics_row(name: str, md: dict) -> None:
        print(f"{name:13s} avg_pts={md['avg_points']:.3f}  acc={md['accuracy']:.3f}  "
              f"brier={md['brier']:.4f}  logloss={md['log_loss']:.4f}  rps={md['rps']:.4f}")
    _print_metrics_row('Hybrid', metrics_h)
    _print_metrics_row('ML-only', metrics_m)
    _print_metrics_row('Poisson-only', metrics_p)

    # Points distribution
    print("\nPOINTS DISTRIBUTION (Season - Hybrid)")
    unique_pts = [0, 2, 3, 4]
    pt_counts = {p: int(np.sum(pts_h == p)) for p in unique_pts}
    print("Counts by points    : " + ", ".join([f"{p}p={pt_counts[p]}" for p in unique_pts]))
    print(f"Avg points          : {np.mean(pts_h) if len(pts_h) else 0.0:.3f}  Total: {int(np.sum(pts_h))}")

    # Strategy comparisons
    print("\n" + "-"*80)
    print("STRATEGY METRICS (Season - Scoreline selection)")
    print("-"*80)
    base_points = compute_points(ph_h, pa_h, ah, aa)

    preds_opt = predictor.predict_optimized(features_all, strategy='balanced')
    ph_opt = np.asarray([int(p.get('predicted_home_score', 0)) for p in preds_opt], dtype=int)
    pa_opt = np.asarray([int(p.get('predicted_away_score', 0)) for p in preds_opt], dtype=int)
    pts_opt = compute_points(ph_opt, pa_opt, ah, aa)

    def _aggressive_from_probs(base_preds: list) -> tuple:
        agg_h = []
        agg_a = []
        for p in base_preds:
            h = int(p.get('predicted_home_score', 0))
            a = int(p.get('predicted_away_score', 0))
            ph = float(p.get('home_win_probability', 1/3))
            pd = float(p.get('draw_probability', 1/3))
            pa = float(p.get('away_win_probability', 1/3))
            if ph > pa and ph > pd:
                h = max(0, h + 1)
            elif pa > ph and pa > pd:
                a = max(0, a + 1)
            else:
                if h == a:
                    h, a = 2, 2
            agg_h.append(h)
            agg_a.append(a)
        return np.asarray(agg_h, dtype=int), np.asarray(agg_a, dtype=int)

    ph_agg, pa_agg = _aggressive_from_probs(hybrid_preds_all)
    pts_agg = compute_points(ph_agg, pa_agg, ah, aa)

    def _safe_from_confidence(base_preds: list, thr: float = float(predictor.confidence_threshold)) -> tuple:
        sh = []
        sa = []
        for p in base_preds:
            h = int(p.get('predicted_home_score', 0))
            a = int(p.get('predicted_away_score', 0))
            conf = float(extract_display_confidence(p))
            if conf < thr:
                ph = float(p.get('home_win_probability', 1/3))
                pd = float(p.get('draw_probability', 1/3))
                pa = float(p.get('away_win_probability', 1/3))
                if ph > pa and ph > pd:
                    h, a = 2, 1
                elif pa > ph and pa > pd:
                    h, a = 1, 2
                else:
                    h, a = 1, 1
            sh.append(h)
            sa.append(a)
        return np.asarray(sh, dtype=int), np.asarray(sa, dtype=int)

    ph_safe, pa_safe = _safe_from_confidence(hybrid_preds_all)
    pts_safe = compute_points(ph_safe, pa_safe, ah, aa)

    def _print_strategy(name: str, ph: np.ndarray, pa: np.ndarray, pts: np.ndarray) -> None:
        print(f"{name:13s} avg_pts={float(np.mean(pts)) if len(pts) else 0.0:.3f}  total={int(np.sum(pts)):4d}  "
              f"scores={int(np.sum((ph==ah) & (pa==aa))):3d}  diffs={int(np.sum((ph-pa)==(ah-aa))):3d}  "
              f"results={int(np.sum(((ph>pa)&(ah>aa)) | ((ph==pa)&(ah==aa)) | ((ph<pa)&(ah<aa)))):3d}")

    _print_strategy('Base', ph_h, pa_h, base_points)
    _print_strategy('Optimized', ph_opt, pa_opt, pts_opt)
    _print_strategy('Aggressive', ph_agg, pa_agg, pts_agg)
    _print_strategy('Safe', ph_safe, pa_safe, pts_safe)

    # Confidence buckets (Hybrid)
    def _confidence_bundle(P: np.ndarray) -> tuple:
        max_prob = np.max(P, axis=1)
        sorted_probs = np.sort(P, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(P * np.log(P), axis=1)
        entropy_conf = 1.0 - (entropy / np.log(3))
        combined = 0.6 * max_prob + 0.4 * margin
        return max_prob, margin, entropy_conf, combined

    _, _, _, combined_all = _confidence_bundle(P_h)
    out_dir = os.path.join('data', 'predictions')
    ensure_dir(out_dir)
    conf_df = bin_by_confidence(combined_all, y_true, P_h, pts_h, n_bins=5)
    conf_season_csv = os.path.join(out_dir, 'confidence_buckets_season.csv')
    conf_df.to_csv(conf_season_csv, index=False)
    try:
        plot_confidence_buckets(conf_df, os.path.join(out_dir, 'confidence_buckets_season.png'))
    except Exception:
        pass
    if len(conf_df) > 0:
        print("\nCONFIDENCE BUCKETS (Season - Hybrid)")
        for _, r in conf_df.iterrows():
            print(f"bin={r['bin']:<14} count={int(r['count']):4d}  avg_pts={float(r['avg_points']):.3f}  "
                  f"acc={float(r['accuracy']):.3f}  avg_conf={float(r['avg_confidence']):.3f}")

    # Calibration curves and confusion matrices
    try:
        curve_H = reliability_diagram(y_true, P_h, 'H', n_bins=10)
        curve_D = reliability_diagram(y_true, P_h, 'D', n_bins=10)
        curve_A = reliability_diagram(y_true, P_h, 'A', n_bins=10)
        plot_reliability_curve(curve_H, 'H', os.path.join(out_dir, 'calibration_home_season.png'))
        plot_reliability_curve(curve_D, 'D', os.path.join(out_dir, 'calibration_draw_season.png'))
        plot_reliability_curve(curve_A, 'A', os.path.join(out_dir, 'calibration_away_season.png'))
    except Exception:
        pass

    cm_stats = confusion_matrix_stats(y_true, P_h)
    cm = np.array(cm_stats['matrix'], dtype=int)
    try:
        plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix_season.png'))
    except Exception:
        pass
    print("\nCONFUSION MATRIX (Season - Hybrid)")
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
    season_metrics = {'hybrid': metrics_h, 'ml': metrics_m, 'poisson': metrics_p}
    save_json(season_metrics, os.path.join(out_dir, 'metrics_season.json'))
    with open(os.path.join(out_dir, 'metrics_table_season.txt'), 'w', encoding='utf-8') as f:
        def _line(name: str, md: dict) -> str:
            return (f"{name:8s}  avg_pts={md['avg_points']:.3f}  acc={md['accuracy']:.3f}  "
                    f"brier={md['brier']:.4f}  logloss={md['log_loss']:.4f}  rps={md['rps']:.4f}")
        f.write(_line('hybrid', metrics_h) + "\n")
        f.write(_line('ml', metrics_m) + "\n")
        f.write(_line('poisson', metrics_p) + "\n")

    # Build and save debug CSV for season (Hybrid canonical)
    # Include teams, matchday, predicted/actual labels, probabilities, points
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    debug_rows = []
    for i in range(len(features_all)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true = float(P_h[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows.append({
            'match_id': int(features_all.iloc[i]['match_id']) if 'match_id' in features_all.columns else None,
            'matchday': int(features_all.iloc[i]['matchday']) if 'matchday' in features_all.columns else None,
            'home_team': features_all.iloc[i]['home_team'] if 'home_team' in features_all.columns else None,
            'away_team': features_all.iloc[i]['away_team'] if 'away_team' in features_all.columns else None,
            'actual': actual_lab,
            'pred': LABELS_ORDER[int(np.argmax(P_h[i]))],
            'pH': float(P_h[i, 0]),
            'pD': float(P_h[i, 1]),
            'pA': float(P_h[i, 2]),
            'p_true': p_true,
            'points': int(pts_h[i]),
        })
    pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, 'debug_season.csv'), index=False)
    print(f"\nSeason artifacts written to {out_dir}")


