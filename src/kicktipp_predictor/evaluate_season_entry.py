from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor
from kicktipp_predictor.models.performance_tracker import PerformanceTracker
from kicktipp_predictor.models.confidence_selector import extract_display_confidence
from kicktipp_predictor.metrics import (
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
    data_loader = DataLoader()
    predictor = MatchPredictor()
    tracker = PerformanceTracker(storage_dir="data/predictions_season_eval")

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

    all_predictions = []

    # Get historical data for feature context (finished-only; augment with previous season if sparse)
    print("Fetching historical data for context...")
    historical_matches_all = data_loader.fetch_season_matches(current_season)
    historical_matches = [m for m in historical_matches_all if m.get('is_finished')]
    if len(historical_matches) < 50:
        print("Fetching additional finished matches from previous season...")
        prev_season_matches = data_loader.fetch_season_matches(current_season - 1)
        historical_matches.extend([m for m in prev_season_matches if m.get('is_finished')])

    for matchday in range(first_matchday, last_matchday + 1):
        print(f"--- Processing Matchday {matchday} ---")

        # Get matches for the current matchday
        matchday_matches = [m for m in finished_matches if m['matchday'] == matchday]
        if not matchday_matches:
            print(f"No finished matches for matchday {matchday}.")
            continue

        # Create features
        features_df = data_loader.create_prediction_features(
            matchday_matches, historical_matches
        )

        if features_df.empty:
            print(f"Could not generate features for matchday {matchday}.")
            continue

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
    features_all = data_loader.create_prediction_features(
        finished_matches, historical_matches
    )
    if features_all is None or len(features_all) == 0:
        print("\nUnable to build features for season-level analysis.")
        return

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

    # Probability matrix
    P = _proba_from_preds(preds_all)

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
    def _confidence_bundle(P: np.ndarray) -> tuple:
        max_prob = np.max(P, axis=1)
        sorted_probs = np.sort(P, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(P * np.log(P), axis=1)
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
    # Include teams, matchday, predicted/actual labels, probabilities, points
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    debug_rows = []
    for i in range(len(features_all)):
        actual_lab = y_true[i]
        true_idx = mapping.get(actual_lab, -1)
        p_true = float(P[i, true_idx]) if true_idx >= 0 else float('nan')
        debug_rows.append({
            'match_id': int(features_all.iloc[i]['match_id']) if 'match_id' in features_all.columns else None,
            'matchday': int(features_all.iloc[i]['matchday']) if 'matchday' in features_all.columns else None,
            'home_team': features_all.iloc[i]['home_team'] if 'home_team' in features_all.columns else None,
            'away_team': features_all.iloc[i]['away_team'] if 'away_team' in features_all.columns else None,
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


