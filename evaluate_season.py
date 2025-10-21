#!/usr/bin/env python3
"""
Script to evaluate predictor performance for the entire current season.
"""

import sys
import argparse
from collections import defaultdict, Counter
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor
from src.models.performance_tracker import PerformanceTracker

def main():
    """
    Main function to run the season evaluation.
    """
    print("="*80)
    print("SEASON PERFORMANCE EVALUATION")
    print("="*80)
    print()

    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()
    # Using a temporary tracker to not interfere with recorded predictions
    tracker = PerformanceTracker(storage_dir="data/predictions_season_eval")

    # CLI args
    parser = argparse.ArgumentParser(description="Evaluate current season performance")
    parser.add_argument("--strategy", choices=["base", "optimized", "both"], default="both",
                        help="Prediction strategy: base = direct hybrid, optimized = grid argmax of expected points, both = compare")
    args, _ = parser.parse_known_args()

    # Load trained models
    print("Loading models...")
    if not predictor.load_models("hybrid"):
        print("\nERROR: No trained models found.")
        print("Please run train_model.py first to train the models.")
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

    # Fit goal temperature once on prior finished matches (train Poisson first)
    try:
        import pandas as pd
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
        import pandas as pd
        md_min_date = min(m['date'] for m in matchday_matches)
        hist_prior = [m for m in historical_matches if m['is_finished'] and m.get('date') is not None and m['date'] < md_min_date]
        hist_df = pd.DataFrame(hist_prior)
        predictor.poisson_predictor.train(hist_df)

        # Generate predictions according to strategy
        if args.strategy == "base":
            predictions = predictor.predict(features_df)
            predictions_alt = []
        elif args.strategy == "optimized":
            predictions = predictor.predict_optimized(features_df)
            predictions_alt = []
        else:
            predictions = predictor.predict(features_df)
            predictions_alt = predictor.predict_optimized(features_df)

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
            pred['strategy'] = 'base'
            all_predictions.append(pred)

        # If comparing, also accumulate optimized predictions with a separate tag
        if predictions_alt:
            for pred, actual in zip(predictions_alt, matchday_matches):
                points = tracker._calculate_points(
                    pred['predicted_home_score'], pred['predicted_away_score'],
                    actual['home_score'], actual['away_score']
                )
                pred['actual_home_score'] = actual['home_score']
                pred['actual_away_score'] = actual['away_score']
                pred['points_earned'] = points
                pred['is_evaluated'] = True
                pred['matchday'] = matchday
                pred['strategy'] = 'optimized'
                all_predictions.append(pred)

        if args.strategy == "both":
            print(f"Generated and evaluated {len(predictions)} base + {len(predictions_alt)} optimized predictions for matchday {matchday}")
        else:
            print(f"Generated and evaluated {len(predictions)} predictions for matchday {matchday}")

    if not all_predictions:
        print("\nNo predictions could be generated for the season.")
        sys.exit(0)

    # --- AGGREGATE AND PRINT COMPREHENSIVE REPORT ---

    # Overall metrics
    total_matches = len(all_predictions)
    total_points = sum(p['points_earned'] for p in all_predictions)
    avg_points = total_points / total_matches if total_matches > 0 else 0

    # Detailed metrics calculation
    metrics = {
        'by_outcome': defaultdict(lambda: {'total': 0, 'correct': 0, 'points': 0}),
        'by_confidence': defaultdict(lambda: {'total': 0, 'points': 0}),
        'score_predictions': Counter(),
        'score_actuals': Counter(),
        'matchday_stats': defaultdict(lambda: {'points': 0, 'matches': 0})
    }
    points_dist = defaultdict(int)

    for pred in all_predictions:
        points_dist[pred['points_earned']] += 1

        ph, pa = pred['predicted_home_score'], pred['predicted_away_score']
        ah, aa = pred['actual_home_score'], pred['actual_away_score']

        # Outcome
        actual_outcome = 'D' if ah == aa else ('H' if ah > aa else 'A')
        pred_outcome = 'D' if ph == pa else ('H' if ph > pa else 'A')

        metrics['by_outcome'][actual_outcome]['total'] += 1
        metrics['by_outcome'][actual_outcome]['points'] += pred['points_earned']
        if pred_outcome == actual_outcome:
            metrics['by_outcome'][actual_outcome]['correct'] += 1

        # Confidence
        confidence = pred.get('confidence', 0)
        conf_bucket = 'low' if confidence < 0.4 else ('medium' if confidence < 0.6 else 'high')
        metrics['by_confidence'][conf_bucket]['total'] += 1
        metrics['by_confidence'][conf_bucket]['points'] += pred['points_earned']

        # Scores
        metrics['score_predictions'][f"{ph}-{pa}"] += 1
        metrics['score_actuals'][f"{ah}-{aa}"] += 1

        # Matchday
        metrics['matchday_stats'][pred['matchday']]['points'] += pred['points_earned']
        metrics['matchday_stats'][pred['matchday']]['matches'] += 1

    # Print Report
    # Calibration and debug for season
    import numpy as np, csv, os, json
    def outcome_idx(h, a): return 0 if h > a else (1 if h == a else 2)
    probs = [[p['home_win_probability'], p['draw_probability'], p['away_win_probability']] for p in all_predictions]
    actual_idx = [outcome_idx(p['actual_home_score'], p['actual_away_score']) for p in all_predictions]
    brier = float(np.mean([np.sum((np.eye(3)[ai] - np.array(pr))**2) for pr, ai in zip(probs, actual_idx)]))
    print("\n--- CALIBRATION (Season) ---")
    print(f"Brier score: {brier:.3f}")

    # Reliability (3 bins)
    def reliability(scores):
        arr = np.array(scores)
        if len(arr)==0: return []
        qs = np.quantile(arr[:,0], [0, 1/3, 2/3, 1])
        rows=[]
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (arr[:,0]>=lo) & (arr[:,0]<=hi)
            if m.sum()==0: continue
            rows.append((arr[m,0].mean(), arr[m,1].mean(), int(m.sum())))
        return rows
    H_rel = reliability([(pr[0], 1 if ai==0 else 0) for pr, ai in zip(probs, actual_idx)])
    D_rel = reliability([(pr[1], 1 if ai==1 else 0) for pr, ai in zip(probs, actual_idx)])
    A_rel = reliability([(pr[2], 1 if ai==2 else 0) for pr, ai in zip(probs, actual_idx)])
    print("H reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in H_rel]))
    print("D reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in D_rel]))
    print("A reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in A_rel]))

    # Per-match debug
    dbg_path = os.path.join('data','predictions','debug_season.csv')
    os.makedirs(os.path.dirname(dbg_path), exist_ok=True)
    with open(dbg_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['matchday','strategy','home','away','pred_h','pred_a','act_h','act_a','lambda_h','lambda_a','pH','pD','pA','confidence','margin','entropy_confidence','points'])
        for p in all_predictions:
            # derive margin/entropy if not present
            pH = float(p['home_win_probability'])
            pD = float(p['draw_probability'])
            pA = float(p['away_win_probability'])
            probs_sorted = sorted([pH, pD, pA], reverse=True)
            margin = float(p.get('margin', probs_sorted[0] - probs_sorted[1]))
            import math
            import numpy as _np
            probs = _np.array([pH, pD, pA], dtype=float)
            probs = _np.clip(probs, 1e-12, 1.0)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            entropy = float(-_np.sum(probs * _np.log(probs)))
            ent_conf = float(p.get('entropy_confidence', 1.0 - (entropy / math.log(3))))
            w.writerow([p['matchday'], p.get('strategy','base'), p['home_team'], p['away_team'],
                        p['predicted_home_score'], p['predicted_away_score'],
                        p['actual_home_score'], p['actual_away_score'],
                        f"{p['home_expected_goals']:.3f}", f"{p['away_expected_goals']:.3f}",
                        f"{pH:.3f}", f"{pD:.3f}", f"{pA:.3f}",
                        f"{p.get('confidence',0):.3f}", f"{margin:.3f}", f"{ent_conf:.3f}", p['points_earned']])
    print(f"Wrote per-match debug: {dbg_path}")

    # Persist run meta
    meta_path = os.path.join('data','predictions','run_meta.json')
    meta = {
        'script': 'evaluate_season.py',
        'ml_weight': predictor.ml_weight,
        'poisson_weight': predictor.poisson_weight,
        'prob_blend_alpha': predictor.prob_blend_alpha,
        'min_lambda': predictor.min_lambda,
        'goal_temperature': predictor.goal_temperature,
        'confidence_threshold': predictor.confidence_threshold,
        'max_goals': getattr(predictor, 'max_goals', 8)
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote run meta: {meta_path}")

    print("\n--- OVERALL PERFORMANCE ---")
    # Per-strategy splits (useful when --strategy both)
    base_preds = [p for p in all_predictions if p.get('strategy', 'base') == 'base']
    opt_preds = [p for p in all_predictions if p.get('strategy') == 'optimized']
    print(f"Evaluated Matchdays: {first_matchday} - {last_matchday}")
    print(f"Total Matches Evaluated: {total_matches}")
    print(f"Total Points Earned: {total_points}")
    print(f"Average Points per Match: {avg_points:.3f}")
    print(f"Projected Season Total (38 matchdays): {avg_points * 38:.1f} points")
    print(f"Point Efficiency: {(total_points / (total_matches * 4)) * 100:.1f}% (of maximum possible)")
    if base_preds and opt_preds:
        bp = sum(p['points_earned'] for p in base_preds)
        op = sum(p['points_earned'] for p in opt_preds)
        print("\nPer-strategy totals:")
        print(f"  Base:      matches={len(base_preds):3d}, points={bp:3d}, avg={bp/len(base_preds):.3f}")
        print(f"  Optimized: matches={len(opt_preds):3d}, points={op:3d}, avg={op/len(opt_preds):.3f}")

    print("\n--- ACCURACY BREAKDOWN ---")
    exact_scores = points_dist[4]
    correct_diffs = points_dist[3]
    correct_results = points_dist[2]
    incorrect = points_dist[0]

    print(f"Exact Scores (4pts):       {exact_scores:4d} ({exact_scores/total_matches*100:5.1f}%)")
    print(f"Correct Differences (3pts): {correct_diffs:4d} ({correct_diffs/total_matches*100:5.1f}%)")
    print(f"Correct Results (2pts):    {correct_results:4d} ({correct_results/total_matches*100:5.1f}%)")
    print(f"Incorrect (0pts):          {incorrect:4d} ({incorrect/total_matches*100:5.1f}%)")

    print("\n--- PERFORMANCE BY ACTUAL OUTCOME ---")
    for outcome, data in sorted(metrics['by_outcome'].items()):
        name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        avg_pts_outcome = data['points'] / data['total'] if data['total'] > 0 else 0
        print(f"{name:10s}: {data['total']:3d} matches, {data['correct']:3d} correct ({acc:5.1f}%), avg {avg_pts_outcome:.2f} pts")

    print("\n--- PERFORMANCE BY CONFIDENCE ---")
    for conf, data in sorted(metrics['by_confidence'].items()):
        avg_pts_conf = data['points'] / data['total'] if data['total'] > 0 else 0
        print(f"{conf.capitalize():8s}: {data['total']:3d} matches, avg {avg_pts_conf:.2f} pts")

    print("\n--- PERFORMANCE BY MATCHDAY ---")
    if base_preds and opt_preds:
        from collections import defaultdict as _dd
        md_base = _dd(lambda: {'points': 0, 'matches': 0})
        md_opt = _dd(lambda: {'points': 0, 'matches': 0})
        for p in base_preds:
            md_base[p['matchday']]['points'] += p['points_earned']
            md_base[p['matchday']]['matches'] += 1
        for p in opt_preds:
            md_opt[p['matchday']]['points'] += p['points_earned']
            md_opt[p['matchday']]['matches'] += 1

        print("(Base)")
        print(f"{'Matchday':<10} {'Matches':<10} {'Points':<10} {'Avg Pts':<10}")
        print("-" * 42)
        for md, data in sorted(md_base.items()):
            avg_pts_md = data['points'] / data['matches'] if data['matches'] > 0 else 0
            print(f"{md:<10} {data['matches']:<10} {data['points']:<10} {avg_pts_md:<10.2f}")

        print("\n(Optimized)")
        print(f"{'Matchday':<10} {'Matches':<10} {'Points':<10} {'Avg Pts':<10}")
        print("-" * 42)
        for md, data in sorted(md_opt.items()):
            avg_pts_md = data['points'] / data['matches'] if data['matches'] > 0 else 0
            print(f"{md:<10} {data['matches']:<10} {data['points']:<10} {avg_pts_md:<10.2f}")
    else:
        print(f"{'Matchday':<10} {'Matches':<10} {'Points':<10} {'Avg Pts':<10}")
        print("-" * 42)
        for md, data in sorted(metrics['matchday_stats'].items()):
            avg_pts_md = data['points'] / data['matches'] if data['matches'] > 0 else 0
            print(f"{md:<10} {data['matches']:<10} {data['points']:<10} {avg_pts_md:<10.2f}")

    print("\n--- TOP 5 PREDICTED SCORES ---")
    for score, count in metrics['score_predictions'].most_common(5):
        print(f"{score:6s}: {count:3d} times ({count/total_matches*100:5.1f}%)")

    print("\n--- TOP 5 ACTUAL SCORES ---")
    for score, count in metrics['score_actuals'].most_common(5):
        print(f"{score:6s}: {count:3d} times ({count/total_matches*100:5.1f}%)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
