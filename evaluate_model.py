#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Provides detailed metrics, confusion matrices, and performance analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor


def calculate_detailed_metrics(predictions, actuals):
    """Calculate comprehensive evaluation metrics."""

    metrics = {
        'total_matches': len(predictions),
        'exact_scores': 0,
        'correct_differences': 0,
        'correct_results': 0,
        'incorrect': 0,
        'total_points': 0,
        'by_outcome': defaultdict(lambda: {'total': 0, 'correct': 0, 'points': 0}),
        'by_confidence': defaultdict(lambda: {'total': 0, 'correct_result': 0, 'points': 0}),
        'score_predictions': defaultdict(int),
        'score_actuals': defaultdict(int),
    }

    for pred, actual in zip(predictions, actuals):
        pred_home = pred['predicted_home_score']
        pred_away = pred['predicted_away_score']
        actual_home = actual['home_score']
        actual_away = actual['away_score']

        # Determine actual outcome
        if actual_home > actual_away:
            actual_outcome = 'H'
        elif actual_away > actual_home:
            actual_outcome = 'A'
        else:
            actual_outcome = 'D'

        # Determine predicted outcome
        if pred_home > pred_away:
            pred_outcome = 'H'
        elif pred_away > pred_home:
            pred_outcome = 'A'
        else:
            pred_outcome = 'D'

        # Track by actual outcome
        metrics['by_outcome'][actual_outcome]['total'] += 1

        # Track score distributions
        score_str = f"{pred_home}-{pred_away}"
        actual_str = f"{actual_home}-{actual_away}"
        metrics['score_predictions'][score_str] += 1
        metrics['score_actuals'][actual_str] += 1

        # Confidence bucket
        confidence = pred.get('confidence', pred.get('max_probability', 0))
        if confidence < 0.4:
            conf_bucket = 'low'
        elif confidence < 0.6:
            conf_bucket = 'medium'
        else:
            conf_bucket = 'high'

        metrics['by_confidence'][conf_bucket]['total'] += 1

        # Calculate points
        points = 0

        # Exact score: 4 points
        if pred_home == actual_home and pred_away == actual_away:
            metrics['exact_scores'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 4
            metrics['by_confidence'][conf_bucket]['points'] += 4
            metrics['by_confidence'][conf_bucket]['correct_result'] += 1
            points = 4
        # Correct goal difference: 3 points
        elif (pred_home - pred_away) == (actual_home - actual_away):
            metrics['correct_differences'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 3
            metrics['by_confidence'][conf_bucket]['points'] += 3
            metrics['by_confidence'][conf_bucket]['correct_result'] += 1
            points = 3
        # Correct winner: 2 points
        elif pred_outcome == actual_outcome:
            metrics['correct_results'] += 1
            metrics['by_outcome'][actual_outcome]['correct'] += 1
            metrics['by_outcome'][actual_outcome]['points'] += 2
            metrics['by_confidence'][conf_bucket]['points'] += 2
            metrics['by_confidence'][conf_bucket]['correct_result'] += 1
            points = 2
        else:
            metrics['incorrect'] += 1

        metrics['total_points'] += points

    return metrics


def print_evaluation_report(metrics):
    """Print a comprehensive evaluation report."""

    n = metrics['total_matches']

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Overall metrics
    print("\n--- OVERALL PERFORMANCE ---")
    print(f"Total Matches: {n}")
    print(f"Total Points: {metrics['total_points']}")
    print(f"Average Points per Match: {metrics['total_points']/n:.3f}")
    print(f"Expected Season Total (38 matches): {(metrics['total_points']/n)*38:.1f} points")
    print()

    # Accuracy breakdown
    print("--- ACCURACY BREAKDOWN ---")
    print(f"Exact Scores (4pts):       {metrics['exact_scores']:4d} ({metrics['exact_scores']/n*100:5.1f}%)")
    print(f"Correct Differences (3pts): {metrics['correct_differences']:4d} ({metrics['correct_differences']/n*100:5.1f}%)")
    print(f"Correct Results (2pts):    {metrics['correct_results']:4d} ({metrics['correct_results']/n*100:5.1f}%)")
    print(f"Incorrect (0pts):          {metrics['incorrect']:4d} ({metrics['incorrect']/n*100:5.1f}%)")
    print()

    # Points distribution
    total_possible = n * 4
    efficiency = (metrics['total_points'] / total_possible) * 100
    print(f"Point Efficiency: {efficiency:.1f}% (of maximum possible)")
    print()

    # By outcome
    print("--- PERFORMANCE BY ACTUAL OUTCOME ---")
    for outcome in ['H', 'D', 'A']:
        outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        data = metrics['by_outcome'][outcome]

        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            avg_pts = data['points'] / data['total']
            print(f"{outcome_name:10s}: {data['total']:3d} matches, "
                  f"{data['correct']:3d} correct ({acc:5.1f}%), "
                  f"avg {avg_pts:.2f} pts")

    print()

    # By confidence
    print("--- PERFORMANCE BY CONFIDENCE LEVEL ---")
    for conf_level in ['low', 'medium', 'high']:
        data = metrics['by_confidence'][conf_level]

        if data['total'] > 0:
            result_acc = data['correct_result'] / data['total'] * 100
            avg_pts = data['points'] / data['total']
            print(f"{conf_level.capitalize():8s}: {data['total']:3d} matches, "
                  f"result accuracy {result_acc:5.1f}%, "
                  f"avg {avg_pts:.2f} pts")

    print()

    # Most common predictions
    print("--- TOP 10 PREDICTED SCORES ---")
    sorted_preds = sorted(metrics['score_predictions'].items(),
                         key=lambda x: x[1], reverse=True)[:10]
    for score, count in sorted_preds:
        print(f"{score:6s}: {count:3d} times ({count/n*100:5.1f}%)")

    print()

    # Most common actual scores
    print("--- TOP 10 ACTUAL SCORES ---")
    sorted_actuals = sorted(metrics['score_actuals'].items(),
                           key=lambda x: x[1], reverse=True)[:10]
    for score, count in sorted_actuals:
        print(f"{score:6s}: {count:3d} times ({count/n*100:5.1f}%)")

    print()
    print("="*80)


def main():
    """Run comprehensive model evaluation."""

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
        print("ERROR: No trained models found. Run train_model.py first.")
        return

    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2

    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    finished = [m for m in all_matches if m['is_finished']]

    print(f"Loaded {len(finished)} finished matches")

    # Poisson will be trained only on prior matches relative to test window (below)

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
    hist_df = pd.DataFrame(hist_prior)
    predictor.poisson_predictor.train(hist_df)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(test_features)

    # Calculate metrics
    print("Calculating metrics...")
    actuals = test_df.to_dict('records')
    metrics = calculate_detailed_metrics(predictions, actuals)

    # Print report
    print_evaluation_report(metrics)

    # Additional realism metrics: 0-0 rate and predicted outcome shares
    print("\n" + "="*80)
    print("REALISM METRICS")
    print("="*80)

    # 0-0 rate (predicted vs actual)
    zero_zero_pred = sum(1 for p in predictions if p['predicted_home_score'] == 0 and p['predicted_away_score'] == 0)
    zero_zero_act = sum(1 for a in actuals if int(a['home_score']) == 0 and int(a['away_score']) == 0)
    n = len(predictions)
    print(f"Predicted 0-0 rate: {zero_zero_pred/n*100:5.1f}% ({zero_zero_pred}/{n})")
    print(f"Actual 0-0 rate:    {zero_zero_act/n*100:5.1f}% ({zero_zero_act}/{n})")

    # Outcome distribution (predicted)
    def outcome(h, a):
        return 'H' if h > a else ('A' if a > h else 'D')

    pred_outcomes = [outcome(p['predicted_home_score'], p['predicted_away_score']) for p in predictions]
    pred_H = pred_outcomes.count('H') / n * 100
    pred_D = pred_outcomes.count('D') / n * 100
    pred_A = pred_outcomes.count('A') / n * 100

    # Outcome distribution (actual)
    act_outcomes = [outcome(int(a['home_score']), int(a['away_score'])) for a in actuals]
    act_H = act_outcomes.count('H') / n * 100
    act_D = act_outcomes.count('D') / n * 100
    act_A = act_outcomes.count('A') / n * 100

    print("\n--- Outcome Distribution (Predicted vs Actual) ---")
    print(f"Home Win: {pred_H:5.1f}% vs {act_H:5.1f}%")
    print(f"Draw:     {pred_D:5.1f}% vs {act_D:5.1f}%")
    print(f"Away Win: {pred_A:5.1f}% vs {act_A:5.1f}%")

    # Additional calibration diagnostics and debug export
    # Build outcome arrays
    def outcome_idx(h, a): return 0 if h > a else (1 if h == a else 2)
    probs = []
    actual_idx = []
    for p, a in zip(predictions, actuals):
        probs.append([p['home_win_probability'], p['draw_probability'], p['away_win_probability']])
        actual_idx.append(outcome_idx(int(a['home_score']), int(a['away_score'])))

    # Brier score
    import numpy as np, json, os, csv
    brier = float(np.mean([np.sum((np.eye(3)[ai] - np.array(pr))**2) for pr, ai in zip(probs, actual_idx)]))
    # Reliability (3 bins per class)
    def reliability(scores):
        arr = np.array(scores)
        if len(arr) == 0:
            return []
        qs = np.quantile(arr[:,0], [0, 1/3, 2/3, 1])
        rows = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (arr[:,0] >= lo) & (arr[:,0] <= hi)
            if m.sum() == 0:
                continue
            rows.append((arr[m,0].mean(), arr[m,1].mean(), int(m.sum())))
        return rows
    H_rel = reliability([(pr[0], 1 if ai==0 else 0) for pr, ai in zip(probs, actual_idx)])
    D_rel = reliability([(pr[1], 1 if ai==1 else 0) for pr, ai in zip(probs, actual_idx)])
    A_rel = reliability([(pr[2], 1 if ai==2 else 0) for pr, ai in zip(probs, actual_idx)])

    # Confusion matrix (argmax)
    pred_idx = [int(np.argmax(pr)) for pr in probs]
    conf = np.zeros((3,3), dtype=int)
    for pi, ai in zip(pred_idx, actual_idx):
        conf[ai, pi] += 1

    print("\n--- CALIBRATION ---")
    print(f"Brier score: {brier:.3f}")
    print("H reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in H_rel]))
    print("D reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in D_rel]))
    print("A reliability:", ", ".join([f"{c:.2f}~{a:.2f}(n={n})" for c,a,n in A_rel]))
    print("Confusion (rows=actual H/D/A, cols=pred):")
    print(conf)

    # Realism warnings
    gaps = []
    if abs(pred_D - act_D) > 10:
        gaps.append("Draw share gap > 10pp")
    if (act_A - pred_A) > 10:
        gaps.append("Away share underpredicted by > 10pp")
    if abs((zero_zero_pred/n*100) - (zero_zero_act/n*100)) > 3:
        gaps.append("0-0 rate gap > 3pp")
    if gaps:
        print("WARN:", "; ".join(gaps))

    # Per-match debug CSV
    dbg_path = os.path.join('data','predictions','debug_eval.csv')
    os.makedirs(os.path.dirname(dbg_path), exist_ok=True)
    with open(dbg_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['match_id','home','away','pred_h','pred_a','act_h','act_a','lambda_h','lambda_a','pH','pD','pA','confidence','points'])
        for p, a in zip(predictions, actuals):
            ah, aa = int(a['home_score']), int(a['away_score'])
            points = 4 if (p['predicted_home_score']==ah and p['predicted_away_score']==aa) else \
                     3 if ((p['predicted_home_score']-p['predicted_away_score'])==(ah-aa)) else \
                     2 if ( (p['predicted_home_score']>p['predicted_away_score'])==(ah>aa) and (p['predicted_home_score']==p['predicted_away_score'])==(ah==aa) ) else 0
            w.writerow([p['match_id'], p['home_team'], p['away_team'],
                        p['predicted_home_score'], p['predicted_away_score'],
                        ah, aa,
                        f"{p['home_expected_goals']:.3f}", f"{p['away_expected_goals']:.3f}",
                        f"{p['home_win_probability']:.3f}", f"{p['draw_probability']:.3f}", f"{p['away_win_probability']:.3f}",
                        f"{p.get('confidence',0):.3f}", points])
    print(f"Wrote per-match debug: {dbg_path}")

    # Persist run meta
    meta_path = os.path.join('data','predictions','run_meta.json')
    meta = {
        'script': 'evaluate_model.py',
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


if __name__ == "__main__":
    main()
