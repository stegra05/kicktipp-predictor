#!/usr/bin/env python3
"""
Feature Ablation Study

Systematically tests different feature subsets to find the optimal balance
between model complexity and performance.

Strategy:
1. Baseline: All features (from kept_features.yaml)
2. Category ablation: Remove feature categories one at a time
3. Importance-based pruning: Remove N% lowest importance features
4. Minimal set: Test core features only

Output:
- Comparative metrics CSV
- Performance plots
- Recommendations report
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import (
    LABELS_ORDER,
    brier_score_multiclass,
    compute_points,
    log_loss_multiclass,
    ranked_probability_score_3c,
)
from kicktipp_predictor.predictor import MatchPredictor


class FeatureAblationStudy:
    """Automated feature ablation study."""

    def __init__(self, output_dir: str = "data/feature_ablation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader()
        self.results: list[dict] = []

        # Load base feature list
        self.all_features = self._load_kept_features()

        # Define feature categories
        self.feature_categories = self._categorize_features()

    def _load_kept_features(self) -> list[str]:
        """Load the current kept features list."""
        kept_path = project_root / "data" / "feature_selection" / "kept_features.yaml"
        if kept_path.exists():
            with open(kept_path) as f:
                features = yaml.safe_load(f)
                if isinstance(features, list):
                    return features

        # Fallback: use all 62 features from analysis
        print("⚠️  Warning: Could not load kept_features.yaml, using default list")
        return self._get_default_features()

    def _get_default_features(self) -> list[str]:
        """Default feature list (62 features from analysis)."""
        return [
            "abs_form_points_diff",
            "abs_momentum_score_diff",
            "attack_defense_form_ratio_away",
            "attack_defense_form_ratio_home",
            "attack_defense_long_ratio_away",
            "attack_defense_long_ratio_home",
            "away_avg_goals_against",
            "away_avg_points",
            "away_form_avg_goals_conceded",
            "away_form_avg_goals_scored",
            "away_form_draws",
            "away_form_goal_diff",
            "away_form_goals_conceded",
            "away_form_goals_scored",
            "away_form_losses",
            "away_form_points",
            "away_goal_diff_ewm5",
            "away_goals_against_ewm5",
            "away_goals_conceded_pg_at_home",
            "away_goals_conceded_pg_away",
            "away_goals_for_ewm5",
            "away_goals_pg_at_home",
            "away_goals_pg_away",
            "away_momentum_conceded",
            "away_momentum_goals",
            "away_momentum_score",
            "away_points_ewm5",
            "away_points_pg_at_home",
            "away_points_pg_away",
            "ewm_points_ratio",
            "form_points_difference",
            "form_points_pg_ratio",
            "home_avg_goals_against",
            "home_avg_goals_for",
            "home_avg_points",
            "home_form_avg_goals_conceded",
            "home_form_avg_goals_scored",
            "home_form_draws",
            "home_form_goal_diff",
            "home_form_goals_conceded",
            "home_form_goals_scored",
            "home_form_losses",
            "home_form_points",
            "home_goal_diff_ewm5",
            "home_goals_against_ewm5",
            "home_goals_conceded_pg_at_home",
            "home_goals_conceded_pg_away",
            "home_goals_for_ewm5",
            "home_goals_pg_at_home",
            "home_goals_pg_away",
            "home_momentum_conceded",
            "home_momentum_goals",
            "home_momentum_score",
            "home_points_ewm5",
            "home_points_pg_at_home",
            "home_points_pg_away",
            "momentum_score_difference",
            "momentum_score_ratio",
            "venue_conceded_delta",
            "venue_goals_delta",
            "venue_points_delta",
        ]

    def _categorize_features(self) -> dict[str, list[str]]:
        """Categorize features for ablation experiments."""
        categories = defaultdict(list)

        for feat in self.all_features:
            # Interaction ratios
            if "ratio" in feat:
                categories["interaction_ratios"].append(feat)
            # Venue-specific
            elif "pg_at_home" in feat or "pg_away" in feat or "venue_" in feat:
                categories["venue_specific"].append(feat)
            # EWMA recency
            elif "ewm" in feat:
                categories["ewma_recency"].append(feat)
            # Momentum
            elif "momentum" in feat:
                categories["momentum"].append(feat)
            # Form
            elif "form" in feat:
                categories["form"].append(feat)
            # Base stats
            elif "avg_goals" in feat or "avg_points" in feat:
                categories["base_stats"].append(feat)
            # Derived differences
            elif "diff" in feat or "difference" in feat:
                categories["derived_diffs"].append(feat)
            else:
                categories["other"].append(feat)

        return dict(categories)

    def _create_temp_feature_file(self, features: list[str]) -> Path:
        """Create a temporary feature selection file."""
        temp_path = self.output_dir / "temp_selected_features.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(features, f)
        return temp_path

    def _train_and_evaluate(
        self,
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        experiment_name: str,
    ) -> dict:
        """Train model with specific features and evaluate."""
        print(f"\n{'=' * 80}")
        print(f"Experiment: {experiment_name}")
        print(f"Features: {len(features)}")
        print(f"{'=' * 80}")

        start_time = time.time()

        # Create predictor
        predictor = MatchPredictor(quiet=True)

        # Train
        try:
            predictor.train(train_df)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {
                "experiment": experiment_name,
                "n_features": len(features),
                "status": "failed",
                "error": str(e),
            }

        # Prepare test data
        test_features = test_df.drop(
            columns=["home_score", "away_score", "goal_difference", "result"],
            errors="ignore",
        )

        # Predict
        try:
            predictions = predictor.predict(test_features)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {
                "experiment": experiment_name,
                "n_features": len(features),
                "status": "failed",
                "error": str(e),
            }

        # Evaluate
        y_true = []
        for _, row in test_df.iterrows():
            if row["home_score"] > row["away_score"]:
                y_true.append("H")
            elif row["away_score"] > row["home_score"]:
                y_true.append("A")
            else:
                y_true.append("D")

        # Build probability matrix
        proba = np.array(
            [
                [
                    float(p.get("home_win_probability", 1 / 3)),
                    float(p.get("draw_probability", 1 / 3)),
                    float(p.get("away_win_probability", 1 / 3)),
                ]
                for p in predictions
            ]
        )
        proba = np.clip(proba, 1e-15, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)

        # Predicted scores and points
        ph = np.array([int(p.get("predicted_home_score", 0)) for p in predictions])
        pa = np.array([int(p.get("predicted_away_score", 0)) for p in predictions])
        ah = test_df["home_score"].values
        aa = test_df["away_score"].values

        points = compute_points(ph, pa, ah, aa)

        # Calculate metrics
        pred_outcomes = np.array([LABELS_ORDER[i] for i in np.argmax(proba, axis=1)])
        accuracy = float(np.mean(pred_outcomes == np.array(y_true)))

        brier = brier_score_multiclass(y_true, proba)
        logloss = log_loss_multiclass(y_true, proba)
        rps = ranked_probability_score_3c(y_true, proba)

        avg_points = float(np.mean(points))
        total_points = int(np.sum(points))

        elapsed = time.time() - start_time

        result = {
            "experiment": experiment_name,
            "n_features": len(features),
            "status": "success",
            "n_samples": len(test_df),
            "accuracy": accuracy,
            "avg_points": avg_points,
            "total_points": total_points,
            "brier_score": brier,
            "log_loss": logloss,
            "rps": rps,
            "training_time_sec": elapsed,
        }

        print("\n✅ Results:")
        print(f"   Accuracy:    {accuracy:.3f}")
        print(f"   Avg Points:  {avg_points:.3f}")
        print(f"   Brier:       {brier:.4f}")
        print(f"   Log Loss:    {logloss:.4f}")
        print(f"   RPS:         {rps:.4f}")
        print(f"   Time:        {elapsed:.1f}s")

        return result

    def run_baseline(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Run baseline with all features."""
        return self._train_and_evaluate(
            self.all_features, train_df, test_df, "baseline_all_features"
        )

    def run_category_ablation(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> list[dict]:
        """Remove each feature category and test."""
        results = []

        for category, cat_features in self.feature_categories.items():
            # Features without this category
            remaining = [f for f in self.all_features if f not in cat_features]

            result = self._train_and_evaluate(
                remaining, train_df, test_df, f"ablate_{category}"
            )
            result["removed_category"] = category
            result["n_removed"] = len(cat_features)
            results.append(result)

        return results

    def run_minimal_core(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Test minimal core feature set."""
        # Core features: base stats + form + momentum (no EWMA, venue, ratios)
        core_features = []
        for feat in self.all_features:
            # Keep base stats, form, and momentum_score
            if any(
                x in feat
                for x in [
                    "avg_goals",
                    "avg_points",
                    "form_points",
                    "form_goal_diff",
                    "momentum_score",
                ]
            ):
                # But skip venue-specific and ratios
                if "pg_at_" not in feat and "ratio" not in feat and "ewm" not in feat:
                    core_features.append(feat)

        # Add simple differences
        core_features.extend(
            [
                f
                for f in self.all_features
                if "difference" in f
                or f in ["abs_form_points_diff", "abs_momentum_score_diff"]
            ]
        )

        # Remove duplicates
        core_features = list(set(core_features))

        return self._train_and_evaluate(
            core_features, train_df, test_df, "minimal_core_set"
        )

    def run_percentage_pruning(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        percentages: list[int] | None = None,
    ) -> list[dict]:
        """Test removing N% of features (simulated importance-based)."""
        results = []

        if percentages is None:
            percentages = [10, 20, 30, 40]

        # Simulate importance by category priority
        # Lower priority = more likely to remove
        priority = {
            "interaction_ratios": 1,  # Remove first
            "venue_specific": 2,
            "derived_diffs": 3,
            "ewma_recency": 4,
            "momentum": 5,
            "form": 6,
            "base_stats": 7,  # Keep last
            "other": 4,
        }

        # Sort features by priority (low to high)
        sorted_features = []
        for cat in sorted(
            self.feature_categories.keys(), key=lambda x: priority.get(x, 5)
        ):
            sorted_features.extend(self.feature_categories[cat])

        # Add any remaining features
        sorted_features.extend(
            [f for f in self.all_features if f not in sorted_features]
        )

        for pct in percentages:
            n_remove = int(len(self.all_features) * pct / 100)
            remaining = sorted_features[n_remove:]  # Keep higher priority

            result = self._train_and_evaluate(
                remaining, train_df, test_df, f"prune_{pct}pct"
            )
            result["percent_removed"] = pct
            result["n_removed"] = n_remove
            results.append(result)

        return results

    def generate_report(self, baseline: dict, all_results: list[dict]):
        """Generate comparison report."""
        print(f"\n{'=' * 80}")
        print("FEATURE ABLATION STUDY - SUMMARY REPORT")
        print(f"{'=' * 80}\n")

        # Save full results
        results_df = pd.DataFrame([baseline] + all_results)
        results_csv = self.output_dir / "ablation_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"📊 Full results saved to: {results_csv}\n")

        # Calculate deltas from baseline
        baseline_points = baseline["avg_points"]
        baseline_acc = baseline["accuracy"]

        print(f"BASELINE ({baseline['n_features']} features):")
        print(f"  Accuracy:   {baseline_acc:.3f}")
        print(f"  Avg Points: {baseline_points:.3f}")
        print(f"  Brier:      {baseline['brier_score']:.4f}")
        print()

        # Sort by performance (avg_points descending)
        sorted_results = sorted(
            [r for r in all_results if r["status"] == "success"],
            key=lambda x: x["avg_points"],
            reverse=True,
        )

        print("TOP CONFIGURATIONS (by avg_points):")
        print(
            f"{'Rank':<6} {'Experiment':<30} {'Features':<10} {'Δ Pts':<10} {'Δ Acc':<10} {'Brier':<10}"
        )
        print("-" * 80)

        for i, result in enumerate(sorted_results[:10], 1):
            delta_pts = result["avg_points"] - baseline_points
            delta_acc = result["accuracy"] - baseline_acc

            print(
                f"{i:<6} {result['experiment']:<30} {result['n_features']:<10} "
                f"{delta_pts:+.3f}{'':>4} {delta_acc:+.3f}{'':>4} {result['brier_score']:.4f}"
            )

        print()

        # Find best trade-offs
        print("RECOMMENDED CONFIGURATIONS:")

        # Best overall (ignore if baseline is best)
        best = sorted_results[0]
        if best["experiment"] != "baseline_all_features":
            print("\n🏆 Best Performance:")
            print(f"   {best['experiment']}: {best['n_features']} features")
            print(
                f"   Avg Points: {best['avg_points']:.3f} ({best['avg_points'] - baseline_points:+.3f})"
            )
            print(
                f"   Accuracy: {best['accuracy']:.3f} ({best['accuracy'] - baseline_acc:+.3f})"
            )

        # Best simplification (minimal features with <0.1 points loss)
        good_simpler = [
            r
            for r in sorted_results
            if r["n_features"] < baseline["n_features"]
            and (baseline_points - r["avg_points"]) < 0.10
        ]

        if good_simpler:
            simplest = min(good_simpler, key=lambda x: x["n_features"])
            reduction_pct = (1 - simplest["n_features"] / baseline["n_features"]) * 100

            print("\n🎯 Best Simplification (<0.1 pts loss):")
            print(
                f"   {simplest['experiment']}: {simplest['n_features']} features (-{reduction_pct:.0f}%)"
            )
            print(
                f"   Avg Points: {simplest['avg_points']:.3f} ({simplest['avg_points'] - baseline_points:+.3f})"
            )
            print(
                f"   Accuracy: {simplest['accuracy']:.3f} ({simplest['accuracy'] - baseline_acc:+.3f})"
            )

        # Category insights
        print("\n📈 CATEGORY ABLATION INSIGHTS:")
        category_results = [
            r
            for r in all_results
            if "removed_category" in r and r["status"] == "success"
        ]
        category_impact = sorted(
            category_results,
            key=lambda x: baseline_points - x["avg_points"],
            reverse=True,
        )

        print(f"{'Category':<25} {'Impact':<15} {'Features':<10} {'Δ Pts':<10}")
        print("-" * 70)
        for r in category_impact:
            delta = baseline_points - r["avg_points"]
            impact = (
                "🔴 High" if delta > 0.15 else "🟡 Medium" if delta > 0.05 else "🟢 Low"
            )
            print(
                f"{r['removed_category']:<25} {impact:<15} {r['n_removed']:<10} {-delta:+.3f}"
            )

        # Low-impact categories can be removed
        low_impact = [
            r for r in category_impact if (baseline_points - r["avg_points"]) < 0.05
        ]
        if low_impact:
            print("\n✅ Safe to remove (< 0.05 pts impact):")
            for r in low_impact:
                print(f"   - {r['removed_category']} ({r['n_removed']} features)")

        print(f"\n{'=' * 80}")

    def run_full_study(self):
        """Run complete ablation study."""
        print("🚀 Starting Feature Ablation Study")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Total features: {len(self.all_features)}")
        print()

        # Load data - USE MORE SEASONS FOR BETTER TRAINING
        print("📊 Loading data...")
        current_season = self.loader.get_current_season()
        start_season = current_season - 4  # Changed from -2 to -4 for more data

        all_matches = self.loader.fetch_historical_seasons(start_season, current_season)
        features_df = self.loader.create_features_from_matches(all_matches)

        # Train/test split (80/20 for more training data)
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]

        print(f"   Train: {len(train_df)} samples")
        print(f"   Test:  {len(test_df)} samples")

        all_results = []

        # 1. Baseline
        print("\n" + "=" * 80)
        print("PHASE 1: BASELINE")
        print("=" * 80)
        baseline = self.run_baseline(train_df, test_df)

        # 2. Category ablation
        print("\n" + "=" * 80)
        print("PHASE 2: CATEGORY ABLATION")
        print("=" * 80)
        category_results = self.run_category_ablation(train_df, test_df)
        all_results.extend(category_results)

        # 3. Percentage pruning
        print("\n" + "=" * 80)
        print("PHASE 3: PERCENTAGE PRUNING")
        print("=" * 80)
        pruning_results = self.run_percentage_pruning(train_df, test_df)
        all_results.extend(pruning_results)

        # 4. Minimal core
        print("\n" + "=" * 80)
        print("PHASE 4: MINIMAL CORE SET")
        print("=" * 80)
        minimal = self.run_minimal_core(train_df, test_df)
        all_results.append(minimal)

        # Generate report
        self.generate_report(baseline, all_results)

        return baseline, all_results


def main():
    """Run feature ablation study."""
    study = FeatureAblationStudy()
    baseline, results = study.run_full_study()

    print("\n✅ Feature ablation study complete!")
    print(f"📁 Results saved to: {study.output_dir}")


if __name__ == "__main__":
    main()
