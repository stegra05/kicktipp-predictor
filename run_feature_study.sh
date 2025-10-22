#!/bin/bash
# Quick runner for feature ablation study

set -e

echo "🚀 Feature Ablation Study"
echo "========================="
echo ""
echo "This will test different feature subsets to find optimal complexity/performance balance."
echo "Expected runtime: 15-30 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if kicktipp_predictor is installed
if ! python -c "import kicktipp_predictor" 2>/dev/null; then
    echo "❌ kicktipp_predictor not installed. Run: pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p data/feature_ablation

echo ""
echo "🏃 Running study..."
echo ""

# Run the study
python experiments/feature_ablation.py | tee data/feature_ablation/study_log.txt

echo ""
echo "✅ Study complete!"
echo ""
echo "📁 Results:"
echo "   - CSV:    data/feature_ablation/ablation_results.csv"
echo "   - Log:    data/feature_ablation/study_log.txt"
echo ""
echo "📖 Next steps:"
echo "   1. Review the recommendations above"
echo "   2. Edit data/feature_selection/kept_features.yaml to remove low-impact features"
echo "   3. Retrain: python -m kicktipp_predictor train"
echo "   4. Validate: python -m kicktipp_predictor evaluate --season"
echo ""

