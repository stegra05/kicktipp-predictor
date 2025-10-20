# Quick Start Guide

Get up and running with the 3. Liga Predictor in 5 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

## Step 2: Train the Models (2-3 minutes)

```bash
python train_model.py
```

This fetches historical data and trains the prediction models. You'll see:
- Data fetching progress
- Training progress for ML models
- Evaluation results on test data
- Model save confirmation

**Expected output**:
```
Fetching historical data...
Total matches fetched: 1500+
Training ML models...
Training completed!
Average points per match: 2.3-2.5
Models saved to data/models/
```

## Step 3: Generate Predictions (30 seconds)

```bash
python predict.py --record
```

This generates predictions for upcoming matches and records them for tracking.

**You'll see**:
- List of upcoming matches
- Predicted scores
- Win/draw/loss probabilities
- Confidence levels

## Step 4: Launch Web Interface (10 seconds)

```bash
python src/web/app.py
```

Open your browser to: **http://localhost:5000**

Explore:
- 📊 **Predictions** - View upcoming match predictions
- 📋 **Table** - Current 3. Liga standings
- 📈 **Statistics** - Your prediction performance

## Quick Commands Reference

```bash
# Generate predictions for next 7 days
python predict.py

# Predict specific matchday
python predict.py --matchday 15

# Use different strategies
python predict.py --strategy aggressive
python predict.py --strategy conservative
python predict.py --strategy safe

# Record predictions for tracking
python predict.py --record

# Update with actual results and see your points
python predict.py --update-results

# Retrain models (do this every 4-6 weeks)
python train_model.py
```

## Weekly Routine

**Monday** (after matches finish):
```bash
python predict.py --update-results
```
See how many points you earned!

**Tuesday/Wednesday** (before next matchday):
```bash
python predict.py --matchday <next_matchday> --record
```
Generate and record new predictions.

## Understanding the Output

### Prediction Format
```
Bayern Munich II vs 1860 München
Predicted Score: 2:1
Probabilities: Home 55.3% | Draw 24.2% | Away 20.5%
Confidence: 55.3%
```

- **Predicted Score**: Most likely final score
- **Probabilities**: Chance of each outcome (Home win / Draw / Away win)
- **Confidence**: Reliability of prediction (higher = more confident)

### Points System
- **4 points**: Exact score (2:1 predicted, 2:1 actual)
- **3 points**: Correct difference (2:1 predicted, 3:2 actual)
- **2 points**: Correct winner (2:1 predicted, 3:1 actual)
- **0 points**: Wrong (2:1 predicted, 1:2 actual)

## Prediction Strategies Explained

- **Balanced** ⚖️: Best all-around, uses standard hybrid predictions
- **Conservative** 🛡️: Safer predictions with lower scores
- **Aggressive** 🎯: Goes for exact scores, higher risk/reward
- **Safe** ✅: Prioritizes getting the winner right

Try different strategies to see what works best!

## Troubleshooting

**Problem**: "No trained models found"
**Solution**: Run `python train_model.py` first

**Problem**: Web interface won't start
**Solution**: Make sure port 5000 is free, or edit `src/web/app.py` to use a different port

**Problem**: No upcoming matches found
**Solution**: Check if there are matches in the next 7 days, or try a specific matchday

## Tips for Success

1. ✅ **Train regularly** - Retrain every 4-6 weeks
2. ✅ **Record predictions** - Track your performance over time
3. ✅ **Update results** - Check your points weekly
4. ✅ **Compare strategies** - See which works best for you
5. ✅ **Check confidence** - Higher confidence = more reliable

## Need Help?

- Read the full **README.md** for detailed documentation
- Check the code comments for implementation details
- Issues? Check the Troubleshooting section in README.md

---

**You're all set! Good luck with your predictions!** ⚽🎯
