# 3. Liga Match Predictor

A hybrid machine learning and statistical predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

## Features

- **Advanced Hybrid Prediction Model**: Combines XGBoost ML models with Dixon-Coles enhanced Poisson models
- **Probability Calibration**: Isotonic regression calibration for ML expected goals, blended outcome probabilities
- **Enhanced Confidence Metrics**: Margin-based confidence calculation with entropy-based alternatives
- **Advanced Feature Engineering**: 80+ features including momentum, strength of schedule, rest days, and goal patterns
- **Multiple Prediction Strategies**: Balanced, Conservative, Aggressive, and Safe strategies to optimize for different scenarios
- **Hyperparameter Tuning**: Cross-validation tools for optimizing model weights and parameters
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis by outcome/confidence
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics
- **Data Scraping**: Automatic fetching of match data from OpenLigaDB API

## Recent Improvements (Latest Version)

### Model Enhancements
1. **Dixon-Coles Correction**: Added low-score interaction adjustments to Poisson model for more realistic 0-0, 1-0, 0-1, 1-1 predictions
2. **Isotonic Calibration**: ML expected goals now calibrated using isotonic regression for better lambda estimates
3. **Blended Probabilities**: Outcome probabilities now blend Poisson grid (65%) with ML classifier (35%) for improved accuracy
4. **Improved Confidence**: New margin-based confidence metric that captures prediction certainty better than max probability alone

### Feature Engineering
5. **Momentum Features**: Exponentially weighted recent form (decay factor 0.9) gives more importance to recent matches
6. **Strength of Schedule**: Tracks average opponent strength over last 5 matches
7. **Rest Features**: Days since last match, fatigue indicators, rest advantage between teams
8. **Better Validation**: H2H features now validate for finished matches with valid scores

### Tools & Utilities
9. **Hyperparameter Tuning**: `tune_hyperparameters.py` - Cross-validation grid search to optimize weights
10. **Comprehensive Evaluation**: `evaluate_model.py` - Detailed metrics, strategy comparison, score distributions
11. **Robustness**: Better error handling, model readiness checks, feature alignment, JSON type safety

## Scoring System

Predictions earn points based on accuracy:
- **4 points**: Exact score prediction (e.g., predicted 2:1, actual 2:1)
- **3 points**: Correct goal difference (e.g., predicted 2:0, actual 3:1)
- **2 points**: Correct winner (e.g., predicted 1:0, actual 2:1)
- **0 points**: Incorrect prediction

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

1. Clone or download this repository:
```bash
cd kicktipp-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Models

Before making predictions, you need to train the models on historical data:

```bash
python train_model.py
```

This will:
- Fetch 3-4 seasons of historical 3. Liga data
- Create features from match statistics
- Train both ML and Poisson models
- Evaluate performance on test data
- Save trained models to `data/models/`

**Note**: Training should be done initially and then periodically (e.g., every few weeks) to update models with new data.

### 2. Generate Predictions

Generate predictions for upcoming matches:

```bash
# Predict matches in the next 7 days
python predict.py

# Predict a specific matchday
python predict.py --matchday 15

# Use different prediction strategies
python predict.py --strategy conservative
python predict.py --strategy aggressive
python predict.py --strategy safe

# Record predictions for performance tracking
python predict.py --record

# Update previous predictions with actual results
python predict.py --update-results
```

### Prediction Strategies

- **Balanced** (default): Standard hybrid predictions
- **Conservative**: Favor lower-scoring, more common results
- **Aggressive**: Try to predict exact common scorelines
- **Safe**: Prioritize getting the winner right over exact scores

### 3. Evaluate Model Performance

Get detailed performance metrics and compare strategies:

```bash
python evaluate_model.py
```

This provides:
- Overall performance (points per match, accuracy breakdown)
- Performance by actual outcome (home win, draw, away win)
- Performance by confidence level (low, medium, high)
- Top predicted and actual scorelines
- Strategy comparison (which strategy performs best)

### 4. Tune Hyperparameters (Optional)

Optimize model weights using cross-validation:

```bash
python tune_hyperparameters.py
```

This performs grid search over:
- `ml_weight`: Weight for ML model (vs Poisson)
- `prob_blend_alpha`: Weight for Poisson probabilities (vs ML classifier)
- `min_lambda`: Minimum expected goals to avoid degenerate predictions

**Note**: This takes 10-20 minutes but can improve performance by 0.1-0.3 points/match

### 5. Run the Web Interface

Start the web server:

```bash
python src/web/app.py
```

Then open your browser to `http://localhost:5000`

The web interface provides:
- **Predictions Page**: View upcoming match predictions with probabilities and confidence
- **Table Page**: Current 3. Liga league table
- **Statistics Page**: Your prediction performance statistics

## Project Structure

```
kicktipp-predictor/
├── src/
│   ├── scraper/
│   │   └── data_fetcher.py       # Fetch match data from API
│   ├── features/
│   │   └── feature_engineering.py # Create prediction features
│   ├── models/
│   │   ├── poisson_model.py      # Statistical Poisson model
│   │   ├── ml_model.py           # XGBoost ML model
│   │   ├── hybrid_predictor.py   # Hybrid ensemble model
│   │   └── performance_tracker.py # Track prediction performance
│   └── web/
│       ├── app.py                # Flask web application
│       ├── templates/            # HTML templates
│       └── static/               # CSS and JS files
├── data/
│   ├── cache/                    # Cached API responses
│   ├── models/                   # Trained model files
│   └── predictions/              # Prediction history
├── train_model.py                # Script to train models
├── predict.py                    # Script to generate predictions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How It Works

### Data Collection

The system uses the free OpenLigaDB API to fetch:
- Current season fixtures and results
- Historical match data for training
- Team statistics and form

### Feature Engineering

For each match, the system calculates:
- **Recent Form**: Last 5 matches (points, goals, wins/draws/losses)
- **Head-to-Head**: Historical matchups between the teams
- **Home/Away Performance**: Team-specific venue statistics
- **Goal Statistics**: Average goals scored and conceded
- **League Position**: Current table position and points

### Prediction Models

1. **Poisson Model**: Uses team attack/defense strengths and league averages
2. **XGBoost Models**: Separate models for home goals, away goals, and match result
3. **Hybrid Ensemble**: Weighted combination (60% ML, 40% Poisson)

### Performance Tracking

The system tracks:
- Prediction accuracy (exact scores, differences, results)
- Points earned per match
- Performance by matchday
- Projected season total

## Tips for Best Results

1. **Train Regularly**: Retrain models every 4-6 weeks to incorporate recent data
2. **Record Predictions**: Use `--record` flag to track your performance over time
3. **Update Results**: Run `--update-results` weekly to calculate your points
4. **Try Different Strategies**: Test different strategies to see which performs best
5. **Consider Confidence**: Higher confidence predictions are more reliable

## Example Workflow

Weekly routine for a fantasy football league:

```bash
# Monday: Update results from last matchday
python predict.py --update-results

# Check your performance
python predict.py --update-results | grep "PERFORMANCE"

# Tuesday: Generate predictions for upcoming matchday
python predict.py --matchday 20 --record --strategy balanced

# View predictions in web interface
python src/web/app.py
# Open http://localhost:5000

# Monthly: Retrain models with latest data
python train_model.py
```

## Data Sources

- **OpenLigaDB API**: Free German football data API
  - Documentation: https://www.openligadb.de/
  - No API key required
  - Rate limits apply (be respectful)

## Troubleshooting

### "No trained models found"
Run `python train_model.py` first to train the models.

### "Not enough historical data"
The API might be temporarily unavailable. Try again later or check your internet connection.

### "Module not found" errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

### Web interface not loading
- Check that port 5000 is not in use
- Try accessing via `http://127.0.0.1:5000` instead of localhost
- Check for error messages in the console

## Performance Expectations

Based on testing with historical data:
- **Average points per match**: 2.2-2.5
- **Exact score accuracy**: 8-12%
- **Correct difference accuracy**: 12-18%
- **Correct result accuracy**: 30-40%
- **Projected season total**: 80-95 points (38 matchdays)

## Future Enhancements

Potential improvements:
- Player injury and suspension data
- Weather conditions
- Referee statistics
- Betting odds integration
- More sophisticated ensemble methods
- Deep learning models

## License

This project is for educational and personal use only.

## Disclaimer

This predictor is for entertainment purposes. Football is unpredictable, and no model can guarantee accurate predictions. Use responsibly and never bet more than you can afford to lose.

---

**Good luck with your predictions!**
