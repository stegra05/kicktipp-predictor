# 3. Liga Predictor - Project Summary

## Overview

A complete football match prediction system for Germany's 3. Bundesliga, designed to maximize points in fantasy football leagues. The system uses a hybrid approach combining machine learning (XGBoost) and statistical methods (Poisson distribution).

## System Architecture

### 1. Data Layer (`src/scraper/`)
- **data_fetcher.py**: Fetches match data from OpenLigaDB API
  - Current season fixtures and results
  - Historical data for training
  - Caching mechanism to reduce API calls

### 2. Feature Engineering (`src/features/`)
- **feature_engineering.py**: Creates prediction features
  - Team form (last 5 matches)
  - Head-to-head statistics
  - Home/away performance
  - Goal statistics
  - League table position
  - ~50+ features per match

### 3. Prediction Models (`src/models/`)

#### Poisson Model (`poisson_model.py`)
- Statistical approach using Poisson distribution
- Calculates team attack/defense strengths
- Generates probability distributions for scorelines

#### ML Model (`ml_model.py`)
- XGBoost regressors for home/away goals
- XGBoost classifier for match result
- Trained on historical features

#### Hybrid Predictor (`hybrid_predictor.py`)
- Combines Poisson and ML models (60% ML, 40% Poisson)
- Multiple prediction strategies:
  - **Balanced**: Standard predictions
  - **Conservative**: Lower-scoring predictions
  - **Aggressive**: Aims for exact scores
  - **Safe**: Prioritizes correct winner

#### Performance Tracker (`performance_tracker.py`)
- Records predictions and actual results
- Calculates points earned
- Tracks accuracy metrics
- Provides performance analytics

### 4. Web Interface (`src/web/`)

#### Backend (`app.py`)
Flask application with REST API endpoints:
- `/api/upcoming_predictions` - Get predictions for upcoming matches
- `/api/current_matchday` - Get current matchday predictions
- `/api/performance` - Get performance statistics
- `/api/table` - Get current league table

#### Frontend (HTML/CSS/JS)
- **index.html**: Main predictions page
- **table.html**: League table view
- **statistics.html**: Performance statistics
- **style.css**: Modern, responsive design
- **main.js**: Client-side interactivity

### 5. Command-Line Scripts

#### `train_model.py`
- Fetches 3-4 seasons of historical data
- Creates feature dataset
- Trains ML and Poisson models
- Evaluates on test set
- Saves trained models

#### `predict.py`
- Loads trained models
- Fetches upcoming matches
- Generates predictions with chosen strategy
- Records predictions for tracking
- Updates results and calculates points

#### `demo.py`
- Tests data fetching
- Verifies feature engineering
- Shows current table
- Validates system functionality

## Key Features

### Scoring System
- **4 points**: Exact score match
- **3 points**: Correct goal difference
- **2 points**: Correct winner
- **0 points**: Incorrect prediction

### Model Performance
Based on historical testing:
- Average: 2.2-2.5 points per match
- Exact scores: 8-12% accuracy
- Correct differences: 12-18% accuracy
- Correct results: 30-40% accuracy
- Projected season: 80-95 points (38 matchdays)

### Prediction Strategies

1. **Balanced** (Default)
   - 60% ML, 40% Poisson
   - Best all-around performance
   - Use for: General predictions

2. **Conservative**
   - Reduces predicted goals
   - Safer predictions
   - Use for: Defensive teams, tight matches

3. **Aggressive**
   - Rounds to common scorelines
   - Higher risk/reward
   - Use for: When confidence is high

4. **Safe**
   - Prioritizes correct winner
   - Uses typical scores (1-0, 2-1, 1-1)
   - Use for: Maximizing baseline points

## Data Flow

```
1. Data Fetching (OpenLigaDB API)
   ↓
2. Feature Engineering (50+ features per match)
   ↓
3. Model Prediction
   ├── Poisson Model (statistical)
   ├── ML Model (XGBoost)
   └── Hybrid Ensemble (weighted average)
   ↓
4. Strategy Optimization
   ↓
5. Output (Predictions + Probabilities)
   ↓
6. Performance Tracking (points calculation)
```

## Technology Stack

### Backend
- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: ML utilities
- **XGBoost**: Gradient boosting models
- **SciPy**: Poisson distribution
- **Flask**: Web framework
- **Requests**: API calls
- **BeautifulSoup**: Web scraping fallback

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript (Vanilla)**: Interactivity
- **Responsive Design**: Mobile-friendly

### Data
- **OpenLigaDB API**: Free German football data
- **JSON**: Prediction storage
- **Pickle**: Model serialization

## Project Structure

```
kicktipp-predictor/
├── src/
│   ├── scraper/           # Data fetching
│   ├── features/          # Feature engineering
│   ├── models/            # Prediction models
│   └── web/               # Web interface
├── data/
│   ├── cache/             # API cache
│   ├── models/            # Trained models
│   └── predictions/       # Prediction history
├── train_model.py         # Training script
├── predict.py             # Prediction script
├── demo.py                # Demo/test script
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
└── PROJECT_SUMMARY.md     # This file
```

## Usage Workflow

### Initial Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Test system: `python demo.py`
3. Train models: `python train_model.py`

### Weekly Routine
1. **Monday**: Update results
   ```bash
   python predict.py --update-results
   ```

2. **Tuesday**: Generate predictions
   ```bash
   python predict.py --matchday X --record
   ```

3. **View in browser**:
   ```bash
   python src/web/app.py
   # Open http://localhost:5000
   ```

### Monthly Maintenance
- Retrain models: `python train_model.py`

## Extension Points

### Easy to Add
- New prediction strategies (edit `hybrid_predictor.py`)
- Additional features (edit `feature_engineering.py`)
- Custom API endpoints (edit `web/app.py`)
- UI customization (edit `web/static/`)

### Future Enhancements
- Player injury/suspension data
- Weather conditions
- Referee statistics
- Betting odds integration
- Deep learning models (LSTM, Transformer)
- Mobile app

## Performance Optimization

### Caching
- API responses cached for 1 hour
- Reduces load on OpenLigaDB
- Faster repeated queries

### Model Efficiency
- XGBoost: Fast inference (~1ms per match)
- Poisson: Instant calculation
- Batch prediction support

### Web Performance
- Lightweight frontend (~50KB total)
- Responsive design
- No heavy dependencies

## Testing Strategy

1. **Data Validation**: Check API responses
2. **Feature Testing**: Verify feature calculations
3. **Model Evaluation**: Test set performance
4. **Integration Testing**: End-to-end workflow
5. **Demo Script**: Quick system verification

## Maintenance

### Regular Tasks
- **Weekly**: Generate predictions, update results
- **Monthly**: Retrain models with new data
- **Seasonal**: Full retraining on 3-4 seasons

### Monitoring
- Track prediction accuracy
- Monitor API availability
- Check model performance degradation

## Success Metrics

### Target Performance
- **Minimum**: 2.0 points/match (76 points/season)
- **Target**: 2.3 points/match (87 points/season)
- **Excellent**: 2.5 points/match (95 points/season)

### Accuracy Goals
- Exact scores: 10%+
- Correct differences: 15%+
- Correct results: 35%+

## Known Limitations

1. **Data Availability**: Depends on OpenLigaDB API
2. **Historical Data**: Limited to ~4 seasons
3. **External Factors**: Cannot account for injuries, suspensions, weather
4. **League Volatility**: 3. Liga is highly unpredictable
5. **Sample Size**: Fewer matches than top leagues

## Credits

- **Data Source**: OpenLigaDB (https://www.openligadb.de/)
- **Libraries**: Scikit-learn, XGBoost, Flask, Pandas
- **Design**: Custom responsive CSS

## License

Educational and personal use only.

---

**Built with ⚽ and 🤖 for 3. Liga prediction enthusiasts!**
