# 3. Liga Match Predictor

A hybrid machine learning and statistical predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

## Getting Started

Follow these steps to set up and run the predictor.

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
pip install -r requirements.txt
```

### 2. System Verification
Run the demo script to ensure all components are working correctly. This will test data fetching and feature engineering.
```bash
python demo.py
```

### 3. Initial Model Training
Train the prediction models using historical data. This process may take a few minutes.
```bash
python train_model.py
```

### 4. Generate Predictions
Generate predictions for the upcoming matches. The `--record` flag saves the predictions for performance tracking.
```bash
python predict.py --record
```

### 5. View in Browser
Start the web interface to see the predictions, league table, and performance statistics.
```bash
python src/web/app.py
```
Open your browser and navigate to `http://localhost:5000`.

## Usage

### Weekly Workflow
This is the recommended weekly routine for generating predictions.

1.  **Update Results:** After a matchday is complete, update the performance tracker with the actual scores.
    ```bash
    python predict.py --update-results
    ```

2.  **Generate New Predictions:** Generate and save predictions for the next matchday.
    ```bash
    python predict.py --record
    ```

3.  **View Predictions:** Use the web interface to view the latest predictions.
    ```bash
    python src/web/app.py
    ```

### Monthly Maintenance
To keep the models accurate, retrain them monthly with the latest match data.
```bash
python train_model.py
```

## Command-Line Scripts

The primary interface for the predictor is through the command-line scripts.

### `demo.py`
Tests the data fetching and feature engineering components to verify that the system is correctly installed and configured.
```bash
python demo.py
```

### `train_model.py`
Trains the machine learning and statistical models on historical data. This script fetches the last three seasons of data, engineers features, and saves the trained models to the `data/models` directory.
```bash
python train_model.py
```

### `predict.py`
Generates predictions for upcoming matches.
```bash
python predict.py [OPTIONS]
```
**Options:**
-   `--matchday <number>`: Predict a specific matchday. If not provided, predicts matches for the next 7 days.
-   `--days <number>`: The number of days ahead to look for upcoming matches (default: 7).
-   `--record`: Save the generated predictions to track performance over time.
-   `--update-results`: Update the performance tracker with the actual results from the latest matches.

### `evaluate_model.py`
Provides a comprehensive evaluation of the model's performance on a held-out test set of historical data. This script outputs detailed metrics, including accuracy, points per match, and performance by confidence level.
```bash
python evaluate_model.py
```

### `evaluate_season.py`
Evaluates the predictor's performance across the entire current season. It generates predictions for each matchday and compares them against the actual results.
```bash
python evaluate_season.py
```

## Prediction Strategy
The model uses a single, highly optimized prediction strategy that is not user-configurable. This strategy is designed to maximize points in fantasy football leagues by balancing the probabilities of exact scores, correct goal differences, and correct match outcomes. The model automatically applies a "safe" strategy for low-confidence predictions to minimize risk.

## Features
- **Advanced Hybrid Prediction Model**: Combines XGBoost ML models with Dixon-Coles enhanced Poisson models
- **Probability Calibration**: Isotonic regression calibration for ML expected goals, blended outcome probabilities
- **Enhanced Confidence Metrics**: Margin-based confidence calculation with entropy-based alternatives
- **Advanced Feature Engineering**: 80+ features including momentum, strength of schedule, rest days, and goal patterns
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis by outcome/confidence
- **Performance Tracking**: Automatic tracking of prediction accuracy and points earned
- **Web Interface**: Clean, responsive web UI to view predictions, league table, and statistics
- **Data Scraping**: Automatic fetching of match data from OpenLigaDB API

## Web Interface
The web interface provides a user-friendly way to view the predictions and other relevant information. To start the web interface, run:
```bash
python src/web/app.py
```
The interface includes the following pages:
-   **Predictions**: Displays the upcoming matches with their predicted scores, outcome probabilities, and confidence levels.
-   **League Table**: Shows the current 3. Liga league table.
-   **Statistics**: Provides an overview of the model's performance, including average points per match and accuracy.

## Project Structure
```
kicktipp-predictor/
├── src/
│   ├── scraper/
│   ├── features/
│   ├── models/
│   └── web/
├── data/
│   ├── cache/
│   ├── models/
│   └── predictions/
├── experiments/
├── config/
├── train_model.py
├── predict.py
├── evaluate_model.py
├── evaluate_season.py
├── demo.py
├── requirements.txt
└── README.md
```

## Troubleshooting
-   **"No trained models found"**: Run `python train_model.py` first.
-   **"Module not found"**: Ensure dependencies are installed with `pip install -r requirements.txt`.
-   **"Not enough historical data"**: The API might be temporarily unavailable. Try again later or check your internet connection.
-   **Web interface not loading**: Check if port 5000 is in use. Try accessing via `http://127.0.0.1:5000`.

## Disclaimer
This predictor is for entertainment purposes only. Football is unpredicted and no model can guarantee accurate predictions.
