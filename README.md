# 3. Liga Match Predictor

A hybrid machine learning and statistical predictor for Germany's 3. Bundesliga football matches, optimized for maximizing prediction points in fantasy football leagues.

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/kicktipp-predictor.git
cd kicktipp-predictor
pip install -r requirements.txt
```

### 2. Train Models
Train the prediction models with historical data:
```bash
python train_model.py
```

### 3. Generate Predictions
Generate predictions for upcoming matches:
```bash
python predict.py --record
```

### 4. View in Browser
Start the web interface to see the predictions:
```bash
python src/web/app.py
```
Open your browser to `http://localhost:5000`.

## Usage

### Weekly Workflow
1.  **Update results from the last matchday:**
    ```bash
    python predict.py --update-results
    ```
2.  **Generate new predictions:**
    ```bash
    python predict.py --record
    ```
3.  **View the predictions in the web interface:**
    ```bash
    python src/web/app.py
    ```

### Monthly Maintenance
Retrain the models to incorporate the latest data:
```bash
python train_model.py
```

## How to Use

The following sections provide detailed information on how to use the different features of the predictor.

### Generating Predictions
The `predict.py` script has several options to customize your predictions:
-   `--matchday`: Predict a specific matchday.
-   `--strategy`: Choose a prediction strategy (`balanced`, `conservative`, `aggressive`, `safe`).
-   `--record`: Record predictions for performance tracking.
-   `--update-results`: Update previous predictions with actual results.

### Evaluating Model Performance
-   **`evaluate_model.py`**: Provides a comprehensive evaluation of the model's performance.
-   **`evaluate_season.py`**: Evaluates the predictor's performance over the entire current season.
-   **`diagnose_model.py`**: Provides detailed insights into model behavior, calibration, and prediction patterns.

### Tuning Hyperparameters
The `tune_hyperparameters.py` script allows you to optimize the model's weights using cross-validation.

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
├── diagnose_model.py
├── tune_hyperparameters.py
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
