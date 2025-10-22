from flask import Flask, render_template, jsonify, request
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor

app = Flask(__name__)

# Initialize components
data_loader = DataLoader()
predictor = MatchPredictor()

# Try to load trained models on startup; cache readiness flag.
# This prevents 500s when endpoints are hit before training.
try:
    predictor.load_models()
    MODELS_READY = True
except FileNotFoundError:
    MODELS_READY = False


def ensure_models_loaded() -> bool:
    """Ensure ML models are loaded; lazily retry once per process.

    Returns True if models are ready, False otherwise.
    """
    global MODELS_READY
    if not MODELS_READY:
        try:
            predictor.load_models()
            MODELS_READY = True
        except FileNotFoundError:
            MODELS_READY = False
    return MODELS_READY


@app.route('/')
def index():
    """Main page showing upcoming predictions."""
    return render_template('index.html')


@app.route('/api/upcoming_predictions')
def get_upcoming_predictions():
    """Get predictions for upcoming matches."""
    try:
        # Ensure models are available; return friendly message instead of 500
        if not ensure_models_loaded():
            return jsonify({
                'success': False,
                'error': 'Models not trained. Run training command first.'
            })

        days = request.args.get('days', 7, type=int)

        # Get upcoming matches
        upcoming_matches = data_loader.get_upcoming_matches(days=days)

        if not upcoming_matches:
            return jsonify({
                'success': True,
                'message': 'No upcoming matches in the next {} days'.format(days),
                'predictions': []
            })

        # Get historical data for features
        current_season = data_loader.get_current_season()
        historical_matches = data_loader.fetch_season_matches(current_season)

        # Create features
        features_df = data_loader.create_prediction_features(
            upcoming_matches, historical_matches
        )

        # Guard against empty features
        if features_df is None or len(features_df) == 0:
            return jsonify({
                'success': True,
                'message': 'Not enough historical data to generate features',
                'predictions': []
            })

        # Get predictions
        predictions = predictor.predict(features_df)

        # Format for JSON
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            match = upcoming_matches[i]
            formatted_predictions.append({
                'match_id': int(pred['match_id']) if pred.get('match_id') is not None else None,
                'date': match['date'].strftime('%Y-%m-%d %H:%M'),
                'matchday': int(match['matchday']) if match.get('matchday') is not None else None,
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'predicted_score': f"{pred['predicted_home_score']}:{pred['predicted_away_score']}",
                'predicted_home_score': int(pred['predicted_home_score']),
                'predicted_away_score': int(pred['predicted_away_score']),
                'home_win_probability': float(round(float(pred['home_win_probability']) * 100, 1)),
                'draw_probability': float(round(float(pred['draw_probability']) * 100, 1)),
                'away_win_probability': float(round(float(pred['away_win_probability']) * 100, 1)),
                'confidence': float(round(float(pred['confidence']) * 100, 1)),
                'home_expected_goals': float(round(float(pred['home_expected_goals']), 2)),
                'away_expected_goals': float(round(float(pred['away_expected_goals']), 2)),
            })

        return jsonify({
            'success': True,
            'predictions': formatted_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/current_matchday')
def get_current_matchday():
    """Get predictions for current matchday."""
    try:
        # Ensure models are available; return friendly message instead of 500
        if not ensure_models_loaded():
            return jsonify({
                'success': False,
                'error': 'Models not trained. Run training command first.'
            })

        current_season = data_loader.get_current_season()
        current_matchday = data_loader.get_current_matchday(current_season)

        # Get matchday matches
        matchday_matches = data_loader.fetch_matchday(current_matchday, current_season)

        # Filter upcoming matches
        upcoming = [m for m in matchday_matches if not m['is_finished']]

        if not upcoming:
            return jsonify({
                'success': True,
                'message': f'No upcoming matches in matchday {current_matchday}',
                'predictions': [],
                'matchday': current_matchday
            })

        # Get historical data
        historical_matches = data_loader.fetch_season_matches(current_season)

        # Create features
        features_df = data_loader.create_prediction_features(
            upcoming, historical_matches
        )

        # Guard against empty features
        if features_df is None or len(features_df) == 0:
            return jsonify({
                'success': True,
                'message': 'Not enough historical data to generate features',
                'predictions': [],
                'matchday': current_matchday
            })

        # Get predictions
        predictions = predictor.predict(features_df)

        # Format predictions
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            match = upcoming[i]
            formatted_predictions.append({
                'match_id': int(pred['match_id']) if pred.get('match_id') is not None else None,
                'date': match['date'].strftime('%Y-%m-%d %H:%M'),
                'matchday': int(match['matchday']) if match.get('matchday') is not None else None,
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'predicted_score': f"{pred['predicted_home_score']}:{pred['predicted_away_score']}",
                'predicted_home_score': int(pred['predicted_home_score']),
                'predicted_away_score': int(pred['predicted_away_score']),
                'home_win_probability': float(round(float(pred['home_win_probability']) * 100, 1)),
                'draw_probability': float(round(float(pred['draw_probability']) * 100, 1)),
                'away_win_probability': float(round(float(pred['away_win_probability']) * 100, 1)),
                'confidence': float(round(float(pred['confidence']) * 100, 1)),
            })

        return jsonify({
            'success': True,
            'matchday': current_matchday,
            'predictions': formatted_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/table')
def get_table():
    """Get current league table."""
    try:
        current_season = data_loader.get_current_season()
        matches = data_loader.fetch_season_matches(current_season)
        finished_matches = [m for m in matches if m.get('is_finished')]

        # Calculate table
        table = data_loader._calculate_table(finished_matches)

        # Calculate EWMA and Form features
        ewma_df = data_loader._compute_ewma_recency_features(finished_matches, span=5)

        # Get the latest EWMA points for each team
        latest_ewma = ewma_df.sort_values('date').groupby('team').last()

        formatted_table = []
        for team, data in table.items():
            # Get Form
            team_history = [m for m in finished_matches if team in (m['home_team'], m['away_team'])]
            form_features = data_loader._get_form_features(team, team_history, 'team', last_n=5)

            form_string = ""
            recent_matches = sorted(team_history, key=lambda x: x['date'], reverse=True)[:5]
            for match in reversed(recent_matches):
                if match['home_team'] == team:
                    if match['home_score'] > match['away_score']:
                        form_string += 'W'
                    elif match['home_score'] < match['away_score']:
                        form_string += 'L'
                    else:
                        form_string += 'D'
                else: # away team
                    if match['away_score'] > match['home_score']:
                        form_string += 'W'
                    elif match['away_score'] < match['home_score']:
                        form_string += 'L'
                    else:
                        form_string += 'D'


            # Get EWMA points
            ewma_points = latest_ewma.loc[team]['points_ewm5'] if team in latest_ewma.index else 0.0

            formatted_table.append({
                'position': data['position'],
                'team': team,
                'played': data['played'],
                'won': data['won'],
                'drawn': data['drawn'],
                'lost': data['lost'],
                'goals_for': data['goals_for'],
                'goals_against': data['goals_against'],
                'goal_difference': data['goals_for'] - data['goals_against'],
                'points': data['points'],
                'form_last_5': form_string,
                'ewma_points': round(ewma_points, 2),
            })

        # Sort by position
        sorted_table = sorted(formatted_table, key=lambda x: x['position'])

        return jsonify({
            'success': True,
            'table': sorted_table
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/statistics')
def statistics():
    """Statistics page."""
    return render_template('statistics.html')


@app.route('/table')
def table():
    """League table page."""
    return render_template('table.html')


import json
import os

@app.route('/api/model_quality')
def get_model_quality():
    """Get model quality metrics."""
    try:
        metrics_path = os.path.join('data', 'predictions', 'metrics.json')
        if not os.path.exists(metrics_path):
            return jsonify({'success': False, 'error': 'Metrics file not found. Run evaluation first.'})

        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Extract the main metrics
        main_metrics = metrics_data.get('main', {})

        metrics = {
            'brier_score': main_metrics.get('brier'),
            'log_loss': main_metrics.get('log_loss'),
            'rps': main_metrics.get('rps'),
            'avg_points': main_metrics.get('avg_points'),
            'total_points': main_metrics.get('total_points'),
            'accuracy': main_metrics.get('accuracy'),
        }
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/match/<int:match_id>')
def match_detail(match_id):
    """Match detail page."""
    return render_template('match_detail.html', match_id=match_id)


@app.route('/api/match/<int:match_id>')
def get_match_detail(match_id):
    """Get details for a single match."""
    try:
        if not ensure_models_loaded():
            return jsonify({
                'success': False,
                'error': 'Models not trained. Run training command first.'
            })

        # Fetch the specific match
        # This is a simplified example; you might need a more direct way
        # to fetch a single match by ID if your DataLoader supports it.
        current_season = data_loader.get_current_season()
        all_matches = data_loader.fetch_season_matches(current_season)

        match_data = next((m for m in all_matches if m['match_id'] == match_id), None)

        if not match_data:
            return jsonify({'success': False, 'error': 'Match not found'}), 404

        # For prediction, we need to create features for this single match
        # We pass it as a list to create_prediction_features
        features_df = data_loader.create_prediction_features(
            [match_data], all_matches
        )

        if features_df is None or len(features_df) == 0:
            return jsonify({
                'success': False,
                'error': 'Could not generate features for this match'
            })

        prediction = predictor.predict(features_df)[0]

        # Format the detailed prediction
        formatted_prediction = {
            'match_id': int(prediction['match_id']),
            'date': match_data['date'].strftime('%Y-%m-%d %H:%M'),
            'matchday': int(match_data['matchday']),
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'predicted_score': f"{prediction['predicted_home_score']}:{prediction['predicted_away_score']}",
            'home_win_probability': float(round(float(prediction['home_win_probability']) * 100, 1)),
            'draw_probability': float(round(float(prediction['draw_probability']) * 100, 1)),
            'away_win_probability': float(round(float(prediction['away_win_probability']) * 100, 1)),
            'confidence': float(round(float(prediction['confidence']) * 100, 1)),
            'home_expected_goals': float(round(float(prediction['home_expected_goals']), 2)),
            'away_expected_goals': float(round(float(prediction['away_expected_goals']), 2)),
            'features': features_df.drop(columns=['match_id', 'date', 'home_team', 'away_team']).iloc[0].to_dict()
        }

        return jsonify({'success': True, 'prediction': formatted_prediction})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

