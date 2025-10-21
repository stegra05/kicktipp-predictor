from flask import Flask, render_template, jsonify, request
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor

# Optional performance tracker (may not exist yet in refactored structure)
try:
    from kicktipp_predictor.models.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

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


@app.route('/api/performance')
def get_performance():
    """Get performance statistics."""
    try:
        if not TRACKER_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Performance tracker not available'
            })

        stats = tracker.get_current_stats()

        return jsonify({
            'success': True,
            'stats': stats
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

        # Calculate table
        table = data_loader._calculate_table(matches)

        # Sort by position
        sorted_table = sorted(table.items(), key=lambda x: x[1]['position'])

        formatted_table = []
        for team, data in sorted_table:
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
            })

        return jsonify({
            'success': True,
            'table': formatted_table
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



