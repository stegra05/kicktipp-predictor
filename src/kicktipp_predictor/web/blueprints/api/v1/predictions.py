from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import jsonify, request

from . import v1_bp
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import GoalDifferencePredictor


def _to_iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@v1_bp.get("/predictions")
def get_predictions():
    # Parse query params
    days_param = request.args.get("days")
    matchday_param = request.args.get("matchday")
    days: int = 7
    matchday: int | None = None
    try:
        if days_param is not None and str(days_param).strip() != "":
            days = max(1, int(days_param))
    except Exception:
        days = 7
    try:
        if matchday_param is not None and str(matchday_param).strip() != "":
            matchday = int(matchday_param)
    except Exception:
        matchday = None

    loader = DataLoader()
    predictor = GoalDifferencePredictor()
    # Ensure model is available
    try:
        predictor.load_model()
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Train the model first."}), 503
    except Exception as exc:  # unexpected load errors
        return jsonify({"error": f"Failed to load model: {exc}"}), 500

    # Fetch matches
    if matchday is not None:
        upcoming_matches = loader.fetch_matchday(matchday)
    else:
        upcoming_matches = loader.get_upcoming_matches(days=days)

    if not upcoming_matches:
        return jsonify([]), 200

    # Historical context and features
    current_season = loader.get_current_season()
    historical_matches = loader.fetch_season_matches(current_season)
    features_df = loader.create_prediction_features(upcoming_matches, historical_matches)
    if features_df is None or len(features_df) == 0:
        return jsonify([]), 200

    preds = predictor.predict(features_df)

    def _map(pred: dict[str, Any]) -> dict[str, Any]:
        # Find original match to source date/team if needed
        match = next((m for m in upcoming_matches if m.get("match_id") == pred.get("match_id")), None)
        date_val = None
        if match is not None:
            date_val = match.get("date")
        if date_val is None and "date" in features_df.columns:
            try:
                row = features_df.loc[features_df["match_id"] == pred.get("match_id")].iloc[0]
                date_val = row.get("date")
            except Exception:
                date_val = None
        return {
            "match_id": pred.get("match_id"),
            "home_team": pred.get("home_team"),
            "away_team": pred.get("away_team"),
            "match_date": _to_iso_z(date_val) if isinstance(date_val, datetime) else None,
            "matchday": pred.get("matchday"),
            "predicted_home_score": pred.get("predicted_home_score"),
            "predicted_away_score": pred.get("predicted_away_score"),
            "predicted_result": pred.get("predicted_result"),
            "home_win_probability": pred.get("home_win_probability"),
            "draw_probability": pred.get("draw_probability"),
            "away_win_probability": pred.get("away_win_probability"),
        }

    response = [_map(p) for p in preds]
    return jsonify(response), 200


