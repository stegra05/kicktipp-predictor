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
    """Returns predictions for upcoming matches.

    This endpoint provides predictions for upcoming matches based on the trained
    model. It accepts query parameters to specify the prediction window (in
    days) or a specific matchday.

    Query Parameters:
        days (int, optional): The number of days ahead to predict. Defaults to 7.
        matchday (int, optional): The specific matchday to predict. If provided,
            this takes precedence over the 'days' parameter.

    Returns:
        A JSON response containing a list of prediction objects, where each
        object includes details about the match, the predicted score, and the
        outcome probabilities.
    """
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
        mid = pred.get("match_id")
        mday = pred.get("matchday")
        phs = pred.get("predicted_home_score")
        pas = pred.get("predicted_away_score")
        hwp = pred.get("home_win_probability")
        dp = pred.get("draw_probability")
        awp = pred.get("away_win_probability")
        return {
            "match_id": int(mid) if mid is not None else None,
            "home_team": str(pred.get("home_team")) if pred.get("home_team") is not None else None,
            "away_team": str(pred.get("away_team")) if pred.get("away_team") is not None else None,
            "match_date": _to_iso_z(date_val) if isinstance(date_val, datetime) else None,
            "matchday": int(mday) if mday is not None else None,
            "predicted_home_score": int(phs) if phs is not None else None,
            "predicted_away_score": int(pas) if pas is not None else None,
            "predicted_result": pred.get("predicted_result"),
            "home_win_probability": float(hwp) if hwp is not None else None,
            "draw_probability": float(dp) if dp is not None else None,
            "away_win_probability": float(awp) if awp is not None else None,
        }

    response = [_map(p) for p in preds]
    return jsonify(response), 200


