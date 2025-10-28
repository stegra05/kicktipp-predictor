from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import jsonify, request

from . import v1_bp
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import GoalDifferencePredictor


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@v1_bp.get("/matches/h2h")
def get_h2h():
    t1 = request.args.get("team1_id")
    t2 = request.args.get("team2_id")
    if not t1 or not t2:
        return jsonify({"error": "team1_id and team2_id required"}), 400
    try:
        team1_id = int(t1)
        team2_id = int(t2)
    except Exception:
        return jsonify({"error": "team IDs must be integers"}), 400

    loader = DataLoader()
    season = loader.get_current_season()
    matches = loader.fetch_season_matches(season)
    teams = sorted({m.get("home_team") for m in matches} | {m.get("away_team") for m in matches})
    if (team1_id < 0 or team1_id >= len(teams)) or (team2_id < 0 or team2_id >= len(teams)):
        return jsonify({"error": "Unknown team_id"}), 404
    team1 = teams[team1_id]
    team2 = teams[team2_id]

    recent: list[dict[str, Any]] = []
    t1_w = t2_w = d = 0
    for m in matches:
        if not m.get("is_finished", False):
            continue
        h = m.get("home_team")
        a = m.get("away_team")
        if {h, a} != {team1, team2}:
            continue
        hs = int(m.get("home_score", 0) or 0)
        as_ = int(m.get("away_score", 0) or 0)
        if hs > as_:
            winner = h
        elif hs < as_:
            winner = a
        else:
            winner = None
        if winner is None:
            d += 1
        elif winner == team1:
            t1_w += 1
        else:
            t2_w += 1
        recent.append({
            "date": _iso(m.get("date")),
            "season": season,
            "home_team": h,
            "away_team": a,
            "score": f"{hs}-{as_}",
        })

    return jsonify({
        "summary": {"team1": team1, "team2": team2, "team1_wins": t1_w, "team2_wins": t2_w, "draws": d},
        "recent_matches": sorted(recent, key=lambda r: r.get("date") or "", reverse=True),
    }), 200


@v1_bp.get("/matches/<string:match_id>")
def get_match_detail(match_id: str):
    loader = DataLoader()
    season = loader.get_current_season()
    matches = loader.fetch_season_matches(season)
    match = next((m for m in matches if str(m.get("match_id")) == match_id), None)
    if match is None:
        return jsonify({"error": "match not found"}), 404

    # Prediction
    predictor = GoalDifferencePredictor()
    try:
        predictor.load_model()
    except Exception:
        predictor = None  # allow detail without prediction

    pred_payload: dict[str, Any] | None = None
    try:
        if predictor is not None:
            hist = loader.fetch_season_matches(season)
            feats = loader.create_prediction_features([match], hist)
            if feats is not None and len(feats) > 0:
                pred = predictor.predict(feats)[0]
                pred_payload = {
                    "predicted_home_score": pred.get("predicted_home_score"),
                    "predicted_away_score": pred.get("predicted_away_score"),
                    "predicted_result": pred.get("predicted_result"),
                    "home_win_probability": pred.get("home_win_probability"),
                    "draw_probability": pred.get("draw_probability"),
                    "away_win_probability": pred.get("away_win_probability"),
                }
    except Exception:
        pred_payload = None

    home = match.get("home_team")
    away = match.get("away_team")
    # recent form (last 5)
    def recent_for(team: str) -> list[dict[str, Any]]:
        arr: list[dict[str, Any]] = []
        for m in matches:
            if not m.get("is_finished", False):
                continue
            if m.get("home_team") == team or m.get("away_team") == team:
                hs = int(m.get("home_score", 0) or 0)
                as_ = int(m.get("away_score", 0) or 0)
                at_home = m.get("home_team") == team
                if hs > as_:
                    res = "W" if at_home else "L"
                elif hs < as_:
                    res = "L" if at_home else "W"
                else:
                    res = "D"
                arr.append({
                    "date": _iso(m.get("date")),
                    "result": res,
                    "score": f"{hs}-{as_}" if at_home else f"{as_}-{hs}",
                })
        return sorted(arr, key=lambda r: r.get("date") or "", reverse=True)[:5]

    # h2h summary (current season scope)
    t1_w = t2_w = d = 0
    for m in matches:
        if not m.get("is_finished", False):
            continue
        h = m.get("home_team")
        a = m.get("away_team")
        if {h, a} != {home, away}:
            continue
        hs = int(m.get("home_score", 0) or 0)
        as_ = int(m.get("away_score", 0) or 0)
        if hs == as_:
            d += 1
        elif (hs > as_ and h == home) or (as_ > hs and a == home):
            t1_w += 1
        else:
            t2_w += 1

    return jsonify({
        "prediction": pred_payload,
        "home_team_form": recent_for(home),
        "away_team_form": recent_for(away),
        "h2h": {"summary": {"team1": home, "team2": away, "team1_wins": t1_w, "team2_wins": t2_w, "draws": d}},
    }), 200


