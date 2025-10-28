from __future__ import annotations

from flask import jsonify, request

from . import v1_bp
from kicktipp_predictor.data import DataLoader


@v1_bp.get("/teams/<int:team_id>/recent-matches")
def get_team_recent_matches(team_id: int):
    """Returns the recent matches for a specific team.

    This endpoint provides a list of recent matches for a given team, including
    the opponent, location (home/away), result, and score.

    Args:
        team_id: The ID of the team to retrieve recent matches for.

    Query Parameters:
        limit (int, optional): The maximum number of recent matches to return.
            Defaults to 5.

    Returns:
        A JSON response containing a list of recent match objects.
    """
    limit_param = request.args.get("limit")
    try:
        limit = max(1, int(limit_param)) if limit_param else 5
    except Exception:
        limit = 5

    loader = DataLoader()
    season = loader.get_current_season()
    matches = loader.fetch_season_matches(season)
    # We don't have stable numeric IDs in data; fallback to matching by position or name hash is unsafe.
    # As a minimal viable approach, interpret team_id as an index based on alphabetical ordering.
    teams = sorted({m.get("home_team") for m in matches} | {m.get("away_team") for m in matches})
    if not teams or team_id < 0 or team_id >= len(teams):
        return jsonify({"error": "Unknown team_id"}), 404
    team_name = teams[team_id]

    recent = []
    for m in matches:
        if not m.get("is_finished", False):
            continue
        if m.get("home_team") == team_name or m.get("away_team") == team_name:
            loc = "home" if m.get("home_team") == team_name else "away"
            hs = int(m.get("home_score", 0) or 0)
            as_ = int(m.get("away_score", 0) or 0)
            if hs > as_:
                res = "H"
            elif hs < as_:
                res = "A"
            else:
                res = "D"
            if loc == "home":
                result_symbol = "W" if res == "H" else ("L" if res == "A" else "D")
                score = f"{hs}-{as_}"
                opponent = m.get("away_team")
            else:
                result_symbol = "W" if res == "A" else ("L" if res == "H" else "D")
                score = f"{as_}-{hs}"
                opponent = m.get("home_team")
            recent.append({
                "opponent": opponent,
                "location": loc,
                "result": result_symbol,
                "score": score,
            })
    # sort by most recent; data has 'date'
    try:
        recent = sorted(recent, key=lambda r: next(
            (m.get("date") for m in matches if (m.get("home_team") == team_name and m.get("away_team") == r["opponent"]) or (m.get("away_team") == team_name and m.get("home_team") == r["opponent"]) ),
            ), reverse=True)
    except Exception:
        pass
    return jsonify(recent[:limit]), 200


