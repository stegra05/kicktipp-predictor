from __future__ import annotations

from collections import defaultdict
from typing import Any

from flask import jsonify, request

from . import v1_bp
from kicktipp_predictor.data import DataLoader


@v1_bp.get("/league/table")
def get_league_table():
    """Returns the league table for a given season.

    This endpoint provides the current league table, including points, wins,
    draws, losses, and goal difference for each team. It accepts an optional
    'season' query parameter to retrieve the table for a specific season.

    Query Parameters:
        season (int, optional): The season year to retrieve the table for.
            Defaults to the current season.

    Returns:
        A JSON response containing a list of team objects, where each object
        represents a row in the league table.
    """
    season_param = request.args.get("season")
    season: int | None = None
    try:
        if season_param:
            season = int(season_param)
    except Exception:
        season = None

    loader = DataLoader()
    if season is None:
        season = loader.get_current_season()
    matches = loader.fetch_season_matches(season)
    if not matches:
        return jsonify([]), 200

    table: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "team_name": None,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "points": 0,
        }
    )

    for m in matches:
        home = str(m.get("home_team"))
        away = str(m.get("away_team"))
        hs = int(m.get("home_score", 0) or 0)
        as_ = int(m.get("away_score", 0) or 0)
        finished = bool(m.get("is_finished", False))
        if not finished:
            continue
        th = table[home]
        ta = table[away]
        th["team_name"] = home
        ta["team_name"] = away
        th["played"] += 1
        ta["played"] += 1
        th["goals_for"] += hs
        th["goals_against"] += as_
        ta["goals_for"] += as_
        ta["goals_against"] += hs
        if hs > as_:
            th["wins"] += 1
            ta["losses"] += 1
            th["points"] += 3
        elif hs < as_:
            ta["wins"] += 1
            th["losses"] += 1
            ta["points"] += 3
        else:
            th["draws"] += 1
            ta["draws"] += 1
            th["points"] += 1
            ta["points"] += 1

    # compute goal diff and sort
    rows = []
    for team, rec in table.items():
        rec["goal_diff"] = int(rec["goals_for"] - rec["goals_against"])
        rows.append(rec)
    rows.sort(key=lambda r: (r["points"], r["goal_diff"], r["goals_for"]), reverse=True)
    # add positions
    for i, rec in enumerate(rows, start=1):
        rec["position"] = i
    return jsonify(rows), 200


