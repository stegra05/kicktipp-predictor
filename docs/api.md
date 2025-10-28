### Kicktipp Predictor API (v1)

Base URL: `http://localhost:8000/api/v1`

Run locally:

```
export FLASK_DEBUG=1
python app.py
```

Health endpoint (not versioned): `GET /health`

---

### Status

GET `/status`

- Description: Simple health/status check for monitoring.
- Response 200 JSON:

```
{ "api": "v1", "status": "ok", "model_loaded": true, "version": "4.0.0a2", "config": { "debug": true, "testing": false } }
```

---

### 1) Core Predictions

GET `/predictions`

- Description: Fetch predictions for upcoming matches.
- Query params:
  - `days` (int, optional, default 7): number of days ahead
  - `matchday` (int, optional): specific matchday overrides `days`
- Response 200 JSON (array of predictions):

```
[
  {
    "match_id": "bl3_2024_15_12345",
    "home_team": "Team A",
    "away_team": "Team B",
    "match_date": "2025-03-15T14:30:00Z",
    "matchday": 15,
    "predicted_home_score": 2,
    "predicted_away_score": 1,
    "predicted_result": "H",
    "home_win_probability": 0.48,
    "draw_probability": 0.28,
    "away_win_probability": 0.24
  }
]
```

Errors:
- 503 when model not found: `{ "error": "Model not found. Train the model first." }`

---

### 2) League & Team Data

GET `/league/table`

- Description: Current league table from finished matches.
- Query params:
  - `season` (int, optional): defaults to current season
- Response 200 JSON (array):

```
[{ "position": 1, "team_name": "FC Leader", "points": 32, "wins": 10, "draws": 2, "losses": 3, "played": 15, "goals_for": 28, "goals_against": 12, "goal_diff": 16 }]
```

GET `/teams/{team_id}/recent-matches`

- Description: Recent match results for a specific team.
- URL params:
  - `team_id` (int, required): temporary mapping via alphabetical index of team names
- Query params:
  - `limit` (int, optional, default 5)
- Response 200 JSON (array):

```
[{ "opponent": "Opponent X", "location": "home", "result": "W", "score": "2-0" }]
```

Errors:
- 404 unknown team: `{ "error": "Unknown team_id" }`

GET `/matches/h2h`

- Description: Head-to-head summary between two teams (current season scope).
- Query params:
  - `team1_id` (int, required)
  - `team2_id` (int, required)
- Response 200 JSON:

```
{
  "summary": { "team1": "Team A", "team2": "Team B", "team1_wins": 5, "team2_wins": 2, "draws": 3 },
  "recent_matches": [ { "season": 2024, "date": "2024-10-01T12:00:00Z", "home_team": "Team A", "away_team": "Team B", "score": "1-1" } ]
}
```

Errors:
- 400 missing params or invalid types
- 404 unknown team IDs

GET `/matches/{match_id}`

- Description: Match detail bundle with prediction, home/away recent form (last 5), and H2H summary.
- URL params:
  - `match_id` (string, required)
- Response 200 JSON:

```
{
  "prediction": { "predicted_home_score": 2, "predicted_away_score": 1, "predicted_result": "H", "home_win_probability": 0.48, "draw_probability": 0.28, "away_win_probability": 0.24 },
  "home_team_form": [ { "date": "2025-03-01T12:00:00Z", "result": "W", "score": "2-0" } ],
  "away_team_form": [ { "date": "2025-03-07T12:00:00Z", "result": "L", "score": "0-1" } ],
  "h2h": { "summary": { "team1": "Team A", "team2": "Team B", "team1_wins": 5, "team2_wins": 2, "draws": 3 } }
}
```

Errors:
- 404 if match not found

---

### 3) Model Performance & Evaluation

GET `/evaluation/summary`

- Description: Overall season metrics loaded from `data/predictions/metrics_season.json`.
- Response 200 JSON:

```
{ "season": 2024, "accuracy": 0.491, "avg_points_per_game": 1.45 }
```

GET `/evaluation/matchdays`

- Description: Per-matchday metrics loaded from `data/predictions/per_matchday_metrics_season.csv`.
- Response 200 JSON (array):

```
[{ "matchday": 1, "n": 9, "avg_points": 1.56, "accuracy": 0.556 }]
```

GET `/evaluation/performance-trends`

- Description: Placeholder for dynamic trend queries on historical predictions.
- Query params:
  - `team_id` (int, optional)
  - `min_prob` (float, optional)
  - `max_prob` (float, optional)
- Response 501 JSON:

```
{ "error": "performance-trends not implemented", "params": { "team_id": "..", "min_prob": "..", "max_prob": ".." } }
```

---

### 4) Model & System Transparency

GET `/model/features`

- Description: Returns feature names from model metadata (`metadata.joblib`).
- Response 200 JSON:

```
{ "model_version": "goal_diff_regressor.joblib", "feature_count": 52, "features": ["tanh_tamed_elo", "weighted_form_points_difference", "..."] }
```

Errors:
- 404 if metadata missing

---

### 5) Administration

POST `/admin/retrain`

- Description: Starts a retraining job (stubbed: returns a generated job_id).
- Response 202 JSON:

```
{ "status": "Training job started", "job_id": "train-20251028-abcdef" }
```

---

### Notes

- `team_id` is a temporary alphabetical index over team names for the current season and should be replaced by stable team identifiers in a future iteration.
- Responses may omit fields when data is unavailable. Treat unspecified fields as null/absent.

