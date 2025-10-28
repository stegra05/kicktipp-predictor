from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import jsonify, request

from . import v1_bp


DATA_DIR = Path(__file__).resolve().parents[7] / "data" / "predictions"


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@v1_bp.get("/evaluation/summary")
def evaluation_summary():
    metrics_path = DATA_DIR / "metrics_season.json"
    data = _load_json(metrics_path)
    if not data:
        return jsonify({}), 200
    return jsonify(data), 200


@v1_bp.get("/evaluation/matchdays")
def evaluation_matchdays():
    # CSV to JSON minimal loader
    path = DATA_DIR / "per_matchday_metrics_season.csv"
    if not path.exists():
        return jsonify([]), 200
    rows: list[dict[str, Any]] = []
    try:
        import csv

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append({
                        "matchday": int(r.get("matchday", "0") or 0),
                        "n": int(r.get("n", "0") or 0),
                        "avg_points": float(r.get("avg_points", "0") or 0.0),
                        "accuracy": float(r.get("accuracy", "0") or 0.0),
                    })
                except Exception:
                    continue
    except Exception:
        return jsonify([]), 200
    rows.sort(key=lambda x: x.get("matchday", 0))
    return jsonify(rows), 200


@v1_bp.get("/evaluation/performance-trends")
def evaluation_performance_trends():
    # Placeholder: depends on detailed historical prediction dataset
    # Return 501 to indicate not yet implemented, but keep contract
    params = {
        "team_id": request.args.get("team_id"),
        "min_prob": request.args.get("min_prob"),
        "max_prob": request.args.get("max_prob"),
    }
    return jsonify({"error": "performance-trends not implemented", "params": params}), 501


