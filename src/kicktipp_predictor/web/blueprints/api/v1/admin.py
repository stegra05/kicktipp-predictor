from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from flask import jsonify

from . import v1_bp


@v1_bp.post("/admin/retrain")
def admin_retrain():
    # Minimal stub: in real deployment, enqueue background job
    job_id = f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    return jsonify({"status": "Training job started", "job_id": job_id}), 202


