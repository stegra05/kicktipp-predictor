from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from flask import jsonify

from . import v1_bp


@v1_bp.post("/admin/retrain")
def admin_retrain():
    """Triggers a retraining of the prediction model.

    This endpoint is intended for administrative purposes and simulates the
    start of a background job to retrain the model. In a real-world
    application, this would enqueue a task in a job queue like Celery or RQ.

    Returns:
        A JSON response indicating that the training job has started, along
        with a unique job ID.
    """
    # Minimal stub: in real deployment, enqueue background job
    job_id = f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    return jsonify({"status": "Training job started", "job_id": job_id}), 202


