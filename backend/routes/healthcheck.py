"""Enhanced healthcheck endpoint with DB and Redis checks."""
from flask import Blueprint, jsonify
from sqlalchemy import text

from ext import db, redis_client

healthcheck_bp = Blueprint("healthcheck", __name__)


@healthcheck_bp.route("/health/detailed")
def detailed_health():
    """Detailed healthcheck with component status
    Returns 200 if all OK, 503 if any component degraded.
    """
    status = {
        "status": "ok",
        "components": {}
    }

    # Check DB
    try:
        db.session.execute(text("SELECT 1")).fetchone()
        status["components"]["database"] = "ok"
    except Exception as e:
        status["components"]["database"] = f"error: {e!s}"
        status["status"] = "degraded"

    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            status["components"]["redis"] = "ok"
        else:
            status["components"]["redis"] = "not_configured"
    except Exception as e:
        status["components"]["redis"] = f"warning: {e!s}"
        # Redis n'est pas critique - on ne dégrade pas le statut

    http_code = 200 if status["status"] == "ok" else 503
    return jsonify(status), http_code
