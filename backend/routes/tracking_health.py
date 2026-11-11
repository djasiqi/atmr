"""API santé tracking ops (N3)."""

from __future__ import annotations

from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required

from models import Company, User
from services.tracking.health_engine import run_health_engine_tick

tracking_health_bp = Blueprint("tracking_health", __name__)


@tracking_health_bp.route("/api/v1/companies/me/tracking-health", methods=["GET"])
@jwt_required()
def company_tracking_health():
    from security.portal_scope import authorized_company_id

    identity = get_jwt_identity()
    user = User.query.filter_by(public_id=identity).first() if identity else None
    company_id = authorized_company_id(user) if user is not None else None
    if company_id is None:
        return jsonify({"error": "unauthorized"}), 403
    company = Company.query.get(company_id)
    if not company:
        return jsonify({"error": "company_not_found"}), 404
    result = run_health_engine_tick(company_id=int(company.id))
    return jsonify({"ok": True, "company_id": company.id, **result}), 200
