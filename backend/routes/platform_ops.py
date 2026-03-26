"""Routes Admin Ops / Platform — lecture seule (agrégateur status)."""

from __future__ import annotations

import logging

from flask import current_app, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import limiter, role_required
from models import UserRole
from security.audit_log import AuditLogger
from security.ip_whitelist import ip_whitelist_required
from services.platform_status_aggregator import build_platform_status_payload
from shared.infrastructure.adapters.auth_adapter import get_current_user_via_use_case

logger = logging.getLogger(__name__)

platform_ops_ns = Namespace("platform", description="Admin Ops / Platform (lecture seule)")


@platform_ops_ns.route("/status")
class PlatformStatusResource(Resource):
    """GET /api/v1/platform/status — agrégat prod/demo + liens observabilité."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("120 per hour")  # ~30s polling max théorique
    def get(self):
        try:
            payload = build_platform_status_payload(current_app.config)
        except Exception as e:
            logger.exception("[platform/status] agrégation: %s", e)
            return {"error": "aggregation_failed", "message": str(e)}, 500

        try:
            current_user = get_current_user_via_use_case()
            AuditLogger.log_action(
                action_type="platform_status_read",
                action_category="platform_ops",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "admin",
                result_status="success",
                action_details={},
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception as audit_error:
            logger.warning("[platform/status] audit: %s", audit_error)

        return payload, 200
