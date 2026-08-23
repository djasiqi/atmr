"""Routes session web institution — heartbeat activité humaine."""

from __future__ import annotations

import logging

from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields

from ext import db
from repositories.user_repository import UserRepository
from security.web_session_service import (
    JWT_SID_CLAIM,
    extract_sid_from_claims,
    is_institution_user,
    record_interactive_activity,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

user_repo = UserRepository()


def register_web_session_routes(auth_ns: Namespace) -> None:
    session_activity_model = auth_ns.model(
        "SessionActivityResponse",
        {
            "ok": fields.Boolean(required=True),
            "updated": fields.Boolean(required=True),
            "sid": fields.String(description="Session web (sid)"),
        },
    )

    @auth_ns.route("/session-activity")
    class SessionActivity(Resource):
        @jwt_required()
        @auth_ns.response(200, "Activité enregistrée", session_activity_model)
        @auth_ns.response(401, "Session invalide")
        def post(self):
            """Heartbeat activité humaine — met à jour last_interactive_activity_at."""
            claims = get_jwt() or {}
            sid = extract_sid_from_claims(claims)
            if not sid:
                err_resp, status = APIErrorHandler.handle_permission_error(
                    "Session web requise",
                    logger_instance=logger,
                )
                return err_resp, status

            identity = get_jwt_identity()
            user = user_repo.find_model_by_public_id(str(identity)) if identity else None
            if user is None or not is_institution_user(user):
                err_resp, status = APIErrorHandler.handle_permission_error(
                    "Compte institution requis",
                    logger_instance=logger,
                )
                return err_resp, status

            updated, error_code = record_interactive_activity(
                sid,
                user_id=user.id,
                min_interval_seconds=30,
            )
            if error_code in {"session_not_found", "session_user_mismatch"}:
                return APIErrorHandler.handle_permission_error(
                    "Session invalide",
                    logger_instance=logger,
                )
            if error_code == "session_revoked":
                return (
                    {"error": "session_revoked", "error_code": "session_revoked"},
                    401,
                )
            if error_code == "session_expired":
                return (
                    {"error": "session_expired", "error_code": "session_expired"},
                    401,
                )

            try:
                db.session.commit()
            except Exception as exc:
                db.session.rollback()
                logger.error("session-activity commit failed: %s", exc)
                return {"error": "service_unavailable"}, 503

            return {"ok": True, "updated": bool(updated), JWT_SID_CLAIM: sid}, 200
