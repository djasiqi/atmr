"""Routes handoff web (quota appareils mobile → session web)."""

from __future__ import annotations

import logging

from flask import request
from flask_restx import Resource

from ext import limiter
from middleware.trace_id import get_trace_id
from models import User
from security.web_handoff_service import (
    WebHandoffError,
    consume_web_handoff_token,
    create_web_handoff_session_response,
    validate_handoff_redirect_path,
)
from services.security.login_origin import validate_login_origin_for_web

logger = logging.getLogger(__name__)


def register_web_handoff_routes(auth_ns) -> None:
    @auth_ns.route("/web-handoff/consume")
    class WebHandoffConsume(Resource):
        @limiter.limit("20 per minute")
        def post(self):
            """Consomme un jeton handoff et crée une session web (cookies HttpOnly)."""
            origin_ok, origin_err = validate_login_origin_for_web()
            if not origin_ok:
                return {
                    "error": origin_err or "origin_not_allowed",
                    "error_code": "origin_not_allowed",
                }, 403

            body = request.get_json(silent=True) or {}
            token = str(body.get("token") or "").strip()
            if not token:
                return {
                    "error": "handoff_token_required",
                    "error_code": "handoff_token_required",
                }, 400

            try:
                payload = consume_web_handoff_token(token=token)
            except WebHandoffError as exc:
                status = (
                    401
                    if exc.code.endswith("_expired") or exc.code.endswith("_invalid")
                    else 503
                )
                return {
                    "error": exc.code,
                    "error_code": exc.code,
                    "message": exc.message,
                }, status

            user_id = payload.get("user_id")
            if not user_id:
                return {
                    "error": "handoff_token_invalid",
                    "error_code": "handoff_token_invalid",
                }, 401

            user = User.query.filter_by(id=int(user_id)).first()
            if user is None:
                return {
                    "error": "user_not_found",
                    "error_code": "user_not_found",
                }, 404

            redirect_path = validate_handoff_redirect_path(
                redirect_path=str(payload.get("redirect_path") or "").strip(),
                role=str(payload.get("role") or (user.role.value if user.role else "")),
            )
            if not redirect_path:
                return {
                    "error": "handoff_token_invalid",
                    "error_code": "handoff_token_invalid",
                }, 401

            try:
                from shared.audit_helpers import audit_log

                audit_log(
                    "web_handoff_consumed",
                    "security",
                    user_id=user.id,
                    action_details={"redirect_path": redirect_path},
                )
            except Exception as audit_exc:
                logger.warning("audit web_handoff_consumed: %s", audit_exc)

            return create_web_handoff_session_response(
                user,
                redirect_path=redirect_path,
                trace_id=get_trace_id(),
            )
