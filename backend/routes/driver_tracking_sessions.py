"""Routes registre sessions + watermark (plan Kafka-first v5)."""

from __future__ import annotations

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import db, role_required
from models.enums import UserRole
from services.tracking.session_registry import (
    SessionRegistryError,
    close_tracking_session,
    register_tracking_session,
)
from services.tracking.watermark_service import get_persisted_watermark

logger = logging.getLogger(__name__)

tracking_sessions_ns = Namespace(
    "driver_tracking_sessions",
    description="Registre sessions tracking GPS + watermark persisted",
)


def _driver_from_token():
    from routes.driver import get_driver_from_token

    return get_driver_from_token()


@tracking_sessions_ns.route("/me/tracking/sessions")
class DriverTrackingSessions(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        driver, error_response, status_code = _driver_from_token()
        if error_response:
            return error_response, status_code
        body = request.get_json(silent=True) or {}
        try:
            result = register_tracking_session(
                db.session,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                tracking_session_id=str(body.get("tracking_session_id") or ""),
                tracking_session_started_at=body.get("tracking_session_started_at"),
            )
            db.session.commit()
            return result, 200
        except SessionRegistryError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, exc.http_status
        except Exception:
            db.session.rollback()
            logger.exception("[tracking_sessions] register failed")
            return {"error": "internal_error"}, 500


@tracking_sessions_ns.route("/me/tracking/sessions/<string:session_id>/close")
class DriverTrackingSessionClose(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, session_id: str):
        driver, error_response, status_code = _driver_from_token()
        if error_response:
            return error_response, status_code
        body = request.get_json(silent=True) or {}
        final_seq = body.get("final_sequence_id")
        try:
            result = close_tracking_session(
                db.session,
                driver_id=int(driver.id),
                tracking_session_id=session_id,
                final_sequence_id=int(final_seq) if final_seq is not None else None,
            )
            db.session.commit()
            return result, 200
        except SessionRegistryError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, exc.http_status
        except Exception:
            db.session.rollback()
            logger.exception("[tracking_sessions] close failed")
            return {"error": "internal_error"}, 500


@tracking_sessions_ns.route("/me/tracking/watermark")
class DriverTrackingWatermark(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        driver, error_response, status_code = _driver_from_token()
        if error_response:
            return error_response, status_code
        sid = request.args.get("tracking_session_id") or ""
        cursor = request.args.get("cursor")
        try:
            result = get_persisted_watermark(
                db.session,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                tracking_session_id=sid,
                cursor=cursor,
            )
            return result, 200
        except PermissionError:
            return {"error": "tracking_session_forbidden"}, 403
        except ValueError as exc:
            if "watermark_cursor_session_mismatch" in str(exc):
                return {"error": "watermark_cursor_session_mismatch"}, 400
            logger.exception("[tracking_watermark] value error")
            return {"error": "invalid_request"}, 400
        except Exception:
            logger.exception("[tracking_watermark] failed")
            return {"error": "internal_error"}, 500
