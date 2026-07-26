# routes/institution_timeline.py
"""API timeline transport institution."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import sentry_sdk
from flask import request
from flask_jwt_extended import jwt_required
from flask_jwt_extended.exceptions import JWTExtendedException
from flask_restx import Namespace, Resource, fields
from jwt.exceptions import PyJWTError

from routes.api_error_models import (
    create_not_found_error_model,
    create_permission_error_model,
)
from routes.institution_requests import get_institution_read_context
from services.institutions.transport_timeline_service import list_timeline_events

logger = logging.getLogger(__name__)

institution_timeline_ns = Namespace(
    "institution_timeline",
    description="Timeline immutable des transports institution",
)

not_found_error_model = create_not_found_error_model(institution_timeline_ns)
permission_error_model = create_permission_error_model(institution_timeline_ns)

timeline_event_model = institution_timeline_ns.model(
    "TransportTimelineEvent",
    {
        "id": fields.Integer(),
        "event_type": fields.String(),
        "label": fields.String(),
        "transport_request_id": fields.Integer(),
        "booking_id": fields.Integer(),
        "payload": fields.Raw(),
        "payload_version": fields.Integer(),
        "source_event_id": fields.Integer(),
        "created_at": fields.String(),
    },
)


def _reraise_auth_errors(exc: Exception) -> None:
    """Ne pas transformer les erreurs JWT/auth en 500 ni les remonter à Sentry."""
    if isinstance(exc, (JWTExtendedException, PyJWTError)):
        raise exc
    if hasattr(exc, "code"):
        raise exc
    lowered = str(exc).lower()
    if "signature has expired" in lowered or "token has expired" in lowered:
        raise exc


def _parse_date(name: str) -> datetime | None:
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _timeline_response(events: list[Any]) -> dict[str, Any]:
    serialized = [e.serialize() for e in events]
    next_cursor = serialized[-1]["id"] if serialized else None
    return {
        "events": serialized,
        "count": len(serialized),
        "next_cursor": next_cursor,
    }


@institution_timeline_ns.route("/requests/<int:request_id>/timeline")
class RequestTimeline(Resource):
    @institution_timeline_ns.doc(security="BearerAuth")
    @jwt_required()
    def get(self, request_id: int):
        try:
            institution_id, _user_id, _role = get_institution_read_context()
            from models import TransportRequest

            tr = TransportRequest.query.filter_by(
                id=request_id, institution_id=institution_id
            ).first()
            if not tr:
                return {"error": "Demande non trouvée"}, 404

            limit = min(int(request.args.get("limit", 200)), 500)
            cursor = request.args.get("cursor")
            cursor_id = int(cursor) if cursor else None

            events = list_timeline_events(
                institution_id=institution_id,
                transport_request_id=request_id,
                date_from=_parse_date("from"),
                date_to=_parse_date("to"),
                limit=limit,
                cursor_id=cursor_id,
            )
            return _timeline_response(events), 200
        except Exception as e:
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            logger.exception("[Timeline] GET request %s: %s", request_id, e)
            return {"error": "Erreur serveur"}, 500


@institution_timeline_ns.route("/bookings/<int:booking_id>/timeline")
class BookingTimeline(Resource):
    @institution_timeline_ns.doc(security="BearerAuth")
    @jwt_required()
    def get(self, booking_id: int):
        try:
            institution_id, _user_id, _role = get_institution_read_context()
            from services.institutions.booking_change_service import (
                resolve_institution_booking,
            )

            ctx = resolve_institution_booking(institution_id, booking_id)
            if not ctx:
                return {"error": "Réservation non trouvée"}, 404

            limit = min(int(request.args.get("limit", 200)), 500)
            cursor = request.args.get("cursor")
            cursor_id = int(cursor) if cursor else None

            events = list_timeline_events(
                institution_id=institution_id,
                booking_id=booking_id,
                date_from=_parse_date("from"),
                date_to=_parse_date("to"),
                limit=limit,
                cursor_id=cursor_id,
            )
            return _timeline_response(events), 200
        except Exception as e:
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            logger.exception("[Timeline] GET booking %s: %s", booking_id, e)
            return {"error": "Erreur serveur"}, 500


@institution_timeline_ns.route("/patients/<int:patient_id>/transport-history")
class PatientTransportHistory(Resource):
    @institution_timeline_ns.doc(security="BearerAuth")
    @jwt_required()
    def get(self, patient_id: int):
        try:
            institution_id, _user_id, _role = get_institution_read_context()
            from models import InstitutionPatient

            patient = InstitutionPatient.query.filter_by(
                id=patient_id, institution_id=institution_id
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            limit = min(int(request.args.get("limit", 200)), 500)
            cursor = request.args.get("cursor")
            cursor_id = int(cursor) if cursor else None

            events = list_timeline_events(
                institution_id=institution_id,
                patient_id=patient_id,
                date_from=_parse_date("from"),
                date_to=_parse_date("to"),
                limit=limit,
                cursor_id=cursor_id,
            )
            return _timeline_response(events), 200
        except Exception as e:
            _reraise_auth_errors(e)
            sentry_sdk.capture_exception(e)
            logger.exception("[Timeline] GET patient %s: %s", patient_id, e)
            return {"error": "Erreur serveur"}, 500
