# routes/institution_bookings.py
"""Édition opérationnelle et audit des bookings institution."""

from __future__ import annotations

import logging
from typing import Any, cast

import sentry_sdk
from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request
from flask_restx import Namespace, Resource, fields
from marshmallow import EXCLUDE, Schema, ValidationError, validate
from marshmallow import fields as ma_fields

from ext import db
from models.enums import InstitutionRole
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.validation_utils import parse_request_json
from security.audit_log import AuditLogger
from services.institutions.booking_change_service import (
    INSTITUTION_OPERATIONAL_FIELDS,
    assert_operational_role,
    cancel_institution_booking,
    list_change_events,
    resolve_institution_booking,
    update_institution_booking,
)

logger = logging.getLogger(__name__)

institution_bookings_ns = Namespace(
    "institution_bookings",
    description="Édition et audit des transports institution (bookings)",
)

api_error_model = create_api_error_model(institution_bookings_ns)
not_found_error_model = create_not_found_error_model(institution_bookings_ns)
permission_error_model = create_permission_error_model(institution_bookings_ns)
validation_error_model = create_validation_error_model(institution_bookings_ns)

booking_patch_model = institution_bookings_ns.model(
    "InstitutionBookingPatch",
    {
        "version": fields.Integer(required=True, description="Version optimiste"),
        "reason": fields.String(description="Motif (obligatoire si EN_ROUTE critique)"),
        "customer_name": fields.String(),
        "pickup_location": fields.String(),
        "dropoff_location": fields.String(),
        "scheduled_time": fields.String(),
        "appointment_time": fields.String(),
        "return_appointment_time": fields.String(),
        "medical_facility": fields.String(),
        "hospital_service": fields.String(),
        "doctor_name": fields.String(),
        "pickup_floor": fields.String(),
        "pickup_door_code": fields.String(),
        "dropoff_floor": fields.String(),
        "dropoff_door_code": fields.String(),
        "pickup_access_notes": fields.String(),
        "dropoff_access_notes": fields.String(),
        "notes_medical": fields.String(),
        "wheelchair_need": fields.Boolean(),
        "wheelchair_client_has": fields.Boolean(),
        "delivery_description": fields.String(),
        "leg_appointments": fields.List(fields.Raw),
    },
)

cancel_model = institution_bookings_ns.model(
    "InstitutionBookingCancel",
    {
        "version": fields.Integer(required=True),
        "reason": fields.String(required=True),
        "reason_code": fields.String(),
    },
)


class InstitutionBookingLegAppointmentSchema(Schema):
    """Heure RDV d’un leg destination — null = « À définir » (clear)."""

    class Meta:
        unknown = EXCLUDE

    index = ma_fields.Integer(required=True)
    scheduled_time = ma_fields.String(required=False, allow_none=True)


class InstitutionBookingPatchSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    version = ma_fields.Integer(required=True)
    reason = ma_fields.String(required=False, allow_none=True)
    customer_name = ma_fields.String(required=False)
    pickup_location = ma_fields.String(required=False)
    dropoff_location = ma_fields.String(required=False)
    pickup_lat = ma_fields.Float(required=False, allow_none=True)
    pickup_lon = ma_fields.Float(required=False, allow_none=True)
    dropoff_lat = ma_fields.Float(required=False, allow_none=True)
    dropoff_lon = ma_fields.Float(required=False, allow_none=True)
    scheduled_time = ma_fields.String(required=False, allow_none=True)
    medical_facility = ma_fields.String(required=False, allow_none=True)
    doctor_name = ma_fields.String(required=False, allow_none=True)
    hospital_service = ma_fields.String(required=False, allow_none=True)
    notes_medical = ma_fields.String(required=False, allow_none=True)
    pickup_access_notes = ma_fields.String(required=False, allow_none=True)
    dropoff_access_notes = ma_fields.String(required=False, allow_none=True)
    pickup_floor = ma_fields.String(required=False, allow_none=True)
    pickup_door_code = ma_fields.String(required=False, allow_none=True)
    dropoff_floor = ma_fields.String(required=False, allow_none=True)
    dropoff_door_code = ma_fields.String(required=False, allow_none=True)
    wheelchair_client_has = ma_fields.Boolean(required=False)
    wheelchair_need = ma_fields.Boolean(required=False)
    mission_type = ma_fields.String(required=False)
    delivery_description = ma_fields.String(required=False, allow_none=True)
    appointment_time = ma_fields.String(required=False, allow_none=True)
    return_appointment_time = ma_fields.String(required=False, allow_none=True)
    leg_appointments = ma_fields.List(
        ma_fields.Nested(InstitutionBookingLegAppointmentSchema),
        required=False,
        allow_none=True,
    )


class InstitutionBookingCancelSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    version = ma_fields.Integer(required=True)
    reason = ma_fields.String(required=True, validate=validate.Length(min=1))
    reason_code = ma_fields.String(required=False, allow_none=True)


booking_patch_schema = InstitutionBookingPatchSchema()
booking_cancel_schema = InstitutionBookingCancelSchema()


def _resolve_user_int_id(jwt_identity: Any) -> int | None:
    """JWT identity = User.public_id (UUID). Résout User.id (entier) pour les FK."""
    if jwt_identity is None:
        return None
    if isinstance(jwt_identity, int):
        return jwt_identity
    raw = str(jwt_identity).strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    try:
        from models import User

        user = User.query.filter_by(public_id=raw).first()
        if user:
            return int(user.id)
    except Exception:
        return None
    return None


def get_institution_booking_context() -> tuple[int, int | None, str | None, str | None]:
    verify_jwt_in_request()
    claims = get_jwt()
    institution_id = claims.get("institution_id")
    institution_role = claims.get("institution_role")
    if not institution_id:
        from flask import abort

        abort(403, description="Accès réservé aux utilisateurs institution")
    user_id = _resolve_user_int_id(get_jwt_identity())
    display = claims.get("name") or claims.get("email")
    return int(institution_id), user_id, institution_role, display


def _actor_display_name(user_id: int | None, fallback: str | None) -> str | None:
    if fallback:
        return str(fallback)
    if not user_id:
        return None
    try:
        from models import User

        user = User.query.get(user_id)
        if user:
            return user.full_name or user.email or f"User #{user_id}"
    except Exception:
        pass
    return f"User #{user_id}"


@institution_bookings_ns.route("/<int:booking_id>")
class InstitutionBookingUpdate(Resource):
    @institution_bookings_ns.doc(
        description="Modification opérationnelle booking (avant boarded_at)",
        security="BearerAuth",
    )
    @institution_bookings_ns.expect(booking_patch_model, validate=False)
    @institution_bookings_ns.response(401, "Non authentifié", permission_error_model)
    @institution_bookings_ns.response(403, "Accès refusé", permission_error_model)
    @institution_bookings_ns.response(404, "Non trouvé", not_found_error_model)
    @institution_bookings_ns.response(409, "Conflit de version", api_error_model)
    def patch(self, booking_id: int):
        try:
            institution_id, user_id, role, display = get_institution_booking_context()
            role_err = assert_operational_role(role)
            if role_err:
                return {"error": role_err}, 403

            data = parse_request_json()
            if not data:
                return {
                    "error": "invalid_json",
                    "message": (
                        "Corps de requête JSON manquant ou invalide. "
                        "Vérifiez le format du body."
                    ),
                }, 400
            extra = set(data.keys()) - set(booking_patch_schema.fields.keys())
            if extra:
                return {
                    "error": "Champs non autorisés.",
                    "rejected_fields": sorted(extra),
                }, 400

            try:
                validated = cast(dict[str, Any], booking_patch_schema.load(data))
            except ValidationError as ve:
                return {"error": "Données invalides", "details": ve.messages}, 400

            ctx = resolve_institution_booking(booking_id, institution_id)
            if not ctx:
                return {"error": "Booking non trouvé"}, 404

            body, code = update_institution_booking(
                ctx,
                payload=validated,
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=_actor_display_name(user_id, display),
            )

            if code == 200:
                try:
                    AuditLogger.log_action(
                        action_type="institution_booking_updated",
                        action_category="institution",
                        user_id=user_id,
                        user_type="institution",
                        institution_id=institution_id,
                        result_status="success",
                        action_details={
                            "booking_id": booking_id,
                            "updated_fields": body.get("updated_fields"),
                            "role": role,
                        },
                    )
                except Exception as audit_err:
                    logger.warning("[InstitutionBookings] audit: %s", audit_err)

            return body, code
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.exception("[InstitutionBookings] PATCH %s: %s", booking_id, e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_bookings_ns.route("/<int:booking_id>/cancel")
class InstitutionBookingCancel(Resource):
    @institution_bookings_ns.doc(
        description="Annulation booking institution (avant boarded_at)",
        security="BearerAuth",
    )
    @institution_bookings_ns.expect(cancel_model, validate=False)
    def post(self, booking_id: int):
        try:
            institution_id, user_id, role, display = get_institution_booking_context()
            role_err = assert_operational_role(role)
            if role_err:
                return {"error": role_err}, 403

            data = parse_request_json()
            if not data:
                return {
                    "error": "invalid_json",
                    "message": (
                        "Corps de requête JSON manquant ou invalide. "
                        "Vérifiez le format du body."
                    ),
                }, 400
            try:
                validated = cast(dict[str, Any], booking_cancel_schema.load(data))
            except ValidationError as ve:
                return {"error": "Données invalides", "details": ve.messages}, 400

            ctx = resolve_institution_booking(booking_id, institution_id)
            if not ctx:
                return {"error": "Booking non trouvé"}, 404

            body, code = cancel_institution_booking(
                ctx,
                reason=validated["reason"],
                reason_code=validated.get("reason_code"),
                actor_user_id=user_id,
                actor_role=role,
                actor_display_name=_actor_display_name(user_id, display),
                client_version=int(validated["version"]),
            )
            return body, code
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_bookings_ns.route("/<int:booking_id>/change-events")
class InstitutionBookingChangeEvents(Resource):
    @institution_bookings_ns.doc(
        description="Historique audit métier du booking",
        security="BearerAuth",
    )
    def get(self, booking_id: int):
        verify_jwt_in_request()
        claims = get_jwt()
        institution_id = claims.get("institution_id")
        institution_role = claims.get("institution_role")
        if not institution_id:
            return {"error": "Accès refusé"}, 403

        ctx = resolve_institution_booking(booking_id, int(institution_id))
        if not ctx:
            return {"error": "Booking non trouvé"}, 404

        events = list_change_events(
            booking_id, institution_id=int(institution_id), limit=200
        )
        activity = [
            e
            for e in events
            if e.get("action_type")
            in (
                "status_changed",
                "cancelled",
                "field_updated",
                "notification_sent",
            )
            and e.get("change_scope") != "billing"
        ]
        audit = events
        return {
            "events": events,
            "activity": activity,
            "audit": audit,
            "allowed_fields": sorted(INSTITUTION_OPERATIONAL_FIELDS),
            "edit_version": ctx.booking.edit_version,
            "can_view_amounts": institution_role
            in (
                InstitutionRole.ADMIN.value,
                InstitutionRole.BILLING.value,
                InstitutionRole.CURATOR.value,
            ),
        }, 200


@institution_bookings_ns.route("/<int:booking_id>/release-for-redispatch")
class InstitutionBookingReleaseForRedispatch(Resource):
    @institution_bookings_ns.doc(
        description="Remet une course en diffusion (escalade / refus transporteur)",
        security="BearerAuth",
    )
    def post(self, booking_id: int):
        try:
            institution_id, user_id, role, _display = get_institution_booking_context()
            role_err = assert_operational_role(role)
            if role_err:
                return {"error": role_err}, 403

            ctx = resolve_institution_booking(booking_id, institution_id)
            if not ctx:
                return {"error": "Booking non trouvé"}, 404

            from application.institutions.release_booking_for_redispatch import (
                ReleaseBookingForRedispatchInput,
                ReleaseBookingForRedispatchUseCase,
            )

            data = request.get_json(silent=True) or {}
            result = ReleaseBookingForRedispatchUseCase().execute(
                ReleaseBookingForRedispatchInput(
                    booking_id=booking_id,
                    institution_id=institution_id,
                    reason=data.get("reason"),
                    actor_user_id=user_id,
                    trigger_redispatch=data.get("trigger_redispatch", True),
                )
            )
            if not result.success:
                return {"error": result.error}, result.status_code

            db.session.commit()
            return {
                "success": True,
                "booking_id": result.booking_id,
                "redispatched": result.redispatched,
                "offers_created": result.offers_created,
            }, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.exception(
                "[InstitutionBookings] POST release-for-redispatch %s: %s",
                booking_id,
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500
