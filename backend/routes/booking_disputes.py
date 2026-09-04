"""Routes contestation — entreprise (répondre) et institution/admin (trancher)."""

from __future__ import annotations

import logging

from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required
from flask_restx import Resource

from application.invoices.booking_dispute.freeze import get_open_dispute_for_booking
from application.invoices.booking_dispute.service import (
    add_carrier_evidence,
    carrier_respond,
    decide_dispute,
    ensure_open_dispute,
    present_dispute,
    submit_dispute_for_validation,
)
from ext import db, role_required
from models import Booking, BookingDispute, User
from models.enums import InstitutionBillingControlStatus
from routes.institution_billing import get_billing_context, institution_billing_ns
from routes.invoices import invoices_ns
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import success_response

logger = logging.getLogger(__name__)


def _current_user_id() -> int | None:
    ident = get_jwt_identity()
    if ident is None:
        return None
    try:
        return int(ident)
    except (TypeError, ValueError):
        pass
    user = User.query.filter_by(public_id=str(ident)).first()
    if user is None:
        user = User.query.filter_by(email=str(ident)).first()
    return int(user.id) if user is not None else None


def _current_role() -> str | None:
    claims = get_jwt() or {}
    return str(claims.get("role") or claims.get("institution_role") or "")


def _load_company_booking(company_id: int, booking_id: int) -> Booking | None:
    return Booking.query.filter_by(id=booking_id, company_id=company_id).first()


def _emit_updated(booking: Booking, reason: str) -> None:
    try:
        from services.realtime.socketio import emit_company_event

        cid = getattr(booking, "company_id", None)
        if cid:
            emit_company_event(
                int(cid),
                "booking_updated",
                {"booking_id": int(booking.id), "reason": reason},
            )
    except Exception as exc:
        logger.warning("[Dispute] fan-out booking_updated: %s", exc)


def _company_guard(company_id: int):
    from routes.companies import _get_current_company_via_use_case

    company, error_response, status_code = _get_current_company_via_use_case()
    if error_response or not company:
        return None, (error_response, status_code or 403)
    if int(getattr(company, "id", 0) or 0) != int(company_id):
        return None, APIErrorHandler.handle_permission_error(
            "Non autorisé", logger_instance=logger
        )
    return company, None


def _has_open_or_legacy_dispute(booking: Booking) -> bool:
    if get_open_dispute_for_booking(int(booking.id)) is not None:
        return True
    persisted = getattr(booking, "institution_control_status", None)
    persisted_v = str(getattr(persisted, "value", persisted) or "")
    # not_billable = résolution déjà tranchée : ne pas rouvrir au GET.
    return persisted_v == InstitutionBillingControlStatus.ANOMALY.value


@invoices_ns.route("/companies/<int:company_id>/bookings/<int:booking_id>/dispute")
class CompanyBookingDispute(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id: int, booking_id: int):
        company, err = _company_guard(company_id)
        if err:
            return err
        booking = _load_company_booking(int(company.id), booking_id)
        if booking is None:
            return APIErrorHandler.handle_not_found_error(
                "Course introuvable", logger_instance=logger
            )
        existing = get_open_dispute_for_booking(booking_id)
        if existing is not None:
            return success_response(data=present_dispute(existing, booking))
        if _has_open_or_legacy_dispute(booking):
            dispute = ensure_open_dispute(
                booking,
                actor_user_id=_current_user_id(),
                actor_role=_current_role() or "company",
            )
            db.session.commit()
            return success_response(data=present_dispute(dispute, booking))
        latest = (
            db.session.query(BookingDispute)
            .filter(BookingDispute.booking_id == booking_id)
            .order_by(BookingDispute.id.desc())
            .first()
        )
        if latest is None:
            return APIErrorHandler.handle_validation_error(
                "Cette course n'est pas contestée.",
                logger_instance=logger,
            )
        return success_response(data=present_dispute(latest, booking))

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id: int, booking_id: int):
        """Réponse transporteur : institution a raison / mission faite / correction."""
        company, err = _company_guard(company_id)
        if err:
            return err
        booking = _load_company_booking(int(company.id), booking_id)
        if booking is None:
            return APIErrorHandler.handle_not_found_error(
                "Course introuvable", logger_instance=logger
            )
        body = request.get_json(silent=True) or {}
        result = carrier_respond(
            booking,
            stance=str(body.get("stance") or ""),
            note=body.get("note"),
            exclusion_reason=body.get("exclusion_reason"),
            proposed_amount_ht=body.get("proposed_amount_ht"),
            proposed_payer_type=body.get("proposed_payer_type"),
            proposed_correction_note=body.get("proposed_correction_note"),
            actor_user_id=_current_user_id(),
            actor_role=_current_role() or "COMPANY",
        )
        if not result.ok:
            return {"error": result.error}, int(result.status_code or 400)
        db.session.commit()
        _emit_updated(booking, "dispute_carrier_responded")
        return success_response(data=present_dispute(result.dispute, booking))


@invoices_ns.route(
    "/companies/<int:company_id>/bookings/<int:booking_id>/dispute/evidence"
)
class CompanyBookingDisputeEvidence(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id: int, booking_id: int):
        company, err = _company_guard(company_id)
        if err:
            return err
        booking = _load_company_booking(int(company.id), booking_id)
        if booking is None:
            return APIErrorHandler.handle_not_found_error(
                "Course introuvable", logger_instance=logger
            )
        body = request.get_json(silent=True) or {}
        result = add_carrier_evidence(
            booking,
            kind=str(body.get("kind") or ""),
            note=body.get("note"),
            actor_user_id=_current_user_id(),
            actor_role=_current_role() or "COMPANY",
        )
        if not result.ok:
            return {"error": result.error}, int(result.status_code or 400)
        db.session.commit()
        return success_response(data=present_dispute(result.dispute, booking))


@invoices_ns.route(
    "/companies/<int:company_id>/bookings/<int:booking_id>/dispute/submit"
)
class CompanyBookingDisputeSubmit(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id: int, booking_id: int):
        company, err = _company_guard(company_id)
        if err:
            return err
        booking = _load_company_booking(int(company.id), booking_id)
        if booking is None:
            return APIErrorHandler.handle_not_found_error(
                "Course introuvable", logger_instance=logger
            )
        result = submit_dispute_for_validation(
            booking,
            actor_user_id=_current_user_id(),
            actor_role=_current_role() or "COMPANY",
        )
        if not result.ok:
            return {"error": result.error}, int(result.status_code or 400)
        db.session.commit()
        _emit_updated(booking, "dispute_evidence_submitted")
        return success_response(data=present_dispute(result.dispute, booking))


@invoices_ns.route(
    "/companies/<int:company_id>/bookings/<int:booking_id>/dispute/decide"
)
class CompanyBookingDisputeDecide(Resource):
    """Option B — opérateur LIRIE (ADMIN) tranche si l'institution ne répond pas."""

    @jwt_required()
    @role_required(["ADMIN"])
    def post(self, company_id: int, booking_id: int):
        company, err = _company_guard(company_id)
        if err:
            return err
        booking = _load_company_booking(int(company.id), booking_id)
        if booking is None:
            return APIErrorHandler.handle_not_found_error(
                "Course introuvable", logger_instance=logger
            )
        body = request.get_json(silent=True) or {}
        result = decide_dispute(
            booking,
            decision=str(body.get("decision") or ""),
            note=body.get("note"),
            actor_user_id=_current_user_id(),
            actor_role="ADMIN",
        )
        if not result.ok:
            return {"error": result.error}, int(result.status_code or 400)
        db.session.commit()
        _emit_updated(booking, "dispute_decided")
        return success_response(data=present_dispute(result.dispute, booking))


@institution_billing_ns.route("/bookings/<int:booking_id>/dispute")
class InstitutionBookingDispute(Resource):
    @jwt_required()
    def get(self, booking_id: int):
        from application.institutions.billing_control.resolve import (
            resolve_institution_billing_control_booking,
        )

        institution_id, _user_id, _role = get_billing_context()
        ctx = resolve_institution_billing_control_booking(booking_id, institution_id)
        if ctx is None:
            return {"error": "Booking non trouvé ou hors de votre institution"}, 404
        existing = get_open_dispute_for_booking(booking_id)
        if existing is None:
            existing = (
                db.session.query(BookingDispute)
                .filter(BookingDispute.booking_id == booking_id)
                .order_by(BookingDispute.id.desc())
                .first()
            )
        if existing is None:
            return {"error": "Aucune contestation pour cette course."}, 404
        return {"success": True, "dispute": present_dispute(existing, ctx.booking)}


@institution_billing_ns.route("/bookings/<int:booking_id>/dispute/decide")
class InstitutionBookingDisputeDecide(Resource):
    @jwt_required()
    def post(self, booking_id: int):
        """Institution : accepte ou refuse la preuve. Jamais le transporteur seul."""
        from application.institutions.billing_control.resolve import (
            resolve_institution_billing_control_booking,
        )

        institution_id, user_id, role = get_billing_context()
        ctx = resolve_institution_billing_control_booking(booking_id, institution_id)
        if ctx is None:
            return {"error": "Booking non trouvé ou hors de votre institution"}, 404
        body = request.get_json(silent=True) or {}
        result = decide_dispute(
            ctx.booking,
            decision=str(body.get("decision") or ""),
            note=body.get("note"),
            actor_user_id=user_id,
            actor_role=role or "institution_billing",
        )
        if not result.ok:
            return {"error": result.error}, int(result.status_code or 400)
        db.session.commit()
        _emit_updated(ctx.booking, "dispute_decided")
        return {
            "success": True,
            "dispute": present_dispute(result.dispute, ctx.booking),
        }
