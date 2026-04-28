from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus

from flask_jwt_extended import jwt_required
from flask_restx import Resource

from ext import db, role_required
from models import Booking, DispatchOffer
from models.enums import BookingStatus, DispatchOfferStatus, UserRole
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _get_current_company
from services.dispatch.scoring_engine import (
    compute_candidates,
    compute_urgency_override_candidates,
    persist_offers_for_threshold,
    persist_urgency_offers,
)
from services.platform_exceptions import PlatformTenantSuspended
from services.platform_tenant_gates import assert_company_not_platform_suspended


@dispatch_ns.route("/scoring/preview/<int:booking_id>")
class DispatchScoringPreviewResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, booking_id: int):
        company = _get_current_company()
        booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
        if not booking:
            return {"error": "Booking introuvable"}, HTTPStatus.NOT_FOUND
        candidates = compute_candidates(
            pickup_geo_unit=booking.pickup_geo_unit,
            drop_geo_unit=booking.dropoff_geo_unit,
        )
        return {
            "booking_id": booking_id,
            "candidates": [
                {"company_id": c.company_id, "score": c.score, "reason": c.reason}
                for c in candidates
            ],
        }, HTTPStatus.OK


@dispatch_ns.route("/scoring/dispatch/<int:booking_id>")
class DispatchScoringDispatchResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id: int):
        company = _get_current_company()
        booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
        if not booking:
            return {"error": "Booking introuvable"}, HTTPStatus.NOT_FOUND

        candidates = compute_candidates(
            pickup_geo_unit=booking.pickup_geo_unit,
            drop_geo_unit=booking.dropoff_geo_unit,
        )
        created_total = []
        for threshold in (100, 70, 50, 10):
            created = persist_offers_for_threshold(
                booking_id=booking.id, candidates=candidates, threshold=threshold
            )
            created_total.extend(created)
            if created:
                break

        pickup_at = booking.scheduled_time
        should_run_urgency = False
        if pickup_at:
            pickup_at_aware = pickup_at.replace(tzinfo=UTC) if pickup_at.tzinfo is None else pickup_at
            should_run_urgency = (pickup_at_aware - datetime.now(UTC)).total_seconds() <= 15 * 60

        if not created_total and should_run_urgency:
            urgent_candidates = compute_urgency_override_candidates(
                pickup_lat=booking.pickup_lat,
                pickup_lon=booking.pickup_lon,
            )
            created_total = persist_urgency_offers(booking.id, urgent_candidates)

        db.session.commit()
        return {
            "booking_id": booking_id,
            "offers_created": len(created_total),
            "offers": [
                {
                    "id": offer.id,
                    "company_id": offer.company_id,
                    "score": offer.score,
                    "status": offer.status.value if hasattr(offer.status, "value") else offer.status,
                    "reason_json": offer.reason_json,
                }
                for offer in created_total
            ],
            "existing_offers": DispatchOffer.query.filter_by(booking_id=booking_id).count(),
        }, HTTPStatus.OK


@dispatch_ns.route("/scoring/offers/<int:offer_id>/accept")
class DispatchOfferAcceptResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, offer_id: int):
        """Première entreprise qui accepte devient propriétaire de la course (booking.company_id)."""
        company = _get_current_company()
        offer = DispatchOffer.query.filter_by(
            id=offer_id,
            company_id=company.id,
            status=DispatchOfferStatus.PROPOSED,
        ).first()
        if not offer:
            return {"error": "Offre introuvable ou déjà traitée"}, HTTPStatus.NOT_FOUND

        booking = offer.booking
        if not booking:
            return {"error": "Réservation introuvable"}, HTTPStatus.NOT_FOUND
        if booking.company_id is not None:
            return {
                "error": "Cette course est déjà attribuée à un transporteur",
            }, HTTPStatus.CONFLICT

        try:
            assert_company_not_platform_suspended(company.id)
        except PlatformTenantSuspended as exc:
            return {"error": exc.message}, HTTPStatus.FORBIDDEN

        if offer.expires_at and offer.expires_at < datetime.now(UTC):
            offer.status = DispatchOfferStatus.EXPIRED
            db.session.commit()
            return {"error": "Offre expirée"}, HTTPStatus.GONE

        booking.company_id = company.id
        offer.status = DispatchOfferStatus.ACCEPTED

        for other in DispatchOffer.query.filter(
            DispatchOffer.booking_id == booking.id,
            DispatchOffer.id != offer.id,
            DispatchOffer.status == DispatchOfferStatus.PROPOSED,
        ).all():
            other.status = DispatchOfferStatus.EXPIRED

        if booking.status == BookingStatus.PENDING:
            booking.status = BookingStatus.ACCEPTED

        for ret in Booking.query.filter_by(
            parent_booking_id=booking.id,
            is_return=True,
        ).all():
            ret.company_id = company.id

        db.session.commit()
        return {
            "success": True,
            "offer_id": offer.id,
            "booking_id": booking.id,
            "company_id": company.id,
            "booking_status": (
                booking.status.value
                if hasattr(booking.status, "value")
                else str(booking.status)
            ),
        }, HTTPStatus.OK


@dispatch_ns.route("/scoring/offers/<int:offer_id>/decline")
class DispatchOfferDeclineResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, offer_id: int):
        company = _get_current_company()
        offer = DispatchOffer.query.filter_by(id=offer_id, company_id=company.id).first()
        if not offer:
            return {"error": "Offer introuvable"}, HTTPStatus.NOT_FOUND
        offer.status = DispatchOfferStatus.DECLINED
        db.session.commit()
        return {"success": True, "offer_id": offer.id, "status": "DECLINED"}, HTTPStatus.OK

