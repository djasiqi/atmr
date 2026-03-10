from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from ext import db, role_required
from models import Booking, DispatchOffer
from models.enums import DispatchOfferStatus, UserRole
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _get_current_company
from services.dispatch.scoring_engine import (
    compute_candidates,
    compute_urgency_override_candidates,
    persist_offers_for_threshold,
    persist_urgency_offers,
)


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

