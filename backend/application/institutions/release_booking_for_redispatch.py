# application/institutions/release_booking_for_redispatch.py
"""Use case: Libérer une course institution pour rediffusion.

Remet une course au marché ouvert :
- détache chauffeur et entreprise exécutante
- repasse le booking en PENDING
- historise l'événement timeline `redispatched`
- déclenche optionnellement la rediffusion (RedispatchInstitutionBookingUseCase)

Utilisé après un refus transporteur (révalidation) ou une escalade
(expiration de la demande de validation avec auto-refus activé).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ext import db
from models import Booking
from models.enums import BookingStatus
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReleaseBookingForRedispatchInput:
    """Input pour la libération d'une course."""

    booking_id: int
    institution_id: int | None = None
    reason: str | None = None
    previous_company_id: int | None = None
    actor_user_id: int | None = None
    trigger_redispatch: bool = True


@dataclass(frozen=True, slots=True)
class ReleaseBookingForRedispatchResult:
    """Résultat de la libération."""

    success: bool
    booking_id: int
    previous_company_id: int | None = None
    redispatched: bool = False
    offers_created: int = 0
    error: str | None = None
    status_code: int = 200


class ReleaseBookingForRedispatchUseCase:
    """Use case: libérer une course et la remettre en diffusion."""

    def execute(
        self, input_data: ReleaseBookingForRedispatchInput
    ) -> ReleaseBookingForRedispatchResult:
        booking = Booking.query.get(input_data.booking_id)
        if not booking:
            return ReleaseBookingForRedispatchResult(
                success=False,
                booking_id=input_data.booking_id,
                error="Course introuvable.",
                status_code=404,
            )

        status = str(getattr(booking.status, "value", booking.status) or "").upper()
        if status in (
            BookingStatus.COMPLETED.value,
            BookingStatus.RETURN_COMPLETED.value,
            BookingStatus.CANCELED.value,
        ):
            return ReleaseBookingForRedispatchResult(
                success=False,
                booking_id=booking.id,
                error=f"Libération impossible (statut {status}).",
                status_code=409,
            )

        previous_company_id = (
            input_data.previous_company_id
            or booking.company_id
            or booking.executing_company_id
        )
        previous_company_name = None
        try:
            from models import Company

            if previous_company_id:
                company = Company.query.get(int(previous_company_id))
                previous_company_name = company.name if company else None
        except Exception:
            previous_company_name = None

        # Détacher le transporteur et remettre au marché
        booking.driver_id = None
        booking.executing_company_id = None
        booking.company_id = None
        booking.status = BookingStatus.PENDING
        booking.active_change_request_id = None
        booking.updated_at = datetime.now(UTC)

        self._record_redispatch_timeline(
            booking=booking,
            institution_id=input_data.institution_id,
            previous_company_id=previous_company_id,
            previous_company_name=previous_company_name,
            actor_user_id=input_data.actor_user_id,
            reason=input_data.reason,
        )
        db.session.flush()

        redispatched = False
        offers_created = 0
        if input_data.trigger_redispatch:
            try:
                from application.institutions.redispatch_institution_booking import (
                    RedispatchInstitutionBookingInput,
                    RedispatchInstitutionBookingUseCase,
                )

                redispatch_result = RedispatchInstitutionBookingUseCase().execute(
                    RedispatchInstitutionBookingInput(
                        booking_id=booking.id,
                        institution_id=input_data.institution_id,
                        previous_company_id=(
                            int(previous_company_id) if previous_company_id else None
                        ),
                    )
                )
                redispatched = redispatch_result.success
                offers_created = redispatch_result.offers_created
            except Exception as redispatch_err:
                logger.warning(
                    "[ReleaseBooking] redispatch failed booking=%s: %s",
                    booking.id,
                    redispatch_err,
                )

        try:
            AuditLogger.log_action(
                action_type="booking_released_for_redispatch",
                action_category="institution",
                user_id=input_data.actor_user_id,
                user_type="system",
                institution_id=input_data.institution_id,
                result_status="success",
                action_details={
                    "booking_id": booking.id,
                    "previous_company_id": previous_company_id,
                    "redispatched": redispatched,
                    "offers_created": offers_created,
                    "reason": input_data.reason,
                },
            )
        except Exception as audit_err:
            logger.warning("[ReleaseBooking] audit failed: %s", audit_err)

        return ReleaseBookingForRedispatchResult(
            success=True,
            booking_id=booking.id,
            previous_company_id=(
                int(previous_company_id) if previous_company_id else None
            ),
            redispatched=redispatched,
            offers_created=offers_created,
        )

    @staticmethod
    def _record_redispatch_timeline(
        *,
        booking: Booking,
        institution_id: int | None,
        previous_company_id: int | None,
        previous_company_name: str | None,
        actor_user_id: int | None,
        reason: str | None,
    ) -> None:
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                record_event,
            )

            transport_request_id = None
            try:
                from models import TransportRequest

                tr = TransportRequest.query.filter_by(booking_id=booking.id).first()
                transport_request_id = tr.id if tr else None
            except Exception:
                transport_request_id = None

            record_event(
                "redispatched",
                institution_id=institution_id,
                transport_request_id=transport_request_id,
                booking_id=booking.id,
                actor=TimelineActor(
                    actor_type="system", actor_user_id=actor_user_id
                ),
                payload={
                    "previous_company_id": previous_company_id,
                    "previous_company_name": previous_company_name,
                    "reason": reason,
                },
                correlation_id=(
                    f"redispatched:{booking.id}:"
                    f"{int(datetime.now(UTC).timestamp())}"
                ),
            )
        except Exception as timeline_err:
            logger.warning(
                "[ReleaseBooking] redispatch timeline failed: %s", timeline_err
            )
