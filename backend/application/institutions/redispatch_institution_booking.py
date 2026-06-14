# application/institutions/redispatch_institution_booking.py
"""Use case: Remettre en diffusion une course institution.

Après libération (refus transporteur ou escalade), la demande de transport
source est repassée en SENT et de nouvelles offres broadcast sont créées vers
les entreprises éligibles (en excluant l'ancien transporteur).

Limitation connue (stop-gate PR2) : la réacceptation d'une offre via
AcceptOfferUseCase crée un nouveau Booking. Le rattachement de l'ancien booking
libéré au nouveau transporteur sera traité dans un PR ultérieur ; ici on se
concentre sur la remise au marché de la demande source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ext import db
from models import OfferStatus, RequestOffer, RequestStatus, TransportRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedispatchInstitutionBookingInput:
    """Input pour la remise en diffusion."""

    booking_id: int
    institution_id: int | None = None
    previous_company_id: int | None = None


@dataclass(frozen=True, slots=True)
class RedispatchInstitutionBookingResult:
    """Résultat de la remise en diffusion."""

    success: bool
    booking_id: int
    transport_request_id: int | None = None
    offers_created: int = 0
    error: str | None = None
    status_code: int = 200


class RedispatchInstitutionBookingUseCase:
    """Use case: recréer des offres pour une demande institution remise au marché."""

    def execute(
        self, input_data: RedispatchInstitutionBookingInput
    ) -> RedispatchInstitutionBookingResult:
        transport_request = (
            db.session.query(TransportRequest)
            .filter(TransportRequest.booking_id == input_data.booking_id)
            .first()
        )
        if not transport_request:
            return RedispatchInstitutionBookingResult(
                success=False,
                booking_id=input_data.booking_id,
                error="Demande de transport source introuvable.",
                status_code=404,
            )

        # Repasser la demande en SENT (rouvre l'acceptation)
        transport_request.status = RequestStatus.SENT.value
        transport_request.accepted_by_company_id = None

        # Clore les offres encore PENDING avant de recréer (évite doublons)
        stale = RequestOffer.query.filter(
            RequestOffer.transport_request_id == transport_request.id,
            RequestOffer.status == OfferStatus.PENDING.value,
        ).all()
        for offer in stale:
            offer.status = OfferStatus.UNAVAILABLE.value

        offers_created = self._create_offers(
            transport_request, exclude_company_id=input_data.previous_company_id
        )
        db.session.flush()

        if offers_created == 0:
            logger.warning(
                "[RedispatchInstitutionBooking] Aucune entreprise éligible "
                "pour redispatch booking=%s request=%s",
                input_data.booking_id,
                transport_request.id,
            )

        return RedispatchInstitutionBookingResult(
            success=True,
            booking_id=input_data.booking_id,
            transport_request_id=transport_request.id,
            offers_created=offers_created,
        )

    @staticmethod
    def _create_offers(
        transport_request: TransportRequest,
        *,
        exclude_company_id: int | None,
    ) -> int:
        """Crée des offres broadcast en excluant l'ancien transporteur."""
        from application.institutions.send_transport_request import (
            SendTransportRequestUseCase,
        )

        excluded: list[int] = []
        if exclude_company_id:
            excluded.append(int(exclude_company_id))

        use_case = SendTransportRequestUseCase()
        return use_case._create_broadcast_offers(
            transport_request=transport_request,
            expires_at=None,
            excluded_company_ids=excluded,
        )
