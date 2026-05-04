# application/institutions/reject_offer.py
"""Use case: Rejeter une offre de transport.

Gère le rejet avec déclenchement de l'escalade si mode séquentiel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ext import db
from models import (
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
)
from security.audit_log import AuditLogger

from .institution_settings_service import calculate_timeout
from .send_transport_request import (
    create_fallback_broadcast_offers,
    create_next_sequential_offer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RejectOfferInput:
    """Input pour le rejet d'une offre."""

    offer_id: int
    company_id: int
    user_id: int  # Utilisateur company qui rejette
    reason: str | None = None  # Raison du rejet


@dataclass(frozen=True, slots=True)
class RejectOfferResult:
    """Résultat du rejet d'une offre."""

    success: bool
    offer_id: int
    escalated: bool = False  # True si escalade déclenchée
    next_offer_id: int | None = None  # ID de la prochaine offre si escalade
    fallback_broadcast: bool = False  # True si fallback broadcast déclenché
    error: str | None = None
    status_code: int = 200


class RejectOfferUseCase:
    """Use case: Rejeter une offre de transport."""

    def execute(self, input_data: RejectOfferInput) -> RejectOfferResult:
        """
        Rejette une offre de transport.

        Si mode séquentiel:
        - Déclenche l'escalade vers la préférence suivante
        - Ou le fallback broadcast si plus de préférences

        Args:
            input_data: Données d'entrée

        Returns:
            RejectOfferResult avec info sur l'escalade
        """
        try:
            # 1. Charger l'offre avec vérification ownership
            offer = RequestOffer.query.get(input_data.offer_id)
            if not offer:
                return RejectOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Offre introuvable",
                    status_code=404,
                )

            if offer.company_id != input_data.company_id:
                return RejectOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Vous ne pouvez pas rejeter cette offre",
                    status_code=403,
                )

            # 2. Vérifier que l'offre peut être rejetée
            if offer.status != OfferStatus.PENDING.value:
                return RejectOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error=f"Offre en statut {offer.status}, rejet impossible",
                    status_code=409,
                )

            # 3. Charger la TransportRequest
            transport_request = TransportRequest.query.get(offer.transport_request_id)
            if not transport_request:
                return RejectOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Demande de transport introuvable",
                    status_code=404,
                )

            # 4. Vérifier le statut de la demande
            if transport_request.status not in [
                RequestStatus.SENT.value,
            ]:
                return RejectOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error=f"Demande en statut {transport_request.status}, rejet impossible",
                    status_code=409,
                )

            # 5. Marquer l'offre comme rejetée
            offer.reject(reason=input_data.reason)

            # 6. Gérer l'escalade si mode séquentiel
            escalated = False
            next_offer_id = None
            fallback_broadcast = False

            if offer.mode == OfferMode.SEQUENTIAL.value:
                # Calculer le timeout pour la prochaine offre (via settings institution)
                timeout_minutes = calculate_timeout(
                    transport_request.institution_id,
                    transport_request.scheduled_time,
                )

                # Essayer de créer la prochaine offre séquentielle
                next_offer = create_next_sequential_offer(
                    transport_request=transport_request,
                    current_order=offer.order,
                    timeout_minutes=timeout_minutes,
                )

                if next_offer:
                    escalated = True
                    next_offer_id = next_offer.id
                    logger.info(
                        "[RejectOffer] Escalade: offer %s -> next offer %s (order=%d)",
                        offer.id,
                        next_offer.id,
                        next_offer.order,
                    )
                else:
                    # Pas de préférence suivante -> fallback broadcast
                    offers_created = create_fallback_broadcast_offers(transport_request)
                    if offers_created > 0:
                        fallback_broadcast = True
                        logger.info(
                            "[RejectOffer] Fallback broadcast: %d offers created",
                            offers_created,
                        )
                    else:
                        # Aucune entreprise éligible -> marquer la demande comme expirée
                        transport_request.status = RequestStatus.EXPIRED.value
                        logger.warning(
                            "[RejectOffer] No eligible companies for request %s -> EXPIRED",
                            transport_request.id,
                        )

            db.session.commit()

            # 7. Audit log
            try:
                AuditLogger.log_action(
                    action_type="offer_rejected",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="company",
                    company_id=input_data.company_id,
                    institution_id=transport_request.institution_id,
                    result_status="success",
                    action_details={
                        "offer_id": offer.id,
                        "transport_request_id": transport_request.id,
                        "reason": input_data.reason,
                        "escalated": escalated,
                        "next_offer_id": next_offer_id,
                        "fallback_broadcast": fallback_broadcast,
                    },
                )

                if escalated:
                    AuditLogger.log_action(
                        action_type="request_escalated",
                        action_category="institution",
                        user_id=input_data.user_id,
                        user_type="company",
                        institution_id=transport_request.institution_id,
                        result_status="success",
                        action_details={
                            "transport_request_id": transport_request.id,
                            "from_offer_id": offer.id,
                            "to_offer_id": next_offer_id,
                        },
                    )
            except Exception as audit_err:
                logger.warning("Échec audit log: %s", audit_err)

            logger.info(
                "[RejectOffer] Offer %s rejected by company %s (escalated=%s, fallback=%s)",
                offer.id,
                input_data.company_id,
                escalated,
                fallback_broadcast,
            )

            return RejectOfferResult(
                success=True,
                offer_id=offer.id,
                escalated=escalated,
                next_offer_id=next_offer_id,
                fallback_broadcast=fallback_broadcast,
            )

        except Exception as e:
            logger.exception(
                "Erreur lors du rejet de l'offre %s",
                input_data.offer_id,
            )
            db.session.rollback()
            return RejectOfferResult(
                success=False,
                offer_id=input_data.offer_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )
