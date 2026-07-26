# application/institutions/assign_external_carrier.py
"""Use case: Affecter un transporteur externe à une demande institution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ext import db
from models import OfferStatus, RequestOffer, RequestStatus, TransportRequest, User
from models.enums import CarrierSource
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)

_ASSIGNABLE_STATUSES = frozenset(
    {
        RequestStatus.DRAFT.value,
        RequestStatus.SENT.value,
        RequestStatus.EXTERNAL_ASSIGNED.value,
    }
)


@dataclass(frozen=True, slots=True)
class AssignExternalCarrierInput:
    """Entrée pour l'affectation d'un transporteur externe."""

    transport_request_id: int
    institution_id: int
    user_id: int
    name: str
    phone: str | None = None
    email: str | None = None
    reference: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AssignExternalCarrierResult:
    """Résultat de l'affectation externe."""

    success: bool
    transport_request_id: int
    switched_from_lirie: bool = False
    error: str | None = None
    status_code: int = 200


class AssignExternalCarrierUseCase:
    """Use case: basculer une demande vers un transporteur externe."""

    def execute(
        self, input_data: AssignExternalCarrierInput
    ) -> AssignExternalCarrierResult:
        try:
            name = (input_data.name or "").strip()
            if not name:
                return AssignExternalCarrierResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Le nom du transporteur externe est requis",
                    status_code=400,
                )

            transport_request = (
                db.session.query(TransportRequest)
                .filter(
                    TransportRequest.id == input_data.transport_request_id,
                    TransportRequest.institution_id == input_data.institution_id,
                )
                .with_for_update()
                .first()
            )
            if not transport_request:
                return AssignExternalCarrierResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Demande de transport introuvable",
                    status_code=404,
                )

            if transport_request.status not in _ASSIGNABLE_STATUSES:
                return AssignExternalCarrierResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error=(
                        f"Demande en statut {transport_request.status}, "
                        "affectation externe impossible"
                    ),
                    status_code=409,
                )

            if (
                transport_request.carrier_source == CarrierSource.EXTERNAL.value
                and transport_request.status
                == RequestStatus.EXTERNAL_DECLARED_COMPLETED.value
            ):
                return AssignExternalCarrierResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error="Mission externe déjà déclarée réalisée",
                    status_code=409,
                )

            pending_offers = (
                RequestOffer.query.filter(
                    RequestOffer.transport_request_id == transport_request.id,
                    RequestOffer.status == OfferStatus.PENDING.value,
                )
                .with_for_update()
                .all()
            )
            switched_from_lirie = bool(pending_offers)
            for offer in pending_offers:
                offer.mark_unavailable()

            now = datetime.now(UTC)
            actor_name = self._user_display_name(input_data.user_id)
            phone = (input_data.phone or "").strip() or None
            email = (input_data.email or "").strip() or None
            reference = (input_data.reference or "").strip() or None
            reason = (input_data.reason or "").strip() or None

            transport_request.carrier_source = CarrierSource.EXTERNAL.value
            transport_request.external_carrier_name = name[:255]
            transport_request.external_carrier_phone = phone[:50] if phone else None
            transport_request.external_carrier_email = email[:255] if email else None
            transport_request.external_carrier_reference = (
                reference[:100] if reference else None
            )
            transport_request.external_carrier_reason = reason[:120] if reason else None
            transport_request.assigned_externally_at = now
            transport_request.externalized_by_user_id = input_data.user_id
            transport_request.status = RequestStatus.EXTERNAL_ASSIGNED.value
            transport_request.accepted_by_company_id = None
            transport_request.accepted_at = None

            self._record_timeline(
                transport_request=transport_request,
                user_id=input_data.user_id,
                switched_from_lirie=switched_from_lirie,
                carrier_name=name,
                carrier_phone=phone,
                carrier_reference=reference,
                reason=reason,
                actor_name=actor_name,
                offers_stopped=len(pending_offers),
            )

            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="external_carrier_assigned",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="institution",
                    institution_id=input_data.institution_id,
                    result_status="success",
                    action_details={
                        "transport_request_id": transport_request.id,
                        "carrier_name": name,
                        "switched_from_lirie": switched_from_lirie,
                        "offers_stopped": len(pending_offers),
                    },
                )
            except Exception as audit_err:
                logger.warning("[AssignExternalCarrier] Audit log error: %s", audit_err)

            return AssignExternalCarrierResult(
                success=True,
                transport_request_id=transport_request.id,
                switched_from_lirie=switched_from_lirie,
            )
        except Exception as exc:
            logger.exception(
                "Erreur lors de l'affectation externe request=%s",
                input_data.transport_request_id,
            )
            db.session.rollback()
            return AssignExternalCarrierResult(
                success=False,
                transport_request_id=input_data.transport_request_id,
                error=f"Erreur inattendue: {exc!s}",
                status_code=500,
            )

    @staticmethod
    def _user_display_name(user_id: int) -> str | None:
        user = User.query.get(user_id)
        if not user:
            return None
        first = getattr(user, "first_name", "") or ""
        last = getattr(user, "last_name", "") or ""
        name = f"{first} {last}".strip()
        return name or getattr(user, "username", None)

    @staticmethod
    def _record_timeline(
        *,
        transport_request: TransportRequest,
        user_id: int,
        switched_from_lirie: bool,
        carrier_name: str,
        carrier_phone: str | None,
        carrier_reference: str | None,
        reason: str | None,
        actor_name: str | None,
        offers_stopped: int,
    ) -> None:
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                record_event,
            )

            payload = {
                "carrier_name": carrier_name,
                "carrier_phone": carrier_phone,
                "carrier_reference": carrier_reference,
                "reason": reason,
                "actor_name": actor_name,
            }
            if switched_from_lirie:
                payload["offers_stopped"] = offers_stopped
                event_type = "external_carrier_switched"
            else:
                event_type = "external_carrier_assigned"

            record_event(
                event_type,
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                actor=TimelineActor(
                    actor_type="institution",
                    actor_user_id=user_id,
                ),
                payload=payload,
                correlation_id=f"{event_type}:{transport_request.id}:{user_id}",
            )
        except Exception as timeline_err:
            logger.warning(
                "[AssignExternalCarrier] Timeline recording failed: %s",
                timeline_err,
            )
