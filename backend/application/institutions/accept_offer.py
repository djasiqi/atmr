# application/institutions/accept_offer.py
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false, reportImportCycles=false
"""Use case: Accepter une offre de transport.

Gère l'acceptation atomique avec:
- SELECT FOR UPDATE sur TransportRequest (first accept wins)
- Conversion en Booking
- Marquage des autres offres comme UNAVAILABLE
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from ext import db
from models import (
    Booking,
    BookingStatus,
    Client,
    Company,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    User,
)
from models.enums import BookingCreatedVia, ClientType, ManagementMode, UserRole
from security.audit_log import AuditLogger
from services.pricing.offer_price_estimator import resolve_institution_price
from shared.time_utils import (
    normalize_mission_wall_clock,
)

if TYPE_CHECKING:
    from models.transport_request_leg import TransportRequestLeg

logger = logging.getLogger(__name__)


def _load_transport_request_legs(
    transport_request: TransportRequest,
) -> list[TransportRequestLeg]:
    """Charge les legs ordonnés (relation ORM ou requête directe)."""
    from sqlalchemy import inspect as sa_inspect

    legs_attr = getattr(transport_request, "legs", None)
    if legs_attr is not None:
        return sorted(legs_attr, key=lambda leg: getattr(leg, "sequence_index", 0))

    if sa_inspect(transport_request, raiseerr=False) is not None:
        return sorted(
            transport_request.legs,
            key=lambda leg: getattr(leg, "sequence_index", 0),
        )

    return []


def _find_return_leg(
    transport_request: TransportRequest,
    legs: list[TransportRequestLeg],
) -> TransportRequestLeg | None:
    """Retourne le leg retour institution (is_return_stop ou dernier leg A/R)."""
    for leg in reversed(legs):
        if getattr(leg, "is_return_stop", False):
            return leg
    if getattr(transport_request, "return_to_institution", False) and len(legs) >= 2:
        return legs[-1]
    return None


def _should_use_legs_conversion(transport_request: TransportRequest) -> bool:
    """Parcours multi-segments : conversion atomique 1 booking par leg."""
    if getattr(transport_request, "multi_stop", False):
        return True
    legs = _load_transport_request_legs(transport_request)
    if len(legs) >= 2:
        return True
    return bool(any(getattr(leg, "is_return_stop", False) for leg in legs))


def _normalize_institution_name(value: str | None) -> str:
    """Normalise un nom d'institution pour matching tolérant."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = (
        normalized.lower().replace("’", "'").replace("`", "'").replace("´", "'")
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _track_accept_conflict(
    *,
    offer_id: int,
    company_id: int,
    transport_request_id: int | None,
    reason: str,
) -> None:
    try:
        from services.metrics.institution_metrics import track_offer_accept_conflict_409

        track_offer_accept_conflict_409(
            offer_id=offer_id,
            company_id=company_id,
            transport_request_id=transport_request_id,
            reason=reason,
        )
    except Exception:
        pass


def _build_conversion_institution_snapshot(
    transport_request: TransportRequest,
) -> dict[str, object]:
    """Snapshot institution figé à la conversion (V2 — payload request_converted)."""
    try:
        from models import Institution
        from services.institutions.mission_report_context import (
            build_institution_snapshot,
        )

        institution = getattr(transport_request, "institution", None)
        if institution is None and transport_request.institution_id:
            institution = Institution.query.get(transport_request.institution_id)
        if institution is None:
            return {}
        return build_institution_snapshot(transport_request, institution)
    except Exception as err:
        logger.warning(
            "[AcceptOffer] institution_snapshot capture failed: %s",
            err,
        )
        return {}


@dataclass(frozen=True, slots=True)
class AcceptOfferInput:
    """Input pour l'acceptation d'une offre."""

    offer_id: int
    company_id: int
    user_id: int  # Utilisateur company qui accepte
    proposed_pickup_time: datetime | None = None  # Horaire proposé par l'entreprise


@dataclass(frozen=True, slots=True)
class AcceptOfferResult:
    """Résultat de l'acceptation d'une offre."""

    success: bool
    offer_id: int
    booking_id: int | None = None
    return_booking_id: int | None = None
    transport_request_id: int | None = None
    error: str | None = None
    error_code: str | None = None
    status_code: int = 200


def _map_request_status_conflict_code(status: str) -> str:
    from models.enums import RequestStatus

    if status == RequestStatus.CANCELLED.value:
        return "REQUEST_CANCELLED"
    if status == RequestStatus.CONVERTED.value:
        return "REQUEST_CONVERTED"
    return "REQUEST_NOT_SENT"


def _map_offer_status_conflict_code(status: str) -> tuple[str, int]:
    from models.enums import OfferStatus

    if status == OfferStatus.ACCEPTED.value:
        return "OFFER_ALREADY_ACCEPTED", 409
    if status == OfferStatus.REJECTED.value:
        return "OFFER_REJECTED", 409
    if status == OfferStatus.UNAVAILABLE.value:
        return "OFFER_UNAVAILABLE", 409
    if status == OfferStatus.EXPIRED.value:
        return "OFFER_EXPIRED", 410
    return "REQUEST_NOT_SENT", 409


class AcceptOfferUseCase:
    """Use case: Accepter une offre de transport et créer le booking."""

    def execute(self, input_data: AcceptOfferInput) -> AcceptOfferResult:
        """
        Accepte une offre de transport (atomique: first accept wins).

        Étapes:
        1. Verrouiller la TransportRequest (SELECT FOR UPDATE)
        2. Vérifier que l'offre est valide et peut être acceptée
        3. Marquer l'offre comme ACCEPTED
        4. Marquer les autres offres comme UNAVAILABLE
        5. Mettre à jour la TransportRequest (ACCEPTED -> CONVERTED)
        6. Créer le Booking
        7. Audit logs

        Args:
            input_data: Données d'entrée

        Returns:
            AcceptOfferResult avec booking_id si succès
        """
        try:
            # 1. Charger l'offre avec vérification ownership
            offer = RequestOffer.query.get(input_data.offer_id)
            if not offer:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Offre introuvable",
                    status_code=404,
                )

            if offer.company_id != input_data.company_id:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Vous ne pouvez pas accepter cette offre",
                    status_code=403,
                )

            company = Company.query.get(input_data.company_id)
            if not company or not company.is_approved:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Entreprise non approuvée",
                    status_code=403,
                )

            from services.platform_billing.capabilities import (
                ERROR_BILLING_ACCESS_RESTRICTED,
                BillingCapability,
                is_billing_capability_allowed,
            )

            if not is_billing_capability_allowed(
                company.id, BillingCapability.ACCEPT_MARKETPLACE_OFFERS
            ):
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error=(
                        "Accès marketplace restreint pour cause de facturation "
                        f"({ERROR_BILLING_ACCESS_RESTRICTED})"
                    ),
                    status_code=403,
                )

            # 2. Verrouiller la TransportRequest (FOR UPDATE)
            transport_request = (
                db.session.query(TransportRequest)
                .filter(TransportRequest.id == offer.transport_request_id)
                .with_for_update()
                .first()
            )

            if not transport_request:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Demande de transport introuvable",
                    status_code=404,
                )

            # 3. Vérifier le statut de la demande (doit être SENT)
            if transport_request.status not in [
                RequestStatus.SENT.value,
                RequestStatus.ACCEPTED.value,
            ]:
                _track_accept_conflict(
                    offer_id=input_data.offer_id,
                    company_id=input_data.company_id,
                    transport_request_id=transport_request.id,
                    reason=f"request_status_{transport_request.status}",
                )
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error=f"Demande en statut {transport_request.status}, acceptation impossible",
                    error_code=_map_request_status_conflict_code(
                        transport_request.status
                    ),
                    status_code=409,
                )

            from models.enums import CarrierSource

            if transport_request.carrier_source == CarrierSource.EXTERNAL.value:
                _track_accept_conflict(
                    offer_id=input_data.offer_id,
                    company_id=input_data.company_id,
                    transport_request_id=transport_request.id,
                    reason="carrier_source_external",
                )
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error=(
                        "Demande basculée vers un transporteur externe, "
                        "acceptation impossible"
                    ),
                    error_code="CARRIER_EXTERNAL",
                    status_code=409,
                )

            # 4. Vérifier le statut de l'offre (doit être PENDING)
            # Recharger l'offre dans la même transaction
            offer = (
                db.session.query(RequestOffer)
                .filter(RequestOffer.id == input_data.offer_id)
                .with_for_update()
                .first()
            )
            if offer is None:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    error="Offre introuvable",
                    status_code=404,
                )

            if offer.status != OfferStatus.PENDING.value:
                _track_accept_conflict(
                    offer_id=input_data.offer_id,
                    company_id=input_data.company_id,
                    transport_request_id=transport_request.id,
                    reason=f"offer_status_{offer.status}",
                )
                conflict_code, conflict_status = _map_offer_status_conflict_code(
                    offer.status
                )
                booking_id = (
                    transport_request.booking_id
                    if conflict_code in {"OFFER_ALREADY_ACCEPTED", "REQUEST_CONVERTED"}
                    else None
                )
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    booking_id=booking_id,
                    error=f"Offre en statut {offer.status}, acceptation impossible",
                    error_code=conflict_code,
                    status_code=conflict_status,
                )

            # 5. Vérifier si l'offre a expiré
            if offer.is_expired:
                offer.mark_expired()
                db.session.commit()
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error="Offre expirée",
                    error_code="OFFER_EXPIRED",
                    status_code=410,
                )

            # 5b. Valider / Planifier — garde-fou horaire (avant point de non-retour)
            from services.institutions.offer_accept_rules import (
                validate_accept_pickup_rules,
            )

            pickup_rule_error = validate_accept_pickup_rules(
                transport_request,
                proposed_pickup_time=input_data.proposed_pickup_time,
            )
            if pickup_rule_error:
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error=pickup_rule_error,
                    error_code="PROPOSED_PICKUP_REQUIRED",
                    status_code=422,
                )

            # === POINT DE NON-RETOUR: First Accept Wins ===

            now = datetime.now(UTC)

            # 6. Marquer l'offre comme ACCEPTED
            offer.accept()

            # 7. Marquer les autres offres comme UNAVAILABLE
            other_offers = RequestOffer.query.filter(
                RequestOffer.transport_request_id == transport_request.id,
                RequestOffer.id != offer.id,
                RequestOffer.status == OfferStatus.PENDING.value,
            ).all()

            for other_offer in other_offers:
                other_offer.mark_unavailable()

            # 8. Mettre à jour la TransportRequest
            transport_request.status = RequestStatus.ACCEPTED.value
            transport_request.accepted_at = now
            transport_request.accepted_by_company_id = input_data.company_id

            if input_data.proposed_pickup_time is not None:
                from models.enums import ScheduledTimeType

                # Règle d'architecture : écriture mission institution → normalize_mission_wall_clock.
                # Idempotent sur naïf Genève déjà validé par validate_proposed_pickup_time.
                transport_request.scheduled_time = normalize_mission_wall_clock(
                    input_data.proposed_pickup_time
                )
                transport_request.scheduled_time_type = (
                    ScheduledTimeType.DEPARTURE.value
                )
                transport_request.pickup_time_confirmed = True

            # 9. Créer le(s) Booking(s)
            if _should_use_legs_conversion(transport_request):
                booking, return_booking = self._create_bookings_from_legs(
                    transport_request=transport_request,
                    company_id=input_data.company_id,
                    user_id=input_data.user_id,
                    proposed_pickup_time=input_data.proposed_pickup_time,
                )
                return_booking_id = return_booking.id if return_booking else None
            else:
                booking, return_booking = self._create_booking_from_request(
                    transport_request=transport_request,
                    company_id=input_data.company_id,
                    user_id=input_data.user_id,
                    proposed_pickup_time=input_data.proposed_pickup_time,
                )
                return_booking_id = return_booking.id if return_booking else None

            # Synchroniser le départ confirmé depuis le booking principal
            booking_scheduled = cast(datetime | None, booking.scheduled_time)
            if booking_scheduled is not None:
                from models.enums import ScheduledTimeType

                transport_request.scheduled_time = booking_scheduled
                transport_request.scheduled_time_type = (
                    ScheduledTimeType.DEPARTURE.value
                )
                transport_request.pickup_time_confirmed = True

            # 10. Lier le booking à la request et marquer comme CONVERTED
            transport_request.booking_id = booking.id
            transport_request.status = RequestStatus.CONVERTED.value
            transport_request.converted_at = now

            # Timeline transport: offer_accepted -> request_converted -> booking_created
            self._record_accept_timeline(
                transport_request=transport_request,
                offer=offer,
                booking=booking,
                company=company,
                user_id=input_data.user_id,
                company_id=input_data.company_id,
            )

            db.session.commit()

            # 11. Audit logs
            try:
                # Log acceptation
                AuditLogger.log_action(
                    action_type="offer_accepted",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="company",
                    company_id=input_data.company_id,
                    institution_id=transport_request.institution_id,
                    result_status="success",
                    action_details={
                        "offer_id": offer.id,
                        "transport_request_id": transport_request.id,
                        "other_offers_unavailable": len(other_offers),
                    },
                )

                # Log conversion
                AuditLogger.log_action(
                    action_type="request_converted",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="company",
                    company_id=input_data.company_id,
                    institution_id=transport_request.institution_id,
                    result_status="success",
                    action_details={
                        "transport_request_id": transport_request.id,
                        "booking_id": booking.id,
                        "return_booking_id": return_booking_id,
                    },
                )

                # Log création booking
                AuditLogger.log_action(
                    action_type="booking_created_from_request",
                    action_category="booking",
                    user_id=input_data.user_id,
                    user_type="company",
                    company_id=input_data.company_id,
                    booking_id=booking.id,
                    result_status="success",
                    action_details={
                        "source_transport_request_id": transport_request.id,
                        "source_institution_id": transport_request.institution_id,
                        "return_booking_id": return_booking_id,
                    },
                )
            except Exception as audit_err:
                logger.warning("Échec audit log: %s", audit_err)

            logger.info(
                "[AcceptOffer] Offer %s accepted by company %s -> Booking %s created",
                offer.id,
                input_data.company_id,
                booking.id,
            )

            # GO-LIVE: Métriques métier (délai send → accept)
            try:
                from services.metrics.institution_metrics import track_accept_event

                send_to_accept_seconds = None
                sent_at = getattr(transport_request, "sent_at", None)
                converted_at = getattr(transport_request, "converted_at", None)

                if sent_at is not None and converted_at is not None:
                    send_to_accept_seconds = (converted_at - sent_at).total_seconds()
                elif sent_at is not None:
                    # Fallback: utiliser now si converted_at pas encore set
                    send_to_accept_seconds = (
                        datetime.now(UTC) - sent_at
                    ).total_seconds()

                if send_to_accept_seconds is not None:
                    track_accept_event(
                        offer_id=offer.id,
                        transport_request_id=transport_request.id,
                        company_id=input_data.company_id,
                        send_to_accept_seconds=send_to_accept_seconds,
                    )
            except Exception as metric_err:
                logger.warning("[AcceptOffer] Error tracking metric: %s", metric_err)

            # ÉTAPE 5: Émettre événements temps réel vers l'institution
            try:
                from services.events.institution_events import (
                    emit_offer_accepted,
                    emit_request_converted,
                )

                # Événement offer_accepted
                emit_offer_accepted(
                    institution_id=transport_request.institution_id,
                    request_id=transport_request.id,
                    public_id=str(transport_request.public_id),
                    offer_id=offer.id,
                    company_name=offer.company.name if offer.company else None,
                )

                # Événement request_converted
                emit_request_converted(
                    institution_id=transport_request.institution_id,
                    request_id=transport_request.id,
                    public_id=str(transport_request.public_id),
                    booking_id=booking.id,
                    company_id=input_data.company_id,
                    company_name=offer.company.name if offer.company else None,
                )
            except Exception as event_err:
                logger.warning("[AcceptOffer] Error emitting events: %s", event_err)

            # Notifier les autres entreprises (broadcast) que l'offre n'est plus disponible
            try:
                from services.events.institution_events import emit_offer_unavailable

                accepted_company_name = offer.company.name if offer.company else None
                for other_offer in other_offers:
                    emit_offer_unavailable(
                        company_id=other_offer.company_id,
                        offer_id=other_offer.id,
                        transport_request=transport_request,
                        reason="accepted_by_peer",
                        accepted_by_company_id=input_data.company_id,
                        accepted_by_company_name=accepted_company_name,
                    )
            except Exception as unavailable_err:
                logger.warning(
                    "[AcceptOffer] Error emitting offer_unavailable: %s",
                    unavailable_err,
                )

            return AcceptOfferResult(
                success=True,
                offer_id=offer.id,
                booking_id=booking.id,
                return_booking_id=return_booking_id,
                transport_request_id=transport_request.id,
            )

        except Exception as e:
            logger.exception(
                "Erreur lors de l'acceptation de l'offre %s",
                input_data.offer_id,
            )
            db.session.rollback()
            return AcceptOfferResult(
                success=False,
                offer_id=input_data.offer_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )

    @staticmethod
    def _record_accept_timeline(
        *,
        transport_request: TransportRequest,
        offer: RequestOffer,
        booking: Booking,
        company: Company | None,
        user_id: int,
        company_id: int,
    ) -> None:
        """Historise l'acceptation, la conversion et la création du booking."""
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                find_latest_event,
                record_event,
            )

            company_name = company.name if company else None
            actor = TimelineActor(
                actor_type="company",
                actor_user_id=user_id,
                company_id=company_id,
            )

            # Chaîne offer_accepted -> offer_sent correspondant
            offer_sent_event = find_latest_event(
                transport_request_id=transport_request.id,
                event_type="offer_sent",
                company_id=company_id,
            )
            accepted_event = record_event(
                "offer_accepted",
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                actor=actor,
                payload={
                    "company_id": company_id,
                    "company_name": company_name,
                    "offer_id": offer.id,
                },
                correlation_id=f"offer_accepted:{offer.id}",
                source_event_id=offer_sent_event.id if offer_sent_event else None,
            )

            converted_event = record_event(
                "request_converted",
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                booking_id=booking.id,
                actor=actor,
                payload={
                    "booking_id": booking.id,
                    "company_id": company_id,
                    "company_name": company_name,
                    "institution_snapshot": _build_conversion_institution_snapshot(
                        transport_request
                    ),
                },
                correlation_id=f"request_converted:{transport_request.id}",
                source_event_id=accepted_event.id if accepted_event else None,
            )

            record_event(
                "booking_created",
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                booking_id=booking.id,
                actor=actor,
                payload={"booking_id": booking.id, "company_id": company_id},
                correlation_id=f"booking_created:{booking.id}",
                source_event_id=converted_event.id if converted_event else None,
            )
        except Exception as timeline_err:
            logger.warning("[AcceptOffer] Timeline recording failed: %s", timeline_err)

    def _create_booking_from_request(
        self,
        transport_request: TransportRequest,
        company_id: int,
        user_id: int,
        proposed_pickup_time: datetime | None = None,
    ) -> tuple[Booking, Booking | None]:
        """Crée un Booking (+ retour si A/R) à partir d'une TransportRequest.

        Args:
            transport_request: Demande de transport source
            company_id: ID de l'entreprise qui accepte
            user_id: ID de l'utilisateur qui crée le booking

        Returns:
            Tuple (outbound_booking, return_booking | None)
        """
        # Résoudre ou créer le client institution dans la base de l'entreprise
        institution_client = self._get_or_create_institution_client(
            transport_request, company_id
        )

        billing_intent = (
            getattr(transport_request, "billing_intent", None) or "patient"
        ).lower()

        # Tarif selon payeur effectif — cohérent avec estimate_offer_price.
        price = resolve_institution_price(
            company_id=company_id,
            effective_billing_intent=billing_intent,
            preferential_rate=getattr(institution_client, "preferential_rate", None),
            pickup_location=transport_request.pickup_location,
            dropoff_location=transport_request.dropoff_location,
            pickup_lat=float(transport_request.pickup_lat)
            if transport_request.pickup_lat
            else None,
            pickup_lon=float(transport_request.pickup_lng)
            if transport_request.pickup_lng
            else None,
            dropoff_lat=float(transport_request.dropoff_lat)
            if transport_request.dropoff_lat
            else None,
            dropoff_lon=float(transport_request.dropoff_lng)
            if transport_request.dropoff_lng
            else None,
            scheduled_time=cast(datetime | None, transport_request.scheduled_time),
            is_round_trip=transport_request.is_round_trip,
        )
        amount = price["amount"]

        # Facturation: la demande (billing_intent) est la source de vérité.
        billed_to_type = "patient"
        billed_to_company_id = None
        if billing_intent == "institution":
            billed_to_type = "clinic"
            billed_to_company_id = None
        elif billing_intent == "patient":
            billed_to_type = "patient"
            billed_to_company_id = None
        else:
            logger.warning(
                "[AcceptOffer] billing_intent non géré '%s' pour request=%s. Fallback sur patient.",
                billing_intent,
                transport_request.id,
            )
            billed_to_type = "patient"
            billed_to_company_id = None

        billed_to_company_id = self._resolve_billed_to_company_id_before_flush(
            billed_to_type=billed_to_type,
            institution_client=institution_client,
            transport_request=transport_request,
            company_id=company_id,
        )

        # Horaire: proposed_pickup_time ou départ institutionnel confirmé uniquement
        from services.institutions.offer_accept_rules import has_confirmed_departure

        raw_pickup: datetime | None = proposed_pickup_time
        if raw_pickup is None and has_confirmed_departure(transport_request):
            raw_pickup = cast(datetime | None, transport_request.scheduled_time)
        effective_pickup_time = (
            normalize_mission_wall_clock(raw_pickup) if raw_pickup is not None else None
        )

        booking = Booking(
            # Identité
            company_id=company_id,
            user_id=user_id,
            client_id=institution_client.id if institution_client else None,
            # Nom client (patient pour l'affichage)
            customer_name=self._get_customer_name(transport_request),
            # Type de mission
            mission_type=transport_request.mission_type,
            delivery_description=transport_request.delivery_description,
            # Horaires
            scheduled_time=effective_pickup_time,
            is_round_trip=transport_request.is_round_trip,
            # Lieux pickup
            pickup_location=transport_request.pickup_location,
            pickup_lat=float(transport_request.pickup_lat)
            if transport_request.pickup_lat
            else None,
            pickup_lon=float(transport_request.pickup_lng)
            if transport_request.pickup_lng
            else None,
            pickup_access_notes=self._format_pickup_notes(transport_request),
            pickup_floor=transport_request.pickup_floor,
            pickup_door_code=transport_request.pickup_door_code,
            # Lieux dropoff
            dropoff_location=transport_request.dropoff_location,
            dropoff_lat=float(transport_request.dropoff_lat)
            if transport_request.dropoff_lat
            else None,
            dropoff_lon=float(transport_request.dropoff_lng)
            if transport_request.dropoff_lng
            else None,
            dropoff_access_notes=self._format_dropoff_notes(transport_request),
            dropoff_floor=transport_request.dropoff_floor,
            dropoff_door_code=transport_request.dropoff_door_code,
            # Mobilité
            wheelchair_client_has=self._get_mobility_flag(
                transport_request, "wheelchair"
            ),
            wheelchair_need=self._get_mobility_flag(transport_request, "stretcher"),
            # Notes
            notes_medical=transport_request.notes,
            # Facturation — billed_to_type d'abord, company_id ajouté après
            billed_to_type=billed_to_type,
            billed_to_company_id=billed_to_company_id,
            status=BookingStatus.ACCEPTED.value,
            # Montant: tarif préférentiel ou minimum par défaut
            amount=amount,
            created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            institution_patient_id=getattr(transport_request, "patient_id", None),
        )
        from services.platform_billing.billing_origin import apply_origin_on_booking

        apply_origin_on_booking(
            booking,
            created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            is_institution_flow=True,
        )

        self._apply_clinical_dropoff_from_request(booking, transport_request)

        # Gel tarifaire : conserver le profil/version/détail si calculé via profil
        if price.get("pricing_profile_id"):
            booking.pricing_profile_id = price["pricing_profile_id"]
            booking.pricing_profile_version_id = price.get("pricing_profile_version_id")
            booking.price_amount = amount
            booking.price_breakdown_json = price.get("breakdown")

        db.session.add(booking)
        db.session.flush()  # Pour obtenir l'ID

        # Logger les metadata source (le modèle Booking n'a pas de colonne metadata_json)
        source_meta = self._build_metadata(
            transport_request,
            proposed_pickup_time=proposed_pickup_time,
        )
        logger.info(
            "[AcceptOffer] Booking %s created from institution request. Meta: %s",
            booking.id,
            source_meta,
        )

        # ── Résolution BillingParty (P0.5: résolution complète intent→BP) ──
        self._resolve_billing_party(
            booking,
            transport_request,
            company_id,
            strict=self._is_institution_billing_intent(transport_request),
        )
        if self._is_institution_billing_intent(transport_request):
            from services.billing.booking_billing_guard import (
                assert_non_patient_billing_complete,
            )

            assert_non_patient_billing_complete(
                booking,
                context="acceptation aller institution",
                require_billing_party_for_clinic=True,
            )

        # ── A/R: créer le booking retour si round-trip ──
        return_booking: Booking | None = None

        logger.info(
            "[AcceptOffer] Round-trip check: is_round_trip=%s, return_time=%s, request_id=%s",
            transport_request.is_round_trip,
            transport_request.return_time,
            transport_request.id,
        )

        return_time_raw = getattr(transport_request, "return_time", None)
        return_date_raw = getattr(transport_request, "return_date", None)
        return_time_confirmed_flag = bool(
            getattr(transport_request, "return_time_confirmed", False)
        )
        return_time_naive = (
            normalize_mission_wall_clock(return_time_raw)
            if return_time_raw is not None
            else None
        )

        has_return_plan = transport_request.is_round_trip and (
            return_time_naive is not None or return_date_raw is not None
        )

        if transport_request.is_round_trip and not has_return_plan:
            logger.warning(
                "[AcceptOffer] A/R sans return_date ni return_time, skip retour request_id=%s",
                transport_request.id,
            )
        elif has_return_plan:
            if return_time_naive is not None:
                ret_scheduled = return_time_naive
                ret_time_confirmed = return_time_confirmed_flag
            else:
                ret_scheduled = None
                ret_time_confirmed = False

            return_booking = Booking(
                company_id=company_id,
                user_id=user_id,
                client_id=booking.client_id,
                customer_name=booking.customer_name,
                mission_type=booking.mission_type,
                delivery_description=booking.delivery_description,
                scheduled_time=ret_scheduled,
                time_confirmed=ret_time_confirmed,
                is_round_trip=False,
                is_return=True,
                parent_booking_id=booking.id,
                # Pickup/dropoff inversés
                pickup_location=booking.dropoff_location,
                pickup_lat=booking.dropoff_lat,
                pickup_lon=booking.dropoff_lon,
                pickup_access_notes=booking.dropoff_access_notes,
                pickup_floor=booking.dropoff_floor,
                pickup_door_code=booking.dropoff_door_code,
                dropoff_location=booking.pickup_location,
                dropoff_lat=booking.pickup_lat,
                dropoff_lon=booking.pickup_lon,
                dropoff_access_notes=booking.pickup_access_notes,
                dropoff_floor=booking.pickup_floor,
                dropoff_door_code=booking.pickup_door_code,
                hospital_service=booking.hospital_service,
                wheelchair_client_has=booking.wheelchair_client_has,
                wheelchair_need=booking.wheelchair_need,
                notes_medical=booking.notes_medical,
                billed_to_type="patient",
                status=BookingStatus.ACCEPTED.value,
                amount=booking.amount,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
                institution_patient_id=booking.institution_patient_id
                or getattr(transport_request, "patient_id", None),
            )
            from services.platform_billing.billing_origin import apply_origin_on_booking

            apply_origin_on_booking(
                return_booking,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
                is_institution_flow=True,
            )
            legs = _load_transport_request_legs(transport_request)
            return_leg = _find_return_leg(transport_request, legs)
            return_effective_intent = self._apply_effective_billing_for_leg(
                return_booking,
                return_leg,
                transport_request,
                company_id,
                institution_client,
            )

            # Le retour reprend le même gel tarifaire que l'aller
            return_booking.pricing_profile_id = booking.pricing_profile_id
            return_booking.pricing_profile_version_id = (
                booking.pricing_profile_version_id
            )
            return_booking.price_amount = booking.price_amount
            return_booking.price_breakdown_json = booking.price_breakdown_json

            db.session.add(return_booking)
            db.session.flush()

            self._finalize_booking_billing_resolution(
                return_booking,
                transport_request,
                company_id,
                return_effective_intent,
                context="retour A/R legacy",
            )

            logger.info(
                "[AcceptOffer] Return booking %s created (parent=%s, time_confirmed=%s) for request %s",
                return_booking.id,
                booking.id,
                ret_time_confirmed,
                transport_request.id,
            )

        if return_booking is None:
            return_booking = self._create_return_booking_for_institution_return(
                transport_request=transport_request,
                outbound_booking=booking,
                company_id=company_id,
                user_id=user_id,
            )

        return booking, return_booking

    def _create_return_booking_for_institution_return(
        self,
        transport_request: TransportRequest,
        outbound_booking: Booking,
        company_id: int,
        user_id: int,
    ) -> Booking | None:
        """Crée le booking retour pour A/R multi-étapes (return_to_institution / legs).

        Filet de sécurité lorsque la demande n'a pas is_round_trip mais possède
        un leg retour ou return_date (modèle multi_stop institution).
        """
        if getattr(transport_request, "is_round_trip", False):
            return None

        return_to_inst = bool(
            getattr(transport_request, "return_to_institution", False)
        )
        return_date_raw = getattr(transport_request, "return_date", None)
        if not return_to_inst and return_date_raw is None:
            legs = _load_transport_request_legs(transport_request)
            if not any(getattr(leg, "is_return_stop", False) for leg in legs):
                return None
        else:
            legs = _load_transport_request_legs(transport_request)
        return_leg = _find_return_leg(transport_request, legs)

        if not return_to_inst and return_leg is None:
            return None

        if return_leg is None and return_date_raw is None:
            logger.warning(
                "[AcceptOffer] return_to_institution sans leg retour ni return_date, skip retour request_id=%s",
                transport_request.id,
            )
            return None

        institution_client = self._get_or_create_institution_client(
            transport_request, company_id
        )

        leg_confirmed = (
            bool(getattr(return_leg, "time_confirmed", False)) if return_leg else False
        )
        return_time_raw = (
            getattr(return_leg, "scheduled_time", None) if return_leg else None
        )
        if return_time_raw is None:
            return_time_raw = getattr(transport_request, "return_time", None)

        return_time_naive = (
            normalize_mission_wall_clock(return_time_raw)
            if return_time_raw is not None
            else None
        )

        if return_time_naive is not None and leg_confirmed:
            ret_scheduled = return_time_naive
            ret_time_confirmed = True
        else:
            ret_scheduled = None
            ret_time_confirmed = False

        if return_leg is not None:
            pickup_location = return_leg.pickup_location
            pickup_lat = float(return_leg.pickup_lat) if return_leg.pickup_lat else None
            pickup_lon = float(return_leg.pickup_lng) if return_leg.pickup_lng else None
            dropoff_location = return_leg.dropoff_location
            dropoff_lat = (
                float(return_leg.dropoff_lat) if return_leg.dropoff_lat else None
            )
            dropoff_lon = (
                float(return_leg.dropoff_lng) if return_leg.dropoff_lng else None
            )
        else:
            pickup_location = outbound_booking.dropoff_location
            pickup_lat = outbound_booking.dropoff_lat
            pickup_lon = outbound_booking.dropoff_lon
            dropoff_location = outbound_booking.pickup_location
            dropoff_lat = outbound_booking.pickup_lat
            dropoff_lon = outbound_booking.pickup_lon

        return_booking = Booking(
            company_id=company_id,
            user_id=user_id,
            client_id=outbound_booking.client_id,
            customer_name=outbound_booking.customer_name,
            mission_type=outbound_booking.mission_type,
            delivery_description=outbound_booking.delivery_description,
            scheduled_time=ret_scheduled,
            time_confirmed=ret_time_confirmed,
            is_round_trip=False,
            is_return=True,
            parent_booking_id=outbound_booking.id,
            pickup_location=pickup_location,
            pickup_lat=pickup_lat,
            pickup_lon=pickup_lon,
            pickup_access_notes=outbound_booking.dropoff_access_notes,
            pickup_floor=outbound_booking.dropoff_floor,
            pickup_door_code=outbound_booking.dropoff_door_code,
            dropoff_location=dropoff_location,
            dropoff_lat=dropoff_lat,
            dropoff_lon=dropoff_lon,
            dropoff_access_notes=outbound_booking.pickup_access_notes,
            dropoff_floor=outbound_booking.pickup_floor,
            dropoff_door_code=outbound_booking.pickup_door_code,
            hospital_service=outbound_booking.hospital_service,
            wheelchair_client_has=outbound_booking.wheelchair_client_has,
            wheelchair_need=outbound_booking.wheelchair_need,
            notes_medical=outbound_booking.notes_medical,
            billed_to_type="patient",
            status=BookingStatus.ACCEPTED.value,
            amount=outbound_booking.amount,
            created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            institution_patient_id=outbound_booking.institution_patient_id
            or getattr(transport_request, "patient_id", None),
        )
        from services.platform_billing.billing_origin import apply_origin_on_booking

        apply_origin_on_booking(
            return_booking,
            created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            is_institution_flow=True,
        )
        return_effective_intent = self._apply_effective_billing_for_leg(
            return_booking,
            return_leg,
            transport_request,
            company_id,
            institution_client,
        )

        return_booking.pricing_profile_id = outbound_booking.pricing_profile_id
        return_booking.pricing_profile_version_id = (
            outbound_booking.pricing_profile_version_id
        )
        return_booking.price_amount = outbound_booking.price_amount
        return_booking.price_breakdown_json = outbound_booking.price_breakdown_json

        if return_leg is not None:
            self._apply_clinical_dropoff_from_leg(return_booking, return_leg)

        db.session.add(return_booking)
        db.session.flush()

        self._finalize_booking_billing_resolution(
            return_booking,
            transport_request,
            company_id,
            return_effective_intent,
            context="retour institution multi-étapes",
        )

        if legs:
            outbound_leg = next(
                (leg for leg in legs if leg.sequence_index == 0),
                legs[0],
            )
            if outbound_leg.booking_id is None:
                outbound_leg.booking_id = outbound_booking.id
            if return_leg is not None:
                return_leg.booking_id = return_booking.id

        logger.info(
            "[AcceptOffer] Institution return booking %s created (parent=%s, bp_id=%s, billed_to_company_id=%s, time_confirmed=%s, from_leg=%s) for request %s",
            return_booking.id,
            outbound_booking.id,
            return_booking.billing_party_id,
            return_booking.billed_to_company_id,
            ret_time_confirmed,
            return_leg is not None,
            transport_request.id,
        )
        return return_booking

    def _create_bookings_from_legs(
        self,
        transport_request: TransportRequest,
        company_id: int,
        user_id: int,
        proposed_pickup_time: datetime | None = None,
    ) -> tuple[Booking, Booking | None]:
        """Conversion atomique multi-stop : 1 booking par leg."""
        from models.transport_request_leg import TransportRequestLeg
        from services.institutions.offer_accept_rules import has_confirmed_departure

        legs = (
            TransportRequestLeg.query.filter_by(
                transport_request_id=transport_request.id
            )
            .order_by(TransportRequestLeg.sequence_index.asc())
            .all()
        )
        if not legs:
            raise ValueError(
                f"Demande multi-stop {transport_request.id} sans legs configurés"
            )

        institution_client = self._get_or_create_institution_client(
            transport_request, company_id
        )
        preferential_rate = getattr(institution_client, "preferential_rate", None)

        from services.billing.destination_billing_resolver import (
            billed_to_type_from_intent,
            effective_billing_for_leg,
        )

        route_group_id = getattr(transport_request, "route_group_id", None)
        customer_name = self._get_customer_name(transport_request)
        primary: Booking | None = None
        return_leg_booking: Booking | None = None

        for leg in legs:
            effective_intent = effective_billing_for_leg(leg, transport_request)
            billed_to_type = billed_to_type_from_intent(effective_intent)
            billed_to_company_id = self._resolve_billed_to_company_id_before_flush(
                billed_to_type=billed_to_type,
                institution_client=institution_client,
                transport_request=transport_request,
                company_id=company_id,
            )

            is_first_leg = leg.sequence_index == 0
            leg_confirmed = bool(getattr(leg, "time_confirmed", False))
            mission_depart_confirmed = has_confirmed_departure(transport_request)
            raw_pickup: datetime | None
            if is_first_leg:
                # Ne jamais utiliser leg.scheduled_time (RDV) comme pickup sans départ confirmé.
                raw_pickup = proposed_pickup_time
                if raw_pickup is None and mission_depart_confirmed:
                    raw_pickup = cast(datetime | None, transport_request.scheduled_time)
                operational = (
                    mission_depart_confirmed or proposed_pickup_time is not None
                )
            else:
                raw_pickup = (
                    cast(datetime | None, leg.scheduled_time) if leg_confirmed else None
                )
                operational = leg_confirmed
            effective_pickup_time = (
                normalize_mission_wall_clock(raw_pickup)
                if raw_pickup is not None
                else None
            )

            time_to_define = effective_pickup_time is None or not operational

            is_return_leg = bool(getattr(leg, "is_return_stop", False)) or (
                bool(getattr(transport_request, "return_to_institution", False))
                and len(legs) > 1
                and leg.sequence_index == len(legs) - 1
            )
            # Retours / étapes suivantes : pas de blocage « heure passée » à l'acceptation.
            if not is_first_leg or is_return_leg:
                if not leg_confirmed:
                    effective_pickup_time = None
                    time_to_define = True
                elif effective_pickup_time is not None:
                    from shared.time_utils import now_local

                    is_sentinel_midnight = (
                        effective_pickup_time.hour == 0
                        and effective_pickup_time.minute == 0
                        and effective_pickup_time.second == 0
                    )
                    if is_sentinel_midnight or effective_pickup_time < now_local():
                        effective_pickup_time = None
                        time_to_define = True

            # Tarif par leg selon payeur effectif (override destination inclus)
            leg_price = resolve_institution_price(
                company_id=company_id,
                effective_billing_intent=effective_intent,
                preferential_rate=preferential_rate,
                pickup_location=leg.pickup_location,
                dropoff_location=leg.dropoff_location,
                pickup_lat=float(leg.pickup_lat) if leg.pickup_lat else None,
                pickup_lon=float(leg.pickup_lng) if leg.pickup_lng else None,
                dropoff_lat=float(leg.dropoff_lat) if leg.dropoff_lat else None,
                dropoff_lon=float(leg.dropoff_lng) if leg.dropoff_lng else None,
                scheduled_time=cast(datetime | None, transport_request.scheduled_time),
                is_round_trip=False,
            )
            amount = leg_price["amount"]

            booking = Booking(
                company_id=company_id,
                user_id=user_id,
                client_id=institution_client.id if institution_client else None,
                customer_name=customer_name,
                mission_type=transport_request.mission_type,
                delivery_description=transport_request.delivery_description,
                is_return=False,
                time_confirmed=not time_to_define,
                scheduled_time=effective_pickup_time,
                is_round_trip=False,
                pickup_location=leg.pickup_location,
                pickup_lat=float(leg.pickup_lat) if leg.pickup_lat else None,
                pickup_lon=float(leg.pickup_lng) if leg.pickup_lng else None,
                pickup_access_notes=self._format_pickup_notes(transport_request),
                pickup_floor=transport_request.pickup_floor,
                pickup_door_code=transport_request.pickup_door_code,
                dropoff_location=leg.dropoff_location,
                dropoff_lat=float(leg.dropoff_lat) if leg.dropoff_lat else None,
                dropoff_lon=float(leg.dropoff_lng) if leg.dropoff_lng else None,
                dropoff_access_notes=self._format_dropoff_notes(transport_request),
                dropoff_floor=transport_request.dropoff_floor,
                dropoff_door_code=transport_request.dropoff_door_code,
                wheelchair_client_has=self._get_mobility_flag(
                    transport_request, "wheelchair"
                ),
                wheelchair_need=self._get_mobility_flag(transport_request, "stretcher"),
                notes_medical=transport_request.notes,
                billed_to_type=billed_to_type,
                billed_to_company_id=billed_to_company_id,
                status=BookingStatus.ACCEPTED.value,
                amount=amount,
                route_group_id=route_group_id,
                route_sequence_number=leg.route_sequence_number,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
                institution_patient_id=getattr(transport_request, "patient_id", None),
            )
            from services.platform_billing.billing_origin import apply_origin_on_booking

            apply_origin_on_booking(
                booking,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
                is_institution_flow=True,
            )

            self._apply_clinical_dropoff_from_leg(booking, leg)

            # Gel tarifaire par leg si calculé via profil
            if leg_price.get("pricing_profile_id"):
                booking.pricing_profile_id = leg_price["pricing_profile_id"]
                booking.pricing_profile_version_id = leg_price.get(
                    "pricing_profile_version_id"
                )
                booking.price_amount = amount
                booking.price_breakdown_json = leg_price.get("breakdown")

            # NB : on NE rattache PAS les legs via parent_booking_id. La relation
            # `return_trip` (remote_side=[id], fk=parent_booking_id) interprète
            # parent_booking_id comme un lien aller/retour, ce qui ferait
            # apparaître les legs comme "Aller-retour" et fausserait
            # l'appariement de facturation A/R. Le regroupement du parcours est
            # assuré par `route_group_id` (suffisant pour l'annulation en cascade).

            db.session.add(booking)
            db.session.flush()
            leg.booking_id = booking.id
            self._resolve_billing_party(
                booking,
                transport_request,
                company_id,
                billing_intent_override=effective_intent,
            )

            if primary is None:
                primary = booking
            if is_return_leg:
                return_leg_booking = booking

        if primary is None:
            raise ValueError("Conversion multi-stop sans booking créé")

        logger.info(
            "[AcceptOffer] Multi-stop request %s -> %s bookings (route_group=%s)",
            transport_request.id,
            len(legs),
            route_group_id,
        )
        return primary, return_leg_booking

    @staticmethod
    def _resolve_billed_to_company_id_before_flush(
        *,
        billed_to_type: str,
        institution_client: Client | None,
        transport_request: TransportRequest,
        company_id: int,
    ) -> int | None:
        """Obligatoire avant flush si ``billed_to_type != patient`` (hook ORM Booking)."""
        from services.billing.institution_billing_resolver import (
            resolve_billed_to_company_id_for_accept,
        )

        btype = (billed_to_type or "patient").strip().lower()
        clinic_company_id = resolve_billed_to_company_id_for_accept(
            billed_to_type=btype,
            institution_client=institution_client,
            institution=transport_request.institution,
            transport_company_id=company_id,
        )
        if btype != "patient" and (
            clinic_company_id is None or int(clinic_company_id) <= 0
        ):
            inst_name = getattr(transport_request.institution, "name", None) or "?"
            msg = (
                f"Impossible de résoudre billed_to_company_id pour billed_to_type={btype} "
                f"(institution {inst_name})"
            )
            raise ValueError(msg)
        return clinic_company_id

    @staticmethod
    def _is_institution_billing_intent(transport_request: TransportRequest) -> bool:
        intent = (
            (getattr(transport_request, "billing_intent", None) or "patient")
            .strip()
            .lower()
        )
        return intent == "institution"

    def _apply_effective_billing_for_leg(
        self,
        booking: Booking,
        leg: TransportRequestLeg | None,
        transport_request: TransportRequest,
        company_id: int,
        institution_client: Client | None,
    ) -> str:
        """Applique billed_to_* depuis le payeur effectif du leg (ou intent global)."""
        from services.billing.destination_billing_resolver import (
            billed_to_type_from_intent,
            effective_billing_for_leg,
            resolve_effective_billing_intent,
        )

        if leg is not None:
            effective_intent = effective_billing_for_leg(leg, transport_request)
        else:
            effective_intent = resolve_effective_billing_intent(
                transport_request.billing_intent,
                None,
            )
        billed_to_type = billed_to_type_from_intent(effective_intent)
        booking.billed_to_type = billed_to_type
        booking.billed_to_company_id = self._resolve_billed_to_company_id_before_flush(
            billed_to_type=billed_to_type,
            institution_client=institution_client,
            transport_request=transport_request,
            company_id=company_id,
        )
        return effective_intent

    def _finalize_booking_billing_resolution(
        self,
        booking: Booking,
        transport_request: TransportRequest,
        company_id: int,
        effective_intent: str,
        *,
        context: str,
    ) -> None:
        """Résout BillingParty et valide la complétude selon le payeur effectif."""
        from services.billing.booking_billing_guard import (
            assert_non_patient_billing_complete,
        )
        from services.billing.destination_billing_resolver import (
            billed_to_type_from_intent,
        )

        requires_non_patient = billed_to_type_from_intent(effective_intent) != "patient"
        if booking.billing_party_id is None:
            self._resolve_billing_party(
                booking,
                transport_request,
                company_id,
                billing_intent_override=effective_intent,
                strict=requires_non_patient,
            )
        if requires_non_patient:
            assert_non_patient_billing_complete(
                booking,
                context=context,
                require_billing_party_for_clinic=True,
            )

    def _resolve_billing_party(
        self,
        booking: Booking,
        transport_request: TransportRequest,
        company_id: int,
        billing_intent_override: str | None = None,
        *,
        strict: bool = False,
    ) -> None:
        """Résout et attache le billing_party au booking."""
        from domain.billing.errors import BillingValidationError
        from services.billing.booking_billing_guard import billing_type_normalized

        try:
            from services.billing.institution_billing_resolver import (
                resolve_billing_party_for_institution_booking,
            )

            billing_result = resolve_billing_party_for_institution_booking(
                booking=booking,
                transport_request=transport_request,
                company_id=company_id,
                billing_intent_override=billing_intent_override,
            )

            res_status = billing_result.get("billing_resolution_status", "unknown")
            if billing_result.get("billing_party_id"):
                logger.info(
                    "[AcceptOffer] BillingParty resolved: booking=%s, bp_id=%s, status=%s, source=%s",
                    booking.id,
                    billing_result["billing_party_id"],
                    res_status,
                    billing_result.get("billing_resolution_source", ""),
                )
            elif res_status.startswith("failed"):
                logger.warning(
                    "[AcceptOffer] BillingParty NOT resolved: booking=%s, status=%s, intent=%s",
                    booking.id,
                    res_status,
                    billing_result.get("billing_intent", ""),
                )
            if strict:
                btype = billing_type_normalized(booking)
                if btype != "patient" and (
                    res_status.startswith("failed")
                    or not billing_result.get("billing_party_id")
                ):
                    raise BillingValidationError(
                        (
                            "Résolution BillingParty institution impossible "
                            f"(booking={booking.id}, status={res_status})."
                        ),
                        field="billing_party_id",
                    )
        except BillingValidationError:
            raise
        except Exception as billing_err:
            if strict:
                raise BillingValidationError(
                    f"Résolution BillingParty institution en échec: {billing_err}",
                    field="billing_party_id",
                ) from billing_err
            logger.warning(
                "[AcceptOffer] BillingParty resolution error (non-blocking): %s",
                billing_err,
            )

    def _get_or_create_institution_client(
        self,
        transport_request: TransportRequest,
        company_id: int,
    ) -> Client | None:
        """Retourne le client institution lié par FK, ou le crée automatiquement."""
        institution = transport_request.institution
        if not institution:
            return None

        client = Client.query.filter(
            Client.company_id == company_id,
            Client.is_institution.is_(True),
            Client.linked_institution_id == institution.id,
        ).first()

        if client:
            self._ensure_institution_client_clinic_company_link(client, institution)
            logger.info(
                "[AcceptOffer] Institution client found by FK link: %s (id=%s, rate=%s)",
                institution.name,
                client.id,
                client.preferential_rate,
            )
            return client

        fallback_by_name = self._find_institution_client_by_name(
            institution_name=getattr(institution, "name", None),
            company_id=company_id,
        )
        if fallback_by_name:
            if not getattr(fallback_by_name, "linked_institution_id", None):
                fallback_by_name.linked_institution_id = institution.id
            self._ensure_institution_client_clinic_company_link(
                fallback_by_name, institution
            )
            logger.info(
                "[AcceptOffer] Institution client found by name fallback: %s (id=%s, rate=%s, linked=%s)",
                institution.name,
                fallback_by_name.id,
                fallback_by_name.preferential_rate,
                fallback_by_name.linked_institution_id,
            )
            return fallback_by_name

        return self._create_institution_client(institution, company_id)

    @staticmethod
    def _ensure_institution_client_clinic_company_link(
        client: Client,
        institution: object,
    ) -> None:
        """Rattache default_billed_to_company_id si absent (clients institution historiques)."""
        if not getattr(client, "is_institution", False):
            return
        if getattr(client, "default_billed_to_company_id", None):
            return
        from services.billing.institution_billing_resolver import (
            resolve_clinic_company_id_for_institution_accept,
        )

        transport_company_id = int(getattr(client, "company_id", 0) or 0)
        if transport_company_id <= 0:
            return
        clinic_id = resolve_clinic_company_id_for_institution_accept(
            institution_client=client,
            institution=institution,  # type: ignore[arg-type]
            transport_company_id=transport_company_id,
        )
        if clinic_id is not None:
            client.default_billed_to_type = "clinic"
            client.default_billed_to_company_id = int(clinic_id)

    def _find_institution_client_by_name(
        self, institution_name: str | None, company_id: int
    ) -> Client | None:
        """Fallback robuste quand linked_institution_id est absent/incomplet."""
        normalized_target = _normalize_institution_name(institution_name)
        if not normalized_target:
            return None

        candidates = Client.query.filter(
            Client.company_id == company_id,
            Client.is_institution.is_(True),
        ).all()

        for candidate in candidates:
            candidate_name = getattr(candidate, "institution_name", None)
            normalized_candidate = _normalize_institution_name(candidate_name)
            if not normalized_candidate:
                continue
            if (
                normalized_candidate == normalized_target
                or normalized_candidate in normalized_target
                or normalized_target in normalized_candidate
            ):
                return candidate

        return None

    def _create_institution_client(
        self,
        institution: object,
        company_id: int,
    ) -> Client | None:
        """Crée un client institution dédié pour une entreprise de transport."""
        inst_id = getattr(institution, "id", None)
        if inst_id is None:
            return None
        inst_id_int = int(inst_id)
        inst_name = getattr(institution, "name", None) or f"Institution {inst_id_int}"

        # Important: création sous savepoint pour ne pas invalider la transaction
        # d'acceptation globale en cas de collision (email/user déjà existant).
        try:
            with db.session.begin_nested():
                inst_user = User()
                inst_user.username = f"inst_{inst_id}_{uuid.uuid4().hex[:8]}"
                # Compte technique: email local unique par institution+entreprise.
                # On conserve l'email métier dans Client.contact_email.
                inst_user.email = (
                    f"institution-{inst_id}-company-{company_id}@lirie.local"
                )
                inst_user.role = UserRole.CLIENT
                inst_user.password = "!auto_institution_client!"
                inst_user.first_name = inst_name[:120]
                inst_user.last_name = "Institution"
                db.session.add(inst_user)
                db.session.flush()

                new_client = Client()
                new_client.user_id = inst_user.id
                new_client.company_id = company_id
                new_client.is_institution = True
                new_client.institution_name = inst_name
                new_client.linked_institution_id = inst_id
                new_client.client_type = ClientType.TRANSPORT
                new_client.management_mode = ManagementMode.CORPORATE
                new_client.billing_address = getattr(
                    institution, "billing_address", None
                ) or getattr(institution, "address", None)
                new_client.contact_email = getattr(
                    institution, "billing_email", None
                ) or getattr(institution, "contact_email", None)
                new_client.contact_phone = getattr(institution, "contact_phone", None)
                inst_name_for_co = (inst_name or "").strip()
                clinic_co = (
                    Company.query.filter(
                        func.lower(Company.name) == func.lower(inst_name_for_co)
                    )
                    .order_by(Company.id.asc())
                    .first()
                    if inst_name_for_co
                    else None
                )
                if clinic_co is not None:
                    new_client.default_billed_to_type = "clinic"
                    new_client.default_billed_to_company_id = int(clinic_co.id)
                else:
                    # Par défaut, facturation au patient tant qu'aucun payeur tiers
                    # n'est explicitement configuré (évite invalidation booking).
                    new_client.default_billed_to_type = "patient"
                db.session.add(new_client)
                db.session.flush()

            logger.info(
                "institution_client_auto_created institution_id=%s client_id=%s company_id=%s",
                inst_id,
                new_client.id,
                company_id,
            )
            try:
                from services.metrics.institution_metrics import (
                    track_institution_client_auto_created,
                )

                track_institution_client_auto_created(
                    institution_id=inst_id_int,
                    client_id=new_client.id,
                    company_id=company_id,
                )
            except Exception:
                pass

            return new_client
        except IntegrityError as create_err:
            logger.warning(
                "[AcceptOffer] IntegrityError auto-create institution client (inst=%s, company=%s): %s",
                inst_id,
                company_id,
                create_err,
            )
            # Concurrence: un autre worker a peut-être créé le client entre-temps.
            existing = Client.query.filter(
                Client.company_id == company_id,
                Client.is_institution.is_(True),
                Client.linked_institution_id == inst_id,
            ).first()
            if existing:
                return existing
        except Exception as create_err:
            logger.exception(
                "[AcceptOffer] Failed to auto-create institution client for %s: %s",
                inst_name,
                create_err,
            )
        return None

    def _find_institution_client(
        self,
        transport_request: TransportRequest,
        company_id: int,
    ) -> Client | None:
        """Deprecated: use _get_or_create_institution_client."""
        return self._get_or_create_institution_client(transport_request, company_id)

    def _get_customer_name(self, transport_request: TransportRequest) -> str:
        """Retourne le nom du client pour le booking."""
        if transport_request.patient:
            return f"{transport_request.patient.first_name} {transport_request.patient.last_name}"
        if transport_request.institution:
            return transport_request.institution.name
        return "Client Institution"

    def _build_metadata(
        self,
        transport_request: TransportRequest,
        proposed_pickup_time: datetime | None = None,
    ) -> dict[str, object]:
        """Construit le metadata_json initial du booking."""
        meta: dict[str, object] = {
            "source": "institution",
            "source_transport_request_id": transport_request.id,
            "source_institution_id": transport_request.institution_id,
            "source_external_reference": transport_request.external_reference,
        }

        # Informations institution
        institution = transport_request.institution
        if institution:
            meta["institution"] = {
                "name": institution.name,
                "id": institution.id,
            }

        # Informations patient (accessibles depuis la demande)
        patient = transport_request.patient
        if patient:
            patient_info: dict[str, object] = {
                "institution_patient_id": patient.id,
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "phone": getattr(patient, "phone", None),
                "date_of_birth": (
                    patient.dob.isoformat() if getattr(patient, "dob", None) else None
                ),
            }
            # Adresse domicile (utile si facturé au patient)
            if getattr(patient, "address", None):
                patient_info["address"] = patient.address
            if getattr(patient, "avs_number", None):
                patient_info["avs_number"] = patient.avs_number
            if getattr(patient, "insurance_name", None):
                patient_info["insurance_name"] = patient.insurance_name
            meta["patient"] = patient_info

        # Facturation
        meta["billing_intent"] = transport_request.billing_intent

        # Horaire proposé par l'entreprise (différent de l'horaire initial)
        if proposed_pickup_time:
            meta["proposed_pickup_time"] = proposed_pickup_time.isoformat()
            original: datetime | None = getattr(
                transport_request, "scheduled_time", None
            )
            if original is not None:
                meta["original_scheduled_time"] = original.isoformat()

        # Routing: utiliser les colonnes directes (priorité) avec fallback billing_details.routing
        pickup_type = getattr(transport_request, "pickup_type", None)
        dropoff_type = getattr(transport_request, "dropoff_type", None)
        pickup_entry_point = getattr(transport_request, "pickup_entry_point", None)
        dropoff_entry_point = getattr(transport_request, "dropoff_entry_point", None)

        if pickup_type or dropoff_type or pickup_entry_point or dropoff_entry_point:
            meta["routing"] = {
                "pickup_type": pickup_type,
                "dropoff_type": dropoff_type,
                "pickup_entry_point": pickup_entry_point,
                "dropoff_entry_point": dropoff_entry_point,
            }
        else:
            # Fallback: copier routing depuis billing_details.routing (rétrocompatibilité)
            billing_details = transport_request.billing_details or {}
            routing = billing_details.get("routing")
            if routing and isinstance(routing, dict):
                meta["routing"] = routing

        return meta

    @staticmethod
    def _apply_clinical_dropoff_fields(
        booking: Booking,
        *,
        establishment: str | None,
        service: str | None,
        doctor: str | None,
    ) -> None:
        """Copie établissement / service / médecin de destination sur le booking."""
        if establishment and str(establishment).strip():
            booking.medical_facility = str(establishment).strip()
        if service and str(service).strip():
            booking.hospital_service = str(service).strip()
        if doctor and str(doctor).strip():
            booking.doctor_name = str(doctor).strip()

    @staticmethod
    def _clinical_dropoff_from_leg(
        leg: object,
    ) -> tuple[str | None, str | None, str | None]:
        return (
            getattr(leg, "dropoff_establishment", None),
            getattr(leg, "dropoff_service", None),
            getattr(leg, "dropoff_doctor", None),
        )

    def _clinical_dropoff_from_request(
        self, transport_request: TransportRequest
    ) -> tuple[str | None, str | None, str | None]:
        legs = sorted(
            getattr(transport_request, "legs", None) or [],
            key=lambda item: getattr(item, "sequence_index", 0),
        )
        if not legs:
            return None, None, None
        return self._clinical_dropoff_from_leg(legs[0])

    def _apply_clinical_dropoff_from_leg(self, booking: Booking, leg: object) -> None:
        est, svc, doc = self._clinical_dropoff_from_leg(leg)
        self._apply_clinical_dropoff_fields(
            booking, establishment=est, service=svc, doctor=doc
        )

    def _apply_clinical_dropoff_from_request(
        self, booking: Booking, transport_request: TransportRequest
    ) -> None:
        est, svc, doc = self._clinical_dropoff_from_request(transport_request)
        self._apply_clinical_dropoff_fields(
            booking, establishment=est, service=svc, doctor=doc
        )

    def _format_pickup_notes(self, transport_request: TransportRequest) -> str | None:
        """Formate les notes d'accès pickup."""
        parts = []

        # Entry point: colonnes directes (priorité) puis fallback billing_details.routing
        entry_point = getattr(transport_request, "pickup_entry_point", None) or ""
        if not entry_point:
            routing = (transport_request.billing_details or {}).get("routing", {})
            entry_point = (
                routing.get("pickup_entry_point", "")
                if isinstance(routing, dict)
                else ""
            )
        if entry_point:
            parts.append(f"Accueil: {entry_point}")

        routing = (transport_request.billing_details or {}).get("routing", {})
        pickup_instructions = (
            routing.get("pickup_instructions", "") if isinstance(routing, dict) else ""
        )
        if pickup_instructions:
            parts.append(pickup_instructions)

        if transport_request.pickup_floor:
            parts.append(f"Étage: {transport_request.pickup_floor}")
        if transport_request.pickup_door_code:
            parts.append(f"Code: {transport_request.pickup_door_code}")
        if transport_request.floor_elevator_info:
            parts.append(transport_request.floor_elevator_info)
        contact = transport_request.contact_on_site
        if contact:
            contact_info = []
            # Support enriched structure (requester + onsite)
            if contact.get("onsite_is_different") and contact.get("onsite_name"):
                contact_info.append(contact["onsite_name"])
                if contact.get("onsite_phone"):
                    contact_info.append(contact["onsite_phone"])
            else:
                if contact.get("name"):
                    contact_info.append(contact["name"])
                if contact.get("phone"):
                    contact_info.append(contact["phone"])
            if contact_info:
                parts.append(f"Contact: {', '.join(contact_info)}")
        return "\n".join(parts) if parts else None

    def _format_dropoff_notes(self, transport_request: TransportRequest) -> str | None:
        """Formate les notes d'accès dropoff."""
        parts = []

        # Entry point: colonnes directes (priorité) puis fallback billing_details.routing
        entry_point = getattr(transport_request, "dropoff_entry_point", None) or ""
        if not entry_point:
            routing = (transport_request.billing_details or {}).get("routing", {})
            entry_point = (
                routing.get("dropoff_entry_point", "")
                if isinstance(routing, dict)
                else ""
            )
        if entry_point:
            parts.append(f"Accueil: {entry_point}")

        routing = (transport_request.billing_details or {}).get("routing", {})
        dropoff_instructions = (
            routing.get("dropoff_instructions", "") if isinstance(routing, dict) else ""
        )
        if dropoff_instructions:
            parts.append(dropoff_instructions)

        if transport_request.dropoff_floor:
            parts.append(f"Étage: {transport_request.dropoff_floor}")
        if transport_request.dropoff_door_code:
            parts.append(f"Code: {transport_request.dropoff_door_code}")
        return "\n".join(parts) if parts else None

    def _get_mobility_flag(
        self, transport_request: TransportRequest, flag: str
    ) -> bool:
        """Retourne un flag de mobilité."""
        mobility = transport_request.mobility or {}
        return bool(mobility.get(flag, False))
