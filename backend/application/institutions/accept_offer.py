# application/institutions/accept_offer.py
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false
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
    api_scheduled_iso_to_naive_geneva,
)

logger = logging.getLogger(__name__)


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
    status_code: int = 200


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

            if offer.status != OfferStatus.PENDING.value:
                _track_accept_conflict(
                    offer_id=input_data.offer_id,
                    company_id=input_data.company_id,
                    transport_request_id=transport_request.id,
                    reason=f"offer_status_{offer.status}",
                )
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error=f"Offre en statut {offer.status}, acceptation impossible",
                    status_code=409,
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
                    status_code=410,
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

                transport_request.scheduled_time = input_data.proposed_pickup_time
                transport_request.scheduled_time_type = (
                    ScheduledTimeType.DEPARTURE.value
                )
                transport_request.pickup_time_confirmed = True

            # 9. Créer le(s) Booking(s)
            if getattr(transport_request, "multi_stop", False):
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
            if booking is not None and booking.scheduled_time is not None:
                from models.enums import ScheduledTimeType

                transport_request.scheduled_time = booking.scheduled_time
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

        # Tarif: préférentiel (client institution) sinon repli sur le profil
        # tarifaire actif de l'entreprise. Cohérent avec l'estimation affichée
        # sur l'offre (resolve_institution_price), pour que montant = estimation.
        price = resolve_institution_price(
            company_id=company_id,
            preferential_rate=getattr(
                institution_client, "preferential_rate", None
            ),
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
            scheduled_time=transport_request.scheduled_time,
            is_round_trip=transport_request.is_round_trip,
        )
        amount = price["amount"]

        # Facturation: la demande (billing_intent) est la source de vérité.
        billed_to_type = "patient"
        billed_to_company_id = None
        billing_intent = (
            getattr(transport_request, "billing_intent", None) or "patient"
        ).lower()
        if billing_intent == "institution":
            billed_to_type = "clinic"
            billed_to_company_id = company_id
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

        # Horaire: utiliser l'horaire proposé par l'entreprise si fourni (naïf Genève)
        raw_pickup = proposed_pickup_time or transport_request.scheduled_time
        effective_pickup_time = api_scheduled_iso_to_naive_geneva(raw_pickup)

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
            status=BookingStatus.ACCEPTED.value,
            # Montant: tarif préférentiel ou minimum par défaut
            amount=amount,
            created_via=BookingCreatedVia.INSTITUTION_PORTAL,
        )

        # Assigner billed_to_company_id APRÈS la construction pour que
        # le @validates voit déjà le billed_to_type correct (sinon il
        # le réinitialise à None car le type par défaut est "patient")
        if billed_to_company_id is not None:
            booking.billed_to_company_id = billed_to_company_id

        self._apply_clinical_dropoff_from_request(booking, transport_request)

        # Gel tarifaire : conserver le profil/version/détail si calculé via profil
        if price.get("pricing_profile_id"):
            booking.pricing_profile_id = price["pricing_profile_id"]
            booking.pricing_profile_version_id = price.get(
                "pricing_profile_version_id"
            )
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
        self._resolve_billing_party(booking, transport_request, company_id)

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
            api_scheduled_iso_to_naive_geneva(return_time_raw)
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
                billed_to_type=booking.billed_to_type,
                status=BookingStatus.ACCEPTED.value,
                amount=booking.amount,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            )
            if booking.billed_to_company_id is not None:
                return_booking.billed_to_company_id = booking.billed_to_company_id

            # Le retour reprend le même gel tarifaire que l'aller
            return_booking.pricing_profile_id = booking.pricing_profile_id
            return_booking.pricing_profile_version_id = (
                booking.pricing_profile_version_id
            )
            return_booking.price_amount = booking.price_amount
            return_booking.price_breakdown_json = booking.price_breakdown_json

            db.session.add(return_booking)
            db.session.flush()

            self._resolve_billing_party(return_booking, transport_request, company_id)

            logger.info(
                "[AcceptOffer] Return booking %s created (parent=%s, time_confirmed=%s) for request %s",
                return_booking.id,
                booking.id,
                ret_time_confirmed,
                transport_request.id,
            )

        return booking, return_booking

    def _create_bookings_from_legs(
        self,
        transport_request: TransportRequest,
        company_id: int,
        user_id: int,
        proposed_pickup_time: datetime | None = None,
    ) -> tuple[Booking, Booking | None]:
        """Conversion atomique multi-stop : 1 booking par leg."""
        from models.transport_request_leg import TransportRequestLeg

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
        preferential_rate = getattr(
            institution_client, "preferential_rate", None
        )

        billing_intent = (
            getattr(transport_request, "billing_intent", None) or "patient"
        ).lower()
        billed_to_type = "patient"
        billed_to_company_id = None
        if billing_intent == "institution":
            billed_to_type = "clinic"
            billed_to_company_id = company_id

        route_group_id = getattr(transport_request, "route_group_id", None)
        customer_name = self._get_customer_name(transport_request)
        primary: Booking | None = None

        for leg in legs:
            is_first_leg = leg.sequence_index == 0
            leg_confirmed = bool(getattr(leg, "time_confirmed", False))
            mission_depart_confirmed = bool(
                getattr(transport_request, "pickup_time_confirmed", False)
                and transport_request.scheduled_time
            )
            if is_first_leg:
                raw_pickup = (
                    proposed_pickup_time
                    or (leg.scheduled_time if leg_confirmed else None)
                    or (
                        transport_request.scheduled_time
                        if mission_depart_confirmed
                        else None
                    )
                )
                operational = (
                    leg_confirmed
                    or mission_depart_confirmed
                    or proposed_pickup_time is not None
                )
            else:
                raw_pickup = leg.scheduled_time if leg_confirmed else None
                operational = leg_confirmed
            effective_pickup_time = (
                api_scheduled_iso_to_naive_geneva(raw_pickup) if raw_pickup else None
            )

            time_to_define = effective_pickup_time is None or not operational

            # Tarif par leg : préférentiel sinon profil tarifaire actif
            leg_price = resolve_institution_price(
                company_id=company_id,
                preferential_rate=preferential_rate,
                pickup_location=leg.pickup_location,
                dropoff_location=leg.dropoff_location,
                pickup_lat=float(leg.pickup_lat) if leg.pickup_lat else None,
                pickup_lon=float(leg.pickup_lng) if leg.pickup_lng else None,
                dropoff_lat=float(leg.dropoff_lat) if leg.dropoff_lat else None,
                dropoff_lon=float(leg.dropoff_lng) if leg.dropoff_lng else None,
                scheduled_time=transport_request.scheduled_time,
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
                wheelchair_need=self._get_mobility_flag(
                    transport_request, "stretcher"
                ),
                notes_medical=transport_request.notes,
                billed_to_type=billed_to_type,
                status=BookingStatus.ACCEPTED.value,
                amount=amount,
                route_group_id=route_group_id,
                route_sequence_number=leg.route_sequence_number,
                created_via=BookingCreatedVia.INSTITUTION_PORTAL,
            )
            if billed_to_company_id is not None:
                booking.billed_to_company_id = billed_to_company_id

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
            self._resolve_billing_party(booking, transport_request, company_id)

            if primary is None:
                primary = booking

        if primary is None:
            raise ValueError("Conversion multi-stop sans booking créé")

        logger.info(
            "[AcceptOffer] Multi-stop request %s -> %s bookings (route_group=%s)",
            transport_request.id,
            len(legs),
            route_group_id,
        )
        return primary, None

    def _resolve_billing_party(
        self,
        booking: Booking,
        transport_request: TransportRequest,
        company_id: int,
    ) -> None:
        """Résout et attache le billing_party au booking."""
        try:
            from services.billing.institution_billing_resolver import (
                resolve_billing_party_for_institution_booking,
            )

            billing_result = resolve_billing_party_for_institution_booking(
                booking=booking,
                transport_request=transport_request,
                company_id=company_id,
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
        except Exception as billing_err:
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
            logger.info(
                "[AcceptOffer] Institution client found by name fallback: %s (id=%s, rate=%s, linked=%s)",
                institution.name,
                fallback_by_name.id,
                fallback_by_name.preferential_rate,
                fallback_by_name.linked_institution_id,
            )
            return fallback_by_name

        return self._create_institution_client(institution, company_id)

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
        inst_name = getattr(institution, "name", None) or f"Institution {inst_id}"

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
                    institution_id=int(inst_id),
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
                    patient.dob.isoformat()
                    if getattr(patient, "dob", None)
                    else None
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
            list(getattr(transport_request, "legs", None) or []),
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
