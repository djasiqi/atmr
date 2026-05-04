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
from dataclasses import dataclass
from datetime import UTC, datetime

from ext import db
from models import (
    Booking,
    BookingStatus,
    Client,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
)
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


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
                return AcceptOfferResult(
                    success=False,
                    offer_id=input_data.offer_id,
                    transport_request_id=transport_request.id,
                    error=f"Demande en statut {transport_request.status}, acceptation impossible",
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

            # 9. Créer le Booking (+ retour si A/R)
            booking, return_booking = self._create_booking_from_request(
                transport_request=transport_request,
                company_id=input_data.company_id,
                user_id=input_data.user_id,
                proposed_pickup_time=input_data.proposed_pickup_time,
            )
            return_booking_id = return_booking.id if return_booking else None

            # 10. Lier le booking à la request et marquer comme CONVERTED
            transport_request.booking_id = booking.id
            transport_request.status = RequestStatus.CONVERTED.value
            transport_request.converted_at = now

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
        # Résoudre le client institution existant dans la base de l'entreprise
        institution_client = self._find_institution_client(
            transport_request, company_id
        )

        # Tarif: utiliser le tarif préférentiel du client institution si défini
        # Le montant minimum est 0.5 (AMOUNT_MINIMUM dans le modèle Booking)
        amount = 0.5  # Montant par défaut minimum
        if institution_client and institution_client.preferential_rate:
            amount = float(institution_client.preferential_rate)

        # Facturation: utiliser le type par défaut du client institution
        billed_to_type = "patient"
        billed_to_company_id = None
        if institution_client:
            billed_to = getattr(institution_client, "default_billed_to_type", None)
            if billed_to and billed_to in ("patient", "clinic", "insurance"):
                billed_to_type = billed_to
            # Si facturé à la clinique ou assurance, il faut renseigner billed_to_company_id
            if billed_to_type != "patient":
                billed_to_company_id = getattr(
                    institution_client, "default_billed_to_company_id", None
                )

        # Réutiliser le champ hospital_service du Booking pour le point d'accueil
        entry_point = getattr(transport_request, "pickup_entry_point", None) or ""

        # Horaire: utiliser l'horaire proposé par l'entreprise si fourni
        effective_pickup_time = proposed_pickup_time or transport_request.scheduled_time

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
            # Lieux dropoff
            dropoff_location=transport_request.dropoff_location,
            dropoff_lat=float(transport_request.dropoff_lat)
            if transport_request.dropoff_lat
            else None,
            dropoff_lon=float(transport_request.dropoff_lng)
            if transport_request.dropoff_lng
            else None,
            dropoff_access_notes=self._format_dropoff_notes(transport_request),
            # Point d'accueil → champ existant hospital_service du Booking
            hospital_service=entry_point if entry_point else None,
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
        )

        # Assigner billed_to_company_id APRÈS la construction pour que
        # le @validates voit déjà le billed_to_type correct (sinon il
        # le réinitialise à None car le type par défaut est "patient")
        if billed_to_company_id is not None:
            booking.billed_to_company_id = billed_to_company_id

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

        return_time = getattr(transport_request, "return_time", None)
        if transport_request.is_round_trip and return_time is None:
            logger.warning(
                "[AcceptOffer] A/R sans return_time, skip retour request_id=%s",
                transport_request.id,
            )
        elif transport_request.is_round_trip and return_time is not None:
            return_booking = Booking(
                company_id=company_id,
                user_id=user_id,
                client_id=booking.client_id,
                customer_name=booking.customer_name,
                mission_type=booking.mission_type,
                delivery_description=booking.delivery_description,
                scheduled_time=return_time,
                is_round_trip=False,
                is_return=True,
                parent_booking_id=booking.id,
                # Pickup/dropoff inversés
                pickup_location=booking.dropoff_location,
                pickup_lat=booking.dropoff_lat,
                pickup_lon=booking.dropoff_lon,
                pickup_access_notes=booking.dropoff_access_notes,
                dropoff_location=booking.pickup_location,
                dropoff_lat=booking.pickup_lat,
                dropoff_lon=booking.pickup_lon,
                dropoff_access_notes=booking.pickup_access_notes,
                hospital_service=booking.hospital_service,
                wheelchair_client_has=booking.wheelchair_client_has,
                wheelchair_need=booking.wheelchair_need,
                notes_medical=booking.notes_medical,
                billed_to_type=booking.billed_to_type,
                status=BookingStatus.ACCEPTED.value,
                amount=booking.amount,
            )
            if booking.billed_to_company_id is not None:
                return_booking.billed_to_company_id = booking.billed_to_company_id

            db.session.add(return_booking)
            db.session.flush()

            self._resolve_billing_party(return_booking, transport_request, company_id)

            logger.info(
                "[AcceptOffer] Return booking %s created (parent=%s) for request %s",
                return_booking.id,
                booking.id,
                transport_request.id,
            )

        return booking, return_booking

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

    def _find_institution_client(
        self,
        transport_request: TransportRequest,
        company_id: int,
    ) -> Client | None:
        """Cherche le client institution existant dans la base de l'entreprise.

        Stratégie de recherche (par priorité) :
        1. Match exact par linked_institution_id (FK formelle)
        2. Fallback: match par nom d'institution (ilike)

        Retourne le Client si trouvé, None sinon.
        """
        institution = transport_request.institution
        if not institution:
            return None

        # 1. Match par FK linked_institution_id (le plus fiable)
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

        # 2. Fallback: match par nom
        inst_name = institution.name
        if inst_name:
            client = Client.query.filter(
                Client.company_id == company_id,
                Client.is_institution.is_(True),
                Client.institution_name.ilike(inst_name),
            ).first()

            if client:
                logger.info(
                    "[AcceptOffer] Institution client found by name: %s (id=%s, rate=%s)",
                    inst_name,
                    client.id,
                    client.preferential_rate,
                )
                return client

        logger.warning(
            "[AcceptOffer] No institution client found for '%s' (inst_id=%s) in company %s",
            institution.name,
            institution.id,
            company_id,
        )
        return None

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
                    patient.date_of_birth.isoformat()
                    if getattr(patient, "date_of_birth", None)
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
