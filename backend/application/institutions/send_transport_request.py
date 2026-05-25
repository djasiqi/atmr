# application/institutions/send_transport_request.py
# pyright: reportCallIssue=false
"""Use case: Envoyer une demande de transport aux entreprises.

Gère la création des RequestOffers selon les préférences de l'institution:
- Si préférences définies: mode séquentiel (1 offer à la fois avec timeout)
- Sinon: mode broadcast (toutes les entreprises éligibles en parallèle)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from application.institutions.institution_settings_service import (
    calculate_timeout,
    get_or_create_settings,
)
from ext import db
from models import (
    Company,
    InstitutionTransportPreference,
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
)
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SendTransportRequestInput:
    """Input pour l'envoi d'une demande de transport."""

    transport_request_id: int
    institution_id: int
    user_id: int | None = None  # Utilisateur qui envoie (ou None si API key)


@dataclass(frozen=True, slots=True)
class SendTransportRequestResult:
    """Résultat de l'envoi d'une demande de transport."""

    success: bool
    transport_request_id: int
    offers_created: int = 0
    mode: str | None = None  # "sequential" ou "broadcast"
    error: str | None = None
    status_code: int = 200


class SendTransportRequestUseCase:
    """Use case: Envoyer une demande de transport aux entreprises."""

    def execute(
        self, input_data: SendTransportRequestInput
    ) -> SendTransportRequestResult:
        """
        Envoie une demande de transport aux entreprises de transport.

        Règles:
        1. Si l'institution a des préférences définies:
           - Mode SEQUENTIAL: Envoyer uniquement à la première préférence
           - Timeout dynamique selon scheduled_time
        2. Sinon:
           - Mode BROADCAST: Envoyer à toutes les entreprises éligibles

        Args:
            input_data: Données d'entrée

        Returns:
            SendTransportRequestResult avec le statut de l'envoi
        """
        try:
            # 1. Charger la demande
            transport_request = TransportRequest.query.get(
                input_data.transport_request_id
            )
            if not transport_request:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Demande de transport introuvable",
                    status_code=404,
                )

            # 2. Vérifier que la demande appartient à l'institution
            if transport_request.institution_id != input_data.institution_id:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Accès non autorisé à cette demande",
                    status_code=403,
                )

            # 3. Vérifier le statut
            # ÉTAPE GO-LIVE: Protection anti double-envoi idempotent
            if transport_request.status == RequestStatus.CONVERTED.value:
                # 409: Demande déjà convertie, annulation doit se faire sur booking
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Demande déjà convertie en booking. Impossible d'envoyer à nouveau.",
                    status_code=409,
                )

            if transport_request.status not in [
                RequestStatus.DRAFT.value,
                RequestStatus.SENT.value,
            ]:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error=f"Impossible d'envoyer une demande en statut {transport_request.status}",
                    status_code=400,
                )

            # 4. Vérifier s'il y a déjà des offres PENDING
            existing_pending = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                status=OfferStatus.PENDING.value,
            ).first()

            if existing_pending:
                # ÉTAPE GO-LIVE: Idempotent si déjà SENT avec offres PENDING
                # → Retourne 200 au lieu de 409 (évite les erreurs UI sur retry)
                if transport_request.status == RequestStatus.SENT.value:
                    # Compter les offres PENDING
                    pending_count = RequestOffer.query.filter_by(
                        transport_request_id=transport_request.id,
                        status=OfferStatus.PENDING.value,
                    ).count()

                    logger.info(
                        "[SendTransportRequest] Idempotent: request %s already SENT with %d pending offers",
                        transport_request.id,
                        pending_count,
                    )

                    # Déterminer le mode depuis les offres existantes
                    mode = existing_pending.mode or OfferMode.BROADCAST.value

                    return SendTransportRequestResult(
                        success=True,
                        transport_request_id=transport_request.id,
                        offers_created=pending_count,
                        mode=mode,
                    )

                # Sinon (DRAFT avec offres pending = état incohérent)
                from services.metrics.institution_metrics import (
                    track_transport_request_duplicate_blocked,
                )

                track_transport_request_duplicate_blocked(
                    transport_request_id=transport_request.id,
                    institution_id=input_data.institution_id,
                )
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Des offres sont déjà en attente pour cette demande",
                    status_code=409,
                )

            # 5. Charger les settings institution + calculer le timeout
            settings = get_or_create_settings(input_data.institution_id)
            timeout_minutes = calculate_timeout(
                input_data.institution_id,
                transport_request.scheduled_time,
            )
            now = datetime.now(UTC)
            expires_at = now + timedelta(minutes=timeout_minutes)

            # 6. Déterminer le mode (configurable par institution)
            preferences = InstitutionTransportPreference.get_ordered_preferences(
                input_data.institution_id
            )
            configured_mode = str(
                getattr(settings, "offer_dispatch_mode", OfferMode.SEQUENTIAL.value)
                or OfferMode.SEQUENTIAL.value
            ).lower()

            if configured_mode == OfferMode.BROADCAST.value:
                mode = OfferMode.BROADCAST.value
                offers_created = self._create_broadcast_offers(
                    transport_request=transport_request,
                    expires_at=None,
                    excluded_company_ids=[],
                )
            elif preferences:
                # Mode séquentiel: envoyer uniquement à la première préférence
                # Si institution démo: ignorer les préférences vers entreprises réelles
                from services.demo.soft_delete_guard import (
                    company_is_demo,
                    institution_is_demo,
                )

                mode = OfferMode.SEQUENTIAL.value
                pref = preferences[0]
                if institution_is_demo(transport_request.institution):
                    demo_prefs = [
                        p
                        for p in preferences
                        if company_is_demo(Company.query.get(p.company_id))
                    ]
                    pref = demo_prefs[0] if demo_prefs else None
                if pref:
                    offers_created = self._create_sequential_offer(
                        transport_request=transport_request,
                        preference=pref,
                        expires_at=expires_at,
                    )
                else:
                    offers_created = 0
            else:
                # Pas de préférence: fallback broadcast
                mode = OfferMode.BROADCAST.value
                offers_created = self._create_broadcast_offers(
                    transport_request=transport_request,
                    expires_at=None,  # Pas d'expiration en broadcast (ou longue)
                    excluded_company_ids=[],
                )

            # 7. Vérifier qu'au moins une offre a été créée
            if offers_created == 0:
                db.session.rollback()
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error="Aucune entreprise de transport éligible trouvée. Vérifiez que des entreprises sont approuvées et ont le dispatch activé.",
                    status_code=422,
                )

            # 8. Mettre à jour le statut de la demande
            if transport_request.status == RequestStatus.DRAFT.value:
                transport_request.status = RequestStatus.SENT.value
                transport_request.sent_at = now

            db.session.commit()

            # 8. Audit log
            try:
                AuditLogger.log_action(
                    action_type="transport_request_sent",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="institution" if input_data.user_id else "api_key",
                    institution_id=input_data.institution_id,
                    result_status="success",
                    action_details={
                        "transport_request_id": transport_request.id,
                        "mode": mode,
                        "offers_created": offers_created,
                        "timeout_minutes": timeout_minutes,
                    },
                )
            except Exception as audit_err:
                logger.warning("Échec audit log: %s", audit_err)

            logger.info(
                "[SendTransportRequest] Request %s sent: mode=%s, offers=%d",
                transport_request.id,
                mode,
                offers_created,
            )

            # 9. Métriques métier (GO-LIVE)
            try:
                from services.metrics.institution_metrics import track_send_event

                track_send_event(
                    transport_request_id=transport_request.id,
                    institution_id=input_data.institution_id,
                    mode=mode,
                    offers_created=offers_created,
                )
            except Exception as metric_err:
                logger.warning(
                    "[SendTransportRequest] Error tracking metric: %s", metric_err
                )

            # ÉTAPE 5: Émettre événement temps réel vers l'institution
            try:
                from services.events.institution_events import emit_request_sent

                emit_request_sent(
                    institution_id=input_data.institution_id,
                    request_id=transport_request.id,
                    public_id=transport_request.public_id,
                    external_reference=transport_request.external_reference,
                    mode=mode,
                    offers_created=offers_created,
                )
            except Exception as event_err:
                logger.warning(
                    "[SendTransportRequest] Error emitting event: %s", event_err
                )

            # ÉTAPE 6: Notifier les entreprises cibles (Socket + bell)
            self._notify_target_companies(transport_request)

            return SendTransportRequestResult(
                success=True,
                transport_request_id=transport_request.id,
                offers_created=offers_created,
                mode=mode,
            )

        except Exception as e:
            logger.exception(
                "Erreur lors de l'envoi de la demande %s",
                input_data.transport_request_id,
            )
            db.session.rollback()
            return SendTransportRequestResult(
                success=False,
                transport_request_id=input_data.transport_request_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )

    def _create_sequential_offer(
        self,
        transport_request: TransportRequest,
        preference: InstitutionTransportPreference,
        expires_at: datetime,
    ) -> int:
        """Crée une offre séquentielle pour une préférence donnée."""
        offer = RequestOffer(
            transport_request_id=transport_request.id,
            company_id=preference.company_id,
            mode=OfferMode.SEQUENTIAL.value,
            order=preference.order,
            status=OfferStatus.PENDING.value,
            expires_at=expires_at,
        )
        db.session.add(offer)
        return 1

    def _create_broadcast_offers(
        self,
        transport_request: TransportRequest,
        expires_at: datetime | None,
        excluded_company_ids: list[int],
    ) -> int:
        """Crée des offres broadcast pour toutes les entreprises éligibles."""
        # Récupérer les entreprises éligibles (démo: uniquement entreprises démo)
        eligible_companies = self._get_eligible_companies(
            excluded_company_ids, transport_request=transport_request
        )

        if not eligible_companies:
            logger.warning(
                "[SendTransportRequest] Aucune entreprise éligible pour request %s",
                transport_request.id,
            )
            return 0

        offers_created = 0
        for company in eligible_companies:
            # Vérifier qu'il n'y a pas déjà une offre pour cette entreprise
            existing = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                company_id=company.id,
            ).first()

            if existing:
                continue

            offer = RequestOffer(
                transport_request_id=transport_request.id,
                company_id=company.id,
                mode=OfferMode.BROADCAST.value,
                order=0,  # Pas d'ordre en broadcast
                status=OfferStatus.PENDING.value,
                expires_at=expires_at,
            )
            db.session.add(offer)
            offers_created += 1

        return offers_created

    def _get_eligible_companies(
        self,
        excluded_ids: list[int],
        transport_request: TransportRequest | None = None,
    ) -> list[Company]:
        """Récupère les entreprises éligibles pour recevoir des offres.

        V1 simple: Toutes les entreprises approuvées et avec dispatch activé.
        Si la demande vient d'une institution démo, seules les entreprises démo
        sont éligibles (évite que des transporteurs réels voient des demandes démo).
        """
        from services.demo.soft_delete_guard import (
            company_is_demo,
            institution_is_demo,
        )

        query = Company.query.filter(
            Company.is_approved == True,  # noqa: E712
            Company.dispatch_enabled == True,  # noqa: E712
        )

        if excluded_ids:
            query = query.filter(Company.id.notin_(excluded_ids))

        companies = query.all()
        if transport_request and institution_is_demo(transport_request.institution):
            companies = [c for c in companies if company_is_demo(c)]
        return companies

    @staticmethod
    def _notify_target_companies(transport_request: TransportRequest) -> None:
        """Notifie chaque entreprise ayant reçu une offre PENDING."""
        try:
            from services.demo.soft_delete_guard import (
                company_is_demo,
                institution_is_demo,
            )
            from services.events.institution_events import (
                persist_company_notification,
            )

            pending_offers = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                status=OfferStatus.PENDING.value,
            ).all()

            institution = transport_request.institution
            inst_is_demo = institution_is_demo(institution)
            inst_name = institution.name if institution else "Institution"
            patient = transport_request.patient
            patient_name = (
                f"{patient.first_name} {patient.last_name}" if patient else ""
            )

            sched = transport_request.scheduled_time
            time_str = sched.strftime("%d.%m.%Y %H:%M")
            round_trip = " (A/R)" if transport_request.is_round_trip else ""

            title = "Nouvelle demande de transport"
            message = f"{inst_name} — {patient_name}{round_trip} — {time_str}".strip(
                " —"
            )

            for offer in pending_offers:
                try:
                    company = Company.query.get(offer.company_id)
                    if inst_is_demo and not company_is_demo(company):
                        continue
                    persist_company_notification(
                        company_id=offer.company_id,
                        event_type="new_request",
                        title=title,
                        message=message,
                        metadata={
                            "request_id": transport_request.id,
                            "public_id": str(transport_request.public_id),
                            "offer_id": offer.id,
                            "institution_name": inst_name,
                        },
                        dedupe_key=f"new_request:{transport_request.id}:{offer.company_id}",
                    )
                except Exception as notif_err:
                    logger.warning(
                        "[SendTransportRequest] Error notifying company %s: %s",
                        offer.company_id,
                        notif_err,
                    )
        except Exception as e:
            logger.warning(
                "[SendTransportRequest] Error notifying target companies: %s", e
            )


def create_next_sequential_offer(
    transport_request: TransportRequest,
    current_order: int,
    timeout_minutes: int,
) -> RequestOffer | None:
    """Crée la prochaine offre séquentielle (utilisé lors de l'escalade).

    Args:
        transport_request: Demande de transport
        current_order: Ordre actuel (la prochaine sera > current_order)
        timeout_minutes: Timeout en minutes

    Returns:
        La nouvelle offre créée, ou None si pas de préférence suivante
    """
    # Trouver la préférence suivante
    next_pref = InstitutionTransportPreference.get_next_preference_after(
        transport_request.institution_id,
        current_order,
    )

    if not next_pref:
        return None

    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=timeout_minutes)

    offer = RequestOffer(
        transport_request_id=transport_request.id,
        company_id=next_pref.company_id,
        mode=OfferMode.SEQUENTIAL.value,
        order=next_pref.order,
        status=OfferStatus.PENDING.value,
        expires_at=expires_at,
    )
    db.session.add(offer)

    return offer


def create_fallback_broadcast_offers(
    transport_request: TransportRequest,
) -> int:
    """Crée des offres broadcast de fallback (après épuisement des préférences).

    Args:
        transport_request: Demande de transport

    Returns:
        Nombre d'offres créées
    """
    use_case = SendTransportRequestUseCase()

    # Récupérer les IDs des entreprises déjà contactées
    existing_offers = RequestOffer.query.filter_by(
        transport_request_id=transport_request.id,
    ).all()
    contacted_company_ids = [o.company_id for o in existing_offers]

    # Créer des offres broadcast pour les entreprises non encore contactées
    return use_case._create_broadcast_offers(
        transport_request=transport_request,
        expires_at=None,  # Pas d'expiration pour le fallback
        excluded_company_ids=contacted_company_ids,
    )
