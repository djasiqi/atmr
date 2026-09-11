# application/institutions/send_transport_request.py
# pyright: reportCallIssue=false, reportImportCycles=false
"""Use case: Envoyer une demande de transport aux entreprises.

Gère la création des RequestOffers selon les préférences de l'institution:
- Si préférences définies: mode séquentiel (1 offer à la fois avec timeout)
- Sinon: mode broadcast (toutes les entreprises éligibles en parallèle)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypedDict

from flask import current_app

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


class _RelaunchOffersResult(TypedDict):
    offers_created: int
    mode: str


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
           - 422 : aucun envoi automatique (pas de fan-out catalogue)

        Args:
            input_data: Données d'entrée

        Returns:
            SendTransportRequestResult avec le statut de l'envoi
        """
        started = time.perf_counter()
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
                RequestStatus.EXPIRED.value,
            ]:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error=f"Impossible d'envoyer une demande en statut {transport_request.status}",
                    status_code=400,
                )

            from models.enums import CarrierSource

            if transport_request.carrier_source == CarrierSource.EXTERNAL.value:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error=(
                        "Mission externalisée : retour au flux LIRIE non supporté en V1"
                    ),
                    status_code=409,
                )

            from services.institutions.mission_schedule import (
                get_effective_dispatch_time,
                has_at_least_one_confirmed_time,
            )

            if not has_at_least_one_confirmed_time(transport_request):
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error=(
                        "Pour envoyer aux transporteurs, confirmez au moins une heure "
                        "(départ, rendez-vous ou retour)."
                    ),
                    status_code=400,
                )

            dispatch_time = get_effective_dispatch_time(transport_request)

            # 4. Vérifier s'il y a déjà des offres PENDING
            existing_pending = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                status=OfferStatus.PENDING.value,
            ).first()

            if existing_pending:
                # ÉTAPE GO-LIVE: Idempotent si déjà SENT/EXPIRED avec offres PENDING actives
                # → Retourne 200 au lieu de 409 (évite les erreurs UI sur retry)
                if transport_request.status in (
                    RequestStatus.SENT.value,
                    RequestStatus.EXPIRED.value,
                ):
                    pending_offers = RequestOffer.query.filter_by(
                        transport_request_id=transport_request.id,
                        status=OfferStatus.PENDING.value,
                    ).all()
                    actionable_pending = [o for o in pending_offers if not o.is_expired]

                    if actionable_pending:
                        pending_count = len(actionable_pending)
                        logger.info(
                            "[SendTransportRequest] Idempotent: request %s already SENT with %d pending offers (%.0fms)",
                            transport_request.id,
                            pending_count,
                            (time.perf_counter() - started) * 1000,
                        )

                        mode = existing_pending.mode or OfferMode.BROADCAST.value
                        schedule_post_send_side_effects(
                            transport_request_id=transport_request.id,
                            institution_id=input_data.institution_id,
                            user_id=input_data.user_id,
                            mode=mode,
                            offers_created=pending_count,
                            is_relaunch=False,
                            include_audit=False,
                        )

                        return SendTransportRequestResult(
                            success=True,
                            transport_request_id=transport_request.id,
                            offers_created=pending_count,
                            mode=mode,
                        )

                    logger.info(
                        "[SendTransportRequest] Relaunch: request %s has %d time-expired pending offers",
                        transport_request.id,
                        len(pending_offers),
                    )
                else:
                    # DRAFT + offres pending : finaliser l'envoi, ne pas 409.
                    pending_offers = RequestOffer.query.filter_by(
                        transport_request_id=transport_request.id,
                        status=OfferStatus.PENDING.value,
                    ).all()
                    actionable_pending = [o for o in pending_offers if not o.is_expired]
                    if actionable_pending:
                        if transport_request.status == RequestStatus.DRAFT.value:
                            transport_request.status = RequestStatus.SENT.value
                            transport_request.sent_at = datetime.now(UTC)
                            db.session.commit()
                        mode = existing_pending.mode or OfferMode.BROADCAST.value
                        logger.info(
                            "[SendTransportRequest] Idempotent DRAFT→SENT request %s (%d pending, %.0fms)",
                            transport_request.id,
                            len(actionable_pending),
                            (time.perf_counter() - started) * 1000,
                        )
                        schedule_post_send_side_effects(
                            transport_request_id=transport_request.id,
                            institution_id=input_data.institution_id,
                            user_id=input_data.user_id,
                            mode=mode,
                            offers_created=len(actionable_pending),
                            is_relaunch=False,
                            include_audit=True,
                        )
                        return SendTransportRequestResult(
                            success=True,
                            transport_request_id=transport_request.id,
                            offers_created=len(actionable_pending),
                            mode=mode,
                        )

            # 5. Charger les settings institution + calculer le timeout
            settings = get_or_create_settings(input_data.institution_id)
            timeout_minutes = calculate_timeout(
                input_data.institution_id,
                dispatch_time,
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

            prior_offers_exist = (
                RequestOffer.query.filter_by(
                    transport_request_id=transport_request.id,
                ).first()
                is not None
            )
            is_relaunch = prior_offers_exist and transport_request.status in [
                RequestStatus.SENT.value,
                RequestStatus.EXPIRED.value,
            ]

            if not preferences and not is_relaunch:
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error=(
                        "Aucun transporteur n'est configuré. Ajoutez au moins un "
                        "transporteur dans les paramètres pour envoyer automatiquement."
                    ),
                    status_code=422,
                )

            if is_relaunch:
                if transport_request.status == RequestStatus.EXPIRED.value:
                    transport_request.status = RequestStatus.SENT.value

                relaunch_result = self._create_relaunch_offers(
                    transport_request=transport_request,
                    preferences=preferences,
                    configured_mode=configured_mode,
                    expires_at=expires_at,
                )
                offers_created = relaunch_result["offers_created"]
                mode = relaunch_result["mode"]

                if offers_created == 0:
                    db.session.rollback()
                    return SendTransportRequestResult(
                        success=False,
                        transport_request_id=transport_request.id,
                        error=(
                            "Aucun transporteur disponible pour relancer la diffusion. "
                            "Vérifiez vos préférences ou contactez le support."
                        ),
                        status_code=422,
                    )
            elif configured_mode == OfferMode.BROADCAST.value:
                mode = OfferMode.BROADCAST.value
                offers_created = self._create_broadcast_offers(
                    transport_request=transport_request,
                    expires_at=None,
                    excluded_company_ids=[],
                    only_company_ids=[p.company_id for p in preferences],
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
                return SendTransportRequestResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error=(
                        "Aucun transporteur n'est configuré. Ajoutez au moins un "
                        "transporteur dans les paramètres pour envoyer automatiquement."
                    ),
                    status_code=422,
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

            logger.info(
                "[SendTransportRequest] Request %s accepted: mode=%s, offers=%d (%.0fms)",
                transport_request.id,
                mode,
                offers_created,
                (time.perf_counter() - started) * 1000,
            )

            schedule_post_send_side_effects(
                transport_request_id=transport_request.id,
                institution_id=input_data.institution_id,
                user_id=input_data.user_id,
                mode=mode,
                offers_created=offers_created,
                is_relaunch=is_relaunch,
                include_audit=True,
                timeout_minutes=timeout_minutes,
            )

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

    @staticmethod
    def _record_send_timeline(
        *,
        transport_request: TransportRequest,
        user_id: int | None,
    ) -> None:
        """Historise l'envoi (request_sent) et chaque offre émise (offer_sent)."""
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                record_event,
                resolve_actor_name,
            )

            actor = TimelineActor(
                actor_type="institution_user" if user_id else "api_key",
                actor_user_id=user_id,
            )
            sent_event = record_event(
                "request_sent",
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                actor=actor,
                payload={"actor_name": resolve_actor_name(user_id)},
                correlation_id=f"request_sent:{transport_request.id}",
            )

            # Flush pour obtenir les ids d'offres fraîchement créées
            db.session.flush()
            pending_offers = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                status=OfferStatus.PENDING.value,
            ).all()
            source_event_id = sent_event.id if sent_event else None
            for offer in pending_offers:
                company = Company.query.get(offer.company_id)
                record_event(
                    "offer_sent",
                    institution_id=transport_request.institution_id,
                    transport_request_id=transport_request.id,
                    actor=TimelineActor(
                        actor_type="system", company_id=offer.company_id
                    ),
                    payload={
                        "company_id": offer.company_id,
                        "company_name": company.name if company else None,
                        "offer_id": offer.id,
                        "expires_at": offer.expires_at.isoformat()
                        if offer.expires_at
                        else None,
                        "dispatch_mode": offer.mode,
                    },
                    correlation_id=f"offer_sent:{offer.id}",
                    source_event_id=source_event_id,
                )
        except Exception as timeline_err:
            logger.warning(
                "[SendTransportRequest] Timeline recording failed: %s", timeline_err
            )

    def _create_sequential_offer(
        self,
        transport_request: TransportRequest,
        preference: InstitutionTransportPreference,
        expires_at: datetime,
    ) -> int:
        """Crée une offre séquentielle pour une préférence donnée."""
        from services.platform_billing.capabilities import (
            BillingCapability,
            is_billing_capability_allowed,
        )

        if not is_billing_capability_allowed(
            preference.company_id,
            BillingCapability.RECEIVE_MARKETPLACE_OFFERS,
        ):
            logger.info(
                "[SendTransportRequest] Offre séquentielle ignorée "
                "(billing_access_restricted) company_id=%s",
                preference.company_id,
            )
            return 0
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
        only_company_ids: list[int] | None = None,
    ) -> int:
        """Crée des offres broadcast pour les entreprises éligibles (liste bornée)."""
        eligible_companies = self._get_eligible_companies(
            excluded_company_ids,
            transport_request=transport_request,
            only_company_ids=only_company_ids,
        )

        if not eligible_companies:
            logger.warning(
                "[SendTransportRequest] Aucune entreprise éligible pour request %s",
                transport_request.id,
            )
            return 0

        existing_ids = {
            company_id
            for (company_id,) in db.session.query(RequestOffer.company_id)
            .filter(RequestOffer.transport_request_id == transport_request.id)
            .all()
        }

        offers_created = 0
        for company in eligible_companies:
            if company.id in existing_ids:
                continue

            db.session.add(
                RequestOffer(
                    transport_request_id=transport_request.id,
                    company_id=company.id,
                    mode=OfferMode.BROADCAST.value,
                    order=0,
                    status=OfferStatus.PENDING.value,
                    expires_at=expires_at,
                )
            )
            offers_created += 1

        return offers_created

    def _get_eligible_companies(
        self,
        excluded_ids: list[int],
        transport_request: TransportRequest | None = None,
        only_company_ids: list[int] | None = None,
    ) -> list[Company]:
        """Entreprises destinataires : liste configurée ou catalogue partenaire.

        `dispatch_enabled` n'est pas un critère (mode MANUAL légitime).
        """
        from application.institutions.eligible_carriers import (
            companies_for_marketplace_dispatch,
        )

        institution = (
            transport_request.institution if transport_request is not None else None
        )
        return companies_for_marketplace_dispatch(
            institution=institution,
            excluded_ids=excluded_ids or None,
            only_company_ids=only_company_ids,
        )

    @staticmethod
    def _reactivate_offer(
        offer: RequestOffer,
        expires_at: datetime | None,
    ) -> None:
        """Remet une offre expirée/indisponible en attente."""
        offer.status = OfferStatus.PENDING.value
        offer.expires_at = expires_at
        offer.sent_at = datetime.now(UTC)
        offer.responded_at = None
        offer.rejection_reason = None

    def _create_relaunch_offers(
        self,
        *,
        transport_request: TransportRequest,
        preferences: list[InstitutionTransportPreference],
        configured_mode: str,
        expires_at: datetime,
    ) -> _RelaunchOffersResult:
        """Relance la diffusion : réactive les offres expirées et complète la liste configurée."""
        del configured_mode
        only_company_ids = [p.company_id for p in preferences] if preferences else None
        offers_created = self._relaunch_broadcast_offers(
            transport_request=transport_request,
            expires_at=expires_at,
            only_company_ids=only_company_ids,
        )
        return {
            "offers_created": offers_created,
            "mode": str(OfferMode.BROADCAST.value),
        }

    def _relaunch_sequential_offers(
        self,
        *,
        transport_request: TransportRequest,
        preferences: list[InstitutionTransportPreference],
        expires_at: datetime,
    ) -> tuple[int, str]:
        """Relance séquentielle : réactive la 1re préférence éligible ou élargit."""
        from services.demo.soft_delete_guard import (
            company_is_demo,
            institution_is_demo,
        )

        ordered_prefs = list(preferences)
        if institution_is_demo(transport_request.institution):
            ordered_prefs = [
                p
                for p in preferences
                if company_is_demo(Company.query.get(p.company_id))
            ]

        reactivatable = {
            OfferStatus.EXPIRED.value,
            OfferStatus.UNAVAILABLE.value,
        }

        for pref in ordered_prefs:
            existing = RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                company_id=pref.company_id,
            ).first()
            if existing:
                if existing.status == OfferStatus.REJECTED.value:
                    continue
                if existing.status == OfferStatus.PENDING.value:
                    if existing.is_expired:
                        self._reactivate_offer(existing, expires_at)
                    return 1, OfferMode.SEQUENTIAL.value
                if existing.status in reactivatable:
                    self._reactivate_offer(existing, expires_at)
                    return 1, OfferMode.SEQUENTIAL.value
                continue

            created = self._create_sequential_offer(
                transport_request=transport_request,
                preference=pref,
                expires_at=expires_at,
            )
            if created:
                return created, OfferMode.SEQUENTIAL.value

        contacted_ids = [
            o.company_id
            for o in RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
            ).all()
        ]
        broadcast_created = self._create_broadcast_offers(
            transport_request=transport_request,
            expires_at=None,
            excluded_company_ids=contacted_ids,
        )
        return broadcast_created, OfferMode.BROADCAST.value

    def _relaunch_broadcast_offers(
        self,
        *,
        transport_request: TransportRequest,
        expires_at: datetime | None,
        only_company_ids: list[int] | None = None,
    ) -> int:
        """Relance broadcast : réactive les offres expirées + contacte les entreprises éligibles restantes."""
        offers = RequestOffer.query.filter_by(
            transport_request_id=transport_request.id,
        ).all()
        reactivated = 0
        for offer in offers:
            if offer.status == OfferStatus.REJECTED.value:
                continue
            if offer.status == OfferStatus.EXPIRED.value or (
                offer.status == OfferStatus.PENDING.value and offer.is_expired
            ):
                self._reactivate_offer(offer, expires_at)
                reactivated += 1

        contacted_ids = [
            o.company_id
            for o in RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
            ).all()
        ]
        new_created = self._create_broadcast_offers(
            transport_request=transport_request,
            expires_at=expires_at,
            excluded_company_ids=contacted_ids,
            only_company_ids=only_company_ids,
        )
        return reactivated + new_created

    def _notify_target_companies(
        self,
        transport_request: TransportRequest,
        *,
        is_relaunch: bool = False,
    ) -> None:
        """Notifie chaque entreprise ayant reçu une offre PENDING."""
        try:
            from ext import socketio
            from services.demo.soft_delete_guard import (
                company_is_demo,
                institution_is_demo,
            )
            from services.events.institution_events import (
                persist_company_notification,
            )
            from services.institutions.mission_schedule import (
                get_effective_dispatch_time,
                get_mission_date,
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

            sched = get_effective_dispatch_time(transport_request)
            time_str = sched.strftime("%d.%m.%Y %H:%M") if sched else "Date à confirmer"
            round_trip = " (A/R)" if transport_request.is_round_trip else ""

            title = (
                "Demande de transport relancée"
                if is_relaunch
                else "Nouvelle demande de transport"
            )
            message = f"{inst_name} — {patient_name}{round_trip} — {time_str}".strip(
                " —"
            )
            relaunch_ts = int(datetime.now(UTC).timestamp())
            mission_day = get_mission_date(transport_request)
            mission_date_iso = mission_day.isoformat() if mission_day else None

            for offer in pending_offers:
                try:
                    company = Company.query.get(offer.company_id)
                    if inst_is_demo and not company_is_demo(company):
                        continue

                    dedupe_key = (
                        f"new_request:{transport_request.id}:{offer.company_id}:relaunch:{relaunch_ts}"
                        if is_relaunch
                        else f"new_request:{transport_request.id}:{offer.company_id}"
                    )
                    notif = persist_company_notification(
                        company_id=offer.company_id,
                        event_type="new_request",
                        title=title,
                        message=message,
                        metadata={
                            "request_id": transport_request.id,
                            "public_id": str(transport_request.public_id),
                            "offer_id": offer.id,
                            "institution_name": inst_name,
                            "is_relaunch": is_relaunch,
                            **(
                                {"mission_date": mission_date_iso}
                                if mission_date_iso
                                else {}
                            ),
                        },
                        dedupe_key=dedupe_key,
                    )
                    if notif is None:
                        logger.info(
                            "[SendTransportRequest] Push skipped (inbox dedupe) company=%s request=%s dedupe_key=%s",
                            offer.company_id,
                            transport_request.id,
                            dedupe_key,
                        )
                        continue

                    expires_at_iso = (
                        offer.expires_at.isoformat()
                        if getattr(offer, "expires_at", None)
                        else None
                    )
                    from services.notifications.institution_new_request_push import (
                        enqueue_institution_new_request_company_push,
                    )

                    enqueue_institution_new_request_company_push(
                        transport_request=transport_request,
                        offer_id=offer.id,
                        company_id=offer.company_id,
                        institution_name=inst_name,
                        patient_name=patient_name,
                        title=title,
                        message=message,
                        dedupe_key=dedupe_key,
                        mission_date_iso=mission_date_iso,
                        expires_at_iso=expires_at_iso,
                        is_relaunch=is_relaunch,
                        sched=sched,
                    )

                    socketio.emit(
                        "institution_offer_updated",
                        {
                            "offer_id": offer.id,
                            "transport_request_id": transport_request.id,
                            "company_id": offer.company_id,
                            "is_relaunch": is_relaunch,
                        },
                        to=f"company_{offer.company_id}",
                        namespace="/",
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


def run_post_send_side_effects(
    transport_request_id: int,
    institution_id: int,
    user_id: int | None,
    mode: str | None,
    offers_created: int,
    *,
    is_relaunch: bool,
    include_audit: bool,
    timeout_minutes: int | None = None,
    remove_session: bool = True,
) -> None:
    """Notifications / audit / métriques après acceptation du dispatch."""
    try:
        transport_request = TransportRequest.query.get(transport_request_id)
        if not transport_request:
            logger.warning(
                "[SendTransportRequest] Side effects skipped: request %s introuvable",
                transport_request_id,
            )
            return

        if include_audit:
            try:
                AuditLogger.log_action(
                    action_type="transport_request_sent",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution" if user_id else "api_key",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "transport_request_id": transport_request_id,
                        "mode": mode,
                        "offers_created": offers_created,
                        **(
                            {"timeout_minutes": timeout_minutes}
                            if timeout_minutes is not None
                            else {}
                        ),
                    },
                )
            except Exception as audit_err:
                logger.warning("Échec audit log: %s", audit_err)

            try:
                from services.metrics.institution_metrics import track_send_event

                track_send_event(
                    transport_request_id=transport_request_id,
                    institution_id=institution_id,
                    mode=mode,
                    offers_created=offers_created,
                )
            except Exception as metric_err:
                logger.warning(
                    "[SendTransportRequest] Error tracking metric: %s", metric_err
                )

        try:
            from services.events.institution_events import (
                emit_request_sent,
                format_institution_patient_bell_name,
            )
            from services.institutions.mission_schedule import (
                get_effective_dispatch_time,
            )

            patient = transport_request.patient
            patient_label = (
                format_institution_patient_bell_name(
                    first_name=getattr(patient, "first_name", None),
                    last_name=getattr(patient, "last_name", None),
                    gender=getattr(patient, "gender", None),
                )
                if patient is not None
                else None
            )
            emit_request_sent(
                institution_id=institution_id,
                request_id=transport_request.id,
                public_id=transport_request.public_id,
                external_reference=transport_request.external_reference,
                mode=mode,
                offers_created=offers_created,
                patient_name=patient_label,
                departure_at=get_effective_dispatch_time(transport_request),
            )
        except Exception as event_err:
            logger.warning("[SendTransportRequest] Error emitting event: %s", event_err)

        if include_audit:
            SendTransportRequestUseCase._record_send_timeline(
                transport_request=transport_request,
                user_id=user_id,
            )

        SendTransportRequestUseCase()._notify_target_companies(
            transport_request,
            is_relaunch=is_relaunch,
        )
    except Exception:
        logger.exception(
            "[SendTransportRequest] Side effects failed for request %s",
            transport_request_id,
        )
    finally:
        if remove_session:
            db.session.remove()


def schedule_post_send_side_effects(
    transport_request_id: int,
    institution_id: int,
    user_id: int | None,
    mode: str | None,
    offers_created: int,
    *,
    is_relaunch: bool,
    include_audit: bool,
    timeout_minutes: int | None = None,
) -> None:
    """Lance les effets secondaires sans bloquer la réponse HTTP (hors tests)."""
    kwargs = {
        "transport_request_id": transport_request_id,
        "institution_id": institution_id,
        "user_id": user_id,
        "mode": mode,
        "offers_created": offers_created,
        "is_relaunch": is_relaunch,
        "include_audit": include_audit,
        "timeout_minutes": timeout_minutes,
    }
    if current_app.config.get("TESTING"):
        run_post_send_side_effects(**kwargs, remove_session=False)
        return

    from tasks.request_offer_tasks import notify_transport_request_after_send

    notify_transport_request_after_send.delay(
        kwargs["transport_request_id"],
        kwargs["institution_id"],
        kwargs["user_id"],
        kwargs["mode"],
        kwargs["offers_created"],
        kwargs["is_relaunch"],
        kwargs["include_audit"],
        kwargs["timeout_minutes"],
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
    # flush pour que offer.id soit disponible immédiatement (réponse API escalade)
    db.session.flush()

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
