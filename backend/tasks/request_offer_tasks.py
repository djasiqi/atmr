# tasks/request_offer_tasks.py
# pyright: reportCallIssue=false, reportArgumentType=false
"""Tâches Celery pour la gestion des offres de transport.

- Expiration des offres PENDING
- Escalade vers la préférence suivante (mode séquentiel)
- Fallback broadcast
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from application.institutions.institution_settings_service import calculate_timeout
from celery_app import celery
from ext import db
from models import (
    InstitutionTransportPreference,
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
)
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


def _notify_company_new_offer(
    transport_request: TransportRequest,
    company_id: int,
    offer_id: int | None = None,
) -> None:
    """Notifie une entreprise d'une nouvelle offre de transport."""
    try:
        from models import Company
        from services.demo.soft_delete_guard import (
            company_is_demo,
            institution_is_demo,
        )
        from services.events.institution_events import persist_company_notification

        institution = transport_request.institution
        if institution_is_demo(institution):
            company = Company.query.get(company_id)
            if not company_is_demo(company):
                return
        inst_name = institution.name if institution else "Institution"
        patient = transport_request.patient
        patient_name = f"{patient.first_name} {patient.last_name}" if patient else ""

        sched = transport_request.scheduled_time
        time_str = sched.strftime("%d.%m.%Y %H:%M") if sched else ""
        round_trip = " (A/R)" if transport_request.is_round_trip else ""

        message = f"{inst_name} — {patient_name}{round_trip} — {time_str}".strip(" —")

        persist_company_notification(
            company_id=company_id,
            event_type="new_request",
            title="Nouvelle demande de transport",
            message=message,
            metadata={
                "request_id": transport_request.id,
                "public_id": str(transport_request.public_id),
                "offer_id": offer_id,
                "institution_name": inst_name,
            },
            dedupe_key=f"new_request:{transport_request.id}:{company_id}",
        )
    except Exception as e:
        logger.warning(
            "[RequestOfferTask] Error notifying company %s: %s", company_id, e
        )


@celery.task(
    name="tasks.request_offer_tasks.process_expired_offers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    queue="default",
)
def process_expired_offers(_self: Any) -> dict[str, int]:
    """Traite les offres expirées et déclenche l'escalade si nécessaire.

    Exécuté périodiquement (toutes les minutes recommandé).

    Étapes:
    1. Trouver les offres PENDING expirées (expires_at < now)
    2. Marquer comme EXPIRED
    3. Pour les requests en mode séquentiel:
       - Si préférence suivante existe -> créer next offer
       - Sinon -> fallback broadcast
       - Si aucune company éligible -> request EXPIRED
    """
    logger.info("[RequestOfferTask] Starting process_expired_offers")

    try:
        now = datetime.now(UTC)

        # 1. Trouver les offres PENDING expirées
        expired_offers = RequestOffer.query.filter(
            RequestOffer.status == OfferStatus.PENDING.value,
            RequestOffer.expires_at.isnot(None),
            RequestOffer.expires_at < now,
        ).all()

        logger.info(
            "[RequestOfferTask] Found %d expired offers",
            len(expired_offers),
        )

        processed_requests: set[int] = set()

        for offer in expired_offers:
            try:
                _process_single_expired_offer(offer, now, processed_requests)
            except Exception:
                logger.exception(
                    "[RequestOfferTask] Error processing offer %s",
                    offer.id,
                )
                db.session.rollback()
                continue

        db.session.commit()

        logger.info(
            "[RequestOfferTask] Processed %d expired offers, %d requests escalated/finalized",
            len(expired_offers),
            len(processed_requests),
        )

        return {
            "expired_offers": len(expired_offers),
            "processed_requests": len(processed_requests),
        }

    except Exception:
        logger.exception("[RequestOfferTask] Error in process_expired_offers")
        db.session.rollback()
        raise


def _process_single_expired_offer(
    offer: RequestOffer,
    now: datetime,
    processed_requests: set[int],
) -> None:
    """Traite une offre expirée individuelle."""
    # Marquer comme expirée
    offer.mark_expired()

    # Audit
    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="offer_expired",
            action_category="institution",
            result_status="success",
            action_details={
                "offer_id": offer.id,
                "transport_request_id": offer.transport_request_id,
                "company_id": offer.company_id,
            },
        )

    # GO-LIVE: Métrique expiration
    with contextlib.suppress(Exception):
        from services.metrics.institution_metrics import track_expiration_event

        track_expiration_event(
            offer_id=offer.id,
            transport_request_id=offer.transport_request_id,
            mode=offer.mode,
        )

    # Charger la request
    transport_request = TransportRequest.query.get(offer.transport_request_id)

    if not transport_request:
        return

    # Ne pas escalader si déjà traité ou dans un état final
    if transport_request.id in processed_requests:
        return

    if transport_request.status not in [
        RequestStatus.SENT.value,
    ]:
        return

    # Vérifier s'il reste des offres PENDING pour cette request
    remaining_pending = RequestOffer.query.filter(
        RequestOffer.transport_request_id == transport_request.id,
        RequestOffer.status == OfferStatus.PENDING.value,
    ).count()

    if remaining_pending > 0:
        # Il reste des offres en attente, pas besoin d'escalader
        return

    # Mode séquentiel: escalader
    if offer.mode == OfferMode.SEQUENTIAL.value:
        _escalate_sequential_request(transport_request, offer.order, now)
    else:
        # Broadcast: toutes les offres ont expiré -> request EXPIRED
        _mark_request_expired(transport_request)

    processed_requests.add(transport_request.id)


def _escalate_sequential_request(
    transport_request: TransportRequest,
    current_order: int,
    now: datetime,
) -> None:
    """Escalade une request séquentielle vers la préférence suivante ou fallback."""
    from models import Company
    from services.demo.soft_delete_guard import (
        company_is_demo,
        institution_is_demo,
    )

    # Calculer le timeout (depuis InstitutionSettings)
    timeout_minutes = calculate_timeout(
        transport_request.institution_id,
        transport_request.scheduled_time,
    )

    # Trouver la préférence suivante (institution démo: ignorer les entreprises réelles)
    next_pref = InstitutionTransportPreference.get_next_preference_after(
        transport_request.institution_id,
        current_order,
    )
    if next_pref and institution_is_demo(transport_request.institution):
        while next_pref:
            company = Company.query.get(next_pref.company_id)
            if company_is_demo(company):
                break
            next_pref = InstitutionTransportPreference.get_next_preference_after(
                transport_request.institution_id,
                next_pref.order,
            )
        else:
            next_pref = None

    if next_pref:
        # Créer l'offre suivante
        expires_at = now + timedelta(minutes=timeout_minutes)

        new_offer = RequestOffer(
            transport_request_id=transport_request.id,
            company_id=next_pref.company_id,
            mode=OfferMode.SEQUENTIAL.value,
            order=next_pref.order,
            status=OfferStatus.PENDING.value,
            expires_at=expires_at,
        )
        db.session.add(new_offer)

        logger.info(
            "[RequestOfferTask] Escalated request %s to company %s (order=%d)",
            transport_request.id,
            next_pref.company_id,
            next_pref.order,
        )

        db.session.flush()
        _notify_company_new_offer(transport_request, next_pref.company_id, new_offer.id)

        # Audit
        with contextlib.suppress(Exception):
            AuditLogger.log_action(
                action_type="request_escalated",
                action_category="institution",
                institution_id=transport_request.institution_id,
                result_status="success",
                action_details={
                    "transport_request_id": transport_request.id,
                    "from_order": current_order,
                    "to_order": next_pref.order,
                    "company_id": next_pref.company_id,
                },
            )

        # GO-LIVE: Métrique escalade
        with contextlib.suppress(Exception):
            from services.metrics.institution_metrics import track_escalation_event

            track_escalation_event(
                transport_request_id=transport_request.id,
                from_order=current_order,
                to_order=next_pref.order,
                is_fallback_broadcast=False,
            )

    else:
        # Pas de préférence suivante -> fallback broadcast
        _create_fallback_broadcast(transport_request)


def _create_fallback_broadcast(transport_request: TransportRequest) -> None:
    """Crée des offres broadcast de fallback après épuisement des préférences."""
    from models import Company
    from services.demo.soft_delete_guard import filter_companies_for_institution

    # Récupérer les IDs des entreprises déjà contactées
    existing_offers = RequestOffer.query.filter_by(
        transport_request_id=transport_request.id,
    ).all()
    contacted_company_ids = {o.company_id for o in existing_offers}

    # Récupérer les entreprises éligibles non contactées
    query = Company.query.filter(
        Company.is_approved == True,  # noqa: E712
        Company.dispatch_enabled == True,  # noqa: E712
    )
    if contacted_company_ids:
        query = query.filter(Company.id.notin_(list(contacted_company_ids)))

    eligible = filter_companies_for_institution(
        query.all(), transport_request.institution
    )

    if not eligible:
        # Aucune entreprise disponible -> request EXPIRED
        _mark_request_expired(transport_request)
        logger.warning(
            "[RequestOfferTask] No eligible companies for fallback broadcast, request %s -> EXPIRED",
            transport_request.id,
        )
        return

    # Créer les offres broadcast
    offers_created = 0
    for company in eligible:
        offer = RequestOffer(
            transport_request_id=transport_request.id,
            company_id=company.id,
            mode=OfferMode.BROADCAST.value,
            order=0,
            status=OfferStatus.PENDING.value,
            expires_at=None,
        )
        db.session.add(offer)
        offers_created += 1

    db.session.flush()

    for company in eligible:
        fb_offer = RequestOffer.query.filter_by(
            transport_request_id=transport_request.id,
            company_id=company.id,
        ).first()
        _notify_company_new_offer(
            transport_request, company.id, fb_offer.id if fb_offer else None
        )

    logger.info(
        "[RequestOfferTask] Fallback broadcast for request %s: %d offers created",
        transport_request.id,
        offers_created,
    )

    # Audit
    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="request_fallback_broadcast",
            action_category="institution",
            institution_id=transport_request.institution_id,
            result_status="success",
            action_details={
                "transport_request_id": transport_request.id,
                "offers_created": offers_created,
            },
        )

    # GO-LIVE: Métrique fallback broadcast
    with contextlib.suppress(Exception):
        from services.metrics.institution_metrics import track_escalation_event

        # Trouver le dernier order des préférences (pour le logging)
        last_sequential = (
            RequestOffer.query.filter_by(
                transport_request_id=transport_request.id,
                mode=OfferMode.SEQUENTIAL.value,
            )
            .order_by(RequestOffer.order.desc())
            .first()
        )
        last_order = last_sequential.order if last_sequential else 0

        track_escalation_event(
            transport_request_id=transport_request.id,
            from_order=last_order,
            to_order=None,
            is_fallback_broadcast=True,
        )


def _mark_request_expired(transport_request: TransportRequest) -> None:
    """Marque une request comme expirée (aucune entreprise n'a accepté)."""
    transport_request.status = RequestStatus.EXPIRED.value

    logger.info(
        "[RequestOfferTask] Request %s marked as EXPIRED",
        transport_request.id,
    )

    # Audit
    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="request_expired",
            action_category="institution",
            institution_id=transport_request.institution_id,
            result_status="success",
            action_details={
                "transport_request_id": transport_request.id,
            },
        )


# =============================================================================
# GO-LIVE: Tâche de reporting des métriques métier
# =============================================================================


@celery.task(
    name="tasks.request_offer_tasks.log_institution_metrics",
    bind=True,
    max_retries=1,
    queue="default",
)
def log_institution_metrics(_self: Any, period_hours: int = 24) -> dict[str, Any]:
    """Logge les métriques métier institution (exécuté périodiquement).

    Métriques loggées:
    - Délai moyen send → accept
    - Taux d'expiration
    - Nombre d'escalades vers fallback global

    Args:
        period_hours: Période en heures (défaut: 24h)

    Returns:
        Dictionnaire avec les métriques calculées
    """
    try:
        from services.metrics.institution_metrics import InstitutionMetricsService

        # Calculer et loguer les métriques
        InstitutionMetricsService.log_metrics(period_hours=period_hours)

        # Retourner les métriques pour inspection Celery
        metrics = InstitutionMetricsService.compute_metrics(period_hours=period_hours)
        return metrics.to_dict()

    except Exception:
        logger.exception("[RequestOfferTask] Error in log_institution_metrics")
        return {"error": "Failed to compute metrics"}
