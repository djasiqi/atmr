# services/metrics/institution_metrics.py
"""Métriques métier pour le module Institution.

Fournit des métriques simples pour le monitoring business:
- Délai moyen send → accept
- Taux d'expiration des offres
- Nombre d'escalades vers fallback global

Ces métriques sont loggées et peuvent être stockées en DB pour historique.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func

from ext import db
from models import OfferMode, OfferStatus, RequestOffer, RequestStatus, TransportRequest

# Constantes pour éviter les magic values
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

logger = logging.getLogger(__name__)


@dataclass
class InstitutionMetricsSnapshot:
    """Snapshot des métriques institution."""

    # Période
    period_start: datetime
    period_end: datetime

    # Métriques envoi → acceptation
    total_requests_sent: int = 0
    total_requests_converted: int = 0
    avg_send_to_accept_seconds: float | None = None
    min_send_to_accept_seconds: float | None = None
    max_send_to_accept_seconds: float | None = None

    # Métriques expiration
    total_offers_created: int = 0
    total_offers_expired: int = 0
    expiration_rate_percent: float = 0.0

    # Métriques escalade
    total_escalations: int = 0
    fallback_broadcast_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour logging/API."""
        return {
            "period_start": self.period_start.isoformat()
            if self.period_start
            else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "requests": {
                "sent": self.total_requests_sent,
                "converted": self.total_requests_converted,
                "conversion_rate_percent": (
                    round(
                        self.total_requests_converted / self.total_requests_sent * 100,
                        2,
                    )
                    if self.total_requests_sent > 0
                    else 0
                ),
            },
            "send_to_accept": {
                "avg_seconds": round(self.avg_send_to_accept_seconds, 2)
                if self.avg_send_to_accept_seconds
                else None,
                "min_seconds": round(self.min_send_to_accept_seconds, 2)
                if self.min_send_to_accept_seconds
                else None,
                "max_seconds": round(self.max_send_to_accept_seconds, 2)
                if self.max_send_to_accept_seconds
                else None,
                "avg_human": _format_duration(self.avg_send_to_accept_seconds),
            },
            "offers": {
                "total_created": self.total_offers_created,
                "total_expired": self.total_offers_expired,
                "expiration_rate_percent": round(self.expiration_rate_percent, 2),
            },
            "escalations": {
                "total": self.total_escalations,
                "fallback_broadcast": self.fallback_broadcast_count,
            },
        }


def _format_duration(seconds: float | None) -> str | None:
    """Formate une durée en format humain."""
    if seconds is None:
        return None
    if seconds < SECONDS_PER_MINUTE:
        return f"{int(seconds)}s"
    if seconds < SECONDS_PER_HOUR:
        minutes = int(seconds // SECONDS_PER_MINUTE)
        secs = int(seconds % SECONDS_PER_MINUTE)
        return f"{minutes}m {secs}s"
    hours = int(seconds // SECONDS_PER_HOUR)
    minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
    return f"{hours}h {minutes}m"


class InstitutionMetricsService:
    """Service pour calculer les métriques métier institution."""

    @staticmethod
    def compute_metrics(
        institution_id: int | None = None,
        period_hours: int = 24,
    ) -> InstitutionMetricsSnapshot:
        """Calcule les métriques sur une période donnée.

        Args:
            institution_id: Filtrer par institution (None = toutes)
            period_hours: Période en heures (défaut: 24h)

        Returns:
            InstitutionMetricsSnapshot avec les métriques calculées
        """
        now = datetime.now(UTC)
        period_start = now - timedelta(hours=period_hours)

        try:
            # 1. Requêtes envoyées et converties
            requests_query = TransportRequest.query.filter(
                TransportRequest.sent_at >= period_start,
                TransportRequest.sent_at <= now,
            )
            if institution_id:
                requests_query = requests_query.filter(
                    TransportRequest.institution_id == institution_id
                )

            total_sent = requests_query.count()
            total_converted = requests_query.filter(
                TransportRequest.status == RequestStatus.CONVERTED.value
            ).count()

            # 2. Délai send → accept (uniquement requests converties)
            converted_requests = requests_query.filter(
                TransportRequest.status == RequestStatus.CONVERTED.value,
                TransportRequest.sent_at.isnot(None),
                TransportRequest.converted_at.isnot(None),
            ).all()

            send_to_accept_times = []
            for req in converted_requests:
                if req.sent_at and req.converted_at:
                    delta = (req.converted_at - req.sent_at).total_seconds()
                    if delta > 0:  # Ignorer les valeurs aberrantes
                        send_to_accept_times.append(delta)

            avg_time = (
                sum(send_to_accept_times) / len(send_to_accept_times)
                if send_to_accept_times
                else None
            )
            min_time = min(send_to_accept_times) if send_to_accept_times else None
            max_time = max(send_to_accept_times) if send_to_accept_times else None

            # 3. Offres créées et expirées
            offers_query = RequestOffer.query.filter(
                RequestOffer.sent_at >= period_start,
                RequestOffer.sent_at <= now,
            )
            if institution_id:
                offers_query = offers_query.join(TransportRequest).filter(
                    TransportRequest.institution_id == institution_id
                )

            total_offers = offers_query.count()
            expired_offers = offers_query.filter(
                RequestOffer.status == OfferStatus.EXPIRED.value
            ).count()

            expiration_rate = (
                (expired_offers / total_offers * 100) if total_offers > 0 else 0.0
            )

            # 4. Escalades (offres en mode SEQUENTIAL avec order > 0)
            # Une escalade = une offre séquentielle créée après la première (order > 0)
            escalations_query = RequestOffer.query.filter(
                RequestOffer.sent_at >= period_start,
                RequestOffer.sent_at <= now,
                RequestOffer.mode == OfferMode.SEQUENTIAL.value,
                RequestOffer.order > 0,  # Escalade = pas la première préférence
            )
            if institution_id:
                escalations_query = escalations_query.join(TransportRequest).filter(
                    TransportRequest.institution_id == institution_id
                )
            total_escalations = escalations_query.count()

            # 5. Fallback broadcast (offres BROADCAST sur requests qui avaient des préférences)
            # Identifié par: mode=BROADCAST mais des offres SEQUENTIAL existent aussi
            fallback_query = db.session.query(
                func.count(func.distinct(RequestOffer.transport_request_id))
            ).filter(
                RequestOffer.sent_at >= period_start,
                RequestOffer.sent_at <= now,
                RequestOffer.mode == OfferMode.BROADCAST.value,
            )
            # Sous-requête: requests qui ont aussi des offres SEQUENTIAL
            sequential_request_ids = (
                db.session.query(RequestOffer.transport_request_id)
                .filter(RequestOffer.mode == OfferMode.SEQUENTIAL.value)
                .distinct()
                .scalar_subquery()
            )
            fallback_query = fallback_query.filter(
                RequestOffer.transport_request_id.in_(sequential_request_ids)
            )
            fallback_broadcast = fallback_query.scalar() or 0

            return InstitutionMetricsSnapshot(
                period_start=period_start,
                period_end=now,
                total_requests_sent=total_sent,
                total_requests_converted=total_converted,
                avg_send_to_accept_seconds=avg_time,
                min_send_to_accept_seconds=min_time,
                max_send_to_accept_seconds=max_time,
                total_offers_created=total_offers,
                total_offers_expired=expired_offers,
                expiration_rate_percent=expiration_rate,
                total_escalations=total_escalations,
                fallback_broadcast_count=fallback_broadcast,
            )

        except Exception as e:
            logger.exception("Erreur lors du calcul des métriques institution: %s", e)
            return InstitutionMetricsSnapshot(period_start=period_start, period_end=now)

    @staticmethod
    def log_metrics(
        institution_id: int | None = None,
        period_hours: int = 24,
    ) -> None:
        """Calcule et logge les métriques (pour monitoring)."""
        metrics = InstitutionMetricsService.compute_metrics(
            institution_id, period_hours
        )
        metrics_dict = metrics.to_dict()

        log_msg = (
            "📊 [InstitutionMetrics] period=%dh | "
            + "requests: sent=%d converted=%d (%.1f%%) | "
            + "send→accept: avg=%s | "
            + "offers: created=%d expired=%d (%.1f%%) | "
            + "escalations: %d, fallback_broadcast: %d"
        )
        logger.info(
            log_msg,
            period_hours,
            metrics.total_requests_sent,
            metrics.total_requests_converted,
            metrics_dict["requests"]["conversion_rate_percent"],
            metrics_dict["send_to_accept"]["avg_human"] or "N/A",
            metrics.total_offers_created,
            metrics.total_offers_expired,
            metrics.expiration_rate_percent,
            metrics.total_escalations,
            metrics.fallback_broadcast_count,
        )


def track_send_event(
    transport_request_id: int, institution_id: int, mode: str, offers_created: int
) -> None:
    """Tracke un événement d'envoi (appelé depuis SendTransportRequestUseCase)."""
    logger.info(
        "📤 [Metric:Send] request_id=%d institution_id=%d mode=%s offers=%d",
        transport_request_id,
        institution_id,
        mode,
        offers_created,
    )


def track_accept_event(
    offer_id: int,
    transport_request_id: int,
    company_id: int,
    send_to_accept_seconds: float,
) -> None:
    """Tracke un événement d'acceptation (appelé depuis AcceptOfferUseCase)."""
    logger.info(
        "✅ [Metric:Accept] offer_id=%d request_id=%d company_id=%d delay=%.1fs (%s)",
        offer_id,
        transport_request_id,
        company_id,
        send_to_accept_seconds,
        _format_duration(send_to_accept_seconds),
    )


def track_expiration_event(offer_id: int, transport_request_id: int, mode: str) -> None:
    """Tracke un événement d'expiration (appelé depuis process_expired_offers)."""
    logger.info(
        "⏰ [Metric:Expire] offer_id=%d request_id=%d mode=%s",
        offer_id,
        transport_request_id,
        mode,
    )


def track_escalation_event(
    transport_request_id: int,
    from_order: int,
    to_order: int | None,
    is_fallback_broadcast: bool = False,
) -> None:
    """Tracke un événement d'escalade (appelé depuis _escalate_sequential_request)."""
    if is_fallback_broadcast:
        logger.info(
            "🔄 [Metric:Escalate] request_id=%d from_order=%d → FALLBACK_BROADCAST",
            transport_request_id,
            from_order,
        )
    else:
        logger.info(
            "🔄 [Metric:Escalate] request_id=%d from_order=%d → to_order=%d",
            transport_request_id,
            from_order,
            to_order,
        )


def track_offer_unavailable_emitted(
    *,
    company_id: int,
    offer_id: int,
    transport_request_id: int,
    reason: str = "accepted_by_peer",
) -> None:
    """KPI: offre devenue indisponible pour une entreprise (broadcast concurrent)."""
    logger.info(
        "offer_unavailable_emitted company_id=%s offer_id=%s request_id=%s reason=%s",
        company_id,
        offer_id,
        transport_request_id,
        reason,
    )


def track_institution_client_auto_created(
    *,
    institution_id: int,
    client_id: int,
    company_id: int,
) -> None:
    """KPI: client institution créé automatiquement à l'acceptation d'offre."""
    logger.info(
        "institution_client_auto_created institution_id=%s client_id=%s company_id=%s",
        institution_id,
        client_id,
        company_id,
    )


def track_transport_request_duplicate_blocked(
    *,
    transport_request_id: int,
    institution_id: int,
) -> None:
    """KPI: envoi institution bloqué (offres PENDING déjà existantes)."""
    logger.info(
        "transport_request_duplicate_blocked request_id=%s institution_id=%s",
        transport_request_id,
        institution_id,
    )


def track_offer_accept_conflict_409(
    *,
    offer_id: int,
    company_id: int,
    transport_request_id: int | None = None,
    reason: str = "",
) -> None:
    """KPI: conflit d'acceptation (second acteur ou offre déjà traitée)."""
    logger.info(
        "offer_accept_conflict_409 offer_id=%s company_id=%s request_id=%s reason=%s",
        offer_id,
        company_id,
        transport_request_id,
        reason,
    )


def track_proposed_pickup_time_validation_failed(
    *,
    company_id: int,
    offer_id: int,
    reason: str,
) -> None:
    """KPI: horaire proposé refusé par validation backend."""
    logger.info(
        "proposed_pickup_time_validation_failed company_id=%s offer_id=%s reason=%s",
        company_id,
        offer_id,
        reason,
    )


def track_company_push_new_request_sent(*, company_id: int) -> None:
    """Push new_request institution enqueue vers entreprise."""
    logger.info(
        "company_push_new_request_sent company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_new_request_sent

        inc_company_push_new_request_sent(company_id=company_id)
    except Exception:
        logger.debug("company_push_new_request_sent prometheus failed", exc_info=True)


def track_company_push_new_request_delivery_failed(
    *,
    company_id: int,
    reason: str,
) -> None:
    """Échec livraison push new_request (FCM/token/timeout)."""
    logger.info(
        "company_push_new_request_delivery_failed company_id=%s reason=%s",
        company_id,
        reason,
    )
    try:
        from services.monitoring.prometheus import (
            inc_company_push_new_request_delivery_failed,
        )

        inc_company_push_new_request_delivery_failed(reason=reason)
    except Exception:
        logger.debug(
            "company_push_new_request_delivery_failed prometheus failed",
            exc_info=True,
        )


def track_company_push_new_request_opened(*, company_id: int | None = None) -> None:
    """Ouverture notification new_request (télémétrie mobile)."""
    logger.info(
        "company_push_new_request_opened company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_new_request_opened

        inc_company_push_new_request_opened()
    except Exception:
        logger.debug("company_push_new_request_opened prometheus failed", exc_info=True)


def track_company_push_new_request_accept(*, company_id: int) -> None:
    """Acceptation offre institution après notification mobile."""
    logger.info(
        "company_push_new_request_accept company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_new_request_accept

        inc_company_push_new_request_accept()
    except Exception:
        logger.debug("company_push_new_request_accept prometheus failed", exc_info=True)


def track_company_push_new_request_reject(*, company_id: int) -> None:
    """Refus offre institution après notification."""
    logger.info(
        "company_push_new_request_reject company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_new_request_reject

        inc_company_push_new_request_reject()
    except Exception:
        logger.debug("company_push_new_request_reject prometheus failed", exc_info=True)


def track_company_push_new_request_expired(*, company_id: int) -> None:
    """Tentative accept sur offre expirée."""
    logger.info(
        "company_push_new_request_expired company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_new_request_expired

        inc_company_push_new_request_expired()
    except Exception:
        logger.debug(
            "company_push_new_request_expired prometheus failed", exc_info=True
        )


def track_company_push_open_to_accept_seconds(*, seconds: float) -> None:
    """Délai entre ouverture notif et acceptation offre."""
    logger.info(
        "company_push_open_to_accept_seconds seconds=%s",
        seconds,
    )
    try:
        from services.monitoring.prometheus import (
            observe_company_push_open_to_accept_seconds,
        )

        observe_company_push_open_to_accept_seconds(seconds=seconds)
    except Exception:
        logger.debug(
            "company_push_open_to_accept_seconds prometheus failed", exc_info=True
        )


def track_company_push_tap_without_network(*, company_id: int | None = None) -> None:
    """Tap notification/offre sans réseau (usage offline)."""
    logger.info(
        "company_push_tap_without_network company_id=%s",
        company_id,
    )
    try:
        from services.monitoring.prometheus import inc_company_push_tap_without_network

        inc_company_push_tap_without_network()
    except Exception:
        logger.debug(
            "company_push_tap_without_network prometheus failed", exc_info=True
        )
