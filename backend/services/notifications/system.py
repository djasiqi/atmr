"""✅ 3.6.2: Service d'alertes proactives.

Détecte les problèmes avant impact utilisateur et envoie des alertes
via webhook Slack/Email.
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List

import requests  # pyright: ignore[reportMissingModuleSource]
from requests import (  # pyright: ignore[reportMissingModuleSource]
    RequestException,
    Timeout,
)
from sqlalchemy.exc import DBAPIError, OperationalError

from ext import redis_client
from models import AssignmentStatus, DelayEvent
from repositories.assignment_repository import AssignmentRepository
from services.websocket_metrics import ws_metrics

logger = logging.getLogger(__name__)

# Seuils d'alerte
WEBSOCKET_DISCONNECTION_RATE_THRESHOLD = 0.10  # 10%
ETA_ACCURACY_THRESHOLD = 0.80  # 80%
DISPATCH_DELAY_RATE_THRESHOLD = 0.15  # 15%
OSRM_DOWN_THRESHOLD_SECONDS = 60  # 1 minute
REDIS_DOWN_THRESHOLD_SECONDS = 30  # 30 secondes

# Configuration webhooks
SLACK_WEBHOOK_URL = os.getenv("ALERTING_SLACK_WEBHOOK_URL", default=None)
EMAIL_WEBHOOK_URL = os.getenv("ALERTING_EMAIL_WEBHOOK_URL", default=None)


@dataclass
class Alert:
    """Alerte détectée."""

    severity: str  # "warning", "critical"
    title: str
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }

    def to_slack_message(self) -> Dict[str, Any]:
        """Convertit en message Slack."""
        color = "danger" if self.severity == "critical" else "warning"
        return {
            "attachments": [
                {
                    "color": color,
                    "title": self.title,
                    "text": self.message,
                    "fields": [
                        {
                            "title": "Métrique",
                            "value": self.metric_name,
                            "short": True,
                        },
                        {
                            "title": "Valeur actuelle",
                            "value": f"{self.current_value:.2%}"
                            if isinstance(self.current_value, float)
                            else str(self.current_value),
                            "short": True,
                        },
                        {
                            "title": "Seuil",
                            "value": f"{self.threshold:.2%}"
                            if isinstance(self.threshold, float)
                            else str(self.threshold),
                            "short": True,
                        },
                        {
                            "title": "Sévérité",
                            "value": self.severity.upper(),
                            "short": True,
                        },
                    ],
                    "ts": int(self.timestamp.timestamp()),
                }
            ]
        }


class AlertingService:
    """Service d'alertes proactives."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le service d'alertes."""
        self.slack_webhook_url = SLACK_WEBHOOK_URL
        self.email_webhook_url = EMAIL_WEBHOOK_URL
        self._last_osrm_check: datetime | None = None
        self._last_redis_check: datetime | None = None
        self._osrm_down_since: datetime | None = None
        self._redis_down_since: datetime | None = None

    def check_all_alerts(self) -> List[Alert]:
        """Vérifie toutes les alertes et retourne celles déclenchées.

        Returns:
            Liste des alertes déclenchées
        """
        alerts: List[Alert] = []

        # Vérifier WebSocket
        ws_alert = self._check_websocket_disconnection_rate()
        if ws_alert:
            alerts.append(ws_alert)

        # Vérifier ETA
        eta_alert = self._check_eta_accuracy()
        if eta_alert:
            alerts.append(eta_alert)

        # Vérifier Dispatch
        dispatch_alert = self._check_dispatch_delay_rate()
        if dispatch_alert:
            alerts.append(dispatch_alert)

        # Vérifier OSRM
        osrm_alert = self._check_osrm_health()
        if osrm_alert:
            alerts.append(osrm_alert)

        # Vérifier Redis
        redis_alert = self._check_redis_health()
        if redis_alert:
            alerts.append(redis_alert)

        return alerts

    def _check_websocket_disconnection_rate(self) -> Alert | None:
        """Vérifie le taux de déconnexion WebSocket."""
        try:
            stats = ws_metrics.get_stats()
            connections = stats.get("connections", {})
            total = connections.get("total", 0)
            disconnections = connections.get("disconnections_total", 0)

            if total == 0:
                return None

            disconnection_rate = disconnections / total

            if disconnection_rate > WEBSOCKET_DISCONNECTION_RATE_THRESHOLD:
                severity = (
                    "critical"
                    if disconnection_rate > WEBSOCKET_DISCONNECTION_RATE_THRESHOLD * 2
                    else "warning"
                )

                return Alert(
                    severity=severity,
                    title="Taux de déconnexion WebSocket élevé",
                    message=(
                        f"Taux de déconnexion: {disconnection_rate:.2%} "
                        f"(seuil: {WEBSOCKET_DISCONNECTION_RATE_THRESHOLD:.2%})"
                    ),
                    metric_name="websocket_disconnection_rate",
                    threshold=WEBSOCKET_DISCONNECTION_RATE_THRESHOLD,
                    current_value=disconnection_rate,
                    timestamp=datetime.now(UTC),
                    metadata={
                        "total_connections": total,
                        "disconnections": disconnections,
                    },
                )
        except (KeyError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : clés manquantes, types incorrects
            logger.error(
                "[AlertingService] Error checking WebSocket (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception("[AlertingService] Error checking WebSocket")

        return None

    def _check_eta_accuracy(self) -> Alert | None:
        """Vérifie la précision ETA depuis la base de données."""
        try:
            # ✅ 3.6.2: Calculer précision ETA depuis EtaAccuracyLog
            from models.eta_accuracy_log import EtaAccuracyLog

            # Calculer précision sur les dernières 24h
            cutoff_time = datetime.now(UTC) - timedelta(hours=24)

            # Récupérer les logs ETA récents avec actual_duration_seconds
            logs = (
                EtaAccuracyLog.query.filter(
                    EtaAccuracyLog.created_at >= cutoff_time,
                    EtaAccuracyLog.actual_duration_seconds.isnot(None),
                    EtaAccuracyLog.predicted_eta_seconds.isnot(None),
                )
                .limit(1000)
                .all()
            )

            if not logs:
                # Pas assez de données pour calculer précision
                return None

            # Calculer précision moyenne
            total_accuracy = 0.0
            count = 0

            for log in logs:
                if (
                    log.actual_duration_seconds is not None
                    and log.predicted_eta_seconds is not None
                    and log.actual_duration_seconds > 0
                ):
                    # Précision = 1 - (erreur relative)
                    error = (
                        abs(log.predicted_eta_seconds - log.actual_duration_seconds)
                        / log.actual_duration_seconds
                    )
                    accuracy = max(0.0, 1.0 - error)  # 0-1
                    total_accuracy += accuracy
                    count += 1

            if count == 0:
                return None

            avg_accuracy = total_accuracy / count

            if avg_accuracy < ETA_ACCURACY_THRESHOLD:
                severity = (
                    "critical"
                    if avg_accuracy < ETA_ACCURACY_THRESHOLD * 0.75
                    else "warning"
                )

                return Alert(
                    severity=severity,
                    title="Précision ETA faible",
                    message=(
                        f"Précision ETA moyenne: {avg_accuracy:.2%} "
                        f"(seuil: {ETA_ACCURACY_THRESHOLD:.2%}) "
                        f"sur {count} calculs (24h)"
                    ),
                    metric_name="eta_accuracy_rate",
                    threshold=ETA_ACCURACY_THRESHOLD,
                    current_value=avg_accuracy,
                    timestamp=datetime.now(UTC),
                    metadata={
                        "sample_count": count,
                        "time_window_hours": 24,
                    },
                )
        except ImportError:
            # Modèle EtaAccuracyLog non disponible (migration en cours)
            logger.debug(
                "[AlertingService] EtaAccuracyLog non disponible pour vérification précision"
            )
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.error(
                "[AlertingService] Error checking ETA (DB error: %s): %s",
                type(e).__name__,
                e,
            )
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides
            logger.error(
                "[AlertingService] Error checking ETA (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception("[AlertingService] Error checking ETA")

        return None

    def _check_dispatch_delay_rate(self) -> Alert | None:
        """Vérifie le taux de retard dispatch depuis la base de données."""
        try:
            # ✅ 3.6.2: Calculer taux de retard depuis DelayEvent
            # Calculer taux sur les dernières 24h
            cutoff_time = datetime.now(UTC) - timedelta(hours=24)

            # Compter retards détectés
            total_delays = DelayEvent.query.filter(
                DelayEvent.detected_at >= cutoff_time
            ).count()

            if total_delays == 0:
                # Pas de retards récents
                return None

            # Compter retards critiques/high (qui comptent pour le seuil)
            critical_delays = DelayEvent.query.filter(
                DelayEvent.detected_at >= cutoff_time,
                DelayEvent.severity.in_(["critical", "high"]),
            ).count()

            # Calculer taux de retard (basé sur retards critiques/high)
            # Note: On pourrait aussi calculer depuis les assignments, mais DelayEvent
            # est plus fiable car il reflète les retards réellement détectés
            delay_rate = critical_delays / max(1, total_delays)

            # Alternative: calculer depuis assignments récents
            # Pour l'instant, on utilise le ratio retards critiques / total retards
            # comme proxy du taux de retard global

            # Si on a beaucoup de retards critiques, c'est un problème
            if delay_rate > DISPATCH_DELAY_RATE_THRESHOLD:
                severity = (
                    "critical"
                    if delay_rate > DISPATCH_DELAY_RATE_THRESHOLD * 1.5
                    else "warning"
                )

                return Alert(
                    severity=severity,
                    title="Taux de retard dispatch élevé",
                    message=(
                        f"Taux de retards critiques/high: {delay_rate:.2%} "
                        f"(seuil: {DISPATCH_DELAY_RATE_THRESHOLD:.2%}) "
                        f"sur {total_delays} retards détectés (24h)"
                    ),
                    metric_name="dispatch_delay_rate",
                    threshold=DISPATCH_DELAY_RATE_THRESHOLD,
                    current_value=delay_rate,
                    timestamp=datetime.now(UTC),
                    metadata={
                        "total_delays": total_delays,
                        "critical_delays": critical_delays,
                        "time_window_hours": 24,
                    },
                )

            # Vérifier aussi le taux absolu de retards par rapport aux assignments
            # (nécessite de compter les assignments récents)
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            recent_assignments = assignment_repo.count_by_statuses(
                [
                    AssignmentStatus.ONBOARD,
                    AssignmentStatus.COMPLETED,
                    AssignmentStatus.EN_ROUTE_PICKUP,
                ]
            )

            if recent_assignments > 0:
                absolute_delay_rate = total_delays / recent_assignments
                if absolute_delay_rate > DISPATCH_DELAY_RATE_THRESHOLD:
                    severity = (
                        "critical"
                        if absolute_delay_rate > DISPATCH_DELAY_RATE_THRESHOLD * 1.5
                        else "warning"
                    )

                    return Alert(
                        severity=severity,
                        title="Taux de retard dispatch élevé",
                        message=(
                            f"Taux de retards: {absolute_delay_rate:.2%} "
                            f"(seuil: {DISPATCH_DELAY_RATE_THRESHOLD:.2%}) "
                            f"({total_delays} retards / {recent_assignments} assignments)"
                        ),
                        metric_name="dispatch_delay_rate",
                        threshold=DISPATCH_DELAY_RATE_THRESHOLD,
                        current_value=absolute_delay_rate,
                        timestamp=datetime.now(UTC),
                        metadata={
                            "total_delays": total_delays,
                            "recent_assignments": recent_assignments,
                            "time_window_hours": 24,
                        },
                    )

        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.error(
                "[AlertingService] Error checking dispatch delay (DB error: %s): %s",
                type(e).__name__,
                e,
            )
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides
            logger.error(
                "[AlertingService] Error checking dispatch delay (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception("[AlertingService] Error checking dispatch delay")

        return None

    def _check_osrm_health(self) -> Alert | None:
        """Vérifie la santé OSRM."""
        try:
            import requests  # pyright: ignore[reportMissingModuleSource]

            osrm_url = os.getenv("UD_OSRM_BASE_URL", "http://osrm:5000")
            timeout = 2

            try:
                response = requests.get(f"{osrm_url}/health", timeout=timeout)
                response.raise_for_status()

                # OSRM est up, réinitialiser le compteur
                if self._osrm_down_since:
                    self._osrm_down_since = None

                self._last_osrm_check = datetime.now(UTC)
                return None
            except (RequestException, Timeout, ConnectionError, OSError):
                # OSRM est down : erreurs réseau attendues
                now = datetime.now(UTC)

                if self._osrm_down_since is None:
                    self._osrm_down_since = now

                down_duration = (now - self._osrm_down_since).total_seconds()

                if down_duration > OSRM_DOWN_THRESHOLD_SECONDS:
                    severity = (
                        "critical"
                        if down_duration > OSRM_DOWN_THRESHOLD_SECONDS * 2
                        else "warning"
                    )

                    return Alert(
                        severity=severity,
                        title="OSRM est down",
                        message=(
                            f"OSRM est inaccessible depuis {down_duration:.0f} secondes "
                            f"(seuil: {OSRM_DOWN_THRESHOLD_SECONDS}s)"
                        ),
                        metric_name="osrm_health",
                        threshold=OSRM_DOWN_THRESHOLD_SECONDS,
                        current_value=down_duration,
                        timestamp=now,
                        metadata={"osrm_url": osrm_url},
                    )

        except (RequestException, Timeout, ConnectionError, OSError) as e:
            # Erreurs réseau attendues : connexion OSRM, timeout
            logger.error(
                "[AlertingService] Error checking OSRM (network error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception("[AlertingService] Error checking OSRM")

        return None

    def _check_redis_health(self) -> Alert | None:
        """Vérifie la santé Redis."""
        try:
            if not redis_client:
                return None

            # Ping Redis
            try:
                redis_client.ping()
                # Redis est up, réinitialiser le compteur
                if self._redis_down_since:
                    self._redis_down_since = None

                self._last_redis_check = datetime.now(UTC)
                return None
            except (ConnectionError, OSError, TimeoutError):
                # Redis est down : erreurs réseau attendues
                now = datetime.now(UTC)

                if self._redis_down_since is None:
                    self._redis_down_since = now

                down_duration = (now - self._redis_down_since).total_seconds()

                if down_duration > REDIS_DOWN_THRESHOLD_SECONDS:
                    severity = (
                        "critical"
                        if down_duration > REDIS_DOWN_THRESHOLD_SECONDS * 2
                        else "warning"
                    )

                    return Alert(
                        severity=severity,
                        title="Redis est down",
                        message=(
                            f"Redis est inaccessible depuis {down_duration:.0f} secondes "
                            f"(seuil: {REDIS_DOWN_THRESHOLD_SECONDS}s)"
                        ),
                        metric_name="redis_health",
                        threshold=REDIS_DOWN_THRESHOLD_SECONDS,
                        current_value=down_duration,
                        timestamp=now,
                    )

        except (ConnectionError, OSError, TimeoutError) as e:
            # Erreurs réseau attendues : connexion Redis, timeout
            logger.error(
                "[AlertingService] Error checking Redis (network error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception("[AlertingService] Error checking Redis")

        return None

    def send_alert(self, alert: Alert) -> bool:
        """Envoie une alerte via webhook.

        Args:
            alert: Alerte à envoyer

        Returns:
            True si envoyé avec succès, False sinon
        """
        success = False

        # Envoyer à Slack
        if self.slack_webhook_url:
            try:
                message = alert.to_slack_message()
                response = requests.post(
                    self.slack_webhook_url,
                    json=message,
                    timeout=5,
                )
                response.raise_for_status()
                logger.info("[AlertingService] Alert sent to Slack: %s", alert.title)
                success = True
            except (RequestException, Timeout, ConnectionError) as e:
                # Erreurs réseau attendues : connexion HTTP, timeout
                logger.error(
                    "[AlertingService] Failed to send Slack alert (network error: %s): %s",
                    type(e).__name__,
                    e,
                )
            except (ValueError, TypeError, KeyError) as e:
                # Erreurs de validation attendues : JSON invalide
                logger.error(
                    "[AlertingService] Failed to send Slack alert (validation error: %s): %s",
                    type(e).__name__,
                    e,
                )
            except Exception:
                # Erreur inattendue : logger avec trace complète
                logger.exception("[AlertingService] Failed to send Slack alert")

        # Envoyer à Email (via webhook)
        if self.email_webhook_url:
            try:
                response = requests.post(
                    self.email_webhook_url,
                    json=alert.to_dict(),
                    timeout=5,
                )
                response.raise_for_status()
                logger.info("[AlertingService] Alert sent to Email: %s", alert.title)
                success = True
            except (RequestException, Timeout, ConnectionError) as e:
                # Erreurs réseau attendues : connexion HTTP, timeout
                logger.error(
                    "[AlertingService] Failed to send Email alert (network error: %s): %s",
                    type(e).__name__,
                    e,
                )
            except (ValueError, TypeError, KeyError) as e:
                # Erreurs de validation attendues : JSON invalide
                logger.error(
                    "[AlertingService] Failed to send Email alert (validation error: %s): %s",
                    type(e).__name__,
                    e,
                )
            except Exception:
                # Erreur inattendue : logger avec trace complète
                logger.exception("[AlertingService] Failed to send Email alert")

        return success

    def check_and_send_alerts(self) -> List[Alert]:
        """Vérifie toutes les alertes et envoie celles déclenchées.

        Returns:
            Liste des alertes envoyées
        """
        alerts = self.check_all_alerts()
        sent_alerts: List[Alert] = []

        for alert in alerts:
            if self.send_alert(alert):
                sent_alerts.append(alert)

        return sent_alerts


# Instance globale
_alerting_service_instance: AlertingService | None = None


def get_alerting_service() -> AlertingService:
    """Retourne l'instance singleton du service d'alertes."""
    global _alerting_service_instance  # noqa: PLW0603
    if _alerting_service_instance is None:
        _alerting_service_instance = AlertingService()
    return _alerting_service_instance
