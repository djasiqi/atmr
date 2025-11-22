"""✅ Export Prometheus metrics pour monitoring complet (dispatch, SLO, OSRM)."""

import logging
import time

from flask import Response, make_response
from flask_restx import Namespace, Resource

from services.secret_rotation_monitor import (
    get_days_since_last_rotation,
    get_rotation_stats,
)
from services.unified_dispatch.slo import get_slo_tracker

logger = logging.getLogger(__name__)

prometheus_metrics_ns = Namespace("prometheus", description="Prometheus metrics export")


@prometheus_metrics_ns.route("/metrics")
class PrometheusMetrics(Resource):
    """Export toutes les métriques au format Prometheus pour scraping.

    Exporte:
    - Toutes les métriques dispatch (Counter, Gauge, Histogram)
    - Métriques SLO (breaches, severity, alerts)
    - Métriques OSRM (cache hits/misses, circuit breaker)
    """

    def get(self):
        """Retourne toutes les métriques au format Prometheus.

        Returns:
            Response avec text/plain content-type et métriques Prometheus
        """
        try:
            # ✅ Exporter toutes les métriques prometheus_client via generate_latest()
            try:
                from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

                # Générer toutes les métriques enregistrées (dispatch, OSRM, etc.)
                metrics_output = generate_latest()

                # Ajouter les métriques SLO personnalisées
                current_time = time.time()
                slo_tracker = get_slo_tracker()
                breach_summary = slo_tracker.get_breach_summary(current_time)

                # Construire les métriques SLO au format Prometheus
                slo_metrics = []
                slo_metrics.append(
                    "# HELP dispatch_slo_breaches_total Total SLO breaches detected"
                )
                slo_metrics.append("# TYPE dispatch_slo_breaches_total counter")
                slo_metrics.append(
                    f'dispatch_slo_breaches_total{{window_minutes="{slo_tracker.window_minutes}"}} {breach_summary["breach_count"]}'
                )
                slo_metrics.append("")
                slo_metrics.append(
                    "# HELP dispatch_slo_breach_severity SLO breach severity (0=info, 1=warning, 2=critical)"
                )
                slo_metrics.append("# TYPE dispatch_slo_breach_severity gauge")
                severity_value = (
                    0
                    if breach_summary["severity"] == "info"
                    else (1 if breach_summary["severity"] == "warning" else 2)
                )
                slo_metrics.append(
                    f'dispatch_slo_breach_severity{{severity="{breach_summary["severity"]}"}} {severity_value}'
                )
                slo_metrics.append("")
                slo_metrics.append(
                    "# HELP dispatch_slo_should_alert Whether to page oncall (0=no, 1=yes)"
                )
                slo_metrics.append("# TYPE dispatch_slo_should_alert gauge")
                slo_metrics.append(
                    f"dispatch_slo_should_alert {1 if breach_summary['should_alert'] else 0}"
                )

                # Ajouter les métriques par type de breach
                for breach_type, count in breach_summary.get("by_type", {}).items():
                    slo_metrics.append("")
                    slo_metrics.append(
                        "# HELP dispatch_slo_breaches_by_type SLO breaches by breach type"
                    )
                    slo_metrics.append("# TYPE dispatch_slo_breaches_by_type counter")
                    slo_metrics.append(
                        f'dispatch_slo_breaches_by_type{{type="{breach_type}"}} {count}'
                    )

                # Combiner toutes les métriques
                slo_metrics_str = "\n".join(slo_metrics)

                # Ajouter les métriques de rotation de secrets
                rotation_metrics = self._get_secret_rotation_metrics()

                # Combiner métriques prometheus_client + métriques SLO + métriques rotations
                # generate_latest() retourne toujours bytes
                metrics_output_str = metrics_output.decode("utf-8")

                full_metrics = (
                    metrics_output_str
                    + "\n"
                    + slo_metrics_str
                    + "\n"
                    + rotation_metrics
                )

                # Utiliser make_response pour éviter la sérialisation JSON de Flask-RESTX
                response = make_response(full_metrics, 200)
                response.mimetype = CONTENT_TYPE_LATEST
                return response

            except ImportError:
                # Fallback si prometheus_client non disponible
                logger.warning(
                    "[PrometheusMetrics] prometheus_client non disponible, export SLO uniquement"
                )
                current_time = time.time()
                return self._export_slo_only(current_time)

        except Exception as e:
            logger.exception("[PrometheusMetrics] Error generating metrics: %s", e)
            return {"error": "Failed to generate metrics", "details": str(e)}, 500

    def _export_slo_only(self, current_time: float) -> Response:
        """Export uniquement les métriques SLO (fallback si prometheus_client absent)."""
        slo_tracker = get_slo_tracker()
        breach_summary = slo_tracker.get_breach_summary(current_time)

        lines = [
            "# HELP dispatch_slo_breaches_total Total SLO breaches detected",
            "# TYPE dispatch_slo_breaches_total counter",
            f'dispatch_slo_breaches_total{{window_minutes="{slo_tracker.window_minutes}"}} {breach_summary["breach_count"]}',
            "",
            "# HELP dispatch_slo_breach_severity SLO breach severity (0=info, 1=warning, 2=critical)",
            "# TYPE dispatch_slo_breach_severity gauge",
            f'dispatch_slo_breach_severity{{severity="{breach_summary["severity"]}"}} {0 if breach_summary["severity"] == "info" else (1 if breach_summary["severity"] == "warning" else 2)}',
            "",
            "# HELP dispatch_slo_should_alert Whether to page oncall (0=no, 1=yes)",
            "# TYPE dispatch_slo_should_alert gauge",
            f"dispatch_slo_should_alert {1 if breach_summary['should_alert'] else 0}",
        ]

        for breach_type, count in breach_summary.get("by_type", {}).items():
            lines.append("")
            lines.append(
                "# HELP dispatch_slo_breaches_by_type SLO breaches by breach type"
            )
            lines.append("# TYPE dispatch_slo_breaches_by_type counter")
            lines.append(
                f'dispatch_slo_breaches_by_type{{type="{breach_type}"}} {count}'
            )

        content = "\n".join(lines)

        # Ajouter les métriques de rotation de secrets
        rotation_metrics = self._get_secret_rotation_metrics()

        content = "\n".join(lines) + "\n" + rotation_metrics

        # Utiliser make_response pour éviter la sérialisation JSON de Flask-RESTX
        response = make_response(content, 200)
        response.mimetype = "text/plain; version=0.0.4; charset=utf-8"
        return response

    def _get_secret_rotation_metrics(self) -> str:
        """Génère les métriques Prometheus pour les rotations de secrets.

        Returns:
            Chaîne de caractères au format Prometheus
        """
        try:
            stats = get_rotation_stats()

            lines = []

            # Counter: Total rotations par type et status
            lines.append(
                "# HELP vault_rotation_total Total secret rotations by type and status"
            )
            lines.append("# TYPE vault_rotation_total counter")

            for secret_type, type_stats in stats.get("by_type", {}).items():
                for status in ["success", "error", "skipped"]:
                    count = type_stats.get(status, 0)
                    if count > 0:
                        lines.append(
                            f'vault_rotation_total{{secret_type="{secret_type}",status="{status}"}} {count}'
                        )

            lines.append("")

            # Gauge: Timestamp de dernière rotation réussie par type
            lines.append(
                "# HELP vault_rotation_last_success_timestamp Timestamp of last successful rotation (Unix seconds)"
            )
            lines.append("# TYPE vault_rotation_last_success_timestamp gauge")

            for secret_type, last_rotation_iso in stats.get(
                "last_rotations", {}
            ).items():
                if last_rotation_iso:
                    try:
                        from datetime import datetime

                        last_dt = datetime.fromisoformat(
                            last_rotation_iso.replace("Z", "+00:00")
                        )
                        timestamp = int(last_dt.timestamp())
                        lines.append(
                            f'vault_rotation_last_success_timestamp{{secret_type="{secret_type}"}} {timestamp}'
                        )
                    except Exception:
                        pass  # Ignorer les erreurs de parsing

            lines.append("")

            # Gauge: Jours depuis dernière rotation réussie par type
            lines.append(
                "# HELP vault_rotation_days_since_last Days since last successful rotation"
            )
            lines.append("# TYPE vault_rotation_days_since_last gauge")

            for secret_type in ["jwt", "encryption", "flask_secret_key"]:
                days = get_days_since_last_rotation(secret_type)
                if days is not None:
                    lines.append(
                        f'vault_rotation_days_since_last{{secret_type="{secret_type}"}} {days}'
                    )

            return "\n".join(lines)

        except Exception as e:
            logger.warning(
                "[PrometheusMetrics] Erreur génération métriques rotations: %s", e
            )
            return ""
