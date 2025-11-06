"""✅ Export Prometheus metrics pour monitoring complet (dispatch, SLO, OSRM)."""

import logging
import time

from flask import Response, make_response
from flask_restx import Namespace, Resource

from services.unified_dispatch.slo import get_slo_tracker

logger = logging.getLogger(__name__)

prometheus_metrics_ns = Namespace(
    "prometheus",
    description="Prometheus metrics export"
)


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
                slo_metrics.append("# HELP dispatch_slo_breaches_total Total SLO breaches detected")
                slo_metrics.append("# TYPE dispatch_slo_breaches_total counter")
                slo_metrics.append(f'dispatch_slo_breaches_total{{window_minutes="{slo_tracker.window_minutes}"}} {breach_summary["breach_count"]}')
                slo_metrics.append("")
                slo_metrics.append("# HELP dispatch_slo_breach_severity SLO breach severity (0=info, 1=warning, 2=critical)")
                slo_metrics.append("# TYPE dispatch_slo_breach_severity gauge")
                severity_value = 0 if breach_summary["severity"] == "info" else (1 if breach_summary["severity"] == "warning" else 2)
                slo_metrics.append(f'dispatch_slo_breach_severity{{severity="{breach_summary["severity"]}"}} {severity_value}')
                slo_metrics.append("")
                slo_metrics.append("# HELP dispatch_slo_should_alert Whether to page oncall (0=no, 1=yes)")
                slo_metrics.append("# TYPE dispatch_slo_should_alert gauge")
                slo_metrics.append(f'dispatch_slo_should_alert {1 if breach_summary["should_alert"] else 0}')
                
                # Ajouter les métriques par type de breach
                for breach_type, count in breach_summary.get("by_type", {}).items():
                    slo_metrics.append("")
                    slo_metrics.append("# HELP dispatch_slo_breaches_by_type SLO breaches by breach type")
                    slo_metrics.append("# TYPE dispatch_slo_breaches_by_type counter")
                    slo_metrics.append(f'dispatch_slo_breaches_by_type{{type="{breach_type}"}} {count}')
                
                # Combiner toutes les métriques
                slo_metrics_str = "\n".join(slo_metrics)
                
                # Combiner métriques prometheus_client + métriques SLO
                # generate_latest() retourne toujours bytes
                metrics_output_str = metrics_output.decode("utf-8")
                
                full_metrics = metrics_output_str + "\n" + slo_metrics_str
                
                # Utiliser make_response pour éviter la sérialisation JSON de Flask-RESTX
                response = make_response(full_metrics, 200)
                response.mimetype = CONTENT_TYPE_LATEST
                return response
                
            except ImportError:
                # Fallback si prometheus_client non disponible
                logger.warning("[PrometheusMetrics] prometheus_client non disponible, export SLO uniquement")
                current_time = time.time()
                return self._export_slo_only(current_time)
                
        except Exception as e:
            logger.exception("[PrometheusMetrics] Error generating metrics: %s", e)
            return {
                "error": "Failed to generate metrics",
                "details": str(e)
            }, 500
    
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
            f'dispatch_slo_should_alert {1 if breach_summary["should_alert"] else 0}',
        ]
        
        for breach_type, count in breach_summary.get("by_type", {}).items():
            lines.append("")
            lines.append("# HELP dispatch_slo_breaches_by_type SLO breaches by breach type")
            lines.append("# TYPE dispatch_slo_breaches_by_type counter")
            lines.append(f'dispatch_slo_breaches_by_type{{type="{breach_type}"}} {count}')
        
        content = "\n".join(lines)
        
        # Utiliser make_response pour éviter la sérialisation JSON de Flask-RESTX
        response = make_response(content, 200)
        response.mimetype = "text/plain; version=0.0.4; charset=utf-8"
        return response

