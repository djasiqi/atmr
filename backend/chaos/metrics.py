"""Métriques et mesures pour le chaos engineering.

Permet de mesurer:
- RTO (Recovery Time Objective)
- Métriques de chaos (injections, fallbacks, latences)
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# ✅ D3: Métriques globales de chaos (thread-safe)
_chaos_metrics_lock = threading.Lock()
_chaos_metrics: dict[str, Any] = {
    # Compteurs d'injections de chaos
    "injections_total": defaultdict(int),  # {type: count}
    # Fallbacks utilisés
    "fallback_used": defaultdict(int),  # {fallback_type: count}
    "fallback_success": defaultdict(int),  # {fallback_type: success_count}
    "fallback_failure": defaultdict(int),  # {fallback_type: failure_count}
    # Latences injectées
    "latency_samples": [],  # Liste de (latency_ms, timestamp)
}


class ChaosMetrics:
    """Gestionnaire de métriques de chaos.

    ✅ D3: Thread-safe pour utilisation dans des environnements concurrents.
    """

    @staticmethod
    def record_injection(chaos_type: str) -> None:
        """Enregistre une injection de chaos.

        Args:
            chaos_type: Type de chaos (ex: "osrm_down", "db_read_only", "latency", "error")
        """
        with _chaos_metrics_lock:
            _chaos_metrics["injections_total"][chaos_type] += 1

        logger.debug("[CHAOS] Injection recorded: type=%s", chaos_type)

    @staticmethod
    def record_fallback(fallback_type: str, success: bool) -> None:
        """Enregistre l'utilisation d'un fallback.

        Args:
            fallback_type: Type de fallback (ex: "haversine", "cache", "default")
            success: True si le fallback a réussi, False sinon
        """
        with _chaos_metrics_lock:
            _chaos_metrics["fallback_used"][fallback_type] += 1
            if success:
                _chaos_metrics["fallback_success"][fallback_type] += 1
            else:
                _chaos_metrics["fallback_failure"][fallback_type] += 1

        logger.debug("[CHAOS] Fallback recorded: type=%s, success=%s", fallback_type, success)

    @staticmethod
    def record_latency(latency_ms: float) -> None:
        """Enregistre une latence injectée.

        Args:
            latency_ms: Latence en millisecondes
        """
        with _chaos_metrics_lock:
            # Garder seulement les 1000 derniers échantillons pour éviter explosion mémoire
            MAX_LATENCY_SAMPLES = 1000
            _chaos_metrics["latency_samples"].append((latency_ms, time.time()))
            if len(_chaos_metrics["latency_samples"]) > MAX_LATENCY_SAMPLES:
                _chaos_metrics["latency_samples"].pop(0)

        logger.debug("[CHAOS] Latency recorded: %.1fms", latency_ms)

    @staticmethod
    def get_injections_total(chaos_type: str | None = None) -> Dict[str, int] | int:
        """Récupère le total des injections de chaos.

        Args:
            chaos_type: Type de chaos spécifique, ou None pour tous les types

        Returns:
            Si chaos_type=None: dict {type: count}
            Sinon: count pour le type spécifié
        """
        with _chaos_metrics_lock:
            if chaos_type is None:
                return dict(_chaos_metrics["injections_total"])
            return _chaos_metrics["injections_total"].get(chaos_type, 0)

    @staticmethod
    def get_fallback_stats() -> Dict[str, Dict[str, int]]:
        """Récupère les statistiques de fallback.

        Returns:
            Dict avec les clés:
            - "used": {fallback_type: count}
            - "success": {fallback_type: success_count}
            - "failure": {fallback_type: failure_count}
            - "success_rate": {fallback_type: rate (0.0-1.0)}
        """
        with _chaos_metrics_lock:
            used = dict(_chaos_metrics["fallback_used"])
            success = dict(_chaos_metrics["fallback_success"])
            failure = dict(_chaos_metrics["fallback_failure"])

            # Calculer les taux de succès
            success_rate = {}
            for fallback_type in used:
                total = used.get(fallback_type, 0)
                if total > 0:
                    success_rate[fallback_type] = success.get(fallback_type, 0) / total
                else:
                    success_rate[fallback_type] = 0.0

            return {"used": used, "success": success, "failure": failure, "success_rate": success_rate}

    @staticmethod
    def get_latency_stats() -> Dict[str, float]:
        """Récupère les statistiques de latence injectée.

        Returns:
            Dict avec les clés:
            - "mean_ms": Latence moyenne en ms
            - "min_ms": Latence minimale en ms
            - "max_ms": Latence maximale en ms
            - "count": Nombre d'échantillons
        """
        with _chaos_metrics_lock:
            samples = _chaos_metrics["latency_samples"].copy()

        if not samples:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "count": 0}

        latencies = [latency for latency, _ in samples]

        return {
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "count": len(latencies),
        }

    @staticmethod
    def reset() -> None:
        """Réinitialise toutes les métriques."""
        with _chaos_metrics_lock:
            _chaos_metrics["injections_total"].clear()
            _chaos_metrics["fallback_used"].clear()
            _chaos_metrics["fallback_success"].clear()
            _chaos_metrics["fallback_failure"].clear()
            _chaos_metrics["latency_samples"].clear()

        logger.info("[CHAOS] Metrics reset")


# Instance globale pour faciliter l'utilisation
_metrics = ChaosMetrics()


def get_chaos_metrics() -> ChaosMetrics:
    """Retourne l'instance globale de ChaosMetrics."""
    return _metrics


def measure_rto(
    service_name: str,
    restore_func: Callable[[], None],
    test_func: Callable[[], Any],
    objective_seconds: float,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> float:
    """Mesure le RTO (Recovery Time Objective) d'un service.

    Le RTO est le temps entre la restauration d'un service et la première
    opération réussie après la restauration.

    Args:
        service_name: Nom du service (ex: "osrm", "db")
        restore_func: Fonction qui restaure le service (ex: lambda: injector.set_osrm_down(False))
        test_func: Fonction qui teste une opération sur le service (doit retourner un résultat ou lever une exception)
        objective_seconds: Objectif RTO en secondes (ex: 30.0 pour OSRM)
        max_attempts: Nombre max de tentatives si la première échoue (défaut: 3)
        retry_delay_seconds: Délai entre tentatives en secondes (défaut: 1.0)

    Returns:
        RTO mesuré en secondes

    Raises:
        Exception: Si toutes les tentatives échouent après restauration

    Example:
        >>> from chaos.injectors import get_chaos_injector
        >>> injector = get_chaos_injector()
        >>>
        >>> def restore():
        ...     injector.set_osrm_down(False)
        >>>
        >>> def test():
        ...     from services.osrm_client import get_matrix
        ...     return get_matrix([(0, 0)], [(1, 1)])
        >>>
        >>> rto = measure_rto("osrm", restore, test, objective_seconds=30.0)
    """
    logger.info("[D3] Starting RTO measurement for service: %s (objective: %.1fs)", service_name, objective_seconds)

    # ✅ Restaurer le service
    restore_start = time.time()
    restore_func()
    restore_duration = time.time() - restore_start

    logger.debug("[D3] Service %s restored in %.3fs", service_name, restore_duration)

    # ✅ Mesurer le temps jusqu'à la première opération réussie
    rto_start = time.time()
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            test_func()  # Appeler la fonction de test (peut retourner une valeur ou None)

            # Opération réussie
            rto_seconds = time.time() - rto_start

            # ✅ Logs structurés pour analyse post-mortem
            status = "PASS" if rto_seconds <= objective_seconds else "FAIL"

            logger.info(
                "[D3] RTO measured",
                extra={
                    "service": service_name,
                    "rto_seconds": rto_seconds,
                    "objective_seconds": objective_seconds,
                    "status": status,
                    "attempt": attempt,
                    "restore_duration_seconds": restore_duration,
                    "success": True,
                },
            )

            # Log détaillé en format lisible
            if rto_seconds <= objective_seconds:
                logger.info(
                    "[D3] ✅ RTO PASSED: %.2fs (objective: %.1fs) for service %s",
                    rto_seconds,
                    objective_seconds,
                    service_name,
                )
            else:
                logger.warning(
                    "[D3] ⚠️ RTO FAILED: %.2fs > %.1fs (objective) for service %s",
                    rto_seconds,
                    objective_seconds,
                    service_name,
                )

            return rto_seconds

        except Exception as e:
            last_error = e
            elapsed = time.time() - rto_start

            logger.debug(
                "[D3] Attempt %d/%d failed for service %s after %.3fs: %s",
                attempt,
                max_attempts,
                service_name,
                elapsed,
                e,
            )

            # Si ce n'est pas la dernière tentative, attendre avant de réessayer
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

    # Toutes les tentatives ont échoué
    rto_seconds = time.time() - rto_start

    logger.error(
        "[D3] RTO measurement FAILED",
        extra={
            "service": service_name,
            "rto_seconds": rto_seconds,
            "objective_seconds": objective_seconds,
            "status": "FAIL",
            "attempts": max_attempts,
            "last_error": str(last_error),
            "success": False,
        },
    )

    logger.error(
        "[D3] ❌ All %d attempts failed for service %s after %.2fs (objective: %.1fs)",
        max_attempts,
        service_name,
        rto_seconds,
        objective_seconds,
    )

    # Re-lever la dernière exception
    if last_error:
        raise last_error from last_error
    error_msg = f"RTO measurement failed: all {max_attempts} attempts failed for service {service_name}"
    raise RuntimeError(error_msg) from None
