"""Injecteurs de chaos pour tester la résilience.

Permet d'injecter:
- Latence réseau
- Erreurs HTTP
- Timeouts
- Pannes de services (OSRM, DB, etc.)

⚠️ ATTENTION: Ne JAMAIS activer en production !
Les variables d'environnement suivantes contrôlent le chaos:
- CHAOS_ENABLED: activer/désactiver chaos (défaut: false)
- CHAOS_OSRM_DOWN: simuler OSRM down (défaut: false)
- CHAOS_DB_READ_ONLY: simuler DB read-only (défaut: false)
"""

import logging
import os
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ChaosInjector:
    """Injecteur de chaos pour les tests de résilience.

    ✅ D3: Lit les variables d'environnement au démarrage pour configuration automatique.
    ⚠️ En production, CHAOS_ENABLED doit être 'false' ou non défini.
    """

    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        # ✅ D3: Lire variables d'environnement (défaut: tout désactivé)
        chaos_enabled = os.getenv("CHAOS_ENABLED", "false").lower() in ("true", "1", "yes")
        chaos_osrm_down = os.getenv("CHAOS_OSRM_DOWN", "false").lower() in ("true", "1", "yes")
        chaos_db_read_only = os.getenv("CHAOS_DB_READ_ONLY", "false").lower() in ("true", "1", "yes")

        self.enabled = chaos_enabled
        self.latency_ms = 0
        self.error_rate = 0.0
        self.timeout_rate = 0.0
        self.osrm_down = chaos_osrm_down
        self.db_read_only = chaos_db_read_only

        # Logging au démarrage si chaos activé (pour visibilité)
        if self.enabled:
            logger.warning(
                "[CHAOS] ⚠️ Chaos injection activé via CHAOS_ENABLED=true !, Vérifier que ce n'est PAS en production !"
            )
            if self.osrm_down:
                logger.warning("[CHAOS] OSRM down activé via CHAOS_OSRM_DOWN=true")
            if self.db_read_only:
                logger.warning("[CHAOS] DB read-only activé via CHAOS_DB_READ_ONLY=true")

    def enable(self):
        """Active l'injection de chaos."""
        self.enabled = True
        logger.warning("[CHAOS] Chaos injection enabled")

    def disable(self):
        """Désactive l'injection de chaos."""
        self.enabled = False
        logger.info("[CHAOS] Chaos injection disabled")

    def set_latency(self, ms: int):
        """Configure la latence injectée (en millisecondes)."""
        self.latency_ms = ms
        logger.info("[CHAOS] Latency set to %sms", ms)

        # ✅ D3: Enregistrer la latence injectée dans les métriques
        if ms > 0:
            try:
                from chaos.metrics import get_chaos_metrics

                get_chaos_metrics().record_latency(float(ms))
            except ImportError:
                pass

    def set_error_rate(self, rate: float):
        """Configure le taux d'erreur (0.0 à 1.0)."""
        self.error_rate = rate
        logger.info("[CHAOS] Error rate set to %.1f%%", rate * 100)

    def set_timeout_rate(self, rate: float):
        """Configure le taux de timeout (0.0 à 1.0)."""
        self.timeout_rate = rate
        logger.info("[CHAOS] Timeout rate set to %.1f%%", rate * 100)

    def set_osrm_down(self, down: bool):
        """Configure l'état OSRM (down ou up)."""
        self.osrm_down = down
        logger.warning("[CHAOS] OSRM down: %s", down)

        # ✅ D3: Enregistrer l'injection de chaos
        if down:
            try:
                from chaos.metrics import get_chaos_metrics

                get_chaos_metrics().record_injection("osrm_down")
            except ImportError:
                pass

    def set_db_read_only(self, read_only: bool):
        """Configure la DB en read-only."""
        self.db_read_only = read_only
        logger.warning("[CHAOS] DB read-only: %s", read_only)

        # ✅ D3: Enregistrer l'injection de chaos
        if read_only:
            try:
                from chaos.metrics import get_chaos_metrics

                get_chaos_metrics().record_injection("db_read_only")
            except ImportError:
                pass

    def inject_latency(self, func: Callable[..., T]) -> Callable[..., T]:
        """Décorateur pour injecter de la latence."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and self.latency_ms > 0:
                time.sleep(self.latency_ms / 1000.0)
            return func(*args, **kwargs)

        return wrapper

    def inject_error(self, func: Callable[..., T]) -> Callable[..., T]:
        """Décorateur pour injecter des erreurs."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and random.random() < self.error_rate:
                # ✅ D3: Enregistrer l'injection d'erreur
                try:
                    from chaos.metrics import get_chaos_metrics

                    get_chaos_metrics().record_injection("error")
                except ImportError:
                    pass
                raise ConnectionError(f"[CHAOS] Injected error in {func.__name__}")
            return func(*args, **kwargs)

        return wrapper

    def inject_timeout(self, func: Callable[..., T], timeout_sec: float = 5.0) -> Callable[..., T]:
        """Décorateur pour injecter des timeouts."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and random.random() < self.timeout_rate:
                # ✅ D3: Enregistrer l'injection de timeout
                try:
                    from chaos.metrics import get_chaos_metrics

                    get_chaos_metrics().record_injection("timeout")
                except ImportError:
                    pass
                time.sleep(timeout_sec)
                raise TimeoutError(f"[CHAOS] Injected timeout in {func.__name__}")
            return func(*args, **kwargs)

        return wrapper


# Singleton global
_chaos_injector = ChaosInjector()


def get_chaos_injector() -> ChaosInjector:
    """Retourne l'injecteur de chaos global."""
    return _chaos_injector


def reset_chaos():
    """Réinitialise l'injecteur de chaos."""
    _chaos_injector.disable()
    _chaos_injector.set_latency(0)
    _chaos_injector.set_error_rate(0.0)
    _chaos_injector.set_timeout_rate(0.0)
    _chaos_injector.set_osrm_down(False)
    _chaos_injector.set_db_read_only(False)
