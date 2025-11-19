"""✅ A5: Métriques cache OSRM avec hit-rate et alarmes.

Objectif: Réduire la latence et surveiller la dérive du cache.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Dict

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import Counter, Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None

logger = logging.getLogger(__name__)

# ==================== Constants ====================

# Seuil hit-rate pour alerter
HIT_RATE_THRESHOLD = 0.70  # 70%

# Fenêtre temporelle pour calculer hit-rate
HIT_RATE_WINDOW_MINUTES = 15

# Constants pour format de cache key
CACHE_KEY_PREFIX = "ud:osrm:matrix:v1"
SLOT_SIZE_MINUTES = 15

# ==================== Prometheus Metrics ====================

# Métriques Prometheus pour cache OSRM (créées uniquement si prometheus_client disponible)
if PROMETHEUS_AVAILABLE and Counter is not None and Gauge is not None:
    OSRM_CACHE_HITS_TOTAL = Counter(
        "osrm_cache_hits_total",
        "Nombre total de hits dans le cache Redis OSRM",
        ["cache_type"],  # cache_type: "route", "table", "matrix"
    )

    OSRM_CACHE_MISSES_TOTAL = Counter(
        "osrm_cache_misses_total",
        "Nombre total de misses dans le cache Redis OSRM",
        ["cache_type"],
    )

    OSRM_CACHE_BYPASS_TOTAL = Counter(
        "osrm_cache_bypass_total",
        "Nombre total de bypass cache (Redis non disponible)",
    )

    OSRM_CACHE_HIT_RATE = Gauge(
        "osrm_cache_hit_rate",
        "Taux de réussite du cache OSRM (0-1)",
    )
else:
    OSRM_CACHE_HITS_TOTAL = None
    OSRM_CACHE_MISSES_TOTAL = None
    OSRM_CACHE_BYPASS_TOTAL = None
    OSRM_CACHE_HIT_RATE = None


# ==================== Cache Metrics Counter ====================


class OSrmCacheMetricsCounter:
    """Compteur thread-safe pour hits/misses cache OSRM."""

    _instance: "OSrmCacheMetricsCounter | None" = None
    _lock = RLock()

    def __init__(self) -> None:  # type: ignore[override]
        """Initialise le compteur."""
        self._hits = 0
        self._misses = 0
        self._bypass_count = 0  # A5: Cache bypass si Redis HS
        self._top_misses: Dict[str, int] = defaultdict(int)
        self._lock_instance = RLock()  # RLock pour éviter deadlock dans _update_hit_rate_gauge()

    @classmethod
    def get_instance(cls) -> "OSrmCacheMetricsCounter":
        """Retourne l'instance singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_hit(self, cache_type: str = "unknown") -> None:
        """Enregistre un hit.

        Args:
            cache_type: Type de cache ("route", "table", "matrix")
        """
        with self._lock_instance:
            self._hits += 1
            # ✅ Exposer métrique Prometheus
            if OSRM_CACHE_HITS_TOTAL:
                OSRM_CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()
            # Mettre à jour hit-rate gauge
            self._update_hit_rate_gauge()

    def record_miss(self, cache_key: str | None = None, cache_type: str = "unknown") -> None:
        """Enregistre un miss (optionnel: clé pour tracking top misses).

        Args:
            cache_key: Clé de cache pour tracking top misses
            cache_type: Type de cache ("route", "table", "matrix")
        """
        with self._lock_instance:
            self._misses += 1
            if cache_key:
                self._top_misses[cache_key] += 1
            # ✅ Exposer métrique Prometheus
            if OSRM_CACHE_MISSES_TOTAL:
                OSRM_CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()
            # Mettre à jour hit-rate gauge
            self._update_hit_rate_gauge()

    def record_bypass(self) -> None:
        """Enregistre un bypass cache (Redis HS)."""
        with self._lock_instance:
            self._bypass_count += 1
            # ✅ Exposer métrique Prometheus
            if OSRM_CACHE_BYPASS_TOTAL:
                OSRM_CACHE_BYPASS_TOTAL.inc()

    def get_hit_rate(self) -> float:
        """Calcule le hit-rate (0-1)."""
        with self._lock_instance:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

    def _update_hit_rate_gauge(self) -> None:
        """Met à jour le gauge Prometheus hit-rate."""
        if OSRM_CACHE_HIT_RATE:
            hit_rate = self.get_hit_rate()
            OSRM_CACHE_HIT_RATE.set(hit_rate)

    def reset(self) -> None:
        """Réinitialise les compteurs."""
        with self._lock_instance:
            self._hits = 0
            self._misses = 0
            self._bypass_count = 0
            self._top_misses.clear()
            # Mettre à jour le gauge Prometheus
            if OSRM_CACHE_HIT_RATE:
                OSRM_CACHE_HIT_RATE.set(0.0)

    def get_top_misses(self, n: int = 10) -> Dict[str, int]:
        """Retourne les N top misses."""
        with self._lock_instance:
            sorted_misses = sorted(self._top_misses.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_misses[:n])

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        with self._lock_instance:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "bypass_count": self._bypass_count,
                "hit_rate": self.get_hit_rate(),
                "total": self._hits + self._misses,
                "top_misses": dict(list(self._top_misses.items())[:10]),
            }


# ==================== Cache Key Generation ====================


def generate_cache_key_v1(profile: str, points: list[tuple[float, float]], date_str: str, slot_15min: int) -> str:
    """Génère une clé de cache stable et reproductible.

    Format: ud:osrm:matrix:v1:{profile}:{YYYYMMDD}:{slot15}:{sha1(points)}

    Args:
        profile: Profile OSRM (driving, walking, etc.)
        points: Liste des coordonnées (lat, lon)
        date_str: Date au format YYYYMMDD
        slot_15min: Slot 15 minutes (0-95)

    Returns:
        Cache key stable
    """
    # Normaliser les points (arrondir à 5 décimales ~1m)
    normalized = [f"{lat:.5f},{lon:.5f}" for lat, lon in points]
    points_str = "|".join(sorted(normalized))  # Trier pour être déterministe

    # Hash SHA256 des points
    sha1_hash = hashlib.sha256(points_str.encode(), usedforsecurity=False).hexdigest()

    # Format: ud:osrm:matrix:v1:{profile}:{YYYYMMDD}:{slot15}:{sha1}
    return f"{CACHE_KEY_PREFIX}:{profile}:{date_str}:{slot_15min}:{sha1_hash}"


def get_slot_15min(now: datetime | None = None) -> int:
    """Retourne le slot 15 min (0-95 par jour).

    Args:
        now: Timestamp (défaut: maintenant)

    Returns:
        Slot 0-95 (0-1425 minutes / 15 = 96 slots)
    """
    if now is None:
        now = datetime.now(UTC)

    minutes_since_midnight = now.hour * 60 + now.minute
    slot = minutes_since_midnight // SLOT_SIZE_MINUTES

    return min(slot, 95)  # Max 95 (0-1425min / 15)


# ==================== Global Functions ====================


def increment_cache_hit(cache_type: str = "unknown") -> None:
    """Incrémente le compteur de hits.

    Args:
        cache_type: Type de cache ("route", "table", "matrix")
    """
    OSrmCacheMetricsCounter.get_instance().record_hit(cache_type=cache_type)


def increment_cache_miss(cache_key: str | None = None, cache_type: str = "unknown") -> None:
    """Incrémente le compteur de misses.

    Args:
        cache_key: Clé de cache pour tracking
        cache_type: Type de cache ("route", "table", "matrix")
    """
    OSrmCacheMetricsCounter.get_instance().record_miss(cache_key, cache_type=cache_type)


def increment_cache_bypass() -> None:
    """Incrémente le compteur de bypass."""
    OSrmCacheMetricsCounter.get_instance().record_bypass()


def get_cache_hit_rate() -> float:
    """Retourne le hit-rate actuel."""
    return OSrmCacheMetricsCounter.get_instance().get_hit_rate()


def reset_cache_metrics() -> None:
    """Réinitialise les métriques (pour tests)."""
    OSrmCacheMetricsCounter.get_instance().reset()


def get_top_misses(n: int = 10) -> Dict[str, int]:
    """Retourne les N top misses."""
    return OSrmCacheMetricsCounter.get_instance().get_top_misses(n)


def get_cache_metrics_dict() -> Dict[str, Any]:
    """Retourne les métriques en dictionnaire."""
    return OSrmCacheMetricsCounter.get_instance().to_dict()


# ==================== Alerts ====================


@dataclass
class CacheAlert:
    """Alerte cache OSRM."""

    severity: str  # "warning" | "critical"
    message: str
    hit_rate: float
    threshold: float
    should_page: bool = False


def check_cache_alert() -> CacheAlert | None:
    """Vérifie si une alerte doit être déclenchée.

    Returns:
        CacheAlert si hit-rate < seuil, None sinon
    """
    hit_rate = get_cache_hit_rate()

    if hit_rate < HIT_RATE_THRESHOLD:
        should_page = hit_rate < (HIT_RATE_THRESHOLD * 0.5)  # < 35% = critical

        severity = "critical" if should_page else "warning"
        message = f"Cache hit-rate bas: {hit_rate:.2%} < {HIT_RATE_THRESHOLD:.2%}" + (
            f" (bypass_count={OSrmCacheMetricsCounter.get_instance()._bypass_count})"
            if OSrmCacheMetricsCounter.get_instance()._bypass_count > 0
            else ""
        )

        return CacheAlert(
            severity=severity, message=message, hit_rate=hit_rate, threshold=HIT_RATE_THRESHOLD, should_page=should_page
        )

    return None
