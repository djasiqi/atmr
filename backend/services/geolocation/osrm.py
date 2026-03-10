from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import requests  # pyright: ignore[reportMissingModuleSource]
from cachetools import LRUCache  # pyright: ignore[reportMissingModuleSource]

from services.unified_dispatch.metrics.osrm_cache import (
    increment_cache_bypass,
    increment_cache_hit,
    increment_cache_miss,
)
from shared.geo_utils import haversine_tuple as _haversine_km
from shared.otel_setup import get_tracer  # ✅ D1: OpenTelemetry
from shared.retry import retry_with_backoff  # ✅ 2.3: Retry uniformisé

# pyright: reportUnnecessaryTypeIgnoreComment=false

# ✅ D3: Import chaos injector (optionnel, évite erreur si module absent)
try:
    from chaos.injectors import get_chaos_injector
except ImportError:
    # Si module chaos non disponible, définir fonction no-op
    class _DummyInjector:
        enabled = False
        osrm_down = False
        latency_ms = 0

    def get_chaos_injector() -> Any:  # type: ignore[misc]
        return _DummyInjector()


# Constantes pour éviter les valeurs magiques
RATE_PER_SEC_ZERO = 0
WAIT_ZERO = 0
N_ONE = 1
N_THRESHOLD = 150
N_PERCENT = 100
ORIG_ZERO = 0
CACHE_KEY_MAX_DISPLAY_LENGTH = (
    50  # Longueur maximale pour afficher la clé de cache dans les logs
)
SINGLEFLIGHT_KEY_MAX_DISPLAY_LENGTH = (
    50  # Longueur maximale pour afficher la clé singleflight dans les logs
)

# Seuils pour timeout adaptatif OSRM
OSRM_TIMEOUT_LARGE_MATRIX_THRESHOLD = 150  # Matrices > 150 points → timeout 120s
OSRM_TIMEOUT_MEDIUM_LARGE_THRESHOLD = 100  # Matrices > 100 points → timeout 90s
OSRM_TIMEOUT_MEDIUM_THRESHOLD = 50  # Matrices > 50 points → timeout 60s
OSRM_TIMEOUT_SMALL_MEDIUM_THRESHOLD = 20  # Matrices > 20 points → timeout 45s

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

# ✅ D1: Tracer OpenTelemetry pour traces OSRM
_tracer = get_tracer("osrm")

# ============================================================
# Configuration timeout et retry
# ============================================================
# ✅ Augmenté pour matrices volumineuses (100+ points)
DEFAULT_TIMEOUT = int(os.getenv("UD_OSRM_TIMEOUT", "45"))
DEFAULT_RETRY_COUNT = int(os.getenv("UD_OSRM_RETRY", "2"))
# ✅ Cache plus long pour routes (peu de changements topographiques)
CACHE_TTL_SECONDS = int(os.getenv("UD_OSRM_CACHE_TTL", "7200"))  # 2h par défaut

# ✅ 3.2.4: TTL adaptatif selon fréquence d'utilisation
CACHE_TTL_FREQUENT = int(
    os.getenv("UD_OSRM_CACHE_TTL_FREQUENT", "86400")
)  # 24h pour routes fréquentes
CACHE_TTL_MEDIUM = int(
    os.getenv("UD_OSRM_CACHE_TTL_MEDIUM", "7200")
)  # 2h pour routes moyennes
CACHE_TTL_RARE = int(
    os.getenv("UD_OSRM_CACHE_TTL_RARE", "3600")
)  # 1h pour routes rares

# ✅ P1: TTL pour matrices OSRM (7 jours comme recommandé dans rapport performance)
OSRM_MATRIX_CACHE_TTL = int(
    os.getenv("UD_OSRM_MATRIX_CACHE_TTL", "604800")
)  # 7 jours (604800 secondes)

# ✅ P1: Cache local LRU pour matrices OSRM fréquentes (L1 cache)
# Max 100 entrées pour éviter explosion mémoire
_OSRM_MATRIX_LOCAL_CACHE: LRUCache[str, List[List[float]]] = LRUCache(maxsize=100)
_OSRM_MATRIX_LOCAL_CACHE_LOCK = threading.Lock()

# ✅ P1: Nombre max de workers pour parallélisation OSRM
OSRM_MAX_PARALLEL_WORKERS = int(os.getenv("UD_OSRM_MAX_PARALLEL_WORKERS", "5"))

# ✅ P1: Seuil pour activer parallélisation (si n > seuil et >1 chunk)
OSRM_PARALLEL_THRESHOLD = 20

# Seuils de fréquence pour classification
FREQUENT_ROUTE_THRESHOLD = int(
    os.getenv("UD_OSRM_FREQUENT_THRESHOLD", "10")
)  # ≥10 accès/jour = fréquent
MEDIUM_ROUTE_THRESHOLD = int(
    os.getenv("UD_OSRM_MEDIUM_THRESHOLD", "3")
)  # 3-9 accès/jour = moyen
# <3 accès/jour = rare

# TTL pour compteur de fréquence (reset quotidien)
FREQUENCY_COUNTER_TTL = 86400  # 24h

# ============================================================
# Optional Redis import (safe) + alias d'exception
# ============================================================
try:
    # Import runtime; on évite l'attribut '.exceptions'
    # que Pylance ne connaît pas toujours
    from redis.exceptions import ConnectionError as _RedisConnError  # type: ignore
except Exception:  # redis absent ou API inattendue
    # Définir une classe de fallback
    class _RedisConnError(Exception):  # type: ignore[no-redef]
        pass


# ------------------------------------------------------------
# In-flight de-dup (singleflight) process-local
# ------------------------------------------------------------
_inflight_lock = threading.Lock()
_inflight: Dict[str, Dict[str, Any]] = {}


def _singleflight_do(
    key: str, fn: Callable[[], Any], max_wait_seconds: float = 10.0
) -> Any:
    """Regroupe les appels concurrents sur la même clé.
    Le premier exécute fn(); les autres attendent le résultat.

    Args:
        key: Clé de déduplication
        fn: Fonction à exécuter
        max_wait_seconds: Temps maximum d'attente pour les followers
            (évite blocage indéfini)
    """
    with _inflight_lock:
        entry = _inflight.get(key)
        if entry is None:
            entry = {
                "evt": threading.Event(),
                "result": None,
                "error": None,
                "leader": True,
            }
            _inflight[key] = entry
        else:
            entry["leader"] = False
    if entry["leader"]:
        logger.info(
            "[OSRM] Singleflight leader: executing function for key=%s",
            key[:SINGLEFLIGHT_KEY_MAX_DISPLAY_LENGTH] + "..."
            if len(key) > SINGLEFLIGHT_KEY_MAX_DISPLAY_LENGTH
            else key,
        )
        try:
            res = fn()
            entry["result"] = res
            logger.info("[OSRM] Singleflight leader: function completed successfully")
        except Exception as e:
            entry["error"] = e
            logger.exception(
                "[OSRM] Singleflight leader: function raised exception: %s", str(e)
            )
        finally:
            entry["evt"].set()
            with _inflight_lock:
                _inflight.pop(key, None)
    else:
        # ⚡ Timeout sur l'attente pour éviter blocage indéfini
        # si la requête leader timeout
        if not entry["evt"].wait(timeout=max_wait_seconds):
            # Timeout d'attente → exécuter directement pour éviter blocage en cascade
            logger.warning(
                (
                    "[OSRM] Singleflight wait timeout (%ds) for key=%s..., "
                    "executing directly"
                ),
                max_wait_seconds,
                key[:16],
            )
            try:
                return fn()
            except Exception as e:
                logger.warning(
                    "[OSRM] Direct execution after wait timeout failed: %s", e
                )
                raise
        if entry["error"]:
            raise entry["error"]
    return entry["result"]


# ============================================================
# Fallback / Helpers (rate-limit, haversine, chunking)
# ============================================================

# --- simple per-process rate limiter ---
_rl_lock = threading.Lock()
_rl_last_ts = {"value": 0.0}


def _rate_limit(rate_per_sec: float | None) -> None:
    """Sleep just enough to respect a per-process rate (req/sec)."""
    if rate_per_sec is None or rate_per_sec <= RATE_PER_SEC_ZERO:
        return
    with _rl_lock:
        now = time.time()
        min_interval = 1.0 / float(rate_per_sec)
        wait = _rl_last_ts["value"] + min_interval - now
        if wait > WAIT_ZERO:
            time.sleep(wait)
        _rl_last_ts["value"] = time.time()


def _fallback_matrix(
    coords: List[Tuple[float, float]], avg_kmh: float = 50.0
) -> List[List[float]]:
    # ✅ Standardisé à 50 km/h selon plan d'audit (au lieu de 25 km/h)
    """Fallback durations matrix (seconds) using haversine distance
    and an average speed.
    Symmetric, diagonal 0.0.
    """
    n = len(coords)
    if n <= N_ONE:
        return [[0.0] * n for _ in range(n)]
    M = [[0.0] * n for _ in range(n)]
    speed = max(avg_kmh, 1e-3)  # avoid divide by zero
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = _haversine_km(coords[i], coords[j])
            M[i][j] = float((km / speed) * 3600.0)
    return M


def _fallback_eta_seconds(
    a: Tuple[float, float], b: Tuple[float, float], avg_kmh: float = 50.0
) -> int:
    # ✅ Standardisé à 50 km/h selon plan d'audit (au lieu de 25 km/h)
    km = _haversine_km(a, b)
    sec = (km / max(avg_kmh, 1e-3)) * 3600.0
    return int(max(1, round(sec)))


def _chunks(indices: Iterable[int], size: int) -> Iterable[list[int]]:
    """Yield consecutive chunks (lists) of indices from any iterable."""
    it = iter(indices)
    while True:
        block = list(itertools.islice(it, size))
        if not block:
            break
        yield block


# ============================================================
# OSRM HTTP helpers (sync)
# ============================================================


def _table(
    base_url: str,
    profile: str,
    coords: List[Tuple[float, float]],
    sources: List[int] | None,
    destinations: List[int] | None,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """Appel OSRM table avec retry automatique sur timeout.

    Args:
        timeout: Timeout en secondes (défaut: env UD_OSRM_TIMEOUT ou 30s)

    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    # ✅ 2.3: Utiliser retry uniformisé avec exponential backoff
    from typing import cast

    # ✅ Gestion des erreurs DNS: NameResolutionError de urllib3
    try:
        from urllib3.exceptions import (  # pyright: ignore[reportMissingImports]
            NameResolutionError as Urllib3NameResolutionError,
        )
    except ImportError:
        Urllib3NameResolutionError = None

    retryable_exceptions = (
        requests.Timeout,
        requests.ConnectionError,
        TimeoutError,
    )
    # Ajouter NameResolutionError si disponible (urllib3)
    if Urllib3NameResolutionError is not None:
        retryable_exceptions = (*retryable_exceptions, Urllib3NameResolutionError)

    return cast(
        Dict[str, Any],
        retry_with_backoff(
            lambda: _table_single_request(
                base_url, profile, coords, sources, destinations, timeout
            ),
            max_retries=DEFAULT_RETRY_COUNT,
            base_delay_ms=250,
            max_delay_ms=2000,
            use_jitter=True,
            retryable_exceptions=retryable_exceptions,
            logger_instance=logger,
        ),
    )


def _table_single_request(
    base_url: str,
    profile: str,
    coords: List[Tuple[float, float]],
    sources: List[int] | None,
    destinations: List[int] | None,
    timeout: int | None,
) -> Dict[str, Any]:
    """Exécute une seule requête OSRM table (appelé par _table avec retry).

    ⚠️ D3: Si chaos injector est activé, peut simuler panne OSRM ou injecter latence.
    Chaos ne doit JAMAIS être activé en production (vérifier CHAOS_ENABLED=false).
    """
    # ✅ D3: Vérifier chaos injector avant l'appel HTTP
    injector = get_chaos_injector()
    if injector.enabled and injector.osrm_down:
        logger.warning("[CHAOS] OSRM down simulation - raising ConnectionError")
        raise requests.ConnectionError("[CHAOS] OSRM down simulation")
    if injector.enabled and injector.latency_ms > 0:
        logger.info(
            "[CHAOS] Injecting %sms latency before OSRM table request",
            injector.latency_ms,
        )
        # ✅ D3: Enregistrer la latence réellement injectée
        try:
            from chaos.metrics import get_chaos_metrics

            get_chaos_metrics().record_latency(float(injector.latency_ms))
        except ImportError:
            pass
        time.sleep(injector.latency_ms / 1000.0)

    # ✅ D1: Span pour requête OSRM table
    with _tracer.start_as_current_span("osrm.table") as span:
        span.set_attribute("profile", profile)
        span.set_attribute("coords_count", len(coords))
        span.set_attribute("sources_count", len(sources) if sources else len(coords))
        span.set_attribute(
            "destinations_count", len(destinations) if destinations else len(coords)
        )

        # 6 décimales pour OSRM; la clé de cache utilisera son propre arrondi.
        coord_str = ";".join(f"{lon},{lat}" for (lat, lon) in coords)
        url = f"{base_url}/table/v1/{profile}/{coord_str}"
        params = {"annotations": "duration"}
        if sources is not None:
            params["sources"] = ";".join(map(str, sources))
        if destinations is not None:
            params["destinations"] = ";".join(map(str, destinations))

        span.set_attribute("http.url", url)

        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as e:
            # ✅ Améliorer le logging pour les erreurs DNS/connexion
            error_msg = str(e)
            if (
                "Failed to resolve" in error_msg
                or "Name or service not known" in error_msg
            ):
                logger.error(
                    "[OSRM] DNS resolution failed for host '%s': %s. OSRM service may not be available. Fallback will be used.",
                    base_url,
                    error_msg,
                )
            span.record_exception(e)
            raise

        span.set_attribute("http.status_code", r.status_code)
        span.set_attribute(
            "response_duration_ms", int(r.elapsed.total_seconds() * 1000)
        )

        # ✅ P1: Protéger parsing JSON contre réponses malformées
        try:
            data: Any = r.json()
        except json.JSONDecodeError as e:
            logger.error(
                (
                    "[OSRM] JSON decode error for URL '%s': %s. Response status: %d, "
                    "Response preview: %s"
                ),
                url,
                e,
                r.status_code,
                r.text[:200] if r.text else "(empty)",
            )
            span.record_exception(e)
            # Lever exception pour déclencher retry (si retryable) ou fallback
            raise ValueError(f"OSRM returned invalid JSON: {e}") from e

        return cast("Dict[str, Any]", data)


def _route(
    base_url: str,
    profile: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    waypoints: List[Tuple[float, float]] | None = None,
    overview: str = "false",  # "false" | "simplified" | "full"
    geometries: str = "geojson",
    steps: bool = False,
    annotations: bool = False,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """Exécute une requête OSRM route.

    ⚠️ D3: Si chaos injector est activé, peut simuler panne OSRM ou injecter latence.
    Chaos ne doit JAMAIS être activé en production (vérifier CHAOS_ENABLED=false).
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    # ✅ D3: Vérifier chaos injector avant l'appel HTTP
    injector = get_chaos_injector()
    if injector.enabled and injector.osrm_down:
        logger.warning("[CHAOS] OSRM down simulation - raising ConnectionError")
        raise requests.ConnectionError("[CHAOS] OSRM down simulation")
    if injector.enabled and injector.latency_ms > 0:
        logger.info(
            "[CHAOS] Injecting %sms latency before OSRM route request",
            injector.latency_ms,
        )
        # ✅ D3: Enregistrer la latence réellement injectée
        try:
            from chaos.metrics import get_chaos_metrics

            get_chaos_metrics().record_latency(float(injector.latency_ms))
        except ImportError:
            pass
        time.sleep(injector.latency_ms / 1000.0)

    # ✅ D1: Span pour requête OSRM route
    with _tracer.start_as_current_span("osrm.route") as span:
        span.set_attribute("profile", profile)
        span.set_attribute("has_waypoints", bool(waypoints))

        pts: List[Tuple[float, float]] = [origin]
        if waypoints:
            pts.extend(waypoints)
        pts.append(destination)
        coord_str = ";".join(f"{lon},{lat}" for (lat, lon) in pts)
        url = f"{base_url}/route/v1/{profile}/{coord_str}"
        params = {
            "overview": overview,
            "geometries": geometries,
            "steps": "true" if steps else "false",
            "annotations": "true" if annotations else "false",
            # "continue_straight": "true"  # optionnel
        }

        span.set_attribute("http.url", url)
        span.set_attribute("waypoints_count", len(waypoints) if waypoints else 0)

        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as e:
            # ✅ Améliorer le logging pour les erreurs DNS/connexion
            error_msg = str(e)
            if (
                "Failed to resolve" in error_msg
                or "Name or service not known" in error_msg
            ):
                logger.error(
                    "[OSRM] DNS resolution failed for host '%s': %s. OSRM service may not be available.",
                    base_url,
                    error_msg,
                )
            span.record_exception(e)
            raise

        span.set_attribute("http.status_code", r.status_code)
        span.set_attribute(
            "response_duration_ms", int(r.elapsed.total_seconds() * 1000)
        )

        # ✅ P1: Protéger parsing JSON contre réponses malformées
        try:
            data: Any = r.json()
        except json.JSONDecodeError as e:
            logger.error(
                (
                    "[OSRM] JSON decode error for route URL '%s': %s. Response status: %d, "
                    "Response preview: %s"
                ),
                url,
                e,
                r.status_code,
                r.text[:200] if r.text else "(empty)",
            )
            span.record_exception(e)
            # Lever exception pour déclencher retry (si retryable) ou fallback
            raise ValueError(f"OSRM returned invalid JSON: {e}") from e

        return cast("Dict[str, Any]", data)


# ============================================================
# ✅ 3.2.4: Cache adaptatif - Compteur de fréquence et TTL
# ============================================================


def _increment_frequency_counter(
    redis_client: Any | None, cache_key: str, cache_type: str = "route"
) -> None:
    """Incrémente le compteur de fréquence pour une route/table.

    Args:
        redis_client: Client Redis (peut être None)
        cache_key: Clé de cache (ex: "osrm:route:{key}" ou "osrm:table:{key}")
        cache_type: Type de cache ("route" ou "table")
    """
    if not redis_client:
        return

    # Clé pour compteur de fréquence
    freq_key = f"osrm:freq:{cache_key}"
    try:
        redis_client.incr(freq_key)
        redis_client.expire(freq_key, FREQUENCY_COUNTER_TTL)  # Reset après 24h
        logger.debug(
            "[OSRM Cache] Incremented frequency counter: key=%s type=%s",
            cache_key[:CACHE_KEY_MAX_DISPLAY_LENGTH] + "..."
            if len(cache_key) > CACHE_KEY_MAX_DISPLAY_LENGTH
            else cache_key,
            cache_type,
        )
    except Exception as e:
        # Non-critique : si Redis échoue, on continue sans compteur
        logger.debug("[OSRM Cache] Failed to increment frequency counter: %s", str(e))


def _get_frequency_count(redis_client: Any | None, cache_key: str) -> int:
    """Récupère le compteur de fréquence pour une route/table.

    Args:
        redis_client: Client Redis (peut être None)
        cache_key: Clé de cache

    Returns:
        Nombre d'accès (0 si non disponible)
    """
    if not redis_client:
        return 0

    freq_key = f"osrm:freq:{cache_key}"
    try:
        count = redis_client.get(freq_key)
        if count is None:
            return 0
        return int(count) if isinstance(count, (int, str, bytes)) else 0
    except Exception:
        return 0


def _get_adaptive_ttl(
    redis_client: Any | None,
    cache_key: str,
    default_ttl: int,
    cache_type: str = "route",
) -> int:
    """Calcule TTL adaptatif selon fréquence d'utilisation.

    Args:
        redis_client: Client Redis (peut être None)
        cache_key: Clé de cache
        default_ttl: TTL par défaut (fallback si Redis indisponible)
        cache_type: Type de cache ("route" ou "table")

    Returns:
        TTL en secondes (adaptatif ou default_ttl)
    """
    if not redis_client:
        return default_ttl

    try:
        freq_count = _get_frequency_count(redis_client, cache_key)

        # Classification selon fréquence
        if freq_count >= FREQUENT_ROUTE_THRESHOLD:
            ttl = CACHE_TTL_FREQUENT  # 24h
            category = "frequent"
        elif freq_count >= MEDIUM_ROUTE_THRESHOLD:
            ttl = CACHE_TTL_MEDIUM  # 2h
            category = "medium"
        else:
            ttl = CACHE_TTL_RARE  # 1h
            category = "rare"

        logger.debug(
            "[OSRM Cache] Adaptive TTL: key=%s type=%s freq=%d category=%s ttl=%ds",
            cache_key[:CACHE_KEY_MAX_DISPLAY_LENGTH] + "..."
            if len(cache_key) > CACHE_KEY_MAX_DISPLAY_LENGTH
            else cache_key,
            cache_type,
            freq_count,
            category,
            ttl,
        )

        # ✅ Métrique Prometheus pour routes fréquentes
        try:
            from services.unified_dispatch.metrics.osrm_cache import (
                OSRM_CACHE_FREQUENT_ROUTES_TOTAL,
            )

            if freq_count >= FREQUENT_ROUTE_THRESHOLD:
                OSRM_CACHE_FREQUENT_ROUTES_TOTAL.labels(cache_type=cache_type).inc()  # type: ignore[reportOptionalMemberAccess]
        except (ImportError, AttributeError):
            # Métrique non disponible (module non importé ou non défini)
            pass

        return ttl
    except Exception as e:
        # En cas d'erreur, fallback vers TTL par défaut
        logger.debug(
            "[OSRM Cache] Failed to get adaptive TTL, using default: %s", str(e)
        )
        return default_ttl


# ============================================================
# Cache keys (stable, coord_precision ~ 1m)
# ============================================================


def _canonical_key_table(
    coords: List[Tuple[float, float]],
    sources: List[int] | None,
    destinations: List[int] | None,
    *,
    coord_precision: int = 5,
) -> str:
    def _round(t):
        lat, lon = t
        return (round(lat, coord_precision), round(lon, coord_precision))

    rounded = [_round(c) for c in coords]
    payload = {
        "coords": rounded,
        "sources": sources or "ALL",
        "destinations": destinations or "ALL",
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def _canonical_key_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    waypoints: List[Tuple[float, float]] | None = None,
    *,
    coord_precision: int = 5,
    profile: str = "driving",
) -> str:
    def _round(t):
        lat, lon = t
        return (round(lat, coord_precision), round(lon, coord_precision))

    pts = (
        [_round(origin)]
        + ([_round(w) for w in waypoints] if waypoints else [])
        + [_round(destination)]
    )
    payload = {"profile": profile, "pts": pts}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


# ============================================================
# PUBLIC: Matrix (utilisé par data.build_time_matrix)
# ============================================================


def _get_redis_client_fallback() -> Any | None:
    """Récupère un client Redis avec fallback vers ext.redis_client.

    ✅ P1: S'assurer que redis_client est toujours disponible pour cache OSRM.

    Returns:
        Client Redis ou None si indisponible
    """
    try:
        from ext import redis_client as ext_redis_client

        if ext_redis_client is not None:
            # Tester la connexion
            ext_redis_client.ping()
            return ext_redis_client
    except Exception:
        pass

    # Fallback : essayer de créer depuis REDIS_URL
    try:
        redis_url = os.getenv("REDIS_URL", None)
        if redis_url:
            import redis  # pyright: ignore[reportMissingImports]

            socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
            socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
            client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
            client.ping()
            return client
    except Exception:
        pass

    return None


def build_distance_matrix_osrm(
    coords: List[Tuple[float, float]],
    *,
    base_url: str,
    profile: str = "driving",
    timeout: int | None = None,  # ✅ Timeout adaptatif basé sur taille
    max_sources_per_call: int = 60,
    rate_limit_per_sec: int = 8,
    max_retries: int = 2,
    backoff_ms: int = 250,
    # Cache/mémo optionnel
    redis_client: Any | None = None,
    coord_precision: int = 5,
) -> List[List[float]]:
    """Retourne une matrice de durées en SECONDES (float), shape NxN, diagonale = 0.0.
    Fallback haversine en cas d'échec.

    ✅ P1: Optimisations performance :
    - Cache Redis systématique (fallback si None passé)
    - Cache local LRU (L1 cache) pour matrices fréquentes
    - Parallélisation des appels OSRM avec ThreadPoolExecutor
    - TTL de 7 jours pour cache Redis
    """
    # ✅ P1: S'assurer que redis_client est toujours disponible
    if redis_client is None:
        redis_client = _get_redis_client_fallback()
        if redis_client:
            logger.debug("[OSRM] Using fallback Redis client for matrix cache")
    # ✅ Timeout adaptatif basé sur nombre de coordonnées (amélioré)
    if timeout is None:
        n = len(coords)
        # Timeout adaptatif selon taille (valeurs fixes par seuil)
        base_timeout = 15
        max_timeout = 120  # Maximum 2 minutes pour très grandes matrices

        if n > OSRM_TIMEOUT_LARGE_MATRIX_THRESHOLD:
            timeout = max_timeout  # 120s pour grandes matrices
        elif n > OSRM_TIMEOUT_MEDIUM_LARGE_THRESHOLD:
            timeout = 90  # 90s pour matrices moyennes-grandes
        elif n > OSRM_TIMEOUT_MEDIUM_THRESHOLD:
            timeout = 60  # 60s pour matrices moyennes
        elif n > OSRM_TIMEOUT_SMALL_MEDIUM_THRESHOLD:
            timeout = 45  # 45s pour petites-moyennes
        else:
            timeout = base_timeout  # 15s pour petites matrices

        logger.debug("[OSRM] Timeout adaptatif: %d points → %ds timeout", n, timeout)

    n = len(coords)
    logger.info(
        "[OSRM] build_distance_matrix_osrm entry: n=%d base_url=%s timeout=%s",
        n,
        base_url,
        timeout,
    )
    if n <= N_ONE:
        logger.info("[OSRM] Early return: n=%d <= N_ONE=%d", n, N_ONE)
        return [[0.0] * n for _ in range(n)]

    logger.info("[OSRM] Creating matrix: n=%d", n)
    M = [[0.0] * n for _ in range(n)]
    all_dests = list(range(n))
    logger.info("[OSRM] Matrix created, starting chunking logic")

    # ✅ PERF: Chunking adaptatif - petits chunks pour grandes matrices
    adaptive_chunk_size = 40 if n > N_PERCENT else max_sources_per_call
    total_chunks = (n + adaptive_chunk_size - 1) // adaptive_chunk_size
    logger.info(
        "[OSRM] Starting chunked requests: total_chunks=%d chunk_size=%d n=%d",
        total_chunks,
        adaptive_chunk_size,
        n,
    )

    # ✅ P1: Créer clé de cache globale pour matrice complète (cache local L1)
    # MD5 utilisé uniquement pour générer une clé de cache (non cryptographique)
    full_matrix_cache_key = f"osrm:matrix:{hashlib.md5(json.dumps(coords, sort_keys=True).encode(), usedforsecurity=False).hexdigest()}:{n}"  # nosec B324

    # ✅ P1: Vérifier cache local LRU (L1 cache - ultra rapide)
    with _OSRM_MATRIX_LOCAL_CACHE_LOCK:
        if full_matrix_cache_key in _OSRM_MATRIX_LOCAL_CACHE:
            logger.info("[OSRM] ✅ L1 cache hit (local LRU) for full matrix")
            return _OSRM_MATRIX_LOCAL_CACHE[full_matrix_cache_key]

    # ✅ P1: Vérifier matrices pré-calculées pour zones fréquentes
    # Note: Logique intégrée directement pour éviter cycle d'importation
    try:
        from ext import redis_client as ext_redis_client

        if ext_redis_client and len(coords) > N_ONE:
            # Vérifier si toutes les coordonnées appartiennent à la même zone
            PRECOMPUTE_GRID_SIZE = 0.1  # Grille de 0.1° ≈ 11km
            PRECOMPUTE_CACHE_PREFIX = "osrm:precomputed:zone:"

            def _round_to_grid(coord: Tuple[float, float]) -> Tuple[float, float]:
                lat, lon = coord
                rounded_lat = round(lat / PRECOMPUTE_GRID_SIZE) * PRECOMPUTE_GRID_SIZE
                rounded_lon = round(lon / PRECOMPUTE_GRID_SIZE) * PRECOMPUTE_GRID_SIZE
                return (rounded_lat, rounded_lon)

            zones = {_round_to_grid(coord) for coord in coords}

            # Si toutes les coordonnées sont dans la même zone, chercher la matrice
            if len(zones) == 1:
                zone = zones.pop()
                zone_id = f"{zone[0]:.3f},{zone[1]:.3f}"
                cache_key = f"{PRECOMPUTE_CACHE_PREFIX}{zone_id}:{profile}"

                cached_data = ext_redis_client.get(cache_key)
                if cached_data:
                    try:
                        precomputed_matrix = json.loads(
                            cast(bytes, cached_data).decode("utf-8")
                        )
                        logger.info(
                            "[OSRM] ✅ Using precomputed matrix for zone %s", zone_id
                        )
                        # Stocker dans cache local pour accès ultérieur
                        with _OSRM_MATRIX_LOCAL_CACHE_LOCK:
                            _OSRM_MATRIX_LOCAL_CACHE[full_matrix_cache_key] = (
                                precomputed_matrix
                            )
                        return precomputed_matrix
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(
                            "[OSRM] Failed to decode precomputed matrix: %s", e
                        )
    except Exception as e:
        logger.debug("[OSRM] Error checking precomputed matrix: %s", e)

    # ✅ P1: Préparer les chunks pour traitement (séquentiel ou parallèle)
    chunks_list = list(_chunks(range(n), max(1, int(adaptive_chunk_size))))
    total_chunks_count = len(chunks_list)

    # ✅ P1: Décider si parallélisation (seulement si >1 chunk et n > seuil)
    use_parallel = total_chunks_count > 1 and n > OSRM_PARALLEL_THRESHOLD

    logger.info(
        "[OSRM] Processing %d chunks: parallel=%s adaptive_chunk_size=%d n=%d",
        total_chunks_count,
        use_parallel,
        adaptive_chunk_size,
        n,
    )

    # ✅ P1: Fonction pour traiter un chunk (réutilisable pour séquentiel et parallèle)
    def _process_chunk(
        src_block: List[int],
    ) -> Tuple[List[int], Dict[str, Any] | None, Exception | None]:
        """Traite un chunk et retourne (src_block, data, error)."""
        try:
            # --- Cache key pour ce sous-bloc ---
            cache_key = _canonical_key_table(
                coords, src_block, all_dests, coord_precision=coord_precision
            )

            # ✅ P1: Vérifier cache Redis (L2 cache)
            cached = None
            redis_available = True
            if redis_client is not None:
                try:
                    raw = redis_client.get(f"osrm:table:{cache_key}")
                    if raw:
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode("utf-8", errors="ignore")
                        cached = json.loads(raw)
                        increment_cache_hit(cache_type="table")
                        _increment_frequency_counter(
                            redis_client, f"osrm:table:{cache_key}", cache_type="table"
                        )
                        logger.debug(
                            "[OSRM] L2 cache hit (Redis) for chunk %s", min(src_block)
                        )
                except _RedisConnError:
                    redis_available = False
                    logger.debug(
                        "[OSRM] Redis unavailable for chunk %s", min(src_block)
                    )
                except Exception:
                    logger.debug(
                        "[OSRM] Redis get failed for chunk %s",
                        min(src_block),
                        exc_info=True,
                    )

            # Si cache hit, retourner les données
            if cached and "durations" in cached:
                return (src_block, cached, None)

            # Cache miss - faire la requête OSRM
            increment_cache_miss(cache_key, cache_type="table")

            def _fetch_table_data():
                _rate_limit(rate_limit_per_sec)
                data = _table(
                    base_url=base_url,
                    profile=profile,
                    coords=coords,
                    sources=src_block,
                    destinations=all_dests,
                    timeout=timeout,
                )
                durs = data.get("durations")
                if not durs:
                    msg = "OSRM /table returned no durations"
                    raise RuntimeError(msg)
                if len(durs) != len(src_block):
                    msg = "OSRM durations shape mismatch"
                    raise RuntimeError(msg)
                return data

            # Utiliser singleflight pour éviter requêtes dupliquées
            data_any = _singleflight_do(
                cache_key,
                lambda: retry_with_backoff(
                    _fetch_table_data,
                    max_retries=max_retries,
                    base_delay_ms=backoff_ms,
                    max_delay_ms=5000,
                    use_jitter=True,
                    logger_instance=logger,
                ),
            )

            if not isinstance(data_any, dict):
                raise RuntimeError("OSRM returned invalid data")

            data = cast("Dict[str, Any]", data_any)

            # ✅ P1: Écrire dans cache Redis avec TTL de 7 jours
            if redis_client is not None and redis_available:
                try:
                    table_cache_key = f"osrm:table:{cache_key}"
                    _increment_frequency_counter(
                        redis_client, table_cache_key, cache_type="table"
                    )
                    # ✅ P1: Utiliser TTL de 7 jours pour matrices (recommandation rapport)
                    redis_client.setex(
                        table_cache_key,
                        OSRM_MATRIX_CACHE_TTL,  # 7 jours
                        json.dumps(data),
                    )
                    logger.debug(
                        "[OSRM] L2 cache write (Redis) for chunk %s", min(src_block)
                    )
                except Exception:
                    logger.debug(
                        "[OSRM] Redis setex failed for chunk %s",
                        min(src_block),
                        exc_info=True,
                    )

            return (src_block, data, None)

        except Exception as e:
            logger.warning("[OSRM] Error processing chunk %s: %s", min(src_block), e)
            return (src_block, None, e)

    # ✅ P1: Traitement parallèle ou séquentiel
    chunk_results: Dict[int, Dict[str, Any]] = {}
    chunk_errors: Dict[int, Exception] = {}

    if use_parallel:
        # ✅ P1: Parallélisation avec ThreadPoolExecutor
        logger.info(
            "[OSRM] 🔄 Parallel processing: %d chunks with max %d workers",
            total_chunks_count,
            OSRM_MAX_PARALLEL_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=OSRM_MAX_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(_process_chunk, list(chunk)): chunk
                for chunk in chunks_list
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    src_block, data, error = future.result()
                    if error:
                        chunk_errors[min(src_block)] = error
                    elif data:
                        chunk_results[min(src_block)] = data
                except Exception as e:
                    chunk_errors[min(chunk)] = e
                    logger.warning("[OSRM] Chunk %s failed: %s", min(chunk), e)
    else:
        # Traitement séquentiel (comportement original)
        logger.info("[OSRM] Sequential processing: %d chunks", total_chunks_count)
        for src_block in chunks_list:
            src_block_list = list(src_block)
            _, data, error = _process_chunk(src_block_list)
            if error:
                chunk_errors[min(src_block_list)] = error
            elif data:
                chunk_results[min(src_block_list)] = data

    # ✅ P1: Vérifier si on a des erreurs critiques
    if len(chunk_errors) > 0 and len(chunk_results) == 0:
        # Toutes les requêtes ont échoué -> fallback
        logger.warning(
            "[OSRM] All chunks failed (%d errors), using haversine fallback",
            len(chunk_errors),
        )
        return _fallback_matrix(coords)

    # ✅ P1: Assembler la matrice depuis les résultats des chunks
    logger.info(
        "[OSRM] Assembling matrix from %d successful chunks (%d errors)",
        len(chunk_results),
        len(chunk_errors),
    )

    for src_block in chunks_list:
        src_block_list = list(src_block)
        data = chunk_results.get(min(src_block_list))

        if not data:
            # Chunk en erreur -> remplir avec valeurs de fallback
            logger.warning(
                "[OSRM] Chunk %s missing, using fallback values", min(src_block_list)
            )
            for src_idx in src_block_list:
                for j in range(n):
                    if src_idx == j:
                        M[src_idx][j] = 0.0
                    else:
                        # Utiliser haversine comme fallback pour ce chunk
                        from shared.geo_utils import haversine_seconds

                        lat1, lon1 = coords[src_idx]
                        lat2, lon2 = coords[j]
                        M[src_idx][j] = float(
                            haversine_seconds(
                                lat1, lon1, lat2, lon2, avg_speed_kmh=25.0
                            )
                        )
            continue

        # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
        durs = data.get("durations", [])
        if durs:
            for local_i, src_idx in enumerate(src_block_list):
                if local_i >= len(durs):
                    break
                row = durs[local_i]
                # ✅ P1: Protéger accès liste pour éviter IndexError
                if len(row) < n:
                    logger.warning(
                        "[OSRM] Row length mismatch: expected %d, got %d",
                        n,
                        len(row),
                    )
                    continue
                for j in range(n):
                    v = row[j]
                    M[src_idx][j] = (
                        999999.0 if (v is None or not math.isfinite(v)) else float(v)
                    )

    # ✅ P1: Diagonale à 0
    for i in range(n):
        M[i][i] = 0.0

    # ✅ P1: Mettre en cache local LRU (L1 cache) pour accès ultra-rapide
    with _OSRM_MATRIX_LOCAL_CACHE_LOCK:
        _OSRM_MATRIX_LOCAL_CACHE[full_matrix_cache_key] = M
        logger.debug("[OSRM] ✅ L1 cache write (local LRU) for full matrix")

    return M


# ============================================================
# PUBLIC: Route & ETA (sync) + cache optionnel
# ============================================================


def route_info(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    base_url: str,
    profile: str = "driving",
    waypoints: List[Tuple[float, float]] | None = None,
    timeout: int = 15,  # ✅ Augmenté pour routes longues
    # multi-points
    redis_client: Any | None = None,
    coord_precision: int = 5,
    overview: str = "false",
    geometries: str = "geojson",
    steps: bool = False,
    annotations: bool = False,
    avg_speed_kmh_fallback: float = 50.0,  # ✅ Standardisé à 50 km/h selon plan d'audit
    cache_ttl_s: int | None = None,
) -> Dict[str, Any]:
    """Retourne un dict: {"duration": sec, "distance": m, "geometry": ...,
    "legs": [...]}
    Fallback: haversine + vitesse moyenne.
    """
    key = _canonical_key_route(
        origin, destination, waypoints, coord_precision=coord_precision, profile=profile
    )
    key = (
        f"{key}:ov={overview}:geo={geometries}:st={int(bool(steps))}:"
        f"an={int(bool(annotations))}"
    )
    cache_key = f"osrm:route:{key}"

    # Cache
    if redis_client is not None:
        try:
            raw = redis_client.get(cache_key)
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="ignore")
                cached = json.loads(raw)
                if "duration" in cached and "distance" in cached:
                    # ✅ Track cache hit
                    increment_cache_hit(cache_type="route")
                    # ✅ 3.2.4: Incrémenter compteur de fréquence à chaque accès
                    _increment_frequency_counter(
                        redis_client, cache_key, cache_type="route"
                    )
                    # ✅ S'assurer que fallback est présent dans le cache
                    # (rétrocompatibilité)
                    if "fallback" not in cached:
                        cached["fallback"] = False
                    return cast(Dict[str, Any], cached)
        except _RedisConnError:
            logger.warning("[OSRM] Redis connection failed - continuing without cache")
            increment_cache_bypass()
        except Exception:
            logger.warning("[OSRM] Redis get failed (route)", exc_info=True)
            increment_cache_bypass()

    # ✅ Track cache miss (pas de cache disponible)
    increment_cache_miss(cache_key, cache_type="route")

    def _do():
        data = _route(
            base_url=base_url,
            profile=profile,
            origin=origin,
            destination=destination,
            waypoints=waypoints,
            overview=overview,
            geometries=geometries,
            steps=steps,
            annotations=annotations,
            timeout=timeout,
        )
        if data.get("code") != "Ok" or not data.get("routes"):
            msg = f"OSRM /route bad response: {data.get('message')}"
            raise RuntimeError(msg)
        # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
        routes = data.get("routes", [])
        if not routes:
            raise RuntimeError("No routes in OSRM response")
        r0 = routes[0]
        return {
            "duration": float(r0.get("duration", 0.0)),
            "distance": float(r0.get("distance", 0.0)),
            "geometry": r0.get("geometry"),
            "legs": r0.get("legs", []),
            "fallback": False,  # ✅ Résultat OSRM réel, pas de fallback
        }

    try:
        # ⚡ Timeout d'attente adaptatif : max_wait = timeout OSRM + 2s de marge
        max_wait = float(timeout + 2) if timeout else 12.0
        res_any: Any = _singleflight_do(cache_key, _do, max_wait_seconds=max_wait)
        if not isinstance(res_any, dict):
            msg = "OSRM /route returned non-dict"
            raise RuntimeError(msg)
        res: Dict[str, Any] = cast("Dict[str, Any]", res_any)
    except (
        ConnectionError,
        TimeoutError,
        requests.Timeout,
        requests.ConnectionError,
        requests.RequestException,
        Exception,
    ) as e:
        # ✅ Gestion d'erreurs améliorée : Fallback pour toutes les erreurs
        # (réseau/timeout/autres)
        error_type = (
            "network/timeout"
            if isinstance(
                e,
                (
                    ConnectionError,
                    TimeoutError,
                    requests.Timeout,
                    requests.ConnectionError,
                    requests.RequestException,
                ),
            )
            else "unexpected"
        )
        logger.warning(
            "[OSRM] route failed (%s error) -> fallback haversine: %s", error_type, e
        )
        pts: List[Tuple[float, float]] = [origin] + (waypoints or []) + [destination]
        dist_m = 0.0
        for a, b in itertools.pairwise(pts):
            dist_m += _haversine_km(a, b) * 1000.0
        sec = (dist_m / 1000.0) / max(avg_speed_kmh_fallback, 1e-3) * 3600.0
        res = {
            "duration": float(sec),
            "distance": float(dist_m),
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for (lat, lon) in pts],
            },
            "legs": [{"duration": float(sec), "distance": float(dist_m)}],
            "fallback": True,  # ✅ Marquer comme fallback pour traçabilité
        }

    # Cache set
    try:
        if redis_client is not None:
            # ✅ 3.2.4: TTL adaptatif selon fréquence (si cache_ttl_s non spécifié)
            if cache_ttl_s is None:
                # Incrémenter compteur avant de calculer TTL adaptatif
                _increment_frequency_counter(
                    redis_client, cache_key, cache_type="route"
                )
                ttl = _get_adaptive_ttl(
                    redis_client, cache_key, CACHE_TTL_SECONDS, cache_type="route"
                )
            else:
                ttl = max(int(cache_ttl_s), 0)
                # Incrémenter compteur même si TTL manuel
                _increment_frequency_counter(
                    redis_client, cache_key, cache_type="route"
                )

            if ttl > 0:
                redis_client.setex(cache_key, ttl, json.dumps(res))
                logger.debug("OSRM cache SET key=%s ttl=%ss", cache_key, ttl)
    except _RedisConnError:
        logger.warning(
            "[OSRM] Redis connection failed when writing to cache - continuing without cache"
        )
    except Exception:
        logger.warning("[OSRM] Redis setex failed (route)", exc_info=True)

    return res


def eta_seconds(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    base_url: str,
    profile: str = "driving",
    waypoints: List[Tuple[float, float]] | None = None,
    timeout: int = 10,  # ✅ Augmenté pour destinations lointaines
    redis_client: Any | None = None,
    coord_precision: int = 5,
    avg_speed_kmh_fallback: float = 50.0,  # ✅ Standardisé à 50 km/h selon plan d'audit
) -> int:
    """Calcule un ETA (secondes) robuste via OSRM /route,
    avec cache + fallback haversine."""
    info = route_info(
        origin,
        destination,
        base_url=base_url,
        profile=profile,
        waypoints=waypoints,
        timeout=timeout,
        redis_client=redis_client,
        coord_precision=coord_precision,
        overview="false",
        geometries="geojson",
        steps=False,
        annotations=False,
        avg_speed_kmh_fallback=avg_speed_kmh_fallback,
    )
    dur = info.get("duration", 0.0)
    try:
        return int(max(1, round(float(dur))))
    except Exception:
        return _fallback_eta_seconds(
            origin, destination, avg_kmh=avg_speed_kmh_fallback
        )


# ============================================================
# ✅ NEW: Circuit-Breaker pattern pour OSRM
# ============================================================
class CircuitBreaker:
    """Circuit-breaker pour protéger OSRM des surcharges.
    États : CLOSED (normal) -> OPEN (échecs) -> HALF_OPEN (test) -> CLOSED.
    """

    def __init__(self, failure_threshold: int = 5, timeout_duration: int = 60):
        super().__init__()
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit-breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                # Vérifier si timeout expiré -> passer en HALF_OPEN
                if self.last_failure_time:
                    time_since_last_failure = time.time() - self.last_failure_time
                    if time_since_last_failure >= self.timeout_duration:
                        logger.info(
                            (
                                "[CircuitBreaker] OPEN -> HALF_OPEN "
                                "(timeout expired: %.1fs >= %ds, allowing test request)"
                            ),
                            time_since_last_failure,
                            self.timeout_duration,
                        )
                        self.state = "HALF_OPEN"
                        # Réinitialiser le compteur pour permettre un test
                        self.failure_count = 0
                        # ⚡ Continuer pour tenter l'appel en HALF_OPEN
                    else:
                        remaining = self.timeout_duration - time_since_last_failure
                        msg = (
                            f"CircuitBreaker OPEN (failures: {self.failure_count}, "
                            f"remaining: {remaining:.1f}s/{self.timeout_duration}s)"
                        )
                        logger.warning("[CircuitBreaker] %s", msg)
                        raise Exception(msg)
                else:
                    # Pas de last_failure_time mais état OPEN
                    # -> passer en HALF_OPEN pour test
                    logger.info(
                        "[CircuitBreaker] OPEN -> HALF_OPEN (no last_failure_time, resetting)"
                    )
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                    # ⚡ Continuer pour tenter l'appel en HALF_OPEN

        try:
            result = func(*args, **kwargs)

            # Succès -> reset
            with self._lock:
                if self.state == "HALF_OPEN":
                    logger.info("[CircuitBreaker] HALF_OPEN -> CLOSED (success)")
                    self.state = "CLOSED"
                self.failure_count = 0
                # ✅ Enregistrer métrique Prometheus
                if PROMETHEUS_METRICS_AVAILABLE:
                    # On n'a pas company_id ici, utiliser 0 (global)
                    with contextlib.suppress(Exception):
                        record_circuit_breaker_state(self.state, company_id=0)

            return result

        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                old_state = self.state

                if self.failure_count >= self.failure_threshold:
                    if self.state != "OPEN":
                        logger.warning(
                            (
                                "[CircuitBreaker] %s -> OPEN "
                                "(failures: %d >= threshold: %d, last_error: %s)"
                            ),
                            old_state,
                            self.failure_count,
                            self.failure_threshold,
                            str(e)[:100],
                        )
                    self.state = "OPEN"
                elif self.state == "HALF_OPEN":
                    # En HALF_OPEN, un seul échec remet en OPEN
                    logger.warning(
                        "[CircuitBreaker] HALF_OPEN -> OPEN (test failed, error: %s)",
                        str(e)[:100],
                    )
                    self.state = "OPEN"
                else:
                    logger.debug(
                        "[CircuitBreaker] Échec %d/%d (state: %s, error: %s)",
                        self.failure_count,
                        self.failure_threshold,
                        self.state,
                        str(e)[:100],
                    )

                # ✅ Enregistrer métrique Prometheus après changement d'état
                if PROMETHEUS_METRICS_AVAILABLE:
                    # On n'a pas company_id ici, utiliser 0 (global)
                    with contextlib.suppress(Exception):
                        record_circuit_breaker_state(self.state, company_id=0)
            raise


# Instance globale du circuit-breaker OSRM
_osrm_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_duration=60)

# ✅ Import métriques Prometheus pour circuit breaker
try:
    from services.unified_dispatch.dispatch_prometheus_metrics import (  # type: ignore[import-not-found]
        record_circuit_breaker_state,
    )

    PROMETHEUS_METRICS_AVAILABLE = True
except ImportError:
    PROMETHEUS_METRICS_AVAILABLE = False

    def record_circuit_breaker_state(state: str, company_id: int) -> None:
        pass


def build_distance_matrix_osrm_with_cb(
    coords: List[Tuple[float, float]], **kwargs
) -> List[List[float]]:
    """Wrapper de build_distance_matrix_osrm avec circuit-breaker.
    En cas de circuit ouvert, fallback immédiat vers haversine.
    """
    n = len(coords)
    base_url = kwargs.get("base_url", "http://osrm:5000")
    timeout = kwargs.get("timeout")
    logger.debug(
        "[OSRM] build_distance_matrix_osrm_with_cb entry: n=%d base_url=%s timeout=%s",
        n,
        base_url,
        timeout,
    )
    try:
        logger.debug("[OSRM] Calling circuit breaker for build_distance_matrix_osrm")
        result = _osrm_circuit_breaker.call(
            build_distance_matrix_osrm, coords, **kwargs
        )
        logger.debug(
            "[OSRM] build_distance_matrix_osrm_with_cb success: shape=%dx%d",
            len(result),
            len(result[0]) if result else 0,
        )
        return cast(List[List[float]], result)
    except Exception as e:
        logger.warning(
            (
                "[OSRM] Circuit-breaker triggered or call failed: %s (type=%s), "
                "using haversine fallback"
            ),
            str(e),
            type(e).__name__,
            exc_info=True,
        )
        avg_kmh = kwargs.get("avg_speed_kmh_fallback", 50.0)  # ✅ Standardisé à 50 km/h
        return _fallback_matrix(coords, avg_kmh=avg_kmh)


# ============================================================
# ✅ OSRMClient: Classe pour fallback robuste
# ============================================================


class OSRMClient:
    """Client OSRM avec fallback robuste.

    Encapsule la logique OSRM avec gestion automatique du fallback haversine
    en cas d'échec des requêtes OSRM.
    """

    def __init__(
        self,
        base_url: str,
        profile: str = "driving",
        avg_speed_kmh: float = 50.0,
        redis_client: Any | None = None,
        timeout: int = 15,
    ):
        """Initialise le client OSRM.

        Args:
            base_url: URL de base du serveur OSRM
            profile: Profil de routage (driving, walking, cycling)
            avg_speed_kmh: Vitesse moyenne pour le fallback haversine (défaut: 50 km/h)
            redis_client: Client Redis optionnel pour le cache
            timeout: Timeout par défaut pour les requêtes (secondes)
        """
        super().__init__()
        self.base_url = base_url
        self.profile = profile
        self.avg_speed_kmh = avg_speed_kmh
        self.redis_client = redis_client
        self.timeout = timeout

    def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        waypoints: List[Tuple[float, float]] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Récupère un itinéraire avec fallback automatique.

        Args:
            origin: Point d'origine (lat, lon)
            destination: Point de destination (lat, lon)
            waypoints: Points intermédiaires optionnels
            timeout: Timeout pour la requête (utilise self.timeout si None)
            **kwargs: Arguments supplémentaires passés à route_info()

        Returns:
            Dict avec duration, distance, geometry, legs, et fallback
                (True si fallback utilisé)
        """
        timeout_used = timeout if timeout is not None else self.timeout
        try:
            # Utiliser route_info existante
            result = route_info(
                origin=origin,
                destination=destination,
                base_url=self.base_url,
                profile=self.profile,
                waypoints=waypoints,
                timeout=timeout_used,
                redis_client=self.redis_client,
                avg_speed_kmh_fallback=self.avg_speed_kmh,
                **kwargs,
            )
            # S'assurer que fallback est False si pas déjà défini
            if "fallback" not in result:
                result["fallback"] = False
            return result
        except (
            ConnectionError,
            TimeoutError,
            requests.RequestException,
            requests.Timeout,
        ) as e:
            # Fallback heuristique en cas d'erreur réseau
            logger.warning("[OSRM] Request failed, using fallback: %s", e)
            return self._heuristic_route(origin, destination, waypoints=waypoints)
        except Exception as e:
            # Autres erreurs -> fallback aussi
            logger.warning("[OSRM] Unexpected error, using fallback: %s", e)
            return self._heuristic_route(origin, destination, waypoints=waypoints)

    def _heuristic_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        waypoints: List[Tuple[float, float]] | None = None,
    ) -> Dict[str, Any]:
        """Calcul heuristique de distance/temps avec haversine.

        Args:
            origin: Point d'origine (lat, lon)
            destination: Point de destination (lat, lon)
            waypoints: Points intermédiaires optionnels

        Returns:
            Dict avec distance (m), duration (s), geometry, legs, et fallback=True
        """
        pts: List[Tuple[float, float]] = [origin] + (waypoints or []) + [destination]
        dist_m = 0.0
        for a, b in itertools.pairwise(pts):
            dist_m += _haversine_km(a, b) * 1000.0

        duration_seconds = (dist_m / 1000.0) / max(self.avg_speed_kmh, 1e-3) * 3600.0

        return {
            "distance": float(dist_m),  # mètres
            "duration": float(duration_seconds),  # secondes
            "fallback": True,  # ⚠️ Marquer comme fallback
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for (lat, lon) in pts],
            },
            "legs": [{"duration": float(duration_seconds), "distance": float(dist_m)}],
        }


# ============================================================
# Helpers de haut niveau pour distance/temps et matrices
# ============================================================


def get_distance_time(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    *,
    base_url: str | None = None,
    profile: str = "driving",
    redis_client: Any | None = None,
) -> Dict[str, float]:
    """Retourne un dict {"distance": m, "duration": s} en utilisant route_info.
    base_url est requis par les appels existants du module
    (utiliser OSRM_BASE_URL sinon).
    """
    # Résolution de l'URL de base
    osrm_base = base_url or os.getenv("OSRM_BASE_URL", "http://localhost:5000")
    # Garantir que osrm_base est toujours un str (os.getenv peut retourner None)
    if not osrm_base:
        osrm_base = "http://localhost:5000"
    info = route_info(
        origin, dest, base_url=osrm_base, profile=profile, redis_client=redis_client
    )
    return {
        "distance": float(info.get("distance", 0.0)),
        "duration": float(info.get("duration", 0.0)),
    }


def get_matrix(
    origins: List[Tuple[float, float]],
    destinations: List[Tuple[float, float]],
    *,
    base_url: str | None = None,
    profile: str = "driving",
    redis_client: Any | None = None,
) -> Dict[str, Any]:
    """Construit une matrice de durées (secondes) entre origines et destinations.
    Retourne {"durations": List[List[float]]}.
    """
    osrm_base = base_url or os.getenv("OSRM_BASE_URL", "http://localhost:5000")
    # Concaténer pour construire une matrice NxN en référençant toutes les coordonnées
    # Ici, on construit une matrice complète sur l'ensemble unique des points
    all_points = list(origins)
    # Assurer que destinations sont incluses; si ce sont les mêmes, pas de duplication
    for pt in destinations:
        if pt not in all_points:
            all_points.append(pt)

    durations = build_distance_matrix_osrm_with_cb(
        all_points,
        base_url=osrm_base,
        profile=profile,
        redis_client=redis_client,
    )

    # Si origins/destinations sont des sous-ensembles/ordres différents,
    # on extrait la sous-matrice correspondante
    idx = {pt: i for i, pt in enumerate(all_points)}
    sub = []
    for o in origins:
        row = []
        oi = idx[o]
        for d in destinations:
            di = idx[d]
            row.append(durations[oi][di])
        sub.append(row)
    return {"durations": sub}


# ============================================================
# Cache Redis pour matrices journalières
# ============================================================


def get_distance_time_cached(origin, dest, date_str=None):
    """Récupère la distance et le temps entre deux points avec cache Redis.

    Args:
        origin: Point d'origine (lat, lon)
        dest: Point de destination (lat, lon)
        date_str: Date pour le cache (optionnel, défaut: aujourd'hui)

    Returns:
        Dict avec 'distance' et 'duration' en mètres et secondes

    """
    if date_str is None:
        from datetime import datetime

        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    # Créer une clé de cache plus robuste
    # (SHA-256 au lieu de MD5 pour meilleures pratiques)
    origin_hash = hashlib.sha256(
        f"{origin[ORIG_ZERO]},{origin[1]}".encode()
    ).hexdigest()[:8]
    dest_hash = hashlib.sha256(f"{dest[0]},{dest[1]}".encode()).hexdigest()[:8]
    cache_key = f"osrm:cache:{date_str}:{origin_hash}:{dest_hash}"

    try:
        from ext import redis_client as rc

        if rc is None:
            raise Exception("Redis client not available")
        raw_any = rc.get(cache_key)
        if raw_any:
            raw = raw_any
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            if not isinstance(raw, str):
                raw = str(raw)
            return json.loads(raw)
    except Exception as e:
        logger.warning("[OSRM] Cache read error: %s", e)

    # Calculer la distance et le temps
    result = get_distance_time(origin, dest)

    try:
        from ext import redis_client as rc

        if rc is None:
            raise Exception("Redis client not available")
        rc.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result))
    except Exception as e:
        logger.warning("[OSRM] Cache write error: %s", e)

    return result


def get_matrix_cached(origins, destinations, date_str=None):
    """Récupère la matrice de distances/temps avec cache Redis par jour.

    Args:
        origins: Liste des points d'origine [(lat, lon), ...]
        destinations: Liste des points de destination [(lat, lon), ...]
        date_str: Date pour le cache (optionnel, défaut: aujourd'hui)

    Returns:
        Dict avec 'distances' et 'durations' (matrices)

    """
    if date_str is None:
        from datetime import datetime

        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    # Créer une clé de cache pour la matrice
    # (SHA-256 au lieu de MD5 pour meilleures pratiques)
    origins_str = ",".join([f"{o[0]},{o[1]}" for o in origins])
    dests_str = ",".join([f"{d[0]},{d[1]}" for d in destinations])
    matrix_hash = hashlib.sha256(f"{origins_str}|{dests_str}".encode()).hexdigest()[:16]
    cache_key = f"osrm:matrix:{date_str}:{matrix_hash}"

    try:
        from ext import redis_client as rc

        if rc is None:
            raise Exception("Redis client not available")
        raw_any = rc.get(cache_key)
        if raw_any:
            logger.info(
                "[OSRM] Matrix cache hit for %sx%s points",
                len(origins),
                len(destinations),
            )
            raw = raw_any
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            if not isinstance(raw, str):
                raw = str(raw)
            return json.loads(raw)
    except Exception as e:
        logger.warning("[OSRM] Matrix cache read error: %s", e)

    # Calculer la matrice
    result = get_matrix(origins, destinations)

    try:
        from ext import redis_client as rc

        if rc is None:
            raise Exception("Redis client not available")
        rc.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result))
        logger.info(
            "[OSRM] Matrix cached for %sx%s points", len(origins), len(destinations)
        )
    except Exception as e:
        logger.warning("[OSRM] Matrix cache write error: %s", e)

    return result
