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
from datetime import UTC
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import requests

from services.unified_dispatch.osrm_cache_metrics import (
    increment_cache_bypass,
    increment_cache_hit,
    increment_cache_miss,
)
from shared.geo_utils import haversine_tuple as _haversine_km
from shared.otel_setup import get_tracer  # ✅ D1: OpenTelemetry
from shared.retry import retry_with_backoff  # ✅ 2.3: Retry uniformisé

# ✅ D3: Import chaos injector (optionnel, évite erreur si module absent)
try:
    from chaos.injectors import get_chaos_injector
except ImportError:
    # Si module chaos non disponible, définir fonction no-op
    class _DummyInjector:
        enabled = False
        osrm_down = False
        latency_ms = 0

    def get_chaos_injector() -> Any:
        return _DummyInjector()


# Constantes pour éviter les valeurs magiques
RATE_PER_SEC_ZERO = 0
WAIT_ZERO = 0
N_ONE = 1
N_THRESHOLD = 150
N_PERCENT = 100
ORIG_ZERO = 0
CACHE_KEY_MAX_DISPLAY_LENGTH = 50  # Longueur maximale pour afficher la clé de cache dans les logs
SINGLEFLIGHT_KEY_MAX_DISPLAY_LENGTH = 50  # Longueur maximale pour afficher la clé singleflight dans les logs

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

# ============================================================
# Optional Redis import (safe) + alias d'exception
# ============================================================
try:
    # Import runtime; on évite l'attribut '.exceptions' que Pylance ne connaît pas toujours
    from redis.exceptions import ConnectionError as _RedisConnError  # type: ignore
except Exception:  # redis absent ou API inattendue

    class _RedisConnError(Exception):
        pass


# ------------------------------------------------------------
# In-flight de-dup (singleflight) process-local
# ------------------------------------------------------------
_inflight_lock = threading.Lock()
_inflight: Dict[str, Dict[str, Any]] = {}


def _singleflight_do(key: str, fn: Callable[[], Any], max_wait_seconds: float = 10.0) -> Any:
    """Regroupe les appels concurrents sur la même clé.
    Le premier exécute fn(); les autres attendent le résultat.

    Args:
        key: Clé de déduplication
        fn: Fonction à exécuter
        max_wait_seconds: Temps maximum d'attente pour les followers (évite blocage indéfini)
    """
    with _inflight_lock:
        entry = _inflight.get(key)
        if entry is None:
            entry = {"evt": threading.Event(), "result": None, "error": None, "leader": True}
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
            logger.exception("[OSRM] Singleflight leader: function raised exception: %s", str(e))
        finally:
            entry["evt"].set()
            with _inflight_lock:
                _inflight.pop(key, None)
    else:
        # ⚡ Timeout sur l'attente pour éviter blocage indéfini si la requête leader timeout
        if not entry["evt"].wait(timeout=max_wait_seconds):
            # Timeout d'attente → exécuter directement pour éviter blocage en cascade
            logger.warning(
                "[OSRM] Singleflight wait timeout (%ds) for key=%s..., executing directly",
                max_wait_seconds,
                key[:16],
            )
            try:
                return fn()
            except Exception as e:
                logger.warning("[OSRM] Direct execution after wait timeout failed: %s", e)
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


def _fallback_matrix(coords: List[Tuple[float, float]], avg_kmh: float = 25.0) -> List[List[float]]:
    """Fallback durations matrix (seconds) using haversine distance and an average speed.
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


def _fallback_eta_seconds(a: Tuple[float, float], b: Tuple[float, float], avg_kmh: float = 25.0) -> int:
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

    return cast(
        Dict[str, Any],
        retry_with_backoff(
            lambda: _table_single_request(base_url, profile, coords, sources, destinations, timeout),
            max_retries=DEFAULT_RETRY_COUNT,
            base_delay_ms=250,
            max_delay_ms=2000,
            use_jitter=True,
            retryable_exceptions=(requests.Timeout, requests.ConnectionError, TimeoutError),
            logger_instance=logger,
        ),
    )


def _table_single_request(base_url, profile, coords, sources, destinations, timeout):
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
        logger.info("[CHAOS] Injecting %sms latency before OSRM table request", injector.latency_ms)
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
        span.set_attribute("destinations_count", len(destinations) if destinations else len(coords))

        # 6 décimales pour OSRM; la clé de cache utilisera son propre arrondi.
        coord_str = ";".join(f"{lon},{lat}" for (lat, lon) in coords)
        url = f"{base_url}/table/v1/{profile}/{coord_str}"
        params = {"annotations": "duration"}
        if sources is not None:
            params["sources"] = ";".join(map(str, sources))
        if destinations is not None:
            params["destinations"] = ";".join(map(str, destinations))

        span.set_attribute("http.url", url)

        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()

        span.set_attribute("http.status_code", r.status_code)
        span.set_attribute("response_duration_ms", int(r.elapsed.total_seconds() * 1000))

        data: Any = r.json()
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
        logger.info("[CHAOS] Injecting %sms latency before OSRM route request", injector.latency_ms)
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

        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()

        span.set_attribute("http.status_code", r.status_code)
        span.set_attribute("response_duration_ms", int(r.elapsed.total_seconds() * 1000))

        data: Any = r.json()
        return cast("Dict[str, Any]", data)


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

    pts = [_round(origin)] + ([_round(w) for w in waypoints] if waypoints else []) + [_round(destination)]
    payload = {"profile": profile, "pts": pts}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


# ============================================================
# PUBLIC: Matrix (utilisé par data.build_time_matrix)
# ============================================================


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
    """
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
    logger.info("[OSRM] build_distance_matrix_osrm entry: n=%d base_url=%s timeout=%s", n, base_url, timeout)
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
        "[OSRM] Starting chunked requests: total_chunks=%d chunk_size=%d n=%d", total_chunks, adaptive_chunk_size, n
    )

    logger.info("[OSRM] Entering chunk loop: adaptive_chunk_size=%d", adaptive_chunk_size)
    for src_block in _chunks(range(n), max(1, int(adaptive_chunk_size))):
        logger.info("[OSRM] Processing chunk: src_block=%s (len=%d)", str(list(src_block)[:5]) + "...", len(src_block))
        # --- Cache key pour ce sous-bloc ---
        logger.info("[OSRM] Creating cache key for chunk...")
        cache_key = _canonical_key_table(coords, list(src_block), all_dests, coord_precision=coord_precision)
        logger.info(
            "[OSRM] Cache key created: %s",
            cache_key[:CACHE_KEY_MAX_DISPLAY_LENGTH] + "..."
            if len(cache_key) > CACHE_KEY_MAX_DISPLAY_LENGTH
            else cache_key,
        )
        cached = None
        redis_available = True
        if redis_client is not None:
            logger.info("[OSRM] Checking Redis cache...")
            try:
                raw = redis_client.get(f"osrm:table:{cache_key}")
                logger.info("[OSRM] Redis get completed: raw=%s", "found" if raw else "not found")
                if raw:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8", errors="ignore")
                    cached = json.loads(raw)
                    logger.info("[OSRM] cache_hit block=%d len=%d", min(src_block), len(src_block))
                    # ✅ A5: Track cache hit
                    increment_cache_hit(cache_type="table")
            except _RedisConnError:
                # Redis HS -> on continue sans cache
                redis_available = False
                logger.warning("[OSRM] Redis connection failed - continuing without cache")
                # ✅ A5: Track bypass
                increment_cache_bypass()
            except Exception:
                logger.warning("[OSRM] Redis get failed", exc_info=True)
                increment_cache_bypass()

        logger.info("[OSRM] After Redis check: cached=%s", "present" if cached else "None")
        if cached and "durations" in cached:
            # ✅ A5: Track cache hit (déjà fait plus haut)
            durs = cached["durations"]
            for local_i, src_idx in enumerate(src_block):
                row = durs[local_i]
                for j in range(n):
                    v = row[j]
                    M[src_idx][j] = 999999.0 if (v is None or not math.isfinite(v)) else float(v)
            continue

        # ✅ A5: Track cache miss si pas de cache
        logger.info("[OSRM] No cache found, incrementing cache miss counter...")
        try:
            increment_cache_miss(cache_key, cache_type="table")
            logger.info("[OSRM] Cache miss counter incremented successfully")
        except Exception as e:
            logger.exception("[OSRM] ERROR in increment_cache_miss: %s (type=%s)", str(e), type(e).__name__)
            # Continue même en cas d'erreur de métriques
        logger.info("[OSRM] Preparing HTTP request to OSRM...")

        def _do_request():
            logger.info("[OSRM] _do_request() called, preparing retry logic...")

            # ✅ 2.3: Utiliser retry uniformisé
            def _fetch_table_data():
                logger.info(
                    "[OSRM] 🔵 Requesting table: sources=%d destinations=%d timeout=%ds base_url=%s",
                    len(src_block),
                    len(all_dests),
                    timeout,
                    base_url,
                )
                request_start = time.time()
                _rate_limit(rate_limit_per_sec)
                try:
                    data = _table(
                        base_url=base_url,
                        profile=profile,
                        coords=coords,
                        sources=list(src_block),
                        destinations=all_dests,
                        timeout=timeout,
                    )
                    request_duration_ms = int((time.time() - request_start) * 1000)
                    logger.info(
                        "[OSRM] ✅ Table request successful: duration_ms=%d sources=%d destinations=%d",
                        request_duration_ms,
                        len(src_block),
                        len(all_dests),
                    )
                    if request_duration_ms > timeout * 1000 * 0.8:  # >80% du timeout
                        logger.warning(
                            "[OSRM] ⚠️ Request took %d ms (close to timeout %ds) - consider increasing timeout",
                            request_duration_ms,
                            timeout,
                        )
                except Exception as req_e:
                    request_duration_ms = int((time.time() - request_start) * 1000)
                    logger.exception(
                        "[OSRM] ❌ Table request FAILED after %d ms: %s (type=%s)",
                        request_duration_ms,
                        str(req_e),
                        type(req_e).__name__,
                    )
                    raise
                durs = data.get("durations")
                if not durs:
                    msg = "OSRM /table returned no durations"
                    raise RuntimeError(msg)
                if len(durs) != len(src_block):
                    msg = "OSRM durations shape mismatch"
                    raise RuntimeError(msg)
                return data

            return retry_with_backoff(
                _fetch_table_data,
                max_retries=max_retries,
                base_delay_ms=backoff_ms,
                max_delay_ms=5000,  # 5s max
                use_jitter=True,
                logger_instance=logger,
            )

        # Déduplication in-flight sur la même clé
        start = time.time()
        try:
            data_any: Any = _singleflight_do(cache_key, _do_request)
            if not isinstance(data_any, dict):
                logger.warning("[OSRM] table_fetch returned non-dict -> fallback")
                return _fallback_matrix(coords)
            data: Dict[str, Any] = cast("Dict[str, Any]", data_any)
        except Exception as e:
            # 🚨 Fallback si toutes les tentatives ont échoué
            logger.warning("[OSRM] All attempts failed, using haversine fallback: %s", e)

            # ✅ D3: Enregistrer l'utilisation du fallback haversine
            try:
                from chaos.metrics import get_chaos_metrics

                metrics = get_chaos_metrics()
                # Essayer le fallback et enregistrer le résultat
                fallback_result = _fallback_matrix(coords)
                metrics.record_fallback("haversine", success=True)
                return fallback_result
            except ImportError:
                # Module chaos non disponible, continuer normalement
                return _fallback_matrix(coords)
        finally:
            dur_ms = int((time.time() - start) * 1000)
            logger.info(
                "[OSRM] table_fetch block_start=%d size=%d duration_ms=%d", min(src_block), len(src_block), dur_ms
            )

        # Écrit dans le cache
        try:
            if redis_client is not None and redis_available:
                redis_client.setex(f"osrm:table:{cache_key}", CACHE_TTL_SECONDS, json.dumps(data))
                logger.debug("OSRM cache SET key=%s ttl=%ss", cache_key, CACHE_TTL_SECONDS)
        except _RedisConnError:
            # Redis HS -> log mais on continue
            redis_available = False
            logger.warning("[OSRM] Redis connection failed when writing to cache - continuing without cache")
        except Exception:
            logger.warning("[OSRM] Redis setex failed", exc_info=True)
            increment_cache_bypass()

        durs = data.get("durations")
        if not isinstance(durs, list):
            logger.warning("[OSRM] durations missing/invalid -> fallback matrix")
            return _fallback_matrix(coords)
        for local_i, src_idx in enumerate(src_block):
            row = durs[local_i]
            for j in range(n):
                v = row[j]
                M[src_idx][j] = 999999.0 if (v is None or not math.isfinite(v)) else float(v)

    for i in range(n):
        M[i][i] = 0.0
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
    timeout: int = 15,  # ✅ Augmenté pour routes longues multi-points
    redis_client: Any | None = None,
    coord_precision: int = 5,
    overview: str = "false",
    geometries: str = "geojson",
    steps: bool = False,
    annotations: bool = False,
    avg_speed_kmh_fallback: float = 25.0,
    cache_ttl_s: int | None = None,
) -> Dict[str, Any]:
    """Retourne un dict: {"duration": sec, "distance": m, "geometry": ..., "legs": [...]}
    Fallback: haversine + vitesse moyenne.
    """
    key = _canonical_key_route(origin, destination, waypoints, coord_precision=coord_precision, profile=profile)
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
                    return cached
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
        r0 = data["routes"][0]
        return {
            "duration": float(r0.get("duration", 0.0)),
            "distance": float(r0.get("distance", 0.0)),
            "geometry": r0.get("geometry"),
            "legs": r0.get("legs", []),
        }

    try:
        # ⚡ Timeout d'attente adaptatif : max_wait = timeout OSRM + 2s de marge
        max_wait = float(timeout + 2) if timeout else 12.0
        res_any: Any = _singleflight_do(cache_key, _do, max_wait_seconds=max_wait)
        if not isinstance(res_any, dict):
            msg = "OSRM /route returned non-dict"
            raise RuntimeError(msg)
        res: Dict[str, Any] = cast("Dict[str, Any]", res_any)
    except Exception as e:
        # Fallback ETA/distance
        logger.warning("[OSRM] route failed -> fallback haversine: %s", e)
        pts: List[Tuple[float, float]] = [origin] + (waypoints or []) + [destination]
        dist_m = 0.0
        for a, b in itertools.pairwise(pts):
            dist_m += _haversine_km(a, b) * 1000.0
        sec = (dist_m / 1000.0) / max(avg_speed_kmh_fallback, 1e-3) * 3600.0
        res = {
            "duration": float(sec),
            "distance": float(dist_m),
            "geometry": {"type": "LineString", "coordinates": [[lon, lat] for (lat, lon) in pts]},
            "legs": [{"duration": float(sec), "distance": float(dist_m)}],
        }

    # Cache set
    try:
        if redis_client is not None:
            ttl = CACHE_TTL_SECONDS if cache_ttl_s is None else max(int(cache_ttl_s), 0)
            if ttl > 0:
                redis_client.setex(cache_key, ttl, json.dumps(res))
                logger.debug("OSRM cache SET key=%s ttl=%ss", cache_key, ttl)
    except _RedisConnError:
        logger.warning("[OSRM] Redis connection failed when writing to cache - continuing without cache")
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
    avg_speed_kmh_fallback: float = 25.0,
) -> int:
    """Calcule un ETA (secondes) robuste via OSRM /route, avec cache + fallback haversine."""
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
        return _fallback_eta_seconds(origin, destination, avg_kmh=avg_speed_kmh_fallback)


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
                            "[CircuitBreaker] OPEN -> HALF_OPEN (timeout expired: %.1fs >= %ds, allowing test request)",
                            time_since_last_failure,
                            self.timeout_duration,
                        )
                        self.state = "HALF_OPEN"
                        # Réinitialiser le compteur pour permettre un test
                        self.failure_count = 0
                        # ⚡ Continuer pour tenter l'appel en HALF_OPEN
                    else:
                        remaining = self.timeout_duration - time_since_last_failure
                        msg = f"CircuitBreaker OPEN (failures: {self.failure_count}, remaining: {remaining:.1f}s/{self.timeout_duration}s)"
                        logger.warning("[CircuitBreaker] %s", msg)
                        raise Exception(msg)
                else:
                    # Pas de last_failure_time mais état OPEN -> passer en HALF_OPEN pour test
                    logger.info("[CircuitBreaker] OPEN -> HALF_OPEN (no last_failure_time, resetting)")
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
                            "[CircuitBreaker] %s -> OPEN (failures: %d >= threshold: %d, last_error: %s)",
                            old_state,
                            self.failure_count,
                            self.failure_threshold,
                            str(e)[:100],
                        )
                    self.state = "OPEN"
                elif self.state == "HALF_OPEN":
                    # En HALF_OPEN, un seul échec remet en OPEN
                    logger.warning("[CircuitBreaker] HALF_OPEN -> OPEN (test failed, error: %s)", str(e)[:100])
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
    from services.unified_dispatch.dispatch_prometheus_metrics import record_circuit_breaker_state

    PROMETHEUS_METRICS_AVAILABLE = True
except ImportError:
    PROMETHEUS_METRICS_AVAILABLE = False

    def record_circuit_breaker_state(*args, **kwargs):
        pass


def build_distance_matrix_osrm_with_cb(coords: List[Tuple[float, float]], **kwargs) -> List[List[float]]:
    """Wrapper de build_distance_matrix_osrm avec circuit-breaker.
    En cas de circuit ouvert, fallback immédiat vers haversine.
    """
    n = len(coords)
    base_url = kwargs.get("base_url", "http://osrm:5000")
    timeout = kwargs.get("timeout")
    logger.info("[OSRM] build_distance_matrix_osrm_with_cb entry: n=%d base_url=%s timeout=%s", n, base_url, timeout)
    try:
        logger.debug("[OSRM] Calling circuit breaker for build_distance_matrix_osrm")
        result = _osrm_circuit_breaker.call(build_distance_matrix_osrm, coords, **kwargs)
        logger.info(
            "[OSRM] build_distance_matrix_osrm_with_cb success: shape=%dx%d",
            len(result),
            len(result[0]) if result else 0,
        )
        return result
    except Exception as e:
        logger.warning(
            "[OSRM] Circuit-breaker triggered or call failed: %s (type=%s), using haversine fallback",
            str(e),
            type(e).__name__,
            exc_info=True,
        )
        avg_kmh = kwargs.get("avg_speed_kmh_fallback", 25.0)
        return _fallback_matrix(coords, avg_kmh=avg_kmh)


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
    base_url est requis par les appels existants du module (utiliser OSRM_BASE_URL sinon).
    """
    # Résolution de l'URL de base
    osrm_base = base_url or os.getenv("OSRM_BASE_URL", "http://localhost:5000")
    info = route_info(origin, dest, base_url=osrm_base, profile=profile, redis_client=redis_client)
    return {"distance": float(info.get("distance", 0.0)), "duration": float(info.get("duration", 0.0))}


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

    # Si origins/destinations sont des sous-ensembles/ordres différents, on extrait la sous-matrice correspondante
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
    origin_hash = hashlib.md5(f"{origin[ORIG_ZERO]},{origin[1]}".encode(), usedforsecurity=False).hexdigest()[:8]
    dest_hash = hashlib.md5(f"{dest[0]},{dest[1]}".encode(), usedforsecurity=False).hexdigest()[:8]
    cache_key = f"osrm:cache:{date_str}:{origin_hash}:{dest_hash}"

    try:
        from ext import redis_client as rc

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
    origins_str = ",".join([f"{o[0]},{o[1]}" for o in origins])
    dests_str = ",".join([f"{d[0]},{d[1]}" for d in destinations])
    matrix_hash = hashlib.md5(f"{origins_str}|{dests_str}".encode(), usedforsecurity=False).hexdigest()[:16]
    cache_key = f"osrm:matrix:{date_str}:{matrix_hash}"

    try:
        from ext import redis_client as rc

        raw_any = rc.get(cache_key)
        if raw_any:
            logger.info("[OSRM] Matrix cache hit for %sx%s points", len(origins), len(destinations))
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

        rc.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result))
        logger.info("[OSRM] Matrix cached for %sx%s points", len(origins), len(destinations))
    except Exception as e:
        logger.warning("[OSRM] Matrix cache write error: %s", e)

    return result
