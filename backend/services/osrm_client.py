from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, Dict, List, Tuple, cast

import requests

logger = logging.getLogger(__name__)

# ============================================================
# Configuration timeout et retry
# ============================================================
DEFAULT_TIMEOUT = int(os.getenv("UD_OSRM_TIMEOUT", "30"))
DEFAULT_RETRY_COUNT = int(os.getenv("UD_OSRM_RETRY", "2"))
CACHE_TTL_SECONDS = int(os.getenv("UD_OSRM_CACHE_TTL", "3600"))  # 1h par défaut

# ============================================================
# Optional Redis import (safe) + alias d'exception
# ============================================================
try:
    # Import runtime; on évite l’attribut '.exceptions' que Pylance ne connaît pas toujours
    from redis.exceptions import ConnectionError as _RedisConnError  # type: ignore
except Exception:  # redis absent ou API inattendue
    class _RedisConnError(Exception):
        pass

# ------------------------------------------------------------
# In-flight de-dup (singleflight) process-local
# ------------------------------------------------------------
_inflight_lock = threading.Lock()
_inflight: Dict[str, Dict[str, Any]] = {}

def _singleflight_do(key: str, fn: Callable[[], Any]) -> Any:
    """
    Regroupe les appels concurrents sur la même clé.
    Le premier exécute fn(); les autres attendent le résultat.
    """
    with _inflight_lock:
        entry = _inflight.get(key)
        if entry is None:
            entry = {"evt": threading.Event(), "result": None, "error": None, "leader": True}
            _inflight[key] = entry
        else:
            entry["leader"] = False
    if entry["leader"]:
        try:
            res = fn()
            entry["result"] = res
        except Exception as e:
            entry["error"] = e
        finally:
            entry["evt"].set()
            with _inflight_lock:
                _inflight.pop(key, None)
    else:
        entry["evt"].wait()
        if entry["error"]:
            raise entry["error"]
    return entry["result"]

# ============================================================
# Fallback / Helpers (rate-limit, haversine, chunking)
# ============================================================

# --- simple per-process rate limiter ---
_rl_lock = threading.Lock()
_rl_last_ts = 0.0

def _rate_limit(rate_per_sec: float) -> None:
    """Sleep just enough to respect a per-process rate (req/sec)."""
    if rate_per_sec is None or rate_per_sec <= 0:
        return
    global _rl_last_ts
    with _rl_lock:
        now = time.time()
        min_interval = 1.0 / float(rate_per_sec)
        wait = _rl_last_ts + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _rl_last_ts = time.time()

def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in km for (lat, lon) pairs."""
    R = 6371.0
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def _fallback_matrix(coords: List[Tuple[float, float]], avg_kmh: float = 25.0) -> List[List[float]]:
    """
    Fallback durations matrix (seconds) using haversine distance and an average speed.
    Symmetric, diagonal 0.0.
    """
    n = len(coords)
    if n <= 1:
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
    """
    Appel OSRM table avec retry automatique sur timeout.

    Args:
        timeout: Timeout en secondes (défaut: env UD_OSRM_TIMEOUT ou 30s)
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    retry_count = DEFAULT_RETRY_COUNT
    last_error = None

    for attempt in range(retry_count):
        try:
            return _table_single_request(base_url, profile, coords, sources, destinations, timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            logger.warning(f"OSRM timeout/connection error (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))  # Backoff: 0.5s, 1s

    raise last_error or RuntimeError("OSRM request failed after retries")

def _table_single_request(base_url, profile, coords, sources, destinations, timeout):
    """Exécute une seule requête OSRM table (appelé par _table avec retry)."""
    # 6 décimales pour OSRM; la clé de cache utilisera son propre arrondi.
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords)
    url = f"{base_url}/table/v1/{profile}/{coord_str}"
    params = {"annotations": "duration"}
    if sources is not None:
        params["sources"] = ";".join(map(str, sources))
    if destinations is not None:
        params["destinations"] = ";".join(map(str, destinations))
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data: Any = r.json()
    return cast(Dict[str, Any], data)

def _route(
    base_url: str,
    profile: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    waypoints: List[Tuple[float, float]] | None = None,
    overview: str = "false",   # "false" | "simplified" | "full"
    geometries: str = "geojson",
    steps: bool = False,
    annotations: bool = False,
    timeout: int | None = None,
) -> Dict[str, Any]:
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    pts: List[Tuple[float, float]] = [origin]
    if waypoints:
        pts.extend(waypoints)
    pts.append(destination)
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in pts)
    url = f"{base_url}/route/v1/{profile}/{coord_str}"
    params = {
        "overview": overview,
        "geometries": geometries,
        "steps": "true" if steps else "false",
        "annotations": "true" if annotations else "false",
        # "continue_straight": "true"  # optionnel
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data: Any = r.json()
    return cast(Dict[str, Any], data)

# ============================================================
# Cache keys (stable, coord_precision ~ 1m)
# ============================================================

def _canonical_key_table(coords: List[Tuple[float, float]],
                         sources: List[int] | None,
                         destinations: List[int] | None,
                         *, coord_precision: int = 5) -> str:
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
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _canonical_key_route(origin: Tuple[float, float],
                         destination: Tuple[float, float],
                         waypoints: List[Tuple[float, float]] | None = None,
                         *, coord_precision: int = 5, profile: str = "driving") -> str:
    def _round(t):
        lat, lon = t
        return (round(lat, coord_precision), round(lon, coord_precision))
    pts = [_round(origin)] + ([_round(w) for w in waypoints] if waypoints else []) + [_round(destination)]
    payload = {"profile": profile, "pts": pts}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# ============================================================
# PUBLIC: Matrix (utilisé par data.build_time_matrix)
# ============================================================

def build_distance_matrix_osrm(
    coords: List[Tuple[float, float]],
    *,
    base_url: str,
    profile: str = "driving",
    timeout: int = 30,  # ✅ PERF: Augmenté à 30s pour matrices volumineuses
    max_sources_per_call: int = 60,
    rate_limit_per_sec: int = 8,
    max_retries: int = 2,
    backoff_ms: int = 250,
    # Cache/mémo optionnel
    redis_client: Any | None = None,
    cache_ttl_s: int = 900,
    coord_precision: int = 5,
) -> List[List[float]]:
    """
    Retourne une matrice de durées en SECONDES (float), shape NxN, diagonale = 0.0.
    Fallback haversine en cas d'échec.
    """
    n = len(coords)
    if n <= 1:
        return [[0.0] * n for _ in range(n)]

    M = [[0.0] * n for _ in range(n)]
    all_dests = list(range(n))

    # ✅ PERF: Chunking adaptatif - petits chunks pour grandes matrices
    adaptive_chunk_size = 40 if n > 100 else max_sources_per_call

    for src_block in _chunks(range(n), max(1, int(adaptive_chunk_size))):
        # --- Cache key pour ce sous-bloc ---
        cache_key = _canonical_key_table(coords, list(src_block), all_dests, coord_precision=coord_precision)
        cached = None
        redis_available = True
        if redis_client is not None:
            try:
                raw = redis_client.get(f"osrm:table:{cache_key}")
                if raw:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8", errors="ignore")
                    cached = json.loads(raw)
                    logger.info("[OSRM] cache_hit block=%d len=%d", min(src_block), len(src_block))
            except _RedisConnError:
                # Redis HS -> on continue sans cache
                redis_available = False
                logger.warning("[OSRM] Redis connection failed - continuing without cache")
            except Exception:
                logger.warning("[OSRM] Redis get failed", exc_info=True)
        if cached and "durations" in cached:
            durs = cached["durations"]
            for local_i, src_idx in enumerate(src_block):
                row = durs[local_i]
                for j in range(n):
                    v = row[j]
                    M[src_idx][j] = 999999.0 if (v is None or not math.isfinite(v)) else float(v)
            continue

        def _do_request():
            for attempt in range(max_retries + 1):
                try:
                    _rate_limit(rate_limit_per_sec)
                    data = _table(
                        base_url=base_url,
                        profile=profile,
                        coords=coords,
                        sources=list(src_block),
                        destinations=all_dests,
                        timeout=timeout,
                    )
                    durs = data.get("durations")
                    if not durs:
                        raise RuntimeError("OSRM /table returned no durations")
                    if len(durs) != len(src_block):
                        raise RuntimeError("OSRM durations shape mismatch")
                    return data
                except Exception as e:
                    if attempt >= max_retries:
                        logger.warning("[OSRM] block failed permanently: %s", e)
                        raise
                    sleep = (backoff_ms / 1000.0) * (2 ** attempt)
                    logger.info("[OSRM] retry in %.3fs after %s", sleep, e)
                    time.sleep(sleep)

        # Déduplication in-flight sur la même clé
        start = time.time()
        try:
            data_any: Any = _singleflight_do(cache_key, _do_request)
            if not isinstance(data_any, dict):
                logger.warning("[OSRM] table_fetch returned non-dict -> fallback")
                return _fallback_matrix(coords)
            data: Dict[str, Any] = cast(Dict[str, Any], data_any)
        except Exception as e:
            # 🚨 Fallback si toutes les tentatives ont échoué
            logger.warning("[OSRM] All attempts failed, using haversine fallback: %s", e)
            return _fallback_matrix(coords)
        finally:
            dur_ms = int((time.time() - start) * 1000)
            logger.info("[OSRM] table_fetch block_start=%d size=%d duration_ms=%d", min(src_block), len(src_block), dur_ms)

        # Écrit dans le cache
        try:
            if redis_client is not None and redis_available:
                redis_client.setex(f"osrm:table:{cache_key}", CACHE_TTL_SECONDS, json.dumps(data))
                logger.debug(f"OSRM cache SET key={cache_key} ttl={CACHE_TTL_SECONDS}s")
        except _RedisConnError:
            # Redis HS -> log mais on continue
            redis_available = False
            logger.warning("[OSRM] Redis connection failed when writing to cache - continuing without cache")
        except Exception:
            logger.warning("[OSRM] Redis setex failed", exc_info=True)

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
    timeout: int = 5,
    redis_client: Any | None = None,
    cache_ttl_s: int = 900,
    coord_precision: int = 5,
    overview: str = "false",
    geometries: str = "geojson",
    steps: bool = False,
    annotations: bool = False,
    avg_speed_kmh_fallback: float = 25.0,
) -> Dict[str, Any]:
    """
    Retourne un dict: {"duration": sec, "distance": m, "geometry": ..., "legs": [...]}
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
                    return cached
        except _RedisConnError:
            logger.warning("[OSRM] Redis connection failed - continuing without cache")
        except Exception:
            logger.warning("[OSRM] Redis get failed (route)", exc_info=True)

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
            raise RuntimeError(f"OSRM /route bad response: {data.get('message')}")
        r0 = data["routes"][0]
        return {
            "duration": float(r0.get("duration", 0.0)),
            "distance": float(r0.get("distance", 0.0)),
            "geometry": r0.get("geometry"),
            "legs": r0.get("legs", []),
        }

    try:
        res_any: Any = _singleflight_do(cache_key, _do)
        if not isinstance(res_any, dict):
            raise RuntimeError("OSRM /route returned non-dict")
        res: Dict[str, Any] = cast(Dict[str, Any], res_any)
    except Exception as e:
        # Fallback ETA/distance
        logger.warning("[OSRM] route failed -> fallback haversine: %s", e)
        pts: List[Tuple[float, float]] = [origin] + (waypoints or []) + [destination]
        dist_m = 0.0
        for a, b in zip(pts[:-1], pts[1:], strict=False):
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
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(res))
            logger.debug(f"OSRM cache SET key={cache_key} ttl={CACHE_TTL_SECONDS}s")
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
    timeout: int = 5,
    redis_client: Any | None = None,
    cache_ttl_s: int = 900,
    coord_precision: int = 5,
    avg_speed_kmh_fallback: float = 25.0,
) -> int:
    """
    Calcule un ETA (secondes) robuste via OSRM /route, avec cache + fallback haversine.
    """
    info = route_info(
        origin, destination,
        base_url=base_url,
        profile=profile,
        waypoints=waypoints,
        timeout=timeout,
        redis_client=redis_client,
        cache_ttl_s=cache_ttl_s,
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
    """
    Circuit-breaker pour protéger OSRM des surcharges.
    États : CLOSED (normal) -> OPEN (échecs) -> HALF_OPEN (test) -> CLOSED
    """
    def __init__(self, failure_threshold: int = 5, timeout_duration: int = 60):
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
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout_duration:
                    logger.info("[CircuitBreaker] OPEN -> HALF_OPEN (timeout expired)")
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"CircuitBreaker OPEN (failures: {self.failure_count})")
        
        try:
            result = func(*args, **kwargs)
            
            # Succès -> reset
            with self._lock:
                if self.state == "HALF_OPEN":
                    logger.info("[CircuitBreaker] HALF_OPEN -> CLOSED (success)")
                    self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != "OPEN":
                        logger.warning(
                            "[CircuitBreaker] CLOSED -> OPEN (failures: %d >= threshold: %d)",
                            self.failure_count, self.failure_threshold
                        )
                    self.state = "OPEN"
            raise


# Instance globale du circuit-breaker OSRM
_osrm_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_duration=60)


def build_distance_matrix_osrm_with_cb(
    coords: List[Tuple[float, float]],
    **kwargs
) -> List[List[float]]:
    """
    Wrapper de build_distance_matrix_osrm avec circuit-breaker.
    En cas de circuit ouvert, fallback immédiat vers haversine.
    """
    try:
        return _osrm_circuit_breaker.call(
            build_distance_matrix_osrm,
            coords,
            **kwargs
        )
    except Exception as e:
        logger.warning(
            "[OSRM] Circuit-breaker triggered or call failed: %s, using haversine fallback",
            e
        )
        avg_kmh = kwargs.get('avg_speed_kmh_fallback', 25.0)
        return _fallback_matrix(coords, avg_kmh=avg_kmh)
