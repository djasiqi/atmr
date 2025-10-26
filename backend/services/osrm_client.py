from __future__ import annotations

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

from shared.geo_utils import haversine_tuple as _haversine_km

# Constantes pour éviter les valeurs magiques
RATE_PER_SEC_ZERO = 0
WAIT_ZERO = 0
N_ONE = 1
N_THRESHOLD = 150
N_PERCENT = 100
ORIG_ZERO = 0

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

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

def _singleflight_do(key: str, fn: Callable[[], Any]) -> Any:
    """Regroupe les appels concurrents sur la même clé.
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

    retry_count = DEFAULT_RETRY_COUNT
    last_error = None

    for attempt in range(retry_count):
        try:
            return _table_single_request(base_url, profile, coords, sources, destinations, timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            logger.warning("OSRM timeout/connection error (attempt %s/%s): %s", attempt+1, retry_count, e)
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))  # Backoff: 0.5s, 1s

    raise last_error or RuntimeError("OSRM request failed after retries")

def _table_single_request(base_url, profile, coords, sources, destinations, timeout):
    """Exécute une seule requête OSRM table (appelé par _table avec retry)."""
    # 6 décimales pour OSRM; la clé de cache utilisera son propre arrondi.
    coord_str = ";".join(f"{lon},{lat}" for (lat, lon) in coords)
    url = f"{base_url}/table/v1/{profile}/{coord_str}"
    params = {"annotations": "duration"}
    if sources is not None:
        params["sources"] = ";".join(map(str, sources))
    if destinations is not None:
        params["destinations"] = ";".join(map(str, destinations))
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data: Any = r.json()
    return cast("Dict[str, Any]", data)

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
    coord_str = ";".join(f"{lon},{lat}" for (lat, lon) in pts)
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
    return cast("Dict[str, Any]", data)

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
    # ✅ Timeout adaptatif basé sur nombre de coordonnées
    if timeout is None:
        n = len(coords)
        if n > N_THRESHOLD:
            timeout = 60
        elif n > N_PERCENT:
            timeout = 45
        elif n > N_THRESHOLD:
            timeout = 30
        else:
            timeout = 15

    n = len(coords)
    if n <= N_ONE:
        return [[0.0] * n for _ in range(n)]

    M = [[0.0] * n for _ in range(n)]
    all_dests = list(range(n))

    # ✅ PERF: Chunking adaptatif - petits chunks pour grandes matrices
    adaptive_chunk_size = 40 if n > N_PERCENT else max_sources_per_call

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
                        msg = "OSRM /table returned no durations"
                        raise RuntimeError(msg)
                    if len(durs) != len(src_block):
                        msg = "OSRM durations shape mismatch"
                        raise RuntimeError(msg)
                    return data
                except Exception as e:
                    if attempt >= max_retries:
                        logger.warning("[OSRM] block failed permanently: %s", e)
                        raise
                    sleep = (backoff_ms / 1000.0) * (2 ** attempt)
                    logger.info("[OSRM] retry in %.3fs after %s", sleep, e)
                    time.sleep(sleep)
            return None

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
            return _fallback_matrix(coords)
        finally:
            dur_ms = int((time.time() - start) * 1000)
            logger.info("[OSRM] table_fetch block_start=%d size=%d duration_ms=%d", min(src_block), len(src_block), dur_ms)

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
        res_any: Any = _singleflight_do(cache_key, _do)
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
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(res))
            logger.debug("OSRM cache SET key=%s ttl=%ss", cache_key, CACHE_TTL_SECONDS)
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
        origin, destination,
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
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout_duration:
                    logger.info("[CircuitBreaker] OPEN -> HALF_OPEN (timeout expired)")
                    self.state = "HALF_OPEN"
                else:
                    msg = f"CircuitBreaker OPEN (failures: {self.failure_count})"
                    raise Exception(msg)

        try:
            result = func(*args, **kwargs)

            # Succès -> reset
            with self._lock:
                if self.state == "HALF_OPEN":
                    logger.info("[CircuitBreaker] HALF_OPEN -> CLOSED (success)")
                    self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception:
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
    """Wrapper de build_distance_matrix_osrm avec circuit-breaker.
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
        avg_kmh = kwargs.get("avg_speed_kmh_fallback", 25.0)
        return _fallback_matrix(coords, avg_kmh=avg_kmh)
# ============================================================
# Helpers de haut niveau pour distance/temps et matrices
# ============================================================

def get_distance_time(origin: Tuple[float, float], dest: Tuple[float, float], *, base_url: str | None = None,
                      profile: str = "driving", redis_client: Any | None = None) -> Dict[str, float]:
    """Retourne un dict {"distance": m, "duration": s} en utilisant route_info.
    base_url est requis par les appels existants du module (utiliser OSRM_BASE_URL sinon).
    """
    # Résolution de l'URL de base
    osrm_base = base_url or os.getenv("OSRM_BASE_URL", "http://localhost:5000")
    info = route_info(origin, dest, base_url=osrm_base, profile=profile, redis_client=redis_client)
    return {"distance": float(info.get("distance", 0.0)), "duration": float(info.get("duration", 0.0))}


def get_matrix(origins: List[Tuple[float, float]], destinations: List[Tuple[float, float]], *,
               base_url: str | None = None, profile: str = "driving",
               redis_client: Any | None = None) -> Dict[str, Any]:
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
    origin_hash = hashlib.md5(f"{origin[ORIG_ZERO]},{origin[1]}".encode()).hexdigest()[:8]
    dest_hash = hashlib.md5(f"{dest[0]},{dest[1]}".encode()).hexdigest()[:8]
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
    matrix_hash = hashlib.md5(f"{origins_str}|{dests_str}".encode()).hexdigest()[:16]
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
