# services/maps.py
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, cast

import requests
from cachetools import LRUCache

from ext import app_logger
from shared.geo_utils import haversine_tuple as _haversine_km
from shared.retry import retry_http_request  # ✅ 2.3: Retry uniformisé

# Constantes pour éviter les valeurs magiques
RATE_ZERO = 0
WAIT_ZERO = 0
OU_ZERO = 0
TTL_VAL_ZERO = 0
DUR_ZERO = 0
COORD_PAIR_LENGTH = 2

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
_GOOGLE_TIMEOUT = 10  # s
_AVG_SPEED_KMH = 40.0  # fallback
MAPS_CIRCUIT_BREAKER_FAILURE_THRESHOLD = max(
    1, int(os.getenv("MAPS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
)
MAPS_CIRCUIT_BREAKER_OPEN_SECONDS = max(
    1, int(os.getenv("MAPS_CIRCUIT_BREAKER_OPEN_SECONDS", "60"))
)


class _InMemoryCircuitBreaker:
    """Circuit breaker local au process pour API Google Maps."""

    def __init__(self, *, failure_threshold: int, open_seconds: int):  # pyright: ignore[reportMissingSuperCall]
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._lock = threading.Lock()
        self._opened_until_ts = 0.0
        self._consecutive_failures = 0
        self._half_open_probe_in_flight = False

    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            if self._opened_until_ts > now:
                return False
            if self._opened_until_ts > 0:
                if self._half_open_probe_in_flight:
                    return False
                self._half_open_probe_in_flight = True
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._opened_until_ts = 0.0
            self._consecutive_failures = 0
            self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._half_open_probe_in_flight = False
            if self._consecutive_failures >= self._failure_threshold:
                self._opened_until_ts = time.time() + self._open_seconds
                self._consecutive_failures = 0
                app_logger.warning(
                    "[Google Maps] Circuit breaker OPEN for %ss",
                    self._open_seconds,
                )


_google_maps_circuit_breaker = _InMemoryCircuitBreaker(
    failure_threshold=MAPS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    open_seconds=MAPS_CIRCUIT_BREAKER_OPEN_SECONDS,
)


def _as_origin_str(addr_or_coord: Any) -> str:
    """Accepte une adresse string OU un tuple (lat, lon)
    -> 'lat,lon' ou adresse telle quelle."""
    if isinstance(addr_or_coord, str):
        return addr_or_coord
    if (
        isinstance(addr_or_coord, (list, tuple))
        and len(addr_or_coord) == COORD_PAIR_LENGTH
    ):
        return f"{float(addr_or_coord[0])},{float(addr_or_coord[1])}"
    return str(addr_or_coord)


def get_distance_duration(
    pickup_address: Any,
    dropoff_address: Any,
    *,
    departure_time: int | None = None,
    traffic_model: str = "best_guess",
    units: str = "metric",
    region: str = "CH",
    language: str = "fr",
    timeout: int = _GOOGLE_TIMEOUT,
) -> Tuple[int, int]:
    """Retourne (duration_seconds, distance_meters).
    Accepte adresse string OU tuple (lat, lon) en entrée.
    Fallback Haversine si la clé API manque ou si l'API renvoie une erreur.
    """
    pick_is_coord = (
        isinstance(pickup_address, (list, tuple))
        and len(pickup_address) == COORD_PAIR_LENGTH
    )
    drop_is_coord = (
        isinstance(dropoff_address, (list, tuple))
        and len(dropoff_address) == COORD_PAIR_LENGTH
    )

    if not GOOGLE_MAPS_API_KEY:
        if pick_is_coord and drop_is_coord:
            dist_km = _haversine_km(
                cast(Tuple[float, float], pickup_address),
                cast(Tuple[float, float], dropoff_address),
            )
            dur_s = int((dist_km / _AVG_SPEED_KMH) * 3600)
            return max(1, dur_s), int(dist_km * 1000)
        if isinstance(pickup_address, str) and isinstance(dropoff_address, str):
            return _distance_duration_from_geocoded_strings(
                pickup_address, dropoff_address, region=region
            )
        msg = "Clé API Google Maps manquante et entrées non géocodables."
        raise OSError(msg)

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params: Dict[str, str] = {
        "origins": _as_origin_str(pickup_address),
        "destinations": _as_origin_str(dropoff_address),
        "key": str(GOOGLE_MAPS_API_KEY),
        "language": language,
        "units": units,
        "region": region,
    }
    if departure_time is not None:
        params["departure_time"] = str(
            int(departure_time)
        )  # <- string pour calmer Pylance
        params["traffic_model"] = traffic_model

    try:
        if not _google_maps_circuit_breaker.allow_request():
            raise RuntimeError("google_maps_circuit_open")
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            raise RuntimeError(data.get("error_message") or data.get("status"))

        # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
        rows = data.get("rows", [])
        if not rows:
            raise RuntimeError("No rows in Google Maps response")
        elements = rows[0].get("elements", [])
        if not elements:
            raise RuntimeError("No elements in Google Maps response")
        elem = elements[0]
        if elem.get("status") != "OK":
            raise RuntimeError(elem.get("status"))

        dur_field = elem.get("duration_in_traffic") or elem.get("duration")
        if not dur_field or "value" not in dur_field:
            raise RuntimeError("Missing duration value in Google Maps response")
        duration_seconds = int(dur_field["value"])
        distance = elem.get("distance", {})
        if "value" not in distance:
            raise RuntimeError("Missing distance value in Google Maps response")
        distance_meters = int(distance["value"])
        _google_maps_circuit_breaker.record_success()
        return duration_seconds, distance_meters

    except Exception as e:
        if str(e) != "google_maps_circuit_open":
            _google_maps_circuit_breaker.record_failure()
        app_logger.warning("⚠️ DistanceMatrix single-pair fallback: %s", e)
        if pick_is_coord and drop_is_coord:
            dist_km = _haversine_km(
                cast(Tuple[float, float], pickup_address),
                cast(Tuple[float, float], dropoff_address),
            )
            dur_s = int((dist_km / _AVG_SPEED_KMH) * 3600)
            return max(1, dur_s), int(dist_km * 1000)
        if isinstance(pickup_address, str) and isinstance(dropoff_address, str):
            try:
                return _distance_duration_from_geocoded_strings(
                    pickup_address, dropoff_address, region=region
                )
            except Exception as e2:
                app_logger.warning(
                    "⚠️ Fallback géocode+Haversine après échec Distance Matrix: %s", e2
                )
        raise


def geocode_address(
    address: str, *, country: str | None = None, language: str = "fr"
) -> Dict[str, float] | None:
    """Géocode une adresse → {'lat': float, 'lon': float} | None.
    - country: code ISO (ex: "CH") pour biaiser la recherche
    - language: "fr" par défaut.

    ✅ P1: Utilise cache Redis/local pour améliorer performance.
    ✅ P1: Utilise retry avec backoff pour améliorer résilience.
    """
    if not GOOGLE_MAPS_API_KEY:
        app_logger.warning(
            "⚠️ Clé API Google Maps manquante, utilisation de Nominatim (OSM)."
        )
        return geocode_address_nominatim(address, country=country)

    address = (address or "").strip()
    if not address:
        msg = "Adresse vide ou invalide pour le géocodage."
        raise ValueError(msg)

    # ✅ P1: Normaliser l'adresse pour améliorer cache hit rate
    address_normalized = _normalize_address_for_cache(address, country)
    cache_key = f"google:{address_normalized}:{language}"

    # ✅ P1: Vérifier cache local (L1 cache) - ultra-rapide
    with _GOOGLE_MAPS_LOCAL_CACHE_LOCK:
        cached_result = _GOOGLE_MAPS_LOCAL_CACHE.get(cache_key)
        if cached_result is not None:
            app_logger.debug(
                "[Google Maps] ✅ L1 cache hit (local LRU) for address: %s",
                address[:50],
            )
            return cached_result

    # ✅ P1: Vérifier cache Redis (L2 cache)
    redis_client = _get_redis_for_geocoding()
    if redis_client:
        try:
            cache_key_hash = hashlib.sha256(
                cache_key.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
            redis_cache_key = f"geocoding:google:{cache_key_hash}"
            cached_data = redis_client.get(redis_cache_key)
            if cached_data:
                try:
                    cached_str: str | None = None
                    if isinstance(cached_data, (bytes, bytearray)):
                        cached_str = cached_data.decode("utf-8", errors="ignore")
                    elif isinstance(cached_data, str):
                        cached_str = cached_data

                    if cached_str:
                        cached_result = json.loads(cached_str)
                        if (
                            isinstance(cached_result, dict)
                            and "lat" in cached_result
                            and "lon" in cached_result
                        ):
                            result = {
                                "lat": float(cached_result["lat"]),
                                "lon": float(cached_result["lon"]),
                            }
                            # Stocker dans cache local pour accès ultérieur
                            with _GOOGLE_MAPS_LOCAL_CACHE_LOCK:
                                _GOOGLE_MAPS_LOCAL_CACHE[cache_key] = result
                            app_logger.debug(
                                "[Google Maps] ✅ L2 cache hit (Redis) for address: %s",
                                address[:50],
                            )
                            return result
                        if cached_str == "null":
                            # Cache des résultats None pour éviter requêtes répétées
                            with _GOOGLE_MAPS_LOCAL_CACHE_LOCK:
                                _GOOGLE_MAPS_LOCAL_CACHE[cache_key] = None
                            return None
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as e:
                    app_logger.debug(
                        "[Google Maps] Failed to decode cached data: %s", e
                    )
        except Exception as e:
            app_logger.debug("[Google Maps] Redis cache check failed: %s", e)

    # ✅ P1: Géocoder avec retry et backoff
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params: Dict[str, str] = {
        "address": address,
        "key": str(GOOGLE_MAPS_API_KEY),
        "language": language,
    }
    if country:
        params["components"] = f"country:{country}"

    result: Dict[str, float] | None = None

    # ✅ P1: Utiliser retry avec backoff pour améliorer résilience
    def _geocode_request() -> Dict[str, float] | None:
        """Fonction interne pour requête Google Maps avec gestion d'erreurs."""
        try:
            if not _google_maps_circuit_breaker.allow_request():
                raise RuntimeError("google_maps_circuit_open")
            resp = requests.get(url, params=params, timeout=_GOOGLE_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "OK" or not data.get("results"):
                app_logger.warning(
                    "⚠️ Aucune coordonnée trouvée pour : '%s' (country=%s)",
                    address,
                    country,
                )
                return None

            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            results = data.get("results", [])
            if results:
                geometry = results[0].get("geometry", {})
                if geometry:
                    location = geometry.get("location", {})
                    if location:
                        lat = location.get("lat")
                        lng = location.get("lng")
                        if lat is not None and lng is not None:
                            _google_maps_circuit_breaker.record_success()
                            return {"lat": float(lat), "lon": float(lng)}
            return None
        except requests.Timeout as e:
            # ✅ P2: Gestion spécifique des timeouts pour meilleure observabilité
            app_logger.error(
                "⏱️ Timeout API Google Maps Geocoding pour '%s' (country=%s, timeout=%ds): %s",
                address,
                country,
                _GOOGLE_TIMEOUT,
                e,
            )
            raise
        except requests.RequestException as e:
            app_logger.error(
                "❌ Erreur API Google Maps pour '%s' (country=%s): %s",
                address,
                country,
                e,
            )
            raise
        except Exception:
            raise

    try:
        # Sprint 2 H13: limiter les retries Google Maps pour préserver la latence.
        result = retry_http_request(
            _geocode_request,
            max_retries=1,
            base_delay_ms=250,
        )
    except Exception as e:
        if str(e) != "google_maps_circuit_open":
            _google_maps_circuit_breaker.record_failure()
        app_logger.warning(
            "[Google Maps] Échec géocodage après retries pour '%s': %s", address[:50], e
        )
        result = None

    # ✅ P1: Mettre en cache le résultat (même si None pour éviter requêtes répétées)
    if redis_client:
        try:
            cache_value = json.dumps(result) if result else "null"
            cache_key_hash = hashlib.sha256(
                cache_key.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
            redis_cache_key = f"geocoding:google:{cache_key_hash}"
            redis_client.setex(redis_cache_key, _GOOGLE_MAPS_CACHE_TTL, cache_value)
            app_logger.debug(
                "[Google Maps] ✅ L2 cache write (Redis) for address: %s", address[:50]
            )
        except Exception as e:
            app_logger.debug("[Google Maps] Failed to write to Redis cache: %s", e)

    # ✅ P1: Mettre en cache local (L1 cache)
    with _GOOGLE_MAPS_LOCAL_CACHE_LOCK:
        _GOOGLE_MAPS_LOCAL_CACHE[cache_key] = result
        app_logger.debug(
            "[Google Maps] ✅ L1 cache write (local LRU) for address: %s", address[:50]
        )

    return result


def _distance_duration_from_geocoded_strings(
    pickup: str,
    dropoff: str,
    *,
    region: str = "CH",
) -> Tuple[int, int]:
    """Géocode les deux adresses puis estime durée / distance (Haversine, vitesse moyenne).

    Utilisé quand la Distance Matrix Google n'est pas utilisable (pas de clé, erreur API,
    réponse invalide). Réutilise ``geocode_address`` (Google ou Nominatim selon config).
    """
    pick = (pickup or "").strip()
    drop = (dropoff or "").strip()
    if not pick or not drop:
        raise ValueError("Adresse de départ ou d'arrivée vide")
    a = geocode_address(pick, country=region)
    b = geocode_address(drop, country=region)
    if not a or not b:
        raise RuntimeError("ZERO_RESULTS")
    coord_a = (float(a["lat"]), float(a["lon"]))
    coord_b = (float(b["lat"]), float(b["lon"]))
    dist_km = _haversine_km(coord_a, coord_b)
    dur_s = int((dist_km / _AVG_SPEED_KMH) * 3600)
    return max(1, dur_s), int(dist_km * 1000)


# ✅ P1: Cache pour géocodage Nominatim (TTL 24h)
_NOMINATIM_CACHE_TTL = int(os.getenv("NOMINATIM_CACHE_TTL", "86400"))  # 24h par défaut
_NOMINATIM_LOCAL_CACHE: LRUCache[str, Dict[str, float] | None] = LRUCache(maxsize=500)
_NOMINATIM_LOCAL_CACHE_LOCK = threading.Lock()

# ✅ P1: Cache pour géocodage Google Maps (TTL 7j)
_GOOGLE_MAPS_CACHE_TTL = int(
    os.getenv("GOOGLE_MAPS_CACHE_TTL", "604800")
)  # 7 jours par défaut
_GOOGLE_MAPS_LOCAL_CACHE: LRUCache[str, Dict[str, float] | None] = LRUCache(maxsize=500)
_GOOGLE_MAPS_LOCAL_CACHE_LOCK = threading.Lock()


def _get_redis_for_geocoding():
    """Récupère un client Redis pour le cache de géocodage."""
    try:
        from ext import redis_client as ext_redis_client

        if ext_redis_client is not None:
            ext_redis_client.ping()
            return ext_redis_client
    except Exception:
        pass

    # Fallback : essayer de créer depuis REDIS_URL
    try:
        redis_url = os.getenv("REDIS_URL", None)
        if redis_url:
            import redis

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


def _normalize_address_for_cache(address: str, country: str | None) -> str:
    """Normalise une adresse pour améliorer le taux de cache hit."""
    # Normaliser : minuscules, supprimer espaces multiples, trim
    normalized = " ".join((address or "").strip().lower().split())
    if country:
        normalized = f"{normalized}|{country.lower()}"
    return normalized


def geocode_address_nominatim(
    address: str, *, country: str | None = None
) -> Dict[str, float] | None:
    """Géocode une adresse avec Nominatim (OpenStreetMap)
    → {'lat': float, 'lon': float} | None.
    - country: code ISO (ex: "CH") pour biaiser la recherche.

    ✅ P1: Optimisations performance :
    - Cache Redis avec TTL 24h pour adresses géocodées
    - Cache local LRU (L1 cache) pour accès ultra-rapide
    - Normalisation des adresses pour améliorer cache hit
    """
    address = (address or "").strip()
    if not address:
        msg = "Adresse vide ou invalide pour le géocodage."
        raise ValueError(msg)

    # ✅ P1: Normaliser l'adresse pour améliorer le cache hit
    cache_key_normalized = _normalize_address_for_cache(address, country)
    cache_key_hash = hashlib.md5(
        cache_key_normalized.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    redis_cache_key = f"nominatim:geocode:{cache_key_hash}"

    # ✅ P1: Vérifier cache local LRU (L1 cache - ultra rapide)
    with _NOMINATIM_LOCAL_CACHE_LOCK:
        cached_result = _NOMINATIM_LOCAL_CACHE.get(cache_key_normalized)
        if cached_result is not None:
            app_logger.debug(
                "[Nominatim] ✅ L1 cache hit (local LRU) for address: %s", address[:50]
            )
            return cached_result

    # ✅ P1: Vérifier cache Redis (L2 cache)
    redis_client = _get_redis_for_geocoding()
    if redis_client:
        try:
            cached_raw = redis_client.get(redis_cache_key)
            if cached_raw:
                # ✅ P1: Gérer types bytes/str pour compatibilité Redis
                cached_str: str | None = None
                if isinstance(cached_raw, (bytes, bytearray)):
                    cached_str = cached_raw.decode("utf-8", errors="ignore")
                elif isinstance(cached_raw, str):
                    cached_str = cached_raw

                if cached_str:
                    try:
                        cached_data = json.loads(cached_str)
                        if (
                            isinstance(cached_data, dict)
                            and "lat" in cached_data
                            and "lon" in cached_data
                        ):
                            result = {
                                "lat": float(cached_data["lat"]),
                                "lon": float(cached_data["lon"]),
                            }
                            # ✅ P1: Mettre en cache local LRU (L1 cache)
                            with _NOMINATIM_LOCAL_CACHE_LOCK:
                                _NOMINATIM_LOCAL_CACHE[cache_key_normalized] = result
                            app_logger.debug(
                                "[Nominatim] ✅ L2 cache hit (Redis) for address: %s",
                                address[:50],
                            )
                            return result
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                        pass
        except Exception as e:
            app_logger.debug("[Nominatim] Redis get failed: %s", e)

    # Cache miss - faire la requête Nominatim
    url = "https://nominatim.openstreetmap.org/search"
    # Typer correctement params pour satisfaire mypy
    params: dict[str, str | int] = {
        "q": address,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }

    if country:
        params["countrycodes"] = country.lower()

    headers = {"User-Agent": "ATMR-Transport-App/1.0"}

    def _fetch_nominatim() -> requests.Response:
        """Fonction interne pour fetch Nominatim avec respect du rate limit."""
        # Nominatim a une limite de 1 requête par seconde
        time.sleep(1.1)
        return requests.get(url, params=params, headers=headers, timeout=10)

    result: Dict[str, float] | None = None
    try:
        # ✅ 2.3: Utiliser retry uniformisé (2 retries, 1s base) pour meilleure résilience
        resp = retry_http_request(
            _fetch_nominatim,
            max_retries=2,
            base_delay_ms=1000,  # 1 seconde comme spécifié dans le rapport
        )
        resp.raise_for_status()
        data = resp.json()

        if not data or len(data) == 0:
            app_logger.warning(
                "⚠️ Nominatim: Aucune coordonnée trouvée pour '%s'", address
            )
            result = None
        else:
            result_data = data[0]
            result = {
                "lat": float(result_data["lat"]),
                "lon": float(result_data["lon"]),
            }

    except requests.RequestException as e:
        app_logger.error(
            "❌ Erreur Nominatim pour '%s' (après retries): %s", address, e
        )
        result = None

    # ✅ P1: Mettre en cache le résultat (même si None pour éviter requêtes répétées)
    # Cache Redis (L2 cache) avec TTL 24h
    if redis_client:
        try:
            cache_value = json.dumps(result) if result else json.dumps(None)
            redis_client.setex(redis_cache_key, _NOMINATIM_CACHE_TTL, cache_value)
            app_logger.debug(
                "[Nominatim] ✅ L2 cache write (Redis) for address: %s", address[:50]
            )
        except Exception as e:
            app_logger.debug("[Nominatim] Redis setex failed: %s", e)

    # ✅ P1: Mettre en cache local LRU (L1 cache) pour accès ultra-rapide
    with _NOMINATIM_LOCAL_CACHE_LOCK:
        _NOMINATIM_LOCAL_CACHE[cache_key_normalized] = result
        app_logger.debug(
            "[Nominatim] ✅ L1 cache write (local LRU) for address: %s", address[:50]
        )

    return result


# ✅ P1: Parallélisation intelligente pour géocodage batch
_NOMINATIM_RATE_LIMITER_LOCK = threading.Lock()
_NOMINATIM_LAST_REQUEST_TIME = {"value": 0.0}
_NOMINATIM_MIN_INTERVAL = 1.1  # 1.1 secondes entre requêtes Nominatim


def _wait_for_nominatim_rate_limit():
    """Attend le respect du rate limit Nominatim (1 req/seconde)."""
    with _NOMINATIM_RATE_LIMITER_LOCK:
        now = time.time()
        elapsed = now - _NOMINATIM_LAST_REQUEST_TIME["value"]
        if elapsed < _NOMINATIM_MIN_INTERVAL:
            wait_time = _NOMINATIM_MIN_INTERVAL - elapsed
            time.sleep(wait_time)
        _NOMINATIM_LAST_REQUEST_TIME["value"] = time.time()


def geocode_addresses_batch(
    addresses: List[str],
    *,
    country: str | None = None,
    max_workers: int | None = None,
    prefer_google: bool = True,
) -> Dict[str, Dict[str, float] | None]:
    """Géocode plusieurs adresses en parallèle avec rate limiting intelligent.

    ✅ P1: Optimisation performance - Batch geocoding parallélisé
    - Parallélisation avec ThreadPoolExecutor pour géocoder plusieurs adresses simultanément
    - Rate limiting intelligent : respecte limite Nominatim (1 req/s) mais parallélise avec Google
    - Priorise Google Maps si disponible (50 QPS vs 1 QPS Nominatim)
    - Utilise le cache existant pour éviter requêtes redondantes
    - Gestion d'erreurs robuste : continue même si certaines adresses échouent

    Args:
        addresses: Liste d'adresses à géocoder
        country: Code pays ISO (ex: "CH")
        max_workers: Nombre max de workers parallèles (défaut: min(5, len(addresses)))
        prefer_google: Si True, utilise Google Maps en priorité (plus rapide, plus de parallélisme)

    Returns:
        Dict[str, Dict[str, float] | None]: Dictionnaire {adresse: {'lat': float, 'lon': float} | None}
        - Clé = adresse originale
        - Valeur = coordonnées ou None si échec

    Exemple:
        >>> results = geocode_addresses_batch(
        ...     ["Rue de Genève 1, Lausanne", "Avenue de la Gare 10, Genève"],
        ...     country="CH"
        ... )
        >>> print(results["Rue de Genève 1, Lausanne"])
        {'lat': 46.5197, 'lon': 6.6323}
    """
    if not addresses:
        return {}

    # ✅ P1: Dédupliquer les adresses pour éviter géocodage redondant
    unique_addresses = list(
        dict.fromkeys(addresses)
    )  # Préserve l'ordre, supprime doublons
    results: Dict[str, Dict[str, float] | None] = {}

    # ✅ P1: Vérifier cache pour toutes les adresses d'abord (très rapide)
    cached_results: Dict[str, Dict[str, float] | None] = {}
    uncached_addresses: List[str] = []

    for address in unique_addresses:
        cached = geocode_address(address, country=country)
        if cached:
            cached_results[address] = cached
        else:
            uncached_addresses.append(address)

    results.update(cached_results)

    # ✅ P1: Si toutes les adresses sont en cache, retourner immédiatement
    if not uncached_addresses:
        app_logger.debug(
            "[Batch Geocoding] ✅ Toutes les %d adresses trouvées en cache",
            len(addresses),
        )
        # Retourner dans l'ordre original
        return {addr: results.get(addr) for addr in addresses}

    app_logger.info(
        "[Batch Geocoding] Géocodage de %d adresses (%d en cache, %d à géocoder)",
        len(addresses),
        len(cached_results),
        len(uncached_addresses),
    )

    # ✅ P1: Déterminer le nombre de workers optimal
    if max_workers is None:
        # Pour Google Maps : jusqu'à 5 workers (50 QPS / 10 = marge de sécurité)
        # Pour Nominatim : 1 worker (rate limit strict 1 req/s)
        if GOOGLE_MAPS_API_KEY and prefer_google:
            max_workers = min(5, len(uncached_addresses))
        else:
            max_workers = 1  # Nominatim : rate limit strict

    # ✅ P1: Fonction wrapper pour géocodage avec gestion d'erreurs
    def _geocode_single(address: str) -> Tuple[str, Dict[str, float] | None]:
        """Géocode une seule adresse avec gestion d'erreurs."""
        try:
            # ✅ P1: Rate limiting pour Nominatim uniquement
            if not GOOGLE_MAPS_API_KEY or not prefer_google:
                _wait_for_nominatim_rate_limit()

            result = geocode_address(address, country=country)
            return (address, result)
        except Exception as e:
            app_logger.warning(
                "[Batch Geocoding] ⚠️ Échec géocodage pour '%s': %s",
                address[:50],
                e,
            )
            return (address, None)

    # ✅ P1: Paralléliser le géocodage des adresses non cachées
    if max_workers > 1 and len(uncached_addresses) > 1:
        # Parallélisation avec ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_address = {
                executor.submit(_geocode_single, addr): addr
                for addr in uncached_addresses
            }

            for future in as_completed(future_to_address):
                address, result = future.result()
                results[address] = result
    else:
        # Géocodage séquentiel (Nominatim ou 1 seule adresse)
        for address in uncached_addresses:
            _, result = _geocode_single(address)
            results[address] = result

    # ✅ P1: Retourner dans l'ordre original des adresses
    final_results = {addr: results.get(addr) for addr in addresses}
    success_count = sum(1 for v in final_results.values() if v is not None)
    app_logger.info(
        "[Batch Geocoding] ✅ Géocodage terminé: %d/%d réussis",
        success_count,
        len(addresses),
    )

    return final_results


# Defaults
UD_MATRIX_MAX_ELEMENTS = int(os.getenv("UD_MATRIX_MAX_ELEMENTS", "100"))
UD_MATRIX_MAX_ROWS = int(os.getenv("UD_MATRIX_MAX_ROWS", "25"))
UD_MATRIX_MAX_COLS = int(os.getenv("UD_MATRIX_MAX_COLS", "25"))
UD_MATRIX_RATE_LIMIT = float(os.getenv("UD_MATRIX_RATE_LIMIT", "8"))
UD_MATRIX_CACHE_TTL = int(os.getenv("UD_MATRIX_CACHE_TTL_SEC", "300"))
UD_MATRIX_GRID_ROUND = int(os.getenv("UD_MATRIX_GRID_ROUND", "50"))
UD_MATRIX_CACHE_MAX_PAIRS = int(os.getenv("UD_MATRIX_CACHE_MAX_PAIRS", "100000"))
UD_MATRIX_CACHE_USE_REDIS = os.getenv("UD_MATRIX_CACHE_USE_REDIS", "0") in (
    "1",
    "true",
    "True",
)
UD_MATRIX_INFLIGHT_TTL_S = int(os.getenv("UD_MATRIX_INFLIGHT_TTL_S", "10"))
REDIS_URL = os.getenv("REDIS_URL", "")

_DM_LOCK = threading.Lock()
_DM_LAST_CALL_TS = {"value": 0.0}

_DM_CACHE: dict[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, int]] = {}
_DM_CACHE_LOCK = threading.Lock()

# ---- singleflight ----
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: Dict[str, Dict[str, Any]] = {}


def _singleflight(key: str, fn):
    with _INFLIGHT_LOCK:
        ent = _INFLIGHT.get(key)
        if not ent:
            ent = {
                "evt": threading.Event(),
                "res": None,
                "err": None,
                "leader": True,
                "ts": time.time(),
            }
            _INFLIGHT[key] = ent
        else:
            ent["leader"] = False
    if ent["leader"]:
        try:
            ent["res"] = fn()
        except Exception as e:
            ent["err"] = e
        finally:
            ent["evt"].set()
            with _INFLIGHT_LOCK:
                for k, v in list(_INFLIGHT.items()):
                    if (
                        v["evt"].is_set()
                        and (time.time() - v.get("ts", 0)) > UD_MATRIX_INFLIGHT_TTL_S
                    ):
                        _INFLIGHT.pop(k, None)
    else:
        ent["evt"].wait()
        if ent["err"]:
            raise ent["err"]
    return ent["res"]


def _decode_cached_duration(v: Any) -> int | None:
    """Convertit une valeur redis (bytes/bytearray/str/int/float) en int.
    Ignore les objets Awaitable éventuels (clients Redis async).
    """
    # éviter import global quand non nécessaire
    try:
        from collections.abc import Awaitable as _AwaitableABC

        if isinstance(v, _AwaitableABC):
            return None
    except Exception:
        pass
    try:
        if isinstance(v, (bytes, bytearray)):
            return int.from_bytes(v, "big", signed=False)
        if isinstance(v, str):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
    except Exception:
        pass
    return None


def _get_redis():
    if not UD_MATRIX_CACHE_USE_REDIS or not REDIS_URL:
        return None
    try:
        import redis

        return redis.from_url(REDIS_URL, decode_responses=False)
    except Exception as e:
        app_logger.warning("⚠️ Redis indisponible: %s", e)
        return None


def _haversine_seconds(
    a: Tuple[float, float], b: Tuple[float, float], avg_kmh: float = 40.0
) -> int:
    # Import centralisé depuis shared.geo_utils
    from shared.geo_utils import haversine_seconds

    lat1, lon1 = float(a[0]), float(a[1])
    lat2, lon2 = float(b[0]), float(b[1])
    result = haversine_seconds(lat1, lon1, lat2, lon2, avg_kmh)
    return max(1, result)


def _round_coord(coord: Tuple[float, float], meters: int) -> Tuple[float, float]:
    """Arrondit une coordonnée sur une grille (≃meters) pour mieux hit le cache."""
    lat, lon = float(coord[0]), float(coord[1])
    dlat = meters / 111_320.0
    dlon = meters / (
        111_320.0 * max(0.1, math.cos(math.radians(max(-89.9, min(89.9, lat)))))
    )
    return (round(lat / dlat) * dlat, round(lon / dlon) * dlon)


def _to_str(c: Tuple[float, float]) -> str:
    return f"{c[0]},{c[1]}"


def _respect_rate_limit(rate: float):
    if rate <= RATE_ZERO:
        return
    with _DM_LOCK:
        now = time.time()
        min_interval = 1.0 / rate
        wait = _DM_LAST_CALL_TS["value"] + min_interval - now
        if wait > WAIT_ZERO:
            time.sleep(wait)
        _DM_LAST_CALL_TS["value"] = time.time()


def _dm_request(
    origins: List[str],
    dests: List[str],
    *,
    departure_time: int | None = None,
    traffic_model: str = "best_guess",
    units: str = "metric",
    region: str = "CH",
    language: str = "fr",
    timeout: int = 12,
    max_retries: int = 1,
    retry_backoff_ms: int = 250,
    rate_limit_per_sec: float = UD_MATRIX_RATE_LIMIT,
) -> List[List[int | None]]:
    """Appelle l'API Distance Matrix une seule fois pour origins x dests.
    Retourne une matrice (len(origins) x len(dests)) en SECONDES (int | None).
    """
    _respect_rate_limit(rate_limit_per_sec)
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params: Dict[str, str] = {
        "origins": "|".join(origins),
        "destinations": "|".join(dests),
        "key": str(GOOGLE_MAPS_API_KEY or ""),
        "language": language,
        "units": units,
        "region": region,
    }
    if departure_time is not None:
        params["departure_time"] = str(int(departure_time))
        params["traffic_model"] = traffic_model

    inflight_key = hashlib.sha256(
        json.dumps({"o": origins, "d": dests, "p": params}, sort_keys=True).encode(
            "utf-8"
        ),
        usedforsecurity=False,
    ).hexdigest()

    def _do() -> List[List[int | None]]:
        """Exécute la requête Distance Matrix avec retry uniformisé."""

        def _fetch_matrix() -> List[List[int | None]]:
            if not _google_maps_circuit_breaker.allow_request():
                raise RuntimeError("google_maps_circuit_open")
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()  # Lève exception pour codes HTTP d'erreur

            # ✅ P1: Protéger parsing JSON contre réponses malformées
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                app_logger.error(
                    (
                        "[GDM] JSON decode error for Google Maps Distance Matrix: %s. "
                        "Response status: %d, Response preview: %s"
                    ),
                    e,
                    resp.status_code,
                    resp.text[:200] if resp.text else "(empty)",
                )
                # Lever exception pour déclencher retry (si retryable) ou fallback
                raise ValueError(f"Google Maps returned invalid JSON: {e}") from e

            if data.get("status") != "OK":
                msg = f"status={data.get('status')} err={data.get('error_message')}"
                raise RuntimeError(msg)

            rows = data.get("rows", [])
            out: List[List[int | None]] = []
            for r in rows:
                row_vals: List[int | None] = []
                for el in r.get("elements", []):
                    if el.get("status") == "OK":
                        dur = (
                            el.get("duration_in_traffic") or el.get("duration") or {}
                        ).get("value", 0)
                        row_vals.append(int(dur))
                    else:
                        row_vals.append(None)
                while len(row_vals) < len(dests):
                    row_vals.append(None)
                out.append(row_vals[: len(dests)])
            while len(out) < len(origins):
                out.append([None for _ in dests])
            _google_maps_circuit_breaker.record_success()
            return out[: len(origins)]

        # ✅ 2.3: Utiliser retry uniformisé avec fallback gracieux
        try:
            result = retry_http_request(
                _fetch_matrix,
                max_retries=max_retries,
                base_delay_ms=retry_backoff_ms,
            )
            return cast(List[List[int | None]], result)
        except Exception as e:
            if str(e) != "google_maps_circuit_open":
                _google_maps_circuit_breaker.record_failure()
            app_logger.warning(
                "⚠️ DistanceMatrix request error (final après retries): %s", e
            )
            # Retourner matrice vide en cas d'échec définitif
            return [[None for _ in dests] for _ in origins]

    start = time.time()
    res = cast("List[List[int | None]]", _singleflight(inflight_key, _do))
    app_logger.info(
        "[GDM] table_fetch o=%s d=%s duration_ms=%s",
        len(origins),
        len(dests),
        int((time.time() - start) * 1000),
    )
    return res


def _all_cached(
    block_o_idx: List[int], block_d_idx: List[int], qcoords: List[Tuple[float, float]]
) -> bool:
    now = time.time()
    r = _get_redis()
    for i in block_o_idx:
        oi = qcoords[i]
        for j in block_d_idx:
            if i == j:
                continue
            dj = qcoords[j]
            if r:
                try:
                    key = f"gdm:{UD_MATRIX_GRID_ROUND}:{oi[0]},{oi[1]},{dj[0]},{dj[1]}"
                    ttl_val = cast("Any", r.ttl(key))
                    # Si le TTL est None/invalid ou <= OU_ZERO -> pas (ou plus) en cache
                    if not isinstance(ttl_val, (int, float)) or ttl_val <= TTL_VAL_ZERO:
                        return False
                except Exception:
                    return False
            else:
                k = (oi, dj)
                with _DM_CACHE_LOCK:
                    v = _DM_CACHE.get(k)
                if not v or v[0] < now:
                    return False
    return True


def _fill_from_cache(
    matrix: List[List[int]], block_o_idx: List[int], block_d_idx: List[int], qcoords
):
    now = time.time()
    r = _get_redis()
    for i in block_o_idx:
        oi = qcoords[i]
        for j in block_d_idx:
            if i == j:
                continue
            dj = qcoords[j]
            if r:
                try:
                    key = f"gdm:{UD_MATRIX_GRID_ROUND}:{oi[0]},{oi[1]},{dj[0]},{dj[1]}"
                    v_raw: Any = r.get(key)
                    val = _decode_cached_duration(v_raw)
                    if val is not None:
                        matrix[i][j] = val
                except Exception:
                    pass
            else:
                with _DM_CACHE_LOCK:
                    v2 = _DM_CACHE.get((oi, dj))
                if v2 and v2[0] >= now:
                    matrix[i][j] = v2[1]


def _update_cache_from_block(
    block: List[List[int | None]],
    block_o_idx: List[int],
    block_d_idx: List[int],
    qcoords: List[Tuple[float, float]],
) -> None:
    expires = time.time() + UD_MATRIX_CACHE_TTL
    r = _get_redis()
    for ii, i in enumerate(block_o_idx):
        oi = qcoords[i]
        for jj, j in enumerate(block_d_idx):
            if i == j:
                continue
            dur = block[ii][jj]
            if dur is not None and dur > DUR_ZERO:
                if r:
                    try:
                        dj = qcoords[j]
                        key = (
                            f"gdm:{UD_MATRIX_GRID_ROUND}:"
                            f"{oi[0]},{oi[1]},{dj[0]},{dj[1]}"
                        )
                        r.setex(
                            key,
                            UD_MATRIX_CACHE_TTL,
                            int(dur).to_bytes(4, "big", signed=False),
                        )
                    except Exception:
                        pass
                else:
                    with _DM_CACHE_LOCK:
                        _DM_CACHE[(oi, qcoords[j])] = (expires, int(dur))
                    if len(_DM_CACHE) > UD_MATRIX_CACHE_MAX_PAIRS:
                        for _ in range(int(UD_MATRIX_CACHE_MAX_PAIRS * 0.1)):
                            try:
                                _DM_CACHE.pop(next(iter(_DM_CACHE)))
                            except Exception:
                                break


def build_distance_matrix_google(
    coords: List[Tuple[float, float]],
    *,
    departure_time: int | None = None,
    traffic_model: str = "best_guess",
    units: str = "metric",
    region: str = "CH",
    language: str = "fr",
    timeout: int = 12,
    max_retries: int = 1,
    retry_backoff_ms: int = 250,
    rate_limit_per_sec: float | None = None,
) -> List[List[int]]:
    """Matrice NxN des durées en SECONDES entre tous les points `coords` (lat, lon)."""
    if not coords:
        return []

    n = len(coords)
    if not GOOGLE_MAPS_API_KEY:
        app_logger.warning("⚠️ GOOGLE_MAPS_API_KEY absente - matrice 100% Haversine.")
    qcoords = [_round_coord(c, UD_MATRIX_GRID_ROUND) for c in coords]

    matrix: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

    base_side = max(1, int(math.sqrt(UD_MATRIX_MAX_ELEMENTS)))
    o_block = min(base_side, UD_MATRIX_MAX_ROWS)
    d_block = min(UD_MATRIX_MAX_ELEMENTS // o_block, UD_MATRIX_MAX_COLS)
    d_block = max(1, d_block)

    o_ranges = [list(range(s, min(s + o_block, n))) for s in range(0, n, o_block)]
    d_ranges = [list(range(s, min(s + d_block, n))) for s in range(0, n, d_block)]

    effective_rate = (
        UD_MATRIX_RATE_LIMIT
        if rate_limit_per_sec is None
        else float(rate_limit_per_sec)
    )

    for o_idx in o_ranges:
        origins = [_to_str(coords[i]) for i in o_idx]

        for d_idx in d_ranges:
            if len(o_idx) == 1 and len(d_idx) == 1 and o_idx[0] == d_idx[0]:
                continue

            if _all_cached(o_idx, d_idx, qcoords):
                _fill_from_cache(matrix, o_idx, d_idx, qcoords)
                continue

            dests = [_to_str(coords[j]) for j in d_idx]
            block: List[List[int | None]]
            if GOOGLE_MAPS_API_KEY:
                block = _dm_request(
                    origins,
                    dests,
                    departure_time=departure_time,
                    traffic_model=traffic_model,
                    units=units,
                    region=region,
                    language=language,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_backoff_ms=retry_backoff_ms,
                    rate_limit_per_sec=effective_rate,
                )
            else:
                block = [[None for _ in dests] for _ in origins]

            for ii, i in enumerate(o_idx):
                oi = coords[i]
                for jj, j in enumerate(d_idx):
                    if i == j:
                        matrix[i][j] = 0
                        continue
                    dur = block[ii][jj]
                    if dur is None or dur <= DUR_ZERO:
                        dur = _haversine_seconds(oi, coords[j])
                    matrix[i][j] = int(dur)

            _update_cache_from_block(block, o_idx, d_idx, qcoords)

    return matrix
