import requests
import os, json, hashlib
from ext import app_logger
import math, time, threading
from typing import List, Tuple, Optional, Dict, Any

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
_GOOGLE_TIMEOUT = 10  # s
_AVG_SPEED_KMH = 40.0  # fallback

def _as_origin_str(addr_or_coord):
    """Accepte une adresse string OU un tuple (lat, lon) -> 'lat,lon' ou adresse telle quelle."""
    if isinstance(addr_or_coord, (list, tuple)) and len(addr_or_coord) == 2:
        return f"{float(addr_or_coord[0])},{float(addr_or_coord[1])}"
    return str(addr_or_coord)

def _haversine_km(a, b):
    lat1, lon1 = float(a[0]), float(a[1])
    lat2, lon2 = float(b[0]), float(b[1])
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(x), math.sqrt(1 - x)))

def get_distance_duration(pickup_address, dropoff_address, *, departure_time: Optional[int] = None,
                          traffic_model: str = "best_guess", units: str = "metric",
                          region: str = "CH", language: str = "fr", timeout: int = _GOOGLE_TIMEOUT):
    """
    Retourne (duration_seconds, distance_meters).
    Accepte adresse string OU tuple (lat, lon) en entrée.
    Fallback Haversine si la clé API manque ou si l’API renvoie une erreur.
    """
    # Si on a des tuples, on peut aussi calculer un fallback fiable
    pick_is_coord = isinstance(pickup_address, (list, tuple)) and len(pickup_address) == 2
    drop_is_coord = isinstance(dropoff_address, (list, tuple)) and len(dropoff_address) == 2

    if not GOOGLE_MAPS_API_KEY:
        if pick_is_coord and drop_is_coord:
            dist_km = _haversine_km(pickup_address, dropoff_address)
            dur_s = int((dist_km / _AVG_SPEED_KMH) * 3600)
            return max(1, dur_s), int(dist_km * 1000)
        raise EnvironmentError("Clé API Google Maps manquante et aucune coordonnée pour fallback.")

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"    
    params = {
        "origins": _as_origin_str(pickup_address),
        "destinations": _as_origin_str(dropoff_address),
        "key": GOOGLE_MAPS_API_KEY,
        "language": language,
        "units": units,
        "region": region,    
        }
    if departure_time is not None:
        params["departure_time"] = int(departure_time)
        params["traffic_model"] = traffic_model
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            raise RuntimeError(data.get("error_message") or data.get("status"))

        elem = data["rows"][0]["elements"][0]
        if elem.get("status") != "OK":
            raise RuntimeError(elem.get("status"))

        # duration_in_traffic si dispo, sinon duration
        duration_seconds = int((elem.get("duration_in_traffic") or elem.get("duration"))["value"])
        distance_meters = int(elem["distance"]["value"])
        return duration_seconds, distance_meters

    except Exception as e:
        app_logger.warning(f"⚠️ DistanceMatrix single-pair fallback: {e}")
        if pick_is_coord and drop_is_coord:
            dist_km = _haversine_km(pickup_address, dropoff_address)
            dur_s = int((dist_km / _AVG_SPEED_KMH) * 3600)
            return max(1, dur_s), int(dist_km * 1000)
        # si adresses textuelles, pas de fallback fiable → propage une erreur claire
        raise

def geocode_address(address: str, *, country: str | None = None, language: str = "fr"):
    """
    Géocode une adresse → (lat, lon) | None.
    - country: code ISO (ex: "CH") pour biaiser la recherche (Google: components=country:CH)
    - language: "fr" par défaut
    """
    if not GOOGLE_MAPS_API_KEY:
        app_logger.error("❌ Clé API Google Maps manquante.")
        raise EnvironmentError("Clé API Google Maps non définie.")

    address = (address or "").strip()
    if not address:
        raise ValueError("Adresse vide ou invalide pour le géocodage.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY,
        "language": language,
    }
    if country:
        # Biais pays côté Google Geocoding
        params["components"] = f"country:{country}"

    try:
        resp = requests.get(url, params=params, timeout=_GOOGLE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK" or not data.get("results"):
            app_logger.warning(f"⚠️ Aucune coordonnée trouvée pour : '{address}' (country={country}).")
            return None

        loc = data["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])

    except requests.RequestException as e:
        app_logger.error(f"❌ Erreur API Google Maps pour '{address}' (country={country}): {e}")
        return None

# Defaults (peuvent être surchargés par ENV)
UD_MATRIX_MAX_ELEMENTS = int(os.getenv("UD_MATRIX_MAX_ELEMENTS", 100))   # elements = origins * destinations
UD_MATRIX_MAX_ROWS     = int(os.getenv("UD_MATRIX_MAX_ROWS", 25))        # max origins par requête
UD_MATRIX_MAX_COLS     = int(os.getenv("UD_MATRIX_MAX_COLS", 25))        # max destinations par requête
UD_MATRIX_RATE_LIMIT   = float(os.getenv("UD_MATRIX_RATE_LIMIT", 8))     # req/sec
UD_MATRIX_CACHE_TTL    = int(os.getenv("UD_MATRIX_CACHE_TTL_SEC", 300))  # seconds
UD_MATRIX_GRID_ROUND   = int(os.getenv("UD_MATRIX_GRID_ROUND", 50))      # meters (≈50m)
UD_MATRIX_CACHE_MAX_PAIRS = int(os.getenv("UD_MATRIX_CACHE_MAX_PAIRS", 100_000))
UD_MATRIX_CACHE_USE_REDIS  = os.getenv("UD_MATRIX_CACHE_USE_REDIS", "0") in ("1","true","True")
UD_MATRIX_INFLIGHT_TTL_S   = int(os.getenv("UD_MATRIX_INFLIGHT_TTL_S", 10))
REDIS_URL = os.getenv("REDIS_URL", "")

_DM_LOCK = threading.Lock()
_DM_LAST_CALL_TS = 0.0  # rate-limit horodaté par process

# Cache: key=((olat,olon),(dlat,dlon)) -> (expires_ts, duration_seconds)
_DM_CACHE: dict[Tuple[Tuple[float,float], Tuple[float,float]], Tuple[float,int]] = {}
_DM_CACHE_LOCK = threading.Lock()


# ---- singleflight (dédup in-flight) : DÉFINITIONS MANQUANTES CORRIGÉES ----
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: Dict[str, Dict[str, Any]] = {}

def _singleflight(key: str, fn):
    """Déduplique les appels concurrents identiques : un leader fait la requête, les autres attendent."""
    with _INFLIGHT_LOCK:
        ent = _INFLIGHT.get(key)
        if not ent:
            ent = {"evt": threading.Event(), "res": None, "err": None, "leader": True, "ts": time.time()}
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
            # GC léger des entrées anciennes
            with _INFLIGHT_LOCK:
                for k, v in list(_INFLIGHT.items()):
                    if v["evt"].is_set() and (time.time() - v.get("ts", 0)) > UD_MATRIX_INFLIGHT_TTL_S:
                        _INFLIGHT.pop(k, None)
    else:
        ent["evt"].wait()
        if ent["err"]:
            raise ent["err"]
    return ent["res"]

def _get_redis():
    if not UD_MATRIX_CACHE_USE_REDIS or not REDIS_URL:
        return None
    try:
        import redis  # type: ignore
        return redis.from_url(REDIS_URL, decode_responses=False)
    except Exception as e:
        app_logger.warning(f"⚠️ Redis indisponible: {e}")
        return None

def _haversine_seconds(a: Tuple[float,float], b: Tuple[float,float], avg_kmh: float = 40.0) -> int:
    lat1, lon1 = float(a[0]), float(a[1])
    lat2, lon2 = float(b[0]), float(b[1])
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    dist_km = R * (2 * math.atan2(math.sqrt(x), math.sqrt(1 - x)))
    return max(1, int((dist_km / avg_kmh) * 3600))

def _round_coord(coord: Tuple[float,float], meters: int) -> Tuple[float,float]:
    """Arrondit une coordonnée sur une grille (≃meters) pour mieux hit le cache."""
    lat, lon = float(coord[0]), float(coord[1])
    # 1° lat ≃ 111_320 m ; 1° lon ≃ 111_320 * cos(lat)
    dlat = meters / 111_320.0
    dlon = meters / (111_320.0 * max(0.1, math.cos(math.radians(max(-89.9, min(89.9, lat))))))
    return (round(lat / dlat) * dlat, round(lon / dlon) * dlon)

def _to_str(c: Tuple[float,float]) -> str:
    return f"{c[0]},{c[1]}"

def _respect_rate_limit(rate: float):
    global _DM_LAST_CALL_TS
    if rate <= 0:
        return
    with _DM_LOCK:
        now = time.time()
        min_interval = 1.0 / rate
        wait = _DM_LAST_CALL_TS + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _DM_LAST_CALL_TS = time.time()

def _dm_request(origins: List[str], dests: List[str], *,
                departure_time: Optional[int] = None,
                traffic_model: str = "best_guess",
                units: str = "metric",
                region: str = "CH",
                language: str = "fr",
                timeout: int = 12,
                max_retries: int = 3,
                retry_backoff_ms: int = 250,
                rate_limit_per_sec: float = UD_MATRIX_RATE_LIMIT) -> List[List[Optional[int]]]:    
    """
    Appelle l'API Distance Matrix une seule fois pour origins × dests.
    Retourne une matrice (len(origins) × len(dests)) en SECONDES.
    Les erreurs ponctuelles se traduisent par None et seront fallback en Haversine par le caller.
    """
    _respect_rate_limit(rate_limit_per_sec)
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(dests),
        "key": GOOGLE_MAPS_API_KEY,
        "language": language,
        "units": units,
        "region": region,
    }
    if departure_time is not None:
        params["departure_time"] = int(departure_time)
        params["traffic_model"] = traffic_model

    # Clé de dédup in-flight basée sur la requête
    inflight_key = hashlib.sha1(json.dumps({"o": origins, "d": dests, "p": params}, sort_keys=True).encode("utf-8")).hexdigest()

    def _do():
        # Retries simples avec backoff linéaire
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                data = resp.json()
                if data.get("status") != "OK":
                    raise RuntimeError(f"status={data.get('status')} err={data.get('error_message')}")

                rows = data.get("rows", [])
                out: List[List[Optional[int]]] = []
                for r in rows:
                    row_vals: List[Optional[int]] = []
                    for el in r.get("elements", []):
                        if el.get("status") == "OK":
                            dur = (el.get("duration_in_traffic") or el.get("duration") or {}).get("value", 0)
                            row_vals.append(int(dur))
                        else:
                            row_vals.append(None)  # ZERO_RESULTS, NOT_FOUND, etc.
                    # padding si mismatch (rare)
                    while len(row_vals) < len(dests):
                        row_vals.append(None)
                    out.append(row_vals[:len(dests)])
                # padding lignes si mismatch
                while len(out) < len(origins):
                    out.append([None for _ in dests])
                return out[:len(origins)]
            except Exception as e:
                if attempt >= max_retries:
                    app_logger.warning(f"⚠️ DistanceMatrix request error (final): {e}")
                    return [[None for _ in dests] for _ in origins]
                app_logger.warning(f"⚠️ DistanceMatrix request error (retry {attempt+1}/{max_retries}): {e}")
                time.sleep((retry_backoff_ms / 1000.0) * (attempt + 1))

    start = time.time()
    res = _singleflight(inflight_key, _do)
    app_logger.info(f"[GDM] table_fetch o={len(origins)} d={len(dests)} duration_ms={int((time.time()-start)*1000)}")
    return res

def _all_cached(block_o_idx: List[int], block_d_idx: List[int], qcoords: List[Tuple[float,float]]) -> bool:
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
                    key = f"gdm:{UD_MATRIX_GRID_ROUND}:{oi[0]:.6f},{oi[1]:.6f}|{dj[0]:.6f},{dj[1]:.6f}"
                    if not r.ttl(key) or r.ttl(key) <= 0:
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

def _fill_from_cache(matrix: List[List[int]], block_o_idx: List[int], block_d_idx: List[int], qcoords, coords):
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
                    key = f"gdm:{UD_MATRIX_GRID_ROUND}:{oi[0]:.6f},{oi[1]:.6f}|{dj[0]:.6f},{dj[1]:.6f}"
                    v = r.get(key)
                    if v:
                        matrix[i][j] = int.from_bytes(v, "big")
                except Exception:
                    pass
            else:
                with _DM_CACHE_LOCK:
                    v = _DM_CACHE.get((oi, dj))
                if v and v[0] >= now:
                    matrix[i][j] = v[1]

def _update_cache_from_block(block: List[List[int]], block_o_idx: List[int], block_d_idx: List[int], qcoords):
    expires = time.time() + UD_MATRIX_CACHE_TTL
    r = _get_redis()
    for ii, i in enumerate(block_o_idx):
        oi = qcoords[i]
        for jj, j in enumerate(block_d_idx):
            if i == j:
                continue
            dur = block[ii][jj]
            if dur is not None and dur > 0:
                if r:
                    try:
                        dj = qcoords[j]
                        key = f"gdm:{UD_MATRIX_GRID_ROUND}:{oi[0]:.6f},{oi[1]:.6f}|{dj[0]:.6f},{dj[1]:.6f}"
                        r.setex(key, UD_MATRIX_CACHE_TTL, int(dur).to_bytes(4, "big", signed=False))
                    except Exception:
                        pass
                else:
                    with _DM_CACHE_LOCK:
                        _DM_CACHE[(oi, qcoords[j])] = (expires, int(dur))
                    # borne douce du cache mémoire
                    if len(_DM_CACHE) > UD_MATRIX_CACHE_MAX_PAIRS:
                        # éviction grossière: on retire ~10% des entrées les plus anciennes/quelconques
                        for _ in range(int(UD_MATRIX_CACHE_MAX_PAIRS * 0.1)):
                            try:
                                _DM_CACHE.pop(next(iter(_DM_CACHE)))
                            except Exception:
                                break

def build_distance_matrix_google(
    coords: List[Tuple[float,float]],
    *,
    departure_time: Optional[int] = None,
    traffic_model: str = "best_guess",
    units: str = "metric",
    region: str = "CH",
    language: str = "fr",
    timeout: int = 12,
    max_retries: int = 3,
    retry_backoff_ms: int = 250,
    rate_limit_per_sec: Optional[float] = None,
) -> List[List[int]]:
    """
    Matrice NxN des durées en SECONDES entre tous les points `coords` (lat, lon).
    - Batchize origins×destinations en respectant UD_MATRIX_MAX_* et UD_MATRIX_MAX_ELEMENTS
    - Cache TTL avec arrondi de grille (UD_MATRIX_GRID_ROUND)
    - Rate-limit configurable (UD_MATRIX_RATE_LIMIT)
    - Fallback Haversine si API échoue sur une case
    """
    if not coords:
        return []

    n = len(coords)
    # Si pas de clé, on passe tout en Haversine (grâce au fallback)
    if not GOOGLE_MAPS_API_KEY:
        app_logger.warning("⚠️ GOOGLE_MAPS_API_KEY absente — matrice 100% Haversine.")
    # coords quantisées pour cache
    qcoords = [_round_coord(c, UD_MATRIX_GRID_ROUND) for c in coords]

    # init matrice
    matrix: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

    # Déterminer des tailles de blocs sûres
    # Contrainte 1: rows ≤ UD_MATRIX_MAX_ROWS, cols ≤ UD_MATRIX_MAX_COLS
    # Contrainte 2: rows * cols ≤ UD_MATRIX_MAX_ELEMENTS
    # On choisit une taille carrée autant que possible puis on ajuste.
    base_side = max(1, int(math.sqrt(UD_MATRIX_MAX_ELEMENTS)))
    o_block = min(base_side, UD_MATRIX_MAX_ROWS)
    d_block = min(UD_MATRIX_MAX_ELEMENTS // o_block, UD_MATRIX_MAX_COLS)
    d_block = max(1, d_block)

    # Indices de blocs
    o_ranges = [list(range(s, min(s + o_block, n))) for s in range(0, n, o_block)]
    d_ranges = [list(range(s, min(s + d_block, n))) for s in range(0, n, d_block)]

    # Rate limit effectif (sans muter la globale)
    effective_rate = UD_MATRIX_RATE_LIMIT if rate_limit_per_sec is None else float(rate_limit_per_sec)

    for o_idx in o_ranges:
        # origins (strings)
        origins = [_to_str(coords[i]) for i in o_idx]

        for d_idx in d_ranges:
            # skip la diagonale dans l’appel, mais on remplit quand même la matrice
            if len(o_idx) == 1 and len(d_idx) == 1 and o_idx[0] == d_idx[0]:
                continue

            # Si tout est déjà en cache valide -> pas d’appel API
            if _all_cached(o_idx, d_idx, qcoords):
                _fill_from_cache(matrix, o_idx, d_idx, qcoords, coords)
                continue

            dests = [_to_str(coords[j]) for j in d_idx]
            # si pas de clé => on simule un block "None" (on tombera en Haversine)
            block = (
                _dm_request(
                    origins, dests,
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
                if GOOGLE_MAPS_API_KEY else
                [[None for _ in dests] for _ in origins]
            )

            # Remplit, en fallback Haversine si None
            for ii, i in enumerate(o_idx):
                oi = coords[i]
                for jj, j in enumerate(d_idx):
                    if i == j:
                        matrix[i][j] = 0
                        continue
                    dur = block[ii][jj]
                    if dur is None or dur <= 0:
                        # fallback
                        dur = _haversine_seconds(oi, coords[j])
                    matrix[i][j] = int(dur)

            # Met à jour le cache pour ce block (avec coords quantisées)
            _update_cache_from_block(block, o_idx, d_idx, qcoords)
    return matrix
