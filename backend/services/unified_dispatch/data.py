
# backend/services/unified_dispatch/data.py
from __future__ import annotations

import functools
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Tuple, cast

import pytz
from sqlalchemy import and_, func, or_  # (optionnel) peut \u00eatre nettoy\u00e9 si non utilis\u00e9s
from sqlalchemy.orm import joinedload

from models import Booking, BookingStatus, Company, Driver
from services.dispatch_utils import count_assigned_bookings_for_day
from services.maps import geocode_address
from services.osrm_client import build_distance_matrix_osrm
from services.unified_dispatch.heuristics import haversine_minutes
from services.unified_dispatch.settings import Settings, driver_work_window_from_config
from shared.time_utils import day_local_bounds, now_local, parse_local_naive

logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = Settings()

# Autoriser (ou non) le g\u00e9ocodage serveur pour compl\u00e9ter des coordonn\u00e9es manquantes.
# Par d\u00e9faut: d\u00e9sactiv\u00e9 (respect du "sans g\u00e9ocodage serveur").
_ALLOW_SERVER_GEOCODE = os.getenv("DISPATCH_ALLOW_SERVER_GEOCODE", "0") not in ("0", "", "false", "False", "FALSE")

# Optionnel: consid\u00e8re une position "fra\u00eeche" si < 5 min
_POS_TTL = timedelta(minutes=5)

# Verrous in-process + (optionnel) Redis Lock pour un run par (company, day)
_dispatch_locks: Dict[str, Any] = {}

# ------------------------------------------------------------
# Helpers cache/m\u00e9mo
# ------------------------------------------------------------
def _get_redis_from_settings(settings) -> Any | None:
    """
    Tente de r\u00e9cup\u00e9rer un client Redis depuis les settings (ou via REDIS_URL).
    Retourne None si indisponible.
    """
    try:
        rc = getattr(getattr(settings, "matrix", None), "redis_client", None)
        if rc is not None:
            return rc
    except Exception:
        pass
    try:
        url = os.getenv("REDIS_URL", None) or getattr(getattr(settings, "matrix", None), "redis_url", None)
        if url:
            # Lazy import pour compat Windows, pas d'obligation de d\u00e9pendance
            import redis  # type: ignore
            return redis.from_url(url, decode_responses=False)
    except Exception:
        logger.warning("[Dispatch] Redis unavailable; continuing without cache.", exc_info=True)
    return None

def _canonical_coords(coords: List[Tuple[float, float]], prec: int = 5) -> List[Tuple[float, float]]:
    """Arrondit les coordonn\u00e9es pour stabiliser les cl\u00e9s cache (~1m par d\u00e9faut)."""
    out: List[Tuple[float, float]] = []
    for (lat, lon) in coords:
        try:
            out.append((round(float(lat), prec), round(float(lon), prec)))
        except Exception:
            out.append((round(46.2044, prec), round(6.1432, prec)))
    return out

# LRU process-local pour limiter les recalculs dans le m\u00eame process
@functools.lru_cache(maxsize=128)
def _haversine_matrix_cached(coords_key_json: str, avg_speed_kmh: float) -> List[List[float]]:
    """
    Cache process-local d'une matrice Haversine.
    coords_key_json: JSON compact d'une liste de coordonn\u00e9es arrondies.
    """
    try:
        coords = json.loads(coords_key_json)
    except Exception as e:
        logger.warning(f"[Dispatch] Failed to parse coords_key_json: {e}")
        # En dernier recours, on utilise des coordonn\u00e9es par d\u00e9faut
        coords = [(46.2044, 6.1432), (46.2044, 6.1432)]  # Gen\u00e8ve par d\u00e9faut

    # Ensure coords is not empty and is a list to avoid errors
    if not coords or len(coords) < 1:
        coords = [(46.2044, 6.1432), (46.2044, 6.1432)]  # Gen\u00e8ve par d\u00e9faut

    if not isinstance(coords, list):
        coords = [(46.2044, 6.1432), (46.2044, 6.1432)]  # Gen\u00e8ve par d\u00e9faut

    # Ensure all coordinates are valid tuples with proper validation
    valid_coords = []
    for coord in coords:
        try:
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                lat, lon = coord
                # Ensure coordinates are valid numbers
                lat_float = float(lat)
                lon_float = float(lon)
                # Validate coordinate ranges
                if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                    valid_coords.append((lat_float, lon_float))
                else:
                    logger.warning(f"[Dispatch] Invalid coordinate range: lat={lat_float}, lon={lon_float}, using fallback")
                    valid_coords.append((46.2044, 6.1432))
            else:
                logger.warning(f"[Dispatch] Invalid coordinate format: {coord}, using fallback")
                valid_coords.append((46.2044, 6.1432))
        except (ValueError, TypeError) as e:
            logger.warning(f"[Dispatch] Failed to parse coordinate {coord}: {e}, using fallback")
            valid_coords.append((46.2044, 6.1432))

    # Ensure we have at least 2 coordinates for matrix calculation
    if len(valid_coords) < 2:
        logger.warning("[Dispatch] Less than 2 valid coordinates, using fallback")
        valid_coords = [(46.2044, 6.1432), (46.2044, 6.1432)]

    logger.info(f"[Dispatch] Using {len(valid_coords)} valid coordinates for haversine matrix")
    return _build_distance_matrix_haversine(valid_coords, avg_speed_kmh)

# build_problem_data, enrich coords, time-matrix
# ============================================================
# 1\ufe0f\u20e3 R\u00e9cup\u00e9ration des donn\u00e9es brutes (bookings & drivers)
# ============================================================

def get_bookings_for_dispatch(company_id: int, horizon_minutes: int) -> List[Booking]:
    now_ts = now_local()
    horizon_local = now_ts + timedelta(minutes=horizon_minutes)

    bookings = (
        Booking.query
        # évite les soucis de typage Pylance sur RelationshipProperty
        .options(joinedload("driver"))  # type: ignore[arg-type]
        .filter(
            Booking.company_id == company_id,
            # caster la colonne pour éviter les bool Python
            or_(
                cast(Any, Booking.status) == BookingStatus.ACCEPTED,
                cast(Any, Booking.status) == BookingStatus.ASSIGNED,
            ),
            Booking.scheduled_time.isnot(None),  # type: ignore[attr-defined]
            Booking.scheduled_time <= horizon_local,  # type: ignore[attr-defined]
        )
        .order_by(Booking.scheduled_time.asc())  # type: ignore[attr-defined]
        .all()
    )

    # Ici, **aucune** conversion : on laisse les datetimes NA\u00cfFS tels quels
    # 🚫 Exclure les retours avec heure à confirmer (00:00)
    # Les retours avec scheduled_time = NULL sont déjà exclus par le filtre SQL ci-dessus
    # 🚫 Exclure les retours avec heure à confirmer (time_confirmed = False)
    filtered_bookings = []
    excluded_count = 0

    logger.warning(f"[DATA] 🔍 Filtrage de {len(bookings)} courses...")

    for b in bookings:
        scheduled = getattr(b, "scheduled_time", None)
        if scheduled is None:
            logger.warning(f"⏸️ Course #{b.id} EXCLUE : scheduled_time est NULL")
            continue

        # Si c'est un retour avec time_confirmed = False → exclure du dispatch
        is_return = bool(getattr(b, "is_return", False))
        time_confirmed = bool(getattr(b, "time_confirmed", True))

        logger.info(f"[DATA] Course #{b.id}: is_return={is_return}, time_confirmed={time_confirmed}")

        if is_return and not time_confirmed:
            excluded_count += 1
            logger.error(f"⏸️ Course #{b.id} ({getattr(b, 'customer_name', 'N/A')}) EXCLUE du dispatch : retour avec heure à confirmer")
            continue

        logger.info(f"[DATA] ✅ Course #{b.id} INCLUSE dans le dispatch")
        filtered_bookings.append(b)

    logger.error(f"[DATA] ✅ {len(filtered_bookings)} courses après filtrage ({excluded_count} retours exclus)")
    return filtered_bookings

def _normalize_booking_time_fields(bookings: List[Booking]) -> List[Booking]:
    """
    Normalise les champs de date/heure des r\u00e9servations pour garantir qu'ils sont
    tous dans le m\u00eame format (UTC avec timezone).
    """
    tz = pytz.timezone('Europe/Zurich')
    for b in bookings:
        b_any: Any = b
        st = getattr(b_any, "scheduled_time", None)
        if st is not None and getattr(st, "tzinfo", None) is None:
            b_any.scheduled_time = tz.localize(cast(datetime, st))
        if hasattr(b_any, "dropoff_time"):
            dt2 = getattr(b_any, "dropoff_time", None)
            if dt2 is not None and getattr(dt2, "tzinfo", None) is None:
                b_any.dropoff_time = tz.localize(cast(datetime, dt2))
    return bookings

def get_bookings_for_day(company_id, day_str, Booking=None, BookingStatus=None):
    """
    Improved version of get_bookings_for_day that handles both timezone-aware and naive datetimes
    
    Args:
        company_id (int): Company ID
        day_str (str): Date string in YYYY-MM-DD format
        Booking: SQLAlchemy Booking model (optional)
        BookingStatus: Booking status enum (optional)
        
    Returns:
        list: List of bookings for the specified day
    """
    import logging
    from datetime import datetime

    import pytz

    logger = logging.getLogger(__name__)

    # Use the imported models if not provided
    if Booking is None:
        from models import Booking
    if BookingStatus is None:
        from models import BookingStatus

    # Ensure day_str is in the correct format
    if not day_str or not isinstance(day_str, str):
        logger.warning(f"[Dispatch] Invalid day_str: {day_str}, using today's date")
        day_str = datetime.now().strftime("%Y-%m-%d")

    # Parse the day string to get year, month, day
    try:
        y, m, d = map(int, day_str.split("-"))
    except (ValueError, AttributeError):
        logger.warning(f"[Dispatch] Failed to parse day_str: {day_str}, using today's date")
        today = datetime.now()
        y, m, d = today.year, today.month, today.day

    # Create local timezone bounds (Europe/Zurich)
    zurich_tz = pytz.timezone('Europe/Zurich')

    # Create start and end datetime objects in Europe/Zurich timezone
    start_local = datetime(y, m, d, 0, 0, 0)
    end_local = datetime(y, m, d, 23, 59, 59)

    # Make them timezone-aware
    start_local_aware = zurich_tz.localize(start_local)
    end_local_aware = zurich_tz.localize(end_local)

    # Convert to UTC for comparison with timezone-aware datetimes
    start_utc = start_local_aware.astimezone(pytz.UTC)
    end_utc = end_local_aware.astimezone(pytz.UTC)

    # Get valid statuses for dispatch
    valid_statuses = []

    # Add enum values
    if hasattr(BookingStatus, 'ACCEPTED'):
        valid_statuses.append(BookingStatus.ACCEPTED)
    if hasattr(BookingStatus, 'ASSIGNED'):
        valid_statuses.append(BookingStatus.ASSIGNED)

    # Create a time expression that checks multiple time fields
    def booking_time_expr():
        """
        Returns a SQLAlchemy expression representing the "time" of a booking.
        Preference order: scheduled_time, pickup_time, date_time, datetime.
        """
        expr = None
        for name in ("scheduled_time", "pickup_time", "date_time", "datetime"):
            col = getattr(Booking, name, None)
            if col is None:
                continue
            expr = col if expr is None else func.coalesce(expr, col)
        return expr or Booking.scheduled_time

    time_expr = booking_time_expr()

    # Query with a more tolerant time window filter
    try:
        # ✅ Inclure PENDING si utilisé, et enlever lower() sur Enum (Postgres enum)
        status_filters = [
            cast(Any, Booking.status) == BookingStatus.ACCEPTED,
            cast(Any, Booking.status) == BookingStatus.ASSIGNED,
        ]
        if hasattr(BookingStatus, 'PENDING'):
            status_filters.append(cast(Any, Booking.status) == BookingStatus.PENDING)

        bookings = (
            Booking.query
            .filter(
                Booking.company_id == company_id,
                or_(*status_filters),
                time_expr.isnot(None),
                # Use OR condition to match both timezone-aware and naive datetimes
                or_(
                    # For timezone-aware datetimes (UTC)
                    and_(
                        time_expr >= start_utc,  # type: ignore[operator]
                        time_expr <= end_utc     # type: ignore[operator]
                    ),
                    # For naive datetimes (local)
                    and_(
                        time_expr >= start_local,
                        time_expr <= end_local
                    ),
                    # For date-only comparison (SQLite)
                    func.date(time_expr) == func.date(start_local)
                )
            )
            .order_by(time_expr.asc())
        )

        result = bookings.all()

        # Log detailed information about the found bookings
        booking_ids = [b.id for b in result]
        booking_times = [getattr(b, 'scheduled_time', None) for b in result]

        logger.info(f"[Dispatch] Found {len(result)} bookings for company {company_id} on {day_str}")
        if result:
            logger.info(f"[Dispatch] Booking IDs: {booking_ids[:3]}...")
            logger.info(f"[Dispatch] Booking times: {booking_times[:3]}...")

        # 🚫 FILTRE PYTHON : Exclure les retours avec heure à confirmer (time_confirmed = False)
        filtered_result = []
        excluded_count = 0

        logger.error(f"[DATA] 🔍 FILTRAGE de {len(result)} courses pour retours non confirmés...")

        for b in result:
            # Si c'est un retour avec time_confirmed = False → exclure du dispatch
            is_return = bool(getattr(b, "is_return", False))
            time_confirmed = bool(getattr(b, "time_confirmed", True))

            if is_return and not time_confirmed:
                excluded_count += 1
                logger.error(f"⏸️ Course #{b.id} ({getattr(b, 'customer_name', 'N/A')}) EXCLUE : retour avec time_confirmed=False")
                continue

            filtered_result.append(b)

        logger.error(f"[DATA] ✅ {len(filtered_result)} courses après filtrage ({excluded_count} retours exclus avec heure à confirmer)")
        return filtered_result
    except Exception as e:
        logger.error(f"[Dispatch] Error querying bookings for day: {e}")
        return []

def get_available_drivers(company_id: int) -> List[Driver]:
    """
    R\u00e9cup\u00e8re les chauffeurs actifs & disponibles pour dispatch,
    et normalise last_position_update en UTC (tz-aware).
    """
    drivers = (
        Driver.query
        .filter(
            Driver.company_id == company_id,
            cast(Any, Driver.is_active).is_(True),
            cast(Any, Driver.is_available).is_(True),
        )
        .all()
    )

    # \ud83d\udd27 Normalisation "en m\u00e9moire" (mode na\u00eff)
    for d in drivers:
        # On conserve les dates na\u00efves telles quelles
        _ = getattr(d, "last_position_update", None)
        # Optionnel: robustesse si des coords arrivent en str
        try:
            if d.latitude is not None:
                d_any: Any = d
                d_any.latitude = float(cast(Any, d.latitude))
            if d.longitude is not None:
                d_any = cast(Any, d)
                d_any.longitude = float(cast(Any, d.longitude))
        except Exception:
            # on laisse l'enrichissement faire le fallback plus tard
            d_any = cast(Any, d)
            d_any.latitude = None
            d_any.longitude = None

    return drivers

def get_available_drivers_split(company_id: int) -> tuple[List[Driver], List[Driver]]:
    """
    Retourne (réguliers, urgences) à partir du pool actif & dispo.
    Tolérant si driver_type est un Enum OU une chaîne ("regular"/"REGULAR"/...).
    """
    import logging
    logger = logging.getLogger(__name__)

    drivers = get_available_drivers(company_id)

    def norm_type(dt):
        # supporte Enum, str, None
        s = str(dt or "").strip().upper()
        # Si c’est un Enum SQLA, str(dt) peut donner "DriverType.REGULAR" → on garde la dernière partie
        if "." in s:
            s = s.split(".")[-1]
        return s

    regs, emgs, unknown = [], [], []
    for d in drivers:
        t = norm_type(getattr(d, "driver_type", None))
        if t == "REGULAR":
            regs.append(d)
        elif t == "EMERGENCY":
            emgs.append(d)
        else:
            # On peut choisir de classer par défaut en REGULAR, ou juste tracer
            unknown.append(d)

    logger.info(
        "[Dispatch] Drivers available: total=%d regular=%d emergency=%d unknown=%d ids(reg)=%s ids(emg)=%s",
        len(drivers), len(regs), len(emgs), len(unknown),
        [getattr(d, 'id', None) for d in regs],
        [getattr(d, 'id', None) for d in emgs],
    )

    # Optionnel: si tout est "unknown", prends-les comme réguliers pour ne pas bloquer
    if not regs and not emgs and unknown:
        logger.warning("[Dispatch] All drivers have unknown driver_type → falling back to REGULAR for all.")
        regs = unknown
        unknown = []

    return regs, emgs


# ============================================================
# 2\ufe0f\u20e3 Enrichissement coordonn\u00e9es (sans Google / sans g\u00e9ocodage)
# ============================================================

def _company_latlon(company: Company) -> tuple[float, float]:
    """Retourne les coords de l'entreprise si dispo, sinon Genève."""
    if company:
        c_any = cast(Any, company)
        lat = getattr(c_any, "latitude", None)
        lon = getattr(c_any, "longitude", None)
        if lat is not None and lon is not None:
            try:
                return float(cast(Any, lat)), float(cast(Any, lon))
            except Exception:
                pass
    # Fallback Gen\u00e8ve (centre-ville)
    return 46.2044, 6.1432

def _to_float_opt(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

@lru_cache(maxsize=256)
def _geocode_safe_cached(address: str) -> tuple[float, float] | None:
    if not address or not address.strip():
        return None
    try:
        res = geocode_address(address)
        # attendu: dict {"lat": ..., "lon": ...} OU (lat, lon)
        if isinstance(res, dict) and "lat" in res and "lon" in res:
            res_d = cast(Dict[str, Any], res)
            lat = _to_float_opt(res_d.get("lat"))
            lon = _to_float_opt(res_d.get("lon"))
            if lat is not None and lon is not None:
                return lat, lon
            return None
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            lat = _to_float_opt(res[0])
            lon = _to_float_opt(res[1])
            if lat is not None and lon is not None:
                return lat, lon
            return None
    except Exception:
        logger.warning("[Dispatch] geocode_address failed for '%s'", address, exc_info=True)
    return None

def enrich_booking_coords(bookings: List[Booking], company: Company) -> None:
    """
    Compl\u00e8te les coordonn\u00e9es pickup/dropoff des bookings MANQUANTES.
    R\u00e8gles :
      - Si le frontend a d\u00e9j\u00e0 fourni lat/lon \u21d2 on ne touche \u00e0 rien.
      - Sinon, si DISPATCH_ALLOW_SERVER_GEOCODE=1 et une adresse est disponible \u21d2 on tente un g\u00e9ocodage.
      - Sinon, fallback : coords de l'entreprise, sinon Gen\u00e8ve.
      - \u00c9crit en m\u00e9moire uniquement (pas de commit).
    """
    fallback_lat, fallback_lon = _company_latlon(company)

    for b in bookings:
        # --- PICKUP ---
        plat = getattr(b, "pickup_lat", None)
        plon = getattr(b, "pickup_lon", None)
        if plat is None or plon is None:
            # tenter g\u00e9ocodage si autoris\u00e9 et texte connu
            addr = getattr(b, "pickup_address", None) or getattr(b, "pickup", None)
            got = _geocode_safe_cached(str(addr)) if (_ALLOW_SERVER_GEOCODE and addr) else None
            if got:
                plat, plon = got
            else:
                plat, plon = fallback_lat, fallback_lon
        try:
            b_any: Any = b
            b_any.pickup_lat = float(cast(Any, plat))
            b_any.pickup_lon = float(cast(Any, plon))
        except Exception:
            b_any = cast(Any, b)
            b_any.pickup_lat, b_any.pickup_lon = fallback_lat, fallback_lon

        # --- DROPOFF ---
        dlat = getattr(b, "dropoff_lat", None)
        dlon = getattr(b, "dropoff_lon", None)
        if dlat is None or dlon is None:
            addr = getattr(b, "dropoff_address", None) or getattr(b, "dropoff", None)
            got = _geocode_safe_cached(str(addr)) if (_ALLOW_SERVER_GEOCODE and addr) else None
            if got:
                dlat, dlon = got
            else:
                # si pas d'adresse ou \u00e9chec g\u00e9ocode, fallback = pickup (plus r\u00e9aliste que centre-ville)
                dlat, dlon = b.pickup_lat, b.pickup_lon
        try:
            b_any = cast(Any, b)
            b_any.dropoff_lat = float(cast(Any, dlat))
            b_any.dropoff_lon = float(cast(Any, dlon))
        except Exception:
            b_any = cast(Any, b)
            b_any.dropoff_lat, b_any.dropoff_lon = b_any.pickup_lat, b_any.pickup_lon

def enrich_driver_coords(drivers: List[Driver], company: Company) -> None:
    """
    Remplit d.current_lat / d.current_lon (attributs \u00e9ph\u00e9m\u00e8res).
    - Utilise driver.latitude/longitude si pr\u00e9sentes ET r\u00e9centes (last_position_update frais).
    - Sinon, fallback sur les coords de l'entreprise, sinon Gen\u00e8ve.
    """
    now = now_local()
    default_latlon = _company_latlon(company)

    for d in drivers:
        lat = getattr(d, "latitude", None)
        lon = getattr(d, "longitude", None)

        # Fra\u00eecheur de la position
        fresh = False
        ts = getattr(d, "last_position_update", None)
        if ts is not None:
            try:
                ts_local = parse_local_naive(cast(Any, ts))
                fresh = bool(ts_local) and (now - cast(datetime, ts_local)) <= _POS_TTL
            except Exception:
                fresh = False

        # Si pas de coord. ou trop ancien -> fallback entreprise (ou Gen\u00e8ve)
        if lat is None or lon is None or not fresh:
            lat, lon = default_latlon

        # cast safe
        try:
            d.current_lat = float(cast(Any, lat))
            d.current_lon = float(cast(Any, lon))
        except Exception:
            d.current_lat = float(default_latlon[0])
            d.current_lon = float(default_latlon[1])

# ============================================================
# 3\ufe0f\u20e3 Matrice de temps / distances
# ============================================================

def build_time_matrix(
    bookings: List[Booking],
    drivers: List[Driver],
    settings=DEFAULT_SETTINGS,
    for_date: str | None = None
) -> Tuple[List[List[int]], List[Tuple[float, float]]]:
    """
    Construit la matrice de temps en minutes entre chaque point
    (drivers start \u2192 pickups \u2192 dropoffs).
    Retourne (time_matrix_minutes, coords_list).
    """
    coords: List[Tuple[float, float]] = []

    def _safe_tuple(lat, lon) -> Tuple[float, float]:
        try:
            return (float(lat), float(lon))
        except Exception:
            return (46.2044, 6.1432)  # Gen\u00e8ve

    # Points de d\u00e9part chauffeurs
    for d in drivers:
        lat = getattr(d, "current_lat", getattr(d, "latitude", None))
        lon = getattr(d, "current_lon", getattr(d, "longitude", None))
        if lat is None or lon is None:
            lat, lon = 46.2044, 6.1432
        coords.append(_safe_tuple(cast(Any, lat), cast(Any, lon)))

    # Pickups & dropoffs
    for b in bookings:
        plat = getattr(b, "pickup_lat", None)
        plon = getattr(b, "pickup_lon", None)
        dlat = getattr(b, "dropoff_lat", None)
        dlon = getattr(b, "dropoff_lon", None)
        if plat is None or plon is None:
            plat, plon = 46.2044, 6.1432
        if dlat is None or dlon is None:
            dlat, dlon = plat, plon
        coords.append(_safe_tuple(cast(Any, plat), cast(Any, plon)))
        coords.append(_safe_tuple(cast(Any, dlat), cast(Any, dlon)))

    n = len(coords)
    if n == 0:
        return [], []
    if n == 1:
        return [[0]], coords

    provider = (settings.matrix.provider or "haversine").lower()
    # Param\u00e8tres cache/OSRM
    coord_prec = int(getattr(getattr(settings, "matrix", None), "coord_precision", 5))
    cache_ttl_s = int(getattr(getattr(settings, "matrix", None), "cache_ttl_s", 900))
    redis_client = _get_redis_from_settings(settings)

    # Cl\u00e9 cache/m\u00e9mo locale (process) pour runs identiques
    coords_canon = _canonical_coords(coords, prec=coord_prec)
    try:
        coords_key = json.dumps(coords_canon, separators=(",", ":"))
    except Exception:
        coords_key = str(coords_canon)

    if provider == "osrm":
        start = time.time()
        try:
            matrix_sec = build_distance_matrix_osrm(
                coords_canon,  # coords arrondies pour stabilit\u00e9 du cache OSRM
                base_url=getattr(settings.matrix, "osrm_url", "http://osrm:5000"),
                profile=getattr(settings.matrix, "osrm_profile", "driving"),
                timeout=int(getattr(settings.matrix, "osrm_timeout_sec", 5)),
                max_sources_per_call=int(getattr(settings.matrix, "osrm_max_sources_per_call", 60)),
                rate_limit_per_sec=int(getattr(settings.matrix, "osrm_rate_limit_per_sec", 8)),
                max_retries=int(getattr(settings.matrix, "osrm_max_retries", 2)),
                backoff_ms=int(getattr(settings.matrix, "osrm_retry_backoff_ms", 250)),
                redis_client=redis_client,
                cache_ttl_s=cache_ttl_s,
                coord_precision=coord_prec,
            )
            # Validation de forme : OSRM doit renvoyer une matrice n x n
            if not matrix_sec or len(matrix_sec) != n or any(len(r) != n for r in matrix_sec):
                raise ValueError(f"OSRM returned invalid matrix shape (got {len(matrix_sec)} rows for n={n})")
        except Exception as e:
            logger.warning("[Dispatch] OSRM matrix failed \u2192 fallback haversine: %s", e)
            avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
            matrix_sec = _haversine_matrix_cached(coords_key, avg_kmh)
        finally:
            dur_ms = int((time.time() - start) * 1000)
            logger.info("[Dispatch] build_time_matrix provider=osrm n=%d redis=%s ttl=%ds prec=%d duration_ms=%d",
                        n, "on" if redis_client else "off", cache_ttl_s, coord_prec, dur_ms)

    else:
        avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
        matrix_sec = _haversine_matrix_cached(coords_key, avg_kmh)
        logger.info("[Dispatch] build_time_matrix provider=haversine n=%d", n)

    # sec -> minutes (ceil), min=1 hors diagonale
    time_matrix_min: List[List[int]] = []
    for i, row in enumerate(matrix_sec):
        row_min: List[int] = []
        for j, t in enumerate(row):
            if i == j:
                row_min.append(0)
            else:
                try:
                    minutes = int(max(2, math.ceil(float(t) / 60.0)))
                except Exception:
                    minutes = 2
                row_min.append(minutes)
        time_matrix_min.append(row_min)

    return time_matrix_min, coords

def _to_minutes_window(win: Any, t0: datetime, horizon: int) -> tuple[int, int]:
    """
    Convertit ce que renvoie driver_work_window_from_config en (start_min, end_min) relatifs à t0.
    Accepte :
      - (start_min:int, end_min:int, *_)
      - (start_dt:datetime, end_dt:datetime)
      - sinon fallback (0, horizon)
    """
    try:
        # 3-tuple ou 2-tuple d'ints
        s = win[0]
        e = win[1]
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            return int(s), int(e)
    except Exception:
        pass
    if isinstance(win, tuple) and len(win) == 2 and all(isinstance(x, datetime) for x in win):
        sdt, edt = win  # type: ignore[assignment]
        s = max(0, int(((sdt - t0).total_seconds()) // 60))
        e = max(s + 1, int(((edt - t0).total_seconds()) // 60))
        return min(s, horizon), min(e, horizon)
    return 0, horizon


def _build_distance_matrix_haversine(coords: List[Tuple[float,float]], avg_speed_kmh: float = 25.0) -> List[List[float]]:
    """
    Fallback Haversine (distances en SECONDES estim\u00e9es, vitesse moyenne ~25 km/h par d\u00e9faut).
    """
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

    n = len(coords)
    if n < 2:
        logger.warning(f"[Dispatch] _build_distance_matrix_haversine: need at least 2 coordinates, got {n}")
        return [[0.0] * max(n, 1) for _ in range(max(n, 1))]

    matrix = [[0.0]*n for _ in range(n)]
    avg_speed_kmh = max(5.0, float(avg_speed_kmh))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
                continue
            try:
                dist_km = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                time_hr = dist_km / avg_speed_kmh
                matrix[i][j] = time_hr * 3600  # sec
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"[Dispatch] Error calculating haversine distance for coords {i},{j}: {e}")
                matrix[i][j] = 3600.0  # 1 hour default fallback

    logger.info(f"[Dispatch] Generated haversine matrix {n}x{n} with avg_speed={avg_speed_kmh} km/h")
    return matrix

# ============================================================
# 4\ufe0f\u20e3 Construction VRPTW problem dict
# ============================================================

def build_vrptw_problem(
    company: Company,
    bookings: List[Booking],
    drivers: List[Driver],
    settings=DEFAULT_SETTINGS,
    *,
    base_time=None,
    for_date: str | None = None,
) -> Dict[str, Any]:
    """
    Construit le dict de donn\u00e9es pour OR-Tools (VRPTW), minutes partout.
    """

    def _clamp_range(s: int | None, e: int | None, horizon: int) -> tuple[int, int]:
        s = 0 if s is None else int(s)
        e = horizon if e is None else int(e)
        # 👇 clamp BOTH start and end
        s = max(0, min(s, horizon - 1))
        e = max(0, min(e, horizon))
        if e <= s:
            # assure une fenêtre minimale de 1 minute dans la borne
            s = max(0, min(s, horizon - 1))
            e = min(horizon, s + 1)
        return s, e


    # 0) \u00c9quit\u00e9
    fairness_counts = count_assigned_bookings_for_day(
        int(cast(Any, company.id)),
        [int(cast(Any, d.id)) for d in drivers],
    )

    # 1) Matrice temps (en minutes) + coordonn\u00e9es ordonn\u00e9es
    time_matrix_min, coords_list = build_time_matrix(
        bookings=bookings,
        drivers=drivers,
        settings=settings,
        for_date=for_date,
    )
    # Harmonise les noms utilis\u00e9s plus loin (et dans le return)
    time_matrix = time_matrix_min
    coords = coords_list

    num_vehicles = len(drivers)
    starts = list(range(num_vehicles))
    ends   = list(range(num_vehicles))

    # 2) Fen\u00eatres de temps et temps de service (minutes)
    time_windows: list[tuple[int, int]] = []
    service_times: list[int] = []
    pair_min_gaps: list[int] = []

    # mapping des noms de champs TimeSettings
    horizon = 1440 if for_date else int(getattr(settings.time, "horizon_min", 480))
    # parse_local_naive(...) peut renvoyer None -> fallback immédiat
    t0 = parse_local_naive(base_time) or now_local()

    for i, b in enumerate(bookings):

        # Index des n\u0153uds pickup et dropoff dans la matrice de temps
        p_node_idx = num_vehicles + (i * 2)
        d_node_idx = p_node_idx + 1

        # R\u00e9cup\u00e9rer la dur\u00e9e du trajet (pickup -> dropoff) depuis la matrice
        # Fallback \u00e0 30 min si la matrice a un probl\u00e8me inattendu.
        trip_duration_min = (
            time_matrix[p_node_idx][d_node_idx]
            if p_node_idx < len(time_matrix) and d_node_idx < len(time_matrix[p_node_idx])
            else 30
        )

        # - Le service au pickup reste "court" (embarquement)
        # - La contrainte de dur\u00e9e (trajet + buffer post-trip) est impos\u00e9e explicitement
        #   dans le solveur via pair_min_gaps.
        pickup_service_time = max(int(getattr(settings.time, "pickup_service_min", 3)), 0)

        # Enregistrer les temps de service (pickup puis dropoff)
        service_times.append(int(pickup_service_time))
        service_times.append(int(getattr(settings.time, "dropoff_service_min", 3)))

        # min_gap : trajet + buffer (et on ajoute la marge de service pickup + 1 min pour \u00e9viter l'\u00e9galit\u00e9 stricte)
        post_buf = int(getattr(settings.time, "post_trip_buffer_min", 15))
        min_gap = int(trip_duration_min) + post_buf
        min_gap = max(min_gap, int(getattr(settings.time, "pickup_service_min", 3)) + 1)
        pair_min_gaps.append(max(2, min_gap))

        # \ud83d\udd0d Debug utile: tracer le gap impos\u00e9 (trajet+buffer) par booking
        try:
            logger.debug("[VRPTW] pair_min_gap booking_id=%s trip=%s buffer=%s -> min_gap=%s",
                         getattr(b, "id", None), int(trip_duration_min),
                         post_buf, int(min_gap))
        except Exception:
            pass

        # Calcul des fen\u00eatres horaires (Time Windows)
        scheduled_local = parse_local_naive(cast(Any, getattr(b, "scheduled_time", None))) or t0
        start_min_raw = int((scheduled_local - t0).total_seconds() // 60)

        # Fen\u00eatre du Pickup
        buf = int(getattr(settings.time, "pickup_buffer_min", 5))
        p_start = max(0, start_min_raw - buf)
        p_end = start_min_raw + buf
        time_windows.append(_clamp_range(p_start, p_end, horizon))

        # Fen\u00eatre du Dropoff (large, car le temps de trajet est d\u00e9j\u00e0 dans le service_time)
        d_start = p_start + trip_duration_min
        d_end = d_start + 120  # Fen\u00eatre large pour le dropoff
        time_windows.append(_clamp_range(d_start, d_end, horizon))

    # 3) Fen\u00eatres de travail chauffeurs (minutes)
    driver_windows: list[tuple[int, int]] = []
    for d in drivers:
        win_any = driver_work_window_from_config(getattr(d, "working_config", None))
        s_raw, e_raw = _to_minutes_window(win_any, t0, horizon)
        s_min, e_min = _clamp_range(int(s_raw), int(e_raw), horizon)
        driver_windows.append((int(s_min), int(e_min)))

    # (Optionnel) petits garde-fous
    assert len(time_windows) == 2 * len(bookings), "TW != 2 par booking"
    assert len(service_times) == 2 * len(bookings), "service_times != 2 par booking"

    return {
        "company_id": company.id,
        "bookings": bookings,
        "drivers": drivers,
        "time_matrix": time_matrix,          # unit\u00e9s: minutes (conversion d\u00e9j\u00e0 faite plus haut)
        "time_windows": time_windows,
        "service_times": service_times,
        "pair_min_gaps": pair_min_gaps,
        "starts": starts,
        "ends": ends,
        "num_vehicles": num_vehicles,
        "coords": coords,
        "fairness_counts": fairness_counts,
        "driver_windows": driver_windows,
        "horizon": horizon,
        "base_time": t0,
        "for_date": for_date,
        # ----- facultatif, pratique pour debug/observabilit\u00e9 -----
        "matrix_provider": (settings.matrix.provider or "haversine").lower(),
        "matrix_units": "minutes",
    }


# ============================================================
# 5\ufe0f\u20e3 Fabrique principale pour engine
# ============================================================

def build_problem_data(
    company_id: int,
    settings=DEFAULT_SETTINGS,
    for_date: str | None = None,
    *,
    regular_first: bool = True,
    allow_emergency: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Pipeline complet de construction des donn\u00e9es pour l'engine.
    - S\u00e9lectionne les r\u00e9servations (par journ\u00e9e locale si for_date est fourni, sinon horizon roulant)
    - R\u00e9cup\u00e8re les chauffeurs disponibles
    - Enrichit coordonn\u00e9es (g\u00e9ocodage serveur optionnel via DISPATCH_ALLOW_SERVER_GEOCODE)
    - Construit le probl\u00e8me VRPTW (avec base_time ancr\u00e9 sur la journ\u00e9e planifi\u00e9e)
    """
    company = Company.query.get(company_id)
    if not company:
        raise ValueError(f"Company {company_id} not found")

    # 1) S\u00e9lection des bookings + base_time (rep\u00e8re temporel du probl\u00e8me)
    if for_date:
        bookings = get_bookings_for_day(company_id, for_date)
        # base_time = minuit local du jour demandé (naïf)
        base_time, _ = day_local_bounds(for_date)
    else:
        horizon_min = int(getattr(getattr(settings, "time", None), "horizon_minutes",
                           getattr(getattr(settings, "time", None), "horizon_min", 480)))
        bookings = get_bookings_for_dispatch(company_id, horizon_min)
        base_time = now_local()

    # Log the number of bookings found
    logger.info(f"[Dispatch] Found {len(bookings)} bookings for company {company_id} for date {for_date or 'today'}")

    # 2) Pool de chauffeurs (actifs & disponibles)
    #    Si desired: priorit\u00e9 aux r\u00e9guliers, puis urgences si allow_emergency
    if regular_first:
        regs, emgs = get_available_drivers_split(company_id)
        drivers = regs + (emgs if allow_emergency else [])

        # Log detailed information about the driver selection
        logger.info(f"[Dispatch] Using {len(regs)} regular drivers for company {company_id}")
        if allow_emergency:
            logger.info(f"[Dispatch] Also using {len(emgs)} emergency drivers for company {company_id}")
        else:
            logger.info("[Dispatch] Not using emergency drivers (allow_emergency=False)")
    else:
        drivers = get_available_drivers(company_id)
        logger.info(f"[Dispatch] Using all {len(drivers)} drivers without priority (regular_first=False)")

    # Log the total number of drivers found
    logger.info(f"[Dispatch] Total: {len(drivers)} available drivers for company {company_id}")

    # 3) Garde-fous : si rien \u00e0 traiter, on renvoie un dict vide
    #    Filtrer les bookings termin\u00e9s/annul\u00e9s avant enrichissement
    try:
        from models import BookingStatus as BS  # type: ignore
        completed_vals = set()
        for name in ("COMPLETED", "RETURN_COMPLETED", "CANCELLED", "CANCELED", "REJECTED"):
            if hasattr(BS, name):
                completed_vals.add(getattr(BS, name))
        if completed_vals:
            bookings = [b for b in bookings if getattr(b, "status", None) not in completed_vals]
    except Exception:
        pass

    if not bookings or not drivers:
        reason = "no_bookings" if not bookings else "no_drivers"
        logger.warning(f"[Dispatch] No dispatch possible for company {company_id}: {reason}")
        return {
            "bookings": [],
            "drivers": [],
            "meta": {
                "reason": reason,
                "for_date": for_date,
                "regular_first": bool(regular_first),
                "allow_emergency": bool(allow_emergency),
            },
        }

    # 4) Enrichissement des coordonn\u00e9es (bookings & chauffeurs)
    enrich_booking_coords(bookings, company)
    enrich_driver_coords(drivers, company)

    # 5) Construction du probl\u00e8me VRPTW avec le bon ancrage temporel
    problem = build_vrptw_problem(
        company=company,
        bookings=bookings,
        drivers=drivers,
        settings=settings,
        base_time=base_time,   # \u2b05 ancr\u00e9 sur minuit du jour si for_date
        for_date=for_date,
    )
    # Renseigner quelques m\u00e9ta-infos utiles pour /preview /logs
    try:
        problem["horizon_minutes"] = int(getattr(getattr(settings, "time", None), "horizon_minutes",
                                          getattr(getattr(settings, "time", None), "horizon_min", 480)))
    except Exception:
        problem["horizon_minutes"] = 480
    problem["regular_first"] = bool(regular_first)
    problem["allow_emergency"] = bool(allow_emergency)
    problem["overrides"] = overrides or {}
    return problem


# ============================================================
# 6\ufe0f\u20e3 S\u00e9lection des retours urgents (utilis\u00e9 par engine.run)
# ============================================================
def pick_urgent_returns(problem: Dict[str, Any], settings=DEFAULT_SETTINGS) -> list[int]:
    """
    Renvoie la liste des booking.id correspondant \u00e0 des *retours urgents* :
    - b.is_return == True
    - scheduled_time dans <= threshold minutes (ou d\u00e9j\u00e0 d\u00e9pass\u00e9e)
    Tri par horaire croissant.
    """
    if not problem or "bookings" not in problem:
        return []

    try:
        threshold = int(getattr(getattr(settings, "emergency", None), "return_urgent_threshold_min", 20))
    except Exception:
        threshold = 20

    now = now_local()
    urgent: list[tuple[int, Any]] = []

    for b in problem.get("bookings", []):
        if not getattr(b, "is_return", False):
            continue
        st = getattr(b, "scheduled_time", None)
        st_local = parse_local_naive(cast(Any, st)) if st else None
        if not st_local:
            continue

        delta_min = (st_local - now).total_seconds() / 60.0
        # urgent si dans la fen\u00eatre threshold OU d\u00e9j\u00e0 en retard
        if delta_min <= threshold:
            urgent.append((int(b.id), st_local))

    urgent.sort(key=lambda t: t[1])
    return [bid for (bid, _) in urgent]


# ============================================================
# 7\ufe0f\u20e3 Verrou de dispatch (jour) \u2013 Redis si dispo, sinon threading.Lock
# ============================================================
def acquire_dispatch_lock(company_id: int, day_str: str, ttl_sec: int = 60) -> bool:
    """
    Emp\u00eache deux passes de dispatch simultan\u00e9es pour (company_id, day).
    - Si Redis est configur\u00e9 dans DEFAULT_SETTINGS.matrix (ou REDIS_URL), on utilise un lock Redis (cross-process).
    - Sinon, fallback sur un lock en m\u00e9moire (prot\u00e8ge uniquement le process courant).
    Retourne True si le verrou est acquis, False sinon.
    """
    key = f"dispatch:{company_id}:{day_str}"

    # Tentative avec Redis
    rc = _get_redis_from_settings(DEFAULT_SETTINGS)
    if rc is not None:
        try:
            lock = rc.lock(key, timeout=int(ttl_sec))
            got = lock.acquire(blocking=False)
            if got:
                _dispatch_locks[key] = lock
            return bool(got)
        except Exception:
            logger.warning("[Dispatch] Redis lock failed; falling back to in-process lock.", exc_info=True)

    # Fallback in-process
    lock = _dispatch_locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _dispatch_locks[key] = lock
    return lock.acquire(blocking=False)


def release_dispatch_lock(company_id: int, day_str: str) -> None:
    """Lib\u00e8re le verrou (Redis ou m\u00e9moire) pour (company_id, day)."""
    key = f"dispatch:{company_id}:{day_str}"
    lock = _dispatch_locks.pop(key, None)
    if lock is None:
        return
    try:
        # Objet Redis Lock a aussi release(); idem pour threading.Lock
        lock.release()
    except Exception:
        # On ne veut jamais planter ici
        logger.debug("[Dispatch] lock release noop.", exc_info=True)


# ============================================================
# 8\ufe0f\u20e3 ETA & d\u00e9tection de retard (pour alertes temps r\u00e9el)
# ============================================================
def calculate_eta(
    driver_position: Tuple[float, float],
    destination: Tuple[float, float],
    settings=DEFAULT_SETTINGS
) -> int:
    """
    Calcule un ETA (en secondes) chauffeur -> destination (pickup le plus souvent).
    Essaie OSRM si provider='osrm', sinon fallback Haversine via vitesse moyenne.
    """
    try:
        provider = str(getattr(getattr(settings, "matrix", None), "provider", "haversine")).lower()
        if provider == "osrm":
            coords = [driver_position, destination]
            matrix = build_distance_matrix_osrm(
                coords,
                base_url=getattr(settings.matrix, "osrm_url", "http://osrm:5000"),
                profile=getattr(settings.matrix, "osrm_profile", "driving"),
                timeout=int(getattr(settings.matrix, "osrm_timeout_sec", 5)),
                max_sources_per_call=int(getattr(settings.matrix, "osrm_max_sources_per_call", 60)),
                rate_limit_per_sec=int(getattr(settings.matrix, "osrm_rate_limit_per_sec", 8)),
                max_retries=int(getattr(settings.matrix, "osrm_max_retries", 2)),
                backoff_ms=int(getattr(settings.matrix, "osrm_retry_backoff_ms", 250)),
                redis_client=_get_redis_from_settings(settings),
                cache_ttl_s=int(getattr(settings.matrix, "cache_ttl_s", 900)),
                coord_precision=int(getattr(settings.matrix, "coord_precision", 5)),
            )
            # OSRM retourne une matrice (sec). On prend la case [0][1].
            if matrix and len(matrix) > 0 and len(matrix[0]) > 1:
                return int(max(0, float(matrix[0][1])))
    except Exception as e:
        logger.warning("[ETA] OSRM failed \u2192 fallback haversine: %s", e)

    # Fallback Haversine
    avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
    minutes = haversine_minutes(driver_position, destination, avg_kmh=avg_kmh)
    return int(max(0, minutes * 60))


def detect_delay(
    driver_position: Tuple[float, float],
    pickup_coords: Tuple[float, float],
    scheduled_time,
    buffer_minutes: int | None = None,
    settings=DEFAULT_SETTINGS
) -> Tuple[bool, int]:
    """
    D\u00e9tecte si le chauffeur arrivera en retard au pickup.
    - scheduled_time : datetime (na\u00efve locale ou tz-aware) de l'horaire pr\u00e9vu du pickup
    - buffer_minutes : marge de tol\u00e9rance (par d\u00e9faut settings.time.buffer_min)
    Retourne (is_delayed, delay_seconds) o\u00f9 delay_seconds > 0 si retard projet\u00e9.
    """
    if buffer_minutes is None:
        try:
            buffer_minutes = int(getattr(getattr(settings, "time", None), "buffer_min", 5))
        except Exception:
            buffer_minutes = 5

    now = now_local()
    scheduled_local = parse_local_naive(cast(Any, scheduled_time))
    if not scheduled_local:
        return False, 0

    # Temps restant jusqu'au pickup pr\u00e9vu
    remaining_seconds = (scheduled_local - now).total_seconds()

    # ETA actuel chauffeur -> pickup
    eta_seconds = calculate_eta(driver_position, pickup_coords, settings=settings)

    # Retard projet\u00e9 (n\u00e9gatif = avance)
    delay_seconds = int(eta_seconds - remaining_seconds)

    # On consid\u00e8re retard si arriv\u00e9e apr\u00e8s (horaire + buffer)
    is_delayed = delay_seconds > int(buffer_minutes * 60)
    return is_delayed, delay_seconds


def get_next_free_at(dropoff_time: datetime, settings=DEFAULT_SETTINGS) -> datetime:
    """
    Retourne l'instant o\u00f9 le chauffeur redevient dispo apr\u00e8s un dropoff.
    Ajoute le buffer post-trajet (par d\u00e9faut 15 min).
    """
    try:
        buf = int(getattr(getattr(settings, "time", None), "post_trip_buffer_min", 15))
    except Exception:
        buf = 15
    return dropoff_time + timedelta(minutes=buf)
