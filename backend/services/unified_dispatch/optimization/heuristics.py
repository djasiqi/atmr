# backend/services/unified_dispatch/heuristics.py
from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, cast

from cachetools import LRUCache  # pyright: ignore[reportMissingModuleSource]

from models import Booking, BookingStatus, Driver
from services.geolocation_service import get_geolocation_service
from services.unified_dispatch.settings import Settings
from shared.constants import DispatchHeuristicsConstants, GeoConstants
from shared.time_utils import minutes_from_now, now_local, sort_key_utc, to_utc

# ✅ REFACTORING: Utilisation de GeolocationService pour centraliser les calculs de distance

# Alias pour compatibilité avec le code existant
AVG_KMH_ZERO = DispatchHeuristicsConstants.AVG_KMH_ZERO
DIST_KM_ONE = DispatchHeuristicsConstants.DIST_KM_ONE
MINS_THRESHOLD = DispatchHeuristicsConstants.MINS_THRESHOLD
CNT_ZERO = DispatchHeuristicsConstants.CNT_ZERO
TO_PICKUP_MIN_THRESHOLD = DispatchHeuristicsConstants.TO_PICKUP_MIN_THRESHOLD
SC_ZERO = DispatchHeuristicsConstants.SC_ZERO
CURRENT_LOAD_THRESHOLD = DispatchHeuristicsConstants.CURRENT_LOAD_THRESHOLD
DID_THRESHOLD = DispatchHeuristicsConstants.DID_THRESHOLD
LATENESS_THRESHOLD_MIN = DispatchHeuristicsConstants.LATENESS_THRESHOLD_MIN
FALLBACK_COORD_DEFAULT = GeoConstants.FALLBACK_COORD_DEFAULT
EMERGENCY_PICKUP_NEAR_THRESHOLD = (
    DispatchHeuristicsConstants.EMERGENCY_PICKUP_NEAR_THRESHOLD
)
EMERGENCY_PICKUP_MEDIUM_THRESHOLD = (
    DispatchHeuristicsConstants.EMERGENCY_PICKUP_MEDIUM_THRESHOLD
)
EMERGENCY_PICKUP_FAR_THRESHOLD = (
    DispatchHeuristicsConstants.EMERGENCY_PICKUP_FAR_THRESHOLD
)
EMERGENCY_TRIP_SHORT_THRESHOLD = (
    DispatchHeuristicsConstants.EMERGENCY_TRIP_SHORT_THRESHOLD
)
EMERGENCY_TRIP_MEDIUM_THRESHOLD = (
    DispatchHeuristicsConstants.EMERGENCY_TRIP_MEDIUM_THRESHOLD
)
MAX_FAIRNESS_GAP = DispatchHeuristicsConstants.MAX_FAIRNESS_GAP


def baseline_and_cap_loads(loads: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    """Normalise les charges brutes en retirant la charge minimale (baseline)
    puis en bornant l'écart maximal à MAX_FAIRNESS_GAP.

    Returns:
        tuple(normalized_loads, baseline)
    """
    if not loads:
        return {}, 0

    numeric_loads: Dict[int, int] = {}
    for raw_id, raw_value in loads.items():
        try:
            did = int(raw_id)
        except (TypeError, ValueError):
            continue
        try:
            count = int(raw_value)
        except (TypeError, ValueError):
            count = 0
        numeric_loads[did] = max(count, 0)

    if not numeric_loads:
        return {}, 0

    baseline = min(numeric_loads.values())
    normalized: Dict[int, int] = {}
    for did, load in numeric_loads.items():
        diff = max(0, load - baseline)
        normalized[did] = min(diff, MAX_FAIRNESS_GAP)
    return normalized, baseline


def _normalized_loads(loads: Dict[int, int]) -> Dict[int, int]:
    """Normalise les charges en retirant la charge minimale et en bornant
    l'écart maximal.

    Cela évite que des historiques trop élevés déséquilibrent la répartition courante :
    seules les différences dans la fenêtre MAX_FAIRNESS_GAP sont conservées.
    """
    if not loads:
        return {}
    min_load = min(loads.values())
    normalized: Dict[int, int] = {}
    for did, load in loads.items():
        diff = load - min_load
        diff = max(diff, 0)
        normalized[did] = min(diff, MAX_FAIRNESS_GAP)
    return normalized


PREFERRED_EXTRA_GAP = 1  # Marge supplémentaire autorisée pour le chauffeur préféré

DEFAULT_SETTINGS = Settings()

# Constantes pour parallélisation
PARALLEL_MIN_BOOKINGS = DispatchHeuristicsConstants.PARALLEL_MIN_BOOKINGS
PARALLEL_MIN_DRIVERS = 5
PARALLEL_MAX_WORKERS = 32

# ✅ P1: Optimisations performance pour heuristique
# Seuil de distance max pour pré-filtrage (km) - drivers trop éloignés exclus
HEURISTIC_MAX_DISTANCE_KM = float(os.getenv("UD_HEURISTIC_MAX_DISTANCE_KM", "50.0"))
# Seuil de score optimal pour early exit (si score >= seuil, arrêter recherche)
HEURISTIC_OPTIMAL_SCORE_THRESHOLD = float(
    os.getenv("UD_HEURISTIC_OPTIMAL_SCORE_THRESHOLD", "0.95")
)
# Cache LRU pour scores (max 500 entrées)
_HEURISTIC_SCORE_CACHE: LRUCache[
    str, Tuple[float, Dict[str, float], Tuple[int, int]]
] = LRUCache(maxsize=500)
_HEURISTIC_SCORE_CACHE_LOCK = threading.Lock()
# Seuil pour activer parallélisation automatique (même si feature flag désactivé)
HEURISTIC_AUTO_PARALLEL_THRESHOLD = int(
    os.getenv("UD_HEURISTIC_AUTO_PARALLEL_THRESHOLD", "30")
)  # Auto-paralléliser si >30 bookings

logger = logging.getLogger(__name__)


# ✅ A1: Compteur thread-safe pour conflits temporels
class TemporalConflictCounter:
    """Compteur thread-safe pour les conflits temporels détectés."""

    _instance: TemporalConflictCounter | None = None

    def __init__(self) -> None:
        super().__init__()
        self._counter: int = 0

    @classmethod
    def get_instance(cls) -> TemporalConflictCounter:
        """Retourne l'instance singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        """Réinitialise le compteur."""
        self._counter = 0

    def increment(self) -> None:
        """Incrémente le compteur."""
        self._counter += 1

    def get_count(self) -> int:
        """Retourne le nombre total de conflits."""
        return self._counter


def reset_temporal_conflict_counter() -> None:
    """Réinitialise le compteur de conflits temporels."""
    TemporalConflictCounter.get_instance().reset()


def get_temporal_conflict_count() -> int:
    """Retourne le nombre de conflits temporels depuis le dernier reset."""
    return TemporalConflictCounter.get_instance().get_count()


def increment_temporal_conflict_counter() -> None:
    """Incrémente le compteur de conflits temporels."""
    TemporalConflictCounter.get_instance().increment()


def _can_be_pooled(b1: Booking, b2: Booking, settings: Settings) -> bool:
    """Vérifie si deux courses peuvent être regroupées (même pickup, même heure)."""
    if not settings.pooling.enabled:
        return False

    # Vérifier que les deux courses ont scheduled_time
    t1 = getattr(b1, "scheduled_time", None)
    t2 = getattr(b2, "scheduled_time", None)
    if not t1 or not t2:
        return False

    # Vérifier que les heures sont proches
    # (±settings.pooling.time_tolerance_min)
    time_diff_min = abs((t1 - t2).total_seconds() / 60)
    if time_diff_min > settings.pooling.time_tolerance_min:
        return False

    # Vérifier que les pickups sont proches (distance GPS)
    lat1 = getattr(b1, "pickup_lat", None)
    lon1 = getattr(b1, "pickup_lon", None)
    lat2 = getattr(b2, "pickup_lat", None)
    lon2 = getattr(b2, "pickup_lon", None)

    if not all([lat1, lon1, lat2, lon2]):
        # Fallback : comparer les adresses textuellement
        addr1 = getattr(b1, "pickup_location", "").lower().replace(" ", "")
        addr2 = getattr(b2, "pickup_location", "").lower().replace(" ", "")
        # Ignorer les différences mineures (majuscules, espaces)
        return bool(addr1 and addr2 and addr1 == addr2)

    # Calculer la distance GPS (vérifier que ce ne sont pas des None)
    lat1safe = float(lat1) if lat1 is not None else 0
    lon1safe = float(lon1) if lon1 is not None else 0
    lat2safe = float(lat2) if lat2 is not None else 0
    lon2safe = float(lon2) if lon2 is not None else 0
    # ✅ REFACTORING: Utilisation de GeolocationService au lieu d'import direct
    geolocation_service = get_geolocation_service()
    distance_m = geolocation_service.distance_meters(
        lat1safe, lon1safe, lat2safe, lon2safe
    )

    if distance_m <= settings.pooling.pickup_distance_m:
        logger.info(
            (
                "[POOLING] 🚗 Courses #%s et #%s peuvent être regroupées "
                "(même pickup à %.0fm, même heure)"
            ),
            b1.id,
            b2.id,
            distance_m,
        )
        return True

    return False


# ⏱️ Temps de service RÉELS (selon utilisateur) - maintenant paramétrables via settings
# PICKUP_SERVICE_MIN, DROPOFF_SERVICE_MIN, MIN_TRANSITION_MARGIN_MIN, etc.
# sont maintenant accessibles via settings.service_times.* et
# settings.pooling.*

# fenêtres travail par chauffeur
# -------------------------------------------------------------------
# Types de retour
# -------------------------------------------------------------------


@dataclass
class HeuristicAssignment:
    booking_id: int
    driver_id: int
    score: float
    reason: str  # "return_urgent" | "regular_scoring"
    estimated_start_min: int
    estimated_finish_min: int
    breakdown: Dict[str, Any] | None = None  # ✅ A1: Détails de scoring + conflits

    # ✅ B2: Explicabilité des décisions (top-3 alternatives & contributions)
    top_alternatives: List[Dict[str, Any]] | None = None  # Top 3 drivers avec scores
    reason_codes: Dict[str, float] | None = (
        None  # distance, fairness, priority, temporal_conflict
    )
    rl_contribution: float = 0.0  # Contribution RL (alpha)
    heuristic_contribution: float = 0.0  # Contribution heuristique (1-alpha)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation compatible avec le contrat Assignment côté API.
        - 'estimated_*' sont renvoyés en datetimes ISO basés sur
          'now_local()' + minutes estimées.
        - 'status' = 'proposed' (l'état final persiste après apply_assignments()).
        """
        base = now_local()
        try:
            est_pickup_dt = base + timedelta(minutes=int(self.estimated_start_min))
            est_drop_dt = base + timedelta(minutes=int(self.estimated_finish_min))
        except Exception:
            est_pickup_dt = base
            est_drop_dt = base
        return {
            "booking_id": int(self.booking_id),
            "driver_id": int(self.driver_id),
            "status": "proposed",
            "estimated_pickup_arrival": est_pickup_dt,
            "estimated_dropoff_arrival": est_drop_dt,
            # Champs facultatifs, utiles au debug
            "score": float(self.score),
            "reason": self.reason,
        }


@dataclass
class HeuristicResult:
    assignments: List[HeuristicAssignment]
    unassigned_booking_ids: List[int]
    debug: Dict[str, Any]


# -------------------------------------------------------------------
# Utilitaires internes
# -------------------------------------------------------------------


def haversine_minutes(
    a: Tuple[float, float],
    b: Tuple[float, float],
    avg_kmh: float = 40,
    *,
    min_minutes: int = 1,
    max_minutes: int | None = None,
    fallback_speed_kmh: float = 30,
) -> int:
    """Estime le temps de trajet (en minutes, arrondi à l'entier supérieur) entre
    deux coordonnées (lat, lon) en utilisant la formule de Haversine et une
    vitesse moyenne `avg_kmh`.

    - Clamp les lat/lon dans les bornes valides.
    - Gère les vitesses non valides (0/NaN/inf) via `fallback_speed_kmh`.
    - Applique un plancher `min_minutes` (par défaut 1) et un plafond
      optionnel `max_minutes`.

    Args:
        a, b: (latitude, longitude) en degrés.
        avg_kmh: vitesse moyenne supposée.
        min_minutes: minute minimale retournée.
        max_minutes: minute maximale retournée (None = pas de plafond).
        fallback_speed_kmh: vitesse utilisée si `avg_kmh` est invalide.

    Returns:
        int: minutes estimées (>= min_minutes, et <= max_minutes si fourni).

    """
    lat1, lon1 = float(a[0]), float(a[1])
    lat2, lon2 = float(b[0]), float(b[1])

    # Clamp des valeurs (robustesse face à des données bruitées)
    lat1 = max(-90, min(90, lat1))
    lat2 = max(-90, min(90, lat2))
    lon1 = ((lon1 + 180) % 360) - 180  # normalise dans [-180, 180)
    lon2 = ((lon2 + 180) % 360) - 180

    # Sécurité vitesse
    if not (math.isfinite(avg_kmh) and avg_kmh > AVG_KMH_ZERO):
        avg_kmh = fallback_speed_kmh
    if not (math.isfinite(avg_kmh) and avg_kmh > AVG_KMH_ZERO):
        # Ultime garde-fou
        avg_kmh = 30

    # ✅ REFACTORING: Utilisation de GeolocationService au lieu d'import direct
    geolocation_service = get_geolocation_service()
    dist_km = geolocation_service.distance_km(lat1, lon1, lat2, lon2)

    # Si quasi le même point, temps minimal
    if dist_km < DIST_KM_ONE - 3:  # ~DIST_KM_ONE mètre
        minutes = 0
    else:
        time_hours = dist_km / avg_kmh
        minutes = math.ceil(time_hours * 60)

    # Appliquer plancher/plafond
    minutes = max(min_minutes, minutes)
    if max_minutes is not None:
        minutes = min(max_minutes, minutes)

    return minutes


def _py_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _current_driver_id(b: Booking) -> int | None:
    return _py_int(getattr(b, "driver_id", None))


def _driver_current_coord(d: Driver) -> Tuple[Tuple[float, float], float]:
    factor = float(getattr(d, "_coord_quality_factor", 1.0) or 1.0)
    coord: Tuple[float, float] | None = None

    cur_lat = getattr(d, "current_lat", None)
    cur_lon = getattr(d, "current_lon", None)
    if cur_lat is not None and cur_lon is not None:
        try:
            coord = (float(cur_lat), float(cur_lon))
        except Exception:
            coord = None

    if coord is None:
        lat = getattr(d, "latitude", None)
        lon = getattr(d, "longitude", None)
        if lat is not None and lon is not None:
            try:
                coord = (float(lat), float(lon))
            except Exception:
                coord = None

    if coord is None:
        coord = FALLBACK_COORD_DEFAULT
        factor = min(factor, 0.5)

    factor = max(0.2, min(factor, 1.0))
    return coord, factor


def _booking_coords(b: Booking) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def _extract(lat_value: Any, lon_value: Any) -> Tuple[float, float]:
        try:
            lat = float(lat_value) if lat_value is not None else float("nan")
            lon = float(lon_value) if lon_value is not None else float("nan")
            if math.isnan(lat) or math.isnan(lon):
                raise ValueError("nan coordinate")
            return (lat, lon)
        except Exception:
            return FALLBACK_COORD_DEFAULT  # Genève par défaut

    pickup_coord = _extract(
        getattr(b, "pickup_lat", None), getattr(b, "pickup_lon", None)
    )
    dropoff_coord = _extract(
        getattr(b, "dropoff_lat", None), getattr(b, "dropoff_lon", None)
    )
    return (pickup_coord, dropoff_coord)


def _is_booking_assigned(b: Booking) -> bool:
    try:
        s = cast("Any", getattr(b, "status", None))
        # compare à l'enum (ou à sa value) pour éviter ColumnElement
        return (s == BookingStatus.ASSIGNED) or (
            getattr(s, "value", None) == BookingStatus.ASSIGNED.value
        )
    except Exception:
        return False


def _priority_weight(b: Booking, weights: Dict[str, float]) -> float:
    """Calcule une "priorité" contextuelle :
    - médical/hôpital => +,
    - VIP/fragile (si vous avez un flag) => +,
    - retard potentiel (pickup imminent) => +,
    - retour déclenché à la demande => + léger (l'urgent est géré à part).
    """
    score: float = 0.0

    # Exemples de signaux - adaptez selon vos champs réels:
    if getattr(b, "medical_facility", None):
        score += weights.get("medical", 0.6)

    if getattr(b, "hospital_service", False):
        score += weights.get("hospital", 0.4)

    # retard potentiel
    mins = minutes_from_now(getattr(b, "scheduled_time", None))
    if mins <= MINS_THRESHOLD:
        score += weights.get("time_pressure", 0.5)
    elif mins <= MINS_THRESHOLD:
        score += weights.get("time_pressure", 0.2)

    # retour (non urgent) => léger bonus
    if getattr(b, "is_return", False):
        score += weights.get("return_generic", 0.1)

    # ⭐ Bonus VIP client (fonctionnalité optionnelle)
    # Si le modèle Client a un champ `is_vip` ou `priority_level`, ajoutez :
    # if getattr(b.client, "is_vip", False):
    #     score += weights.get("vip_client", 0.3)

    return score


def _is_return_urgent(b: Booking, settings: Settings) -> bool:
    if not getattr(b, "is_return", False):
        return False
    mins = minutes_from_now(getattr(b, "scheduled_time", None))
    # compat: certains settings utilisent emergency_threshold_min
    thr = cast(
        "Any",
        getattr(
            settings.emergency,
            "return_urgent_threshold_min",
            getattr(settings.emergency, "emergency_threshold_min", 30),
        ),
    )
    return mins <= int(thr)


def _driver_fairness_penalty(driver_id: int, fairness_counts: Dict[int, int]) -> float:
    """Plus le chauffeur a déjà de courses (dans la fenêtre équité), plus la
    pénalité augmente.
    """
    normalized = _normalized_loads(fairness_counts)
    diff = normalized.get(driver_id, 0)
    if diff <= 0:
        return 0.0
    # Échelle agressive : diff=1 → 0.35, diff=2 → 0.7 (fort découragement)
    penalty = 0.35 * diff
    return min(0.8, penalty)


def _regular_driver_bonus(b: Booking, d: Driver) -> float:
    """Bonus si le driver est "régulier" du client (ex: même driver_id référencé
    sur les dernières courses du client). Ici placeholder: si already assigned
    au même chauffeur, neutre (on évite de casser la relation).
    """
    try:
        bid_raw = cast("Any", getattr(b, "driver_id", None))
        did_raw = cast("Any", getattr(d, "id", None))
        bid = int(bid_raw) if bid_raw is not None else None
        did = int(did_raw) if did_raw is not None else None
    except Exception:
        return 0
    if bid is not None and did is not None and bid == did:
        return 0.15
    return 0


def _check_driver_window_feasible(
    driver_window: Tuple[int, int], est_start_min: int
) -> bool:
    start_w, end_w = driver_window

    # ⚠️ CORRECTION CRITIQUE : driver_window (0-1440) représente la journée du chauffeur
    # mais est_start_min est en "minutes depuis maintenant"
    # Pour les courses futures (demain+), la fenêtre d'aujourd'hui ne s'applique PAS
    # → On accepte toujours les courses qui sont dans le futur (planning à l'avance)

    # Si la course commence après la fin de la fenêtre (après minuit), c'est
    # pour demain → accepter
    if est_start_min > end_w:
        return True

    # Si la course finit après la fenêtre mais commence dedans, c'est OK
    # (elle chevauche minuit)
    # On vérifie seulement que le début est dans la fenêtre
    return est_start_min >= start_w


# -------------------------------------------------------------------
# Scoring principal
# -------------------------------------------------------------------


def _score_driver_for_booking(
    b: Booking,
    d: Driver,
    driver_window: Tuple[int, int],
    settings: Settings,
    fairness_counts: Dict[int, int],
    company_coords: Tuple[float, float]
    | None = None,  # ⚡ Coordonnées du bureau (lat, lon)
    preferred_driver_id: int
    | None = None,  # ⚡ Chauffeur préféré pour bonus de préférence
    last_dropoff_coord: Tuple[float, float]
    | None = None,  # ⚡ Position de dropoff de la dernière course assignée
) -> Tuple[float, Dict[str, float], Tuple[int, int]]:
    """Renvoie (score_total, breakdown, (est_start_min, est_finish_min))
    - score en [0..1+] (plus est grand, mieux c'est)
    - breakdown : contributions par facteur
    - estimation temps (start/finish) pour quick-feasibility.
    ⚡ NOUVEAU: company_coords pour prioriser la proximité au bureau pour les
      chauffeurs d'urgence.
    ⚡ NOUVEAU: last_dropoff_coord pour prioriser les courses proches de la
      dernière course assignée.

    ✅ P1: Optimisations performance :
    - Cache LRU pour scores similaires
    - Pré-filtrage distance max avant calcul complet
    """
    # ✅ P1: Créer clé de cache pour score (booking + driver + contexte)
    b_id = int(cast("Any", getattr(b, "id", 0)) or 0)
    d_id = int(cast("Any", getattr(d, "id", 0)) or 0)
    cache_key_parts = [
        f"b:{b_id}",
        f"d:{d_id}",
        f"w:{driver_window[0]}-{driver_window[1]}",
        f"f:{hash(str(sorted(fairness_counts.items())))}",
    ]
    if company_coords:
        cache_key_parts.append(f"c:{company_coords[0]:.4f},{company_coords[1]:.4f}")
    if preferred_driver_id:
        cache_key_parts.append(f"p:{preferred_driver_id}")
    if last_dropoff_coord:
        cache_key_parts.append(
            f"l:{last_dropoff_coord[0]:.4f},{last_dropoff_coord[1]:.4f}"
        )

    cache_key = "|".join(cache_key_parts)

    # ✅ P1: Vérifier cache LRU
    with _HEURISTIC_SCORE_CACHE_LOCK:
        if cache_key in _HEURISTIC_SCORE_CACHE:
            logger.debug("[HEURISTIC] ✅ Score cache hit for b=%s d=%s", b_id, d_id)
            return _HEURISTIC_SCORE_CACHE[cache_key]

    # 1) Proximité / coûts temps (paramétrable via settings)
    avg_kmh = getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25)
    # mapping des noms vers TimeSettings actuels
    buffer_min = int(getattr(settings.time, "pickup_buffer_min", 5))
    # ✅ Utiliser settings.service_times (configurables par le client)
    pickup_service = int(getattr(settings.service_times, "pickup_service_min", 5))
    drop_service = int(getattr(settings.service_times, "dropoff_service_min", 10))

    # ✅ Vérifier si le driver est un urgent via driver_type (pas is_emergency)
    driver_type = getattr(d, "driver_type", None)
    driver_type_str = str(driver_type or "").strip().upper()
    if "." in driver_type_str:
        driver_type_str = driver_type_str.split(".")[-1]
    is_emergency = driver_type_str == "EMERGENCY"

    # ⚡ AMÉLIORATION: Utiliser le meilleur des deux points de départ pour le
    # calcul de base
    # Cela évite de pénaliser inutilement les courses quand last_dropoff_coord est loin
    # Puis ajouter un bonus de continuité si last_dropoff_coord est utilisé et proche
    current_coord, driver_quality_factor = _driver_current_coord(d)
    p_coord, d_coord = _booking_coords(b)
    booking_quality_factor = float(getattr(b, "_coord_quality_factor", 1.0) or 1.0)
    coord_quality_factor = max(0.2, min(driver_quality_factor, booking_quality_factor))

    # ✅ P1: Pré-filtrage distance max (évite calculs coûteux pour drivers trop éloignés)
    # ✅ REFACTORING: Utilisation de GeolocationService au lieu d'import direct
    geolocation_service = get_geolocation_service()
    distance_km = geolocation_service.distance_km(
        current_coord[0], current_coord[1], p_coord[0], p_coord[1]
    )

    # ✅ P1: Exclure drivers trop éloignés avant scoring complet
    if distance_km > HEURISTIC_MAX_DISTANCE_KM:
        logger.debug(
            "[HEURISTIC] Driver %s too far (%.1f km > %.1f km) for booking %s",
            d_id,
            distance_km,
            HEURISTIC_MAX_DISTANCE_KM,
            b_id,
        )
        # Retourner score négatif pour indiquer exclusion
        result = (-1.0, {"distance_filtered": 1.0}, (0, 0))
        with _HEURISTIC_SCORE_CACHE_LOCK:
            _HEURISTIC_SCORE_CACHE[cache_key] = result
        return result

    # Initialiser use_last_dropoff_for_bonus
    use_last_dropoff_for_bonus = False

    # Calculer les distances depuis les deux points possibles
    to_pickup_from_current = haversine_minutes(
        current_coord, p_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=180
    )

    to_pickup_from_last_dropoff = 999
    if last_dropoff_coord:
        to_pickup_from_last_dropoff = haversine_minutes(
            last_dropoff_coord, p_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=180
        )

    # Utiliser le point de départ qui donne la distance la plus courte
    # Cela garantit que le prox_score est toujours optimal
    to_pickup_min = to_pickup_from_current
    if last_dropoff_coord and to_pickup_from_last_dropoff < to_pickup_from_current:
        to_pickup_min = to_pickup_from_last_dropoff
        use_last_dropoff_for_bonus = True
    elif is_emergency and company_coords:
        to_pickup_min = haversine_minutes(
            company_coords, p_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=180
        )

    # Estimations robustes (plancher/plafond pour éviter les valeurs extrêmes
    # en heuristique)
    to_drop_min = haversine_minutes(
        p_coord, d_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240
    )

    # ⚡ Pour les chauffeurs d'urgence : bonus pour trajets courts (pickup
    # proche + trajet court)
    emergency_trip_bonus = 0.0
    if is_emergency:
        # Bonus si pickup proche du bureau ET trajet court
        if (
            to_pickup_min <= EMERGENCY_PICKUP_NEAR_THRESHOLD
            and to_drop_min <= EMERGENCY_TRIP_SHORT_THRESHOLD
        ):
            emergency_trip_bonus = (
                0.5  # Fort bonus pour trajets courts depuis le bureau
            )
        elif (
            to_pickup_min <= EMERGENCY_PICKUP_MEDIUM_THRESHOLD
            and to_drop_min <= EMERGENCY_TRIP_MEDIUM_THRESHOLD
        ):
            emergency_trip_bonus = 0.3  # Bonus moyen
        elif to_pickup_min <= EMERGENCY_PICKUP_FAR_THRESHOLD:
            emergency_trip_bonus = 0.1  # Bonus faible

    # Estimations de début/fin (minutes depuis maintenant)
    # ⚠️ IMPORTANT: on doit prendre en compte l'heure réelle de la course
    # (scheduled_time)
    mins_to_booking = minutes_from_now(getattr(b, "scheduled_time", None))
    # Le chauffeur doit arriver au pickup AVANT scheduled_time
    # Pour la faisabilité, on utilise quand le chauffeur ARRIVE au pickup (=
    # scheduled_time)
    est_start_min = max(0, mins_to_booking)
    est_finish_min = est_start_min + pickup_service + to_drop_min + drop_service

    # Pré-faisabilité : fenêtre de travail chauffeur
    # Si on dépasse déjà la fenêtre, inutile d'aller plus loin.
    if not _check_driver_window_feasible(driver_window, est_start_min):
        return (-1, {"feasible": 0}, (est_start_min, est_finish_min))

    # Garde "pickup trop tard" : si le chauffeur ne peut pas arriver à temps
    # (on a déjà mins_to_booking calculé ci-dessus)
    lateness_penalty = 0.6 if to_pickup_min > mins_to_booking + buffer_min else 0

    # 2) Équité (driver_load_balance)
    did_safe = int(cast("Any", getattr(d, "id", 0)) or 0)
    fairness_pen = _driver_fairness_penalty(did_safe, fairness_counts)

    # 3) Priorité booking
    pr = _priority_weight(
        b,
        {
            "medical": 0.6,
            "hospital": 0.4,
            "time_pressure": 0.5,
            "return_generic": 0.1,
        },
    )

    # 4) Regular driver bonus
    reg_bonus = _regular_driver_bonus(b, d)

    # 5) ✅ Bonus pour chauffeur préféré (si configuré)
    preferred_bonus = 0.0
    if preferred_driver_id is not None:
        did_safe = int(cast("Any", getattr(d, "id", 0)) or 0)
        if did_safe == preferred_driver_id:
            # ✅ Fort bonus pour le chauffeur préféré (ajuste le poids selon
            # settings si nécessaire)
            # Bonus de 3.0 = très fort pour prioriser ce chauffeur (surmonte
            # proximité, équité, etc.)
            preferred_bonus = 3.0
            logger.info(
                (
                    "[HEURISTIC] 🎯 Bonus préférence FORT appliqué pour "
                    "chauffeur #%d (+%.1f) booking_id=%s"
                ),
                did_safe,
                preferred_bonus,
                int(cast("Any", getattr(b, "id", 0))),
            )

    # Normalisations simples
    # Proximité -> transformer to_pickup_min en score (0..1)
    # 0-5 min ~ 1 ; 30min+ ~ 0
    if to_pickup_min <= TO_PICKUP_MIN_THRESHOLD:
        prox_score: float = 1.0
    elif to_pickup_min >= TO_PICKUP_MIN_THRESHOLD:
        prox_score = 0.0
    else:
        prox_score = max(0, 1 - (to_pickup_min - 5) / 25)
    prox_score *= coord_quality_factor

    # ⚡ Bonus de continuité géographique si last_dropoff_coord est utilisé
    # Cela récompense les courses qui minimisent les trajets entre courses consécutives
    # Seuils étendus et bonus augmentés pour avoir un impact significatif
    CONTINUITY_BONUS_NEAR_MIN = 15  # Distance en minutes pour bonus fort
    CONTINUITY_BONUS_MEDIUM_MIN = 30  # Distance en minutes pour bonus moyen
    CONTINUITY_BONUS_FAR_MIN = 45  # Distance en minutes pour bonus faible
    CONTINUITY_BONUS_VERY_FAR_MIN = 60  # Distance en minutes pour bonus très faible
    CONTINUITY_BONUS_NEAR = 0.5  # Bonus fort pour courses très proches
    CONTINUITY_BONUS_MEDIUM = 0.3  # Bonus moyen
    CONTINUITY_BONUS_FAR = 0.2  # Bonus faible
    CONTINUITY_BONUS_VERY_FAR = 0.1  # Bonus très faible

    continuity_bonus = 0.0
    # ⚡ Bonus de continuité seulement si last_dropoff_coord est utilisé ET
    # proche
    # On utilise to_pickup_from_last_dropoff pour le bonus (pas to_pickup_min
    # qui peut venir de current_coord)
    if use_last_dropoff_for_bonus and last_dropoff_coord:
        # Bonus décroissant avec la distance depuis last_dropoff : 0-15min =
        # +0.5, 15-30min = +0.3, 30-45min = +0.2, 45-60min = +0.1
        if to_pickup_from_last_dropoff <= CONTINUITY_BONUS_NEAR_MIN:
            continuity_bonus = CONTINUITY_BONUS_NEAR
        elif to_pickup_from_last_dropoff <= CONTINUITY_BONUS_MEDIUM_MIN:
            continuity_bonus = CONTINUITY_BONUS_MEDIUM
        elif to_pickup_from_last_dropoff <= CONTINUITY_BONUS_FAR_MIN:
            continuity_bonus = CONTINUITY_BONUS_FAR
        elif to_pickup_from_last_dropoff <= CONTINUITY_BONUS_VERY_FAR_MIN:
            continuity_bonus = CONTINUITY_BONUS_VERY_FAR
        # Au-delà de 60min, pas de bonus (trop loin de la dernière dropoff)

    # Agrégation pondérée
    w = settings.heuristic  # déjà normalisé
    base = (
        prox_score * w.proximity
        + (1 - fairness_pen) * w.driver_load_balance
        + pr * w.priority
        + reg_bonus * w.regular_driver_bonus
    )
    # Urgence "non-critique" déjà dans pr via return_generic
    # Appliquer malus de retard potentiel
    heuristic_score = max(0, base - lateness_penalty)

    # ⚡ Ajouter le bonus de continuité géographique (ajouté après pour avoir
    # un impact fort)
    heuristic_score += continuity_bonus

    # ⚡ Bonus pour chauffeurs d'urgence avec trajets courts depuis le bureau
    if is_emergency:
        heuristic_score += emergency_trip_bonus

    # ✅ Bonus pour chauffeur préféré (ajouté après les autres calculs pour
    # avoir un impact fort)
    heuristic_score += (
        preferred_bonus * 1.0
    )  # Poids fort (1.0) pour prioriser significativement

    breakdown: Dict[str, Any] = {
        "proximity": prox_score * w.proximity,
        "fairness": (1 - fairness_pen) * w.driver_load_balance,
        "priority": pr * w.priority,
        "regular": reg_bonus * w.regular_driver_bonus,
        "preferred_driver_bonus": preferred_bonus
        * 1.0,  # ✅ Ajout du bonus préférence dans le breakdown
        "lateness_penalty": -lateness_penalty,
        "continuity_bonus": continuity_bonus,  # ⚡ Bonus de continuité géographique
        "coord_quality": coord_quality_factor,
    }

    # Fusion avec score RL si activé
    if getattr(settings.features, "enable_rl", False) and getattr(
        settings.features, "enable_rl_apply", False
    ):
        # Normaliser le score heuristique de 0-1 vers 0-100
        heuristic_score_100 = heuristic_score * 100

        # TODO: Récupérer le score RL (à implémenter avec le système RL)
        rl_score = 0.5  # Placeholder: score RL par défaut
        alpha = getattr(settings.rl, "alpha", 0.2)

        from services.unified_dispatch.score_fusion import fuse_scores

        final_score_100, fusion_breakdown = fuse_scores(
            heuristic_score=heuristic_score_100, rl_score=rl_score, alpha=alpha
        )

        # Reconvertir en 0-1
        total = final_score_100 / 100

        # Ajouter le breakdown de fusion
        breakdown["rl_fusion"] = fusion_breakdown
        breakdown["heuristic_raw"] = heuristic_score
    else:
        total = heuristic_score

    # ✅ P1: Mettre en cache le résultat
    result = (total, breakdown, (est_start_min, est_finish_min))
    with _HEURISTIC_SCORE_CACHE_LOCK:
        _HEURISTIC_SCORE_CACHE[cache_key] = result

    return result


# -------------------------------------------------------------------
# Parallélisation du scoring
# -------------------------------------------------------------------


def _score_booking_driver_pair(
    b: Booking,
    d: Driver,
    _driver_window: Tuple[int, int],  # Renommé pour indiquer usage intentionnel
    settings: Settings,
    fairness_counts: Dict[int, int],
    _driver_index: Dict[int, int],  # Renommé pour indiquer usage intentionnel
    company_coords: Tuple[float, float] | None = None,  # ⚡ Coordonnées du bureau
    preferred_driver_id: int
    | None = None,  # ⚡ Chauffeur préféré pour bonus de préférence
    last_dropoff_coord: Tuple[float, float]
    | None = None,  # ⚡ Position de dropoff de la dernière course assignée
) -> Tuple[int, int, float, Dict[str, float], Tuple[int, int]]:
    """Score un couple (booking, driver) de manière thread-safe.

    Returns:
        (booking_id, driver_id, score, breakdown, (est_start, est_finish))
    """
    try:
        b_id = int(cast("Any", b.id))
        d_id = int(cast("Any", d.id))
        dw = (
            0,
            24 * 60,
        )  # Default window (driver_window non utilisé dans cette version simplifiée)

        normalized_counts = _normalized_loads(fairness_counts)

        sc, breakdown, time_est = _score_driver_for_booking(
            b,
            d,
            dw,
            settings,
            normalized_counts,
            company_coords=company_coords,
            preferred_driver_id=preferred_driver_id,
            last_dropoff_coord=last_dropoff_coord,
        )

        return (b_id, d_id, sc, breakdown, time_est)
    except Exception as e:
        logger.error("[ParallelScoring] Error scoring b=%s d=%s: %s", b.id, d.id, e)
        return (int(cast("Any", b.id)), int(cast("Any", d.id)), 0.0, {}, (0, 0))


# -------------------------------------------------------------------
# Assignation heuristique
# -------------------------------------------------------------------


def assign(
    problem: Dict[str, Any], settings: Settings = DEFAULT_SETTINGS
) -> HeuristicResult:
    """Algorithme glouton :
    1) Traite en premier les "retours urgents".
    2) Trie le reste par scheduled_time croissante puis score décroissant.
    3) Respecte un plafond global par chauffeur
      (settings.solver.max_bookings_per_driver).
    4) Évite les réassignations inutiles (ASSIGNED au même driver).
    """
    if not problem:
        return HeuristicResult(
            assignments=[], unassigned_booking_ids=[], debug={"reason": "empty_problem"}
        )

    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts_raw: Dict[int, int] = problem.get("fairness_counts", {})
    fairness_counts, fairness_baseline = baseline_and_cap_loads(fairness_counts_raw)
    problem["fairness_counts"] = fairness_counts
    problem["fairness_baseline"] = fairness_baseline
    company_coords: Tuple[float, float] | None = problem.get(
        "company_coords"
    )  # ⚡ Coordonnées du bureau
    driver_load_multipliers: Dict[int, float] = problem.get(
        "driver_load_multipliers", {}
    )  # ⚡ Multiplicateurs de charge par chauffeur
    preferred_driver_id: int | None = problem.get(
        "preferred_driver_id"
    )  # ⚡ Chauffeur préféré

    # Log pour debug
    total_fairness = sum(fairness_counts.values())
    max_fairness = max(fairness_counts.values()) if fairness_counts else 0
    non_zero_fairness = {k: v for k, v in fairness_counts.items() if v}
    logger.info(
        (
            "[HEURISTIC] 🎯 assign() entry: preferred_driver_id=%s, "
            "bookings=%d, drivers=%d, fairness_total=%d, fairness_max=%d, "
            "map=%s"
        ),
        preferred_driver_id,
        len(bookings),
        len(drivers),
        total_fairness,
        max_fairness,
        non_zero_fairness or "{}",
    )
    if preferred_driver_id:
        # ✅ P1: Utiliser set pour vérification O(1) au lieu de O(n)
        driver_ids = {int(cast("Any", d.id)) for d in drivers}
        logger.info(
            "[HEURISTIC] 🎯 Chauffeur préféré %s dans drivers disponibles: %s",
            preferred_driver_id,
            preferred_driver_id in driver_ids,
        )
        logger.info(
            "[HEURISTIC] 🎯 Chauffeur préféré détecté dans le problème: %s",
            preferred_driver_id,
        )
    if company_coords:
        logger.debug(
            "[HEURISTIC] 📍 Coordonnées bureau disponibles: (%s, %s)",
            company_coords[0],
            company_coords[1],
        )

    # 📅 Récupérer les états précédents depuis problem (ou initialiser à zéro)
    previous_busy = problem.get("busy_until", {})
    previous_times = problem.get("driver_scheduled_times", {})
    previous_load = problem.get("proposed_load", {})

    # État local : nombre d'assignations *proposées* dans cette passe (ids
    # castés en int)
    proposed_load: Dict[int, int] = {
        int(cast("Any", d.id)): previous_load.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }
    fairness_effective: Dict[int, int] = {
        int(cast("Any", d.id)): fairness_counts.get(int(cast("Any", d.id)), 0)
        + proposed_load.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }
    driver_index: Dict[int, int] = {
        int(cast("Any", d.id)): i for i, d in enumerate(drivers)
    }

    max_cap = settings.solver.max_bookings_per_driver

    # ⚡ Calculer les caps ajustés selon les préférences de charge par chauffeur
    def get_adjusted_max_cap(driver_id: int) -> int:
        """Retourne le cap maximum ajusté pour un chauffeur selon ses préférences."""
        multiplier = driver_load_multipliers.get(driver_id, 1.0)
        return int(max_cap * multiplier)

    # ⚡ Fonction helper pour obtenir les chauffeurs éligibles selon équité
    # stricte ou préférence
    def get_eligible_drivers(
        all_drivers: List[Driver], current_loads: Dict[int, int]
    ) -> List[Driver]:
        """Retourne la liste des chauffeurs éligibles selon la préférence ou
        l'équité stricte.

        ⚡ CORRECTION: Le chauffeur préféré est inclus dans la liste éligible
        avec un bonus de +3.0 dans le scoring, plutôt que d'être exclusivement
        sélectionné. Cela permet au bonus de prioriser le préféré tout en
        gardant la flexibilité pour d'autres assignations si nécessaire.
        """
        # Équité stricte : filtrer selon MAX_FAIRNESS_GAP
        if not current_loads:
            return all_drivers

        min_load = min(current_loads.values())

        # Priorité absolue aux chauffeurs avec la charge minimale
        eligible = [
            d
            for d in all_drivers
            if current_loads.get(int(cast("Any", d.id)), 0) == min_load
        ]

        # Si tout le monde a déjà au moins min_load+1, élargir
        # progressivement jusqu'à MAX_FAIRNESS_GAP
        gap = 1
        while not eligible and gap <= MAX_FAIRNESS_GAP:
            eligible = [
                d
                for d in all_drivers
                if current_loads.get(int(cast("Any", d.id)), 0) <= min_load + gap
            ]
            gap += 1

        # Si malgré tout aucun chauffeur n'est éligible (cas extrême),
        # retourner la liste complète
        if not eligible:
            eligible = all_drivers

        max_allowed_for_log = min_load + MAX_FAIRNESS_GAP
        preferred_gap_limit = max_allowed_for_log + PREFERRED_EXTRA_GAP

        # ⚡ CORRECTION: Si un chauffeur préféré est défini, l'inclure dans
        # la liste éligible
        # Le bonus de +3.0 dans le scoring fera la priorisation
        if preferred_driver_id:
            preferred_driver = next(
                (
                    d
                    for d in all_drivers
                    if int(cast("Any", d.id)) == preferred_driver_id
                ),
                None,
            )
            if preferred_driver:
                preferred_did = int(cast("Any", preferred_driver.id))
                adjusted_cap = get_adjusted_max_cap(preferred_did)
                current_load = current_loads.get(preferred_did, 0)

                # Toujours inclure le préféré s'il est sous le cap et dans la
                # marge d'équité élargie
                if (
                    current_load < adjusted_cap
                    and current_load <= preferred_gap_limit
                    and preferred_driver not in eligible
                ):
                    logger.info(
                        (
                            "[HEURISTIC] 🎯 Ajout chauffeur préféré #%s à la "
                            "liste éligible (load: %d/%d, bonus: +3.0)"
                        ),
                        preferred_did,
                        current_load,
                        adjusted_cap,
                    )
                    eligible.append(preferred_driver)
                elif (
                    current_load < adjusted_cap and current_load <= preferred_gap_limit
                ):
                    logger.debug(
                        (
                            "[HEURISTIC] 🎯 Chauffeur préféré #%s déjà "
                            "éligible (load: %d/%d, bonus: +3.0)"
                        ),
                        preferred_did,
                        current_load,
                        adjusted_cap,
                    )
                else:
                    logger.warning(
                        (
                            "[HEURISTIC] ⚠️ Chauffeur préféré #%s au cap "
                            "(load: %d/%d), bonus non appliqué"
                        ),
                        preferred_did,
                        current_load,
                        adjusted_cap,
                    )
        logger.debug(
            (
                "[HEURISTIC] 📊 Équité stricte: %d chauffeurs éligibles "
                "(min_load: %s, max_allowed: %s)"
            ),
            len(eligible),
            min_load,
            max_allowed_for_log,
        )
        return eligible if eligible else all_drivers

    urgent: List[Booking] = [b for b in bookings if _is_return_urgent(b, settings)]
    urgent_ids = {int(cast("Any", b.id)) for b in urgent}
    regular: List[Booking] = [
        b for b in bookings if int(cast("Any", b.id)) not in urgent_ids
    ]

    # Trier
    urgent.sort(
        key=lambda b: sort_key_utc(cast("Any", getattr(b, "scheduled_time", None)))
    )  # plus proches
    regular.sort(
        key=lambda b: sort_key_utc(cast("Any", getattr(b, "scheduled_time", None)))
    )  # FIFO temporel

    assignments: List[HeuristicAssignment] = []

    # ⚡ AMÉLIORATION: Construire un dictionnaire driver_last_dropoff AVANT le scoring
    # en utilisant les bookings déjà assignés (status=ASSIGNED avec driver_id)
    # Cela permet de minimiser les trajets dès le scoring initial
    # On garde pour chaque chauffeur la dernière course assignée (par scheduled_time)
    driver_last_dropoff_initial: Dict[int, Tuple[float, float]] = {}
    driver_last_booking_time: Dict[
        int, datetime
    ] = {}  # Pour comparer les scheduled_time

    for booking in bookings:
        # Vérifier si le booking est déjà assigné
        booking_driver_id = getattr(booking, "driver_id", None)
        booking_status = getattr(booking, "status", None)
        if booking_driver_id and booking_status == BookingStatus.ASSIGNED:
            did = int(booking_driver_id)
            # Récupérer la position de dropoff de cette course
            _, dropoff_coord = _booking_coords(booking)
            if dropoff_coord:
                booking_scheduled = getattr(booking, "scheduled_time", None)
                # Si le chauffeur n'a pas encore de course, ou si cette
                # course est plus récente
                if did not in driver_last_dropoff_initial:
                    driver_last_dropoff_initial[did] = dropoff_coord
                    if booking_scheduled:
                        driver_last_booking_time[did] = booking_scheduled
                elif booking_scheduled:
                    # Comparer les scheduled_time pour garder la plus récente
                    last_time = driver_last_booking_time.get(did)
                    if last_time is None or booking_scheduled > last_time:
                        driver_last_dropoff_initial[did] = dropoff_coord
                        driver_last_booking_time[did] = booking_scheduled

    # Timeline par chauffeur (en minutes depuis maintenant)
    busy_until: Dict[int, int] = {
        int(cast("Any", d.id)): previous_busy.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }
    # 🆕 Tracker les scheduled_time par chauffeur pour éviter les conflits
    driver_scheduled_times: Dict[int, List[int]] = {
        int(cast("Any", d.id)): list(previous_times.get(int(cast("Any", d.id)), []))
        for d in drivers
    }

    unassigned: List[int] = []
    # ✅ A1: Tracker les rejets de conflits temporels pour observabilité
    temporal_conflict_rejects: List[Dict[str, Any]] = []

    # --- 1) Retours urgents (hard priority) ---
    logger.info("=" * 80)
    logger.info(
        "[DISPATCH HEURISTIC] 🚨 %d retours urgents, %d courses régulières",
        len(urgent),
        len(regular),
    )
    logger.info("[DISPATCH HEURISTIC] 👥 %d chauffeurs disponibles", len(drivers))
    if previous_busy or previous_times or previous_load:
        logger.info(
            "[DISPATCH HEURISTIC] 📥 États récupérés: busy_until=%s, proposed_load=%s",
            busy_until,
            proposed_load,
        )
    logger.info("=" * 80)

    for b in urgent:
        best: Tuple[float, HeuristicAssignment] | None = None
        b_id = int(cast("Any", b.id))
        logger.debug("[DISPATCH] Assignation urgente #$%s...", b_id)

        # ⚡ Calculer les charges actuelles pour tous les chauffeurs
        raw_loads = {
            int(cast("Any", d.id)): fairness_effective.get(int(cast("Any", d.id)), 0)
            for d in drivers
        }
        current_loads = _normalized_loads(raw_loads)
        # ⚡ Filtrer les chauffeurs éligibles selon préférence ou équité stricte
        eligible_drivers = get_eligible_drivers(drivers, current_loads)

        for d in eligible_drivers:
            # Cap par chauffeur (ajusté selon préférences)
            did = int(cast("Any", d.id))
            adjusted_cap = get_adjusted_max_cap(did)
            if fairness_effective.get(did, 0) >= adjusted_cap:
                continue

            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            sc, _, (est_s, est_f) = _score_driver_for_booking(
                b,
                d,
                dw,
                settings,
                current_loads,
                company_coords=company_coords,
                preferred_driver_id=preferred_driver_id,
            )

            # ✅ A1: VALIDATION STRICTE DES CONFLITS TEMPORELS
            # Récupérer min_gap_minutes depuis settings
            min_gap_minutes = int(getattr(settings.safety, "min_gap_minutes", 30))
            post_trip_buffer = int(getattr(settings.safety, "post_trip_buffer_min", 15))
            strict_check = bool(
                getattr(
                    settings.features, "enable_strict_temporal_conflict_check", True
                )
            )

            has_conflict = False
            conflict_reasons = []

            # 🚫 Règle 1 (AMÉLIORÉE): Vérifier scheduled_time avec marge
            for existing_time in driver_scheduled_times[did]:
                gap_minutes = abs(est_s - existing_time)
                if gap_minutes < min_gap_minutes:
                    logger.debug(
                        (
                            "[DISPATCH] ⏰ Chauffeur #%s a déjà une course à "
                            "%smin, course #%s à %smin (écart: %smin < %smin) "
                            "→ CONFLIT"
                        ),
                        did,
                        existing_time,
                        b_id,
                        est_s,
                        gap_minutes,
                        min_gap_minutes,
                    )
                    has_conflict = True
                    conflict_reasons.append(f"scheduled_time_gap:{gap_minutes}min")
                    break

            # 🚫 Règle 2 (AMÉLIORÉE): Vérifier busy_until avec buffer
            # est_s = quand le chauffeur doit ARRIVER au pickup
            # busy_until[did] = quand le chauffeur finit la précédente
            # Il faut: busy_until + post_trip_buffer <= est_s (avec marge)
            if strict_check and busy_until[did] > 0:
                required_free_time = busy_until[did] + post_trip_buffer
                if est_s < required_free_time:
                    logger.debug(
                        (
                            "[DISPATCH] ⏰ Chauffeur #%s occupé jusqu'à "
                            "%smin (+%smin buffer = %smin), course #%s "
                            "démarre à %smin (écart: %smin) → CONFLIT"
                        ),
                        did,
                        busy_until[did],
                        post_trip_buffer,
                        required_free_time,
                        b_id,
                        est_s,
                        est_s - required_free_time,
                    )
                    has_conflict = True
                    conflict_reasons.append(
                        f"busy_until:{busy_until[did]}→{required_free_time}"
                    )
                elif est_s < busy_until[did]:
                    # Cas edge : chauffeur pas encore libre
                    logger.debug(
                        (
                            "[DISPATCH] ⏰ Chauffeur #%s pas encore libre "
                            "(busy_until=%smin), course #%s à %smin → CONFLIT"
                        ),
                        did,
                        busy_until[did],
                        b_id,
                        est_s,
                    )
                    has_conflict = True
                    conflict_reasons.append(f"driver_not_free:{busy_until[did]}")

            if has_conflict:
                logger.warning(
                    (
                        "[DISPATCH] 🔴 Conflit temporel détecté pour booking "
                        "#%s + driver #%s: %s"
                    ),
                    b_id,
                    did,
                    ", ".join(conflict_reasons),
                )
                # ✅ A1: Incrémenter métrique
                increment_temporal_conflict_counter()
                # ✅ A1: Marquer le rejet avec conflict_penalty dans le debug
                temporal_conflict_rejects.append(
                    {
                        "booking_id": b_id,
                        "driver_id": did,
                        "conflict_reasons": conflict_reasons,
                        "conflict_penalty": -9999.0,  # Score négatif symbolique
                        "estimated_start_min": est_s,
                        "busy_until": busy_until[did],
                        "gap_minutes": min_gap_minutes,
                        "post_trip_buffer": post_trip_buffer,
                    }
                )
                continue
            if sc <= SC_ZERO:
                continue

            # 🎯 Bonus/malus pour équilibrer la charge
            current_load = fairness_effective.get(did, 0)

            # 📈 Pénalité PROGRESSIVE plus douce
            if current_load <= CURRENT_LOAD_THRESHOLD:
                load_penalty = current_load * 0.1
            elif current_load == CURRENT_LOAD_THRESHOLD + 1:
                load_penalty = 0.3
            elif current_load == CURRENT_LOAD_THRESHOLD + 2:
                load_penalty = 0.6
            else:
                load_penalty = 1 + (current_load - 5) * 0.5

            sc -= load_penalty

            # 🏆 Bonus FORT pour chauffeur moins chargé
            # ⚡ CORRECTION: Calculer min_load avec fairness_counts inclus
            # (charge totale réelle)
            current_loads_all = [
                fairness_effective.get(int(cast("Any", d.id)), 0) for d in drivers
            ]
            min_load = min(current_loads_all) if current_loads_all else 0
            if current_load == min_load:
                sc += 0.8
            elif current_load == min_load + 1:
                sc += 0.4

            # ⚠️ Malus pour chauffeur d'urgence
            # ✅ Utiliser le paramètre configurable par le client
            # (settings.emergency.emergency_penalty)
            # ✅ Vérifier via driver_type (pas is_emergency)
            driver_type = getattr(d, "driver_type", None)
            driver_type_str = str(driver_type or "").strip().upper()
            if "." in driver_type_str:
                driver_type_str = driver_type_str.split(".")[-1]
            if driver_type_str == "EMERGENCY":
                # Convertir la pénalité (0-1000) en malus de score
                emergency_penalty = float(
                    getattr(settings.emergency, "emergency_penalty", 900.0)
                )
                malus = -(
                    emergency_penalty / 180.0
                )  # 900 / 180 = 5.0, 500 / 180 = 2.78
                sc += malus

            cand = HeuristicAssignment(
                booking_id=int(cast("Any", b.id)),
                driver_id=did,
                score=sc,
                reason="return_urgent",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if (best is None) or (sc > best[0]):
                best = (sc, cand)

            # ✅ P1: Early exit si score optimal trouvé (seuil de qualité)
            if sc >= HEURISTIC_OPTIMAL_SCORE_THRESHOLD:
                logger.info(
                    "[HEURISTIC] ✅ Optimal score (%.3f >= %.3f) found for booking %s, driver %s - early exit",
                    sc,
                    HEURISTIC_OPTIMAL_SCORE_THRESHOLD,
                    b_id,
                    did,
                )
                break

        if best:
            chosen = best[1]
            assignments.append(chosen)
            proposed_load[int(chosen.driver_id)] += 1
            did2 = int(chosen.driver_id)
            fairness_effective[did2] = fairness_effective.get(did2, 0) + 1

            # ⏱️ CORRECTION: Calculer scheduled_min du booking et utiliser
            # durée OSRM réelle
            scheduled_time_dt = getattr(b, "scheduled_time", None)
            base_time = problem.get("base_time")
            if base_time and scheduled_time_dt:
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = (
                    scheduled_dt_utc - base_dt_utc
                    if scheduled_dt_utc and base_dt_utc
                    else None
                )
                scheduled_min = (
                    int(delta.total_seconds() // 60)
                    if delta
                    else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
                )
            else:
                scheduled_min = (
                    scheduled_time_dt.hour * 60 + scheduled_time_dt.minute
                    if scheduled_time_dt
                    else chosen.estimated_start_min
                )

            # Calculer la durée réelle de la course selon OSRM (pickup + trajet
            # OSRM + dropoff)
            duration_osrm = chosen.estimated_finish_min - chosen.estimated_start_min
            realistic_finish = scheduled_min + duration_osrm
            busy_until[did2] = max(busy_until[did2], realistic_finish)

            # 📅 Enregistrer le scheduled_time RÉEL
            driver_scheduled_times[did2].append(scheduled_min)
            logger.info(
                (
                    "[DISPATCH] ✅ Urgent #%s → Chauffeur #%s (score: %.2f, "
                    "start: %smin, busy_until: %smin)"
                ),
                chosen.booking_id,
                chosen.driver_id,
                chosen.score,
                scheduled_min,
                busy_until[did2],
            )
        else:
            unassigned.append(int(cast("Any", b.id)))
            logger.warning(
                (
                    "[DISPATCH] ⚠️ Impossible d'assigner urgent #%s (aucun "
                    "chauffeur disponible)"
                ),
                b_id,
            )

    # --- 2) Assignations régulières ---
    # Pré-scorage rapide pour limiter la combinatoire
    scored_pool: List[Tuple[float, HeuristicAssignment, Booking]] = []

    # Vérifier si parallélisation activée
    use_parallel = getattr(settings.features, "enable_parallel_heuristics", False)
    scores_dict: dict[
        tuple[int, int], tuple[float, Dict[str, float], int, int]
    ] = {}  # Initialiser pour éviter "unbound"

    logger.debug(
        (
            "[HEURISTIC] 🔍 Début scoring de %s courses régulières avec %s "
            "chauffeurs (parallel=%s)..."
        ),
        len(regular),
        len(drivers),
        use_parallel,
    )

    # ✅ C2: Scoring parallèle optimisé pour 100+ courses
    if (
        use_parallel
        and len(regular) > PARALLEL_MIN_BOOKINGS
        and len(drivers) > PARALLEL_MIN_DRIVERS
    ):
        # Pré-scorer toutes les combinaisons en parallèle
        # ⚡ Calculer les charges actuelles pour tous les chauffeurs (pour
        # le scoring parallèle)
        raw_loads_parallel = {
            int(cast("Any", d.id)): fairness_effective.get(int(cast("Any", d.id)), 0)
            for d in drivers
        }
        current_loads_parallel = _normalized_loads(raw_loads_parallel)
        eligible_drivers_parallel = get_eligible_drivers(
            drivers, current_loads_parallel
        )
        scoring_tasks = []
        for b in regular:
            b_id = int(cast("Any", b.id))
            for d in eligible_drivers_parallel:
                did = int(cast("Any", d.id))
                adjusted_cap = get_adjusted_max_cap(did)
                if fairness_effective.get(did, 0) >= adjusted_cap:
                    continue
                # ✅ C2: Réduire allocations - stocker seulement les IDs
                scoring_tasks.append((b_id, did, b, d))

        # ✅ C2: Exécuter en parallèle avec ThreadPoolExecutor (scores_dict
        # déjà initialisé ligne 1182)
        max_workers = min(len(scoring_tasks), PARALLEL_MAX_WORKERS)

        start_parallel = time.time()  # ✅ C2: Mesurer temps parallélisation

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _score_booking_driver_pair,
                    b,
                    d,
                    (0, 24 * 60),
                    settings,
                    _normalized_loads(fairness_effective),
                    driver_index,
                    company_coords,
                    preferred_driver_id=preferred_driver_id,
                    last_dropoff_coord=driver_last_dropoff_initial.get(
                        did
                    ),  # ⚡ Utiliser last_dropoff_coord si disponible
                ): (b_id, did)
                for b_id, did, b, d in scoring_tasks
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    b_id, d_id, sc, breakdown, (est_s, est_f) = result
                    # ✅ C2: Éviter copies inutiles - stocker directement
                    scores_dict[(b_id, d_id)] = (sc, breakdown, est_s, est_f)
                except Exception as e:
                    logger.error("[ParallelScoring] Error: %s", e)

        parallel_time = time.time() - start_parallel
        logger.info(
            "[C2] ParallelScoring completed %d tasks in %.2fs (speedup: ~%.1fx)",
            len(scores_dict),
            parallel_time,
            len(scoring_tasks) / max_workers,
        )

    for b in regular:
        b_id = int(cast("Any", b.id))
        best_for_b: Tuple[float, HeuristicAssignment] | None = None
        rejected_reasons = []

        # ⚡ Calculer les charges actuelles pour tous les chauffeurs
        raw_loads_regular = {
            int(cast("Any", d.id)): fairness_effective.get(int(cast("Any", d.id)), 0)
            for d in drivers
        }
        current_loads = _normalized_loads(raw_loads_regular)
        # ⚡ Filtrer les chauffeurs éligibles selon préférence ou équité stricte
        eligible_drivers = get_eligible_drivers(drivers, current_loads)

        for d in eligible_drivers:
            did = int(cast("Any", d.id))
            adjusted_cap = get_adjusted_max_cap(did)
            if fairness_effective.get(did, 0) >= adjusted_cap:
                rejected_reasons.append(f"driver#{did}:cap_reached")
                continue

            # Si la course est déjà ASSIGNED à ce driver, gardons une
            # préférence (éviter churn)
            is_assigned = _is_booking_assigned(b)
            cur_driver_id = _current_driver_id(b)
            prefer_assigned = bool(is_assigned and (cur_driver_id == did))

            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            # ⚡ AMÉLIORATION: Utiliser last_dropoff_coord si disponible pour
            # ce chauffeur
            # Cela permet de minimiser les trajets entre courses consécutives
            # dès le scoring initial
            last_dropoff_for_driver = driver_last_dropoff_initial.get(did)

            # Utiliser le score parallèle si disponible
            if (
                use_parallel
                and len(regular) > PARALLEL_MIN_BOOKINGS
                and len(drivers) > PARALLEL_MIN_DRIVERS
                and len(scores_dict) > 0
            ):
                score_key = (b_id, did)
                if score_key in scores_dict:
                    sc, _breakdown, est_s, est_f = scores_dict[score_key]
                    # ⚡ Le scoring parallèle a déjà utilisé
                    # last_dropoff_coord, pas besoin de re-scorer
                else:
                    # Fallback sur scoring normal si pas dans le cache
                    sc, _breakdown, (est_s, est_f) = _score_driver_for_booking(
                        b,
                        d,
                        dw,
                        settings,
                        current_loads,
                        company_coords=company_coords,
                        preferred_driver_id=preferred_driver_id,
                        last_dropoff_coord=last_dropoff_for_driver,
                    )
            else:
                sc, _breakdown, (est_s, est_f) = _score_driver_for_booking(
                    b,
                    d,
                    dw,
                    settings,
                    current_loads,
                    company_coords=company_coords,
                    preferred_driver_id=preferred_driver_id,
                    last_dropoff_coord=last_dropoff_for_driver,
                )

            # 🚫 CORRECTION CRITIQUE: Utiliser scheduled_time (heure
            # demandée par le client)
            # au lieu de est_s (optimisé OSRM) pour vérifier la faisabilité !
            scheduled_time_dt = getattr(b, "scheduled_time", None)
            if not scheduled_time_dt:
                rejected_reasons.append(f"driver#{did}:no_scheduled_time")
                continue

            # Convertir scheduled_time en minutes depuis minuit du jour concerné
            # (même logique que dans data.py pour la cohérence)
            base_time = problem.get("base_time")
            if base_time:
                # Si base_time est fourni, calculer depuis ce moment
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = (
                    scheduled_dt_utc - base_dt_utc
                    if scheduled_dt_utc and base_dt_utc
                    else None
                )
                scheduled_min = (
                    int(delta.total_seconds() // 60)
                    if delta
                    else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
                )
            else:
                # Sinon, utiliser les heures/minutes du jour
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

            # 🔍 Logs détaillés pour debug
            if b_id in [106, 109, 11, 115] and did == DID_THRESHOLD:
                logger.error("[DEBUG] Course #%s + Giuseppe (#%s):", b_id, did)
                logger.error(
                    "  - scheduled_time: %s (%smin)", scheduled_time_dt, scheduled_min
                )
                logger.error("  - est_start_min (OSRM optimisé): %smin", est_s)
                logger.error("  - est_finish_min: %smin", est_f)
                logger.error("  - busy_until[%s]: %smin", did, busy_until[did])
                logger.error(
                    "  - driver_scheduled_times[%s]: %s",
                    did,
                    driver_scheduled_times[did],
                )
                logger.error("  - score: %.3f", sc)

            # ✅ A1: VALIDATION STRICTE DES CONFLITS TEMPORELS (section regular)
            min_gap_minutes = int(getattr(settings.safety, "min_gap_minutes", 30))
            post_trip_buffer = int(getattr(settings.safety, "post_trip_buffer_min", 15))
            strict_check = bool(
                getattr(
                    settings.features, "enable_strict_temporal_conflict_check", True
                )
            )

            has_conflict = False
            can_pool = False
            conflict_reasons_reg = []

            # 🚫 Règle 1 (AMÉLIORÉE): Vérifier scheduled_time avec calcul du temps réel
            # nécessaire
            # Calculer les temps de service configurables depuis settings.service_times
            # (configurables par le client)
            pickup_service_min = int(
                getattr(settings.service_times, "pickup_service_min", 5)
            )
            dropoff_service_min = int(
                getattr(settings.service_times, "dropoff_service_min", 10)
            )
            min_transition_margin_min = int(
                getattr(settings.service_times, "min_transition_margin_min", 15)
            )

            for existing_time in driver_scheduled_times[did]:
                # Chercher la course existante pour calculer le temps réel nécessaire
                # ✅ CORRECTION: Vérifier aussi les courses qui sont en train
                # d'être assignées dans le même batch
                existing_booking = None
                # D'abord chercher dans les assignments déjà faits
                for assigned in [a for a in assignments if a.driver_id == did]:
                    assigned_booking = next(
                        (
                            bk
                            for bk in bookings
                            if int(cast("Any", bk.id)) == assigned.booking_id
                        ),
                        None,
                    )
                    if assigned_booking:
                        assigned_time_dt = getattr(
                            assigned_booking, "scheduled_time", None
                        )
                        if assigned_time_dt:
                            if base_time:
                                assigned_dt_utc = to_utc(assigned_time_dt)
                                base_dt_utc = to_utc(base_time)
                                delta = (
                                    assigned_dt_utc - base_dt_utc
                                    if assigned_dt_utc and base_dt_utc
                                    else None
                                )
                                assigned_min = (
                                    int(delta.total_seconds() // 60)
                                    if delta
                                    else (
                                        assigned_time_dt.hour * 60
                                        + assigned_time_dt.minute
                                    )
                                )
                            else:
                                assigned_min = (
                                    assigned_time_dt.hour * 60 + assigned_time_dt.minute
                                )
                            if assigned_min == existing_time:
                                existing_booking = assigned_booking
                                break

                # ✅ Si pas trouvé dans assignments, chercher dans toutes les bookings
                # du problème
                # (pour détecter les conflits avec les courses qui seront assignées
                # dans le même batch)
                if not existing_booking:
                    for other_booking in bookings:
                        if int(cast("Any", other_booking.id)) == b_id:
                            continue  # Ignorer la course actuelle
                        other_time_dt = getattr(other_booking, "scheduled_time", None)
                        if other_time_dt:
                            if base_time:
                                other_dt_utc = to_utc(other_time_dt)
                                base_dt_utc = to_utc(base_time)
                                delta = (
                                    other_dt_utc - base_dt_utc
                                    if other_dt_utc and base_dt_utc
                                    else None
                                )
                                other_min = (
                                    int(delta.total_seconds() // 60)
                                    if delta
                                    else (
                                        other_time_dt.hour * 60 + other_time_dt.minute
                                    )
                                )
                            else:
                                other_min = (
                                    other_time_dt.hour * 60 + other_time_dt.minute
                                )
                            if other_min == existing_time:
                                # Vérifier si cette course est déjà assignée
                                # à ce chauffeur ou pourrait l'être
                                # (pour éviter les faux positifs, on vérifie seulement
                                # si elle est dans driver_scheduled_times)
                                existing_booking = other_booking
                                break

                if not existing_booking:
                    # Si on ne trouve pas la course, utiliser la vérification simple
                    gap_minutes = abs(scheduled_min - existing_time)
                    if gap_minutes < min_gap_minutes:
                        has_conflict = True
                        conflict_reasons_reg.append(f"time_gap:{gap_minutes}min")
                        break
                    continue

                # À ce point, existing_booking est défini
                # (sinon on aurait fait continue)
                # Vérifier si regroupement possible
                if _can_be_pooled(b, existing_booking, settings):
                    can_pool = True
                    logger.info(
                        (
                            "[POOLING] 🚗 Course #%s peut être regroupée avec #%s "
                            "(chauffeur #%s)"
                        ),
                        b_id,
                        existing_booking.id,
                        did,
                    )
                    break

                # Calculer le temps réel nécessaire entre les deux courses
                # 1. Temps de trajet de la course précédente (pickup → dropoff)
                existing_pickup_coord = _booking_coords(existing_booking)[0]
                existing_dropoff_coord = _booking_coords(existing_booking)[1]
                booking_pickup_coord = _booking_coords(b)[0]

                # Utiliser la matrice de temps si disponible, sinon haversine
                trip_time_min = 20  # Estimation par défaut
                transition_time_min = 15  # Estimation par défaut

                if "time_matrix" in problem and "coords" in problem:
                    try:
                        coords = problem["coords"]
                        time_matrix = problem["time_matrix"]

                        # Trouver les indices dans la matrice
                        existing_pickup_idx = None
                        existing_dropoff_idx = None
                        booking_pickup_idx = None

                        for idx, coord in enumerate(coords):
                            if coord == existing_pickup_coord:
                                existing_pickup_idx = idx
                            if coord == existing_dropoff_coord:
                                existing_dropoff_idx = idx
                            if coord == booking_pickup_coord:
                                booking_pickup_idx = idx

                        # Calculer temps de trajet course précédente
                        if (
                            existing_pickup_idx is not None
                            and existing_dropoff_idx is not None
                            and existing_pickup_idx < len(time_matrix)
                            and existing_dropoff_idx
                            < len(time_matrix[existing_pickup_idx])
                        ):
                            trip_time_min = int(
                                time_matrix[existing_pickup_idx][existing_dropoff_idx]
                            )

                        # Calculer temps de transition
                        # (dropoff précédent → pickup suivant)
                        if (
                            existing_dropoff_idx is not None
                            and booking_pickup_idx is not None
                            and existing_dropoff_idx < len(time_matrix)
                            and booking_pickup_idx
                            < len(time_matrix[existing_dropoff_idx])
                        ):
                            transition_time_min = int(
                                time_matrix[existing_dropoff_idx][booking_pickup_idx]
                            )
                    except Exception as e:
                        logger.debug(
                            (
                                "[DISPATCH] Erreur calcul matrice temps, "
                                "utilisation haversine: %s"
                            ),
                            e,
                        )
                        # Fallback: utiliser haversine
                        if existing_pickup_coord and existing_dropoff_coord:
                            trip_time_min = haversine_minutes(
                                existing_pickup_coord,
                                existing_dropoff_coord,
                                avg_kmh=getattr(
                                    getattr(settings, "matrix", None),
                                    "avg_speed_kmh",
                                    25,
                                ),
                            )
                        if existing_dropoff_coord and booking_pickup_coord:
                            transition_time_min = haversine_minutes(
                                existing_dropoff_coord,
                                booking_pickup_coord,
                                avg_kmh=getattr(
                                    getattr(settings, "matrix", None),
                                    "avg_speed_kmh",
                                    25,
                                ),
                            )

                # Temps total nécessaire entre les deux courses
                total_time_needed = (
                    trip_time_min  # Temps de trajet course précédente
                    + dropoff_service_min  # Temps de dropoff
                    + transition_time_min  # Temps de trajet entre courses
                    + pickup_service_min  # Temps de pickup
                    + min_transition_margin_min  # Marge de sécurité
                )

                # Calculer l'heure de fin estimée de la course précédente
                # Utiliser datetime.timedelta directement pour éviter problème de scope
                # avec le linter
                from datetime import timedelta as dt_timedelta

                existing_scheduled_dt = getattr(
                    existing_booking, "scheduled_time", None
                )
                if not existing_scheduled_dt:
                    continue

                existing_end_time = existing_scheduled_dt + dt_timedelta(
                    minutes=trip_time_min + pickup_service_min + dropoff_service_min
                )

                # Calculer l'heure de début nécessaire pour la nouvelle course
                booking_scheduled_dt = getattr(b, "scheduled_time", None)
                if booking_scheduled_dt:
                    required_start_time = booking_scheduled_dt - dt_timedelta(
                        minutes=transition_time_min
                        + pickup_service_min
                        + min_transition_margin_min
                    )

                    # Vérifier si on a assez de temps
                    if existing_end_time > required_start_time:
                        time_gap = (
                            required_start_time - existing_end_time
                        ).total_seconds() / 60
                        has_conflict = True
                        conflict_msg = (
                            f"temps_insuffisant: nécessaire={total_time_needed}min, "
                            f"écart={time_gap:.1f}min "
                            f"(course #{existing_booking.id} fin "
                            f"{existing_end_time:%H:%M} vs course #{b_id} début "
                            f"{booking_scheduled_dt:%H:%M})"
                        )
                        conflict_reasons_reg.append(conflict_msg)
                        logger.warning(
                            (
                                "[DISPATCH] ⚠️ Conflit temporel détaillé: course #%s "
                                "(fin %s) et #%s (début %s) → temps nécessaire: %dmin, "
                                "écart disponible: %.1fmin"
                            ),
                            existing_booking.id,
                            existing_end_time.strftime("%H:%M"),
                            b_id,
                            booking_scheduled_dt.strftime("%H:%M"),
                            total_time_needed,
                            time_gap,
                        )
                        break

            if has_conflict and not can_pool:
                logger.warning(
                    (
                        "[DISPATCH] 🔴 Conflit temporel (regular) booking #%s + "
                        "driver #%s: %s"
                    ),
                    b_id,
                    did,
                    ", ".join(conflict_reasons_reg),
                )
                # ✅ A1: Marquer le rejet avec conflict_penalty
                temporal_conflict_rejects.append(
                    {
                        "booking_id": b_id,
                        "driver_id": did,
                        "conflict_reasons": conflict_reasons_reg,
                        "conflict_penalty": -9999.0,
                        "estimated_start_min": est_s,
                        "scheduled_min": scheduled_min,
                    }
                )
                continue

            # 🚫 Règle 2 (AMÉLIORÉE): Vérifier busy_until avec buffer configurable
            required_free_time = (
                busy_until[did] + post_trip_buffer if strict_check else busy_until[did]
            )
            if scheduled_min < required_free_time:
                rejected_reasons.append(f"driver#{did}:busy")
                conflict_reasons_reg.append(
                    f"busy_until:{busy_until[did]}→{required_free_time}"
                )
                if b_id in [106, 109, 11, 115] and did == DID_THRESHOLD:
                    logger.error(
                        "  ❌ BUSY: scheduled_min=%smin < busy_until+margin=%smin",
                        scheduled_min,
                        required_free_time,
                    )
                logger.warning(
                    "[DISPATCH] 🔴 Conflit busy_until booking #%s + driver #%s: %s",
                    b_id,
                    did,
                    ", ".join(conflict_reasons_reg),
                )
                continue
            if sc <= SC_ZERO:
                rejected_reasons.append(f"driver#{did}:score_negative")
                continue

            # 🎯 Bonus/malus pour équilibrer la charge entre chauffeurs
            current_load = fairness_effective.get(did, 0)

            # 📈 Pénalité PROGRESSIVE plus douce pour assigner TOUTES les courses
            # en favorisant l'équilibre
            # 0-2 courses : pénalité faible (0-0.2)
            # 3 courses : 0.3 pénalité (acceptable)
            # 4 courses : 0.6 pénalité (forte mais pas bloquante)
            # 5+ courses : 1+ pénalité (très forte mais permet quand même
            # l'assignation si nécessaire)
            if current_load <= CURRENT_LOAD_THRESHOLD:
                load_penalty = current_load * 0.1
            elif current_load == CURRENT_LOAD_THRESHOLD + 1:
                load_penalty = 0.3
            elif current_load == CURRENT_LOAD_THRESHOLD + 2:
                load_penalty = 0.6
            else:
                load_penalty = 1 + (current_load - 5) * 0.5

            sc -= load_penalty

            # 🏆 Bonus FORT pour chauffeur moins chargé (favoriser l'équilibrage)
            # ⚡ CORRECTION: Calculer min_load avec fairness_counts inclus
            # (charge totale réelle)
            current_loads_all = [
                fairness_effective.get(int(cast("Any", d.id)), 0) for d in drivers
            ]
            min_load = min(current_loads_all) if current_loads_all else 0
            if current_load == min_load:
                sc += 0.8  # Fort bonus pour le chauffeur le moins chargé
            elif current_load == min_load + 1:
                sc += 0.4  # Bonus moyen si proche du minimum

            # ⚠️ Malus pour chauffeur d'urgence (dernier recours uniquement)
            # ✅ Utiliser le paramètre configurable par le client
            # (settings.emergency.emergency_penalty)
            # ✅ Vérifier via driver_type (pas is_emergency)
            driver_type = getattr(d, "driver_type", None)
            driver_type_str = str(driver_type or "").strip().upper()
            if "." in driver_type_str:
                driver_type_str = driver_type_str.split(".")[-1]
            if driver_type_str == "EMERGENCY":
                # Convertir la pénalité (0-1000) en malus de score
                # Plus la pénalité est élevée, plus le malus est fort
                # 900 = malus très fort, 500 = malus modéré, 0 = pas de malus
                emergency_penalty = float(
                    getattr(settings.emergency, "emergency_penalty", 900.0)
                )
                # Normaliser: 900 → -5.0, 500 → -2.5, 0 → 0
                malus = -(
                    emergency_penalty / 180.0
                )  # 900 / 180 = 5.0, 500 / 180 = 2.78
                sc += malus

            if prefer_assigned:
                sc += 0.2  # stabilité de planning

            cand = HeuristicAssignment(
                booking_id=int(cast("Any", b.id)),
                driver_id=did,
                score=sc,
                reason="regular_scoring",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if (best_for_b is None) or (sc > best_for_b[0]):
                best_for_b = (sc, cand)

            # ✅ P1: Early exit si score optimal trouvé (seuil de qualité)
            if sc >= HEURISTIC_OPTIMAL_SCORE_THRESHOLD:
                logger.info(
                    "[HEURISTIC] ✅ Optimal score (%.3f >= %.3f) found for booking %s, driver %s - early exit",
                    sc,
                    HEURISTIC_OPTIMAL_SCORE_THRESHOLD,
                    b_id,
                    did,
                )
                break

        if best_for_b:
            # Log pour tracer les décisions de sélection
            if preferred_driver_id and best_for_b[1].driver_id == preferred_driver_id:
                logger.info(
                    (
                        "[HEURISTIC] ✅ Booking #%s → Chauffeur préféré #%s "
                        "(score: %.2f, reason: preferred_bonus)"
                    ),
                    b_id,
                    preferred_driver_id,
                    best_for_b[0],
                )
            elif preferred_driver_id:
                logger.debug(
                    (
                        "[HEURISTIC] ⚠️ Booking #%s → Chauffeur #%s (score: %.2f) "
                        "au lieu du préféré #%s"
                    ),
                    b_id,
                    best_for_b[1].driver_id,
                    best_for_b[0],
                    preferred_driver_id,
                )
            scored_pool.append((best_for_b[0], best_for_b[1], b))
            logger.debug(
                (
                    "[HEURISTIC] ✅ Course #%s peut être assignée au driver #%s "
                    "(score: %.2f)"
                ),
                b_id,
                best_for_b[1].driver_id,
                best_for_b[0],
            )
        else:
            unassigned.append(int(cast("Any", b.id)))
            # ✅ FIX: Réduire la verbosité de ce log (WARNING → DEBUG)
            # car il peut être très verbeux avec de nombreux drivers
            logger.debug(
                "[HEURISTIC] ❌ Course #%s REJETÉE par tous les chauffeurs: %s",
                b_id,
                ", ".join(rejected_reasons) if rejected_reasons else "aucune raison",
            )

    # 🕐 CORRECTION: Ordonner par scheduled_time CHRONOLOGIQUE d'abord, puis par score
    # Cela évite d'assigner les courses tardives (bon score) avant les courses matinales
    # (moins bon score)
    # et d'avoir des conflits "busy_until" absurdes
    scored_pool.sort(
        key=lambda x: (
            sort_key_utc(cast("Any", getattr(x[2], "scheduled_time", None))),
            -x[0],
        )
    )

    pooled_bookings = set()  # Track bookings that were pooled to skip other candidates

    # ⚡ Dictionnaire pour suivre la position de dropoff de la dernière course assignée
    # à chaque chauffeur
    # Cela permet de minimiser les trajets entre courses consécutives
    driver_last_dropoff: Dict[int, Tuple[float, float]] = {}

    logger.info(
        "[DISPATCH] 🔍 Début boucle scored_pool: %d courses à traiter", len(scored_pool)
    )

    for sc_original, cand, b in scored_pool:
        # Si cette course a déjà été assignée via regroupement, skip les autres
        # candidats
        if int(cast("Any", b.id)) in pooled_bookings:
            continue

        # Double check cap
        did = int(cand.driver_id)
        adjusted_cap = get_adjusted_max_cap(did)
        if fairness_effective.get(did, 0) >= adjusted_cap:
            logger.debug(
                "[DISPATCH] ⏭️ Chauffeur #%s a atteint le cap (%s), skipped",
                did,
                max_cap,
            )
            continue

        # ⚡ AMÉLIORATION: Re-scorer en utilisant la position de dropoff de la dernière
        # course assignée
        # Cela permet de minimiser les trajets entre courses consécutives
        # On cherche dans : 1) driver_last_dropoff
        # (courses déjà assignées dans le batch),
        # 2) assignments (courses assignées dans le batch en cours)
        # 2) assignments (courses assignées dans le batch en cours)
        sc = sc_original
        last_dropoff = driver_last_dropoff.get(did)

        # Log de diagnostic pour comprendre pourquoi le re-scoring ne se déclenche pas
        # Utiliser INFO pour s'assurer que les logs apparaissent
        logger.info(
            (
                "[DISPATCH] 🔍 Re-scoring check pour course #%s + chauffeur #%s: "
                "driver_last_dropoff=%s, assignments_count=%d"
            ),
            int(cast("Any", b.id)),
            did,
            "présent" if last_dropoff else "absent",
            len(assignments),
        )

        # ⚡ Si pas trouvé dans driver_last_dropoff, chercher dans assignments
        # (courses assignées dans le batch en cours)
        if not last_dropoff:
            # Trouver la dernière course assignée à ce chauffeur dans le batch
            # (par scheduled_time)
            b_scheduled = getattr(b, "scheduled_time", None)
            if b_scheduled:
                last_assigned_booking = None
                last_assigned_time = None

                # Parcourir les assignments déjà faits pour ce chauffeur
                assignments_for_driver = [a for a in assignments if a.driver_id == did]
                logger.info(
                    (
                        "[DISPATCH] 🔍 Course #%s: Recherche dans assignments pour "
                        "chauffeur #%s: %d assignments trouvés"
                    ),
                    int(cast("Any", b.id)),
                    did,
                    len(assignments_for_driver),
                )

                for assigned in assignments_for_driver:
                    assigned_booking = next(
                        (
                            bk
                            for bk in bookings
                            if int(cast("Any", bk.id)) == assigned.booking_id
                        ),
                        None,
                    )
                    if assigned_booking:
                        assigned_scheduled = getattr(
                            assigned_booking, "scheduled_time", None
                        )
                        logger.info(
                            (
                                "[DISPATCH] 🔍 Course #%s: Assignment #%s "
                                "(booking_id=%s) pour chauffeur #%s: scheduled=%s, "
                                "b_scheduled=%s"
                            ),
                            int(cast("Any", b.id)),
                            assigned.booking_id,
                            assigned.booking_id,
                            did,
                            assigned_scheduled,
                            b_scheduled,
                        )
                        # Garder la course assignée la plus récente qui se
                        # termine AVANT la course actuelle
                        if (
                            assigned_scheduled
                            and assigned_scheduled < b_scheduled
                            and (
                                last_assigned_time is None
                                or assigned_scheduled > last_assigned_time
                            )
                        ):
                            last_assigned_booking = assigned_booking
                            last_assigned_time = assigned_scheduled
                            logger.info(
                                (
                                    "[DISPATCH] 🔍 Course #%s: Nouvelle "
                                    "meilleure course trouvée: #%s à %s"
                                ),
                                int(cast("Any", b.id)),
                                last_assigned_booking.id,
                                last_assigned_time,
                            )

                # Si on a trouvé une course assignée, utiliser sa position de dropoff
                if last_assigned_booking:
                    _, dropoff_coord = _booking_coords(last_assigned_booking)
                    if dropoff_coord:
                        last_dropoff = dropoff_coord
                        logger.info(
                            (
                                "[DISPATCH] 🔍 Course #%s: Utilisation "
                                "dropoff de course #%s (assignée dans le "
                                "batch à %s) pour chauffeur #%s"
                            ),
                            int(cast("Any", b.id)),
                            last_assigned_booking.id,
                            last_assigned_time,
                            did,
                        )
                    else:
                        logger.warning(
                            (
                                "[DISPATCH] ⚠️ Course #%s: Dropoff coord "
                                "non trouvée pour course #%s (chauffeur #%s)"
                            ),
                            int(cast("Any", b.id)),
                            last_assigned_booking.id,
                            did,
                        )
                else:
                    logger.info(
                        (
                            "[DISPATCH] 🔍 Course #%s: Aucune course "
                            "assignée trouvée dans le batch pour chauffeur "
                            "#%s (scheduled_time=%s)"
                        ),
                        int(cast("Any", b.id)),
                        did,
                        b_scheduled,
                    )

        # ⚡ Utiliser aussi driver_last_dropoff_initial (courses déjà
        # assignées avant le batch)
        if not last_dropoff:
            last_dropoff = driver_last_dropoff_initial.get(did)
            if last_dropoff:
                logger.info(
                    (
                        "[DISPATCH] 🔍 Course #%s: Utilisation dropoff "
                        "initial (course déjà assignée avant batch) pour "
                        "chauffeur #%s"
                    ),
                    int(cast("Any", b.id)),
                    did,
                )

        if last_dropoff:
            # Trouver le chauffeur correspondant
            driver_obj = (
                drivers[driver_index.get(did, 0)]
                if driver_index.get(did) is not None
                else None
            )
            if driver_obj:
                di = driver_index.get(did, 0)
                dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
                # Re-scorer avec last_dropoff_coord
                sc_improved, breakdown_improved, (est_s_improved, est_f_improved) = (
                    _score_driver_for_booking(
                        b,
                        driver_obj,
                        dw,
                        settings,
                        fairness_effective,
                        company_coords=company_coords,
                        preferred_driver_id=preferred_driver_id,
                        last_dropoff_coord=last_dropoff,
                    )
                )
                # ⚡ TOUJOURS utiliser le score amélioré si last_dropoff est
                # disponible
                # La proximité à la dernière course assignée est un critère
                # important pour minimiser les trajets
                # Même si le score n'est pas strictement meilleur, on
                # privilégie la continuité géographique
                sc = sc_improved
                cand.estimated_start_min = est_s_improved
                cand.estimated_finish_min = est_f_improved
                cand.score = sc_improved

                # Log détaillé pour comprendre l'impact
                score_delta = sc_improved - sc_original
                proximity_contrib = breakdown_improved.get("proximity", 0)
                continuity_bonus_contrib = breakdown_improved.get("continuity_bonus", 0)
                logger.info(
                    (
                        "[DISPATCH] ⚡ Re-scoring avec dropoff précédente "
                        "pour course #%s + chauffeur #%s: %.2f → %.2f "
                        "(Δ=%.2f, proximité=%.2f, continuité=%.2f)"
                    ),
                    int(cast("Any", b.id)),
                    did,
                    sc_original,
                    sc_improved,
                    score_delta,
                    proximity_contrib,
                    continuity_bonus_contrib,
                )

        # 🚫 Récupérer le scheduled_time réel du booking pour les vérifications finales
        scheduled_time_dt = getattr(b, "scheduled_time", None)
        base_time = problem.get("base_time")
        if base_time:
            scheduled_dt_utc = to_utc(scheduled_time_dt)
            base_dt_utc = to_utc(base_time)
            delta = (
                scheduled_dt_utc - base_dt_utc
                if scheduled_dt_utc and base_dt_utc
                else None
            )
            scheduled_min = (
                int(delta.total_seconds() // 60)
                if delta
                else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
                if scheduled_time_dt
                else 0
            )
        else:
            scheduled_min = (
                scheduled_time_dt.hour * 60 + scheduled_time_dt.minute
                if scheduled_time_dt
                else 0
            )

        # ✅ A1: VÉRIFICATION FINALE CONFLITS TEMPORELS (scored_pool)
        min_gap_minutes = int(getattr(settings.safety, "min_gap_minutes", 30))
        post_trip_buffer = int(getattr(settings.safety, "post_trip_buffer_min", 15))
        strict_check = bool(
            getattr(settings.features, "enable_strict_temporal_conflict_check", True)
        )

        has_conflict = False
        can_pool = False
        pooled_with = None
        conflict_reasons_final = []

        # ✅ AMÉLIORATION: Utiliser le même calcul détaillé que dans la
        # section "regular"
        # Calculer les temps de service configurables depuis settings.service_times
        pickup_service_min = int(
            getattr(settings.service_times, "pickup_service_min", 5)
        )
        dropoff_service_min = int(
            getattr(settings.service_times, "dropoff_service_min", 10)
        )
        min_transition_margin_min = int(
            getattr(settings.service_times, "min_transition_margin_min", 15)
        )

        for existing_time in driver_scheduled_times[did]:
            gap_minutes = abs(scheduled_min - existing_time)
            if gap_minutes < min_gap_minutes:
                # Chercher la course existante déjà assignée à ce chauffeur
                existing_booking = None
                for assigned in [a for a in assignments if a.driver_id == did]:
                    assigned_booking = next(
                        (
                            bk
                            for bk in bookings
                            if int(cast("Any", bk.id)) == assigned.booking_id
                        ),
                        None,
                    )
                    if assigned_booking:
                        assigned_time_dt = getattr(
                            assigned_booking, "scheduled_time", None
                        )
                        if assigned_time_dt:
                            base_time = problem.get("base_time")
                            if base_time:
                                assigned_dt_utc = to_utc(assigned_time_dt)
                                base_dt_utc = to_utc(base_time)
                                delta = (
                                    assigned_dt_utc - base_dt_utc
                                    if assigned_dt_utc and base_dt_utc
                                    else None
                                )
                                assigned_min = (
                                    int(delta.total_seconds() // 60)
                                    if delta
                                    else (
                                        assigned_time_dt.hour * 60
                                        + assigned_time_dt.minute
                                    )
                                )
                            else:
                                assigned_min = (
                                    assigned_time_dt.hour * 60 + assigned_time_dt.minute
                                )

                            if assigned_min == existing_time:
                                existing_booking = assigned_booking
                                break

                # Vérifier si regroupement possible
                if existing_booking and _can_be_pooled(b, existing_booking, settings):
                    can_pool = True
                    pooled_with = existing_booking.id
                    logger.warning(
                        (
                            "[POOLING] 🚗 Course #%s FORCÉE au chauffeur "
                            "#%s (regroupement avec #%s, priorité absolue)"
                        ),
                        cand.booking_id,
                        did,
                        existing_booking.id,
                    )
                    pooled_bookings.add(int(cast("Any", b.id)))
                    break

                # ✅ CALCUL DÉTAILLÉ du temps réel nécessaire (comme dans
                # la section "regular")
                if existing_booking:
                    # Calculer le temps réel nécessaire entre les deux courses
                    existing_pickup_coord = _booking_coords(existing_booking)[0]
                    existing_dropoff_coord = _booking_coords(existing_booking)[1]
                    booking_pickup_coord = _booking_coords(b)[0]

                    # Utiliser la matrice de temps si disponible, sinon haversine
                    trip_time_min = 20  # Estimation par défaut
                    transition_time_min = 15  # Estimation par défaut

                    if "time_matrix" in problem and "coords" in problem:
                        try:
                            coords = problem["coords"]
                            time_matrix = problem["time_matrix"]

                            # Trouver les indices dans la matrice
                            existing_pickup_idx = None
                            existing_dropoff_idx = None
                            booking_pickup_idx = None

                            for idx, coord in enumerate(coords):
                                if coord == existing_pickup_coord:
                                    existing_pickup_idx = idx
                                if coord == existing_dropoff_coord:
                                    existing_dropoff_idx = idx
                                if coord == booking_pickup_coord:
                                    booking_pickup_idx = idx

                            # Calculer temps de trajet course précédente
                            if (
                                existing_pickup_idx is not None
                                and existing_dropoff_idx is not None
                                and existing_pickup_idx < len(time_matrix)
                                and existing_dropoff_idx
                                < len(time_matrix[existing_pickup_idx])
                            ):
                                trip_time_min = int(
                                    time_matrix[existing_pickup_idx][
                                        existing_dropoff_idx
                                    ]
                                )

                            # Calculer temps de transition (dropoff précédent
                            # → pickup suivant)
                            if (
                                existing_dropoff_idx is not None
                                and booking_pickup_idx is not None
                                and existing_dropoff_idx < len(time_matrix)
                                and booking_pickup_idx
                                < len(time_matrix[existing_dropoff_idx])
                            ):
                                transition_time_min = int(
                                    time_matrix[existing_dropoff_idx][
                                        booking_pickup_idx
                                    ]
                                )
                        except Exception as e:
                            logger.debug(
                                (
                                    "[DISPATCH] Erreur calcul matrice temps "
                                    "(scored_pool), utilisation haversine: %s"
                                ),
                                e,
                            )
                            # Fallback: utiliser haversine
                            if existing_pickup_coord and existing_dropoff_coord:
                                trip_time_min = haversine_minutes(
                                    existing_pickup_coord,
                                    existing_dropoff_coord,
                                    avg_kmh=getattr(
                                        getattr(settings, "matrix", None),
                                        "avg_speed_kmh",
                                        25,
                                    ),
                                )
                            if existing_dropoff_coord and booking_pickup_coord:
                                transition_time_min = haversine_minutes(
                                    existing_dropoff_coord,
                                    booking_pickup_coord,
                                    avg_kmh=getattr(
                                        getattr(settings, "matrix", None),
                                        "avg_speed_kmh",
                                        25,
                                    ),
                                )

                    # Temps total nécessaire
                    total_time_needed = (
                        trip_time_min  # Temps de trajet course précédente
                        + dropoff_service_min  # Temps de dropoff
                        + transition_time_min  # Temps de trajet entre courses
                        + pickup_service_min  # Temps de pickup
                        + min_transition_margin_min  # Marge de sécurité
                    )

                    # Calculer l'heure de fin estimée de la course précédente
                    from datetime import timedelta

                    existing_scheduled_dt = getattr(
                        existing_booking, "scheduled_time", None
                    )
                    booking_scheduled_dt = getattr(b, "scheduled_time", None)

                    # Si les deux courses ont des heures planifiées, faire
                    # le calcul détaillé
                    if existing_scheduled_dt and booking_scheduled_dt:
                        existing_end_time = existing_scheduled_dt + timedelta(
                            minutes=trip_time_min
                            + pickup_service_min
                            + dropoff_service_min
                        )

                        # Calculer l'heure de début nécessaire pour la nouvelle course
                        required_start_time = booking_scheduled_dt - timedelta(
                            minutes=transition_time_min
                            + pickup_service_min
                            + min_transition_margin_min
                        )

                        # Vérifier si on a assez de temps
                        if existing_end_time > required_start_time:
                            time_gap = (
                                required_start_time - existing_end_time
                            ).total_seconds() / 60
                            conflict_msg = (
                                f"temps_insuffisant: nécessaire="
                                f"{total_time_needed}min, "
                                f"écart={time_gap:.1f}min (course "
                                f"#{existing_booking.id} fin "
                                f"{existing_end_time:%H:%M} "
                                f"vs course #{cand.booking_id} début "
                                f"{booking_scheduled_dt:%H:%M})"
                            )
                            conflict_reasons_final.append(conflict_msg)
                            logger.warning(
                                (
                                    "[DISPATCH] ⚠️ Conflit temporel détaillé "
                                    "(scored_pool): course #%s (fin %s) et "
                                    "#%s (début %s) → temps nécessaire: "
                                    "%dmin, écart disponible: %.1fmin"
                                ),
                                existing_booking.id,
                                existing_end_time.strftime("%H:%M"),
                                cand.booking_id,
                                booking_scheduled_dt.strftime("%H:%M"),
                                total_time_needed,
                                time_gap,
                            )
                            has_conflict = True
                            break
                        # Si pas de conflit détecté par le calcul détaillé,
                        # continuer la boucle
                    else:
                        # Si les heures ne sont pas disponibles, utiliser la
                        # vérification simple
                        conflict_reasons_final.append(
                            f"time_gap:{gap_minutes}min "
                            + "(heures non disponibles pour calcul détaillé)"
                        )
                        conflict_msg = (
                            f"⚠️ CONFLIT: Chauffeur #{did} a course à "
                            f"{existing_time}min, course #{cand.booking_id} à "
                            f"{scheduled_min}min (écart: {gap_minutes}min)"
                        )
                        # ✅ FIX: Réduire le niveau de log en mode testing
                        # (attendu dans tests de validation temporelle)
                        is_testing = False
                        try:
                            is_testing = os.getenv("FLASK_CONFIG") == "testing"
                            try:
                                from flask import (  # pyright: ignore[reportMissingImports]
                                    current_app,
                                )

                                is_testing = is_testing or current_app.config.get(
                                    "TESTING", False
                                )
                            except RuntimeError:
                                pass
                        except Exception:
                            pass

                        log_level = logger.debug if is_testing else logger.warning
                        log_level("[DISPATCH] %s → SKIP", conflict_msg)
                        has_conflict = True
                        break
                else:
                    # Si pas de calcul détaillé possible (existing_booking
                    # non trouvé), utiliser la vérification simple
                    conflict_reasons_final.append(f"time_gap:{gap_minutes}min")
                    conflict_msg = (
                        f"⚠️ CONFLIT: Chauffeur #{did} a course à "
                        f"{existing_time}min, course #{cand.booking_id} à "
                        f"{scheduled_min}min (écart: {gap_minutes}min)"
                    )
                    # ✅ FIX: Réduire le niveau de log en mode testing
                    # (attendu dans tests de validation temporelle)
                    is_testing = False
                    try:
                        is_testing = os.getenv("FLASK_CONFIG") == "testing"
                        try:
                            from flask import (  # pyright: ignore[reportMissingImports]
                                current_app,
                            )

                            is_testing = is_testing or current_app.config.get(
                                "TESTING", False
                            )
                        except RuntimeError:
                            pass
                    except Exception:
                        pass

                    log_level = logger.debug if is_testing else logger.warning
                    log_level("[DISPATCH] %s → SKIP", conflict_msg)
                    has_conflict = True
                    break

        if has_conflict and not can_pool:
            # ✅ FIX: Réduire le niveau de log en mode testing
            # (attendu dans tests de validation temporelle)
            is_testing = False
            try:
                # Essayer d'abord via variable d'environnement
                # (plus sûr, fonctionne partout)
                is_testing = os.getenv("FLASK_CONFIG") == "testing"
                # Si current_app est disponible, utiliser sa config (plus précis)
                try:
                    from flask import (  # pyright: ignore[reportMissingImports]
                        current_app,
                    )

                    is_testing = is_testing or current_app.config.get("TESTING", False)
                except RuntimeError:
                    # current_app pas disponible (hors contexte Flask),
                    # utiliser seulement env var
                    pass
            except Exception:
                # En cas d'erreur, utiliser warning par défaut
                pass

            log_level = logger.debug if is_testing else logger.warning
            log_level(
                "[DISPATCH] 🔴 Conflit temporel (final) booking #%s + driver #%s: %s",
                cand.booking_id,
                did,
                ", ".join(conflict_reasons_final),
            )
            # ✅ A1: Incrémenter métrique
            increment_temporal_conflict_counter()
            # ✅ A1: Marquer le rejet avec conflict_penalty
            temporal_conflict_rejects.append(
                {
                    "booking_id": int(cast("Any", b.id)),
                    "driver_id": did,
                    "conflict_reasons": conflict_reasons_final,
                    "conflict_penalty": -9999.0,
                    "estimated_start_min": scheduled_min,
                }
            )
            continue

        # ✅ A1: Vérifier busy_until avec buffer configurable
        if not can_pool and strict_check and busy_until[did] > 0:
            required_free_time = busy_until[did] + post_trip_buffer
            if scheduled_min < required_free_time:
                conflict_reasons_final.append(
                    f"busy_until:{busy_until[did]}→{required_free_time}"
                )
                logger.warning(
                    (
                        "[DISPATCH] ⚠️ CONFLIT BUSY: Chauffeur #%s occupé "
                        "jusqu'à %smin (+%smin buffer = %smin), "
                        "course #%s démarre à %smin → SKIP"
                    ),
                    did,
                    busy_until[did],
                    post_trip_buffer,
                    required_free_time,
                    cand.booking_id,
                    scheduled_min,
                )
                continue

        # Si déjà pris (par un meilleur match urgent par ex.)
        if any(a.booking_id == int(cast("Any", b.id)) for a in assignments):
            continue

        assignments.append(cand)
        proposed_load[did] += 1
        fairness_effective[did] = fairness_effective.get(did, 0) + 1

        # ✅ CRITIQUE: Mettre à jour driver_scheduled_times IMMÉDIATEMENT
        # après l'assignation
        # pour que les courses suivantes dans le même batch voient cette assignation
        if scheduled_min not in driver_scheduled_times[did]:
            driver_scheduled_times[did].append(scheduled_min)

        # 🚗 Vérifier si c'est un regroupement avec une course existante
        is_pooled = False
        pooled_with = None
        for existing_time in driver_scheduled_times[did]:
            if abs(scheduled_min - existing_time) < settings.pooling.time_tolerance_min:
                # Trouver la course existante
                for assigned in [
                    a for a in assignments if a.driver_id == did and a != cand
                ]:
                    assigned_booking = next(
                        (
                            bk
                            for bk in bookings
                            if int(cast("Any", bk.id)) == assigned.booking_id
                        ),
                        None,
                    )
                    if assigned_booking and _can_be_pooled(
                        b, assigned_booking, settings
                    ):
                        is_pooled = True
                        pooled_with = assigned.booking_id
                        break
                if is_pooled:
                    break

        # ⏱️ CORRECTION: Utiliser durée OSRM réelle + temps de service
        duration_osrm = cand.estimated_finish_min - cand.estimated_start_min

        if is_pooled:
            # 🚗 REGROUPEMENT: Ajouter un détour supplémentaire pour le 2ème dropoff
            # Pickup commun → Dropoff 1 → Dropoff 2 (détour estimé)
            realistic_finish = (
                scheduled_min + duration_osrm + settings.pooling.max_detour_min
            )
            logger.info(
                (
                    "[POOLING] 🚗 Course #%s regroupée avec #%s → busy_until += %smin "
                    "détour"
                ),
                cand.booking_id,
                pooled_with,
                settings.pooling.max_detour_min,
            )
        else:
            realistic_finish = scheduled_min + duration_osrm

        busy_until[did] = max(busy_until[did], realistic_finish)

        # 📅 Enregistrer le scheduled_time RÉEL
        # (sauf si déjà enregistré pour regroupement)
        if scheduled_min not in driver_scheduled_times[did]:
            driver_scheduled_times[did].append(scheduled_min)

        # ⚡ Mettre à jour driver_last_dropoff avec la position de dropoff
        # de cette course
        # Cela permettra aux courses suivantes d'utiliser cette position
        # pour minimiser les trajets
        _, dropoff_coord = _booking_coords(b)
        driver_last_dropoff[did] = dropoff_coord

        pool_indicator = f" [GROUPÉ avec #{pooled_with}]" if is_pooled else ""
        assign_msg = (
            f"✅ Course #{cand.booking_id} → Chauffeur #{did} "
            f"(score: {sc:.2f}, start: {scheduled_min}min, "
            f"busy_until: {busy_until[did]}min){pool_indicator}"
        )
        logger.info("[DISPATCH] %s", assign_msg)

    # ⚡ Passe supplémentaire : réassigner les courses non assignées
    # avec les chauffeurs d'urgence
    # Prioriser les courses proches et rapides pendant le rush (13:30-14:30)
    allow_emergency_flag = problem.get(
        "allow_emergency", True
    )  # Par défaut, autoriser les urgences
    # ⚡ Définir les constantes de rush hour en dehors du bloc
    # pour éviter "possibly unbound"
    rush_start = 13 * 60 + 30  # 13:30
    rush_end = 14 * 60 + 30  # 14:30

    if unassigned and allow_emergency_flag:
        # ✅ Vérifier via driver_type (pas is_emergency)
        def _is_emergency_driver(driver):
            driver_type = getattr(driver, "driver_type", None)
            if not driver_type:
                return False
            driver_type_str = str(driver_type).strip().upper()
            if "." in driver_type_str:
                driver_type_str = driver_type_str.split(".")[-1]
            return driver_type_str == "EMERGENCY"

        emergency_drivers = [d for d in drivers if _is_emergency_driver(d)]
        if emergency_drivers:
            logger.info(
                (
                    "[DISPATCH] 🚨 Tentative de réassignation avec %d chauffeur(s) "
                    "d'urgence pour %d courses non assignées"
                ),
                len(emergency_drivers),
                len(unassigned),
            )

            # Filtrer les courses non assignées
            unassigned_bookings = [
                b for b in bookings if int(cast("Any", b.id)) in unassigned
            ]

            # Trier par priorité : rush hour (13:30-14:30) d'abord,
            # puis proximité au bureau
            def get_priority_for_emergency(b: Booking) -> Tuple[int, float]:
                scheduled_time_dt = getattr(b, "scheduled_time", None)
                if not scheduled_time_dt:
                    return (9999, 9999.0)  # Dernière priorité si pas d'heure

                # Calculer l'heure en minutes depuis minuit
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

                # Bonus si dans le rush (13:30-14:30 = 810-870 minutes)
                rush_start = 13 * 60 + 30  # 13:30
                rush_end = 14 * 60 + 30  # 14:30
                is_rush = rush_start <= scheduled_min <= rush_end
                priority_time = 0 if is_rush else 1000  # Priorité au rush

                # Calculer la distance au bureau pour prioriser les plus proches
                if company_coords:
                    p_coord, _ = _booking_coords(b)
                    distance_to_office = float(
                        haversine_minutes(
                            company_coords,
                            p_coord,
                            avg_kmh=25,
                            min_minutes=1,
                            max_minutes=180,
                        )
                    )
                else:
                    distance_to_office = 999.0

                return (priority_time, distance_to_office)

            # Trier par priorité (rush d'abord, puis distance)
            unassigned_bookings.sort(key=get_priority_for_emergency)

            # Essayer d'assigner avec les chauffeurs d'urgence
            for b in unassigned_bookings:
                b_id = int(cast("Any", b.id))
                best_emergency = None
                best_score = -9999.0

                for d_emg in emergency_drivers:
                    d_emg_id = int(cast("Any", d_emg.id))

                    # Calculer le score avec le chauffeur d'urgence
                    driver_window_emg = (
                        driver_windows[drivers.index(d_emg)]
                        if d_emg in drivers
                        else (0, 24 * 60)
                    )
                    sc_emg, _breakdown_emg, (est_s_emg, est_f_emg) = (
                        _score_driver_for_booking(
                            b,
                            d_emg,
                            driver_window_emg,
                            settings,
                            fairness_effective,
                            company_coords=company_coords,
                            preferred_driver_id=preferred_driver_id,
                        )
                    )

                    # Vérifier la faisabilité
                    if sc_emg <= SC_ZERO:
                        continue

                    # Vérifier les conflits temporels
                    scheduled_time_dt = getattr(b, "scheduled_time", None)
                    base_time = problem.get("base_time")
                    if base_time and scheduled_time_dt:
                        scheduled_dt_utc = to_utc(scheduled_time_dt)
                        base_dt_utc = to_utc(base_time)
                        delta = (
                            scheduled_dt_utc - base_dt_utc
                            if scheduled_dt_utc and base_dt_utc
                            else None
                        )
                        scheduled_min_emg = (
                            int(delta.total_seconds() // 60)
                            if delta
                            else (
                                scheduled_time_dt.hour * 60 + scheduled_time_dt.minute
                            )
                        )
                    else:
                        scheduled_min_emg = (
                            scheduled_time_dt.hour * 60 + scheduled_time_dt.minute
                            if scheduled_time_dt
                            else 0
                        )

                    # Vérifier les conflits
                    min_gap_minutes_emg = int(
                        getattr(settings.safety, "min_gap_minutes", 30)
                    )
                    has_conflict_emg = False
                    for existing_time in driver_scheduled_times.get(d_emg_id, []):
                        if abs(scheduled_min_emg - existing_time) < min_gap_minutes_emg:
                            has_conflict_emg = True
                            break

                    if has_conflict_emg:
                        continue

                    # Bonus si dans le rush (13:30-14:30)
                    # rush_start et rush_end sont définis avant la boucle
                    if rush_start <= scheduled_min_emg <= rush_end:
                        sc_emg += 1.0  # Bonus fort pour rush

                    if sc_emg > best_score:
                        best_score = sc_emg
                        best_emergency = (
                            d_emg,
                            sc_emg,
                            est_s_emg,
                            est_f_emg,
                            scheduled_min_emg,
                        )

                if best_emergency:
                    d_emg, sc_emg, est_s_emg, est_f_emg, scheduled_min_emg = (
                        best_emergency
                    )
                    d_emg_id = int(cast("Any", d_emg.id))

                    cand_emg = HeuristicAssignment(
                        booking_id=b_id,
                        driver_id=d_emg_id,
                        score=sc_emg,
                        reason="emergency_reassignment",
                        estimated_start_min=est_s_emg,
                        estimated_finish_min=est_f_emg,
                    )

                    assignments.append(cand_emg)
                    proposed_load[d_emg_id] += 1
                    fairness_effective[d_emg_id] = (
                        fairness_effective.get(d_emg_id, 0) + 1
                    )
                    unassigned.remove(b_id)

                    # Mettre à jour busy_until et scheduled_times
                    duration_osrm_emg = est_f_emg - est_s_emg
                    realistic_finish_emg = scheduled_min_emg + duration_osrm_emg
                    busy_until[d_emg_id] = max(
                        busy_until.get(d_emg_id, 0), realistic_finish_emg
                    )

                    if d_emg_id not in driver_scheduled_times:
                        driver_scheduled_times[d_emg_id] = []
                    if scheduled_min_emg not in driver_scheduled_times[d_emg_id]:
                        driver_scheduled_times[d_emg_id].append(scheduled_min_emg)

                    logger.info(
                        (
                            "[DISPATCH] 🚨 Course #%s réassignée avec chauffeur "
                            "d'urgence #%s (score: %.2f, rush: %s)"
                        ),
                        b_id,
                        d_emg_id,
                        sc_emg,
                        rush_start <= scheduled_min_emg <= rush_end,
                    )

    debug = {
        "proposed_load": proposed_load,
        "fairness_counts": fairness_counts,
        "fairness_baseline": fairness_baseline,
        "urgent_count": len(urgent),
        "regular_count": len(regular),
        "max_cap": max_cap,
        "busy_until": busy_until,  # 📅 Pour transmettre au fallback
        "driver_scheduled_times": driver_scheduled_times,  # 📅 Pour transmettre
        # au fallback
        "temporal_conflict_rejects": temporal_conflict_rejects,  # ✅ A1: Rejets
        # avec conflict_penalty
    }

    logger.info(
        "[DISPATCH] 📊 Résultat: %s assignations, %s non-assignées",
        len(assignments),
        len(unassigned),
    )
    logger.info("[DISPATCH] 📈 Charge par chauffeur: %s", dict(proposed_load))

    return HeuristicResult(
        assignments=assignments, unassigned_booking_ids=unassigned, debug=debug
    )


# -------------------------------------------------------------------
# Assignation "retours urgents" (pré-tri) : réguliers d'abord, urgence si nécessaire
# -------------------------------------------------------------------


def assign_urgent(
    problem: Dict[str, Any],
    urgent_booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> HeuristicResult:
    if not problem or not urgent_booking_ids:
        return HeuristicResult(
            assignments=[], unassigned_booking_ids=[], debug={"reason": "no_urgent"}
        )

    allow_emergency = bool(getattr(settings.emergency, "allow_emergency", True))
    logger.info(
        "[Heuristics] assign_urgent start urgent=%s allow_emergency=%s",
        len(urgent_booking_ids),
        allow_emergency,
    )

    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts_raw: Dict[int, int] = problem.get("fairness_counts", {})
    fairness_counts, fairness_baseline = baseline_and_cap_loads(fairness_counts_raw)
    problem["fairness_counts"] = fairness_counts
    problem["fairness_baseline"] = fairness_baseline
    company_coords: Tuple[float, float] | None = problem.get(
        "company_coords"
    )  # ⚡ Coordonnées du bureau
    driver_load_multipliers: Dict[int, float] = problem.get(
        "driver_load_multipliers", {}
    )  # ⚡ Multiplicateurs de charge
    preferred_driver_id: int | None = problem.get(
        "preferred_driver_id"
    )  # ⚡ Chauffeur préféré
    max_cap = settings.solver.max_bookings_per_driver

    # ⚡ Calculer les caps ajustés selon les préférences de charge par chauffeur
    def get_adjusted_max_cap(driver_id: int) -> int:
        """Retourne le cap maximum ajusté pour un chauffeur selon ses préférences."""
        multiplier = driver_load_multipliers.get(driver_id, 1.0)
        return int(max_cap * multiplier)

    by_id: Dict[int, Booking] = {int(cast("Any", b.id)): b for b in bookings}
    driver_index: Dict[int, int] = {
        int(cast("Any", d.id)): i for i, d in enumerate(drivers)
    }
    proposed_load: Dict[int, int] = {int(cast("Any", d.id)): 0 for d in drivers}
    busy_until: Dict[int, int] = {int(cast("Any", d.id)): 0 for d in drivers}
    fairness_effective_local: Dict[int, int] = {
        int(cast("Any", d.id)): fairness_counts.get(int(cast("Any", d.id)), 0)
        + proposed_load.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }

    def _choose_best(b: Booking, regular_only: bool) -> HeuristicAssignment | None:
        best: Tuple[float, HeuristicAssignment] | None = None
        norm_loads = _normalized_loads(fairness_effective_local)
        emergency_candidate_logged = False
        for d in drivers:
            # Évite l'ouverture des chauffeurs d'urgence si regular_only
            driver_type_attr = getattr(d, "driver_type", None)
            driver_type_str = str(driver_type_attr or "").strip().upper()
            if "." in driver_type_str:
                driver_type_str = driver_type_str.split(".")[-1]
            if regular_only and driver_type_str == "EMERGENCY":
                continue
            # Cap fairness (ajusté selon préférences)
            did = int(cast("Any", d.id))
            adjusted_cap = get_adjusted_max_cap(did)
            if fairness_effective_local.get(did, 0) >= adjusted_cap:
                continue
            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            sc, _br, (est_s, est_f) = _score_driver_for_booking(
                b,
                d,
                dw,
                settings,
                norm_loads,
                company_coords=company_coords,
                preferred_driver_id=preferred_driver_id,
            )
            if est_s < busy_until[did]:
                continue
            if sc <= SC_ZERO:
                continue
            # Bonus stabilité si déjà ASSIGNED à ce driver
            if _is_booking_assigned(b) and (_current_driver_id(b) == did):
                sc += 0.3

            # Malus sur "emergency" pour ne l'utiliser qu'en dernier recours
            if driver_type_str == "EMERGENCY":
                emergency_penalty = float(
                    getattr(settings.emergency, "emergency_penalty", 900.0)
                )
                malus = -(
                    emergency_penalty / 180.0
                )  # 900 / 180 = 5.0, 500 / 180 = 2.78
                sc += malus
                if not regular_only and not emergency_candidate_logged:
                    logger.info(
                        (
                            "[Heuristics] Emergency driver candidate driver_id=%s "
                            "booking_id=%s allow_emergency=%s score=%.2f duration=%s"
                        ),
                        did,
                        getattr(b, "id", None),
                        allow_emergency,
                        sc,
                        est_f - est_s,
                    )
                    emergency_candidate_logged = True

        return best[1] if best else None

    # Ordonner les urgents par horaire (si dispo)
    ordered: List[Booking] = []
    for bid in urgent_booking_ids:
        b = by_id.get(int(cast("Any", bid)))
        if b:
            ordered.append(b)
    ordered.sort(
        key=lambda x: sort_key_utc(cast("Any", getattr(x, "scheduled_time", None)))
    )

    assignments: List[HeuristicAssignment] = []
    unassigned: List[int] = []

    for b in ordered:
        # 1) RÉguliers d'abord
        chosen = _choose_best(b, regular_only=True)
        # 2) Sinon, autoriser l'urgence si activée
        if not chosen and settings.emergency.allow_emergency_drivers:
            chosen = _choose_best(b, regular_only=False)
        if chosen:
            assignments.append(chosen)
            did = int(chosen.driver_id)
            proposed_load[did] += 1
            fairness_effective_local[did] = fairness_effective_local.get(did, 0) + 1
            busy_until[did] = max(busy_until[did], chosen.estimated_finish_min)
        else:
            unassigned.append(int(cast("Any", b.id)))

    debug = {
        "urgent_input": urgent_booking_ids,
        "picked": [int(a.booking_id) for a in assignments],
        "unassigned": unassigned,
        "proposed_load": proposed_load,
        "fairness_baseline": fairness_baseline,
    }
    return HeuristicResult(
        assignments=assignments, unassigned_booking_ids=unassigned, debug=debug
    )


# -------------------------------------------------------------------
# Fallback simple : "closest feasible" pour le reliquat non couvert
# -------------------------------------------------------------------
def closest_feasible(
    problem: Dict[str, Any],
    booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> HeuristicResult:
    if not problem or not booking_ids:
        return HeuristicResult(
            assignments=[],
            unassigned_booking_ids=[],
            debug={"reason": "empty_fallback"},
        )

    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts_raw: Dict[int, int] = problem.get("fairness_counts", {})
    fairness_counts, fairness_baseline = baseline_and_cap_loads(fairness_counts_raw)
    problem["fairness_counts"] = fairness_counts
    problem.setdefault("fairness_baseline", fairness_baseline)
    max_cap = settings.solver.max_bookings_per_driver
    preferred_driver_id: int | None = problem.get(
        "preferred_driver_id"
    )  # ⚡ Chauffeur préféré

    by_id: Dict[int, Booking] = {int(cast("Any", b.id)): b for b in bookings}
    driver_index: Dict[int, int] = {
        int(cast("Any", d.id)): i for i, d in enumerate(drivers)
    }

    # 📅 RÉCUPÉRER les états de l'heuristique principale si disponibles
    # (pour éviter les conflits)
    previous_busy = problem.get("busy_until", {})
    previous_times = problem.get("driver_scheduled_times", {})
    previous_load = problem.get("proposed_load", {})

    proposed_load: Dict[int, int] = {
        int(cast("Any", d.id)): previous_load.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }
    fairness_effective_fb: Dict[int, int] = {
        int(cast("Any", d.id)): fairness_counts.get(int(cast("Any", d.id)), 0)
        + proposed_load.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }
    busy_until: Dict[int, int] = {
        int(cast("Any", d.id)): previous_busy.get(int(cast("Any", d.id)), 0)
        for d in drivers
    }

    # 📅 Traçabilité des temps exacts assignés à chaque chauffeur
    # (pour détecter les doublons d'heure)
    driver_scheduled_times: Dict[int, List[int]] = {
        int(cast("Any", d.id)): list(previous_times.get(int(cast("Any", d.id)), []))
        for d in drivers
    }

    # ✅ FIX: Vérifier la cohérence de l'état fallback injecté
    import os

    is_testing = False
    try:
        is_testing = os.getenv("FLASK_CONFIG") == "testing"
        try:
            from flask import current_app  # pyright: ignore[reportMissingImports]

            is_testing = is_testing or current_app.config.get("TESTING", False)
        except RuntimeError:
            pass
    except Exception:
        pass

    # Vérifier cohérence entre busy_until et scheduled_times
    inconsistencies = []
    for did in busy_until:
        if did in driver_scheduled_times:
            scheduled_list = driver_scheduled_times[did]
            if scheduled_list:
                max_scheduled = max(scheduled_list)
                busy = busy_until.get(did, 0)
                # busy_until devrait être >= au dernier scheduled_time
                if busy > 0 and max_scheduled > busy:
                    inconsistencies.append(
                        f"Driver {did}: busy_until={busy} < "
                        + f"max_scheduled={max_scheduled}"
                    )
                # proposed_load devrait correspondre au nombre de scheduled_times
                proposed = proposed_load.get(did, 0)
                if proposed != len(scheduled_list):
                    inconsistencies.append(
                        f"Driver {did}: proposed_load={proposed} != "
                        + f"len(scheduled_times)={len(scheduled_list)}"
                    )

    if inconsistencies:
        log_level = logger.debug if is_testing else logger.warning
        log_level(
            "[FALLBACK] ⚠️ Incohérences détectées dans l'état fallback: %s",
            "; ".join(inconsistencies),
        )

    # ✅ FIX: Réduire le niveau de log en mode testing
    # (normal, état injecté depuis heuristique principale)
    log_level_info = logger.debug if is_testing else logger.warning
    log_level_info(
        "[FALLBACK] 📥 Récupération état précédent: busy_until=%s, scheduled_times=%s",
        dict(busy_until),
        dict(driver_scheduled_times),
    )
    if preferred_driver_id:
        logger.info(
            "[FALLBACK] 🎯 Chauffeur préféré détecté: %s - bonus +3.0 sera appliqué",
            preferred_driver_id,
        )

    assignments: List[HeuristicAssignment] = []
    unassigned: List[int] = []

    min_effective_load = (
        min(fairness_effective_fb.values()) if fairness_effective_fb else 0
    )

    for bid in booking_ids:
        b = by_id.get(int(cast("Any", bid)))
        if not b:
            continue
        best: Tuple[float, HeuristicAssignment] | None = None
        normalized_fb = _normalized_loads(fairness_effective_fb)
        for d in drivers:
            did = int(cast("Any", d.id))
            # Cap ajusté selon préférences (si disponible)
            adjusted_cap = max_cap
            if "driver_load_multipliers" in problem:
                multiplier = problem["driver_load_multipliers"].get(did, 1.0)
                adjusted_cap = int(max_cap * multiplier)
            if fairness_effective_fb.get(did, 0) >= adjusted_cap:
                continue

            effective_load = fairness_effective_fb.get(did, 0)
            allowed_gap = MAX_FAIRNESS_GAP
            if preferred_driver_id and did == preferred_driver_id:
                allowed_gap += PREFERRED_EXTRA_GAP
            if (effective_load - min_effective_load) > allowed_gap:
                logger.debug(
                    "[FALLBACK] ⛔ Skip driver #%s (load=%s, min=%s, allowed=%s)",
                    did,
                    effective_load,
                    min_effective_load,
                    allowed_gap,
                )
                continue
            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            company_coords = problem.get("company_coords")  # ⚡ Coordonnées du bureau
            sc, _br, (est_s, est_f) = _score_driver_for_booking(
                b,
                d,
                dw,
                settings,
                normalized_fb,
                company_coords=company_coords,
                preferred_driver_id=preferred_driver_id,
            )

            # 🚫 CORRECTION CRITIQUE: Calculer scheduled_min (heure demandée par client)
            scheduled_time_dt = getattr(b, "scheduled_time", None)
            if not scheduled_time_dt:
                continue

            base_time = problem.get("base_time")
            if base_time:
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = (
                    scheduled_dt_utc - base_dt_utc
                    if scheduled_dt_utc and base_dt_utc
                    else None
                )
                scheduled_min = (
                    int(delta.total_seconds() // 60)
                    if delta
                    else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
                )
            else:
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

            # ✅ A1: VÉRIFICATION CONFLITS TEMPORELS (closest_feasible fallback)
            min_gap_minutes = int(getattr(settings.safety, "min_gap_minutes", 30))
            post_trip_buffer = int(getattr(settings.safety, "post_trip_buffer_min", 15))
            strict_check = bool(
                getattr(
                    settings.features, "enable_strict_temporal_conflict_check", True
                )
            )

            has_conflict = False
            can_pool = False
            conflict_reasons_fb = []

            for existing_time in driver_scheduled_times[did]:
                gap_minutes = abs(scheduled_min - existing_time)
                if gap_minutes < min_gap_minutes:
                    # Chercher la course existante pour vérifier si
                    # regroupement possible
                    existing_booking = None
                    for assigned in [a for a in assignments if a.driver_id == did]:
                        assigned_booking = by_id.get(int(assigned.booking_id))
                        if assigned_booking:
                            assigned_time_dt = getattr(
                                assigned_booking, "scheduled_time", None
                            )
                            if assigned_time_dt:
                                if base_time:
                                    assigned_dt_utc = to_utc(assigned_time_dt)
                                    base_dt_utc = to_utc(base_time)
                                    delta = (
                                        assigned_dt_utc - base_dt_utc
                                        if assigned_dt_utc and base_dt_utc
                                        else None
                                    )
                                    assigned_min = (
                                        int(delta.total_seconds() // 60)
                                        if delta
                                        else (
                                            assigned_time_dt.hour * 60
                                            + assigned_time_dt.minute
                                        )
                                    )
                                else:
                                    assigned_min = (
                                        assigned_time_dt.hour * 60
                                        + assigned_time_dt.minute
                                    )

                                if assigned_min == existing_time:
                                    existing_booking = assigned_booking
                                    break

                    # Vérifier si regroupement possible
                    if existing_booking and _can_be_pooled(
                        b, existing_booking, settings
                    ):
                        can_pool = True
                        logger.info(
                            (
                                "[POOLING] 🚗 [FALLBACK] Course #%s peut être "
                                "regroupée avec #%s (chauffeur #%s)"
                            ),
                            bid,
                            existing_booking.id,
                            did,
                        )
                        break

                    conflict_reasons_fb.append(f"time_gap:{gap_minutes}min")
                    # ✅ FIX: Réduire le niveau de log en mode testing
                    # (attendu dans fallback, géré correctement)
                    is_testing_fb = False
                    try:
                        is_testing_fb = os.getenv("FLASK_CONFIG") == "testing"
                        try:
                            from flask import (  # pyright: ignore[reportMissingImports]
                                current_app,
                            )

                            is_testing_fb = is_testing_fb or current_app.config.get(
                                "TESTING", False
                            )
                        except RuntimeError:
                            pass
                    except Exception:
                        pass

                    log_level_fb = logger.debug if is_testing_fb else logger.warning
                    log_level_fb(
                        (
                            "[FALLBACK] ⚠️ CONFLIT: Chauffeur #%s a course à %smin, "
                            "course #%s à %smin (écart: %smin) → SKIP"
                        ),
                        did,
                        existing_time,
                        bid,
                        scheduled_min,
                        gap_minutes,
                    )
                    has_conflict = True
                    break

            if has_conflict and not can_pool:
                logger.warning(
                    "[FALLBACK] 🔴 Conflit temporel booking #%s + driver #%s: %s",
                    bid,
                    did,
                    ", ".join(conflict_reasons_fb),
                )
                # ✅ A1: Incrémenter métrique
                increment_temporal_conflict_counter()
                continue

            # ✅ A1: VÉRIFICATION 2 busy_until avec buffer configurable
            if not can_pool and strict_check and busy_until[did] > 0:
                required_free_time = busy_until[did] + post_trip_buffer
                if scheduled_min < required_free_time:
                    conflict_reasons_fb.append(
                        f"busy_until:{busy_until[did]}→{required_free_time}"
                    )
                    logger.warning(
                        (
                            "[FALLBACK] ⚠️ BUSY: Chauffeur #%s occupé jusqu'à %smin "
                            "(+%smin buffer = %smin), course #%s démarre à %smin → SKIP"
                        ),
                        did,
                        busy_until[did],
                        post_trip_buffer,
                        required_free_time,
                        bid,
                        scheduled_min,
                    )
                    continue

            # 🚗 REGROUPEMENT : Si détecté, assigner IMMÉDIATEMENT
            # sans chercher d'autres chauffeurs
            if can_pool:
                logger.warning(
                    (
                        "[POOLING] 🚗 [FALLBACK] Course #%s FORCÉE au chauffeur #%s "
                        "(regroupement prioritaire)"
                    ),
                    bid,
                    did,
                )
                best = (
                    sc,
                    HeuristicAssignment(
                        booking_id=int(cast("Any", b.id)),
                        driver_id=did,
                        score=sc,
                        reason="fallback_pooled",
                        estimated_start_min=est_s,
                        estimated_finish_min=est_f,
                    ),
                )
                break  # ⚠️ CRUCIAL: Sortir de la boucle des chauffeurs

            # 🚫 VÉRIFICATION 3: Score négatif
            if sc <= SC_ZERO:
                continue

            # 🎯 Bonus/malus pour équilibrer la charge
            current_load = fairness_effective_fb.get(did, 0)

            # Pénalité progressive douce
            if current_load <= CURRENT_LOAD_THRESHOLD:
                load_penalty = current_load * 0.1
            elif current_load == CURRENT_LOAD_THRESHOLD + 1:
                load_penalty = 0.3
            elif current_load == CURRENT_LOAD_THRESHOLD + 2:
                load_penalty = 0.6
            else:
                load_penalty = 1 + (current_load - 5) * 0.5

            sc -= load_penalty

            # ⚡ CORRECTION: Calculer min_load avec fairness_counts inclus
            # (charge totale réelle)
            current_loads_all = [
                fairness_effective_fb.get(int(cast("Any", d.id)), 0) for d in drivers
            ]
            min_load = min(current_loads_all) if current_loads_all else 0
            if current_load == min_load:
                sc += 0.8

            # Bonus stabilité si déjà ASSIGNED à ce driver
            if _is_booking_assigned(b) and (_current_driver_id(b) == did):
                sc += 0.2

            cand = HeuristicAssignment(
                booking_id=int(cast("Any", b.id)),
                driver_id=did,
                score=sc,
                reason="fallback_closest",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if best is None or sc > best[0]:
                best = (sc, cand)
        if best:
            chosen = best[1]
            assignments.append(chosen)
            did2 = int(chosen.driver_id)
            proposed_load[did2] += 1
            fairness_effective_fb[did2] = fairness_effective_fb.get(did2, 0) + 1

            # ⏱️ CORRECTION: Calculer scheduled_min et utiliser durée OSRM réelle
            scheduled_time_dt = getattr(b, "scheduled_time", None)
            base_time = problem.get("base_time")
            if base_time and scheduled_time_dt:
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = (
                    scheduled_dt_utc - base_dt_utc
                    if scheduled_dt_utc and base_dt_utc
                    else None
                )
                scheduled_min = (
                    int(delta.total_seconds() // 60)
                    if delta
                    else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
                )
            else:
                scheduled_min = (
                    scheduled_time_dt.hour * 60 + scheduled_time_dt.minute
                    if scheduled_time_dt
                    else chosen.estimated_start_min
                )

            # 🚗 Vérifier si c'est un regroupement avec une course existante
            is_pooled = False
            pooled_with = None
            for existing_time in driver_scheduled_times[did2]:
                if (
                    abs(scheduled_min - existing_time)
                    < settings.pooling.time_tolerance_min
                ):
                    # Trouver la course existante
                    for assigned in [
                        a for a in assignments if a.driver_id == did2 and a != chosen
                    ]:
                        assigned_booking = by_id.get(int(assigned.booking_id))
                        if assigned_booking and _can_be_pooled(
                            b, assigned_booking, settings
                        ):
                            is_pooled = True
                            pooled_with = assigned.booking_id
                            break
                    if is_pooled:
                        break

            # Calculer la durée réelle de la course selon OSRM
            duration_osrm = chosen.estimated_finish_min - chosen.estimated_start_min

            if is_pooled:
                # 🚗 REGROUPEMENT: Ajouter détour pour 2ème dropoff
                realistic_finish = (
                    scheduled_min + duration_osrm + settings.pooling.max_detour_min
                )
                logger.info(
                    (
                        "[POOLING] 🚗 [FALLBACK] Course #%s regroupée avec #%s → "
                        "+%smin détour"
                    ),
                    chosen.booking_id,
                    pooled_with,
                    settings.pooling.max_detour_min,
                )
            else:
                realistic_finish = scheduled_min + duration_osrm

            busy_until[did2] = max(busy_until[did2], realistic_finish)

            # 📅 Enregistrer le scheduled_time RÉEL
            # (sauf si déjà enregistré pour regroupement)
            if scheduled_min not in driver_scheduled_times[did2]:
                driver_scheduled_times[did2].append(scheduled_min)

            pool_indicator = f" [GROUPÉ avec #{pooled_with}]" if is_pooled else ""
            logger.info(
                (
                    "[FALLBACK] ✅ Course #%s → Chauffeur #%s (score: %.2f, "
                    "start: %smin, busy_until: %smin)%s"
                ),
                chosen.booking_id,
                did2,
                best[0],
                scheduled_min,
                busy_until[did2],
                pool_indicator,
            )
        else:
            unassigned.append(int(cast("Any", b.id)))
            logger.warning(
                (
                    "[FALLBACK] ❌ Course #%s impossible à assigner "
                    "(aucun chauffeur disponible)"
                ),
                bid,
            )

    debug = {
        "input_unassigned": booking_ids,
        "picked": [int(a.booking_id) for a in assignments],
        "still_unassigned": unassigned,
        "proposed_load": proposed_load,
        "busy_until": busy_until,
        "driver_scheduled_times": driver_scheduled_times,
        "fairness_counts": fairness_counts,
        "fairness_baseline": problem.get("fairness_baseline", fairness_baseline),
    }
    return HeuristicResult(
        assignments=assignments, unassigned_booking_ids=unassigned, debug=debug
    )


# -------------------------------------------------------------------
# Recommandations post-run : estimation des attentes / ressources
# -------------------------------------------------------------------
def estimate_wait_or_require_extra(
    problem: Dict[str, Any],
    remaining_booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> Dict[str, Any]:
    """Donne des indications simples pour les courses non assignées :
    - ETA approximatif depuis le chauffeur le plus proche (Haversine).
    - Lateness estimée vs. horaire (min).
    - Suggestions: "ouvrir urgences", "ajouter chauffeur", "élargir fenêtres".
    """
    if not problem or not remaining_booking_ids:
        return {"summary": "no_remaining", "items": []}

    bookings: List[Booking] = problem.get("bookings", [])
    drivers: List[Driver] = problem.get("drivers", [])
    by_id: Dict[int, Booking] = {int(cast("Any", b.id)): b for b in bookings}

    # Coords chauffeurs (courantes si dispo, sinon latitude/longitude)
    driver_coords: List[Tuple[float, float]] = []
    for d in drivers:
        cur_lat = getattr(d, "current_lat", None)
        cur_lon = getattr(d, "current_lon", None)
        if cur_lat is not None and cur_lon is not None:
            driver_coords.append((float(cur_lat), float(cur_lon)))
            continue
        lat = getattr(d, "latitude", None)
        lon = getattr(d, "longitude", None)
        if lat is not None and lon is not None:
            driver_coords.append((float(lat), float(lon)))
        else:
            driver_coords.append((46.2044, 6.1432))  # Genève

    now = now_local()
    items: List[Dict[str, Any]] = []
    avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25))
    # mapping vers la clé réellement présente dans TimeSettings
    buf_min = int(getattr(getattr(settings, "time", None), "pickup_buffer_min", 5))

    for bid in remaining_booking_ids:
        b = by_id.get(int(cast("Any", bid)))
        if not b:
            continue
        try:
            pick = (
                float(b.pickup_lat),  # type: ignore[reportArgumentType]
                float(b.pickup_lon),  # type: ignore[reportArgumentType]
            )
        except Exception:
            # si coordonnées manquent, on saute (devrait être enrichi par
            # data.py)
            continue

        # ETA min depuis n'importe quel chauffeur
        etas = [
            haversine_minutes(dc, pick, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240)
            for dc in driver_coords
        ] or [999]
        eta_min = min(etas)

        st = cast("Any", getattr(b, "scheduled_time", None))
        try:
            # minutes_from_now gère déjà, mais gardons simple
            dt = st if isinstance(st, datetime) else now
            mins_to_pickup = minutes_from_now(dt)
        except Exception:
            mins_to_pickup = 0
        lateness = int(max(0, (eta_min - mins_to_pickup)))
        items.append(
            {
                "booking_id": int(cast("Any", b.id)),
                "eta_min": int(eta_min),
                "lateness_min": int(lateness - buf_min) if lateness > buf_min else 0,
            }
        )

    # Synthèse basique
    allow_emg = bool(
        getattr(getattr(settings, "emergency", None), "allow_emergency_drivers", True)
    )
    suggestions: List[str] = []
    if not allow_emg:
        suggestions.append("Autoriser les chauffeurs d'urgence pour absorber le pic.")
    if len(drivers) == 0:
        suggestions.append("Aucun chauffeur disponible : en ajouter au planning.")
    elif any(it.get("lateness_min", 0) > LATENESS_THRESHOLD_MIN for it in items):
        suggestions.append(
            "Ajouter au moins 1 chauffeur sur le créneau ou élargir "
            + "les fenêtres de temps."
        )
    elif any(it.get("lateness_min", 0) > 0 for it in items):
        suggestions.append("Élargir légèrement les fenêtres ou ajuster les priorités.")

    return {"summary": "ok", "items": items, "suggestions": suggestions}
