# backend/services/unified_dispatch/heuristics.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, cast

from models import Booking, BookingStatus, Driver
from services.unified_dispatch.settings import Settings
from shared.geo_utils import haversine_distance
from shared.geo_utils import haversine_distance_meters as _haversine_distance
from shared.time_utils import minutes_from_now, now_local, sort_key_utc

DEFAULT_SETTINGS = Settings()

logger = logging.getLogger(__name__)

def _can_be_pooled(b1: Booking, b2: Booking, settings: Settings) -> bool:
    """Vérifie si deux courses peuvent être regroupées (même pickup, même heure)"""
    if not settings.pooling.enabled:
        return False

    # Vérifier que les deux courses ont scheduled_time
    t1 = getattr(b1, 'scheduled_time', None)
    t2 = getattr(b2, 'scheduled_time', None)
    if not t1 or not t2:
        return False

    # Vérifier que les heures sont proches (±settings.pooling.time_tolerance_min)
    time_diff_min = abs((t1 - t2).total_seconds() / 60)
    if time_diff_min > settings.pooling.time_tolerance_min:
        return False

    # Vérifier que les pickups sont proches (distance GPS)
    lat1 = getattr(b1, 'pickup_lat', None)
    lon1 = getattr(b1, 'pickup_lon', None)
    lat2 = getattr(b2, 'pickup_lat', None)
    lon2 = getattr(b2, 'pickup_lon', None)

    if not all([lat1, lon1, lat2, lon2]):
        # Fallback : comparer les adresses textuellement
        addr1 = getattr(b1, 'pickup_location', '').lower().replace(' ', '')
        addr2 = getattr(b2, 'pickup_location', '').lower().replace(' ', '')
        # Ignorer les différences mineures (majuscules, espaces)
        return bool(addr1 and addr2 and addr1 == addr2)

    # Calculer la distance GPS (vérifier que ce ne sont pas des None)
    lat1_safe = float(lat1) if lat1 is not None else 0.0
    lon1_safe = float(lon1) if lon1 is not None else 0.0
    lat2_safe = float(lat2) if lat2 is not None else 0.0
    lon2_safe = float(lon2) if lon2 is not None else 0.0
    distance_m = _haversine_distance(lat1_safe, lon1_safe, lat2_safe, lon2_safe)

    if distance_m <= settings.pooling.pickup_distance_m:
        logger.info(f"[POOLING] 🚗 Courses #{b1.id} et #{b2.id} peuvent être regroupées (même pickup à {distance_m:.0f}m, même heure)")
        return True

    return False

# ⏱️ Temps de service RÉELS (selon utilisateur) - maintenant paramétrables via settings
# PICKUP_SERVICE_MIN, DROPOFF_SERVICE_MIN, MIN_TRANSITION_MARGIN_MIN, etc.
# sont maintenant accessibles via settings.service_times.* et settings.pooling.*

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


    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialisation compatible avec le contrat Assignment côté API.
        - 'estimated_*' sont renvoyés en datetimes ISO basés sur 'now_local()' + minutes estimées.
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
    avg_kmh: float = 40.0,
    *,
    min_minutes: int = 1,
    max_minutes: int | None = None,
    fallback_speed_kmh: float = 30.0,
) -> int:
    """
    Estime le temps de trajet (en minutes, arrondi à l'entier supérieur) entre
    deux coordonnées (lat, lon) en utilisant la formule de Haversine et une
    vitesse moyenne `avg_kmh`.

    - Clamp les lat/lon dans les bornes valides.
    - Gère les vitesses non valides (0/NaN/inf) via `fallback_speed_kmh`.
    - Applique un plancher `min_minutes` (par défaut 1) et un plafond optionnel `max_minutes`.

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
    lat1 = max(-90.0, min(90.0, lat1))
    lat2 = max(-90.0, min(90.0, lat2))
    lon1 = ((lon1 + 180.0) % 360.0) - 180.0  # normalise dans [-180, 180)
    lon2 = ((lon2 + 180.0) % 360.0) - 180.0

    # Sécurité vitesse
    if not (math.isfinite(avg_kmh) and avg_kmh > 0.0):
        avg_kmh = fallback_speed_kmh
    if not (math.isfinite(avg_kmh) and avg_kmh > 0.0):
        # Ultime garde-fou
        avg_kmh = 30.0

    # Haversine (distance en km) - Import centralisé
    dist_km = haversine_distance(lat1, lon1, lat2, lon2)

    # Si quasi le même point, temps minimal
    if dist_km < 1e-3:  # ~1 mètre
        minutes = 0
    else:
        time_hours = dist_km / avg_kmh
        minutes = int(math.ceil(time_hours * 60.0))

    # Appliquer plancher/plafond
    minutes = max(min_minutes, minutes)
    if max_minutes is not None:
        minutes = min(max_minutes, minutes)

    return minutes

def _py_int(v: Any) -> int | None:
    try:
        return int(cast(Any, v)) if v is not None else None
    except Exception:
        return None

def _current_driver_id(b: Booking) -> int | None:
    return _py_int(getattr(b, "driver_id", None))

def _driver_current_coord(d: Driver) -> Tuple[float, float]:
    # On assume que data.py a mis à jour current_lat/current_lon
    cur_lat = getattr(d, "current_lat", None)
    cur_lon = getattr(d, "current_lon", None)
    if cur_lat is not None and cur_lon is not None:
        return (float(cast(Any, cur_lat)), float(cast(Any, cur_lon)))
    # fallback sur base chauffeur
    lat = getattr(d, "latitude", None)
    lon = getattr(d, "longitude", None)
    if lat is not None and lon is not None:
        return (float(cast(Any, lat)), float(cast(Any, lon)))
    # fallback Genève
    return (46.2044, 6.1432)


def _booking_coords(b: Booking) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (float(cast(Any, b.pickup_lat)), float(cast(Any, b.pickup_lon))),
        (float(cast(Any, b.dropoff_lat)), float(cast(Any, b.dropoff_lon))),
    )

def _is_booking_assigned(b: Booking) -> bool:
    try:
        s = cast(Any, getattr(b, "status", None))
        # compare à l’enum (ou à sa value) pour éviter ColumnElement
        return (s == BookingStatus.ASSIGNED) or (getattr(s, "value", None) == BookingStatus.ASSIGNED.value)
    except Exception:
        return False

def _priority_weight(b: Booking, weights: Dict[str, float]) -> float:
    """
    Calcule une "priorité" contextuelle :
    - médical/hôpital => +,
    - VIP/fragile (si vous avez un flag) => +,
    - retard potentiel (pickup imminent) => +,
    - retour déclenché à la demande => + léger (l’urgent est géré à part).
    """
    score = 0.0

    # Exemples de signaux — adaptez selon vos champs réels:
    if getattr(b, "medical_facility", None):
        score += weights.get("medical", 0.6)

    if getattr(b, "hospital_service", False):
        score += weights.get("hospital", 0.4)

    # retard potentiel
    mins = minutes_from_now(getattr(b, "scheduled_time", None))
    if mins <= 20:
        score += weights.get("time_pressure", 0.5)
    elif mins <= 40:
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
    thr = cast(Any, getattr(settings.emergency, "return_urgent_threshold_min",
                            getattr(settings.emergency, "emergency_threshold_min", 30)))
    return mins <= int(thr)


def _driver_fairness_penalty(driver_id: int, fairness_counts: Dict[int, int]) -> float:
    """
    Plus le chauffeur a déjà de courses aujourd'hui, plus la pénalité augmente.
    Renvoie une valeur [0..1] (à soustraire au score final).
    """
    cnt = fairness_counts.get(driver_id, 0)
    if cnt <= 0:
        return 0.0
    # échelle simple : 1 course = 0.05, 5 courses = 0.25, cap à 0.4
    return min(0.4, 0.05 * cnt)


def _regular_driver_bonus(b: Booking, d: Driver) -> float:
    """
    Bonus si le driver est "régulier" du client (ex: même driver_id référencé
    sur les dernières courses du client). Ici placeholder: si already assigned
    au même chauffeur, neutre (on évite de casser la relation).
    """
    try:
        bid_raw = cast(Any, getattr(b, "driver_id", None))
        did_raw = cast(Any, getattr(d, "id", None))
        bid = int(bid_raw) if bid_raw is not None else None
        did = int(did_raw) if did_raw is not None else None
    except Exception:
        return 0.0
    if bid is not None and did is not None and bid == did:
        return 0.15
    return 0.0


def _check_driver_window_feasible(driver_window: Tuple[int, int], est_start_min: int, est_finish_min: int) -> bool:
    start_w, end_w = driver_window

    # ⚠️ CORRECTION CRITIQUE : driver_window (0-1440) représente la journée du chauffeur
    # mais est_start_min est en "minutes depuis maintenant"
    # Pour les courses futures (demain+), la fenêtre d'aujourd'hui ne s'applique PAS
    # → On accepte toujours les courses qui sont dans le futur (planning à l'avance)

    # Si la course commence après la fin de la fenêtre (après minuit), c'est pour demain → accepter
    if est_start_min > end_w:
        return True

    # Si la course finit après la fenêtre mais commence dedans, c'est OK (elle chevauche minuit)
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
) -> Tuple[float, Dict[str, float], Tuple[int, int]]:
    """
    Renvoie (score_total, breakdown, (est_start_min, est_finish_min))
    - score en [0..1+] (plus est grand, mieux c'est)
    - breakdown : contributions par facteur
    - estimation temps (start/finish) pour quick-feasibility
    """
    # 1) Proximité / coûts temps (paramétrable via settings)
    avg_kmh = getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0)
    # mapping des noms vers TimeSettings actuels
    buffer_min = int(getattr(settings.time, "pickup_buffer_min", 5))
    pickup_service = int(getattr(settings.time, "pickup_service_min", 3))
    drop_service = int(getattr(settings.time, "dropoff_service_min", 3))

    # (lat, lon) chauffeur (courant/fallback)
    dp = _driver_current_coord(d)
    # (pickup), (dropoff)
    p_coord, d_coord = _booking_coords(b)

    # Estimations robustes (plancher/plafond pour éviter les valeurs extrêmes en heuristique)
    to_pickup_min = haversine_minutes(
        dp, p_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=180
    )
    to_drop_min = haversine_minutes(
        p_coord, d_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240
    )

    # Estimations de début/fin (minutes depuis maintenant)
    # ⚠️ IMPORTANT: on doit prendre en compte l'heure réelle de la course (scheduled_time)
    mins_to_booking = minutes_from_now(getattr(b, "scheduled_time", None))
    # Le chauffeur doit arriver au pickup AVANT scheduled_time
    # Pour la faisabilité, on utilise quand le chauffeur ARRIVE au pickup (= scheduled_time)
    est_start_min = max(0, mins_to_booking)
    est_finish_min = est_start_min + pickup_service + to_drop_min + drop_service

    # Pré‑faisabilité : fenêtre de travail chauffeur
    # Si on dépasse déjà la fenêtre, inutile d'aller plus loin.
    if not _check_driver_window_feasible(driver_window, est_start_min, est_finish_min):
        return (-1.0, {"feasible": 0.0}, (est_start_min, est_finish_min))

    # Garde "pickup trop tard" : si le chauffeur ne peut pas arriver à temps
    # (on a déjà mins_to_booking calculé ci-dessus)
    lateness_penalty = 0.6 if to_pickup_min > mins_to_booking + buffer_min else 0.0

    # 2) Équité (driver_load_balance)
    did_safe = int(cast(Any, getattr(d, "id", 0)) or 0)
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

    # Normalisations simples
    # Proximité -> transformer to_pickup_min en score (0..1)
    # 0-5 min ~ 1.0 ; 30min+ ~ 0.0
    if to_pickup_min <= 5:
        prox_score = 1.0
    elif to_pickup_min >= 30:
        prox_score = 0.0
    else:
        prox_score = max(0.0, 1.0 - (to_pickup_min - 5) / 25.0)

    # Agrégation pondérée
    w = settings.heuristic  # déjà normalisé
    base = (
        prox_score * w.proximity
        + (1.0 - fairness_pen) * w.driver_load_balance
        + pr * w.priority
        + reg_bonus * w.regular_driver_bonus
    )
    # Urgence "non-critique" déjà dans pr via return_generic
    # Appliquer malus de retard potentiel
    total = max(0.0, base - lateness_penalty)

    breakdown = {
        "proximity": prox_score * w.proximity,
        "fairness": (1.0 - fairness_pen) * w.driver_load_balance,
        "priority": pr * w.priority,
        "regular": reg_bonus * w.regular_driver_bonus,
        "lateness_penalty": -lateness_penalty,
    }
    return (total, breakdown, (est_start_min, est_finish_min))


# -------------------------------------------------------------------
# Assignation heuristique
# -------------------------------------------------------------------

def assign(problem: Dict[str, Any], settings: Settings = DEFAULT_SETTINGS) -> HeuristicResult:
    """
    Algorithme glouton :
      1) Traite en premier les "retours urgents".
      2) Trie le reste par scheduled_time croissante puis score décroissant.
      3) Respecte un plafond global par chauffeur (settings.solver.max_bookings_per_driver).
      4) Évite les réassignations inutiles (ASSIGNED au même driver).
    """
    if not problem:
        return HeuristicResult(assignments=[], unassigned_booking_ids=[], debug={"reason": "empty_problem"})


    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts: Dict[int, int] = problem.get("fairness_counts", {})

    # 📅 Récupérer les états précédents depuis problem (ou initialiser à zéro)
    previous_busy = problem.get("busy_until", {})
    previous_times = problem.get("driver_scheduled_times", {})
    previous_load = problem.get("proposed_load", {})

    # État local : nombre d'assignations *proposées* dans cette passe (ids castés en int)
    proposed_load: Dict[int, int] = {int(cast(Any, d.id)): previous_load.get(int(cast(Any, d.id)), 0) for d in drivers}
    driver_index: Dict[int, int] = {int(cast(Any, d.id)): i for i, d in enumerate(drivers)}

    max_cap = settings.solver.max_bookings_per_driver

    urgent: List[Booking] = [b for b in bookings if _is_return_urgent(b, settings)]
    urgent_ids = {int(cast(Any, b.id)) for b in urgent}
    regular: List[Booking] = [b for b in bookings if int(cast(Any, b.id)) not in urgent_ids]

    # Trier
    urgent.sort(key=lambda b: sort_key_utc(cast(Any, getattr(b, "scheduled_time", None))))    # plus proches
    regular.sort(key=lambda b: sort_key_utc(cast(Any, getattr(b, "scheduled_time", None))))   # FIFO temporel

    assignments: List[HeuristicAssignment] = []

    # Timeline par chauffeur (en minutes depuis maintenant)
    busy_until: Dict[int, int] = {int(cast(Any, d.id)): previous_busy.get(int(cast(Any, d.id)), 0) for d in drivers}
    # 🆕 Tracker les scheduled_time par chauffeur pour éviter les conflits
    driver_scheduled_times: Dict[int, List[int]] = {int(cast(Any, d.id)): list(previous_times.get(int(cast(Any, d.id)), [])) for d in drivers}

    unassigned: List[int] = []

    # --- 1) Retours urgents (hard priority) ---
    logger.info("="*80)
    logger.info("[DISPATCH HEURISTIC] 🚨 %d retours urgents, %d courses régulières", len(urgent), len(regular))
    logger.info("[DISPATCH HEURISTIC] 👥 %d chauffeurs disponibles", len(drivers))
    if previous_busy or previous_times or previous_load:
        logger.info("[DISPATCH HEURISTIC] 📥 États récupérés: busy_until=%s, proposed_load=%s", busy_until, proposed_load)
    logger.info("="*80)

    for b in urgent:
        best: Tuple[float, HeuristicAssignment] | None = None
        b_id = int(cast(Any, b.id))
        logger.debug(f"[DISPATCH] Assignation urgente #${b_id}...")

        for d in drivers:
            # Cap par chauffeur
            did = int(cast(Any, d.id))
            if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
                continue

            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            sc, breakdown, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)

            # 🚫 Règle 1: Vérifier que le chauffeur n'a pas déjà une course trop proche
            # Deux courses à moins de 30 min d'intervalle = impossible pour le même chauffeur
            min_gap_minutes = 30  # Marge minimum entre deux courses
            has_conflict = False
            for existing_time in driver_scheduled_times[did]:
                if abs(est_s - existing_time) < min_gap_minutes:
                    logger.debug(f"[DISPATCH] ⏰ Chauffeur #{did} a déjà une course à {existing_time}min, course #{b_id} à {est_s}min (écart: {abs(est_s - existing_time)}min < {min_gap_minutes}min) → CONFLIT")
                    has_conflict = True
                    break
            if has_conflict:
                continue

            # 🚫 Règle 2: Vérifier si le chauffeur peut être disponible à temps
            # Le chauffeur doit finir sa course précédente (busy_until) + avoir le temps d'aller au pickup
            # est_s = quand le chauffeur doit ARRIVER au pickup (= scheduled_time)
            # Il faut vérifier que busy_until[did] <= est_s (avec une petite marge pour le trajet)
            if est_s < busy_until[did]:
                logger.debug(f"[DISPATCH] ⏰ Chauffeur #{did} occupé jusqu'à {busy_until[did]}min, course #{b_id} démarre à {est_s}min → CONFLIT")
                continue
            if sc <= 0:
                continue

            # 🎯 Bonus/malus pour équilibrer la charge
            current_load = proposed_load[did] + fairness_counts.get(did, 0)

            # 📈 Pénalité PROGRESSIVE plus douce
            if current_load <= 2:
                load_penalty = current_load * 0.1
            elif current_load == 3:
                load_penalty = 0.3
            elif current_load == 4:
                load_penalty = 0.6
            else:
                load_penalty = 1.0 + (current_load - 5) * 0.5

            sc -= load_penalty

            # 🏆 Bonus FORT pour chauffeur moins chargé
            min_load = min(proposed_load.values()) if proposed_load else 0
            if current_load == min_load:
                sc += 0.8
            elif current_load == min_load + 1:
                sc += 0.4

            # ⚠️ Malus FORT pour chauffeur d'urgence
            if getattr(d, "is_emergency", False):
                sc -= 0.60

            cand = HeuristicAssignment(
                booking_id=int(cast(Any, b.id)),
                driver_id=did,
                score=sc,
                reason="return_urgent",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if (best is None) or (sc > best[0]):
                best = (sc, cand)

        if best:
            chosen = best[1]
            assignments.append(chosen)
            proposed_load[int(chosen.driver_id)] += 1
            did2 = int(chosen.driver_id)

            # ⏱️ CORRECTION: Calculer scheduled_min du booking et utiliser durée OSRM réelle
            scheduled_time_dt = getattr(b, 'scheduled_time', None)
            base_time = problem.get("base_time")
            if base_time and scheduled_time_dt:
                from shared.time_utils import to_utc
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = scheduled_dt_utc - base_dt_utc if scheduled_dt_utc and base_dt_utc else None
                scheduled_min = int(delta.total_seconds() // 60) if delta else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
            else:
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute if scheduled_time_dt else chosen.estimated_start_min

            # Calculer la durée réelle de la course selon OSRM (pickup + trajet OSRM + dropoff)
            duration_osrm = chosen.estimated_finish_min - chosen.estimated_start_min
            realistic_finish = scheduled_min + duration_osrm
            busy_until[did2] = max(busy_until[did2], realistic_finish)

            # 📅 Enregistrer le scheduled_time RÉEL
            driver_scheduled_times[did2].append(scheduled_min)
            logger.info(f"[DISPATCH] ✅ Urgent #{chosen.booking_id} → Chauffeur #{chosen.driver_id} (score: {chosen.score:.2f}, start: {scheduled_min}min, busy_until: {busy_until[did2]}min)")
        else:
            unassigned.append(int(cast(Any, b.id)))
            logger.warning(f"[DISPATCH] ⚠️ Impossible d'assigner urgent #{b_id} (aucun chauffeur disponible)")

    # --- 2) Assignations régulières ---
    # Pré‑scorage rapide pour limiter la combinatoire
    scored_pool: List[Tuple[float, HeuristicAssignment, Booking]] = []

    logger.warning(f"[HEURISTIC] 🔍 Début scoring de {len(regular)} courses régulières avec {len(drivers)} chauffeurs...")

    for b in regular:
        b_id = int(cast(Any, b.id))
        best_for_b: Tuple[float, HeuristicAssignment] | None = None
        rejected_reasons = []

        for d in drivers:
            did = int(cast(Any, d.id))
            if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
                rejected_reasons.append(f"driver#{did}:cap_reached")
                continue

            # Si la course est déjà ASSIGNED à ce driver, gardons une préférence (éviter churn)
            is_assigned = _is_booking_assigned(b)
            cur_driver_id = _current_driver_id(b)
            prefer_assigned = bool(is_assigned and (cur_driver_id == did))


            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            sc, breakdown, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)

            # 🚫 CORRECTION CRITIQUE: Utiliser scheduled_time (heure demandée par le client)
            # au lieu de est_s (optimisé OSRM) pour vérifier la faisabilité !
            scheduled_time_dt = getattr(b, 'scheduled_time', None)
            if not scheduled_time_dt:
                rejected_reasons.append(f"driver#{did}:no_scheduled_time")
                continue

            # Convertir scheduled_time en minutes depuis minuit du jour concerné
            # (même logique que dans data.py pour la cohérence)
            base_time = problem.get("base_time")
            if base_time:
                # Si base_time est fourni, calculer depuis ce moment
                from shared.time_utils import to_utc
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = scheduled_dt_utc - base_dt_utc if scheduled_dt_utc and base_dt_utc else None
                scheduled_min = int(delta.total_seconds() // 60) if delta else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
            else:
                # Sinon, utiliser les heures/minutes du jour
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

            # 🔍 Logs détaillés pour debug
            if b_id in [106, 109, 113, 115] and did == 3:
                logger.error(f"[DEBUG] Course #{b_id} + Giuseppe (#{did}):")
                logger.error(f"  - scheduled_time: {scheduled_time_dt} ({scheduled_min}min)")
                logger.error(f"  - est_start_min (OSRM optimisé): {est_s}min")
                logger.error(f"  - est_finish_min: {est_f}min")
                logger.error(f"  - busy_until[{did}]: {busy_until[did]}min")
                logger.error(f"  - driver_scheduled_times[{did}]: {driver_scheduled_times[did]}")
                logger.error(f"  - score: {sc:.3f}")

            # 🚫 Règle 1: Vérifier que le pickup demandé n'est PAS pendant qu'une autre course est en cours
            # SAUF si les courses peuvent être regroupées (même pickup, même heure)
            min_gap_minutes = 30
            has_conflict = False
            can_pool = False

            for existing_time in driver_scheduled_times[did]:
                if abs(scheduled_min - existing_time) < min_gap_minutes:
                    # Chercher la course existante pour vérifier si on peut la grouper avec celle-ci
                    existing_booking = None
                    for assigned in [a for a in assignments if a.driver_id == did]:
                        assigned_booking = next((bk for bk in bookings if int(cast(Any, bk.id)) == assigned.booking_id), None)
                        if assigned_booking:
                            assigned_time_dt = getattr(assigned_booking, 'scheduled_time', None)
                            if assigned_time_dt:
                                assigned_min = assigned_time_dt.hour * 60 + assigned_time_dt.minute
                                if assigned_min == existing_time:
                                    existing_booking = assigned_booking
                                    break

                    # Vérifier si regroupement possible
                    if existing_booking and _can_be_pooled(b, existing_booking, settings):
                        can_pool = True
                        logger.info(f"[POOLING] 🚗 Course #{b_id} peut être regroupée avec #{existing_booking.id} (chauffeur #{did})")
                        break
                    else:
                        has_conflict = True
                        rejected_reasons.append(f"driver#{did}:time_conflict")
                        if b_id in [106, 109, 113, 115] and did == 3:
                            logger.error(f"  ❌ CONFLIT: scheduled_min={scheduled_min}min vs existing={existing_time}min (écart: {abs(scheduled_min - existing_time)}min)")
                        break

            if has_conflict and not can_pool:
                continue

            # 🚫 Règle 2: Vérifier que le chauffeur sera libre AVANT l'heure de pickup demandée
            # + marge de sécurité pour la transition (15min minimum)
            required_free_time = busy_until[did] + settings.service_times.min_transition_margin_min
            if scheduled_min < required_free_time:
                rejected_reasons.append(f"driver#{did}:busy")
                if b_id in [106, 109, 113, 115] and did == 3:
                    logger.error(f"  ❌ BUSY: scheduled_min={scheduled_min}min < busy_until+margin={required_free_time}min")
                continue
            if sc <= 0:
                rejected_reasons.append(f"driver#{did}:score_negative")
                continue

            # 🎯 Bonus/malus pour équilibrer la charge entre chauffeurs
            current_load = proposed_load[did] + fairness_counts.get(did, 0)

            # 📈 Pénalité PROGRESSIVE plus douce pour assigner TOUTES les courses en favorisant l'équilibre
            # 0-2 courses : pénalité faible (0.0-0.2)
            # 3 courses : 0.3 pénalité (acceptable)
            # 4 courses : 0.6 pénalité (forte mais pas bloquante)
            # 5+ courses : 1.0+ pénalité (très forte mais permet quand même l'assignation si nécessaire)
            if current_load <= 2:
                load_penalty = current_load * 0.1
            elif current_load == 3:
                load_penalty = 0.3
            elif current_load == 4:
                load_penalty = 0.6
            else:
                load_penalty = 1.0 + (current_load - 5) * 0.5

            sc -= load_penalty

            # 🏆 Bonus FORT pour chauffeur moins chargé (favoriser l'équilibrage)
            min_load = min(proposed_load.values()) if proposed_load else 0
            if current_load == min_load:
                sc += 0.8  # Fort bonus pour le chauffeur le moins chargé
            elif current_load == min_load + 1:
                sc += 0.4  # Bonus moyen si proche du minimum

            # ⚠️ Malus FORT pour chauffeur d'urgence (dernier recours uniquement)
            if getattr(d, "is_emergency", False):
                sc -= 0.60  # Malus augmenté de 0.05 → 0.60

            if prefer_assigned:
                sc += 0.2  # stabilité de planning

            cand = HeuristicAssignment(
                booking_id=int(cast(Any, b.id)),
                driver_id=did,
                score=sc,
                reason="regular_scoring",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if (best_for_b is None) or (sc > best_for_b[0]):
                best_for_b = (sc, cand)

        if best_for_b:
            scored_pool.append((best_for_b[0], best_for_b[1], b))
            logger.debug(f"[HEURISTIC] ✅ Course #{b_id} peut être assignée au driver #{best_for_b[1].driver_id} (score: {best_for_b[0]:.2f})")
        else:
            unassigned.append(int(cast(Any, b.id)))
            logger.warning(f"[HEURISTIC] ❌ Course #{b_id} REJETÉE par tous les chauffeurs: {', '.join(rejected_reasons) if rejected_reasons else 'aucune raison'}")

    # 🕐 CORRECTION: Ordonner par scheduled_time CHRONOLOGIQUE d'abord, puis par score
    # Cela évite d'assigner les courses tardives (bon score) avant les courses matinales (moins bon score)
    # et d'avoir des conflits "busy_until" absurdes
    scored_pool.sort(key=lambda x: (sort_key_utc(cast(Any, getattr(x[2], "scheduled_time", None))), -x[0]))

    pooled_bookings = set()  # Track bookings that were pooled to skip other candidates

    for sc, cand, b in scored_pool:
        # Si cette course a déjà été assignée via regroupement, skip les autres candidats
        if int(cast(Any, b.id)) in pooled_bookings:
            continue

        # Double check cap
        did = int(cand.driver_id)
        if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
            logger.debug(f"[DISPATCH] ⏭️ Chauffeur #{did} a atteint le cap ({max_cap}), skipped")
            continue

        # 🚫 Récupérer le scheduled_time réel du booking pour les vérifications finales
        scheduled_time_dt = getattr(b, 'scheduled_time', None)
        base_time = problem.get("base_time")
        if base_time:
            from shared.time_utils import to_utc
            scheduled_dt_utc = to_utc(scheduled_time_dt)
            base_dt_utc = to_utc(base_time)
            delta = scheduled_dt_utc - base_dt_utc if scheduled_dt_utc and base_dt_utc else None
            scheduled_min = int(delta.total_seconds() // 60) if delta else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
        else:
            scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

        # 🚫 VÉRIFICATION FINALE: Conflit temporel avec courses déjà assignées
        # SAUF si regroupement possible (même pickup, même heure)
        min_gap_minutes = 30
        has_conflict = False
        can_pool = False
        pooled_with = None

        for existing_time in driver_scheduled_times[did]:
            if abs(scheduled_min - existing_time) < min_gap_minutes:
                # Chercher la course existante déjà assignée à ce chauffeur
                existing_booking = None
                for assigned in [a for a in assignments if a.driver_id == did]:
                    assigned_booking = next((bk for bk in bookings if int(cast(Any, bk.id)) == assigned.booking_id), None)
                    if assigned_booking:
                        assigned_time_dt = getattr(assigned_booking, 'scheduled_time', None)
                        if assigned_time_dt:
                            base_time = problem.get("base_time")
                            if base_time:
                                from shared.time_utils import to_utc
                                assigned_dt_utc = to_utc(assigned_time_dt)
                                base_dt_utc = to_utc(base_time)
                                delta = assigned_dt_utc - base_dt_utc if assigned_dt_utc and base_dt_utc else None
                                assigned_min = int(delta.total_seconds() // 60) if delta else (assigned_time_dt.hour * 60 + assigned_time_dt.minute)
                            else:
                                assigned_min = assigned_time_dt.hour * 60 + assigned_time_dt.minute

                            if assigned_min == existing_time:
                                existing_booking = assigned_booking
                                break

                # Vérifier si regroupement possible
                if existing_booking and _can_be_pooled(b, existing_booking, settings):
                    can_pool = True
                    pooled_with = existing_booking.id
                    logger.warning(f"[POOLING] 🚗 Course #{cand.booking_id} FORCÉE au chauffeur #{did} (regroupement avec #{existing_booking.id}, priorité absolue)")
                    pooled_bookings.add(int(cast(Any, b.id)))  # Marquer pour skip les autres candidats
                    break
                else:
                    conflict_msg = f"⚠️ CONFLIT: Chauffeur #{did} a course à {existing_time}min, course #{cand.booking_id} à {scheduled_min}min (écart: {abs(scheduled_min - existing_time)}min)"
                    logger.warning("[DISPATCH] %s → SKIP", conflict_msg)
                    has_conflict = True
                    break

        if has_conflict and not can_pool:
            continue

        # Vérifier aussi busy_until + marge de transition (utiliser scheduled_min)
        # SAUF si c'est un regroupement (le chauffeur prend les 2 clients au même moment)
        if not can_pool:
            required_free_time = busy_until[did] + settings.service_times.min_transition_margin_min
            if scheduled_min < required_free_time:
                logger.warning(f"[DISPATCH] ⚠️ CONFLIT BUSY: Chauffeur #{did} occupé jusqu'à {busy_until[did]}min (+{settings.service_times.min_transition_margin_min}min marge = {required_free_time}min), course #{cand.booking_id} démarre à {scheduled_min}min → SKIP")
                continue

        # Si déjà pris (par un meilleur match urgent par ex.)
        if any(a.booking_id == int(cast(Any, b.id)) for a in assignments):
            continue

        assignments.append(cand)
        proposed_load[did] += 1

        # 🚗 Vérifier si c'est un regroupement avec une course existante
        is_pooled = False
        pooled_with = None
        for existing_time in driver_scheduled_times[did]:
            if abs(scheduled_min - existing_time) < settings.pooling.time_tolerance_min:
                # Trouver la course existante
                for assigned in [a for a in assignments if a.driver_id == did and a != cand]:
                    assigned_booking = next((bk for bk in bookings if int(cast(Any, bk.id)) == assigned.booking_id), None)
                    if assigned_booking and _can_be_pooled(b, assigned_booking, settings):
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
            realistic_finish = scheduled_min + duration_osrm + settings.pooling.max_detour_min
            logger.info(f"[POOLING] 🚗 Course #{cand.booking_id} regroupée avec #{pooled_with} → busy_until += {settings.pooling.max_detour_min}min détour")
        else:
            realistic_finish = scheduled_min + duration_osrm

        busy_until[did] = max(busy_until[did], realistic_finish)

        # 📅 Enregistrer le scheduled_time RÉEL (sauf si déjà enregistré pour regroupement)
        if scheduled_min not in driver_scheduled_times[did]:
            driver_scheduled_times[did].append(scheduled_min)

        pool_indicator = f" [GROUPÉ avec #{pooled_with}]" if is_pooled else ""
        assign_msg = f"✅ Course #{cand.booking_id} → Chauffeur #{did} (score: {sc:.2f}, start: {scheduled_min}min, busy_until: {busy_until[did]}min){pool_indicator}"
        logger.info("[DISPATCH] %s", assign_msg)

    debug = {
        "proposed_load": proposed_load,
        "fairness_counts": fairness_counts,
        "urgent_count": len(urgent),
        "regular_count": len(regular),
        "max_cap": max_cap,
        "busy_until": busy_until,  # 📅 Pour transmettre au fallback
        "driver_scheduled_times": driver_scheduled_times,  # 📅 Pour transmettre au fallback
    }

    logger.info(f"[DISPATCH] 📊 Résultat: {len(assignments)} assignations, {len(unassigned)} non-assignées")
    logger.info(f"[DISPATCH] 📈 Charge par chauffeur: {dict(proposed_load)}")

    return HeuristicResult(assignments=assignments, unassigned_booking_ids=unassigned, debug=debug)

# -------------------------------------------------------------------
# Assignation "retours urgents" (pré-tri) : réguliers d'abord, urgence si nécessaire
# -------------------------------------------------------------------
def assign_urgent(
    problem: Dict[str, Any],
    urgent_booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> HeuristicResult:
    if not problem or not urgent_booking_ids:
        return HeuristicResult(assignments=[], unassigned_booking_ids=[], debug={"reason": "no_urgent"})

    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts: Dict[int, int] = problem.get("fairness_counts", {})
    max_cap = settings.solver.max_bookings_per_driver

    by_id: Dict[int, Booking] = {int(cast(Any, b.id)): b for b in bookings}
    driver_index: Dict[int, int] = {int(cast(Any, d.id)): i for i, d in enumerate(drivers)}
    proposed_load: Dict[int, int] = {int(cast(Any, d.id)): 0 for d in drivers}
    busy_until: Dict[int, int] = {int(cast(Any, d.id)): 0 for d in drivers}

    def _choose_best(b: Booking, regular_only: bool) -> HeuristicAssignment | None:
        best: Tuple[float, HeuristicAssignment] | None = None
        for d in drivers:
            # Évite l'ouverture des chauffeurs d'urgence si regular_only
            if regular_only and getattr(d, "is_emergency", False):
                continue
            # Cap fairness
            did = int(cast(Any, d.id))
            if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
                continue
            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            sc, _br, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)
            if est_s < busy_until[did]:
                continue
            if sc <= 0:
                continue
            # Bonus stabilité si déjà ASSIGNED à ce driver
            if _is_booking_assigned(b) and (_current_driver_id(b) == did):
                sc += 0.3

            # Léger malus sur "emergency" pour ne l'utiliser qu'en dernier recours
            if getattr(d, "is_emergency", False):
                sc -= 0.05
            cand = HeuristicAssignment(
                booking_id=int(cast(Any, b.id)),
                driver_id=did,
                score=sc,
                reason="return_urgent",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if best is None or sc > best[0]:
                best = (sc, cand)
        return best[1] if best else None

    # Ordonner les urgents par horaire (si dispo)
    ordered: List[Booking] = []
    for bid in urgent_booking_ids:
        b = by_id.get(int(cast(Any, bid)))
        if b:
            ordered.append(b)
    ordered.sort(key=lambda x: sort_key_utc(cast(Any, getattr(x, "scheduled_time", None))))

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
            busy_until[did] = max(busy_until[did], chosen.estimated_finish_min)
        else:
            unassigned.append(int(cast(Any, b.id)))

    debug = {
        "urgent_input": urgent_booking_ids,
        "picked": [int(a.booking_id) for a in assignments],
        "unassigned": unassigned,
        "proposed_load": proposed_load,
    }
    return HeuristicResult(assignments=assignments, unassigned_booking_ids=unassigned, debug=debug)


# -------------------------------------------------------------------
# Fallback simple : "closest feasible" pour le reliquat non couvert
# -------------------------------------------------------------------
def closest_feasible(
    problem: Dict[str, Any],
    booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> HeuristicResult:
    if not problem or not booking_ids:
        return HeuristicResult(assignments=[], unassigned_booking_ids=[], debug={"reason": "empty_fallback"})


    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    fairness_counts: Dict[int, int] = problem.get("fairness_counts", {})
    max_cap = settings.solver.max_bookings_per_driver

    by_id: Dict[int, Booking] = {int(cast(Any, b.id)): b for b in bookings}
    driver_index: Dict[int, int] = {int(cast(Any, d.id)): i for i, d in enumerate(drivers)}

    # 📅 RÉCUPÉRER les états de l'heuristique principale si disponibles (pour éviter les conflits)
    previous_busy = problem.get("busy_until", {})
    previous_times = problem.get("driver_scheduled_times", {})
    previous_load = problem.get("proposed_load", {})

    proposed_load: Dict[int, int] = {int(cast(Any, d.id)): previous_load.get(int(cast(Any, d.id)), 0) for d in drivers}
    busy_until: Dict[int, int] = {int(cast(Any, d.id)): previous_busy.get(int(cast(Any, d.id)), 0) for d in drivers}

    # 📅 Traçabilité des temps exacts assignés à chaque chauffeur (pour détecter les doublons d'heure)
    driver_scheduled_times: Dict[int, List[int]] = {int(cast(Any, d.id)): list(previous_times.get(int(cast(Any, d.id)), [])) for d in drivers}

    logger.warning(f"[FALLBACK] 📥 Récupération état précédent: busy_until={dict(busy_until)}, scheduled_times={dict(driver_scheduled_times)}")

    assignments: List[HeuristicAssignment] = []
    unassigned: List[int] = []

    for bid in booking_ids:
        b = by_id.get(int(cast(Any, bid)))
        if not b:
            continue
        best: Tuple[float, HeuristicAssignment] | None = None
        for d in drivers:
            did = int(cast(Any, d.id))
            did = int(cast(Any, d.id))
            if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
                continue
            di = driver_index[did]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            sc, _br, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)

            # 🚫 CORRECTION CRITIQUE: Calculer scheduled_min (heure demandée par client)
            scheduled_time_dt = getattr(b, 'scheduled_time', None)
            if not scheduled_time_dt:
                continue

            base_time = problem.get("base_time")
            if base_time:
                from shared.time_utils import to_utc
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = scheduled_dt_utc - base_dt_utc if scheduled_dt_utc and base_dt_utc else None
                scheduled_min = int(delta.total_seconds() // 60) if delta else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
            else:
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute

            # 🚫 VÉRIFICATION 1: Conflit temporel avec courses déjà assignées
            # SAUF si regroupement possible (même pickup, même heure)
            min_gap_minutes = 30
            has_conflict = False
            can_pool = False

            for existing_time in driver_scheduled_times[did]:
                if abs(scheduled_min - existing_time) < min_gap_minutes:
                    # Chercher la course existante pour vérifier si regroupement possible
                    existing_booking = None
                    for assigned in [a for a in assignments if a.driver_id == did]:
                        assigned_booking = by_id.get(int(assigned.booking_id))
                        if assigned_booking:
                            assigned_time_dt = getattr(assigned_booking, 'scheduled_time', None)
                            if assigned_time_dt:
                                if base_time:
                                    from shared.time_utils import to_utc
                                    assigned_dt_utc = to_utc(assigned_time_dt)
                                    base_dt_utc = to_utc(base_time)
                                    delta = assigned_dt_utc - base_dt_utc if assigned_dt_utc and base_dt_utc else None
                                    assigned_min = int(delta.total_seconds() // 60) if delta else (assigned_time_dt.hour * 60 + assigned_time_dt.minute)
                                else:
                                    assigned_min = assigned_time_dt.hour * 60 + assigned_time_dt.minute

                                if assigned_min == existing_time:
                                    existing_booking = assigned_booking
                                    break

                    # Vérifier si regroupement possible
                    if existing_booking and _can_be_pooled(b, existing_booking, settings):
                        can_pool = True
                        logger.info(f"[POOLING] 🚗 [FALLBACK] Course #{bid} peut être regroupée avec #{existing_booking.id} (chauffeur #{did})")
                        break
                    else:
                        logger.warning(f"[FALLBACK] ⚠️ CONFLIT: Chauffeur #{did} a course à {existing_time}min, course #{bid} à {scheduled_min}min (écart: {abs(scheduled_min - existing_time)}min) → SKIP")
                        has_conflict = True
                        break

            if has_conflict and not can_pool:
                continue

            # 🚫 VÉRIFICATION 2: Chauffeur occupé (busy_until) + marge de transition
            # SAUF si regroupement (le chauffeur prend les 2 clients au même moment)
            if not can_pool:
                required_free_time = busy_until[did] + settings.service_times.min_transition_margin_min
                if scheduled_min < required_free_time:
                    logger.warning(f"[FALLBACK] ⚠️ BUSY: Chauffeur #{did} occupé jusqu'à {busy_until[did]}min (+{settings.service_times.min_transition_margin_min}min marge = {required_free_time}min), course #{bid} démarre à {scheduled_min}min → SKIP")
                    continue

            # 🚗 REGROUPEMENT : Si détecté, assigner IMMÉDIATEMENT sans chercher d'autres chauffeurs
            if can_pool:
                logger.warning(f"[POOLING] 🚗 [FALLBACK] Course #{bid} FORCÉE au chauffeur #{did} (regroupement prioritaire)")
                best = (sc, HeuristicAssignment(
                    booking_id=int(cast(Any, b.id)),
                    driver_id=did,
                    score=sc,
                    reason="fallback_pooled",
                    estimated_start_min=est_s,
                    estimated_finish_min=est_f,
                ))
                break  # ⚠️ CRUCIAL: Sortir de la boucle des chauffeurs

            # 🚫 VÉRIFICATION 3: Score négatif
            if sc <= 0:
                continue

            # 🎯 Bonus/malus pour équilibrer la charge
            current_load = proposed_load[did] + fairness_counts.get(did, 0)

            # Pénalité progressive douce
            if current_load <= 2:
                load_penalty = current_load * 0.1
            elif current_load == 3:
                load_penalty = 0.3
            elif current_load == 4:
                load_penalty = 0.6
            else:
                load_penalty = 1.0 + (current_load - 5) * 0.5

            sc -= load_penalty

            min_load = min(proposed_load.values()) if proposed_load else 0
            if current_load == min_load:
                sc += 0.8

            # Bonus stabilité si déjà ASSIGNED à ce driver
            if _is_booking_assigned(b) and (_current_driver_id(b) == did):
                sc += 0.2

            cand = HeuristicAssignment(
                booking_id=int(cast(Any, b.id)),
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

            # ⏱️ CORRECTION: Calculer scheduled_min et utiliser durée OSRM réelle
            scheduled_time_dt = getattr(b, 'scheduled_time', None)
            base_time = problem.get("base_time")
            if base_time and scheduled_time_dt:
                from shared.time_utils import to_utc
                scheduled_dt_utc = to_utc(scheduled_time_dt)
                base_dt_utc = to_utc(base_time)
                delta = scheduled_dt_utc - base_dt_utc if scheduled_dt_utc and base_dt_utc else None
                scheduled_min = int(delta.total_seconds() // 60) if delta else (scheduled_time_dt.hour * 60 + scheduled_time_dt.minute)
            else:
                scheduled_min = scheduled_time_dt.hour * 60 + scheduled_time_dt.minute if scheduled_time_dt else chosen.estimated_start_min

            # 🚗 Vérifier si c'est un regroupement avec une course existante
            is_pooled = False
            pooled_with = None
            for existing_time in driver_scheduled_times[did2]:
                if abs(scheduled_min - existing_time) < settings.pooling.time_tolerance_min:
                    # Trouver la course existante
                    for assigned in [a for a in assignments if a.driver_id == did2 and a != chosen]:
                        assigned_booking = by_id.get(int(assigned.booking_id))
                        if assigned_booking and _can_be_pooled(b, assigned_booking, settings):
                            is_pooled = True
                            pooled_with = assigned.booking_id
                            break
                    if is_pooled:
                        break

            # Calculer la durée réelle de la course selon OSRM
            duration_osrm = chosen.estimated_finish_min - chosen.estimated_start_min

            if is_pooled:
                # 🚗 REGROUPEMENT: Ajouter détour pour 2ème dropoff
                realistic_finish = scheduled_min + duration_osrm + settings.pooling.max_detour_min
                logger.info(f"[POOLING] 🚗 [FALLBACK] Course #{chosen.booking_id} regroupée avec #{pooled_with} → +{settings.pooling.max_detour_min}min détour")
            else:
                realistic_finish = scheduled_min + duration_osrm

            busy_until[did2] = max(busy_until[did2], realistic_finish)

            # 📅 Enregistrer le scheduled_time RÉEL (sauf si déjà enregistré pour regroupement)
            if scheduled_min not in driver_scheduled_times[did2]:
                driver_scheduled_times[did2].append(scheduled_min)

            pool_indicator = f" [GROUPÉ avec #{pooled_with}]" if is_pooled else ""
            logger.info(f"[FALLBACK] ✅ Course #{chosen.booking_id} → Chauffeur #{did2} (score: {best[0]:.2f}, start: {scheduled_min}min, busy_until: {busy_until[did2]}min){pool_indicator}")
        else:
            unassigned.append(int(cast(Any, b.id)))
            logger.warning(f"[FALLBACK] ❌ Course #{bid} impossible à assigner (aucun chauffeur disponible)")

    debug = {
        "input_unassigned": booking_ids,
        "picked": [int(a.booking_id) for a in assignments],
        "still_unassigned": unassigned,
        "proposed_load": proposed_load,
        "busy_until": busy_until,
        "driver_scheduled_times": driver_scheduled_times,
    }
    return HeuristicResult(assignments=assignments, unassigned_booking_ids=unassigned, debug=debug)


# -------------------------------------------------------------------
# Recommandations post-run : estimation des attentes / ressources
# -------------------------------------------------------------------
def estimate_wait_or_require_extra(
    problem: Dict[str, Any],
    remaining_booking_ids: List[int],
    settings: Settings = DEFAULT_SETTINGS,
) -> Dict[str, Any]:
    """
    Donne des indications simples pour les courses non assignées :
      - ETA approximatif depuis le chauffeur le plus proche (Haversine).
      - Lateness estimée vs. horaire (min).
      - Suggestions: "ouvrir urgences", "ajouter chauffeur", "élargir fenêtres".
    """
    if not problem or not remaining_booking_ids:
        return {"summary": "no_remaining", "items": []}

    bookings: List[Booking] = problem.get("bookings", [])
    drivers: List[Driver] = problem.get("drivers", [])
    by_id: Dict[int, Booking] = {int(cast(Any, b.id)): b for b in bookings}

    # Coords chauffeurs (courantes si dispo, sinon latitude/longitude)
    driver_coords: List[Tuple[float, float]] = []
    for d in drivers:
        cur_lat = getattr(d, "current_lat", None)
        cur_lon = getattr(d, "current_lon", None)
        if cur_lat is not None and cur_lon is not None:
            driver_coords.append((float(cast(Any, cur_lat)), float(cast(Any, cur_lon))))
            continue
        lat = getattr(d, "latitude", None)
        lon = getattr(d, "longitude", None)
        if lat is not None and lon is not None:
            driver_coords.append((float(cast(Any, lat)), float(cast(Any, lon))))
        else:
            driver_coords.append((46.2044, 6.1432))  # Genève

    now = now_local()
    items: List[Dict[str, Any]] = []
    avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
    # mapping vers la clé réellement présente dans TimeSettings
    buf_min = int(getattr(getattr(settings, "time", None), "pickup_buffer_min", 5))

    for bid in remaining_booking_ids:
        b = by_id.get(int(cast(Any, bid)))
        if not b:
            continue
        try:
            pick = (
                float(cast(Any, b.pickup_lat)),
                float(cast(Any, b.pickup_lon)),
            )
        except Exception:
            # si coordonnées manquent, on saute (devrait être enrichi par data.py)
            continue

        # ETA min depuis n'importe quel chauffeur
        etas = [
            haversine_minutes(dc, pick, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240)
            for dc in driver_coords
        ] or [999]
        eta_min = min(etas)

        st = cast(Any, getattr(b, "scheduled_time", None))
        try:
            dt = st if isinstance(st, datetime) else now  # minutes_from_now gère déjà, mais gardons simple
            mins_to_pickup = minutes_from_now(dt)
        except Exception:
            mins_to_pickup = 0
        lateness = int(max(0, (eta_min - mins_to_pickup)))
        items.append(
            {
                "booking_id": int(cast(Any, b.id)),
                "eta_min": int(eta_min),
                "lateness_min": int(lateness - buf_min) if lateness > buf_min else 0,
            }
        )

    # Synthèse basique
    allow_emg = bool(getattr(getattr(settings, "emergency", None), "allow_emergency_drivers", True))
    suggestions: List[str] = []
    if not allow_emg:
        suggestions.append("Autoriser les chauffeurs d'urgence pour absorber le pic.")
    if len(drivers) == 0:
        suggestions.append("Aucun chauffeur disponible : en ajouter au planning.")
    elif any(it.get("lateness_min", 0) > 15 for it in items):
        suggestions.append("Ajouter au moins 1 chauffeur sur le créneau ou élargir les fenêtres de temps.")
    elif any(it.get("lateness_min", 0) > 0 for it in items):
        suggestions.append("Élargir légèrement les fenêtres ou ajuster les priorités.")

    return {"summary": "ok", "items": items, "suggestions": suggestions}
