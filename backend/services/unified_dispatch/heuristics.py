# backend/services/unified_dispatch/heuristics.py
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple, Optional

from models import Booking, Driver, BookingStatus
from services.unified_dispatch.settings import Settings
DEFAULT_SETTINGS = Settings()
from shared.time_utils import minutes_from_now, sort_key_utc, now_local


logger = logging.getLogger(__name__)

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
    max_minutes: Optional[int] = None,
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

    # Haversine (distance en km)
    R = 6371.0088  # rayon moyen de la Terre en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    # Utilise fsum pour une addition légèrement plus stable numériquement
    sin_dphi = math.sin(dphi / 2.0)
    sin_dlam = math.sin(dlambda / 2.0)
    h = math.fsum([
        sin_dphi * sin_dphi,
        math.cos(phi1) * math.cos(phi2) * sin_dlam * sin_dlam
    ])
    # Protéger contre les erreurs d'arrondi
    h = min(1.0, max(0.0, h))
    dist_km = R * (2.0 * math.atan2(math.sqrt(h), math.sqrt(1.0 - h)))

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



def _driver_current_coord(d: Driver) -> Tuple[float, float]:
    # On assume que data.py a mis à jour current_lat/current_lon
    if d.current_lat is not None and d.current_lon is not None:
        return (float(d.current_lat), float(d.current_lon))
    # fallback sur base chauffeur
    if d.latitude is not None and d.longitude is not None:
        return (float(d.latitude), float(d.longitude))
    # fallback Genève
    return (46.2044, 6.1432)


def _booking_coords(b: Booking) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ( (float(b.pickup_lat), float(b.pickup_lon)),
             (float(b.dropoff_lat), float(b.dropoff_lon)) )


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

    # TODO : bonus « VIP client » si vous avez ce champ
    return score


def _is_return_urgent(b: Booking, settings: Settings) -> bool:
    if not getattr(b, "is_return", False):
        return False
    mins = minutes_from_now(getattr(b, "scheduled_time", None))
    return mins <= settings.emergency.return_urgent_threshold_min


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
    if b.driver_id and b.driver_id == d.id:
        return 0.15
    return 0.0


def _check_driver_window_feasible(driver_window: Tuple[int, int], est_start_min: int, est_finish_min: int) -> bool:
    start_w, end_w = driver_window
    return (est_start_min >= start_w) and (est_finish_min <= end_w)


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
    buffer_min = int(settings.time.buffer_min)
    pickup_service = int(settings.time.service_time_pickup_min)
    drop_service = int(settings.time.service_time_dropoff_min)

    dp = _driver_current_coord(d)                 # (lat, lon) chauffeur (courant/fallback)
    p_coord, d_coord = _booking_coords(b)         # (pickup), (dropoff)

    # Estimations robustes (plancher/plafond pour éviter les valeurs extrêmes en heuristique)
    to_pickup_min = haversine_minutes(
        dp, p_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=180
    )
    to_drop_min = haversine_minutes(
        p_coord, d_coord, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240
    )

    # Estimations de début/fin (minutes depuis maintenant)
    est_start_min = max(0, to_pickup_min)  # temps pour arriver au pickup (sera borné par busy_until côté appelant)
    est_finish_min = est_start_min + pickup_service + to_drop_min + drop_service

    # Pré‑faisabilité : fenêtre de travail chauffeur
    # Si on dépasse déjà la fenêtre, inutile d'aller plus loin.
    if not _check_driver_window_feasible(driver_window, est_start_min, est_finish_min):
        return (-1.0, {"feasible": 0.0}, (est_start_min, est_finish_min))

    # Garde "pickup trop tard" : si on sait déjà qu'on part trop tard, malus fort
    mins_to_pickup = minutes_from_now(getattr(b, "scheduled_time", None))
    if est_start_min > mins_to_pickup + buffer_min:
        # arriverait trop tard -> malus fort
        lateness_penalty = 0.6
    else:
        lateness_penalty = 0.0

    # 2) Équité (driver_load_balance)
    fairness_pen = _driver_fairness_penalty(d.id, fairness_counts)

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

    # État local : nombre d'assignations *proposées* dans cette passe
    proposed_load: Dict[int, int] = {d.id: 0 for d in drivers}
    driver_index: Dict[int, int] = {d.id: i for i, d in enumerate(drivers)}

    max_cap = settings.solver.max_bookings_per_driver

    urgent: List[Booking] = [b for b in bookings if _is_return_urgent(b, settings)]
    regular: List[Booking] = [b for b in bookings if b not in urgent]

    # Trier
    urgent.sort(key=lambda b: sort_key_utc(getattr(b, "scheduled_time", None)))       # plus proches dans le temps d'abord
    regular.sort(key=lambda b: sort_key_utc(getattr(b, "scheduled_time", None)))        # FIFO temporel, puis scoring à l'intérieur

    assignments: List[HeuristicAssignment] = []
    # Timeline par chauffeur (en minutes depuis maintenant)
    busy_until: Dict[int, int] = {d.id: 0 for d in drivers}
    unassigned: List[int] = []

    # --- 1) Retours urgents (hard priority) ---
    for b in urgent:
        best: Optional[Tuple[float, HeuristicAssignment]] = None

        for d in drivers:
            # Cap par chauffeur
            if proposed_load[d.id] + fairness_counts.get(d.id, 0) >= max_cap:
                continue

            di = driver_index[d.id]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            sc, breakdown, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)
            # 🚫 Conflit temporel: chauffeur encore occupé à ce moment
            if est_s < busy_until[d.id]:
                continue
            if sc <= 0:
                continue

            # Bonus d'astreinte si vous avez des chauffeurs d'urgence (ex: flag d.is_emergency)
            is_emergency = getattr(d, "is_emergency", False)
            # fallback si tu utilises DriverType:
            try:
                from models import DriverType
                if getattr(d, "driver_type", None) == DriverType.EMERGENCY:
                    is_emergency = True
            except Exception:
                pass
            if settings.emergency.allow_emergency_drivers and is_emergency:
                sc += 0.1  # petit bonus

            cand = HeuristicAssignment(
                booking_id=b.id,
                driver_id=d.id,
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
            proposed_load[chosen.driver_id] += 1
            # ⏱️ le chauffeur est occupé jusqu'à la fin estimée
            busy_until[chosen.driver_id] = max(busy_until[chosen.driver_id], chosen.estimated_finish_min)
        else:
            unassigned.append(b.id)

    # --- 2) Assignations régulières ---
    # Pré‑scorage rapide pour limiter la combinatoire
    scored_pool: List[Tuple[float, HeuristicAssignment, Booking]] = []

    for b in regular:
        best_for_b: Optional[Tuple[float, HeuristicAssignment]] = None
        for d in drivers:
            if proposed_load[d.id] + fairness_counts.get(d.id, 0) >= max_cap:
                continue

            # Si la course est déjà ASSIGNED à ce driver, gardons une préférence (éviter churn)
            prefer_assigned = (b.status == BookingStatus.ASSIGNED and b.driver_id == d.id)

            di = driver_index[d.id]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)

            sc, breakdown, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)
            if est_s < busy_until[d.id]:
                continue
            if sc <= 0:
                continue
            if prefer_assigned:
                sc += 0.2  # stabilité de planning

            cand = HeuristicAssignment(
                booking_id=b.id,
                driver_id=d.id,
                score=sc,
                reason="regular_scoring",
                estimated_start_min=est_s,
                estimated_finish_min=est_f,
            )
            if (best_for_b is None) or (sc > best_for_b[0]):
                best_for_b = (sc, cand)

        if best_for_b:
            scored_pool.append((best_for_b[0], best_for_b[1], b))
        else:
            unassigned.append(b.id)

    # Ordonner par score décroissant (meilleures paires d'abord)
    scored_pool.sort(key=lambda x: (-x[0], sort_key_utc(x[2].scheduled_time)))

    for sc, cand, b in scored_pool:
        # Double check cap
        if proposed_load[cand.driver_id] + fairness_counts.get(cand.driver_id, 0) >= max_cap:
            continue
        # Si déjà pris (par un meilleur match urgent par ex.)
        if any(a.booking_id == b.id for a in assignments):
            continue
        assignments.append(cand)
        proposed_load[cand.driver_id] += 1
        busy_until[cand.driver_id] = max(busy_until[cand.driver_id], cand.estimated_finish_min)

    debug = {
        "proposed_load": proposed_load,
        "fairness_counts": fairness_counts,
        "urgent_count": len(urgent),
        "regular_count": len(regular),
        "max_cap": max_cap,
    }
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

    by_id: Dict[int, Booking] = {int(b.id): b for b in bookings}
    driver_index: Dict[int, int] = {d.id: i for i, d in enumerate(drivers)}
    proposed_load: Dict[int, int] = {d.id: 0 for d in drivers}
    busy_until: Dict[int, int] = {d.id: 0 for d in drivers}

    def _choose_best(b: Booking, regular_only: bool) -> Optional[HeuristicAssignment]:
        best: Optional[Tuple[float, HeuristicAssignment]] = None
        for d in drivers:
            # Évite l'ouverture des chauffeurs d'urgence si regular_only
            if regular_only and getattr(d, "is_emergency", False):
                continue
            # Cap fairness
            if proposed_load[d.id] + fairness_counts.get(d.id, 0) >= max_cap:
                continue
            di = driver_index[d.id]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            sc, _br, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)
            if est_s < busy_until[d.id]:
                continue
            if sc <= 0:
                continue
            # Bonus stabilité si déjà ASSIGNED à ce driver
            if b.status == BookingStatus.ASSIGNED and b.driver_id == d.id:
                sc += 0.3
            # Léger malus sur "emergency" pour ne l'utiliser qu'en dernier recours
            if getattr(d, "is_emergency", False):
                sc -= 0.05
            cand = HeuristicAssignment(
                booking_id=b.id,
                driver_id=d.id,
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
        b = by_id.get(int(bid))
        if b:
            ordered.append(b)
    ordered.sort(key=lambda x: sort_key_utc(getattr(x, "scheduled_time", None)))

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
            proposed_load[chosen.driver_id] += 1
            busy_until[chosen.driver_id] = max(busy_until[chosen.driver_id], chosen.estimated_finish_min)
        else:
            unassigned.append(b.id)

    debug = {
        "urgent_input": urgent_booking_ids,
        "picked": [a.booking_id for a in assignments],
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

    by_id: Dict[int, Booking] = {int(b.id): b for b in bookings}
    driver_index: Dict[int, int] = {d.id: i for i, d in enumerate(drivers)}
    proposed_load: Dict[int, int] = {d.id: 0 for d in drivers}
    busy_until: Dict[int, int] = {d.id: 0 for d in drivers}

    assignments: List[HeuristicAssignment] = []
    unassigned: List[int] = []

    for bid in booking_ids:
        b = by_id.get(int(bid))
        if not b:
            continue
        best: Optional[Tuple[float, HeuristicAssignment]] = None
        for d in drivers:
            if proposed_load[d.id] + fairness_counts.get(d.id, 0) >= max_cap:
                continue
            di = driver_index[d.id]
            dw = driver_windows[di] if di < len(driver_windows) else (0, 24 * 60)
            sc, _br, (est_s, est_f) = _score_driver_for_booking(b, d, dw, settings, fairness_counts)
            if est_s < busy_until[d.id]:
                continue
            if sc <= 0:
                continue
            # Bonus stabilité si déjà ASSIGNED à ce driver
            if b.status == BookingStatus.ASSIGNED and b.driver_id == d.id:
                sc += 0.2
            cand = HeuristicAssignment(
                booking_id=b.id,
                driver_id=d.id,
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
            proposed_load[chosen.driver_id] += 1
            busy_until[chosen.driver_id] = max(busy_until[chosen.driver_id], chosen.estimated_finish_min)
        else:
            unassigned.append(b.id)

    debug = {
        "input_unassigned": booking_ids,
        "picked": [a.booking_id for a in assignments],
        "still_unassigned": unassigned,
        "proposed_load": proposed_load,
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
    by_id: Dict[int, Booking] = {int(b.id): b for b in bookings}

    # Coords chauffeurs (courantes si dispo, sinon latitude/longitude)
    driver_coords: List[Tuple[float, float]] = []
    for d in drivers:
        if getattr(d, "current_lat", None) is not None and getattr(d, "current_lon", None) is not None:
            driver_coords.append((float(d.current_lat), float(d.current_lon)))
        elif getattr(d, "latitude", None) is not None and getattr(d, "longitude", None) is not None:
            driver_coords.append((float(d.latitude), float(d.longitude)))
        else:
            driver_coords.append((46.2044, 6.1432))  # Genève

    now = now_local()
    items: List[Dict[str, Any]] = []
    avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
    buf_min = int(getattr(getattr(settings, "time", None), "buffer_min", 5))

    for bid in remaining_booking_ids:
        b = by_id.get(int(bid))
        if not b:
            continue
        try:
            pick = (float(b.pickup_lat), float(b.pickup_lon))
        except Exception:
            # si coordonnées manquent, on saute (devrait être enrichi par data.py)
            continue

        # ETA min depuis n'importe quel chauffeur
        etas = [
            haversine_minutes(dc, pick, avg_kmh=avg_kmh, min_minutes=1, max_minutes=240)
            for dc in driver_coords
        ] or [999]
        eta_min = min(etas)

        st = getattr(b, "scheduled_time", None)
        try:
            dt = st if isinstance(st, datetime) else now  # minutes_from_now gère déjà, mais gardons simple
            mins_to_pickup = minutes_from_now(dt)
        except Exception:
            mins_to_pickup = 0
        lateness = int(max(0, (eta_min - mins_to_pickup)))
        items.append(
            {
                "booking_id": int(b.id),
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
