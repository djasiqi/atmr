# backend/services/unified_dispatch/validation.py
"""Validation des assignations pour empêcher les conflits temporels.
Détecte les courses qui se chevauchent pour un même chauffeur.
Gère également les courses groupables (même heure, même départ, même destination).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Tuple

from shared.geo_utils import haversine_distance

logger = logging.getLogger(__name__)

# ✅ Variables d'environnement pour tolérances courses groupées
GROUP_RIDE_TIME_TOLERANCE_MIN = int(
    os.getenv("GROUP_RIDE_TIME_TOLERANCE_MIN", "5")
)  # ±5 minutes par défaut
GROUP_RIDE_DISTANCE_TOLERANCE_M = float(
    os.getenv("GROUP_RIDE_DISTANCE_TOLERANCE_M", "100.0")
)  # 100 mètres par défaut


def validate_no_temporal_conflicts(
    assignments: List[Dict[str, Any]], tolerance_minutes: int = 30
) -> Tuple[bool, List[str]]:
    """Vérifie qu'aucun chauffeur n'a deux courses qui se chevauchent temporellement.

    Args:
        assignments: Liste des assignations à valider
        tolerance_minutes: Temps minimum entre deux courses (incluant service + trajet)

    Returns:
        (is_valid, errors)
            - is_valid: True si aucun conflit
            - errors: Liste des messages d'erreur détaillés

    """
    errors = []

    # Grouper assignments par driver_id
    by_driver: Dict[int, List[Dict[str, Any]]] = {}

    for assignment in assignments:
        driver_id = assignment.get("driver_id")
        if not driver_id:
            continue

        if driver_id not in by_driver:
            by_driver[driver_id] = []
        by_driver[driver_id].append(assignment)

    # Vérifier chaque chauffeur
    for driver_id, driver_assignments in by_driver.items():
        # Trier par scheduled_time
        # Utiliser datetime(1900, 1, 1) comme fallback au lieu de datetime.min
        sorted_assignments = sorted(
            driver_assignments,
            key=lambda a: a.get("scheduled_time") or datetime(1900, 1, 1, tzinfo=UTC),
        )

        # Vérifier overlaps
        for i in range(len(sorted_assignments) - 1):
            current = sorted_assignments[i]
            next_assign = sorted_assignments[i + 1]

            current_time = current.get("scheduled_time")
            next_time = next_assign.get("scheduled_time")

            if not current_time or not next_time:
                continue

            # Convertir en datetime si c'est des strings
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(
                    current_time.replace("Z", "+00:00")
                )
            if isinstance(next_time, str):
                next_time = datetime.fromisoformat(next_time.replace("Z", "+00:00"))

            # Calculer fin estimée de la course actuelle
            # Durée estimée = temps de service + trajet moyen
            estimated_duration_minutes = estimate_trip_duration(current)
            current_end = current_time + timedelta(
                minutes=estimated_duration_minutes + tolerance_minutes
            )

            # Conflit si next_time < current_end
            if next_time < current_end:
                time_gap = (next_time - current_end).total_seconds() / 60

                errors.append(
                    f"⚠️ Chauffeur #{driver_id}: Conflit temporel "
                    + (
                        f"entre courses #{current.get('booking_id')} "
                        f"(fin estimée {current_end:%H:%M}) "
                    )
                    + f"et #{next_assign.get('booking_id')} (début {next_time:%H:%M}) "
                    + f"→ Écart: {abs(time_gap):.1f}min"
                )

    return (len(errors) == 0, errors)


def validate_no_duplicate_times(
    assignments: List[Dict[str, Any]], max_same_time: int = 1
) -> Tuple[bool, List[str]]:
    """Vérifie qu'aucun chauffeur n'a plusieurs courses exactement au même moment.

    Args:
        assignments: Liste des assignations à valider
        max_same_time: Nombre maximum de courses autorisées au même moment

    Returns:
        (is_valid, errors)

    """
    errors = []

    # Grouper par (driver_id, scheduled_time)
    by_driver_time: Dict[Tuple[int, datetime], List[Dict[str, Any]]] = {}

    for assignment in assignments:
        driver_id = assignment.get("driver_id")
        scheduled_time = assignment.get("scheduled_time")

        if not driver_id or not scheduled_time:
            continue

        # Convertir en datetime
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(
                scheduled_time.replace("Z", "+00:00")
            )

        # Arrondir à la minute pour regrouper (ignorer secondes)
        scheduled_time = scheduled_time.replace(second=0, microsecond=0)

        key = (driver_id, scheduled_time)
        if key not in by_driver_time:
            by_driver_time[key] = []
        by_driver_time[key].append(assignment)

    # Détecter duplicatas
    for (driver_id, scheduled_time), driver_assignments in by_driver_time.items():
        if len(driver_assignments) > max_same_time:
            booking_ids = [a.get("booking_id") for a in driver_assignments]
            errors.append(
                f"🔴 Chauffeur #{driver_id}: {len(driver_assignments)} courses AU MÊME MOMENT ({scheduled_time:%H:%M}) → Courses: {booking_ids} (IMPOSSIBLE : un chauffeur ne peut pas être à plusieurs endroits simultanément)"
            )

    return (len(errors) == 0, errors)


def estimate_trip_duration(assignment: Dict[str, Any]) -> int:
    """Estime la durée totale d'une course (pickup + trajet + dropoff).

    Args:
        assignment: Dictionnaire de l'assignation

    Returns:
        Durée estimée en minutes

    """
    # Valeurs par défaut
    pickup_service = 5  # 5 min pour embarquer
    dropoff_service = 10  # 10 min pour déposer

    # Estimer trajet selon distance si disponible
    # Sinon, utiliser moyenne de 20 min
    trip_duration_raw = assignment.get("estimated_duration_minutes", 20)
    trip_duration: int = int(trip_duration_raw) if trip_duration_raw is not None else 20

    return pickup_service + trip_duration + dropoff_service


def validate_driver_capacity(
    assignments: List[Dict[str, Any]], max_bookings_per_driver: int = 10
) -> Tuple[bool, List[str]]:
    """Vérifie qu'aucun chauffeur ne dépasse la capacité maximale de courses.

    Args:
        assignments: Liste des assignations
        max_bookings_per_driver: Nombre maximum de courses par chauffeur

    Returns:
        (is_valid, errors)

    """
    errors = []

    # Compter par chauffeur
    by_driver: Dict[int, int] = {}

    for assignment in assignments:
        driver_id = assignment.get("driver_id")
        if not driver_id:
            continue

        by_driver[driver_id] = by_driver.get(driver_id, 0) + 1

    # Vérifier limites
    for driver_id, count in by_driver.items():
        if count > max_bookings_per_driver:
            errors.append(
                f"⚠️ Chauffeur #{driver_id}: {count} courses assignées "
                + f"(maximum autorisé: {max_bookings_per_driver}) "
                + "→ Risque de fatigue et retards"
            )

    return (len(errors) == 0, errors)


def validate_assignments(
    assignments: List[Dict[str, Any]], strict: bool = False
) -> Dict[str, Any]:
    """Validation complète des assignations.

    Args:
        assignments: Liste des assignations à valider
        strict: Si True, rejette le dispatch si erreurs critiques

    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "stats": Dict
        }

    """
    errors = []
    warnings = []

    # 1. Vérifier duplicatas exacts (CRITIQUE)
    is_valid_dup, dup_errors = validate_no_duplicate_times(assignments)
    if not is_valid_dup:
        errors.extend(dup_errors)

    # 2. Vérifier chevauchements temporels (CRITIQUE)
    is_valid_temp, temp_errors = validate_no_temporal_conflicts(
        assignments, tolerance_minutes=30
    )
    if not is_valid_temp:
        if strict:
            errors.extend(temp_errors)
        else:
            warnings.extend(temp_errors)

    # 3. Vérifier capacité chauffeurs (WARNING)
    is_valid_cap, cap_errors = validate_driver_capacity(
        assignments, max_bookings_per_driver=10
    )
    if not is_valid_cap:
        warnings.extend(cap_errors)

    # Stats
    total_assignments = len(assignments)
    drivers_used = len({a.get("driver_id") for a in assignments if a.get("driver_id")})

    stats = {
        "total_assignments": total_assignments,
        "drivers_used": drivers_used,
        "avg_per_driver": round(total_assignments / drivers_used, 1)
        if drivers_used > 0
        else 0,
        "critical_errors": len([e for e in errors if "🔴" in e]),
        "warnings": len(warnings),
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def is_groupable(new_booking: Any, existing_booking: Any) -> Tuple[bool, str | None]:
    """Vérifie si deux courses peuvent être groupées.

    ✅ NOUVELLE LOGIQUE: Si deux courses ont le même départ OU la même arrivée,
    et que le départ est à moins de 5 minutes d'écart, un chauffeur peut effectuer
    les deux courses en même temps.

    Args:
        new_booking: Booking à assigner
        existing_booking: Booking existant déjà assigné au chauffeur

    Returns:
        (is_groupable, reason_message)
        - is_groupable: True si les courses peuvent être groupées
        - reason_message: Message explicatif (None si groupable)

    """
    # Initialiser le résultat (non groupable par défaut)
    is_groupable_result = False
    reason_message: str | None = None

    # 1. Vérifier que les deux courses ont un scheduled_time
    if not new_booking.scheduled_time or not existing_booking.scheduled_time:
        return (False, "Une des courses n'a pas d'heure planifiée")

    # 2. Vérifier écart temporel ≤ 5 minutes
    time_diff_minutes = abs(
        (new_booking.scheduled_time - existing_booking.scheduled_time).total_seconds()
        / 60
    )
    if time_diff_minutes > GROUP_RIDE_TIME_TOLERANCE_MIN:
        return (
            False,
            f"Écart temporel trop important ({time_diff_minutes:.1f} min > {GROUP_RIDE_TIME_TOLERANCE_MIN} min)",
        )

    # 3. Vérifier même départ OU même arrivée (coordonnées ou adresse)
    pickup_match = False
    dropoff_match = False

    # Vérifier même départ (coordonnées GPS)
    if (
        new_booking.pickup_lat is not None
        and new_booking.pickup_lon is not None
        and existing_booking.pickup_lat is not None
        and existing_booking.pickup_lon is not None
    ):
        pickup_distance_m = (
            haversine_distance(
                float(new_booking.pickup_lat),
                float(new_booking.pickup_lon),
                float(existing_booking.pickup_lat),
                float(existing_booking.pickup_lon),
            )
            * 1000
        )  # Convertir en mètres
        pickup_match = pickup_distance_m <= GROUP_RIDE_DISTANCE_TOLERANCE_M
    else:
        # Fallback: comparer les adresses normalisées (insensible à la casse)
        new_pickup = str(new_booking.pickup_location or "").strip().lower()
        existing_pickup = str(existing_booking.pickup_location or "").strip().lower()
        pickup_match = new_pickup == existing_pickup and new_pickup != ""

    # Vérifier même arrivée (coordonnées GPS)
    if (
        new_booking.dropoff_lat is not None
        and new_booking.dropoff_lon is not None
        and existing_booking.dropoff_lat is not None
        and existing_booking.dropoff_lon is not None
    ):
        dropoff_distance_m = (
            haversine_distance(
                float(new_booking.dropoff_lat),
                float(new_booking.dropoff_lon),
                float(existing_booking.dropoff_lat),
                float(existing_booking.dropoff_lon),
            )
            * 1000
        )  # Convertir en mètres
        dropoff_match = dropoff_distance_m <= GROUP_RIDE_DISTANCE_TOLERANCE_M
    else:
        # Fallback: comparer les adresses normalisées (insensible à la casse)
        new_dropoff = str(new_booking.dropoff_location or "").strip().lower()
        existing_dropoff = str(existing_booking.dropoff_location or "").strip().lower()
        dropoff_match = new_dropoff == existing_dropoff and new_dropoff != ""

    # ✅ NOUVELLE LOGIQUE: Groupable si même départ OU même arrivée
    if pickup_match or dropoff_match:
        is_groupable_result = True
    else:
        reason_message = "Ni le départ ni l'arrivée ne correspondent"

    # 5. Vérifier statut compatible (pas completed/cancelled)
    if is_groupable_result:
        from models.enums import BookingStatus

        incompatible_statuses = {BookingStatus.COMPLETED, BookingStatus.CANCELED}
        if new_booking.status in incompatible_statuses:
            is_groupable_result = False
            reason_message = (
                f"Statut de la nouvelle course incompatible: {new_booking.status}"
            )
        elif existing_booking.status in incompatible_statuses:
            is_groupable_result = False
            reason_message = (
                f"Statut de la course existante incompatible: {existing_booking.status}"
            )

    # Retourner le résultat (un seul return)
    return (is_groupable_result, reason_message)


def check_existing_assignment_conflict(
    driver_id: int,
    scheduled_time: datetime,
    booking_id: int | None = None,
    tolerance_minutes: int = 30,
    new_booking: Any | None = None,
) -> Tuple[bool, str | None, Any | None]:
    """Vérifie si une nouvelle assignation créerait un conflit
    avec les assignations existantes.
    Utilisé lors d'assignation manuelle ou réassignation.

    ✅ NOUVEAU: Support des courses groupables.
    Si deux courses ont le même départ OU la même arrivée, et que le départ
    est à moins de 5 minutes d'écart, un chauffeur peut effectuer les deux
    courses en même temps (pas de conflit).

    Args:
        driver_id: ID du chauffeur
        scheduled_time: Heure de la course
        booking_id: ID du booking (pour exclure lors de modification)
        tolerance_minutes: Marge de sécurité
        new_booking: Booking à assigner (optionnel, requis pour vérification groupable)

    Returns:
        (has_conflict, error_message, conflicting_booking)
        - has_conflict: True si conflit détecté
        - error_message: Message d'erreur détaillé avec temps nécessaire et écart disponible
        - conflicting_booking: Booking en conflit (None si pas de conflit ou groupable)

    """
    # ✅ Utilisation du repository pour découpler de SQLAlchemy
    from repositories.assignment_repository import AssignmentRepository

    assignment_repo = AssignmentRepository()
    existing_assignments = assignment_repo.find_active_by_driver_and_time_range(
        driver_id=driver_id,
        booking_id=booking_id,
    )

    # Vérifier chaque assignation existante
    for assignment in existing_assignments:
        existing_booking = assignment.booking
        if not existing_booking or not existing_booking.scheduled_time:
            continue

        existing_time = existing_booking.scheduled_time

        # ✅ NOUVEAU: Vérifier si les courses sont groupables AVANT de vérifier le conflit temporel
        # Cela permet d'autoriser les courses avec même départ/arrivée et < 5 min d'écart
        # Vérifier pour TOUTES les assignations, pas seulement celles dans la fenêtre temporelle
        if new_booking is not None:
            is_groupable_result, _ = is_groupable(new_booking, existing_booking)
            if is_groupable_result:
                # Courses groupables : pas de conflit, ignorer cette assignation
                logger.info(
                    "[Validation] Courses groupables détectées: booking %d et %d (écart: %.1f min)",
                    new_booking.id if new_booking.id else 0,
                    existing_booking.id,
                    abs((scheduled_time - existing_time).total_seconds() / 60),
                )
                continue  # Passer à la prochaine assignation

        # Calculer fenêtre occupée avec durée estimée plus précise
        # Utiliser les coordonnées GPS si disponibles pour calculer la durée réelle
        estimated_duration = 35  # Par défaut
        if (
            existing_booking.pickup_lat is not None
            and existing_booking.pickup_lon is not None
            and existing_booking.dropoff_lat is not None
            and existing_booking.dropoff_lon is not None
        ):
            trip_distance_km = haversine_distance(
                float(existing_booking.pickup_lat),
                float(existing_booking.pickup_lon),
                float(existing_booking.dropoff_lat),
                float(existing_booking.dropoff_lon),
            )
            # Vitesse moyenne 25 km/h en ville
            trip_time_min = int((trip_distance_km / 25) * 60)
            estimated_duration = (
                trip_time_min + 15
            )  # +15 min pour pickup + dropoff + marge

        time_start = existing_time - timedelta(minutes=tolerance_minutes)
        time_end = existing_time + timedelta(
            minutes=estimated_duration + tolerance_minutes
        )

        # Vérifier si conflit temporel
        if time_start <= scheduled_time <= time_end:
            # Calculer le temps nécessaire et l'écart disponible pour un message plus informatif
            existing_end_time = existing_time + timedelta(minutes=estimated_duration)

            # Temps de transition entre les deux courses
            transition_time_min = 0
            if (
                new_booking is not None
                and existing_booking.dropoff_lat is not None
                and existing_booking.dropoff_lon is not None
                and new_booking.pickup_lat is not None
                and new_booking.pickup_lon is not None
            ):
                transition_distance_km = haversine_distance(
                    float(existing_booking.dropoff_lat),
                    float(existing_booking.dropoff_lon),
                    float(new_booking.pickup_lat),
                    float(new_booking.pickup_lon),
                )
                transition_time_min = int((transition_distance_km / 25) * 60)
            else:
                transition_time_min = 15  # Estimation par défaut

            # Temps total nécessaire
            total_time_needed = estimated_duration + transition_time_min

            # Écart disponible (négatif si conflit)
            time_gap_minutes = (scheduled_time - existing_end_time).total_seconds() / 60

            return (
                True,
                f"Conflit temporel avec course #{existing_booking.id} à {existing_time:%H:%M}. "
                + f"Temps nécessaire: {total_time_needed}min, écart disponible: {time_gap_minutes:.1f}min",
                existing_booking,
            )

    return (False, None, None)
