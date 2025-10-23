# backend/services/unified_dispatch/validation.py
# ruff: noqa: DTZ003, W293
"""
Validation des assignations pour empêcher les conflits temporels.
Détecte les courses qui se chevauchent pour un même chauffeur.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Tuple

from models import Assignment, Booking

logger = logging.getLogger(__name__)


def validate_no_temporal_conflicts(
    assignments: List[Dict[str, Any]],
    tolerance_minutes: int = 30
) -> Tuple[bool, List[str]]:
    """
    Vérifie qu'aucun chauffeur n'a deux courses qui se chevauchent temporellement.
    
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
        driver_id = assignment.get('driver_id')
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
            key=lambda a: a.get('scheduled_time') or datetime(1900, 1, 1, tzinfo=UTC)
        )

        # Vérifier overlaps
        for i in range(len(sorted_assignments) - 1):
            current = sorted_assignments[i]
            next_assign = sorted_assignments[i + 1]

            current_time = current.get('scheduled_time')
            next_time = next_assign.get('scheduled_time')

            if not current_time or not next_time:
                continue

            # Convertir en datetime si c'est des strings
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            if isinstance(next_time, str):
                next_time = datetime.fromisoformat(next_time.replace('Z', '+00:00'))

            # Calculer fin estimée de la course actuelle
            # Durée estimée = temps de service + trajet moyen
            estimated_duration_minutes = estimate_trip_duration(current)
            current_end = current_time + timedelta(minutes=estimated_duration_minutes + tolerance_minutes)

            # Conflit si next_time < current_end
            if next_time < current_end:
                time_gap = (next_time - current_end).total_seconds() / 60

                errors.append(
                    f"⚠️ Chauffeur #{driver_id}: Conflit temporel "
                    f"entre courses #{current.get('booking_id')} (fin estimée {current_end:%H:%M}) "
                    f"et #{next_assign.get('booking_id')} (début {next_time:%H:%M}) "
                    f"→ Écart: {abs(time_gap):.0f} min (minimum requis: {tolerance_minutes} min)"
                )

    return (len(errors) == 0, errors)


def validate_no_duplicate_times(
    assignments: List[Dict[str, Any]],
    max_same_time: int = 1
) -> Tuple[bool, List[str]]:
    """
    Vérifie qu'aucun chauffeur n'a plusieurs courses exactement au même moment.
    
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
        driver_id = assignment.get('driver_id')
        scheduled_time = assignment.get('scheduled_time')

        if not driver_id or not scheduled_time:
            continue

        # Convertir en datetime
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))

        # Arrondir à la minute pour regrouper (ignorer secondes)
        scheduled_time = scheduled_time.replace(second=0, microsecond=0)

        key = (driver_id, scheduled_time)
        if key not in by_driver_time:
            by_driver_time[key] = []
        by_driver_time[key].append(assignment)

    # Détecter duplicatas
    for (driver_id, scheduled_time), driver_assignments in by_driver_time.items():
        if len(driver_assignments) > max_same_time:
            booking_ids = [a.get('booking_id') for a in driver_assignments]
            errors.append(
                f"🔴 Chauffeur #{driver_id}: {len(driver_assignments)} courses AU MÊME MOMENT "
                f"({scheduled_time:%H:%M}) → Courses: {booking_ids} "
                f"(IMPOSSIBLE : un chauffeur ne peut pas être à plusieurs endroits simultanément)"
            )

    return (len(errors) == 0, errors)


def estimate_trip_duration(assignment: Dict[str, Any]) -> int:
    """
    Estime la durée totale d'une course (pickup + trajet + dropoff).
    
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
    trip_duration = assignment.get('estimated_duration_minutes', 20)

    return pickup_service + trip_duration + dropoff_service


def validate_driver_capacity(
    assignments: List[Dict[str, Any]],
    max_bookings_per_driver: int = 10
) -> Tuple[bool, List[str]]:
    """
    Vérifie qu'aucun chauffeur ne dépasse la capacité maximale de courses.
    
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
        driver_id = assignment.get('driver_id')
        if not driver_id:
            continue

        by_driver[driver_id] = by_driver.get(driver_id, 0) + 1

    # Vérifier limites
    for driver_id, count in by_driver.items():
        if count > max_bookings_per_driver:
            errors.append(
                f"⚠️ Chauffeur #{driver_id}: {count} courses assignées "
                f"(maximum autorisé: {max_bookings_per_driver}) "
                f"→ Risque de fatigue et retards"
            )

    return (len(errors) == 0, errors)


def validate_assignments(
    assignments: List[Dict[str, Any]],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validation complète des assignations.
    
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
    is_valid_temp, temp_errors = validate_no_temporal_conflicts(assignments, tolerance_minutes=30)
    if not is_valid_temp:
        if strict:
            errors.extend(temp_errors)
        else:
            warnings.extend(temp_errors)

    # 3. Vérifier capacité chauffeurs (WARNING)
    is_valid_cap, cap_errors = validate_driver_capacity(assignments, max_bookings_per_driver=10)
    if not is_valid_cap:
        warnings.extend(cap_errors)

    # Stats
    total_assignments = len(assignments)
    drivers_used = len({a.get('driver_id') for a in assignments if a.get('driver_id')})

    stats = {
        "total_assignments": total_assignments,
        "drivers_used": drivers_used,
        "avg_per_driver": round(total_assignments / drivers_used, 1) if drivers_used > 0 else 0,
        "critical_errors": len([e for e in errors if "🔴" in e]),
        "warnings": len(warnings)
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }


def check_existing_assignment_conflict(
    driver_id: int,
    scheduled_time: datetime,
    booking_id: int | None = None,
    tolerance_minutes: int = 30
) -> Tuple[bool, str | None]:
    """
    Vérifie si une nouvelle assignation créerait un conflit avec les assignations existantes.
    Utilisé lors d'assignation manuelle ou réassignation.
    
    Args:
        driver_id: ID du chauffeur
        scheduled_time: Heure de la course
        booking_id: ID du booking (pour exclure lors de modification)
        tolerance_minutes: Marge de sécurité
    
    Returns:
        (has_conflict, error_message)
    """
    from models import AssignmentStatus

    # Chercher assignations existantes pour ce chauffeur
    query = Assignment.query.join(Booking).filter(
        Assignment.driver_id == driver_id,
        Assignment.status.in_([
            AssignmentStatus.SCHEDULED,
            AssignmentStatus.EN_ROUTE_PICKUP,
            AssignmentStatus.ARRIVED_PICKUP,
            AssignmentStatus.ONBOARD,
            AssignmentStatus.EN_ROUTE_DROPOFF
        ])
    )

    # Exclure le booking actuel si fourni (cas de modification)
    if booking_id:
        query = query.filter(Booking.id != booking_id)

    existing_assignments = query.all()

    # Vérifier chaque assignation existante
    for assignment in existing_assignments:
        booking = assignment.booking
        if not booking or not booking.scheduled_time:
            continue

        existing_time = booking.scheduled_time

        # Calculer fenêtre occupée
        estimated_duration = 35  # Durée moyenne course (pickup 5 + trajet 20 + dropoff 10)
        time_start = existing_time - timedelta(minutes=tolerance_minutes)
        time_end = existing_time + timedelta(minutes=estimated_duration + tolerance_minutes)

        # Vérifier si conflit
        if time_start <= scheduled_time <= time_end:
            time_diff = abs((scheduled_time - existing_time).total_seconds() / 60)
            return (
                True,
                f"Conflit avec course #{booking.id} à {existing_time:%H:%M} "
                f"(écart: {time_diff:.0f} min, minimum: {tolerance_minutes} min)"
            )

    return (False, None)

