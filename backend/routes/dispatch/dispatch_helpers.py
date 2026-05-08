# backend/routes/dispatch/dispatch_helpers.py
"""Fonctions utilitaires partagées pour les routes de dispatch.

⚠️ NOTE : Certaines fonctions peuvent apparaître comme non utilisées lors du linting,
mais elles seront utilisées lors de l'extraction progressive des autres endpoints
(assignments, delays, optimizer, etc.).
"""

import re
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, cast

from models import Booking, Company
from models.enums import BookingStatus
from repositories.assignment_repository import AssignmentRepository

# Import du namespace depuis __init__.py
from routes.dispatch import dispatch_ns

# Initialisation des repositories
assignment_repo = AssignmentRepository()


def _make_json_safe(value: Any) -> Any:
    """Convertit récursivement les objets vers des types compatibles JSON."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return value


def _coerce_bool_param(v: str | None, default: bool = False) -> bool:
    """Interprète un paramètre bool venant de la query-string."""
    if v is None:
        return default
    v = v.strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _get_current_company() -> Company:
    """Récupère l'entreprise courante via use-case (DDD).

    ✅ DDD: Utilise use-case au lieu de service directement.
    """
    from routes.companies import _get_current_company_via_use_case

    company, err, code = _get_current_company_via_use_case()
    if err or company is None:
        # err est typiquement {"error": "..."}
        msg = (err or {}).get("error") if isinstance(err, dict) else "Accès refusé"
        dispatch_ns.abort(code or 403, msg)
        msg = "Company should not be None after abort"
        raise AssertionError(msg)
    return company


def _current_company_id() -> int:
    """Récupère l'ID de l'entreprise courante."""
    c = _get_current_company()
    cid = getattr(c, "id", None)
    return cid if isinstance(cid, int) else int(cast("Any", cid))


def _validate_date_format(date_str: str | None) -> str:
    """Valide le format YYYY-MM-DD d'une date.

    Args:
        date_str: Chaîne de date à valider

    Returns:
        La chaîne de date si valide

    Raises:
        Abort 400 si format invalide
    """
    if not date_str:
        dispatch_ns.abort(400, "Paramètre date requis (format: YYYY-MM-DD)")
        msg = "Should not continue after abort"
        raise AssertionError(msg)

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        dispatch_ns.abort(
            400,
            f"Format de date invalide: {date_str} (attendu: YYYY-MM-DD)",
        )
        msg = "Should not continue after abort"
        raise AssertionError(msg)

    # Vérifier que la date est valide (ex: pas 2025-13-45)
    try:
        date.fromisoformat(date_str)
    except ValueError as err:
        dispatch_ns.abort(
            400,
            f"Date invalide: {date_str} (format correct mais date inexistante)",
        )
        msg = "Should not continue after abort"
        raise AssertionError(msg) from err

    return date_str


def _parse_date(date_str: str | None) -> date:
    """Parse une date YYYY-MM-DD. Si None ou vide, retourne aujourd'hui."""
    if not date_str:
        return datetime.now(UTC).date()
    # Valider le format avant de parser
    _validate_date_format(date_str)
    try:
        # Parse sans timezone (intentionnel car on veut juste la date)
        return date.fromisoformat(date_str)
    except ValueError as err:
        dispatch_ns.abort(
            400, f"Format de date invalide: {date_str} (attendu: YYYY-MM-DD)"
        )
        msg = "Date parsing should not continue after abort"
        raise AssertionError(msg) from err


def _booking_time_expr() -> Any:
    """Retourne l'expression SQLAlchemy pour scheduled_time de Booking."""
    B = cast("Any", Booking)
    return B.scheduled_time


# Statuts où l’on n’agrège plus de retard (course close, annulée, client à bord, ou pas encore dispatchable).
_DELAYS_EXCLUDED_BOOKING_STATUSES_ROLLUP: frozenset[BookingStatus] = frozenset(
    {
        BookingStatus.COMPLETED,
        BookingStatus.RETURN_COMPLETED,
        BookingStatus.CANCELED,
        BookingStatus.IN_PROGRESS,
        BookingStatus.AWAITING_CLIENT_PAYMENT,
    }
)


def _booking_status_enum_from_model(booking: Any) -> BookingStatus | None:
    st = getattr(booking, "status", None)
    if isinstance(st, BookingStatus):
        return st
    if isinstance(st, str) and st.strip():
        raw = st.strip()
        try:
            return BookingStatus(raw)
        except ValueError:
            upper = raw.upper()
            for member in BookingStatus:
                if member.value == upper:
                    return member
    return None


def _exclude_booking_from_delay_rollups(booking: Any) -> bool:
    """True si la réservation ne doit pas compter dans GET /delays (header, filtres)."""
    bst = _booking_status_enum_from_model(booking)
    return bst is not None and bst in _DELAYS_EXCLUDED_BOOKING_STATUSES_ROLLUP


def _classify_delay_severity(delay_minutes: int) -> str:
    """Classifie la sévérité d'un retard en 3 niveaux.

    Args:
        delay_minutes: Retard en minutes (positif = retard, négatif = avance)

    Returns:
        - "on_time" : Pas de retard (<= 0 min)
        - "reasonable" : Retard raisonnable (1-5 min)
        - "moderate" : Retard modéré (5-10 min)
        - "critical" : Retard critique (>10 min)
        - "early" : En avance (négatif)
    """
    # Constantes de classification
    DELAY_MINUTES_THRESHOLD = 5
    DELAY_MINUTES_REASONABLE_MAX = 5
    DELAY_MINUTES_MODERATE_MAX = 10

    if delay_minutes <= 0:
        # En avance ou à l'heure
        if delay_minutes < -DELAY_MINUTES_THRESHOLD:
            return "early"
        return "on_time"
    if delay_minutes <= DELAY_MINUTES_REASONABLE_MAX:
        return "reasonable"  # 1-5 min : raisonnable
    if delay_minutes <= DELAY_MINUTES_MODERATE_MAX:
        return "moderate"  # 5-10 min : modéré
    return "critical"  # >10 min : critique


def _get_driver_previous_booking(
    driver_id: int,
    current_booking: Booking,
    company_id: int,
    current_date_start: datetime,
    current_date_end: datetime,
) -> tuple[Booking, Any] | tuple[None, None]:
    """Trouve la course précédente du chauffeur avant la course courante.

    Args:
        driver_id: ID du chauffeur
        current_booking: Course courante pour laquelle on cherche la précédente
        company_id: ID de l'entreprise
        current_date_start: Début de la journée (pour filtrer)
        current_date_end: Fin de la journée (pour filtrer)

    Returns:
        Tuple (Booking, Assignment) de la course précédente, ou (None, None) si aucune
    """
    try:
        current_scheduled_time = getattr(current_booking, "scheduled_time", None)
        if not current_scheduled_time:
            return None, None

        # Récupérer toutes les assignations du chauffeur pour la même date
        # ✅ P1: Eager loading pour éviter N+1 queries
        prev_assignment = (
            assignment_repo.find_previous_assignment_for_driver_before_booking(
                driver_id=driver_id,
                company_id=company_id,
                current_date_start=current_date_start,
                current_date_end=current_date_end,
                current_scheduled_time=current_scheduled_time,
                excluded_statuses=[
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                    BookingStatus.CANCELED,
                ],
            )
        )

        if prev_assignment:
            prev_booking = prev_assignment.booking
            return prev_booking, prev_assignment

        return None, None
    except Exception as e:
        # Import logger depuis le module appelant si nécessaire
        import logging

        logger_instance = logging.getLogger(__name__)
        logger_instance.warning(
            "[LiveDelays] Error finding previous booking for driver %d: %s",
            driver_id,
            e,
        )
        return None, None


def _calculate_eta_for_assignment(
    driver_pos: tuple[float, float] | None,
    pickup_pos: tuple[float, float] | None,
    use_haversine_only: bool = False,
) -> int | None:
    """Calcule l'ETA (en secondes) entre driver_pos et pickup_pos.

    Retourne None si les positions ne sont pas disponibles ou en cas d'erreur.
    Cette fonction est thread-safe et peut être appelée en parallèle.

    Args:
        driver_pos: Position actuelle du chauffeur
        pickup_pos: Position de pickup
        use_haversine_only: Si True, utilise uniquement Haversine (bypass OSRM)
    """
    if not driver_pos or not pickup_pos:
        return None

    # ✅ Si OSRM est indisponible (circuit breaker OPEN), utiliser directement Haversine
    if use_haversine_only:
        try:
            from infrastructure.dispatch.settings_adapter import Settings
            from shared.geo_utils import haversine_distance

            # Créer une instance par défaut
            default_settings = Settings()
            distance_km = haversine_distance(
                driver_pos[0], driver_pos[1], pickup_pos[0], pickup_pos[1]
            )
            avg_speed_kmh = float(
                getattr(getattr(default_settings, "matrix", None), "avg_speed_kmh", 25)
            )
            eta_seconds = int((distance_km / max(avg_speed_kmh, 1e-3)) * 3600.0)
            return max(1, eta_seconds)
        except Exception as e:
            import logging

            logger_instance = logging.getLogger(__name__)
            logger_instance.warning("[LiveDelays] Haversine fallback failed: %s", e)
            return None

    try:
        from infrastructure.dispatch import data_adapter as data

        return data.calculate_eta(driver_pos, pickup_pos)
    except Exception as e:
        import logging

        logger_instance = logging.getLogger(__name__)
        logger_instance.warning("[LiveDelays] Failed to calculate ETA: %s", e)
        return None


# Exports publics - ces fonctions sont utilisées par les modules d'endpoints
__all__ = [
    "_booking_time_expr",
    "_calculate_eta_for_assignment",
    "_classify_delay_severity",
    "_coerce_bool_param",
    "_current_company_id",
    "_get_current_company",
    "_get_driver_previous_booking",
    "_make_json_safe",
    "_parse_date",
    "_validate_date_format",
]
