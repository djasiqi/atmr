from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from ext import db
from models import DriverShift
from shared.time_utils import to_geneva_local


def serialize_shift(shift: DriverShift) -> dict:
    return {
        "id": shift.id,
        "company_id": shift.company_id,
        "driver_id": shift.driver_id,
        "start_local": shift.start_local.isoformat() if shift.start_local else None,
        "end_local": shift.end_local.isoformat() if shift.end_local else None,
        "timezone": getattr(shift, "timezone", "Europe/Zurich"),
        "type": getattr(shift.type, "value", str(shift.type)).lower(),
        "status": getattr(shift.status, "value", str(shift.status)).lower(),
        "site": getattr(shift, "site", None),
        "zone": getattr(shift, "zone", None),
        "client_ref": getattr(shift, "client_ref", None),
        "pay_code": getattr(shift, "pay_code", None),
        "vehicle_id": getattr(shift, "vehicle_id", None),
        "notes_internal": getattr(shift, "notes_internal", None),
        "notes_employee": getattr(shift, "notes_employee", None),
        "created_by_user_id": shift.created_by_user_id,
        "updated_by_user_id": getattr(shift, "updated_by_user_id", None),
        "created_at": shift.created_at.isoformat() if shift.created_at else None,
        "updated_at": shift.updated_at.isoformat() if shift.updated_at else None,
        "version": getattr(shift, "version", 1),
        # placeholders for future enrichment
        "breaks": [],
        "assignments": [],
        "compliance_flags": getattr(shift, "compliance_flags", []) or [],
    }


def _normalize_to_local_naive(dt: datetime) -> datetime:
    """Ensure a datetime is naive in local company timezone (Europe/Zurich).

    - If aware: convert to Europe/Zurich then drop tzinfo
    - If naive: assume already local and keep as-is
    """
    if dt is None:
        return dt
    if dt.tzinfo is not None:
        local_tz = ZoneInfo("Europe/Zurich")
        return dt.astimezone(local_tz).replace(tzinfo=None)
    return dt


def validate_shift_overlap(company_id: int, driver_id: int, start_local: datetime, end_local: datetime, *, exclude_id: int | None = None) -> None:
    """Squelette: lève une ValueError si un chevauchement est détecté."""
    # Normalize to naive local for consistent comparisons & DB filters
    start_local = _normalize_to_local_naive(start_local)
    end_local = _normalize_to_local_naive(end_local)
    if end_local <= start_local:
        raise ValueError("end_local doit être > start_local")

    q = db.session.query(DriverShift).filter(
        DriverShift.company_id == company_id,
        DriverShift.driver_id == driver_id,
        DriverShift.start_local < end_local,
        DriverShift.end_local > start_local,
    )
    if exclude_id is not None:
        q = q.filter(DriverShift.id != exclude_id)
    if db.session.query(q.exists()).scalar():
        raise ValueError("Chevauchement de shift détecté")


def compute_driver_availability(company_id: int, driver_id: int, from_dt: datetime, to_dt: datetime) -> dict:
    """Retourne un squelette de calendrier busy/free pour l'intervalle demandé."""
    return {"busy": [], "free": []}


def materialize_template(company_id: int, driver_id: int, from_d: date, to_d: date) -> int:
    """Génère des DriverShift depuis les templates (squelette). Retourne le nombre créés."""
    return 0


def is_driver_available_at(company_id: int, driver_id: int, dt: datetime) -> bool:
    """
    Vérifie si un chauffeur est disponible à une date/heure donnée.
    
    Vérifie :
    1. Qu'il a un shift actif à cette heure
    2. Qu'il n'est pas en indisponibilité (congé, maladie, etc.)
    3. Qu'il n'est pas en pause
    
    Args:
        company_id: ID de l'entreprise
        driver_id: ID du chauffeur
        dt: datetime à vérifier
    
    Returns:
        True si le chauffeur est disponible, False sinon
    """
    from sqlalchemy import and_

    from models import DriverShift, DriverUnavailability, ShiftStatus

    try:
        # Convertir en heure locale Genève
        local_dt = to_geneva_local(dt)
        local_date = local_dt.date()

        # 1. Vérifier qu'il a un shift actif ce jour-là
        shift = DriverShift.query.filter(
            and_(
                DriverShift.company_id == company_id,
                DriverShift.driver_id == driver_id,
                DriverShift.start_local <= local_dt,
                DriverShift.end_local >= local_dt,
                DriverShift.status.in_([ShiftStatus.SCHEDULED, ShiftStatus.ACTIVE])
            )
        ).first()

        if not shift:
            # Pas de shift actif à cette heure
            return False

        # 2. Vérifier qu'il n'est pas en indisponibilité
        unavailability = DriverUnavailability.query.filter(
            and_(
                DriverUnavailability.company_id == company_id,
                DriverUnavailability.driver_id == driver_id,
                DriverUnavailability.start_date <= local_date,
                DriverUnavailability.end_date >= local_date
            )
        ).first()

        if unavailability:
            # Le chauffeur est indisponible (congé, maladie, etc.)
            return False

        # 3. TODO futur : Vérifier les pauses (DriverBreak)
        # Pour l'instant, on considère disponible si shift actif et pas indispo

        return True

    except Exception as e:
        # En cas d'erreur, considérer comme non disponible par sécurité
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Erreur vérification disponibilité driver #{driver_id} à {dt}: {e}")
        return False


