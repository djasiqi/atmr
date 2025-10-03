# services/dispatch_utils.py
from typing import Dict, Iterable, List
from sqlalchemy import func, select
from datetime import datetime, timedelta, timezone
from models import Booking, BookingStatus
from ext import db

DEFAULT_STATUSES_FOR_COUNT = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE if hasattr(BookingStatus, "EN_ROUTE") else BookingStatus.ASSIGNED,
    BookingStatus.IN_PROGRESS if hasattr(BookingStatus, "IN_PROGRESS") else BookingStatus.ASSIGNED,
)

def _to_utc(dt: datetime) -> datetime:
    # utilitaire de secours si routes.utils.to_utc n’existe pas / crash
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def count_assigned_bookings_for_day(
    company_id: int,
    driver_ids: List[int],
    day: datetime | None = None,
    statuses: Iterable[BookingStatus] = DEFAULT_STATUSES_FOR_COUNT,
) -> Dict[int, int]:
    """
    Compte, pour chaque chauffeur, le nombre de réservations 'actives' du jour.
    - company_id : filtre par entreprise
    - driver_ids : liste d'IDs chauffeur
    - day        : jour de référence (UTC si non tz-aware). Défaut = maintenant.
    - statuses   : statuts considérés (ASSIGNED/EN_ROUTE/IN_PROGRESS par défaut)
    """
    if not driver_ids:
        return {}

    if day is None:
        day = datetime.now(timezone.utc)

    # bornes du jour (en UTC)
    local_day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_start = local_day_start + timedelta(days=1)
    day_start_utc = _to_utc(local_day_start)
    next_day_start_utc = _to_utc(next_day_start)

    # Pylance voit parfois les attributs ORM comme des types natifs -> on passe par getattr
    B = Booking
    driver_id_col = getattr(B, "driver_id")
    company_id_col = getattr(B, "company_id")
    status_col = getattr(B, "status")
    scheduled_col = getattr(B, "scheduled_time")
    id_col = getattr(B, "id")

    stmt = (
        select(driver_id_col, func.count(id_col))
        .where(
            company_id_col == company_id,
            driver_id_col.in_(driver_ids),
            status_col.in_(tuple(statuses)),
            scheduled_col.isnot(None),
            scheduled_col >= day_start_utc,
            scheduled_col < next_day_start_utc,
        )
        .group_by(driver_id_col)
    )

    # SQLAlchemy 2.0 style
    rows = db.session.execute(stmt).all()

    # Toujours renvoyer tous les drivers avec 0 par défaut
    result = {int(did): 0 for did in driver_ids}
    for did, cnt in rows:
        if did is not None:
            result[int(did)] = int(cnt)
    return result
