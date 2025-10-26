# services/dispatch_utils.py
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Dict, List

from sqlalchemy import and_, func, select

from ext import db
from models import Booking, BookingStatus

DEFAULT_STATUSES_FOR_COUNT = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE if hasattr(BookingStatus, "EN_ROUTE") else BookingStatus.ASSIGNED,
    BookingStatus.IN_PROGRESS if hasattr(BookingStatus, "IN_PROGRESS") else BookingStatus.ASSIGNED,
)

def _to_utc(dt: datetime) -> datetime:
    # utilitaire de secours si routes.utils.to_utc n'existe pas / crash
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

def count_assigned_bookings_for_day(
    company_id: int,
    driver_ids: List[int],
    day: datetime | None = None,
    statuses: Iterable[BookingStatus] = DEFAULT_STATUSES_FOR_COUNT,
) -> Dict[int, int]:
    """Compte, pour chaque chauffeur, le nombre de réservations 'actives' du jour.
    - company_id : filtre par entreprise
    - driver_ids : liste d'IDs chauffeur
    - day        : jour de référence (UTC si non tz-aware). Défaut = maintenant.
    - statuses   : statuts considérés (ASSIGNED/EN_ROUTE/IN_PROGRESS par défaut).
    """
    if not driver_ids:
        return {}

    if day is None:
        day = datetime.now(UTC)

    # bornes du jour (en UTC)
    local_day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_start = local_day_start + timedelta(days=1)
    day_start_utc = _to_utc(local_day_start)
    next_day_start_utc = _to_utc(next_day_start)

    stmt = select(Booking.driver_id, func.count())  # type: ignore[arg-type]
    stmt = stmt.where(
        and_(
            Booking.company_id == company_id,
            Booking.driver_id.in_(driver_ids),
            Booking.status.in_(statuses),
            Booking.scheduled_time.isnot(None),
            Booking.scheduled_time >= day_start_utc,
            Booking.scheduled_time < next_day_start_utc,
        )
    )
    stmt = stmt.group_by(Booking.driver_id)

    # SQLAlchemy 2.0 style
    rows = db.session.execute(stmt).all()

    # Toujours renvoyer tous les drivers avec 0 par défaut
    result = {int(did): 0 for did in driver_ids}
    for did, cnt in rows:
        if did is not None:
            result[int(did)] = int(cnt)
    return result
