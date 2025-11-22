# services/vacation_service.py
from datetime import date, timedelta
from typing import Any, cast

from ext import db
from models import DriverVacation

from .holidays_service import (
    is_holiday_in_geneva,  # par ex. un module qui wrap python-holidays
)

DEFAULT_ANNUAL_VACATION_DAYS = 20
# 0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche
WEEKDAY_FRIDAY = 4  # dernier jour ouvrable


def count_working_days_in_period(start_date: date, end_date: date) -> int:
    # Calcule le nombre de jours ouvrables, hors fériés genevois
    if end_date < start_date:
        return 0
    total_days = 0
    current = start_date
    while current <= end_date:
        # Ignore samedi/dimanche et jours fériés
        if current.weekday() <= WEEKDAY_FRIDAY and not is_holiday_in_geneva(current):
            total_days += 1
        current += timedelta(days=1)
    return total_days


def create_vacation(
    driver,  # int id ou objet avec .id
    start_date: date,
    end_date: date,
    vacation_type: str = "VACANCES",
) -> bool:
    # Récupère un driver_id robuste (int ou objet avec .id)
    if isinstance(driver, int):
        driver_id = driver
    else:
        did = getattr(driver, "id", None)
        if did is None:
            msg = "driver doit être un int (id) ou un objet avec attribut .id"
            raise TypeError(msg)
        driver_id = int(did)

    requested_days = count_working_days_in_period(start_date, end_date)

    used_days = sum(
        count_working_days_in_period(vac.start_date, vac.end_date)
        for vac in DriverVacation.query.filter_by(driver_id=driver_id).all()
    )

    if used_days + requested_days > DEFAULT_ANNUAL_VACATION_DAYS:
        return False

    # Calme Pylance sur kwargs SQLAlchemy
    DV = cast("Any", DriverVacation)
    new_vac = DV(
        driver_id=driver_id,
        start_date=None,
        end_date=None,
        vacation_type=str(vacation_type),
    )

    db.session.add(new_vac)
    try:
        db.session.commit()
        return True
    except Exception:
        db.session.rollback()
        raise
