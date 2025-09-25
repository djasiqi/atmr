# services/vacation_service.py
from datetime import date, timedelta
from models import db, DriverVacation, Driver
from .holidays_service import is_holiday_in_geneva  # par ex. un module qui wrap python-holidays

DEFAULT_ANNUAL_VACATION_DAYS = 20

def count_working_days_in_period(start_date: date, end_date: date) -> int:
    # Calcule le nombre de jours ouvrables, hors fériés genevois
    if end_date < start_date:
        return 0
    total_days = 0
    current = start_date
    while current <= end_date:
        # Ignore samedi/dimanche
        if current.weekday() < 5:
            # Ignore jour férié
            if not is_holiday_in_geneva(current):
                total_days += 1
        current += timedelta(days=1)
    return total_days

def create_vacation(driver, start_date, end_date, vacation_type="VACANCES"):
    requested_days = count_working_days_in_period(start_date, end_date)
    
    used_days = sum(
        count_working_days_in_period(vac.start_date, vac.end_date)
        for vac in DriverVacation.query.filter_by(driver_id=driver.id).all()
    )
    
    if used_days + requested_days > DEFAULT_ANNUAL_VACATION_DAYS:
        return False

    new_vac = DriverVacation(
        driver_id=driver.id,
        start_date=start_date,
        end_date=end_date,
        vacation_type=vacation_type
    )
    
    db.session.add(new_vac)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e  # Vous pourriez logger l'erreur ici avant de raise
    
    return True
