# holidays_service.py
import holidays
from datetime import date
from functools import lru_cache

@lru_cache(maxsize=None)
def get_geneva_holidays(year: int):
    geneva_hols = holidays.Switzerland(years=[year], subdiv='Geneva')
    return {day for day in geneva_hols}

def is_holiday_in_geneva(check_date: date) -> bool:
    """
    Renvoie True si 'check_date' tombe sur un jour férié officiel de Genève,
    False sinon.
    """
    # On récupère les jours fériés pour l'année de la date check_date
    year_holidays = get_geneva_holidays(check_date.year)
    return check_date in year_holidays
