# holidays_service.py
from datetime import date
from functools import lru_cache
from typing import Set
from holidays import country_holidays

@lru_cache(maxsize=None)
def get_geneva_holidays(year: int) -> Set[date]:
    # Suisse = CH ; subdivision Genève = GE
    hols = country_holidays("CH", subdiv="GE", years=year)
    # country_holidays est itérable sur les dates fériées
    return {d for d in hols}

def is_holiday_in_geneva(check_date: date) -> bool:
    """
    Renvoie True si 'check_date' tombe sur un jour férié officiel de Genève,
    False sinon.
    """
    # On récupère les jours fériés pour l'année de la date check_date
    return check_date in get_geneva_holidays(check_date.year)
