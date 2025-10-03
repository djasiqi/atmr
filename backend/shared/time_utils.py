from __future__ import annotations

import os
import pytz
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Union, Tuple


# ---------------------------------------------------------------------------
# MODE NAÏF LOCAL (Europe/Zurich implicite) — AUCUNE CONVERSION UTC
# ---------------------------------------------------------------------------

# Fuseau business par défaut (surchargeable par variable d'env)
LOCAL_TZ = ZoneInfo(os.getenv("UD_DEFAULT_TZ", "Europe/Zurich"))

def _normalize_local_string(s: str) -> str:
    """
    Normalise quelques variations communes (sans TZ) :
    - remplace séparateur espace par 'T' si nécessaire
    - conserve les fractions de secondes si présentes
    """
    s = s.strip()
    if " " in s and "T" not in s:
        s = s.replace(" ", "T")
    return s

def parse_local_naive(dt: Union[str, datetime, None]) -> Optional[datetime]:
    """Retourne un datetime **naïf** (ou None) en interprétant toute entrée comme locale."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        # On retire toute tzinfo éventuelle
        return dt.replace(tzinfo=None)
    s = _normalize_local_string(str(dt))
    return datetime.fromisoformat(s)

def now_local() -> datetime:
    """Horloge unique : maintenant en **naïf local** (Europe/Zurich)."""
    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    # Convert to Europe/Zurich
    now_zurich = now_utc.astimezone(pytz.timezone('Europe/Zurich'))
    # Return as naive datetime
    return now_zurich.replace(tzinfo=None)

def minutes_from_now_local(dt: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre maintenant (local naïf) et dt (naïf)."""
    if not dt:
        return 10**9
    try:
        target = parse_local_naive(dt)
    except Exception:
        return 10**9
    if target is None:
        return 10**9
    delta = target - now_local()
    return max(0, int(delta.total_seconds() // 60))

def minutes_between_local(a: Union[str, datetime, None], b: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre a et b (tous naïfs)."""
    if not a or not b:
        return 0
    try:
        aa = parse_local_naive(a)
        bb = parse_local_naive(b)
    except Exception:
        return 0
    if aa is None or bb is None:
        return 0
    return max(0, int((aa - bb).total_seconds() // 60))

def sort_key_local(dt: Union[str, datetime, None]) -> datetime:
    """Clé de tri sûre (naïf local). None => +50 ans."""
    if not dt:
        return now_local() + timedelta(days=365 * 50)
    try:
        parsed = parse_local_naive(dt)
    except Exception:
        return now_local() + timedelta(days=365 * 50)
    if parsed is None:
        return now_local() + timedelta(days=365 * 50)
    return parsed

def split_date_time_local(dt: Union[str, datetime, None]) -> Tuple[Optional[str], Optional[str]]:
    """
    Retourne ('YYYY-MM-DD', 'HH:MM') sans conversions — pour l’affichage.
    """
    d = parse_local_naive(dt)
    if not d:
        return None, None
    return d.strftime("%Y-%m-%d"), d.strftime("%H:%M")

# ---------------------------------------------------------------------------
# Bornes locales naïves pour un jour 'YYYY-MM-DD'
# ---------------------------------------------------------------------------

def day_local_bounds(day_str: str) -> tuple[datetime, datetime]:
    """Minuit inclus → minuit du jour suivant (naïf)."""
    y, m, d = map(int, day_str.split("-"))
    start = datetime(y, m, d, 0, 0, 0)
    end   = start + timedelta(days=1)
    return start, end

def coerce_local_day(value: Union[str, date]) -> str:
    """Normalise en 'YYYY-MM-DD'."""
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip().replace("/", "-")
    y, m, d = map(int, s.split("-"))
    _ = date(y, m, d)
    return f"{y:04d}-{m:02d}-{d:02d}"

__all__ = [
    "parse_local_naive",
    "now_local",
    "minutes_from_now_local",
    "minutes_between_local",
    "sort_key_local",
    "split_date_time_local",
    "day_local_bounds",
    "coerce_local_day",
]


# RÉTRO-COMPAT UTC : **ici, on corrige pour du vrai UTC aware**
# ---------------------------------------------------------------------------
def _ensure_dt(obj: Union[str, datetime, None]) -> Optional[datetime]:
    if obj is None:
        return None
    if isinstance(obj, datetime):
        return obj
    # chaîne → essaye ISO, interprété comme local naïf si sans TZ
    d = parse_local_naive(obj)
    return d

def to_utc(dt: Union[str, datetime, None]) -> Optional[datetime]:
    """
    Normalise en **UTC aware**.
    - Si dt est naïf: supposé en LOCAL_TZ → converti en UTC.
    - Si dt est aware: astimezone(UTC).
    """
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(timezone.utc)

def to_utc_from_db(dt: Union[str, datetime, None]) -> Optional[datetime]:
    """
    Datetime venant de la DB (souvent aware si timezone=True).
    Ramène toujours en **UTC aware**.
    """
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(timezone.utc)

def ensure_aware_utc(dt: Union[str, datetime, None]) -> Optional[datetime]:
    """Alias explicite vers to_utc (UTC aware)."""
    return to_utc(dt)

def now_utc() -> datetime:
    """Maintenant en **UTC aware** (pour comparer avec des DateTime timezone=True)."""
    return datetime.now(timezone.utc)

def minutes_from_now(dt: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre maintenant (UTC) et dt (normalisé UTC)."""
    d = to_utc(dt)
    if d is None:
        return 10**9
    delta = d - now_utc()
    return max(0, int(delta.total_seconds() // 60))

def minutes_between(a: Union[str, datetime, None], b: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre a et b (tous deux normalisés UTC)."""
    aa, bb = to_utc(a), to_utc(b)
    if aa is None or bb is None:
        return 0
    return max(0, int((aa - bb).total_seconds() // 60))

def sort_key_utc(dt: Union[str, datetime, None]) -> datetime:
    """Clé de tri sûre en **UTC aware**."""
    d = to_utc(dt)
    return d if d is not None else (now_utc() + timedelta(days=365 * 50))

def to_geneva_local(dt: Union[str, datetime, None]) -> Optional[datetime]:
    """Retourne un datetime **aware** en fuseau LOCAL_TZ."""
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(LOCAL_TZ)

def format_geneva(dt: Union[str, datetime, None]) -> Tuple[Optional[str], Optional[str]]:
    """
    Retourne ('YYYY-MM-DD','HH:MM') en **LOCAL_TZ** (aware), pratique pour l'affichage.
    """
    d = to_geneva_local(dt)
    if not d:
        return None, None
    return d.strftime("%Y-%m-%d"), d.strftime("%H:%M")

def iso_utc_z(dt: Union[str, datetime, None]) -> Optional[str]:
    """ISO 8601 en UTC, suffixé 'Z' (ex: 2025-09-21T10:30:00Z)."""
    d = to_utc(dt)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ") if d else None

# Exposer également ces noms pour ne pas casser les imports
__all__ += [
    "to_utc",
    "to_utc_from_db",
    "ensure_aware_utc",
    "now_utc",
    "minutes_from_now",
    "minutes_between",
    "sort_key_utc",
    "to_geneva_local",
    "format_geneva",
    "iso_utc_z",
]